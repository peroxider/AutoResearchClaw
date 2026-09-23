"""Opt-in real-daemon attack verification for the formal Docker sandbox.

Round 53 of the high-quality paper development plan (gap-analysis item 5):
round 39 added the formal-isolation gate at the ``docker run`` command level
and the mocked suite verifies those flags, but nothing had executed the real
command against a real daemon and tried to escape from inside. This suite
drives the production ``DockerSandbox.run_project`` path with
``formal_isolation=True`` and probes the running container for every
isolation property the flag set promises.

The container-side probes are stdlib-only and write their findings to
``/workspace/results.json``, which lands back in the host staging directory.
The fixture image is built from a local base plus the repository's real
``researchclaw/docker/entrypoint.sh``, so neither building nor running needs
network access. Host-cached pulls are never attempted.

Opt-in (needs a local docker daemon and minutes of wall time):

    ARC_RUN_DOCKER_ATTACKS=1 python -m pytest tests/test_real_docker_attacks.py -q

Known calibration: this host runs as root, so production ``--user <uid>:<gid>``
puts a root shell in the container. The setuid-to-root probe is therefore
conditional on the host uid; the binding control for a root user is the fully
dropped capability set (``CapEff == 0``), which is asserted unconditionally.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from researchclaw.config import DockerSandboxConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox

BASE_IMAGE = "openclaw-sandbox:bookworm-slim"
ENTRYPOINT_SOURCE = Path(__file__).resolve().parent.parent / "researchclaw" / "docker" / "entrypoint.sh"


def _daemon_reachable() -> bool:
    return DockerSandbox.check_docker_available()


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ARC_RUN_DOCKER_ATTACKS") != "1",
        reason="opt-in suite: set ARC_RUN_DOCKER_ATTACKS=1 to run against a real daemon",
    ),
    pytest.mark.skipif(not _daemon_reachable(), reason="docker daemon unreachable"),
    pytest.mark.skipif(not DockerSandbox.ensure_image(BASE_IMAGE), reason=f"local base image {BASE_IMAGE} missing"),
]


# ---------------------------------------------------------------------------
# Container-side probes (stdlib only, executed inside the container)
# ---------------------------------------------------------------------------

_FACTS_PROBE = r'''
import ctypes
import json
import os
import socket
import stat
import subprocess

facts = {}


def probe_write(path, append):
    try:
        with open(path, "a" if append else "w", encoding="utf-8") as handle:
            handle.write("rc-attack-probe")
        return "writable"
    except OSError as exc:
        return "blocked errno=%s" % exc.errno


def probe_exec(directory):
    target = os.path.join(directory, "rcattack_exec_copy")
    try:
        with open("/bin/true", "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        os.chmod(target, 0o755)
        completed = subprocess.run([target], capture_output=True, timeout=10)
        return "exit=%s" % completed.returncode
    except OSError as exc:
        return "exec_blocked errno=%s" % exc.errno
    except subprocess.SubprocessError:
        return "exec_failed"
    finally:
        try:
            os.unlink(target)
        except OSError:
            pass


# Identity and privileges
facts["uid"] = os.getuid()
facts["gid"] = os.getgid()
cap_eff = None
no_new_privs = None
with open("/proc/self/status", encoding="ascii") as handle:
    for line in handle:
        if line.startswith("CapEff:"):
            cap_eff = line.split()[1]
        elif line.startswith("NoNewPrivs:"):
            no_new_privs = line.split()[1]
facts["cap_eff"] = cap_eff
facts["no_new_privs"] = no_new_privs
try:
    os.setuid(0)
    facts["setuid_zero"] = "ok uid_now=%s" % os.getuid()
    os.setuid(facts["uid"])
except OSError as exc:
    facts["setuid_zero"] = "blocked errno=%s" % exc.errno

# Read-only rootfs vs writable volumes
facts["write_root_dir"] = probe_write("/rcattack_probe", False)
facts["write_etc_passwd"] = probe_write("/etc/passwd", True)
facts["write_usr"] = probe_write("/usr/bin/rcattack_probe", False)
facts["write_root_home"] = probe_write("/root/rcattack_probe", False)
facts["write_workspace"] = probe_write("/workspace/rcattack_probe", False)
facts["write_tmp"] = probe_write("/tmp/rcattack_probe", False)

# noexec tmpfs on /tmp, executable bind mount on /workspace
facts["exec_tmp"] = probe_exec("/tmp")
facts["exec_workspace"] = probe_exec("/workspace")

# Device creation and mounts require capabilities we dropped
try:
    os.mknod("/tmp/rcattack_zero", 0o600 | stat.S_IFCHR, os.makedev(1, 5))
    facts["mknod_chr"] = "created"
    os.unlink("/tmp/rcattack_zero")
except OSError as exc:
    facts["mknod_chr"] = "blocked errno=%s" % exc.errno
os.makedirs("/tmp/rcattack_bind_target", exist_ok=True)
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_void_p]
rc = libc.mount(b"/workspace", b"/tmp/rcattack_bind_target", b"none", 1, None)
facts["mount_bind"] = "mounted" if rc == 0 else "blocked errno=%s" % ctypes.get_errno()

# --network none
facts["interfaces"] = sorted(name for _, name in socket.if_nameindex())
default_routes = -1
try:
    with open("/proc/net/route", encoding="ascii") as handle:
        rows = [line.split() for line in handle.read().splitlines()[1:] if line.strip()]
    default_routes = sum(1 for row in rows if len(row) > 1 and row[1] == "00000000")
except OSError:
    pass
facts["default_routes"] = default_routes
probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
probe_socket.settimeout(4)
try:
    probe_socket.connect(("1.1.1.1", 53))
    facts["external_connect"] = "succeeded"
except OSError as exc:
    facts["external_connect"] = "blocked %s errno=%s" % (type(exc).__name__, getattr(exc, "errno", None))
finally:
    probe_socket.close()
try:
    socket.getaddrinfo("example.com", 443)
    facts["dns_resolve"] = "succeeded"
except socket.gaierror as exc:
    facts["dns_resolve"] = "blocked %s" % exc

# Docker socket exposure and host cache mounts
facts["docker_sock_exists"] = os.path.exists("/var/run/docker.sock")
with open("/proc/mounts", encoding="ascii") as handle:
    facts["mount_lines"] = handle.read().splitlines()

# Environment forwarding
facts["env_hf_token"] = os.environ.get("HF_TOKEN") or ""
facts["env_hf_token_alt"] = os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
facts["env_hf_hub_cache"] = os.environ.get("HF_HUB_CACHE") or ""
facts["env_control"] = os.environ.get("RC_PROBE_CONTROL") or ""
facts["env_home"] = os.environ.get("HOME") or ""
canary = ""
hub_dir = os.environ.get("HF_HUB_CACHE") or "/home/researcher/.cache/huggingface/hub"
try:
    with open(os.path.join(hub_dir, "rc-canary-token"), encoding="utf-8") as handle:
        canary = handle.read().strip()
except OSError:
    canary = ""
facts["hf_canary_read"] = canary

with open("/workspace/results.json", "w", encoding="utf-8") as handle:
    json.dump(facts, handle, indent=1)
'''

_PIDS_PROBE = r'''
import json
import subprocess

max_children = 0
children = []
attempts = 0
spawn_error = None
while attempts < 320:
    attempts += 1
    try:
        children.append(subprocess.Popen(["/bin/sleep", "30"]))
        max_children = max(max_children, len(children))
    except OSError as exc:
        spawn_error = "%s errno=%s" % (type(exc).__name__, exc.errno)
        break
for child in children:
    child.kill()
reaped = 0
for child in children:
    try:
        child.wait(timeout=15)
        reaped += 1
    except subprocess.SubprocessError:
        pass
with open("/workspace/results.json", "w", encoding="utf-8") as handle:
    json.dump({"max_children": max_children, "attempts": attempts,
               "spawn_error": spawn_error or "", "reaped": reaped}, handle)
'''

_BRIDGE_SERVER_PROBE = (
    "import socket\n"
    "server = socket.socket()\n"
    "server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "server.bind(('0.0.0.0', 8099))\n"
    "server.listen(1)\n"
    "conn, _ = server.accept()\n"
    "conn.sendall(b'pong' + conn.recv(4))\n"
    "conn.close()\n"
)

_BRIDGE_CONNECT_PROBE = (
    "import socket, sys, time\n"
    "host = sys.argv[1]\n"
    "last = None\n"
    "for _ in range(20):\n"
    "    try:\n"
    "        sock = socket.create_connection((host, 8099), timeout=5)\n"
    "        sock.sendall(b'ping')\n"
    "        print(sock.recv(8).decode())\n"
    "        break\n"
    "    except OSError as exc:\n"
    "        last = exc\n"
    "        time.sleep(0.5)\n"
    "else:\n"
    "    raise SystemExit('connector failed: %r' % last)\n"
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_image(tmp_path_factory):
    context = tmp_path_factory.mktemp("rc-attack-build")
    shutil.copyfile(ENTRYPOINT_SOURCE, context / "entrypoint.sh")
    (context / "Dockerfile").write_text(
        "FROM " + BASE_IMAGE + "\n"
        "COPY entrypoint.sh /usr/local/bin/rc-entrypoint.sh\n"
        # The base image defaults to a non-root user ("sandbox"); the
        # fixture build needs root for the chmod, and production overrides
        # the image user with --user anyway.
        "USER root\n"
        "RUN chmod 0755 /usr/local/bin/rc-entrypoint.sh\n"
        'ENTRYPOINT ["/usr/local/bin/rc-entrypoint.sh"]\n'
        'CMD ["main.py"]\n',
        encoding="utf-8")
    tag = f"rc-attack-fixture:{os.getpid()}"
    # The daemon's bundled buildx (v0.12.1) pins API 1.43 and is rejected
    # by this daemon (minimum 1.44); the legacy builder works and is all a
    # COPY+chmod Dockerfile needs.
    build = subprocess.run(["docker", "build", "-t", tag, str(context)],
                           capture_output=True, text=True, timeout=600, check=False,
                           env={**os.environ, "DOCKER_BUILDKIT": "0"})
    assert build.returncode == 0, build.stderr[-2000:]
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False)


@pytest.fixture()
def formal_sandbox(fixture_image, tmp_path):
    config = DockerSandboxConfig(
        image=fixture_image,
        network_policy="none",
        gpu_enabled=False,
        auto_install_deps=False,
        memory_limit_mb=2048,
        shm_size_mb=256,
    )
    return DockerSandbox(config, tmp_path / "sandbox-work")


def _run_probe(sandbox, tmp_path, code, **kwargs):
    project = tmp_path / "probe"
    project.mkdir(exist_ok=True)
    (project / "main.py").write_text(code, encoding="utf-8", newline="\n")
    result = sandbox.run_project(project, formal_isolation=True, timeout_sec=240, **kwargs)
    facts = {}
    output = Path(result.output_dir) / "results.json"
    if output.exists():
        facts = json.loads(output.read_text(encoding="utf-8"))
    return result, facts


def _docker(*args, check=True, timeout=120):
    completed = subprocess.run(["docker", *args], capture_output=True, text=True,
                               timeout=timeout, check=False)
    if check and completed.returncode != 0:
        pytest.fail(f"docker {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


# ---------------------------------------------------------------------------
# Attack probes against the real formal-isolation container
# ---------------------------------------------------------------------------

def test_formal_container_drops_privileges_and_pins_user(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["no_new_privs"] == "1"
    assert facts["cap_eff"] == "0000000000000000"
    assert facts["uid"] == os.getuid() and facts["gid"] == os.getgid()
    assert facts["env_home"] == "/workspace/.home"
    if os.getuid() != 0:
        assert facts["setuid_zero"].startswith("blocked")
    else:
        # Host root means the container is root too (--user 0:0), so
        # setuid(0) is a no-op; the binding control is the dropped
        # capability set pinned to zero above.
        assert facts["setuid_zero"] == "ok uid_now=0"


def test_readonly_rootfs_blocks_system_writes_but_workspace_is_writable(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["write_root_dir"].startswith("blocked")
    assert facts["write_etc_passwd"].startswith("blocked")
    assert facts["write_usr"].startswith("blocked")
    assert facts["write_root_home"].startswith("blocked")
    assert facts["write_workspace"] == "writable"
    assert facts["write_tmp"] == "writable"


def test_tmp_is_noexec_while_workspace_still_executes(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["exec_tmp"].startswith("exec_blocked")
    assert "errno=13" in facts["exec_tmp"]
    assert facts["exec_workspace"] == "exit=0"


def test_device_creation_and_bind_mounts_are_blocked(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    # EPERM specifically: the calls reached the kernel and were denied by
    # the dropped capability set, not failed on a missing path.
    assert facts["mknod_chr"] == "blocked errno=1"
    assert facts["mount_bind"] == "blocked errno=1"


def test_network_none_removes_all_connectivity(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["interfaces"] == ["lo"]
    assert facts["default_routes"] == 0
    assert facts["external_connect"].startswith("blocked")
    assert facts["dns_resolve"].startswith("blocked")


def test_pids_limit_caps_process_explosion(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _PIDS_PROBE)
    assert result.returncode == 0, result.stderr
    # The fork bomb must be stopped by the limit (a spawn error), not by
    # exhausting this probe's own attempt cap.
    assert facts["spawn_error"], facts
    assert facts["max_children"] < 256
    assert facts["max_children"] >= 200
    assert facts["reaped"] == facts["max_children"]


def test_docker_socket_and_host_caches_are_not_mounted(formal_sandbox, tmp_path):
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["docker_sock_exists"] is False
    lines = facts["mount_lines"]
    assert all("docker.sock" not in line for line in lines)
    assert all("huggingface" not in line.lower() for line in lines)
    assert all(" /workspace/data" not in line for line in lines)
    workspace = [line.split() for line in lines if len(line.split()) > 3 and line.split()[1] == "/workspace"]
    assert workspace, lines
    assert "rw" in workspace[0][3].split(",") and "ro" not in workspace[0][3].split(",")
    tmp_line = [line.split() for line in lines if len(line.split()) > 3 and line.split()[1] == "/tmp"]
    assert tmp_line, lines
    assert tmp_line[0][2] == "tmpfs" and "noexec" in tmp_line[0][3].split(",")


def test_host_hf_token_is_not_forwarded_in_formal_mode(monkeypatch, formal_sandbox, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "rc-canary-hf-token-9f13c")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "rc-canary-alt-token-40b2")
    result, facts = _run_probe(
        formal_sandbox, tmp_path, _FACTS_PROBE,
        env_overrides={"RC_PROBE_CONTROL": "control-visible"})
    assert result.returncode == 0, result.stderr
    # The control variable proves -e forwarding works at all, so the
    # absent tokens are a formal-mode decision, not a broken pipe.
    assert facts["env_control"] == "control-visible"
    assert facts["env_hf_token"] == ""
    assert facts["env_hf_token_alt"] == ""


def test_host_hf_home_cache_is_not_mounted_in_formal_mode(monkeypatch, formal_sandbox, tmp_path):
    # An HF cache directory carries the login token file, not just model
    # weights: mounting it read-only would still leak the credential, so
    # formal mode must skip the HF_HOME mount entirely (found by the
    # round-53 verification, fixed with this real-daemon probe + a mocked
    # counterexample).
    cache = tmp_path / "hf-home"
    cache.mkdir()
    (cache / "rc-canary-token").write_text("rc-canary-hf-home-secret-5b21", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))
    result, facts = _run_probe(formal_sandbox, tmp_path, _FACTS_PROBE)
    assert result.returncode == 0, result.stderr
    assert facts["env_hf_hub_cache"] == ""
    assert facts["hf_canary_read"] == ""
    assert all("huggingface" not in line.lower() for line in facts["mount_lines"])


def test_control_daemon_networking_works_on_a_user_defined_bridge(tmp_path):
    """Control for the network probes: same daemon, network NOT disabled.

    Proves the ``--network none`` failures above come from the flag, not
    from a daemon or WSL environment that simply cannot network. Uses the
    base image directly (no entrypoint) with raw docker commands; a
    user-defined bridge needs no external connectivity.
    """
    network = f"rc-attack-ctl-{os.getpid()}"
    listener = f"rc-attack-listener-{os.getpid()}"
    _docker("network", "create", network)
    try:
        # Detached run: the container exists as soon as this returns, so
        # the running-state poll below cannot race its creation.
        _docker("run", "-d", "--rm", "--name", listener, "--network", network,
                BASE_IMAGE, "python3", "-c", _BRIDGE_SERVER_PROBE)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if _docker("inspect", "-f", "{{.State.Running}}", listener).strip() == "true":
                break
            time.sleep(0.3)
        else:
            pytest.fail("control listener container never entered the running state")
        # This daemon's embedded DNS does not resolve user-bridge aliases,
        # so the connector targets the listener's bridge IP directly; the
        # control is about container-to-container reachability, not DNS.
        ip = _docker("inspect", "-f",
                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                     listener).strip()
        assert ip, "control listener was assigned no bridge IP"
        reply = _docker("run", "--rm", "--network", network, BASE_IMAGE,
                        "python3", "-c", _BRIDGE_CONNECT_PROBE, ip, timeout=90)
        # The listener answers b'pong' + echo of what it received.
        assert reply.strip().startswith("pong"), reply
    finally:
        _docker("rm", "-f", listener, check=False)
        _docker("network", "rm", network, check=False)
