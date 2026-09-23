"""Host execution guard: network/spawn blocking and write confinement.

Every hook-behavior check runs in a CHILD process: sys.addaudithook cannot
be removed, so installing the guard inside the pytest process would confine
every later test in the session. Only the pure helpers run in-process.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from researchclaw.experiment.execution_guard import (
    GUARD_ACTIVE_PREFIX, GUARD_MARKER, VIOLATION_PREFIX, ExecutionGuard, _is_write,
)
from researchclaw.experiment.protocol_runner import read_ledger, run_matrix, verify_execution_bundle
from researchclaw.pipeline.evidence_store import file_hash
from researchclaw.pipeline.experiment_protocol import ProtocolError
from tests.test_dependency_manifest import _forge_chain
from tests.test_experiment_protocol import spec
from tests.test_protocol_runner import SCRIPT, setup
from tests.test_research_inputs import inputs

GUARD = Path(__file__).resolve().parents[1] / "researchclaw" / "experiment" / "execution_guard.py"


def launch(project: Path, entry: str = "main.py", env: dict | None = None, timeout: int = 120):
    environment = None if env is None else {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-I", "-u", str(GUARD), entry],
        cwd=project, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=environment, check=False,
    )


def make_project(tmp_path: Path, code: str, name: str = "main.py") -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / name).write_text(code, encoding="utf-8", newline="")
    return project


def escape_target() -> str:
    # The parent of the temp directory is the nearest deterministic location
    # outside every allowed root; the pid suffix keeps concurrent pytest
    # processes from observing each other's probes on a shared host.
    return str(Path(tempfile.gettempdir()).parent / f"arc_guard_escape_probe_{os.getpid()}.txt")


# ── Pure helpers (in-process, no hook installed) ────────────────────────────


def test_write_detection_covers_modes_and_flags(tmp_path):
    assert _is_write("w", 0) and _is_write("a", 0) and _is_write("x", 0) and _is_write("+", 0)
    assert _is_write(None, os.O_WRONLY) and _is_write(None, os.O_RDWR) and _is_write(None, os.O_CREAT)
    assert _is_write(None, os.O_APPEND) and _is_write(None, os.O_TRUNC)
    assert not _is_write("r", 0) and not _is_write("rb", 0) and not _is_write(None, os.O_RDONLY)


def test_containment_rejects_sibling_prefixes_and_escapes(tmp_path):
    guard = ExecutionGuard([tmp_path])
    assert guard._within(tmp_path)
    assert guard._within(tmp_path / "sub" / "file.txt")
    assert not guard._within(tmp_path.parent)
    assert not guard._within(str(tmp_path) + "sibling")
    assert not guard._within(escape_target())


# ── Hook behavior (child processes only) ────────────────────────────────────


def test_benign_project_runs_and_may_write_inside_the_sandbox(tmp_path):
    project = make_project(tmp_path, (
        "from pathlib import Path\n"
        "Path('out').mkdir()\n"
        "Path('out/result.txt').write_text('ok')\n"
        "print('accuracy: 0.5')\n"
    ))
    completed = launch(project)
    assert completed.returncode == 0, completed.stderr
    assert VIOLATION_PREFIX not in completed.stderr
    assert (project / "out" / "result.txt").read_text(encoding="utf-8") == "ok"
    assert "0.5" in completed.stdout


def test_network_access_is_blocked(tmp_path):
    project = make_project(tmp_path, (
        "import socket\n"
        "socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "socket" in completed.stderr
    assert "PermissionError" in completed.stderr


def test_process_spawning_is_blocked(tmp_path):
    project = make_project(tmp_path, (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(1)'])\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "process spawn" in completed.stderr


def test_multiprocessing_import_is_blocked(tmp_path):
    # Windows multiprocessing spawns via _winapi.CreateProcess, which emits
    # no spawn audit event — the guard refuses the package import instead.
    project = make_project(tmp_path, (
        "import multiprocessing\n"
        "print('child-ran')\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "multiprocessing" in completed.stderr
    assert "child-ran" not in completed.stdout


def test_os_spawn_family_is_blocked(tmp_path):
    # os.spawn* emits no audit event either; the guard patches it at install.
    project = make_project(tmp_path, (
        "import os, sys\n"
        "os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, '-c', 'print(1)'])\n"
        "print('parent-done')\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "process spawn" in completed.stderr
    assert "parent-done" not in completed.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="_winapi is Windows-only")
def test_winapi_createprocess_is_blocked(tmp_path):
    # The exact C-level primitive the multiprocessing spawn path uses: its
    # call emits no spawn audit event, so the guard patches it at install.
    target = escape_target()
    escaped = target.replace("\\", "\\\\")  # cmd.exe redirect path inside a Python literal.
    project = make_project(tmp_path, (
        "import _winapi\n"
        f"r = _winapi.CreateProcess(None, 'cmd.exe /c echo spawned > {escaped}', "
        "None, None, False, 0, None, None, None)\n"
        "print('createprocess-returned')\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "process spawn" in completed.stderr
    assert "createprocess-returned" not in completed.stdout
    assert not os.path.exists(target)


def test_unconnected_udp_sends_are_blocked(tmp_path):
    project = make_project(tmp_path, (
        "import socket\n"
        "socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'leak', ('127.0.0.1', 9))\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "network" in completed.stderr


def test_metadata_writes_outside_the_sandbox_are_blocked(tmp_path):
    target_dir = str(Path(tempfile.gettempdir()).parent / f"arc_guard_escape_dir_{os.getpid()}")
    outside = Path(target_dir + "_chmod_victim.txt")
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("victim", encoding="utf-8")
    try:
        before = outside.stat().st_mtime_ns
        project = make_project(tmp_path, (
            "import os\n"
            f"os.makedirs({target_dir!r}, exist_ok=True)\n"
            f"os.chmod({str(outside)!r}, 0o444)\n"
            f"os.utime({str(outside)!r}, None)\n"
        ))
        completed = launch(project)
        assert completed.returncode != 0
        assert VIOLATION_PREFIX in completed.stderr
        assert not os.path.exists(target_dir)
        assert outside.read_text(encoding="utf-8") == "victim"
        assert outside.stat().st_mtime_ns == before  # utime blocked before effect.
    finally:
        outside.unlink(missing_ok=True)


def test_writes_to_the_null_device_are_allowed(tmp_path):
    project = make_project(tmp_path, (
        "import os, sys\n"
        "with open(os.devnull, 'w') as sink:\n"
        "    print('discarded', file=sink)\n"
        "print('done')\n"
    ))
    completed = launch(project)
    assert completed.returncode == 0, completed.stderr
    assert VIOLATION_PREFIX not in completed.stderr


def test_write_outside_the_sandbox_is_blocked(tmp_path):
    target = escape_target()
    project = make_project(tmp_path, (
        "import os\n"
        f"open({target!r}, 'w').write('escape')\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert not os.path.exists(target)


def test_reads_stay_unrestricted_by_design(tmp_path):
    outside = Path(escape_target())
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("host bytes", encoding="utf-8")
    try:
        project = make_project(tmp_path, (
            f"print(open({str(outside)!r}).read())\n"
        ))
        completed = launch(project)
        assert completed.returncode == 0, completed.stderr
        assert "host bytes" in completed.stdout
    finally:
        outside.unlink(missing_ok=True)


def test_removal_outside_the_sandbox_is_blocked_before_the_filesystem(tmp_path):
    outside = Path(escape_target())
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("keep me", encoding="utf-8")
    try:
        project = make_project(tmp_path, (
            f"import os\nos.remove({str(outside)!r})\n"
        ))
        completed = launch(project)
        assert completed.returncode != 0
        assert VIOLATION_PREFIX in completed.stderr
        assert outside.read_text(encoding="utf-8") == "keep me"
    finally:
        outside.unlink(missing_ok=True)


def test_rename_crossing_the_boundary_is_blocked(tmp_path):
    project = make_project(tmp_path, (
        "import os\n"
        "open('inside.txt', 'w').write('x')\n"
        f"os.rename('inside.txt', {escape_target()!r})\n"
    ))
    completed = launch(project)
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert (project / "inside.txt").is_file()


def test_writes_to_the_system_temp_directory_are_allowed(tmp_path):
    project = make_project(tmp_path, (
        "import tempfile, os\n"
        "directory = tempfile.mkdtemp()\n"
        "open(os.path.join(directory, 'scratch.txt'), 'w').write('ok')\n"
        "print('done')\n"
    ))
    completed = launch(project)
    assert completed.returncode == 0, completed.stderr
    assert VIOLATION_PREFIX not in completed.stderr


def test_violations_fail_the_attempt_even_when_caught_by_the_experiment(tmp_path):
    target = escape_target()
    project = make_project(tmp_path, (
        "import sys\n"
        f"try:\n    open({target!r}, 'w')\n"
        "except PermissionError:\n    pass\n"
        "open('out.txt', 'w').write('still ok')\n"
        "print('finished')\n"
    ))
    completed = launch(project)
    # The guard's shutdown finalizer forces a nonzero exit: a caught
    # violation can never exit 0, and the markers stay on the frozen logs.
    assert completed.returncode == 3
    assert VIOLATION_PREFIX in completed.stderr  # Recorded on the frozen logs for audit.
    assert (project / "out.txt").read_text(encoding="utf-8") == "still ok"


def test_stderr_reassignment_cannot_silence_violation_markers(tmp_path):
    # Markers go to a dup() of the original stderr descriptor, not to
    # sys.stderr at call time — reassigning the stream cannot hide them.
    project = make_project(tmp_path, (
        "import sys\n"
        "sink = open('sink.txt', 'w')\n"
        "sys.stderr = sink\n"
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
        "except PermissionError:\n"
        "    pass\n"
        "print('done-cleanly')\n"
    ))
    completed = launch(project)
    assert completed.returncode == 3
    assert VIOLATION_PREFIX in completed.stderr
    assert "socket" in completed.stderr
    assert "done-cleanly" in completed.stdout


def test_dup2_over_stderr_cannot_silence_violation_markers(tmp_path):
    # The evidence descriptor was duplicated before the entry point ran, so
    # dup2-ing a staging sink over fd 2 cannot divert the markers either.
    project = make_project(tmp_path, (
        "import os, sys\n"
        "fd = os.open('sink2.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)\n"
        "os.dup2(fd, 2)\n"
        "import subprocess\n"
        "try:\n"
        "    subprocess.run([sys.executable, '-c', 'print(1)'])\n"
        "except PermissionError:\n"
        "    pass\n"
        "print('done-cleanly')\n"
    ))
    completed = launch(project)
    assert completed.returncode == 3
    assert VIOLATION_PREFIX in completed.stderr
    assert "process spawn" in completed.stderr
    assert "done-cleanly" in completed.stdout


def test_entry_point_outside_the_sandbox_fails_closed(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("print('escaped')\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    completed = launch(project, entry=f"../{outside.name}")
    assert completed.returncode != 0
    assert VIOLATION_PREFIX in completed.stderr
    assert "escaped" not in completed.stdout


def test_missing_entry_point_fails_cleanly(tmp_path):
    project = make_project(tmp_path, "print('never')\n")
    completed = launch(project, entry="absent.py")
    assert completed.returncode != 0
    assert "entry point not found" in completed.stderr
    assert "never" not in completed.stdout
    # Marker constant is a frozen contract: receipts and portable verify
    # branch on this exact string.
    assert GUARD_MARKER == "execution_guard/v1"


def test_active_guard_announces_itself_on_stderr(tmp_path):
    project = make_project(tmp_path, "print('plain')\n")
    completed = launch(project)
    assert completed.returncode == 0
    assert f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}" in completed.stderr


# ── ExperimentSandbox integration ───────────────────────────────────────────


def sandbox_for(tmp_path):
    from researchclaw.config import SandboxConfig
    from researchclaw.experiment.sandbox import ExperimentSandbox
    return ExperimentSandbox(SandboxConfig(python_path=sys.executable), tmp_path / "work")


def test_sandbox_guarded_run_injects_and_runs_the_guard(tmp_path):
    project = make_project(tmp_path, (
        "from pathlib import Path\n"
        "Path('out.txt').write_text('ok')\n"
        "print('accuracy: 0.5')\n"
    ))
    result = sandbox_for(tmp_path).run_project(project, timeout_sec=120, guarded=True)
    assert result.returncode == 0, result.stderr
    assert f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}" in result.stderr
    assert (Path(result.output_dir) / "execution_guard.py").is_file()
    assert (Path(result.output_dir) / "out.txt").read_text(encoding="utf-8") == "ok"
    assert result.metrics == {"accuracy": 0.5}


def test_sandbox_guarded_run_blocks_network_and_subprocess(tmp_path):
    project = make_project(tmp_path, (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(1)'])\n"
    ))
    result = sandbox_for(tmp_path).run_project(project, timeout_sec=120, guarded=True)
    assert result.returncode != 0
    assert VIOLATION_PREFIX in result.stderr


def test_sandbox_unguarded_default_is_unchanged(tmp_path):
    project = make_project(tmp_path, "print('accuracy: 0.5')\n")
    result = sandbox_for(tmp_path).run_project(project, timeout_sec=120)
    assert result.returncode == 0
    assert VIOLATION_PREFIX not in result.stderr
    assert GUARD_ACTIVE_PREFIX not in result.stderr
    assert not (Path(result.output_dir) / "execution_guard.py").exists()


def test_project_cannot_shadow_the_immutable_guard(tmp_path):
    project = make_project(tmp_path, "print('benign')\n")
    (project / "execution_guard.py").write_text("print('SHADOWED')\n", encoding="utf-8")
    result = sandbox_for(tmp_path).run_project(project, timeout_sec=120, guarded=True)
    assert result.returncode == 0, result.stderr
    assert f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}" in result.stderr
    assert "SHADOWED" not in result.stdout


# ── Protocol-runner enforcement (formal host matrix) ────────────────────────


def test_host_matrix_binds_guard_isolation_and_freezes_banner_evidence(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    report = run_matrix(root, project, cfg.experiment)
    assert report["status"] == "complete"
    code = json.loads((root / "protocol_code.json").read_text(encoding="utf-8"))
    assert code["isolation"]["kind"] == "host_guarded/v1"
    assert code["isolation"]["in_process_guard"] == GUARD_MARKER
    receipt = json.loads(
        (root / "evidence_artifacts/protocol_runs/attempt-000001/execution.json").read_text(
            encoding="utf-8"))
    assert receipt["isolation"] == code["isolation"]
    stderr_text = (root / "evidence_artifacts/protocol_runs/attempt-000001/stderr.txt").read_text(
        encoding="utf-8", errors="replace")
    assert f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}" in stderr_text
    assert VIOLATION_PREFIX not in stderr_text
    assert verify_execution_bundle(root, protocol)["success"]


def test_runner_rejects_a_guard_violation_even_when_the_experiment_exits_cleanly(inputs, spec):
    suffix = (
        "import tempfile\n"
        "try:\n"
        "    open(os.path.join(os.path.dirname(tempfile.gettempdir()), "
        "'arc_runner_violation_probe.txt'), 'w').write('escape')\n"
        "except PermissionError:\n"
        "    pass\n"
    )
    root, project, protocol, cfg = setup(inputs, spec, SCRIPT + suffix)
    # The forced guard exit fails every violating attempt (recorded in the
    # hash-bound receipt), the cell never succeeds, and the frozen matrix
    # can no longer be accepted.
    report = run_matrix(root, project, cfg.experiment)
    assert report["status"] == "incomplete"
    finished = [event["attempt"] for event in read_ledger(root, protocol)
                if event["kind"] == "finish"]
    assert finished
    for aid in finished:
        receipt = json.loads(
            (root / f"evidence_artifacts/protocol_runs/{aid}/execution.json").read_text(
                encoding="utf-8"))
        assert receipt["status"] == "failed"
        assert receipt["returncode"] != 0  # Forced guard exit cannot mask a violation.
        stderr_text = (root / f"evidence_artifacts/protocol_runs/{aid}/stderr.txt").read_text(
            encoding="utf-8", errors="replace")
        assert f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}" in stderr_text
        assert VIOLATION_PREFIX in stderr_text  # Frozen audit record of the attempt.
    with pytest.raises(ProtocolError, match="incomplete"):
        verify_execution_bundle(root, protocol)


def test_verify_rejects_a_success_receipt_whose_frozen_banner_was_stripped(inputs, spec):
    root, project, protocol, cfg = setup(inputs, spec)
    run_matrix(root, project, cfg.experiment)
    aid = "attempt-000001"
    stderr_path = root / f"evidence_artifacts/protocol_runs/{aid}/stderr.txt"
    stderr_path.write_text(
        stderr_path.read_text(encoding="utf-8", errors="replace").replace(
            f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}\n", ""),
        encoding="utf-8")
    receipt_path = root / f"evidence_artifacts/protocol_runs/{aid}/execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["logs"][f"evidence_artifacts/protocol_runs/{aid}/stderr.txt"] = file_hash(stderr_path)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    events = read_ledger(root, protocol)
    for event in events:
        if event["kind"] == "finish" and event["attempt"] == aid:
            event["receipt_sha256"] = file_hash(receipt_path)
    _forge_chain(root, events)  # The forger also rebuilds the chain and budget anchors.
    with pytest.raises(ProtocolError, match="lacks execution-guard evidence"):
        verify_execution_bundle(root, protocol)
