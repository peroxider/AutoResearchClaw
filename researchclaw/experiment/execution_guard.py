"""In-process execution guard for host-side formal matrix runs.

The wrapper is injected next to the frozen experiment source and launched as
``python -I -u execution_guard.py <entry_point>``. Before the entry point
runs, it installs a stdlib audit hook, which cannot be removed, that:

- blocks network use (connect, bind, name resolution, unconnected UDP sends)
  outright;
- blocks process spawning: the audited spawn events (subprocess, os.system/
  exec/posix_spawn/fork, startfile), the ``multiprocessing`` package import,
  and the C-level primitives whose calls emit no spawn audit event on
  Windows/CPython (``_winapi.CreateProcess`` — the exact primitive the
  multiprocessing spawn path uses — plus ``os.spawn*`` and
  ``_posixsubprocess.fork_exec``), all patched to raise at install time;
- confines file writes, removals, renames, truncation, and the un-audited
  directory/metadata writers (``os.mkdir``, ``os.chmod``, ``os.utime``,
  ``os.symlink``, ``os.link``, also patched at install time) to the run's
  staging directory and the system temp directory. Reads and the null device
  stay unrestricted.

Evidence is tamper-resistant by construction: every violation marker and the
activation banner are written to a ``dup()`` of the original stderr file
descriptor captured before the entry point runs, so reassigning
``sys.stderr`` or ``os.dup2``-ing over fd 2 cannot silence them. An
``atexit`` finalizer additionally forces exit code 3 when violations were
recorded — best-effort only, because the experiment's own ``os._exit`` skips
shutdown hooks; the real-time markers are the load-bearing channel, and the
formal runner rejects any captured stderr carrying a violation marker
regardless of exit code.

Every violation prints a distinctive ``ARC_GUARD_VIOLATION`` marker to the
captured stderr — the frozen attempt logs preserve it for audit.

Honest boundary: the guard stops accidental escapes and casual tampering and
records deliberate ones; it is not a security boundary against a determined
adversary, who can still hunt file descriptors, restore patched attributes,
or use the deliberately open ctypes pathway (scientific libraries need it).
Container backends remain the strong isolation layer. One availability note:
a cold Python installation whose ``__pycache__`` is unmaterialized makes the
first guarded import of a stdlib module attempt a pyc write outside the
roots, failing that attempt; blocking the write is the safe direction.
"""
from __future__ import annotations

import atexit
import os
import runpy
import socket
import sys
import tempfile
from pathlib import Path

GUARD_MARKER = "execution_guard/v1"
VIOLATION_PREFIX = "ARC_GUARD_VIOLATION"
GUARD_ACTIVE_PREFIX = "ARC_GUARD_ACTIVE"
FORCED_EXIT_CODE = 3

_BLOCKED_EVENTS = {
    "socket.connect": "network connect",
    "socket.bind": "network bind",
    "socket.getaddrinfo": "name resolution",
    "socket.gethostbyname": "name resolution",
    "socket.gethostbyname_ex": "name resolution",
    "subprocess.Popen": "process spawn",
    "os.system": "process spawn",
    "os.exec": "process spawn",
    "os.posix_spawn": "process spawn",
    "os.posix_spawnp": "process spawn",
    "os.fork": "process spawn",
    "os.forkpty": "process spawn",
    "os.startfile": "process spawn",
}

# Un-audited spawn paths on Windows/CPython: closed by patching at install.
_SPAWN_FUNCTIONS = (
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
)
_SPAWN_PRIMITIVES = (("_winapi", "CreateProcess"), ("_posixsubprocess", "fork_exec"))
# Un-audited directory/metadata writers: confined by patching at install.
_METADATA_FUNCTIONS = ("mkdir", "chmod", "utime", "symlink", "link")

_DEVNULL_NAMES = frozenset({os.devnull, "nul", "/dev/null", "\\\\.\\nul"})

_WRITE_MODE_CHARS = ("w", "a", "x", "+")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _is_write(mode, flags) -> bool:
    if isinstance(mode, str) and any(char in mode for char in _WRITE_MODE_CHARS):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _is_devnull(path) -> bool:
    try:
        text = os.fspath(path)
    except TypeError:
        return False
    return isinstance(text, str) and text.strip().lower() in _DEVNULL_NAMES


class ExecutionGuard:
    """Audit hook confining writes to the declared roots; blocking network
    use and process spawning outright."""

    def __init__(self, allowed_roots) -> None:
        self._roots = [Path(root).resolve() for root in allowed_roots]
        self._violations = 0
        try:
            # A private duplicate of the original stderr descriptor: the
            # tamper-resistant evidence channel (sys.stderr reassignment and
            # dup2 over fd 2 cannot touch it).
            self._evidence_fd = os.dup(2)
        except OSError:
            self._evidence_fd = 2

    def _within(self, path) -> bool:
        try:
            candidate = Path(os.path.abspath(os.fspath(path))).resolve()
        except (TypeError, ValueError, OSError):
            return False
        return any(candidate == root or candidate.is_relative_to(root) for root in self._roots)

    def _record(self, message: str) -> None:
        try:
            os.write(self._evidence_fd, (message + "\n").encode("utf-8", "replace"))
        except OSError:
            pass

    def _violate(self, event: str, detail: str) -> None:
        self._violations += 1
        self._record(f"{VIOLATION_PREFIX} {event}: {detail}")
        raise PermissionError(f"execution guard blocked {event}: {detail}")

    def hook(self, event: str, args) -> None:
        blocked = _BLOCKED_EVENTS.get(event)
        if blocked is not None:
            self._violate(event, blocked)
        elif event == "open":
            path, mode, flags = args
            if isinstance(path, int) or not _is_write(mode, flags):
                return
            if not _is_devnull(path) and not self._within(path):
                self._violate(event, f"write outside the sandbox: {path!r}")
        elif event == "import":
            name = args[0]
            if name == "multiprocessing" or name.startswith("multiprocessing."):
                # The Windows spawn machinery emits no process-spawn audit
                # event; refusing the import keeps "process spawn: blocked"
                # honest (formal runs spawn nothing).
                self._violate(event, "multiprocessing import (process spawn)")
        elif event in ("os.remove", "os.rmdir", "os.truncate"):
            if not self._within(args[0]):
                self._violate(event, f"write outside the sandbox: {args[0]!r}")
        elif event == "os.rename":
            if not (self._within(args[0]) and self._within(args[1])):
                self._violate(event, f"rename crossing the sandbox boundary: {args[0]!r} -> {args[1]!r}")

    def _harden(self) -> None:
        """Close the un-audited stdlib gaps (no audit events exist for these
        on CPython): the os.spawn* family and unconnected socket sends, plus
        confinement for the directory/metadata writers."""
        guard = self

        for name in _SPAWN_FUNCTIONS:
            if hasattr(os, name):
                setattr(os, name, guard._blocker("os." + name, "process spawn"))
        for name in ("sendto", "sendmsg"):
            if hasattr(socket.socket, name):
                setattr(socket.socket, name,
                        guard._blocker("socket." + name, "network send without connect"))

        # C-level spawn primitives whose calls emit no audit event: patched
        # at install (the audited callers fail first via their own events,
        # so ordinary subprocess/multiprocessing users never reach these).
        for module_name, attribute in _SPAWN_PRIMITIVES:
            try:
                module = __import__(module_name)
            except ImportError:
                continue
            if hasattr(module, attribute):
                setattr(module, attribute,
                        guard._blocker(f"{module_name}.{attribute}", "process spawn"))

        originals = {}
        for name in _METADATA_FUNCTIONS:
            function = getattr(os, name, None)
            if function is not None:
                originals[name] = function

        def confined_mkdir(path, *args, **kwargs):
            if not guard._within(path):
                guard._violate("os.mkdir", f"write outside the sandbox: {path!r}")
            return originals["mkdir"](path, *args, **kwargs)

        def confined_chmod(path, *args, **kwargs):
            if not guard._within(path):
                guard._violate("os.chmod", f"metadata write outside the sandbox: {path!r}")
            return originals["chmod"](path, *args, **kwargs)

        if "mkdir" in originals:
            os.mkdir = confined_mkdir  # os.makedirs funnels through os.mkdir.
        if "chmod" in originals:
            os.chmod = confined_chmod
        if "utime" in originals:
            def confined_utime(path, *args, **kwargs):
                if not guard._within(path):
                    guard._violate("os.utime", f"metadata write outside the sandbox: {path!r}")
                return originals["utime"](path, *args, **kwargs)
            os.utime = confined_utime
        if "symlink" in originals:
            def confined_symlink(src, dst, *args, **kwargs):
                if not guard._within(dst):
                    guard._violate("os.symlink", f"link outside the sandbox: {dst!r}")
                return originals["symlink"](src, dst, *args, **kwargs)
            os.symlink = confined_symlink
        if "link" in originals:
            def confined_link(src, dst, *args, **kwargs):
                if not guard._within(dst):
                    guard._violate("os.link", f"link outside the sandbox: {dst!r}")
                return originals["link"](src, dst, *args, **kwargs)
            os.link = confined_link

    def _blocker(self, event: str, detail: str):
        guard = self

        def blocked(*args, **kwargs):
            guard._violate(event, detail)

        return blocked

    def _finalize(self) -> None:
        # Best-effort second channel: atexit runs at ordinary interpreter
        # shutdown, but the experiment's own os._exit (or a preempting atexit
        # handler) skips it. The real-time markers on the evidence descriptor
        # and the runner's marker checks are the load-bearing enforcement.
        try:
            if self._violations:
                self._record(f"{VIOLATION_PREFIX} summary: {self._violations} guard "
                             "violation(s) recorded in this process; attempt forced to fail")
                for stream in (sys.stdout, sys.stderr):
                    try:
                        stream.flush()
                    except Exception:
                        pass
                os._exit(FORCED_EXIT_CODE)
        except Exception:
            pass


def install_guard(allowed_roots) -> ExecutionGuard:
    guard = ExecutionGuard(allowed_roots)
    sys.addaudithook(guard.hook)
    guard._harden()
    atexit.register(guard._finalize)
    return guard


def guarded_main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("usage: execution_guard.py <entry_point> [args ...]")
    root = Path.cwd().resolve()
    entry = Path(os.path.abspath(argv[1])).resolve()
    if not (entry == root or entry.is_relative_to(root)):
        raise SystemExit(f"{VIOLATION_PREFIX} entry point escapes the sandbox: {argv[1]}")
    if not entry.is_file():
        raise SystemExit(f"entry point not found: {argv[1]}")
    guard = install_guard([root, Path(tempfile.gettempdir())])
    # One line of proof, captured in the frozen attempt logs, that the guard
    # actually ran in this process — on the tamper-resistant evidence channel.
    guard._record(f"{GUARD_ACTIVE_PREFIX} {GUARD_MARKER}")
    sys.argv = [str(entry), *argv[2:]]
    sys.path.insert(0, str(root))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    guarded_main(sys.argv)
