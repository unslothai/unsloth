# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The live probe that decides whether a sandbox backend actually confines anything.

Every control is paired with the same control run on the HOST first, so a typo
in a path cannot read as a boundary, nor a quirk of the machine as a breakage.

"A write outside the workdir must raise" is wrong: bwrap mounts a private tmpfs
over /tmp, where this probe's scratch root usually lives. What must not happen
is the byte arriving on the host, checked from out here after the run.
"""

from __future__ import annotations
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from loggers import get_logger

logger = get_logger(__name__)

_PR_SET_NO_NEW_PRIVS = 38
try:
    # Resolved at import: an import after the fork can deadlock on the import lock.
    _libc = ctypes.CDLL(None, use_errno = True) if sys.platform == "linux" else None
except OSError:  # pragma: no cover - a libc that will not load
    _libc = None

# ALONE on stdout: a payload that printed it early and died would satisfy "in".
PROBE_TOKEN = "UNSLOTH_SANDBOX_PROBE_OK"
_SENTINEL_TOKEN = "unsloth-host-sentinel-must-not-be-readable"
# Finding this on the host afterwards is the escape; the write need not raise.
_OUTSIDE_WRITE_TOKEN = "unsloth-sandbox-escaped-to-the-host"

PROBE_TIMEOUT_SECONDS = 30.0
# Short enough that installing the AppArmor profile takes effect without a restart.
_CACHE_TTL_SECONDS = 60.0
_CACHE_MAX_ENTRIES = 8

# ``sun_path`` is 108 bytes and multiprocessing appends about 32 of its own, so a
# longer scratch root fails a POSITIVE control for a reason unrelated to isolation.
_MAX_PROBE_BASE_LEN = 59

_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, bool, str]] = {}


def reset_probe_cache() -> None:
    """Forget every cached verdict. For tests only."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple[str, str]) -> tuple[bool, str] | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, available, reason = entry
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        return available, reason


def _cache_put(key: tuple[str, str], available: bool, reason: str) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, available, reason)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))


_PREAMBLE = '''import multiprocessing.reduction, os, socket, subprocess, sys


def must_raise(label, fn):
    """A negative control: the host proved it works, so in here it must not."""
    try:
        fn()
    except OSError:
        return
    raise AssertionError(label)
'''


def _negative_controls(
    sentinel: str, escape: str, outside: str, interpreter_writable: bool, abstract: "bytes | None"
) -> str:
    """What a confined process must NOT be able to do.

    The sentinel is read twice, directly and through a workdir symlink, since a
    boundary drawn on a path's spelling lets the second one out. The outside write
    is NOT required to raise (a private tmpfs may accept it); the host decides
    afterwards. The abstract-socket leg proves the Landlock scope took hold,
    which the ABI version cannot say.
    """
    interpreter_leg = ""
    if interpreter_writable:
        interpreter_leg = (
            '\nmust_raise("opened the interpreter for writing", '
            'lambda: open(sys.executable, "ab").close())\n'
        )
    abstract_leg = ""
    if abstract is not None:
        abstract_leg = (
            f'\nmust_raise("connected to a host abstract unix socket", lambda: '
            f"socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect({abstract!r}))\n"
        )
    return f"""
must_raise("read the host sentinel", lambda: open({sentinel!r}, "rb").close())
must_raise("followed a workdir symlink to the host sentinel",
           lambda: open({escape!r}, "rb").close())
{interpreter_leg}{abstract_leg}
# Allowed to succeed against a private tmpfs; the host checks afterwards that
# nothing arrived. See _negative_controls.
try:
    with open({outside!r}, "w", encoding = "utf-8") as handle:
        handle.write({_OUTSIDE_WRITE_TOKEN!r})
except OSError:
    pass
"""


def _positive_controls(workdir: str) -> str:
    return f"""
private = os.path.join({workdir!r}, "private.txt")
with open(private, "w", encoding = "utf-8") as handle:
    handle.write("sandbox-write-ok")
with open(private, encoding = "utf-8") as handle:
    if handle.read() != "sandbox-write-ok":
        raise AssertionError("workdir read-back did not match what was written")

# numpy/torch in a tool call fork workers, and a sandbox that breaks the
# socketpair breaks every one of them.
left, right = socket.socketpair()
try:
    left.sendall(b"ping")
    if right.recv(4) != b"ping":
        raise AssertionError("socketpair did not deliver")
    right.sendall(b"pong")
    if left.recv(4) != b"pong":
        raise AssertionError("socketpair did not deliver in reverse")
    # Descriptor passing through the resource sharer, which is how Pool and
    # ProcessPoolExecutor hand file handles between processes.
    duplicated = multiprocessing.reduction.DupFd(left.fileno()).detach()
    os.close(duplicated)
finally:
    left.close()
    right.close()

# A child interpreter, because tool code shells out to python constantly.
child = subprocess.run(
    [sys.executable, "-I", "-S", "-c", "print(42)"],
    stdin = subprocess.DEVNULL,
    capture_output = True,
    timeout = 20,
)
if child.returncode != 0 or child.stdout.strip() != b"42":
    raise AssertionError("a python child could not run: " + repr(child.stderr[-200:]))
"""


def _payload(
    workdir: str,
    sentinel: str,
    escape: str,
    outside: str,
    interpreter_writable: bool,
    abstract: "bytes | None",
) -> str:
    """Passed as source on the command line, not as a workdir file, so the probe
    does not depend on how the backend exposes the workdir."""
    return (
        _PREAMBLE
        + _negative_controls(sentinel, escape, outside, interpreter_writable, abstract)
        + _positive_controls(workdir)
        + f"\nprint({PROBE_TOKEN!r})\n"
    )


def _no_new_privs() -> None:
    """Logged rather than swallowed: failing to set it makes the probe MORE
    permissive than the launch it stands in for."""
    if _libc is None:
        return
    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        logger.warning("The sandbox probe could not set PR_SET_NO_NEW_PRIVS")


def _abstract_control() -> "tuple[bytes | None, Any]":
    """Nothing where there is no scope to test, and nothing when the host cannot
    connect to its own socket, since a refusal inside would then prove nothing."""
    if sys.platform != "linux":
        return None, None
    from . import sandbox_landlock

    if not sandbox_landlock.abstract_scope_supported():
        return None, None
    name = b"\0unsloth-probe-" + os.urandom(8).hex().encode()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(name)
        listener.listen(1)
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.settimeout(5)
        try:
            control.connect(name)
        finally:
            control.close()
    except OSError:
        listener.close()
        return None, None
    return name, listener


def _host_saw_the_write(outside: str) -> bool:
    """The only honest form of the escape check: a private tmpfs makes the write
    succeed inside while nothing arrives here."""
    try:
        with open(outside, encoding = "utf-8") as handle:
            return _OUTSIDE_WRITE_TOKEN in handle.read()
    except OSError:
        return False


def _host_payload(workdir: str) -> str:
    return _PREAMBLE + _positive_controls(workdir) + f"\nprint({PROBE_TOKEN!r})\n"


def _probe_base() -> str:
    """A scratch root short enough for the fd-passing control's AF_UNIX address;
    failing that, the host control below reports the problem."""
    # Private roots FIRST, the environment-derived default last. TMPDIR can point
    # inside a directory the backend deliberately exposes (/opt/tmp is enough),
    # which puts the host sentinel under a read-only bind: the negative read then
    # succeeds and a working backend is reported unavailable.
    roots: list[str | None] = [root for root in ("/tmp", "/var/tmp") if os.path.isdir(root)]
    roots.append(None)  # None = the platform default
    fallback = None
    for root in roots:
        try:
            base = tempfile.mkdtemp(prefix = "unsloth-probe-", dir = root)
        except OSError:
            continue
        if len(base) <= _MAX_PROBE_BASE_LEN:
            if fallback is not None:
                shutil.rmtree(fallback, ignore_errors = True)
            return base
        if fallback is None:
            fallback = base
        else:
            shutil.rmtree(base, ignore_errors = True)
    if fallback is not None:
        return fallback
    return tempfile.mkdtemp(prefix = "unsloth-probe-")


def _host_positive_controls(workdir: str, sentinel: str, outside: str, env: dict[str, str]) -> str:
    """Prove on the host that every control would otherwise come out the other
    way. Returns "" when the host is sane, else why the probe cannot conclude."""
    try:
        with open(sentinel, encoding = "utf-8") as handle:
            if handle.read() != _SENTINEL_TOKEN:
                return "the probe sentinel did not contain what was written to it"
    except OSError as exc:
        return f"the host itself could not read the probe sentinel ({exc})"
    try:
        with open(outside, "w", encoding = "utf-8") as handle:
            handle.write("host-write-ok")
        os.unlink(outside)
    except OSError as exc:
        return f"the host itself could not write outside the probe workdir ({exc})"
    try:
        completed = subprocess.run(
            (sys.executable, "-I", "-S", "-c", _host_payload(workdir)),
            cwd = workdir,
            env = env,
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the host itself could not run the probe's positive controls ({exc})"
    if completed.returncode != 0 or completed.stdout.strip() != PROBE_TOKEN.encode():
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()[-300:]
        return (
            "the host itself could not pass the probe's positive controls "
            f"(exit {completed.returncode}): {detail or 'no output'}"
        )
    return ""


def probe(backend: Any, *, force: bool = False) -> tuple[bool, str]:
    """Never raises: every failure becomes ``(False, reason)``, since the caller
    either falls back or refuses."""
    backend_name = str(getattr(backend, "BACKEND_NAME", "unknown"))
    try:
        from .os_sandbox import ToolLaunchPlan, _runtime_identity
        key = (backend_name, _runtime_identity())
    except Exception as exc:  # noqa: BLE001 - a probe never breaks its caller
        return False, f"the sandbox probe could not identify this runtime: {exc}"

    if not force:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    available, reason = _run_probe(backend, backend_name, ToolLaunchPlan)
    _cache_put(key, available, reason)
    return available, reason


def _run_probe(backend: Any, backend_name: str, plan_cls: Any) -> tuple[bool, str]:
    """The verdict is reached INSIDE the try: the escape check reads a host file
    the process may have written, and a cleanup that ran first would report every
    escape as a pass."""
    base = None
    prepared = None
    abstract_listener = None
    try:
        base = _probe_base()
        workdir = os.path.join(base, "work")
        os.mkdir(workdir)
        # The host's would test a directory the sandbox must not expose.
        temp_dir = os.path.join(workdir, "tmp")
        os.mkdir(temp_dir)
        sentinel = os.path.join(base, "host-sentinel.txt")
        with open(sentinel, "w", encoding = "utf-8") as handle:
            handle.write(_SENTINEL_TOKEN)
        escape = os.path.join(workdir, "escape")
        os.symlink(sentinel, escape)
        outside = os.path.join(base, "outside-write.txt")
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": workdir,
            "TMPDIR": temp_dir,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
        }

        blocked = _host_positive_controls(workdir, sentinel, outside, env)
        if blocked:
            return False, blocked

        # Asking a confined process to fail at appending to a file the host
        # cannot write either proves nothing, so that leg is dropped, not passed.
        interpreter_writable = os.access(sys.executable, os.W_OK)
        # Proven reachable from out here first, so a refusal inside is the scope.
        abstract, abstract_listener = _abstract_control()

        plan = plan_cls(
            argv = (
                sys.executable,
                "-I",
                "-S",
                "-c",
                _payload(workdir, sentinel, escape, outside, interpreter_writable, abstract),
            ),
            workdir = workdir,
            env = env,
            # A setuid bwrap cannot raise privileges once no_new_privs is set,
            # so without this the probe qualifies a backend whose every real
            # launch dies after Popen, where auto can no longer fall back.
            preexec_fn = _no_new_privs,
            requested_mode = "required",
            execution_kind = "python",
        )
        prepared = backend.prepare(plan)
        completed = subprocess.run(
            prepared.argv,
            cwd = prepared.workdir,
            env = prepared.env,
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            preexec_fn = prepared.preexec_fn,
            pass_fds = tuple(prepared.pass_fds),
            timeout = PROBE_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0 or completed.stdout.strip() != PROBE_TOKEN.encode():
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()[-300:]
            if not detail:
                detail = (completed.stdout or b"").decode("utf-8", "replace").strip()[-300:]
            return False, (
                f"the {backend_name} live probe failed "
                f"(exit {completed.returncode}): {detail or 'no output'}"
            )
        if _host_saw_the_write(outside):
            return False, (
                f"the {backend_name} live probe wrote through to the host: a file created "
                "outside the sandbox workdir arrived on the real filesystem"
            )
        caveat = (
            ""
            if interpreter_writable
            else " (the interpreter is not host-writable, so that leg was skipped)"
        )
        return True, f"the {backend_name} live isolation probe passed{caveat}"
    except subprocess.TimeoutExpired:
        return False, f"the {backend_name} live probe timed out after {PROBE_TIMEOUT_SECONDS:.0f}s"
    except Exception as exc:  # noqa: BLE001 - see the docstring on probe()
        logger.debug("sandbox probe for %s could not run", backend_name, exc_info = True)
        return False, f"the {backend_name} live probe could not run: {type(exc).__name__}: {exc}"
    finally:
        if abstract_listener is not None:
            abstract_listener.close()
        if prepared is not None:
            try:
                prepared.cleanup()
            except Exception:  # noqa: BLE001 - cleanup failure is not a verdict
                logger.debug("sandbox probe cleanup failed", exc_info = True)
        if base is not None:
            shutil.rmtree(base, ignore_errors = True)
