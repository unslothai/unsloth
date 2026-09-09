# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The live probe that decides whether a sandbox backend actually confines anything.

``available`` is never inferred from a binary on disk. bubblewrap installed on a
host that denies unprivileged user namespaces looks identical to a working one
until you ask it to build a sandbox, so this module builds one and checks that
the things which must fail do fail.

Every control here is paired with the same control run on the HOST first. That
pairing is the whole point, in both directions:

* a negative control (reading the sentinel must raise) means nothing unless the
  host could read that file a moment earlier -- otherwise a typo in a path reads
  as a boundary;
* a positive control (multiprocessing must still work) means nothing unless the
  host can do it either -- otherwise a quirk of the machine, such as a temp
  directory too deep for an ``AF_UNIX`` address, reads as a sandbox that broke
  something.

So the host runs the positive half by itself, in the same environment, before
the sandbox is asked to reproduce it. When the host fails, the probe says it
could not conclude rather than blaming the backend.

The same care applies to what a negative control is allowed to assume. "A write
outside the workdir must raise" sounds right and is wrong: bubblewrap mounts a
private tmpfs over ``/tmp``, which is usually where this probe's own scratch
root lives, so the write succeeds inside a perfectly good sandbox and reaches
nothing. What must not happen is the byte arriving on the host, so that is what
is checked, from out here, after the run.

The launch is built through the backend's own ``prepare()``, never a parallel
argv builder. A probe that assembles its own command line stops testing the code
path that really runs, which is how a sandbox comes to be advertised on the
strength of a command nobody executes.

Network controls are deliberately absent: this sandbox confines the filesystem
and leaves the network alone on purpose (tool calls still pip-install), so there
is nothing to assert about it. See ``os_sandbox`` for why the record says so out
loud rather than staying quiet.
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
    # Resolved at import, never inside the forked child: an import after the fork
    # can deadlock on the import lock a thread held at fork time.
    _libc = ctypes.CDLL(None, use_errno = True) if sys.platform == "linux" else None
except OSError:  # pragma: no cover - a libc that will not load
    _libc = None

# Success is this token ALONE on stdout, not merely present in it. A payload that
# printed the token early and then died would satisfy "in" and prove nothing.
PROBE_TOKEN = "UNSLOTH_SANDBOX_PROBE_OK"
_SENTINEL_TOKEN = "unsloth-host-sentinel-must-not-be-readable"
# Written by the sandboxed process to a path outside its workdir. Finding it on
# the host afterwards is the escape; the write itself raising is not required.
_OUTSIDE_WRITE_TOKEN = "unsloth-sandbox-escaped-to-the-host"

# A wedged bwrap must not hang a tool call. The work here is a few file opens and
# one tiny interpreter start; 30s is far past anything healthy and still bounded.
PROBE_TIMEOUT_SECONDS = 30.0
# Long enough that a chat's worth of tool calls pays for one probe, short enough
# that installing the AppArmor profile takes effect without restarting Studio.
_CACHE_TTL_SECONDS = 60.0
# Keyed on backend + runtime identity, both of which are few in practice. Bounded
# anyway so a pathological caller cannot grow it without limit.
_CACHE_MAX_ENTRIES = 8

# ``sun_path`` is 108 bytes including the NUL. The fd-passing control below binds
# a listener under the launch's TMPDIR, and multiprocessing appends about 32
# bytes of its own (``/pymp-XXXXXXXX/listener-XXXXXXXX``) plus this probe's
# ``/work/tmp``. A scratch root longer than this cannot host that socket, which
# would fail a POSITIVE control for a reason that has nothing to do with
# isolation -- so the root is chosen to fit rather than the failure reported.
_MAX_PROBE_BASE_LEN = 59

_cache_lock = threading.Lock()
# key -> (expires_at, available, reason)
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
            # Insertion-ordered, so this drops the oldest verdict, which is also
            # the one closest to expiring.
            _cache.pop(next(iter(_cache)))


# ── the program that runs on both sides ──────────────────────────────

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

    The sentinel is read twice, by two different names. Once directly, and once
    through a symlink that lives inside the workdir, because a boundary drawn on
    the spelling of a path rather than on the resolved target lets the second one
    straight out.

    The write legs are two different questions, and only one of them is "did it
    raise". A sandbox is entitled to hand the process a private tmpfs -- bwrap
    mounts one over /tmp, which is where this probe's own scratch root usually
    lives -- so a write outside the workdir may well succeed INSIDE and reach
    nothing. What must not happen is the byte landing on the host, and that is
    checked after the run, by the host, in ``_host_saw_the_write``. Opening the
    interpreter for append is the leg that must raise: no sandbox shadows the
    system root with something writable, so a success there is a real escape.

    Reaching a host ABSTRACT unix socket is the leg with no filesystem in it at
    all. Those live in the network namespace this sandbox deliberately shares, so
    the Landlock scope is the only thing that closes them, and whether that scope
    actually took hold cannot be read off the kernel's ABI version: an outer
    sandbox or the nesting limit can deny ``landlock_restrict_self`` on a kernel
    new enough to offer it. Proven here rather than inferred, which is the same
    rule the rest of this module follows.
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
    """What must keep working, sandbox or no sandbox.

    Run on the host first. These are the everyday things a tool call does -- write
    a file, fork a worker, shell out to python -- and a sandbox that breaks any of
    them is not usable no matter how well it confines.
    """
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
    """The full program that runs INSIDE the sandbox.

    Negatives first: a boundary that is not there should be reported as such
    rather than after the slower positive half has run. Passed as source on the
    command line rather than as a file in the workdir, so the probe does not
    depend on the backend exposing the workdir in any particular way and a
    half-written scratch file can never be mistaken for a passing run.
    """
    return (
        _PREAMBLE
        + _negative_controls(sentinel, escape, outside, interpreter_writable, abstract)
        + _positive_controls(workdir)
        + f"\nprint({PROBE_TOKEN!r})\n"
    )


def _no_new_privs() -> None:
    """PR_SET_NO_NEW_PRIVS, exactly as ``tools._sandbox_preexec`` sets it.

    Runs in the forked child, so it resolves nothing it did not already have:
    ``_libc`` is bound at import. Best effort, like the pre-exec it mirrors, but
    a failure to set it would only make the probe MORE permissive than the launch
    it stands in for, so it is reported rather than swallowed silently.
    """
    if _libc is None:
        return
    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        logger.warning("The sandbox probe could not set PR_SET_NO_NEW_PRIVS")


def _abstract_control() -> "tuple[bytes | None, Any]":
    """A host abstract socket the sandboxed payload must NOT be able to reach.

    Returns nothing where there is no scope to test: off Linux, and on a kernel
    too old for it, where ``sandbox_linux.LIMITATIONS`` already says the boundary
    is not there. Nothing either when the host cannot connect to its own socket,
    since a refusal inside would then prove nothing -- the same pairing every
    other control in here is held to.
    """
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
    """Whether the sandboxed process's write outside its workdir reached the host.

    The definitive escape check, and the only honest form of it: a private tmpfs
    makes the write succeed inside while nothing arrives here, and a sandbox is
    entitled to give the process one.
    """
    try:
        with open(outside, encoding = "utf-8") as handle:
            return _OUTSIDE_WRITE_TOKEN in handle.read()
    except OSError:
        return False


def _host_payload(workdir: str) -> str:
    """The positive half alone, for the host to prove it can do these things."""
    return _PREAMBLE + _positive_controls(workdir) + f"\nprint({PROBE_TOKEN!r})\n"


# ── host-side setup and controls ─────────────────────────────────────


def _probe_base() -> str:
    """A scratch root the fd-passing control can actually live under.

    The platform temp directory first, since that is where a probe belongs. On a
    host whose TMPDIR is deep (a workspace-scoped TMPDIR, a long home) the
    ``AF_UNIX`` address would not fit, so a shorter well-known root is used
    instead; failing that, the deep one is used anyway and the host control below
    reports honestly that nothing could be concluded.
    """
    roots: list[str | None] = [None]  # None = the platform default
    roots.extend(root for root in ("/tmp", "/var/tmp") if os.path.isdir(root))
    fallback = None
    for root in roots:
        try:
            base = tempfile.mkdtemp(prefix = "unsloth-probe-", dir = root)
        except OSError:
            continue  # an unwritable candidate is not a failure; try the next
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
    # Nothing was usable; let mkdtemp raise into the probe's own handler, which
    # turns it into a verdict rather than an exception.
    return tempfile.mkdtemp(prefix = "unsloth-probe-")


def _host_positive_controls(workdir: str, sentinel: str, outside: str, env: dict[str, str]) -> str:
    """Prove on the host that every control would otherwise come out the other way.

    Returns "" when the host is sane, or the reason the probe cannot conclude
    anything. An unreadable sentinel makes a confined read meaningless; an
    unwritable base makes a confined write meaningless; a host that cannot itself
    pass a descriptor means a sandbox failing to do so says nothing about the
    sandbox.
    """
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


# ── the probe ────────────────────────────────────────────────────────


def probe(backend: Any, *, force: bool = False) -> tuple[bool, str]:
    """Whether ``backend`` really isolates on this host, and why.

    Never raises. Every failure -- a missing backend module, a bwrap that will
    not start, a timeout, a control that came out the wrong way -- becomes
    ``(False, reason)``, because the caller either falls back to software
    safeguards or refuses; neither is served by an exception escaping here.
    """
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
    """One live launch, start to verdict.

    The verdict is reached INSIDE the try, before the finally removes the scratch
    root: the escape check reads a host file the sandboxed process may have
    written, and a cleanup that ran first would report every escape as a pass.
    """
    base = None
    prepared = None
    abstract_listener = None
    try:
        base = _probe_base()
        workdir = os.path.join(base, "work")
        os.mkdir(workdir)
        # TMPDIR inside the workdir, matching what a real tool launch gets: the
        # resource sharer needs a writable temp dir, and pointing it at the host's
        # would test a directory the sandbox is not supposed to expose.
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

        # Positive control for the interpreter leg: asking a confined process to
        # fail at appending to a file the host cannot write either proves nothing,
        # so that leg is dropped and disclosed rather than passed for free.
        interpreter_writable = os.access(sys.executable, os.W_OK)
        # And for the abstract-socket leg, which only exists where the kernel has
        # the scope to enforce it: bound and proven reachable from out here first,
        # so a refusal inside is the scope and not a socket nobody could reach.
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
            # The one part of a real launch's pre-exec that changes whether the
            # sandbox starts at all. The setsid and the rlimits are the caller's
            # concern, but PR_SET_NO_NEW_PRIVS is not: a bubblewrap installed
            # setuid (how a host with unprivileged user namespaces disabled gets
            # one at all) cannot raise privileges once it is set, so without this
            # the probe would qualify a backend on which every real launch dies
            # after Popen, where auto can no longer fall back.
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
        # Every control inside came out right; the last one can only be answered
        # from out here.
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
