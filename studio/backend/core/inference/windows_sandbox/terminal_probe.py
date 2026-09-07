# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed Terminal compatibility evidence, never backend qualification.

The selected shell runs only inside the raw LPAC launch prepared by
``prepare_terminal_launch``. A host shell control is deliberately omitted:
without using this owner, a same-shell child could outlive its host process.
No failed command is retried or replayed outside the sandbox.
"""

from dataclasses import dataclass
import ctypes
from ctypes import wintypes as W
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path, PureWindowsPath
import secrets
import shutil
import subprocess
import threading
import time

from ..os_sandbox import ToolLaunchPlan, spawn_prepared_launch
from .content_files import native_files
from .native_io import pipe_api
from .profiles import WindowsRuntimeError
from .terminal_launch import prepare_terminal_launch
from . import terminal_runtime as runtime

TERMINAL_CHECKS = (
    "private_workdir_io",
    "quoted_script",
    "nested_same_shell_child",
)
_TOKENS = ("INLINE_OK", "BATCH_OK", "CHILD_OK")
_MAX_OUTPUT = 16 * 1024
_FIXTURE_PREFIX = "unsloth-terminal-probe-"
_pending_fixtures = set()
_pending_lock = threading.Lock()


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_PROBE_INVALID", message)


def _failed(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED", message)


@dataclass(frozen = True)
class TerminalProbeObservations:
    """Compatibility measurements only; no availability or qualification verdict.

    ``runtime_digest`` binds canonical locations to pinned volume/file IDs and,
    for copied shells, their content digest. ``content_digest`` is empty for
    the direct Windows-serviced cmd runtime, which is not a content snapshot.
    Neither digest establishes enforcement or qualification.
    """

    selected_executable: str
    runtime_roots: tuple[str, ...]
    runtime_digest: str
    checks: tuple[str, ...]
    elapsed_seconds: float
    content_digest: str = ""


def _validate_timeout(timeout):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid fixed Terminal probe timeout.")


def _local_path(value):
    drive, tail = ntpath.splitdrive(value) if type(value) is str else ("", "")
    return not (
        type(value) is not str
        or not 0 < len(value) <= 32768
        or "\0" in value
        or value.startswith(("\\\\", "//"))
        or len(drive) != 2
        or not drive[0].isascii()
        or not drive[0].isalpha()
        or tail in ("", "\\", "/")
        or not tail.startswith(("\\", "/"))
        or ".." in PureWindowsPath(value).parts
    )


def _selected(value):
    if not _local_path(value):
        raise _invalid("The selected Terminal executable must be an absolute local path.")
    name = ntpath.basename(value).lower()
    if name == "cmd.exe":
        return "cmd"
    if name in {"bash", "bash.exe"}:
        return "bash"
    raise _invalid("Only an explicitly selected cmd.exe or bash executable can be probed.")


def _remaining(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_CANCELLED", "Fixed Terminal probe was cancelled."
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _failed("Fixed Terminal probe exceeded its deadline.")
    return remaining


def _private_fixture():
    configured = os.environ.get("TEMP")
    if configured is None or not _local_path(configured):
        raise _invalid("The configured TEMP directory is not an absolute local path.")
    parent = Path(os.path.realpath(configured))
    if not runtime._local_path(str(parent)) or not parent.is_dir():
        raise _invalid("The configured TEMP directory is unavailable.")
    files = native_files()
    files.require_ntfs(parent)
    root = parent / (_FIXTURE_PREFIX + secrets.token_hex(16))
    files.mkdir(root)
    try:
        workdir = root / "work"
        # Payload-created files need both the user and package access checks.
        # The content store's non-inheriting ACL is for broker-created files,
        # not this writable fixture. Grant inheritance only inside this new dir.
        sddl = f"O:{files.owner}D:P(A;OICI;FA;;;{files.owner})(A;OICI;FA;;;SY)"
        with files.security_attributes(sddl) as attributes:
            if not files.kernel.CreateDirectoryW(
                files.native_path(workdir), ctypes.byref(attributes)
            ):
                raise _failed("Private Terminal workdir creation failed.")
        return root, workdir, parent
    except BaseException:
        os.rmdir(root)
        raise


def _remove_fixture(root, parent):
    if not os.path.lexists(root):
        return
    if (
        root.parent != parent
        or not root.name.startswith(_FIXTURE_PREFIX)
        or getattr(os.lstat(root), "st_file_attributes", 0) & 0x400
    ):
        raise _invalid("Refusing to remove an invalid Terminal probe fixture.")
    shutil.rmtree(root)


class _FixtureOwner:
    """Retain exact cleanup scope when the facade turns an error into a result."""

    def __init__(self, root, parent):
        self.root, self.parent = root, parent
        self.launch = None
        self.closed = False
        self.lock = threading.Lock()

    def retain(
        self,
        error,
        launch = None,
    ):
        if launch is not None:
            self.launch = launch
        with _pending_lock:
            _pending_fixtures.add(self)
        error.retained_fixture_owner = self

    def cleanup(self):
        acquired = self.lock.acquire(timeout = 5)
        try:
            if not acquired:
                raise _failed("Terminal fixture cleanup is still in progress.")
            if self.closed:
                with _pending_lock:
                    _pending_fixtures.discard(self)
                return
            if self.launch is not None:
                self.launch.cleanup()
                if not self.launch.closed:
                    raise _failed("Terminal launch still owns the probe fixture.")
                self.launch = None
            _remove_fixture(self.root, self.parent)
            self.closed = True
            with _pending_lock:
                _pending_fixtures.discard(self)
        except BaseException as error:
            with _pending_lock:
                _pending_fixtures.add(self)
            failure = WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED", "Fixed Terminal probe cleanup failed."
            )
            failure.retained_fixture_owner = self
            raise failure from error
        finally:
            if acquired:
                self.lock.release()


def _retry_pending(deadline, cancel):
    with _pending_lock:
        pending = tuple(_pending_fixtures)
    for owner in pending:
        _remaining(deadline, cancel)
        owner.cleanup()
        _remaining(deadline, cancel)
    with _pending_lock:
        if _pending_fixtures:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                "A prior Terminal probe still owns its private fixture.",
            )


def _plan(family, shell, workdir, index):
    script = None
    if family == "cmd":
        if index == 0:
            command = (
                'echo INLINE_OK>"inline io.txt"&& '
                'for /f "usebackq delims=" %A in ("inline io.txt") do '
                '@if "%A"=="INLINE_OK" echo INLINE_OK'
            )
        elif index == 1:
            script = workdir / "quoted script-é.cmd"
            command = f'"{script}"'
        else:
            command = f'"{shell}" /d /s /c "echo CHILD_OK"'
        argv = (shell, "/d", "/s", "/c", command)
    else:
        if index == 0:
            command = (
                "printf '%s\\n' INLINE_OK > 'inline io.txt' && "
                "IFS= read -r line < 'inline io.txt' && "
                "test \"$line\" = INLINE_OK && printf '%s\\n' INLINE_OK"
            )
        elif index == 1:
            script = workdir / "quoted script-é.sh"
            command = ". './quoted script-é.sh'"
        else:
            # Bash reports its current executable here, including a protected
            # copy. This fixed probe does not rewrite any user tool command.
            command = '"$BASH" --noprofile --norc -c \'printf "%s\\n" CHILD_OK\''
        argv = (shell, "--noprofile", "--norc", "-c", command)
    return ToolLaunchPlan(argv, str(workdir), {}, execution_kind = "terminal"), script


def _write_script(script, family):
    data = b"@echo off\r\necho BATCH_OK\r\n" if family == "cmd" else b"printf '%s\\n' BATCH_OK\n"
    with open(script, "xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _parse_output(data, token):
    if type(data) is not bytes or not 0 < len(data) <= _MAX_OUTPUT:
        raise _failed("Fixed Terminal probe returned an invalid output size.")
    try:
        lines = data.decode("utf-8", errors = "strict").splitlines()
    except UnicodeDecodeError as error:
        raise _failed("Fixed Terminal probe output was not UTF-8.") from error
    if lines != [token]:
        tail = data[-1000:].decode("utf-8", errors = "replace")
        raise _failed(f"Fixed Terminal probe returned unexpected output: {tail!r}")


def _collect_output(process, deadline, cancel):
    handle = process.stdout.buffer.raw._handle
    api, data = pipe_api(), bytearray()
    while True:
        _remaining(deadline, cancel)
        available = W.DWORD()
        if not api.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            if ctypes.get_last_error() != 109:
                raise _failed("Fixed Terminal probe stdout query failed.")
            if process.poll() is not None:
                break
        if available.value:
            if available.value > _MAX_OUTPUT - len(data):
                raise _failed("Fixed Terminal probe output exceeded its bound.")
            buffer, count = ctypes.create_string_buffer(available.value), W.DWORD()
            if (
                not api.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None)
                or not count.value
            ):
                raise _failed("Fixed Terminal probe stdout read failed.")
            data.extend(buffer.raw[: count.value])
        elif cancel is not None:
            cancel.wait(min(0.01, _remaining(deadline, cancel)))
        else:
            time.sleep(min(0.01, _remaining(deadline, cancel)))
    return bytes(data)


def _runtime_identity(owner):
    selected = owner.selected
    rows = []
    for root in selected.runtime_roots:
        handle = owner.pins.handles.get(Path(root))
        if handle is None:
            raise _failed("Prepared Terminal launch lost a pinned runtime root.")
        info = owner.pins.api.info(handle, directory = True)
        rows.append((runtime._spelling(root), info.volume, info.index_high, info.index_low))
    executable = None
    handle = owner.pins.handles.get(Path(selected.argv[0]))
    if handle is not None:
        info = owner.pins.api.info(handle)
        executable = (info.volume, info.index_high, info.index_low, info.size)
    snapshot = getattr(owner, "snapshot", None)
    content_digest = snapshot.digest if snapshot is not None else ""
    payload = {
        "schema": 2,
        "selected_executable": runtime._spelling(selected.argv[0]),
        "selected_identity": executable,
        "runtime_roots": rows,
        "content_digest": content_digest,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()
    selected_executable = snapshot.source.argv[0] if snapshot is not None else selected.argv[0]
    return selected_executable, selected.runtime_roots, digest, content_digest


def _cleanup_prepared(prepared, owner):
    prepared.cleanup()
    if prepared.cleanup_diagnostics or not owner.closed:
        failure = WindowsRuntimeError(
            "WINDOWS_SANDBOX_CLEANUP_FAILED", "Fixed Terminal probe cleanup failed."
        )
        failure.retained_launch = owner
        raise failure


def _run_check(
    family,
    shell,
    workdir,
    index,
    check,
    token,
    deadline,
    cancel,
    store_root = None,
):
    spec, script = _plan(family, shell, workdir, index)
    prepared = prepare_terminal_launch(
        spec, timeout = _remaining(deadline, cancel), cancel = cancel, store_root = store_root
    )
    owner = prepared.spawn_callback.__self__
    try:
        current = _runtime_identity(owner)
        if _selected(current[0]) != family:
            raise _failed("Terminal preparation changed the selected shell family.")
        try:
            if script is not None:
                _write_script(script, family)
            process = spawn_prepared_launch(
                prepared,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                close_fds = True,
                creationflags = subprocess.CREATE_NO_WINDOW,
                cwd = prepared.workdir,
                env = prepared.env,
            )
            data = _collect_output(process, deadline, cancel)
            if process.returncode != 0:
                tail = data[-1000:].decode("utf-8", errors = "replace")
                raise _failed(f"Terminal check {check} exited with {process.returncode}: {tail}")
            _parse_output(data, token)
            return current
        except WindowsRuntimeError as error:
            if error.code in {
                "WINDOWS_SANDBOX_CANCELLED",
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                "WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED",
            }:
                raise
            raise _failed(f"Terminal check {check} could not run: {error}") from error
        except BaseException as error:
            raise _failed(f"Terminal check {check} could not run: {error}") from error
    finally:
        _cleanup_prepared(prepared, owner)


def run_terminal_probe(
    selected_executable,
    *,
    timeout = 30,
    cancel = None,
    store_root = None,
):
    """Run the three fixed shell checks and return only post-cleanup observations."""
    _validate_timeout(timeout)
    family = _selected(selected_executable)
    start = time.monotonic()
    deadline = start + timeout
    _remaining(deadline, cancel)
    _retry_pending(deadline, cancel)
    root, workdir, parent = _private_fixture()
    fixture = _FixtureOwner(root, parent)
    identity, retained = None, False
    try:
        for index, (check, token) in enumerate(zip(TERMINAL_CHECKS, _TOKENS)):
            try:
                current = _run_check(
                    family,
                    selected_executable,
                    workdir,
                    index,
                    check,
                    token,
                    deadline,
                    cancel,
                    store_root = store_root,
                )
            except BaseException as error:
                launch = getattr(error, "retained_launch", None)
                if launch is not None:
                    retained = True
                    fixture.retain(error, launch)
                raise
            if identity is None:
                identity = current
            elif current != identity:
                raise _failed("Terminal runtime identity changed between fixed checks.")
        _remaining(deadline, cancel)
    finally:
        if not retained:
            try:
                fixture.cleanup()
            except BaseException as error:
                fixture.retain(error)
                raise
    _remaining(deadline, cancel)
    if identity is None:
        raise _failed("Fixed Terminal probe produced no observations.")
    return TerminalProbeObservations(
        identity[0],
        identity[1],
        identity[2],
        TERMINAL_CHECKS,
        time.monotonic() - start,
        content_digest = identity[3],
    )
