# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Owned Terminal preparation and raw LPAC launch; not backend qualification.

The fixed host worker never executes the selected shell. Terminal copies only
its shell runtime; Python's embedding helper and startup token are not used.
"""

import ctypes
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time

from . import terminal_native as lpac
from .admission import _capture_broker_runtime, _scanner_executable
from .content_files import PathLease
from .identity import (
    InvocationRecipe,
    InvocationReservation,
    RuntimeReaderRecipe,
    _profile_location,
    _validate,
)
from .launch_transfer import _environment, adopt_failure
from .preparation import MAX_RESULT, _check_deadline, _decode, _json, _run_worker
from .profiles import WindowsRuntimeError
from . import terminal_runtime as runtime
from .terminal_job import TerminalJobOwner

_pending_cleanup: set["_TerminalLaunch"] = set()
_pending_lock = threading.Lock()


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_PREPARATION_FAILED", message)


def _command_line(argv):
    if Path(argv[0]).name.lower() != "cmd.exe":
        return None
    if (
        len(argv) < 3
        or argv[-2].lower() != "/c"
        or any(flag.lower() not in {"/d", "/s"} for flag in argv[1:-2])
    ):
        raise _invalid("Terminal cmd requires one explicit /c command.")
    # cmd is not a CRT argv consumer. /s removes these outer quotes while
    # preserving the command's own quotes; /d disables registry AutoRun hooks.
    # https://learn.microsoft.com/windows-server/administration/windows-commands/cmd
    return subprocess.list2cmdline((argv[0], "/d", "/s", "/c")) + ' "' + argv[-1] + '"'


def _terminal_environment(env, selected, identity):
    # Windows variable names are case-insensitive. Remove every spelling before
    # installing the fixed private values, including inherited Python settings.
    replaced = {
        "APPDATA",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "SYSTEMROOT",
        "DOCKER_HOST",
        "SSH_AUTH_SOCK",
    }
    safe = {
        k: v
        for k, v in _environment(env).items()
        if k.upper() not in replaced and not k.upper().startswith("PYTHON")
    }
    windows = os.environ.get("SystemRoot")
    if windows is None or not runtime._local_path(windows):
        raise _invalid("The broker's Windows SystemRoot is unavailable.")
    paths = tuple(dict.fromkeys((*selected.runtime_roots, str(Path(windows) / "System32"))))
    safe.update(
        {
            "SystemRoot": windows,
            "PATH": os.pathsep.join(paths),
            "APPDATA": identity.private_temp,
            "LOCALAPPDATA": identity.profile_folder,
            "HOME": selected.workdir,
            "USERPROFILE": selected.workdir,
            "TEMP": identity.private_temp,
            "TMP": identity.private_temp,
        }
    )
    return _environment(safe)


def _pin_paths(selected, temporary):
    paths = {Path(selected.workdir), Path(temporary)}
    if any(runtime._within(selected.argv[0], root) for root in selected.acl_roots):
        paths.add(Path(selected.argv[0]))
    paths.update(Path(root) for root in selected.runtime_roots)
    return paths | {parent for path in paths for parent in path.parents}


def _traversal_roots(roots, user_profile):
    """Pure counterpart of the fixed worker's ancestor selection, no path opens."""
    selected: list[str] = []
    normalized = {runtime._spelling(root) for root in roots}
    for root in roots:
        for parent in Path(root).parents:
            if parent == parent.parent:
                break
            key = runtime._spelling(str(parent))
            if key not in normalized and key not in {runtime._spelling(p) for p in selected}:
                selected.append(str(parent))
            if key == runtime._spelling(user_profile):
                break
    return tuple(selected)


def _retry_pending(deadline, cancel):
    with _pending_lock:
        pending = tuple(_pending_cleanup)
    for owner in pending:
        _check_deadline(deadline, cancel)
        owner.cleanup()
    _check_deadline(deadline, cancel)
    with _pending_lock:
        if _pending_cleanup:
            raise _invalid("Prior Terminal preparation still owns unclosed resources.")


class _TerminalLaunch:
    def __init__(
        self,
        spec,
        deadline,
        cancel,
        store_root = None,
    ):
        self.spec, self.deadline, self.cancel = spec, deadline, cancel
        self.reservation = InvocationReservation(InvocationRecipe.new())
        self.nonce = secrets.token_hex(32)
        self.request = runtime._input(
            {"argv": list(spec.argv), "workdir": spec.workdir, "env": spec.env}
        )
        self.identity = self.selected = self.native = None
        self.snapshot_store = self.reader_name = self.snapshot = self.access = None
        if Path(spec.argv[0]).name.lower() in ("bash", "bash.exe"):
            if store_root is None or not runtime._local_path(str(store_root)):
                raise _invalid("Copied Terminal runtime requires an explicit trusted store root.")
            self.snapshot_store = str(store_root)
            self.reader_name = secrets.token_hex(16)
        self.job_owner = TerminalJobOwner()
        self.pins = PathLease()
        self.handles = set()
        self.retained_processes, self.retained_raw = [], []
        self.started = self.closed = False
        self.lock = threading.Lock()

    def _adopt_failure(self, error):
        process = getattr(error, "retained_process", None)
        if process is not None and process not in self.retained_processes:
            self.retained_processes.append(process)
        self.handles.update(getattr(error, "retained_control_handles", ()))
        job = getattr(error, "retained_job", None)
        if job is not None and not any(item[0] is job for item in self.retained_raw):
            self.retained_raw.append((job, list(getattr(error, "retained_native_handles", ()))))

    def transfer(self, process, data):
        _check_deadline(self.deadline, self.cancel)
        if type(data) is not bytes or not 0 < len(data) <= (
            MAX_RESULT if self.snapshot_store else 65536
        ):
            raise _invalid("Invalid Terminal handoff size.")
        value = _json(data)
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "nonce",
                "request_digest",
                "pid",
                "result",
                "error",
                "identity",
                "pins",
                "collision",
            }
            | ({"snapshot", "reader_pins"} if self.snapshot_store else set())
            or type(value["schema"]) is not int
            or value["schema"] != (2 if self.snapshot_store else 1)
            or value["nonce"] != self.nonce
            or value["request_digest"] != runtime._digest(self.request)
            or type(value["pid"]) is not int
            or value["pid"] != process.pid
        ):
            raise _invalid("Terminal handoff belongs to another invocation.")
        if value["error"] is not None:
            if (
                type(value["error"]) is not str
                or not 0 < len(value["error"]) <= 4096
                or value["identity"] is not None
                or value["result"] is not None
            ):
                raise _invalid("Malformed Terminal preparation failure.")
            adopt_failure(self, {"collision": value["collision"]})
            raise _invalid(value["error"])
        if value["collision"] is not None:
            raise _invalid("A successful Terminal handoff cannot report a collision.")
        if self.snapshot_store:
            from .terminal_snapshot_transfer import decode_snapshot
            self.snapshot, selected = decode_snapshot(self, value, process)
        else:
            selected = runtime._response(
                json.dumps(
                    {
                        k: value[k]
                        for k in ("schema", "nonce", "request_digest", "pid", "result", "error")
                    }
                ).encode("utf-8"),
                process.pid,
                self.nonce,
                self.request,
            )
        record = value["identity"]
        recipe = self.reservation.recipe
        if type(record) is not dict or set(record) != {"record", "path", "traverse"}:
            raise _invalid("Invalid Terminal identity handoff.")
        journal = record["record"]
        if type(journal) is not dict or type(journal.get("sid")) is not str:
            raise _invalid("Missing Terminal identity journal.")
        # Allocate a local SID; a pointer from the worker is never accepted.
        api, sid = lpac._api(), self.reservation.created_sid
        derive = api.userenv.DeriveAppContainerSidFromAppContainerName
        derive.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
        derive.restype = ctypes.c_long
        result = derive(recipe.moniker, ctypes.byref(sid))
        if result != 0 or not sid:
            raise lpac._hresult_error("DeriveAppContainerSidFromAppContainerName", result)
        sid_text = lpac._sid_string(api, sid)
        reader = _validate(journal, recipe, sid_text)
        expected_reader = None
        if self.snapshot_store:
            if self.snapshot is None:
                raise _invalid("Terminal snapshot handoff lost its content inventory.")
            expected_reader = RuntimeReaderRecipe(
                self.snapshot_store, self.reader_name, self.snapshot.digest
            )
        if (
            journal["version"] != (6 if self.snapshot_store else 5)
            or journal["state"] != "ready"
            or journal["workdir"] != selected.workdir
            or (
                reader != expected_reader
                if self.snapshot_store
                else journal["runtime_roots"] != list(selected.acl_roots)
            )
        ):
            raise _invalid("Terminal handoff changed its durable runtime ownership.")
        location = _profile_location(sid_text)
        if location is None or runtime._spelling(location) != runtime._spelling(
            journal["profile_folder"]
        ):
            raise _invalid("Terminal handoff differs from Windows' profile location.")
        if not runtime._local_path(record["path"]):
            raise _invalid("Invalid Terminal journal path.")
        path, temporary = Path(record["path"]), Path(location) / "Temp"
        expected_path = (
            Path(self.reservation.worker_environment["LOCALAPPDATA"])
            / "Unsloth"
            / "Studio"
            / "lpac-manifests"
            / "python-bootstrap-v2"
            / recipe.filename()
        )
        if runtime._spelling(str(path)) != runtime._spelling(str(expected_path)):
            raise _invalid("Terminal journal belongs to another reservation.")
        traverse = _decode(tuple[str, ...], record["traverse"])
        roots = (
            selected.workdir,
            str(temporary),
            *(selected.acl_roots if not self.snapshot_store else ()),
        )
        expected_traverse = _traversal_roots(
            roots, self.reservation.worker_environment["USERPROFILE"]
        )
        if tuple(map(runtime._spelling, traverse)) != tuple(
            map(runtime._spelling, expected_traverse)
        ):
            raise _invalid("Invalid Terminal traversal inventory.")
        rows = _decode(tuple[tuple[str, int], ...], value["pins"])
        expected = _pin_paths(selected, temporary)
        if len(rows) != len(expected) or {Path(p) for p, _ in rows} != expected:
            raise _invalid("Terminal pin inventory differs from its selected runtime.")
        actual = lpac._InvocationIdentity(
            recipe.moniker,
            sid,
            sid_text,
            location,
            str(temporary),
            str(path),
            (*roots, *traverse),
            traverse,
            recipe.owner_pid,
            recipe.owner_created,
        )
        self.reservation.identity = self.identity = actual
        self.reservation.created_sid = ctypes.c_void_p()
        self.reservation.path = path
        self.selected = selected
        self.pins.duplicate_from(
            process._handle, rows, lambda: _check_deadline(self.deadline, self.cancel)
        )
        for pin_path, handle in self.pins.handles.items():
            _check_deadline(self.deadline, self.cancel)
            self.pins.api.require_path(
                handle, pin_path, directory = pin_path != Path(selected.argv[0])
            )
        if self.snapshot_store:
            from .terminal_snapshot_transfer import adopt_reader
            adopt_reader(self, value, process)
        return bytes.fromhex(self.nonce)

    def spawn(self, prepared, kwargs):
        if self.started or self.closed or self.selected is None or self.identity is None:
            raise _invalid("A Terminal launch cannot be reused or replayed.")
        self.started = True
        try:
            _check_deadline(self.deadline, self.cancel)
            if (
                prepared.argv != self.selected.argv
                or prepared.workdir != self.selected.workdir
                or prepared.env
                != _terminal_environment(self.request["env"], self.selected, self.identity)
                or kwargs.get("cwd") != prepared.workdir
                or kwargs.get("env") != prepared.env
                or int(kwargs.get("creationflags", 0)) & ~0x08000000
            ):
                raise _invalid("Terminal launch changed its prepared process plan.")
            _retry_pending(self.deadline, self.cancel)
            # Cleanup cannot revoke the SID or pins while CreateProcessW is
            # consuming them. The existing Job owns creation and descendants.
            acquired = self.lock.acquire(timeout = min(5, max(0, self.deadline - time.monotonic())))
            try:
                if not acquired or self.closed:
                    raise _invalid("Terminal launch lost its preparation ownership.")
                _check_deadline(self.deadline, self.cancel)
                process = lpac._spawn_lpac(
                    prepared,
                    kwargs,
                    self.identity,
                    command_line = _command_line(self.selected.argv),
                    before_resume = self.job_owner.bind_process,
                )
            finally:
                if acquired:
                    self.lock.release()
            _check_deadline(self.deadline, self.cancel)
            return process
        except BaseException as error:
            self._fail(error)

    def _fail(self, error):
        self._adopt_failure(error)
        try:
            self.cleanup()
        except BaseException as cleanup:
            failure = WindowsRuntimeError("WINDOWS_SANDBOX_CLEANUP_FAILED", str(cleanup))
            failure.retained_launch = self
            raise failure from error
        raise error

    def cleanup(self):
        acquired = self.lock.acquire(timeout = 5)
        try:
            if not acquired:
                raise _invalid("Terminal cleanup is still in progress.")
            if self.closed:
                with _pending_lock:
                    _pending_cleanup.discard(self)
                return
            # Reap workers and payloads before releasing any pinned path or ACE.
            while self.retained_processes:
                process = self.retained_processes[-1]
                process.reap(timeout = 5)
                process.close()
                self.retained_processes.pop()
            while self.retained_raw:
                job, handles = self.retained_raw[-1]
                kernel = lpac._api().kernel32
                if handles:
                    if not job.terminate() or kernel.WaitForSingleObject(handles[0], 5000) != 0:
                        raise _invalid("Terminal cleanup could not reap an unreturned worker.")
                    while handles:
                        if not kernel.CloseHandle(handles[-1]):
                            raise lpac._winerror("CloseHandle(Terminal worker)")
                        handles.pop()
                job.close()
                self.retained_raw.pop()
            self.job_owner.cleanup()
            if self.native is not None:
                self.native.cleanup()
            for handle in tuple(self.handles):
                if not lpac._api().kernel32.CloseHandle(handle):
                    raise lpac._winerror("CloseHandle(Terminal preparation)")
                self.handles.remove(handle)
            self.pins.close()
            if self.access is not None:
                # Terminal Job emptiness and native handle cleanup precede any
                # snapshot ACE/lease release; Python's process policy is unused.
                self.access.close()
            self.reservation.cleanup(in_worker = True)
            self.closed = True
            with _pending_lock:
                _pending_cleanup.discard(self)
        except BaseException as error:
            self._adopt_failure(error)
            with _pending_lock:
                _pending_cleanup.add(self)
            raise
        finally:
            if acquired:
                self.lock.release()


def prepare_terminal_launch(
    spec,
    *,
    timeout = 30,
    cancel = None,
    store_root = None,
):
    """Private raw-LPAC path; the backend must qualify this shell before selecting it."""
    from ..os_sandbox import ToolLaunchPlan
    from .prepared import PreparedPythonLaunch

    if (
        type(spec) is not ToolLaunchPlan
        or spec.execution_kind != "terminal"
        or spec.requested_mode != "os_isolation_required"
        or type(spec.argv) is not tuple
        or spec.close_fds is not True
        or spec.terminate_descendants is not True
    ):
        raise _invalid("Terminal requires its explicit owned Required launch plan.")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid Terminal preparation timeout.")
    deadline = time.monotonic() + timeout
    _check_deadline(deadline, cancel)
    _retry_pending(deadline, cancel)
    owner = _TerminalLaunch(spec, deadline, cancel, store_root)
    try:
        broker = _capture_broker_runtime()
        environment = owner.reservation.reserve_worker()
        request = {
            "schema": 2 if owner.snapshot_store else 1,
            "nonce": owner.nonce,
            "input": owner.request,
            "recipe": asdict(owner.reservation.recipe),
        }
        if owner.snapshot_store:
            request["content"] = {"store_root": owner.snapshot_store, "reader": owner.reader_name}
        worker = Path(__file__).with_name("terminal_prepare_worker.py")
        _run_worker(
            [_scanner_executable(broker), "-I", "-S", "-B", str(worker), json.dumps(request)],
            environment,
            str(worker.parent),
            deadline = deadline,
            cancel = cancel,
            transfer = owner.transfer,
        )
        _check_deadline(deadline, cancel)
        if owner.identity is None or owner.selected is None or not owner.pins.handles:
            raise _invalid("Terminal preparation did not transfer its owned runtime.")
        owner.native = lpac._NativeLaunchOwner(owner.identity)
        prepared = PreparedPythonLaunch(
            owner.selected.argv,
            owner.selected.workdir,
            _terminal_environment(owner.request["env"], owner.selected, owner.identity),
            None,
            "windows-lpac",
            timeout_seconds = spec.timeout_seconds,
            spawn_callback = owner.spawn,
            cleanup_callbacks = [owner.cleanup],
        )
        prepared._lpac_native_owner = owner.native
        return prepared
    except BaseException as error:
        owner._fail(error)
