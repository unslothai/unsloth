# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Hard-deadline static admission worker, never a tool execution backend.

Only Studio's running interpreter and fixed admission entrypoint are launched.
This trusted host worker reads metadata and prepares filesystem access; it must
never load candidate libraries, run package hooks or execute a tool script.
Its Job is ownership, not LPAC.
"""

from dataclasses import asdict, dataclass, fields
import ctypes
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import types
from typing import get_args, get_origin, get_type_hints

from .admission import AdmittedCore, _BrokerRuntime, _capture_broker_runtime, _scanner_executable
from .artifacts import AdmittedArtifacts
from .content import SnapshotFile, SnapshotSpec
from .dependencies import FileIdentity, ImageManifest, NativeImage
from .native_plan import DependencyEdge, DependencyPlan, ScanBounds, SystemLoaderPolicy
from .profiles import PYTHON_PROFILE, WindowsRuntimeError
from .runtime import RuntimeDescriptor

MAX_RESULT = 8 * 1024 * 1024


@dataclass(frozen = True)
class PublishedRuntime:
    core: AdmittedCore
    artifacts: AdmittedArtifacts
    store_root: str
    content_digest: str

    def spec(self):
        return SnapshotSpec(
            self.core.files + self.artifacts.files,
            self.core.digest,
            self.core.dependencies.digest,
            PYTHON_PROFILE.digest,
            self.artifacts.digest,
        )


_SCHEMAS = {
    cls: get_type_hints(cls)
    for cls in (
        AdmittedCore,
        _BrokerRuntime,
        SnapshotFile,
        FileIdentity,
        ImageManifest,
        NativeImage,
        DependencyEdge,
        DependencyPlan,
        ScanBounds,
        SystemLoaderPolicy,
        RuntimeDescriptor,
        AdmittedArtifacts,
        PublishedRuntime,
    )
}


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_PREPARATION_FAILED", message)


def _decode(
    cls,
    value,
    depth = 0,
):
    """Fixed dataclass schemas only; no type tags, pickle or dynamic imports."""
    if depth > 20:
        raise _invalid("Preparation result nesting exceeded.")
    if cls in _SCHEMAS:
        schema = _SCHEMAS[cls]
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise _invalid("Preparation result fields do not match the schema.")
        return cls(**{key: _decode(kind, value[key], depth + 1) for key, kind in schema.items()})
    origin, args = get_origin(cls), get_args(cls)
    if origin is tuple:
        if type(value) is not list or len(value) > 16384:
            raise _invalid("Preparation sequence has an invalid size or type.")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item, depth + 1) for item in value)
        if len(value) != len(args):
            raise _invalid("Preparation tuple has the wrong size.")
        return tuple(_decode(kind, item, depth + 1) for kind, item in zip(args, value))
    if origin is types.UnionType and type(None) in args and len(args) == 2:
        return (
            None
            if value is None
            else _decode(next(x for x in args if x is not type(None)), value, depth + 1)
        )
    if cls not in (str, bool, int) or type(value) is not cls:
        raise _invalid("Preparation result has an invalid scalar type.")
    if cls is str and (len(value) > 32768 or "\0" in value):
        raise _invalid("Preparation string exceeded its bound.")
    if cls is int and not 0 <= value < 2**128:
        raise _invalid("Preparation integer exceeded its bound.")
    return value


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _invalid("Duplicate preparation field.")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook = pairs)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise _invalid("Malformed preparation data.") from error


def _check_deadline(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Runtime preparation was cancelled.")
    if time.monotonic() >= deadline:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PREPARATION_TIMEOUT", "Runtime preparation exceeded its hard deadline."
        )


class _WorkerChannels:
    """Own native channels until checked close; no CRT descriptor transfers."""

    def __init__(self):
        from . import native_compat as lpac
        self.api = lpac._api().kernel32
        self.handles = set()

    def pipe(self):
        from . import native_compat as lpac
        from .content_files import _SecurityAttributes

        self.api.CreatePipe.argtypes = [
            ctypes.POINTER(W.HANDLE),
            ctypes.POINTER(W.HANDLE),
            ctypes.POINTER(_SecurityAttributes),
            W.DWORD,
        ]
        self.api.CreatePipe.restype = W.BOOL
        security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, False)
        read, write = W.HANDLE(), W.HANDLE()
        if not self.api.CreatePipe(
            ctypes.byref(read), ctypes.byref(write), ctypes.byref(security), 4096
        ):
            raise lpac._winerror("CreatePipe(static preparation)")
        self.handles.update((read.value, write.value))
        return read.value, write.value

    def null_input(self):
        from . import native_compat as lpac
        from .content_files import native_files

        handle = native_files().kernel.CreateFileW("NUL", 0x80000000, 3, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise lpac._winerror("CreateFileW(static preparation stdin)")
        self.handles.add(handle)
        return handle

    def close(self, handle):
        from . import native_compat as lpac
        if handle in self.handles:
            if not self.api.CloseHandle(handle):
                raise lpac._winerror("CloseHandle(static preparation channel)")
            self.handles.remove(handle)


def _run_worker(
    argv,
    environment,
    directory,
    *,
    deadline,
    cancel,
    transfer = None,
):
    """Own a fixed worker through EOF and exit, optionally handing off its pins.

    transfer is broker code, never a request operation or payload callback. It
    validates one length-prefixed response and duplicates pins before returning
    the fixed 32-byte acknowledgement. Its owner must retain partial duplicates
    on failure. Acknowledgement alone is not success: EOF and zero exit follow.
    """
    from . import native_compat as lpac
    from .protocol import _pipe_api

    _check_deadline(deadline, cancel)
    command = subprocess.list2cmdline(argv)
    if len(command.encode("utf-16-le")) // 2 >= 32767 or "\0" in command:
        raise _invalid("Preparation request exceeds the Windows command-line bound.")
    api = lpac._api().kernel32
    info = lpac._PROCESS_INFORMATION()
    job, attribute_list = None, None
    channels = _WorkerChannels()
    process = None
    try:
        job = lpac._create_job(None, active_process_limit = 1)
        reader, writer = channels.pipe()
        if transfer is not None:
            input_handle, ack_write = channels.pipe()
        else:
            input_handle = channels.null_input()
        for handle in (writer, input_handle):
            os.set_handle_inheritable(handle, True)
        inherited = (W.HANDLE * 2)(writer, input_handle)
        jobs = (W.HANDLE * 1)(job._handle)
        size = ctypes.c_size_t()
        api.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        if ctypes.get_last_error() != 122 or not 0 < size.value < 65536:
            raise lpac._winerror("InitializeProcThreadAttributeList(worker size)")
        storage = ctypes.create_string_buffer(size.value)
        pointer = ctypes.cast(storage, ctypes.c_void_p)
        if not api.InitializeProcThreadAttributeList(pointer, 2, 0, ctypes.byref(size)):
            raise lpac._winerror("InitializeProcThreadAttributeList(worker)")
        attribute_list = pointer
        for key, value in ((0x20002, inherited), (0x2000D, jobs)):
            if not api.UpdateProcThreadAttribute(
                pointer, 0, key, ctypes.byref(value), ctypes.sizeof(value), None, None
            ):
                raise lpac._winerror("UpdateProcThreadAttribute(worker handles/Job)")
        startup = lpac._STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = 0x100
        startup.StartupInfo.hStdInput = inherited[1]
        # The fixed worker executes no payload. Preserve its bounded startup
        # diagnostics; any non-JSON output still fails result validation.
        startup.StartupInfo.hStdError = inherited[0]
        startup.StartupInfo.hStdOutput = inherited[0]
        startup.lpAttributeList = pointer
        _check_deadline(deadline, cancel)
        # JOB_LIST assigns ownership during creation, including if the parent
        # dies before ResumeThread. The Job handle itself is never inherited.
        if not api.CreateProcessW(
            argv[0],
            ctypes.create_unicode_buffer(command),
            None,
            None,
            True,
            0x4 | 0x400 | 0x80000 | 0x08000000,
            lpac._environment_block(environment),
            directory,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(lpac._STARTUPINFOW)),
            ctypes.byref(info),
        ):
            raise lpac._winerror("CreateProcessW(static preparation)")
        process = lpac.WindowsLpacProcess(
            tuple(argv), info.hProcess, info.hThread, int(info.dwProcessId), None, job
        )
        channels.close(writer)
        channels.close(input_handle)
        _check_deadline(deadline, cancel)
        if api.ResumeThread(info.hThread) != 1:
            raise lpac._winerror("ResumeThread(static preparation)")
        pipe_api = _pipe_api()
        result = bytearray()
        acknowledged = False
        frame_size = None
        while True:
            _check_deadline(deadline, cancel)
            available = W.DWORD()
            if not pipe_api.PeekNamedPipe(reader, None, 0, None, ctypes.byref(available), None):
                if ctypes.get_last_error() == 109:
                    break
                raise lpac._winerror("PeekNamedPipe(preparation)")
            if available.value:
                if len(result) + available.value > MAX_RESULT + (4 if transfer is not None else 0):
                    raise _invalid("Preparation result exceeded its size bound.")
                buffer = ctypes.create_string_buffer(min(available.value, 65536))
                count = W.DWORD()
                if (
                    not pipe_api.ReadFile(reader, buffer, len(buffer), ctypes.byref(count), None)
                    or not count.value
                ):
                    raise lpac._winerror("ReadFile(preparation)")
                result.extend(buffer.raw[: count.value])
                if transfer is not None:
                    if frame_size is None and len(result) >= 4:
                        frame_size = int.from_bytes(result[:4], "little")
                        if not 0 < frame_size <= MAX_RESULT:
                            raise _invalid("Invalid preparation handoff frame size.")
                    if frame_size is not None:
                        if len(result) > frame_size + 4:
                            raise _invalid("Unexpected data after preparation handoff frame.")
                        if len(result) == frame_size + 4 and not acknowledged:
                            _check_deadline(deadline, cancel)
                            if process.poll() is not None:
                                raise _invalid("Preparation worker exited before handoff.")
                            ack = transfer(process, bytes(result[4:]))
                            if type(ack) is not bytes or len(ack) != 32:
                                raise _invalid("Invalid preparation handoff acknowledgement.")
                            _check_deadline(deadline, cancel)
                            written = W.DWORD()
                            if not pipe_api.WriteFile(
                                ack_write, ack, len(ack), ctypes.byref(written), None
                            ) or written.value != len(ack):
                                raise _invalid("Incomplete preparation handoff acknowledgement.")
                            channels.close(ack_write)
                            acknowledged = True
            else:
                time.sleep(0.005)
        if transfer is not None and not acknowledged:
            raise _invalid("Preparation worker ended before a complete handoff.")
        # EOF alone is not success: a worker can close stdout and remain stuck.
        while process.poll() is None:
            _check_deadline(deadline, cancel)
            time.sleep(0.005)
        _check_deadline(deadline, cancel)
        code = process.wait(timeout = 0)
        if code != 0:
            diagnostic = bytes(result[:4096]).decode("utf-8", errors = "replace")
            raise _invalid(
                f"The static preparation worker exited unsuccessfully ({code}). {diagnostic}"
            )
        return bytes(result[4:] if transfer is not None else result), process.pid
    finally:
        cleanup_error = None
        try:
            if process is not None:
                try:
                    process.reap(timeout = 5)
                    process.close()
                except BaseException as error:
                    failure = WindowsRuntimeError(
                        "WINDOWS_SANDBOX_CLEANUP_FAILED",
                        "Static preparation worker cleanup retained ownership.",
                    )
                    failure.retained_process = process
                    # Retain Job/process ownership for the caller's recovery.
                    raise failure from error
            elif job is not None:
                from .native_process import close_unreturned_process
                close_unreturned_process(job, info.hProcess, info.hThread)
        except BaseException as error:
            cleanup_error = error
        if attribute_list is not None:
            api.DeleteProcThreadAttributeList(attribute_list)
        # A failed channel close must not mask retained process/Job ownership.
        # Try every channel, then hand the exact remaining handles to recovery.
        for handle in tuple(channels.handles):
            try:
                channels.close(handle)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = WindowsRuntimeError(
                        "WINDOWS_SANDBOX_CLEANUP_FAILED",
                        "Static preparation channel cleanup retained ownership.",
                    )
                    cleanup_error.__cause__ = error
                cleanup_error.add_note(str(error))
        if cleanup_error is not None:
            cleanup_error.retained_control_handles = tuple(channels.handles)
            raise cleanup_error


def prepare_admitted_core(
    selected_executable,
    *,
    timeout = 30,
    cancel = None,
    bounds = ScanBounds(),
):
    """Bound static runtime admission; no payload launch or qualification here."""
    return _prepare(selected_executable, timeout, cancel, bounds, None)


def cleanup_runtime_reader(
    store_root,
    name,
    sid,
    *,
    timeout = 30,
):
    """Revoke one dead launch's reader in a fixed, Job-owned deadline worker.

    Call only after its process is reaped and its pin handles are released. The
    caller retains the exact recipe on failure. This can never replay a payload.
    """
    from .content_access import appcontainer_sid

    root = os.fspath(store_root)
    if (
        type(root) is not str
        or not Path(root).is_absolute()
        or Path(root).parent == Path(root)
        or len(root) > 32768
        or "\0" in root
        or type(name) is not str
        or re.fullmatch(r"[0-9a-f]{32}", name) is None
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 120
    ):
        raise _invalid("Invalid bounded runtime-reader cleanup recipe.")
    appcontainer_sid(sid)
    broker = _capture_broker_runtime()
    worker = Path(__file__).with_name("reader_cleanup_worker.py")
    nonce = secrets.token_hex(32)
    request = json.dumps({"root": root, "name": name, "sid": sid, "nonce": nonce})
    output, pid = _run_worker(
        [_scanner_executable(broker), "-I", "-S", "-B", str(worker), request],
        {},
        str(worker.parent),
        deadline = time.monotonic() + timeout,
        cancel = None,
    )
    value = _json(output)
    if (
        type(value) is not dict
        or set(value) != {"nonce", "pid", "error"}
        or value["nonce"] != nonce
        or type(value["pid"]) is not int
        or value["pid"] != pid
    ):
        raise _invalid("Runtime-reader cleanup response belongs to another worker.")
    if value["error"] is not None:
        if type(value["error"]) is not str or len(value["error"]) > 4096:
            raise _invalid("Malformed runtime-reader cleanup failure.")
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CLEANUP_FAILED", value["error"])


def _profile_environment(environment = None):
    keys = ("LOCALAPPDATA", "USERPROFILE", "SystemRoot")
    if environment is None:
        environment = {key: os.environ[key] for key in keys if key in os.environ}
    from .identity import _local_path

    if (
        type(environment) is not dict
        or not {"LOCALAPPDATA", "USERPROFILE"} <= set(environment) <= set(keys)
        or not all(_local_path(value) for value in environment.values())
    ):
        raise _invalid("The broker's profile cleanup environment is unavailable.")
    return dict(environment)


def cleanup_invocation_profile(
    recipe,
    expected_path,
    collision_record = None,
    *,
    timeout = 30,
    environment = None,
):
    """Run the fixed profile cleanup only after owned processes/pins are gone."""
    from .identity import InvocationRecipe, _local_path

    if type(recipe) is not InvocationRecipe:
        raise _invalid("Invalid profile cleanup recipe.")
    recipe.filename()
    if (
        (
            expected_path is not None
            and (not _local_path(expected_path) or Path(expected_path).name != recipe.filename())
        )
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 120
    ):
        raise _invalid("Invalid profile cleanup bounds.")
    # These are Studio-owned host values, never tool environment fields. The
    # worker recomputes the fixed namespace and compares the journal path.
    environment = _profile_environment(environment)
    broker = _capture_broker_runtime()
    worker = Path(__file__).with_name("profile_cleanup_worker.py")
    nonce = secrets.token_hex(32)
    request = json.dumps(
        {
            "recipe": asdict(recipe),
            "path": expected_path,
            "collision": collision_record,
            "nonce": nonce,
        }
    )
    output, pid = _run_worker(
        [_scanner_executable(broker), "-I", "-S", "-B", str(worker), request],
        environment,
        str(worker.parent),
        deadline = time.monotonic() + timeout,
        cancel = None,
    )
    value = _json(output)
    if (
        type(value) is not dict
        or set(value) != {"nonce", "pid", "error"}
        or value["nonce"] != nonce
        or type(value["pid"]) is not int
        or value["pid"] != pid
    ):
        raise _invalid("Profile cleanup response belongs to another worker.")
    if value["error"] is not None:
        if type(value["error"]) is not str or len(value["error"]) > 4096:
            raise _invalid("Malformed profile cleanup failure.")
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CLEANUP_FAILED", value["error"])


def prepare_runtime_snapshot(
    selected_executable,
    store_root,
    *,
    timeout = 90,
    cancel = None,
    bounds = ScanBounds(),
    pins = None,
    launch = None,
):
    """Admit installed artifacts and publish in the same hard-deadline worker.

    store_root is a dedicated backend-owned cache, never a tool argument. A killed
    publisher leaves only quarantined build files; the next worker recovers them
    under the store's exclusive kernel lock before attempting another build.
    When pins is supplied, it owns copied locks before the worker exits and
    retains partial copies on failure. The launch owner must close it on cleanup.
    """
    root = os.fspath(store_root)
    if (
        type(root) is not str
        or not Path(root).is_absolute()
        or Path(root).parent == Path(root)
        or "\0" in root
    ):
        raise _invalid("A dedicated absolute publication root is required.")
    if pins is not None:
        from .content_files import PathLease
        if type(pins) is not PathLease or pins.handles:
            raise _invalid("Snapshot handoff requires its empty broker-owned pin lease.")
    if launch is not None and pins is None:
        raise _invalid("Filesystem launch preparation requires a pin handoff.")
    return _prepare(selected_executable, timeout, cancel, bounds, root, pins, launch)


def _prepare(
    selected_executable,
    timeout,
    cancel,
    bounds,
    store_root,
    pins = None,
    launch = None,
):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid preparation timeout.")
    deadline = time.monotonic() + timeout
    _check_deadline(deadline, cancel)
    broker = _capture_broker_runtime()
    # Compare process-owned paths before any candidate filesystem operation.
    if type(selected_executable) is not str or os.path.normcase(
        selected_executable
    ) != os.path.normcase(broker.executable):
        raise _invalid("Only the running Studio interpreter may own preparation.")
    # pefile is a pinned Studio dependency, not a package selected by the tool.
    try:
        import pefile
    except ImportError as error:
        raise _invalid("The pinned static parser is unavailable.") from error
    from .dependencies import PEFILE_VERSION

    if pefile.__version__ != PEFILE_VERSION:
        raise _invalid("The pinned static parser is unavailable.")
    worker = Path(__file__).with_name("preparation_worker.py")
    parser_root = str(Path(pefile.__file__).parent)
    nonce = secrets.token_hex(32)
    from . import launch_transfer

    launch_request = launch_transfer.request_for(launch) if launch is not None else None
    request = json.dumps(
        {
            "broker": asdict(broker),
            "bounds": asdict(bounds),
            "nonce": nonce,
            "profile": PYTHON_PROFILE.digest,
            "store_root": store_root,
            **({"pins": True} if pins is not None else {}),
            **({"launch": launch_request} if launch is not None else {}),
        },
        separators = (",", ":"),
    )
    # No environment-derived import paths, user site, activation or .pth hooks.
    # A venv redirector would need a second process. Use its already-running
    # base CPython for static scanning, retaining the original venv descriptor.
    transferred = None

    def transfer(process, data):
        nonlocal transferred
        value = _json(data)
        if type(value) is not dict or "pins" not in value:
            raise _invalid("Preparation response has no pin inventory.")
        if launch is not None:
            # Bind failure ownership too, before accepting a collision tombstone.
            if (
                "launch" not in value
                or value.get("nonce") != nonce
                or value.get("profile") != PYTHON_PROFILE.digest
                or type(value.get("pid")) is not int
                or value["pid"] != process.pid
            ):
                raise _invalid("Filesystem handoff belongs to another worker.")
        rows = _decode(tuple[tuple[str, int], ...], value["pins"])
        published = _validated_response(
            {
                key: item
                for key, item in value.items()
                if key != "pins" and not (launch is not None and key == "launch")
            },
            process.pid,
            nonce,
            broker,
            store_root,
            deadline,
            cancel,
            on_failure = (
                (lambda: launch_transfer.adopt_failure(launch, value["launch"]))
                if launch is not None
                else None
            ),
        )
        directory = Path(published.store_root) / published.content_digest
        expected = {directory / ".lease", directory / "manifest.json"}
        expected.update(directory / "files" / item.relative_path for item in published.spec().files)
        expected.update(parent for path in tuple(expected) for parent in path.parents)
        if {Path(path) for path, _ in rows} != expected or len(rows) != len(expected):
            raise _invalid("Preparation pins do not match the published runtime inventory.")
        pins.duplicate_from(process._handle, rows, lambda: _check_deadline(deadline, cancel))
        if launch is not None:
            launch_transfer.adopt_launch(
                launch,
                value["launch"],
                published,
                process,
                rows,
                lambda: _check_deadline(deadline, cancel),
            )
        transferred = published
        return bytes.fromhex(nonce)

    try:
        output, pid = _run_worker(
            [_scanner_executable(broker), "-I", "-S", "-B", str(worker), parser_root, request],
            launch.reservation.worker_environment if launch is not None else {},
            str(worker.parent),
            deadline = deadline,
            cancel = cancel,
            **({"transfer": transfer} if pins is not None else {}),
        )
    except OSError as error:
        raise _invalid(f"Static preparation could not start or communicate: {error}") from error
    if pins is not None:
        if transferred is None:
            raise _invalid("The preparation worker did not transfer its runtime pins.")
        return transferred
    return _validated_response(_json(output), pid, nonce, broker, store_root, deadline, cancel)


def _validated_response(
    value,
    pid,
    nonce,
    broker,
    store_root,
    deadline,
    cancel,
    *,
    on_failure = None,
):
    if (
        type(value) is not dict
        or set(value) != {"nonce", "profile", "pid", "core", "error", "publication"}
        or value["nonce"] != nonce
        or value["profile"] != PYTHON_PROFILE.digest
        or type(value["pid"]) is not int
        or value["pid"] != pid
    ):
        raise _invalid("Preparation response belongs to a different invocation.")
    if value["error"] is not None:
        error = value["error"]
        if (
            value["core"] is not None
            or value["publication"] is not None
            or type(error) is not dict
            or set(error) != {"code", "message"}
            or type(error["code"]) is not str
            or not error["code"].startswith("WINDOWS_SANDBOX_")
            or type(error["message"]) is not str
            or len(error["message"]) > 4096
        ):
            raise _invalid("Malformed preparation failure.")
        if on_failure is not None:
            on_failure()
        raise WindowsRuntimeError(error["code"], error["message"])
    core = _decode(AdmittedCore, value["core"])
    if (
        core.broker_pid != broker.pid
        or core.runtime.version != broker.version
        or os.path.normcase(core.runtime.executable.file.path)
        != os.path.normcase(broker.executable)
        or core.origin != "running_studio_cpython_core_v1"
        or os.path.normcase(core.runtime.runtime_dll.file.path)
        != os.path.normcase(broker.loaded_dll)
        or os.path.normcase(core.runtime.prefix) != os.path.normcase(broker.prefix)
        or os.path.normcase(core.runtime.base_prefix) != os.path.normcase(broker.base_prefix)
        or core.runtime.trust_classification != "payload_only"
        or core.dependencies.trust_classification != "payload_only"
    ):
        raise _invalid("Admission did not retain the parent runtime identity.")
    _check_deadline(deadline, cancel)
    if store_root is not None:
        import hashlib

        published = _decode(PublishedRuntime, value["publication"])
        if (
            published.core != core
            or os.path.normcase(published.store_root) != os.path.normcase(store_root)
            or published.artifacts.abi
            != f"cpython-{core.runtime.version[0]}{core.runtime.version[1]}-{core.runtime.architecture}-release"
            or hashlib.sha256(published.spec().manifest()).hexdigest() != published.content_digest
        ):
            raise _invalid("Published runtime does not match this preparation.")
        _check_deadline(deadline, cancel)
        return published
    if value["publication"] is not None:
        raise _invalid("An admission-only worker unexpectedly published content.")
    return core
