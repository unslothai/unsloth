# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Assemble the admitted Python core launch; not backend qualification.

The fixed worker prepares filesystem access and transfers pinned content under
a hard deadline. The broker retains native channels and the startup drop gate.
This assembly is not itself runtime qualification.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as W
import io
import math
import os
from pathlib import Path
import subprocess
import secrets
import threading
import time

from . import native_compat as lpac
from .content import RuntimeContentStore
from .content_files import PathLease, _SecurityAttributes, native_files
from .dependencies import checked_path
from .host_config import HostConfiguration, HostPaths
from .native_process import create_suspended_host, attach_delayed_startup
from .preparation import prepare_runtime_snapshot
from .profiles import PYTHON_PROFILE, WindowsRuntimeError
from .identity import (
    InvocationRecipe,
    InvocationReservation,
    RuntimeReaderRecipe,
    recover_identities,
)
from .protocol import (
    LaunchBinding,
    authorize_startup,
    new_launch_nonce,
    _pipe_api,
    startup_permission,
)
from .native_io import NativePipeReader as _NativePipeReader


# Tool callers may discard both an exception and its prepared launch. Keep a
# failed cleanup's actual native/ACL owner, not just its PID or diagnostic text.
_pending_cleanup = set()
_pending_lock = threading.Lock()


def _retry_pending_cleanup(deadline, cancel):
    with _pending_lock:
        pending = tuple(_pending_cleanup)
    for owner in pending:
        _remaining(deadline, cancel)
        try:
            owner.cleanup()
        except Exception as error:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                "A previous Python sandbox still owns resources that could not be cleaned up. "
                "The new tool was not run. Retry after resolving the cleanup failure.",
            ) from error
    with _pending_lock:
        if _pending_cleanup:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                "Another Python sandbox reported a cleanup failure. The new tool was not run.",
            )


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_LAUNCH_FAILED", message)


def _remaining(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Python launch was cancelled.")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_STARTUP_TIMEOUT", "Python launch expired.")
    return min(120, remaining)


def _overlap(first, second):
    return lpac._is_within(str(first), str(second)) or lpac._is_within(str(second), str(first))


def _validate_paths(spec, store_root, prefixes):
    """Filesystem validation belongs only to the fixed preparation worker."""
    workdir = Path(lpac._validate_workdir(spec.workdir))
    script = checked_path(spec.argv[2])
    if not script.is_file() or not script.is_relative_to(workdir) or script.stat().st_nlink != 1:
        raise _invalid(
            "The Python script must be a regular unlinked file within its session workdir."
        )
    root = checked_path(store_root) if Path(store_root).exists() else Path(store_root)
    if not root.is_absolute() or _overlap(workdir, root):
        raise _invalid("The session workdir overlaps runtime storage.")
    for path in (*prefixes, Path(__file__).parents[3]):
        if _overlap(workdir, checked_path(path)):
            raise _invalid("The session workdir overlaps the interpreter or Studio installation.")
    return workdir, script


def prepare_python_launch(
    spec: ToolLaunchPlan,
    store_root,
    *,
    timeout = 90,
    cancel = None,
):
    """Prepare only the current interpreter's Python tool script, never a shell."""
    from ..os_sandbox import ToolLaunchPlan

    if (
        type(spec) is not ToolLaunchPlan
        or spec.execution_kind != "python"
        or spec.requested_mode != "os_isolation_required"
        or type(spec.argv) is not tuple
        or len(spec.argv) != 3
        or spec.argv[1] != "-u"
        or not spec.close_fds
        or not spec.terminate_descendants
    ):
        raise _invalid("The Python bootstrap requires the explicit isolated Python tool plan.")
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid Python preparation timeout.")
    deadline = time.monotonic() + timeout
    _remaining(deadline, cancel)
    _retry_pending_cleanup(deadline, cancel)
    workdir, script = Path(spec.workdir), Path(spec.argv[2])
    owner = _PythonLaunch(spec, workdir, script, None, deadline, cancel)
    return _prepare_owned_launch(owner, spec.argv[0], store_root)


def _prepare_owned_launch(owner, selected_executable, store_root):
    from .prepared import PreparedPythonLaunch
    try:
        owner.published = prepare_runtime_snapshot(
            selected_executable,
            store_root,
            timeout = _remaining(owner.deadline, owner.cancel),
            cancel = owner.cancel,
            pins = owner.pins,
            launch = owner,
        )
        prepared = PreparedPythonLaunch(
            argv = owner.spec.argv,
            workdir = str(owner.workdir),
            env = dict(owner.spec.env),
            preexec_fn = None,
            backend = "windows-lpac",
            timeout_seconds = owner.spec.timeout_seconds,
            spawn_callback = owner.spawn,
            cleanup_callbacks = [owner.cleanup],
        )
        owner.prepare(prepared)
        return prepared
    except BaseException as original:
        owner._adopt_failure(original)
        try:
            owner.cleanup()
        except Exception as cleanup:
            failure = WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED", f"Python preparation cleanup failed: {cleanup}"
            )
            failure.retained_launch = owner
            raise failure from original
        raise


class _PythonLaunch:
    def __init__(self, spec, workdir, script, published, deadline, cancel):
        self.spec, self.workdir, self.script = spec, workdir, script
        self.probe_executable = None
        self.probe_ports = ()
        self.published, self.deadline, self.cancel = published, deadline, cancel
        self.identity = self.access = self.process = None
        self.catalog = None
        self.reservation = InvocationReservation(InvocationRecipe.new())
        self.reader_name = secrets.token_hex(16)
        self.nonce = new_launch_nonce()
        self.pins = PathLease()
        self.file_pins = PathLease()
        self.handles = set()
        self.retained_processes = []
        self.retained_raw = []
        self.started = self.closed = False
        self.startup_binding = None
        self.expected_catalog_binding = None
        self.stdout = None
        self._cleanup_lock = threading.RLock()

    def _adopt_failure(self, error):
        catalog = getattr(error, "catalog_owner", None)
        if catalog is not None:
            self.catalog = catalog
        worker = getattr(error, "retained_process", None)
        processes = (
            *getattr(error, "retained_processes", ()),
            *((worker,) if worker is not None else ()),
        )
        for process in processes:
            if not any(existing is process for existing in self.retained_processes):
                self.retained_processes.append(process)
        self.handles.update(getattr(error, "retained_token_handles", ()))
        self.handles.update(getattr(error, "retained_control_handles", ()))
        job = getattr(error, "retained_job", None)
        if job is not None and not any(existing is job for existing, _pending in self.retained_raw):
            # Nested failure handlers may forward the same owner. Keep its one
            # mutable pending list: a second copy can later close reused handles.
            self.retained_raw.append((job, list(getattr(error, "retained_native_handles", ()))))

    def _pipe(self):
        api = lpac._api().kernel32
        api.CreatePipe.argtypes = [
            ctypes.POINTER(W.HANDLE),
            ctypes.POINTER(W.HANDLE),
            ctypes.POINTER(_SecurityAttributes),
            W.DWORD,
        ]
        api.CreatePipe.restype = W.BOOL
        security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, False)
        read, write = W.HANDLE(), W.HANDLE()
        if not api.CreatePipe(
            ctypes.byref(read), ctypes.byref(write), ctypes.byref(security), 4096
        ):
            raise lpac._winerror("CreatePipe(Python launch)")
        self.handles.update((read.value, write.value))
        return read.value, write.value

    def _close_handle(self, handle):
        if handle in self.handles:
            if not lpac._api().kernel32.CloseHandle(handle):
                raise lpac._winerror("CloseHandle(Python launch)")
            self.handles.remove(handle)

    def prepare_files(self):
        """Run only in the Job-owned preparation worker, before its handoff."""
        _remaining(self.deadline, self.cancel)
        recover_identities()
        reader = RuntimeReaderRecipe(
            self.published.store_root, self.reader_name, self.published.content_digest
        )
        if self.probe_executable is None:
            self.identity = self.reservation.create(self.workdir, reader = reader)
        else:
            self.identity = self.reservation.create_private(reader = reader)
            from .probe import prepare_probe_files
            prepare_probe_files(self)
        identity = self.identity
        from .private_catalog import prepare_private_catalog

        self.catalog = prepare_private_catalog(
            Path(identity.private_temp), package_sid = identity.sid_string
        )
        lpac._grant_modify(str(self.workdir), identity.sid)
        lpac._grant_modify(identity.private_temp, identity.sid)
        if self.probe_executable is not None:
            lpac._grant_read_execute(str(self.script), identity.sid)
            self.file_pins.file(self.script)
        for path in identity.traverse_roots:
            lpac._grant_traverse(path, identity.sid)
        _remaining(self.deadline, self.cancel)
        store = RuntimeContentStore(self.published.store_root)
        self.access = store.read_access(
            self.published.content_digest, identity.sid_string, name = self.reader_name
        )
        self.access.__enter__()
        files = self.access.generation.directory / "files"
        runtime = self.published.core.runtime
        home = files / "runtime"
        self.binary = str(files / "trusted/python_host.exe")
        api = native_files()
        sentinel = Path(identity.private_temp) / "startup-aap-control"
        api.create(sentinel, b"Unsloth startup AAP access control")
        handle = api.open(sentinel, write_dac = True)
        try:
            api.set_owned_dacl(handle, api.private_sddl + "(A;;FR;;;S-1-15-2-1)")
        finally:
            api.kernel.CloseHandle(handle)
        self.file_pins.file(sentinel)
        mapping = {item.source.path: item.relative_path for item in self.published.spec().files}
        from .activation_plan import build_activation_plan
        from .activation_manifest import inspect_activation_image, require_empty_activation_manifest

        selected = []
        # Packages remain payload-only. Preparing a manifest context never calls
        # a package initializer or adds that image to the native preload graph.
        for item in self.published.spec().files:
            if not item.relative_path.startswith(
                "packages/"
            ) or not item.relative_path.lower().endswith((".pyd", ".dll")):
                continue
            try:
                image = inspect_activation_image(files / item.relative_path)
                manifest = require_empty_activation_manifest(image)
            except WindowsRuntimeError:
                # Unsupported semantics keep the ordinary restricted loader path.
                # They are never silently stripped or granted startup authority.
                continue
            if manifest is not None:
                selected.append(item.relative_path)
        activation = build_activation_plan(
            self.access.generation,
            self.published.spec(),
            tuple(selected),
            nonce = self.nonce,
            profile_digest = bytes.fromhex(PYTHON_PROFILE.digest),
        )
        config = HostConfiguration(
            HostPaths(
                str(home / Path(runtime.runtime_dll.file.path).name),
                str(home / "Lib"),
                str(home / "DLLs"),
                str(home),
                runtime.executable.file.path,
                runtime.prefix,
                runtime.base_prefix,
                str(files / "trusted/policy.py"),
                str(files / "trusted/sitecustomize.py"),
                str(self.workdir),
                str(self.script),
                identity.private_temp,
                str(sentinel),
            ),
            identity.sid_string,
            runtime.version,
            self.nonce,
            bytes.fromhex(PYTHON_PROFILE.digest),
            bytes.fromhex(self.published.content_digest),
            native_images = tuple(
                str(files / mapping[path])
                for path in self.published.core.dependencies.ordered_loads
            ),
            packages = tuple(
                str(files / "packages" / str(index))
                for index in range(len(runtime.package_paths))
                if any(
                    item.relative_path.startswith(f"packages/{index}/")
                    for item in self.published.spec().files
                )
            ),
            activation_plan = activation.encode(),
        )
        config_path = Path(identity.private_temp) / "startup-config"
        api.create(config_path, config.encode())
        self.config = self.file_pins.file(config_path)
        self.environment = lpac._safe_environment(
            self.spec.env, str(self.workdir), identity, self.spec.argv
        )
        self.environment = {
            key: value for key, value in self.environment.items() if key.upper() != "SYSTEMROOT"
        }
        if not os.environ.get("SystemRoot"):
            raise _invalid("The broker's Windows SystemRoot is unavailable.")
        self.environment["SystemRoot"] = str(checked_path(os.environ["SystemRoot"]))

    def prepare(self, prepared):
        _remaining(self.deadline, self.cancel)
        prepared.workdir = str(self.workdir)
        prepared.env = dict(self.environment)
        api = native_files()
        # Move the transferred config handle to channel ownership before it can
        # be inherited. The worker has exited; no filesystem reopen is needed.
        self.config = self.file_pins.handles.pop(
            Path(self.identity.private_temp) / "startup-config"
        )
        self.handles.add(self.config)
        self.status_read, self.status_write = self._pipe()
        self.ack_read, self.ack_write = self._pipe()
        output_read, self.output_write = self._pipe()
        # Keep the native handle in self.handles even if either wrapper fails.
        # Raw close removes it only after CloseHandle succeeds; no CRT fd can
        # be freed/reused behind the retained cleanup owner's back.
        self.stdout = _NativePipeReader(output_read, self._close_handle)
        self.stdout = io.BufferedReader(self.stdout)
        self.stdout = io.TextIOWrapper(self.stdout, encoding = "utf-8", errors = "replace")
        self.null = api.kernel.CreateFileW("NUL", 0x80000000, 3, None, 3, 0, None)
        if self.null == ctypes.c_void_p(-1).value:
            raise lpac._winerror("CreateFileW(Python null stdin)")
        self.handles.add(self.null)
        kernel = lpac._api().kernel32
        kernel.SetHandleInformation.argtypes = [W.HANDLE, W.DWORD, W.DWORD]
        kernel.SetHandleInformation.restype = W.BOOL
        for handle in (self.null, self.output_write, self.config, self.status_write, self.ack_read):
            if not kernel.SetHandleInformation(handle, 1, 1):
                raise lpac._winerror("SetHandleInformation(Python child channel)")
        _remaining(self.deadline, self.cancel)

    def spawn(self, prepared, kwargs):
        # Creation, owner adoption and cleanup must share one lifecycle lock.
        # Error cleanup re-enters it; another thread cannot release inherited
        # channels or mark this owner closed while CreateProcess is in flight.
        with self._cleanup_lock:
            return self._spawn_owned(prepared, kwargs)

    def _spawn_owned(self, prepared, kwargs):
        if self.started or self.closed:
            raise _invalid("A Python launch cannot be reused or replayed.")
        self.started = True
        try:
            _retry_pending_cleanup(self.deadline, self.cancel)
            if self.catalog is None or (
                self.expected_catalog_binding is not None
                and self.catalog.binding_digest != self.expected_catalog_binding
            ):
                raise _invalid("Private catalog binding changed after qualification.")
            if (
                kwargs.get("stdout") != subprocess.PIPE
                or kwargs.get("stderr") != subprocess.STDOUT
                or kwargs.get("stdin") != subprocess.DEVNULL
                or not kwargs.get("close_fds", True)
                or kwargs.get("cwd") != prepared.workdir
                or kwargs.get("env") != prepared.env
                or kwargs.get("encoding", "utf-8") != "utf-8"
                or kwargs.get("errors", "replace") != "replace"
                or not kwargs.get("text", True)
                or int(kwargs.get("creationflags", 0)) & ~0x08000000
            ):
                raise _invalid(
                    "The Python bootstrap requires Studio's closed-descriptor streaming plan."
                )
            argv = (self.binary, str(self.config), str(self.status_write), str(self.ack_read))
            try:
                self.process = create_suspended_host(
                    self.binary,
                    argv,
                    self.identity,
                    prepared.env,
                    self.identity.private_temp,
                    stdin = self.null,
                    stdout = self.output_write,
                    control_handles = (self.status_write, self.ack_read, self.config),
                    timeout = _remaining(self.deadline, self.cancel),
                    cancel = self.cancel,
                    delayed_startup = True,
                )
            except BaseException as error:
                self._adopt_failure(error)
                raise
            self.process.stdout = self.stdout
            self.access.bind_process(self.process)
            _remaining(self.deadline, self.cancel)
            if lpac._api().kernel32.ResumeThread(self.process._thread_handle) != 1:
                raise lpac._winerror("ResumeThread(Python bootstrap)")
            for handle in (
                self.null,
                self.output_write,
                self.config,
                self.status_write,
                self.ack_read,
            ):
                self._close_handle(handle)
            binding = LaunchBinding(
                self.process.pid,
                self.nonce,
                bytes.fromhex(PYTHON_PROFILE.digest),
                bytes.fromhex(self.published.content_digest),
            )
            remaining = _remaining(self.deadline, self.cancel)

            def grant_startup():
                _remaining(self.deadline, self.cancel)
                attach_delayed_startup(self.process, deadline = self.deadline, cancel = self.cancel)
                return startup_permission(
                    binding, str(self.catalog.hive_path), self.catalog.identity
                )

            self.handles.difference_update((self.status_read, self.ack_write))
            try:
                self.startup_binding = authorize_startup(
                    self.process,
                    self.status_read,
                    self.ack_write,
                    binding,
                    timeout = remaining,
                    cancel = self.cancel,
                    grant_startup = grant_startup,
                )
            except BaseException as error:
                self.handles.update(getattr(error, "retained_control_handles", ()))
                raise
            return self.process
        except BaseException as original:
            self._adopt_failure(original)
            try:
                self.cleanup()
            except Exception as cleanup:
                failure = WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CLEANUP_FAILED", f"Python launch cleanup failed: {cleanup}"
                )
                failure.retained_launch = self
                raise failure from original
            raise

    def cleanup(self):
        # A later preparation and a caller's finally can retry the same owner.
        # Do not release handles/ACEs twice, or drop the owner on any failure.
        with self._cleanup_lock:
            try:
                self._cleanup()
            except BaseException as error:
                self._adopt_failure(error)
                with _pending_lock:
                    _pending_cleanup.add(self)
                raise
            else:
                with _pending_lock:
                    _pending_cleanup.discard(self)

    def _cleanup(self):
        if self.closed:
            return
        # Never release file grants/control channels while a process may live.
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
                    raise _invalid("Python cleanup could not reap an unreturned native process.")
                while handles:
                    if not kernel.CloseHandle(handles[-1]):
                        raise lpac._winerror("CloseHandle(retained Python creation)")
                    handles.pop()
            job.close()
            self.retained_raw.pop()
        if self.access is not None and self.access.process is self.process:
            # The bound lease owns reaping even when later ACL removal fails.
            # Do not keep a second reference that mistakes its closed process
            # for an unbound one on the next cleanup attempt.
            self.process = None
        if self.process is not None:
            self.process.reap(timeout = 5)
            self.process.close()
            self.process = None
        if self.access is not None:
            self.access.close()
            self.access = self.process = None
        if self.catalog is not None:
            from .private_catalog import PrivateCatalog
            if isinstance(self.catalog, PrivateCatalog):
                self.catalog.close()
            self.catalog = None
        for handle in tuple(self.handles):
            self._close_handle(handle)
        if self.stdout is not None:
            self.stdout.close()
            self.stdout = None
        self.pins.close()
        self.file_pins.close()
        self.reservation.cleanup(in_worker = True)
        self.identity = None
        self.closed = True
