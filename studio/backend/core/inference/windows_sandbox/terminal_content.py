# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Protected copies of selected non-system shell files, not launch authority.

Fixed preparation workers may publish these files without changing the source
installation's ACLs. The Terminal launch owner must separately acquire a lease,
grant its invocation SID read access and retain ownership through Job cleanup.
Python's single-process reader contract is not borrowed for Terminal children.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time

from .content import (
    ContentGeneration,
    MAX_BYTES,
    MAX_FILES,
    MAX_FILE_BYTES,
    RuntimeContentStore,
    SnapshotFile,
    SnapshotSpec,
)
from .dependencies import checked_path, read_regular_file
from .native_plan import ScanBounds
from .profiles import WindowsRuntimeError
from . import terminal_runtime as runtime
from .preparation import MAX_RESULT, _check_deadline, _decode, _json, _run_worker

_PROFILE = runtime._digest({"kind": "terminal-payload-copy", "version": 2})
_NO_HELPER = hashlib.sha256(b"terminal-raw-winapi-no-embedding-helper").hexdigest()
# Git's wrapper uses these directory names to locate its installation. Preserve
# existing directory structure only, never the unselected binaries beneath it.
_LAYOUT_DIRECTORIES = ("mingw64/bin", "mingw32/bin", "ucrt64/bin", "clangarm64/bin", "mingw/bin")


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_CONTENT_INVALID", message)


@dataclass(frozen = True)
class TerminalContentSnapshot:
    source: runtime.TerminalRuntimeRoots
    spec: SnapshotSpec
    executable: str
    roots: tuple[str, ...]

    @property
    def digest(self):
        return hashlib.sha256(self.spec.manifest()).hexdigest()

    def relocated(self, generation):
        """Derive argv only from an independently acquired content-store lease."""
        if type(generation) is not ContentGeneration or generation.digest != self.digest:
            raise _invalid("The leased Terminal content differs from its snapshot.")
        directory = generation.directory / "files"
        expected = {directory / item.relative_path for item in self.spec.files}
        if set(generation.files) != expected or directory / self.executable not in expected:
            raise _invalid("The Terminal generation has a different file inventory.")
        if set(generation.directories) != {directory / name for name in self.spec.directories}:
            raise _invalid("The Terminal generation has a different directory inventory.")
        return runtime.TerminalRuntimeRoots(
            (str(directory / self.executable), *self.source.argv[1:]),
            self.source.workdir,
            tuple(str(directory / root) for root in self.roots),
            # These are copy locations, never the original Program Files roots.
            tuple(str(directory / root) for root in self.roots),
        )


def plan_terminal_content(
    source,
    *,
    bounds = ScanBounds(),
    cancel = None,
):
    """Inventory only already-selected roots; run inside a hard-deadline worker.

    This does not load an executable, resolve extra DLLs, bless startup code or
    grant permissions. Unknown dynamic dependencies remain a probe failure.
    """
    if type(source) is not runtime.TerminalRuntimeRoots or type(bounds) is not ScanBounds:
        raise _invalid("Terminal content requires fixed runtime observations and bounds.")
    if any(type(value) is not int or value <= 0 for value in asdict(bounds).values()):
        raise _invalid("Invalid Terminal content scan bounds.")
    deadline = time.monotonic() + min(bounds.seconds, 120)

    def check():
        if cancel is not None and cancel.is_set():
            raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Terminal copy was cancelled.")
        if time.monotonic() >= deadline:
            raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Terminal copy scan timed out.")

    check()
    current = runtime._inspect({"argv": list(source.argv), "workdir": source.workdir, "env": {}})
    if source != current:
        raise _invalid("The selected Terminal runtime changed before its content scan.")
    executable = checked_path(current.argv[0])
    # Windows-serviced executables keep their OS ownership and servicing links.
    # Only the existing non-system Bash runtime boundary is copied here.
    if executable.name.lower() not in ("bash", "bash.exe") or not current.acl_roots:
        raise _invalid("This snapshot path requires a selected non-system Bash runtime.")
    if current.runtime_roots != current.acl_roots:
        raise _invalid("System and application runtime roots cannot share a shell snapshot.")
    runtime._check_roots(current.argv[0], current.workdir, current.runtime_roots, current.acl_roots)
    layout = executable.parent.parent
    roots = tuple(checked_path(root) for root in current.runtime_roots)
    if any(not root.is_relative_to(layout) or root == layout for root in roots):
        raise _invalid("Shell snapshot roots must preserve the selected installation layout.")
    if any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise _invalid("Overlapping Terminal snapshot roots are not permitted.")
    device = executable.stat().st_dev
    layout_directories = []
    for name in _LAYOUT_DIRECTORIES:
        check()
        try:
            marker = checked_path(layout / name)
        except FileNotFoundError:
            continue
        info = marker.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_dev != device:
            raise _invalid("Terminal layout markers must be local non-reparse directories.")
        layout_directories.append("shell/" + name)
    files: list[SnapshotFile] = []
    names: set[str] = set()
    entries = total = 0
    directories = len(layout_directories)

    def visit(directory, depth):
        nonlocal entries, directories, total
        check()
        directories += 1
        if directories > bounds.directories or depth > min(bounds.depth, 12):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_SCAN_LIMIT", "Terminal directory limit exceeded."
            )
        path = checked_path(directory)
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_dev != device:
            raise _invalid("Terminal content cannot traverse a foreign filesystem.")
        with os.scandir(path) as iterator:
            for entry in iterator:
                check()
                entries += 1
                if entries > bounds.entries:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Terminal entry limit exceeded."
                    )
                # Windows DirEntry metadata omits device and link counts. Query
                # the path without following links; do not relax either check.
                info = os.stat(entry.path, follow_symlinks = False)
                if (
                    info.st_dev != device
                    or stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & 0x400
                ):
                    raise _invalid(
                        "Terminal content contains a reparse point or foreign filesystem."
                    )
                if stat.S_ISDIR(info.st_mode):
                    visit(entry.path, depth + 1)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise _invalid("Terminal content contains a special or hardlinked file.")
                if len(files) >= MAX_FILES or info.st_size > min(MAX_FILE_BYTES, bounds.bytes):
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Terminal file limit exceeded."
                    )
                if total + info.st_size > min(MAX_BYTES, bounds.bytes):
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Terminal byte limit exceeded."
                    )
                identity, _ = read_regular_file(
                    entry.path,
                    limit = min(
                        MAX_FILE_BYTES, bounds.bytes, MAX_BYTES - total, bounds.bytes - total
                    ),
                )
                check()
                relative = "shell/" + Path(identity.path).relative_to(layout).as_posix()
                if relative.casefold() in names:
                    raise _invalid("Terminal snapshot filenames collide on Windows.")
                total += identity.size
                names.add(relative.casefold())
                files.append(SnapshotFile(identity, relative))

    for root in roots:
        visit(root, 0)
    selected = "shell/" + executable.relative_to(layout).as_posix()
    if selected.casefold() not in names:
        raise _invalid("The selected shell executable is absent from its inventory.")
    ordered = tuple(sorted(files, key = lambda item: item.relative_path))
    # File content and selected installation identity both participate. This is
    # not a native dependency graph or approval of the runtime's initializers.
    declared_directories = tuple(sorted(layout_directories))
    inventory = runtime._digest(
        {"files": [asdict(item) for item in ordered], "directories": declared_directories}
    )
    spec = SnapshotSpec(
        ordered,
        runtime._digest({"executable": current.argv[0], "roots": current.runtime_roots}),
        inventory,
        _PROFILE,
        _NO_HELPER,
        directories = declared_directories,
    )
    spec.manifest()  # Reuse the store's relative-path, collision and size policy.
    check()
    return TerminalContentSnapshot(
        current,
        spec,
        selected,
        tuple("shell/" + root.relative_to(layout).as_posix() for root in roots),
    )


def publish_terminal_content(
    source,
    store_root,
    *,
    bounds = ScanBounds(),
    cancel = None,
):
    """Copy through the existing protected store, without any source ACL grant.

    Call only inside the fixed preparation worker: its owning Job bounds native
    filesystem operations which cannot be interrupted by a Python deadline.
    Publication confers no SID access and creates no process or execution record.
    """
    snapshot = plan_terminal_content(source, bounds = bounds, cancel = cancel)
    root = str(store_root)
    if not runtime._local_path(root) or any(
        runtime._within(root, path) or runtime._within(path, root)
        for path in (source.workdir, *source.runtime_roots)
    ):
        raise _invalid("The Terminal store must be separate from the workdir and source runtime.")
    if cancel is not None and cancel.is_set():
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Terminal copy was cancelled.")
    store = RuntimeContentStore(root)
    digest = store.publish(snapshot.spec)
    if digest != snapshot.digest:
        raise _invalid("The published Terminal snapshot has a different digest.")
    if cancel is not None and cancel.is_set():
        # A complete immutable cache generation may remain; no launch or reader
        # grant is created. The existing store owns recovery and garbage collection.
        raise WindowsRuntimeError("WINDOWS_SANDBOX_CANCELLED", "Terminal copy was cancelled.")
    return snapshot


def _snapshot_response(data, pid, nonce, request):
    if type(data) is not bytes or not 0 < len(data) <= MAX_RESULT:
        raise _invalid("Invalid Terminal snapshot response size.")
    value = _json(data)
    if (
        type(value) is not dict
        or set(value) != {"schema", "nonce", "request_digest", "pid", "snapshot", "error"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or value["nonce"] != nonce
        or value["request_digest"] != runtime._digest(request)
        or type(value["pid"]) is not int
        or value["pid"] != pid
    ):
        raise _invalid("Terminal content response belongs to another preparation.")
    if value["error"] is not None:
        if (
            value["snapshot"] is not None
            or type(value["error"]) is not str
            or not 0 < len(value["error"]) <= 4096
        ):
            raise _invalid("Invalid Terminal content failure response.")
        raise _invalid(value["error"])
    item = value["snapshot"]
    if type(item) is not dict or set(item) != {"source", "spec", "executable", "roots"}:
        raise _invalid("Invalid Terminal snapshot fields.")
    source = runtime._response(
        json.dumps(
            {
                "schema": 1,
                "nonce": nonce,
                "request_digest": runtime._digest(request["input"]),
                "pid": pid,
                "result": item["source"],
                "error": None,
            }
        ).encode(),
        pid,
        nonce,
        request["input"],
    )
    spec = item["spec"]
    if type(spec) is not dict or set(spec) != {
        "files",
        "runtime_digest",
        "dependency_digest",
        "profile_digest",
        "helper_digest",
        "directories",
    }:
        raise _invalid("Invalid Terminal snapshot manifest fields.")
    files = _decode(tuple[SnapshotFile, ...], spec["files"])
    decoded = SnapshotSpec(
        files,
        runtime_digest = _decode(str, spec["runtime_digest"]),
        dependency_digest = _decode(str, spec["dependency_digest"]),
        profile_digest = _decode(str, spec["profile_digest"]),
        helper_digest = _decode(str, spec["helper_digest"]),
        directories = _decode(tuple[str, ...], spec["directories"]),
    )
    decoded.manifest()
    layout = Path(source.argv[0]).parent.parent
    relative_executable = "shell/" + Path(source.argv[0]).relative_to(layout).as_posix()
    relative_roots = tuple(
        "shell/" + Path(root).relative_to(layout).as_posix() for root in source.runtime_roots
    )
    if (
        Path(source.argv[0]).name.lower() not in ("bash", "bash.exe")
        or source.runtime_roots != source.acl_roots
        or not source.acl_roots
        or _decode(str, item["executable"]) != relative_executable
        or _decode(tuple[str, ...], item["roots"]) != relative_roots
        or decoded.runtime_digest
        != runtime._digest({"executable": source.argv[0], "roots": source.runtime_roots})
        or decoded.dependency_digest
        != runtime._digest(
            {"files": [asdict(entry) for entry in files], "directories": decoded.directories}
        )
        or any(
            name not in {"shell/" + item for item in _LAYOUT_DIRECTORIES}
            for name in decoded.directories
        )
        or decoded.profile_digest != _PROFILE
        or decoded.helper_digest != _NO_HELPER
    ):
        raise _invalid("Terminal snapshot changed its selected layout or copy policy.")
    for entry in files:
        if not runtime._local_path(entry.source.path) or not any(
            runtime._within(entry.source.path, root) for root in source.runtime_roots
        ):
            raise _invalid("Terminal snapshot includes files outside its selected roots.")
        expected = "shell/" + Path(entry.source.path).relative_to(layout).as_posix()
        if entry.relative_path != expected:
            raise _invalid("Terminal snapshot changed the relative source layout.")
    if relative_executable not in {entry.relative_path for entry in files}:
        raise _invalid("Terminal snapshot omitted its selected executable.")
    return TerminalContentSnapshot(source, decoded, relative_executable, relative_roots)


def prepare_terminal_content(
    spec,
    store_root,
    *,
    timeout = 30,
    cancel = None,
):
    """Publish a snapshot in the fixed, Job-owned filesystem worker.

    No caller-supplied descriptors or digests enter the worker, and a failed
    worker never retries filesystem traversal on the Studio thread.
    """
    from ..os_sandbox import ToolLaunchPlan

    if (
        type(spec) is not ToolLaunchPlan
        or spec.execution_kind != "terminal"
        or spec.requested_mode != "os_isolation_required"
        or type(spec.argv) is not tuple
        or spec.close_fds is not True
        or spec.terminate_descendants is not True
        or not runtime._local_path(store_root)
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 120
    ):
        raise _invalid(
            "Terminal copying requires its explicit owned Required plan and local store."
        )
    deadline = time.monotonic() + timeout
    _check_deadline(deadline, cancel)
    request = {
        "input": runtime._input(
            {"argv": list(spec.argv), "workdir": spec.workdir, "env": spec.env}
        ),
        "store_root": store_root,
    }
    broker = runtime._capture_broker_runtime()
    nonce = secrets.token_hex(32)
    worker = Path(__file__).with_name("terminal_content_worker.py")
    data, pid = _run_worker(
        [
            runtime._scanner_executable(broker),
            "-I",
            "-S",
            "-B",
            str(worker),
            json.dumps({"schema": 1, "nonce": nonce, "request": request}),
        ],
        runtime._profile_environment(),
        str(worker.parent),
        deadline = deadline,
        cancel = cancel,
    )
    _check_deadline(deadline, cancel)
    result = _snapshot_response(data, pid, nonce, request)
    _check_deadline(deadline, cancel)
    return result
