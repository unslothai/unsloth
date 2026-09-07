# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded static Terminal discovery, not a qualified launch or an ACL grant.

All candidate filesystem operations run in the fixed Job-owned worker. Like
the Python preparation primitive, cleanup errors retain native ownership for
the enclosing launch owner to adopt. No failure retries on the broker thread.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import time

from . import terminal_native as lpac
from .admission import _capture_broker_runtime, _scanner_executable
from .identity import _local_path
from .launch_transfer import _environment
from .preparation import _check_deadline, _json, _profile_environment, _run_worker
from .profiles import WindowsRuntimeError

_MAX_RESPONSE = 65536


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_TERMINAL_RUNTIME_INVALID", message)


@dataclass(frozen = True)
class TerminalRuntimeRoots:
    argv: tuple[str, ...]
    workdir: str
    runtime_roots: tuple[str, ...]
    acl_roots: tuple[str, ...]


def _arguments(value):
    if (
        type(value) is not list
        or not 0 < len(value) <= 128
        or any(type(arg) is not str or "\0" in arg for arg in value)
        or not value[0]
        or sum(len(arg) for arg in value) > 16384
    ):
        raise _invalid("Invalid bounded Terminal argv.")
    return tuple(value)


def _input(value):
    if type(value) is not dict or set(value) != {"argv", "workdir", "env"}:
        raise _invalid("Invalid Terminal discovery fields.")
    argv = _arguments(value["argv"])
    if not _local_path(value["workdir"]):
        raise _invalid("Terminal requires an absolute local workdir.")
    env = _environment(value["env"])
    if not os.path.isabs(argv[0]) and not env.get("PATH"):
        raise _invalid("A relative Terminal executable requires its trusted PATH.")
    return {"argv": list(argv), "workdir": value["workdir"], "env": env}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()


def _inspect(value):
    """Only the fixed worker calls this; no shell or package code is executed."""
    value = _input(value)
    workdir = lpac._validate_workdir(value["workdir"])
    argv = lpac._canonical_inner_argv(tuple(value["argv"]), value["env"])
    roots = lpac._runtime_roots(workdir, argv, "terminal")
    acl_roots = tuple(root for root in roots if lpac._needs_explicit_acl(root))
    lpac._validate_runtime_trees(acl_roots)
    return TerminalRuntimeRoots(argv, workdir, roots, acl_roots)


def _paths(value):
    if (
        type(value) is not list
        or len(value) > 64
        or not all(_local_path(path) for path in value)
        or len({_spelling(path) for path in value}) != len(value)
    ):
        raise _invalid("Invalid Terminal runtime root list.")
    return tuple(value)


def _spelling(path):
    return os.path.normcase(os.path.normpath(path))


def _matches_selection(executable, request):
    selected = request["argv"][0]
    if os.path.isabs(selected):
        return _spelling(executable) == _spelling(selected)
    # Only explicit absolute PATH entries may resolve a relative native shell.
    # No broker filesystem lookup, current-directory search or alias fallback.
    if os.path.basename(selected) != selected:
        return False
    names = (selected, selected + ".exe")
    return any(
        _local_path(directory)
        and any(_spelling(executable) == _spelling(os.path.join(directory, name)) for name in names)
        for directory in request["env"].get("PATH", "").split(os.pathsep)
    )


def _within(path, root):
    try:
        return os.path.commonpath((_spelling(path), _spelling(root))) == _spelling(root)
    except ValueError:
        return False


def _check_roots(executable, workdir, roots, acl_roots):
    parent = os.path.dirname(executable)
    allowed = {_spelling(parent)}
    if os.path.basename(executable).lower() in {"bash", "bash.exe"}:
        allowed.add(_spelling(os.path.join(os.path.dirname(parent), "usr", "bin")))
    selected = {_spelling(root) for root in roots}
    windows = os.environ.get("SystemRoot", r"C:\Windows")
    expected_acl = {_spelling(root) for root in roots if not _within(root, windows)}
    if (
        _spelling(parent) not in selected
        or not selected <= allowed
        or {_spelling(root) for root in acl_roots} != expected_acl
        or any(_within(root, workdir) or _within(workdir, root) for root in roots)
    ):
        raise _invalid("Terminal discovery returned inconsistent runtime boundaries.")


def _response(data, pid, nonce, request):
    if type(data) is not bytes or not 0 < len(data) <= _MAX_RESPONSE:
        raise _invalid("Invalid Terminal discovery response size.")
    value = _json(data)
    if (
        type(value) is not dict
        or set(value) != {"schema", "nonce", "request_digest", "pid", "result", "error"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or value["nonce"] != nonce
        or value["request_digest"] != _digest(request)
        or type(value["pid"]) is not int
        or value["pid"] != pid
    ):
        raise _invalid("Terminal discovery belongs to a different invocation.")
    error, result = value["error"], value["result"]
    if error is not None:
        if result is not None or type(error) is not str or not 0 < len(error) <= 4096:
            raise _invalid("Malformed Terminal discovery failure.")
        raise _invalid(error)
    if type(result) is not dict or set(result) != {"argv", "workdir", "runtime_roots", "acl_roots"}:
        raise _invalid("Malformed Terminal runtime result.")
    argv = _arguments(result["argv"])
    roots, acl_roots = _paths(result["runtime_roots"]), _paths(result["acl_roots"])
    if (
        not _local_path(result["workdir"])
        or not _local_path(argv[0])
        or _spelling(result["workdir"]) != _spelling(request["workdir"])
        or not _matches_selection(argv[0], request)
        or argv[1:] != tuple(request["argv"][1:])
        or not roots
        or not set(acl_roots) <= set(roots)
    ):
        raise _invalid("Terminal result changed its arguments or runtime roots.")
    _check_roots(argv[0], result["workdir"], roots, acl_roots)
    # These are observations only. Do not reopen paths on the broker thread or
    # use this structural check as authorization for later ACLs or execution.
    return TerminalRuntimeRoots(argv, result["workdir"], roots, acl_roots)


def inspect_terminal_runtime(
    spec,
    *,
    timeout = 30,
    cancel = None,
):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise _invalid("Invalid Terminal preparation timeout.")
    deadline = time.monotonic() + timeout
    from ..os_sandbox import ToolLaunchPlan

    _check_deadline(deadline, cancel)
    if (
        type(spec) is not ToolLaunchPlan
        or spec.execution_kind != "terminal"
        or spec.requested_mode != "os_isolation_required"
        or spec.close_fds is not True
        or spec.terminate_descendants is not True
        or type(spec.argv) is not tuple
    ):
        raise _invalid("Terminal discovery requires its explicit owned Required launch plan.")
    request = _input({"argv": list(spec.argv), "workdir": spec.workdir, "env": spec.env})
    broker = _capture_broker_runtime()
    nonce = secrets.token_hex(32)
    worker = Path(__file__).with_name("terminal_runtime_worker.py")
    data, pid = _run_worker(
        [
            _scanner_executable(broker),
            "-I",
            "-S",
            "-B",
            str(worker),
            json.dumps({"schema": 1, "nonce": nonce, "input": request}),
        ],
        _profile_environment(),
        str(worker.parent),
        deadline = deadline,
        cancel = cancel,
    )
    _check_deadline(deadline, cancel)
    result = _response(data, pid, nonce, request)
    _check_deadline(deadline, cancel)
    return result
