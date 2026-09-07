# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Finite filesystem-preparation handoff for the fixed Python launch worker.

No payload execution, arbitrary operations, or filesystem reopens in the parent.
The parent owns the recovery recipe before sending the request.
"""

import ctypes
from dataclasses import asdict
import os
from pathlib import Path
import re
import time

from . import native_compat as lpac
from types import SimpleNamespace
from .content_access import RuntimeReadLease
from .identity import (
    InvocationRecipe,
    InvocationReservation,
    _derived_sid,
    _local_path,
    _validate,
    _profile_location,
)
from .profiles import WindowsRuntimeError


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_PREPARATION_FAILED", message)


def _environment(value):
    if (
        type(value) is not dict
        or len(value) > 256
        or any(
            type(k) is not str or type(v) is not str or not k or "=" in k or "\0" in k + v
            for k, v in value.items()
        )
        or sum(len(k) + len(v) for k, v in value.items()) > 16384
    ):
        raise _invalid("Invalid bounded launch environment.")
    return dict(value)


def request_for(owner):
    from .launch import _PythonLaunch

    if type(owner) is not _PythonLaunch:
        raise _invalid("The fixed worker requires its broker launch owner.")
    request = {
        "recipe": asdict(owner.reservation.recipe),
        "reader": owner.reader_name,
        "nonce": owner.nonce.hex(),
    }
    if owner.probe_executable is not None:
        if owner.spec is not None or owner.workdir is not None or owner.script is not None:
            raise _invalid("A fixed probe cannot accept a tool spec or external workdir.")
        request["probe"] = owner.probe_executable
        from .probe import validate_network_ports

        request["network_ports"] = list(validate_network_ports(owner.probe_ports))
    else:
        request.update(
            argv = list(owner.spec.argv), workdir = str(owner.workdir), env = _environment(owner.spec.env)
        )
    owner.reservation.reserve_worker()
    return request


def worker_owner(request, broker, store_root):
    from .launch import _PythonLaunch, _validate_paths

    private_probe = type(request) is dict and set(request) == {
        "probe",
        "network_ports",
        "recipe",
        "reader",
        "nonce",
    }
    if (
        type(request) is not dict
        or (
            not private_probe
            and (
                set(request) != {"argv", "workdir", "env", "recipe", "reader", "nonce"}
                or type(request["argv"]) is not list
                or len(request["argv"]) != 3
                or request["argv"][:2] != [broker.executable, "-u"]
                or not _local_path(request["argv"][2])
                or not _local_path(request["workdir"])
            )
        )
        or (
            private_probe
            and (type(request["probe"]) is not str or request["probe"] != broker.executable)
        )
        or type(request["recipe"]) is not dict
        or set(request["recipe"]) != {"moniker", "owner_pid", "owner_created"}
        or type(request["reader"]) is not str
        or not re.fullmatch(r"[0-9a-f]{32}", request["reader"])
        or type(request["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", request["nonce"])
    ):
        raise _invalid("Invalid fixed Python filesystem request.")
    recipe = InvocationRecipe(**request["recipe"])
    recipe.filename()
    if recipe.owner_pid != broker.pid:
        raise _invalid("Filesystem preparation belongs to another broker.")
    spec = workdir = script = None
    if not private_probe:
        spec = SimpleNamespace(
            argv = tuple(request["argv"]),
            workdir = request["workdir"],
            env = _environment(request["env"]),
            execution_kind = "python",
        )
        workdir, script = _validate_paths(spec, store_root, (broker.prefix, broker.base_prefix))
    owner = _PythonLaunch(spec, workdir, script, None, time.monotonic() + 120, None)
    if private_probe:
        from .probe import validate_network_ports

        if type(request["network_ports"]) is not list:
            raise _invalid("Invalid fixed probe network controls.")
        owner.probe_executable = broker.executable
        owner.probe_ports = validate_network_ports(tuple(request["network_ports"]))
    owner.reservation = InvocationReservation(recipe)
    owner.reader_name, owner.nonce = request["reader"], bytes.fromhex(request["nonce"])
    return owner


def worker_response(owner, *, failed):
    collision = owner.reservation.collision_record if owner is not None else None
    if failed:
        return {"collision": collision}
    actual = owner.identity
    return {
        "collision": None,
        "sid": actual.sid_string,
        "profile": actual.profile_folder,
        "temp": actual.private_temp,
        "path": actual.manifest_path,
        "traverse": list(actual.traverse_roots),
        "workdir": str(owner.workdir),
        "script": str(owner.script),
        "env": owner.environment,
        "reader_pins": [[str(p), h] for p, h in owner.access.pins.handles.items()],
        "file_pins": [[str(p), h] for p, h in owner.file_pins.handles.items()],
    }


def adopt_failure(owner, value):
    if type(value) is not dict or set(value) != {"collision"}:
        raise _invalid("Malformed filesystem preparation failure.")
    collision = value["collision"]
    if collision is not None:
        recipe = owner.reservation.recipe
        with _derived_sid(recipe.moniker) as (_, sid):
            _validate(collision, recipe, sid)
        if collision["state"] != "collision":
            raise _invalid("Unexpected filesystem failure ownership state.")
        owner.reservation.collision_record = collision


def _with_parents(paths):
    paths = set(paths)
    return paths | {parent for path in paths for parent in path.parents}


def adopt_launch(owner, value, published, process, runtime_rows, check_deadline):
    from .preparation import _decode

    if (
        type(value) is not dict
        or set(value)
        != {
            "collision",
            "sid",
            "profile",
            "temp",
            "path",
            "traverse",
            "workdir",
            "script",
            "env",
            "reader_pins",
            "file_pins",
        }
        or value["collision"] is not None
    ):
        raise _invalid("Invalid filesystem launch handoff fields.")
    for key in ("profile", "temp", "path", "workdir", "script"):
        if not _local_path(value[key]):
            raise _invalid("Invalid filesystem handoff path.")
    recipe = owner.reservation.recipe
    profile, temporary, journal = Path(value["profile"]), Path(value["temp"]), Path(value["path"])
    expected_workdir, expected_script = owner.workdir, owner.script
    if owner.probe_executable is not None:
        from .probe import PROBE_FILENAME
        expected_workdir, expected_script = temporary, temporary / PROBE_FILENAME
    if (
        profile.name.lower() != "ac"
        or profile.parent.name != recipe.moniker
        or profile.parent.parent.name.lower() != "packages"
        or temporary != profile / "Temp"
        or journal.name != recipe.filename()
        or journal.parent.name != "python-bootstrap-v2"
        or os.path.normcase(value["workdir"]) != os.path.normcase(str(expected_workdir))
        or os.path.normcase(value["script"]) != os.path.normcase(str(expected_script))
    ):
        raise _invalid("Filesystem handoff changed the invocation paths.")
    traverse = _decode(tuple[str, ...], value["traverse"])
    ancestors = set(Path(value["workdir"]).parents) | set(temporary.parents)
    if len(set(traverse)) != len(traverse) or any(
        not _local_path(path) or Path(path) not in ancestors for path in traverse
    ):
        raise _invalid("Invalid traversal grant inventory.")
    environment = _environment(value["env"])
    reader_rows = _decode(tuple[tuple[str, int], ...], value["reader_pins"])
    file_rows = _decode(tuple[tuple[str, int], ...], value["file_pins"])
    runtime_paths = {Path(path) for path, _ in runtime_rows}
    expected_reader = _with_parents(
        runtime_paths | {Path(published.store_root) / ".readers" / (owner.reader_name + ".lock")}
    )
    expected_files = {temporary / "startup-config", temporary / "startup-aap-control"}
    if owner.probe_executable is not None:
        expected_files.add(expected_script)
        from .probe import HOST_CONTROL_FILENAME
        expected_files.add(profile / HOST_CONTROL_FILENAME)
    expected_files = _with_parents(expected_files)
    for rows, expected in ((reader_rows, expected_reader), (file_rows, expected_files)):
        if len(rows) != len(expected) or {Path(path) for path, _ in rows} != expected:
            raise _invalid("Filesystem pins differ from the expected invocation inventory.")
    all_handles = [h for rows in (runtime_rows, reader_rows, file_rows) for _, h in rows]
    if len(set(all_handles)) != len(all_handles):
        raise _invalid("Filesystem handoff aliases distinct pin handles.")
    # Allocate only a local SID. A worker's pointer never crosses the boundary.
    api, sid = lpac._api(), owner.reservation.created_sid
    derive = api.userenv.DeriveAppContainerSidFromAppContainerName
    derive.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    derive.restype = ctypes.c_long
    result = derive(recipe.moniker, ctypes.byref(sid))
    if result != 0 or not sid:
        raise lpac._hresult_error("DeriveAppContainerSidFromAppContainerName", result)
    if lpac._sid_string(api, sid) != value["sid"]:
        raise _invalid("Filesystem handoff belongs to another AppContainer SID.")
    actual_location = _profile_location(value["sid"])
    if actual_location is None or os.path.normcase(actual_location) != os.path.normcase(
        str(profile)
    ):
        raise _invalid("Filesystem handoff differs from Windows' actual profile location.")
    actual = lpac._InvocationIdentity(
        recipe.moniker,
        sid,
        value["sid"],
        str(profile),
        str(temporary),
        str(journal),
        tuple(dict.fromkeys((value["workdir"], str(temporary), *traverse))),
        traverse,
        recipe.owner_pid,
        recipe.owner_created,
    )
    owner.reservation.identity = owner.identity = actual
    owner.reservation.created_sid = ctypes.c_void_p()
    owner.reservation.path = journal
    owner.workdir, owner.script = Path(value["workdir"]), Path(value["script"])
    if owner.probe_executable is not None:
        from ..os_sandbox import ToolLaunchPlan
        owner.spec = ToolLaunchPlan(
            argv = (owner.probe_executable, "-u", str(owner.script)),
            workdir = str(owner.workdir),
            env = {},
            execution_kind = "python",
        )
    owner.environment = environment
    owner.binary = str(
        Path(published.store_root) / published.content_digest / "files/trusted/python_host.exe"
    )
    owner.access = RuntimeReadLease._from_handoff(published, owner.reader_name, actual.sid_string)
    owner.access.pins.duplicate_from(process._handle, reader_rows, check_deadline)
    owner.file_pins.duplicate_from(process._handle, file_rows, check_deadline)


def release_worker_pins(owner):
    """After ACK or failure, release local handles only; parent owns recovery."""
    if owner is None:
        return
    if owner.access is not None:
        owner.access.pins.close()
    owner.file_pins.close()
    for sid in (
        owner.reservation.created_sid,
        owner.identity.sid if owner.identity is not None else None,
    ):
        if sid and lpac._api().advapi32.FreeSid(sid):
            raise _invalid("Worker could not release its local SID allocation.")
