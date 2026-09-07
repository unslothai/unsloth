# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Write-ahead ownership for random Python and Terminal invocation profiles.

The fixed preparation worker writes this private journal before granting access.
It is not authorization or an execution record. Concurrent hostile host mutation
remains outside the trusted-local boundary.
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes as W
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import secrets

from . import native_compat as lpac
from .content_files import PathLease, native_files
from .dependencies import checked_path
from .profiles import WindowsRuntimeError

_NAME = re.compile(
    r"(unsloth\.studio\.[0-9a-f]{32})\.([1-9][0-9]{0,9})\.([1-9][0-9]{0,19})\.json(?:\.tmp)?"
)
_LIMIT = 65536


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_IDENTITY_INVALID", message)


@dataclass(frozen = True)
class InvocationRecipe:
    moniker: str
    owner_pid: int
    owner_created: int

    def filename(self):
        if (
            type(self.moniker) is not str
            or not re.fullmatch(r"unsloth\.studio\.[0-9a-f]{32}", self.moniker)
            or type(self.owner_pid) is not int
            or not 0 < self.owner_pid < 2**32
            or type(self.owner_created) is not int
            or not 0 < self.owner_created < 2**64
        ):
            raise _invalid("Invalid broker identity recipe.")
        return f"{self.moniker}.{self.owner_pid}.{self.owner_created}.json"

    @classmethod
    def new(cls):
        owner = lpac._process_identity()
        if owner is None:
            raise _invalid("The broker's process identity is unavailable.")
        return cls(lpac._PROFILE_PREFIX + secrets.token_hex(16), *owner)


@contextmanager
def _derived_sid(moniker):
    api, sid = lpac._api(), ctypes.c_void_p()
    derive = api.userenv.DeriveAppContainerSidFromAppContainerName
    derive.argtypes = [W.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    derive.restype = ctypes.c_long
    result = derive(moniker, ctypes.byref(sid))
    if result != 0 or not sid:
        raise lpac._hresult_error("DeriveAppContainerSidFromAppContainerName", result)
    try:
        yield sid, lpac._sid_string(api, sid)
    finally:
        api.advapi32.FreeSid(sid)


@dataclass(frozen = True)
class RuntimeReaderRecipe:
    store_root: str
    name: str
    digest: str

    def validate(self):
        if (
            not _local_path(self.store_root)
            or type(self.name) is not str
            or not re.fullmatch(r"[0-9a-f]{32}", self.name)
            or type(self.digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", self.digest)
        ):
            raise _invalid("Invalid durable runtime reader recipe.")

    @classmethod
    def from_value(cls, value):
        if type(value) is not dict or set(value) != {"store_root", "name", "digest"}:
            raise _invalid("Invalid durable runtime reader fields.")
        result = cls(**value)
        result.validate()
        return result


def _profile_path(sid_text):
    location = _profile_location(sid_text)
    return None if location is None else str(checked_path(location))


def _profile_location(sid_text):
    """Ask Windows for the SID's location without reopening its filesystem."""
    api, value = lpac._api(), W.LPWSTR()
    result = api.userenv.GetAppContainerFolderPath(sid_text, ctypes.byref(value))
    try:
        if ctypes.c_uint32(result).value == 0x80070002:
            return None
        if result != 0 or not value.value:
            raise lpac._hresult_error("GetAppContainerFolderPath", result)
        if not _local_path(value.value):
            raise _invalid("Windows returned an invalid profile location.")
        return value.value
    finally:
        if value:
            api.ole32.CoTaskMemFree(value)


@contextmanager
def _journal_root():
    api = native_files()
    base = checked_path(lpac._manifest_root())
    root = base / "python-bootstrap-v2"
    with PathLease() as pins:
        pins.directory(base)
        api.require_ntfs(base)
        if not os.path.lexists(root):
            try:
                api.mkdir(root)
            except WindowsRuntimeError:
                # A concurrent trusted broker may have created the same root.
                if not root.is_dir():
                    raise
        api.require_private(pins.directory(root))
        yield root


def _write(path, value):
    encoded = json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf-8")
    if len(encoded) > _LIMIT:
        raise _invalid("Identity journal exceeded its size bound.")
    temporary = Path(str(path) + ".tmp")
    native_files().create(temporary, encoded)  # CREATE_NEW, private ACL, flushed
    os.replace(temporary, path)


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _invalid("Duplicate identity journal field.")
            result[key] = value
        return result

    with PathLease() as pins:
        handle = pins.file(path)
        api = native_files()
        api.require_private(handle)
        data = api.read(handle, _LIMIT)
    try:
        return json.loads(data, object_pairs_hook = pairs)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _invalid("Malformed identity journal.") from error


def _unlink_private(path):
    if os.path.lexists(path):
        with PathLease() as pins:
            native_files().require_private(pins.file(path))
        path.unlink()


def _payload(
    recipe,
    sid,
    state,
    workdir,
    profile = None,
    reader = None,
    *,
    private_workdir = False,
    terminal_roots = None,
):
    if terminal_roots is not None and (private_workdir or reader is not None):
        raise _invalid("Terminal ownership cannot include Python runtime access.")
    return {
        "version": 5
        if terminal_roots is not None
        else (4 if private_workdir else (2 if reader is None else 3)),
        "state": state,
        "moniker": recipe.moniker,
        "owner_pid": recipe.owner_pid,
        "owner_created": recipe.owner_created,
        "sid": sid,
        "workdir": workdir,
        "profile_folder": profile,
        **(
            {"reader": asdict(reader)}
            if reader is not None
            else ({"reader": None} if private_workdir else {})
        ),
        **({"purpose": "qualification"} if private_workdir else {}),
        **(
            {"purpose": "terminal", "runtime_roots": list(terminal_roots)}
            if terminal_roots is not None
            else {}
        ),
    }


def _validate(value, recipe, sid):
    version = value.get("version") if type(value) is dict else None
    extra = (
        {"purpose", "runtime_roots"}
        if version == 5
        else ({"reader", "purpose"} if version == 4 else ({"reader"} if version == 3 else set()))
    )
    if (
        type(value) is not dict
        or set(value) != set(_payload(recipe, sid, "creating", "")) | extra
        or type(value["version"]) is not int
        or value["version"] not in (2, 3, 4, 5)
        or value["state"] not in ("creating", "ready", "collision")
        or value["moniker"] != recipe.moniker
        or value["sid"] != sid
        or type(value["owner_pid"]) is not int
        or value["owner_pid"] != recipe.owner_pid
        or type(value["owner_created"]) is not int
        or value["owner_created"] != recipe.owner_created
        or (version != 4 and not _local_path(value["workdir"]))
        or (value["state"] != "ready" and value["profile_folder"] is not None)
        or (value["state"] == "ready" and not _local_path(value["profile_folder"]))
    ):
        raise _invalid("Identity journal does not match its broker recipe.")
    if version == 4 and (
        value["purpose"] != "qualification"
        or (value["state"] != "ready" and value["workdir"] is not None)
        or (
            value["state"] == "ready"
            and (
                not _local_path(value["workdir"])
                or Path(value["workdir"]) != Path(value["profile_folder"]) / "Temp"
            )
        )
    ):
        raise _invalid("Private qualification ownership changed its purpose or workdir.")
    if version == 5:
        _terminal_roots(value["runtime_roots"], value["workdir"])
        if value["purpose"] != "terminal":
            raise _invalid("Terminal ownership changed its purpose.")
    return (
        RuntimeReaderRecipe.from_value(value["reader"])
        if version == 3 or (version == 4 and value["reader"] is not None)
        else None
    )


def _local_path(value):
    return (
        type(value) is str
        and 0 < len(value) <= 32768
        and "\0" not in value
        and Path(value).is_absolute()
        and len(Path(value).drive) == 2
        and Path(value).drive[0].isascii()
        and Path(value).drive[0].isalpha()
        and str(Path(value)) != Path(value).anchor
        and ".." not in Path(value).parts
    )


def _terminal_roots(value, workdir):
    """Validate durable cleanup scope without rediscovering or authorizing grants."""
    if (
        type(value) is not list
        or len(value) > 64
        or not all(_local_path(root) for root in value)
        or len({os.path.normcase(os.path.normpath(root)) for root in value}) != len(value)
    ):
        raise _invalid("Invalid Terminal ownership roots.")
    for root in value:
        root, workdir = (os.path.normcase(os.path.normpath(path)) for path in (root, workdir))
        try:
            common = os.path.commonpath((root, workdir))
        except ValueError:  # Different local drives cannot overlap.
            continue
        if common in (root, workdir):
            raise _invalid("Terminal ownership overlaps its writable workdir.")
    return tuple(value)


def _identity(
    recipe,
    sid,
    sid_text,
    profile,
    workdir,
    path,
    terminal_roots = (),
):
    private_temp = os.path.join(profile, "Temp")
    lpac._validated_private_temp(profile, private_temp)
    roots = tuple(dict.fromkeys((workdir, private_temp, *terminal_roots)))
    traverse = lpac._traverse_ancestors(roots)
    return lpac._InvocationIdentity(
        recipe.moniker,
        sid,
        sid_text,
        profile,
        private_temp,
        str(path),
        (*roots, *traverse),
        traverse,
        recipe.owner_pid,
        recipe.owner_created,
    )


class InvocationReservation:
    """Broker-owned recipe exists before any worker/profile mutation."""

    def __init__(self, recipe):
        if type(recipe) is not InvocationRecipe:
            raise _invalid("A fixed invocation recipe is required.")
        recipe.filename()
        self.recipe = recipe
        self.path = self.identity = None
        self.owned = self.closed = False
        self.started = False
        self.collision_record = None
        self.reader = None
        self.created_sid = ctypes.c_void_p()
        self.worker_environment = None

    def reserve_worker(self):
        """Own recovery before the worker can mutate, even if it never replies."""
        if self.started or self.closed or self.owned:
            raise _invalid("An invocation reservation cannot be reused.")
        from .preparation import _profile_environment

        self.worker_environment = _profile_environment()
        self.started = self.owned = True
        return dict(self.worker_environment)

    def create(
        self,
        workdir,
        *,
        reader = None,
    ):
        if workdir is None:
            raise _invalid("An ordinary invocation requires an explicit workdir.")
        return self._create(workdir, reader = reader, private_workdir = False)

    def create_private(self, *, reader = None):
        """Own a fresh qualification workdir without granting any external root.

        Called only by fixed qualification preparation, never tool arguments.
        As with create(), this records ownership but does not grant access.
        """
        return self._create(None, reader = reader, private_workdir = True)

    def create_terminal(self, runtime):
        """Fixed preparation worker only: revalidate before durable ownership.

        A discovery response is not grant authority. This repeats the bounded
        filesystem checks in the caller's Job-owned worker, never the broker.
        Like create(), it does not grant ACLs or start the selected executable.
        """
        from .terminal_runtime import TerminalRuntimeRoots, _inspect

        if self.started or self.closed:
            raise _invalid("An invocation reservation cannot be reused.")
        if type(runtime) is not TerminalRuntimeRoots:
            raise _invalid("Terminal ownership requires fixed runtime observations.")
        current = _inspect({"argv": list(runtime.argv), "workdir": runtime.workdir, "env": {}})
        if current != runtime:
            raise _invalid("Terminal runtime changed before ownership preparation.")
        roots = _terminal_roots(list(current.acl_roots), current.workdir)
        return self._create(
            current.workdir, reader = None, private_workdir = False, terminal_roots = roots
        )

    def _create(
        self,
        workdir,
        *,
        reader,
        private_workdir,
        terminal_roots = None,
    ):
        if self.started or self.closed:
            raise _invalid("An invocation reservation cannot be reused.")
        self.started = True
        if reader is not None:
            if type(reader) is not RuntimeReaderRecipe:
                raise _invalid("A fixed runtime reader recipe is required.")
            reader.validate()
        self.reader = reader
        if lpac._process_identity(self.recipe.owner_pid) != (
            self.recipe.owner_pid,
            self.recipe.owner_created,
        ):
            raise _invalid("The owning broker is no longer the process in the recipe.")
        workdir = None if private_workdir else lpac._validate_workdir(str(workdir))
        with _journal_root() as root, _derived_sid(self.recipe.moniker) as (_, sid_text):
            if _profile_path(sid_text) is not None:
                raise _invalid("The invocation profile already exists; it was not reused.")
            self.path = root / self.recipe.filename()
            if os.path.lexists(self.path) or os.path.lexists(str(self.path) + ".tmp"):
                raise _invalid("The invocation ownership filename already exists.")
            # The private random name is reserved before CREATE_NEW can fail
            # partway through writing. No profile exists before durable intent.
            self.owned = True
            creating = _payload(
                self.recipe,
                sid_text,
                "creating",
                workdir,
                reader = reader,
                private_workdir = private_workdir,
                terminal_roots = terminal_roots,
            )
            _write(self.path, creating)
            result = lpac._api().userenv.CreateAppContainerProfile(
                self.recipe.moniker,
                "Unsloth Studio tool",
                "Transient zero-capability Studio LPAC",
                None,
                0,
                ctypes.byref(self.created_sid),
            )
            if ctypes.c_uint32(result).value == 0x800700B7:
                # Never delete a profile for an observed creation collision.
                self.collision_record = _payload(
                    self.recipe,
                    sid_text,
                    "collision",
                    workdir,
                    reader = reader,
                    private_workdir = private_workdir,
                    terminal_roots = terminal_roots,
                )
                _write(self.path, self.collision_record)
                raise _invalid("Windows reported an invocation identity collision.")
            if result != 0 or not self.created_sid:
                raise lpac._hresult_error("CreateAppContainerProfile", result)
            if lpac._sid_string(lpac._api(), self.created_sid) != sid_text:
                raise _invalid("Created profile SID differs from its reserved identity.")
            profile = _profile_path(sid_text)
            if profile is None:
                raise _invalid("The created profile directory is unavailable.")
            os.makedirs(os.path.join(profile, "Temp"), mode = 0o700, exist_ok = True)
            if private_workdir:
                workdir = lpac._validate_workdir(os.path.join(profile, "Temp"))
            self.identity = _identity(
                self.recipe,
                self.created_sid,
                sid_text,
                profile,
                workdir,
                self.path,
                terminal_roots or (),
            )
            self.created_sid = ctypes.c_void_p()  # now owned by the identity
            _write(
                self.path,
                _payload(
                    self.recipe,
                    sid_text,
                    "ready",
                    workdir,
                    profile,
                    reader,
                    private_workdir = private_workdir,
                    terminal_roots = terminal_roots,
                ),
            )
            return self.identity

    def cleanup(self, *, in_worker = False):
        if self.closed:
            return
        if in_worker and self.owned:
            from .preparation import cleanup_invocation_profile

            cleanup_invocation_profile(
                self.recipe,
                str(self.path) if self.path is not None else None,
                self.collision_record,
                environment = self.worker_environment,
            )
            # Only successful worker exit releases the broker's SID allocations.
            # The worker has its own derived SID; pointers never cross processes.
            if self.identity is not None:
                if self.identity.sid:
                    if lpac._api().advapi32.FreeSid(self.identity.sid):
                        raise _invalid("Could not release the broker's profile SID allocation.")
                    self.identity.sid = ctypes.c_void_p()
                self.identity.cleaned = True
            self.owned = False
        if self.owned:
            if self.worker_environment is not None:
                raise _invalid("Worker-owned preparation requires bounded recovery.")
            with _journal_root() as root:
                if self.path != root / self.recipe.filename():
                    raise _invalid("Invocation ownership directory changed.")
                if self.collision_record is not None:
                    _unlink_private(Path(str(self.path) + ".tmp"))
                    _write(self.path, self.collision_record)
                    _unlink_private(self.path)
                elif self.identity is not None:
                    if self.reader is not None:
                        _recover_runtime_reader(self.reader, self.identity.sid_string)
                    self.identity.cleanup()
                    _unlink_private(Path(str(self.path) + ".tmp"))
                else:
                    _recover(self.path, self.recipe)
        if self.created_sid:
            if lpac._api().advapi32.FreeSid(self.created_sid):
                raise _invalid("Could not release the unadopted profile SID allocation.")
            self.created_sid = ctypes.c_void_p()
        self.closed = True


def cleanup_recipe(
    recipe,
    expected_path,
    collision_record = None,
):
    """Fixed-worker cleanup for one broker-owned recipe, including live brokers."""
    recipe.filename()
    with _journal_root() as root:
        path = root / recipe.filename()
        if expected_path is not None and str(path) != expected_path:
            raise _invalid("Invocation cleanup journal directory changed.")
        if collision_record is None:
            _recover(path, recipe)
            return
        with _derived_sid(recipe.moniker) as (_, sid_text):
            _validate(collision_record, recipe, sid_text)
            if collision_record["state"] != "collision":
                raise _invalid("Invalid observed profile collision.")
            if os.path.lexists(path):
                value = _read(path)
                _validate(value, recipe, sid_text)
                if value["state"] not in ("creating", "collision") or value != {
                    **collision_record,
                    "state": value["state"],
                }:
                    raise _invalid("Collision cleanup differs from durable ownership.")
            # Persist the observed collision before removing intent. Never call
            # DeleteAppContainerProfile for this identity, including on retry.
            _unlink_private(Path(str(path) + ".tmp"))
            _write(path, collision_record)
            _unlink_private(path)


def _recover_runtime_reader(reader, sid):
    from .content import RuntimeContentStore
    from .content_access import recover_readers

    store = RuntimeContentStore(reader.store_root, _existing_only = True)
    recover_readers(store, owner = (reader.name, sid), expected_digest = reader.digest)


def _recover(path, recipe):
    temporary = Path(str(path) + ".tmp")
    if not os.path.lexists(path):
        # No published intent means profile creation could not have started.
        with _derived_sid(recipe.moniker) as (_, sid_text):
            if _profile_path(sid_text) is not None:
                raise _invalid("An existing profile has no durable ownership journal.")
        _unlink_private(temporary)
        return
    with _derived_sid(recipe.moniker) as (_, sid_text):
        value = _read(path)
        reader = _validate(value, recipe, sid_text)
        if value["state"] == "ready":
            if reader is not None:
                _recover_runtime_reader(reader, sid_text)
            profile = _profile_path(sid_text)
            if profile is None:
                profile = value["profile_folder"]
                if os.path.lexists(profile):
                    raise _invalid("Deleted profile still has an unconfirmed storage path.")
            elif profile != value["profile_folder"]:
                raise _invalid("Recovered profile path differs from its recorded SID.")
            if os.path.lexists(value["workdir"]):
                checked_path(value["workdir"])
            # _InvocationIdentity owns this separate native SID allocation.
            api, sid = lpac._api(), ctypes.c_void_p()
            result = api.userenv.DeriveAppContainerSidFromAppContainerName(
                recipe.moniker, ctypes.byref(sid)
            )
            if result != 0 or not sid:
                raise lpac._hresult_error("DeriveAppContainerSidFromAppContainerName", result)
            try:
                terminal_roots = tuple(value.get("runtime_roots", ()))
                for root in terminal_roots:
                    if os.path.lexists(root):
                        checked_path(root)
                identity = _identity(
                    recipe, sid, sid_text, profile, value["workdir"], path, terminal_roots
                )
                identity.cleanup()
                sid = ctypes.c_void_p()
            finally:
                if sid:
                    api.advapi32.FreeSid(sid)
        elif value["state"] == "creating":
            result = lpac._api().userenv.DeleteAppContainerProfile(recipe.moniker)
            if ctypes.c_uint32(result).value not in (0, 0x80070002):
                raise lpac._hresult_error("DeleteAppContainerProfile", result)
        _unlink_private(temporary)
        _unlink_private(path)


def recover_identities():
    """Reconcile dead brokers only, never the short-lived preparation worker."""
    with _journal_root() as root:
        entries: list[Path] = []
        for path in root.iterdir():
            if len(entries) >= 256 or _NAME.fullmatch(path.name) is None:
                raise _invalid("Unexpected or excessive Python identity journals.")
            entries.append(path)
        names = {path.name.removesuffix(".tmp") for path in entries}
        for name in sorted(names):
            match = _NAME.fullmatch(name)
            if match is None:
                raise _invalid("Invalid invocation ownership filename.")
            recipe = InvocationRecipe(match[1], int(match[2]), int(match[3]))
            recipe.filename()
            if lpac._process_identity(recipe.owner_pid) == (recipe.owner_pid, recipe.owner_created):
                continue
            _recover(root / name, recipe)
