# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-invocation snapshot readers, not a capability or startup approval service.

Only broker-owned store ACLs change here. The launcher still owns AppContainer
identity creation, any ancestor traversal outside the store, and the native
startup/drop gate. A random journal is durable before the first read grant.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import re
import secrets

from .content_files import PathLease
from .profiles import WindowsRuntimeError

_SID = re.compile(r"S-1-15-2(?:-[0-9]{1,10}){7}")
_READER = re.compile(r"[0-9a-f]{32}")
_ACE = re.compile(r"\(A;;([^;()]+);;;([^;()]+)\)")
READ_EXECUTE = 0x1200A9
TRAVERSE = 0x20
MAX_READERS = 128


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", message)


def appcontainer_sid(value):
    if (
        not isinstance(value, str)
        or not _SID.fullmatch(value)
        or any(str(int(part)) != part or int(part) > 0xFFFFFFFF for part in value.split("-")[4:])
    ):
        raise _invalid("A canonical per-invocation AppContainer SID is required.")
    return value


def _encode(value):
    return json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf-8")


def read_readers(store, pins):
    """Read only protected broker journals; their contents never come from tools."""
    directory = store.root / ".readers"
    store.api.require_private(pins.directory(directory))
    names = set()
    for index, path in enumerate(directory.iterdir()):
        if index >= MAX_READERS * 3:
            raise _invalid("Runtime reader journal limit exceeded.")
        if path.suffix not in (".json", ".lock", ".tmp") or not _READER.fullmatch(path.stem):
            raise _invalid("Unowned runtime reader journal entry.")
        if path.suffix == ".tmp":
            with PathLease() as temporary:
                store.api.require_private(temporary.file(path))
            continue  # No grants may be applied before atomic journal publication.
        names.add(path.name)
    readers: dict[str, dict] = {}
    sids = set()
    for name in sorted(names):
        if name.endswith(".lock"):
            if name[:-5] + ".json" not in names:
                raise _invalid("Runtime reader lock has no ownership journal.")
            continue
        if len(readers) >= MAX_READERS:
            raise _invalid("Runtime reader journal limit exceeded.")
        with PathLease() as journal_pins:
            handle = journal_pins.file(directory / name)
            store.api.require_private(handle)
            data = store.api.read(handle, 2048)
        try:
            record = json.loads(data)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise _invalid("Malformed runtime reader journal.") from exc
        if (
            not isinstance(record, dict)
            or set(record) != {"version", "digest", "sid"}
            or type(record["version"]) is not int
            or record["version"] != 1
            or not isinstance(record["digest"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["digest"])
            or _encode(record) != data
        ):
            raise _invalid("Unknown runtime reader journal schema.")
        sid = appcontainer_sid(record["sid"])
        if sid in sids:
            raise _invalid("A runtime invocation SID was reused.")
        sids.add(sid)
        readers[name[:-5]] = record
    return readers


def expected_readers(store, path, readers):
    """Only the root, selected generation and its inventoried data receive ACEs."""
    if path == store.root:
        return {record["sid"]: TRAVERSE for record in readers.values()}
    relative = path.relative_to(store.root)
    parts = relative.parts
    if not parts:
        return {}
    selected = [record for record in readers.values() if record["digest"] == parts[0]]
    if len(parts) == 1:
        return {record["sid"]: TRAVERSE for record in selected}
    if parts[1] == "files":
        return {record["sid"]: READ_EXECUTE for record in selected}
    return {}


def validate_acl(store, handle, path, readers):
    """Recognize a strict subset of journaled grants, including interrupted setup.

    Subsets permit recovery after a mid-update crash. Before launching, preparation
    separately requires this invocation's ACE on every intended object. Unknown
    ACEs are an error; they are never overwritten to repair host permissions.
    """
    text = store.api.security_text(handle)
    prefix = f"O:{store.api.owner}D:P"
    if not text.startswith(prefix):
        raise _invalid("Runtime store owner or private DACL changed.")
    body = text[len(prefix) :]
    entries = _ACE.findall(body)
    if "".join(f"(A;;{mask};;;{sid})" for mask, sid in entries) != body:
        raise _invalid("Unexpected inherited or non-allow runtime ACE.")
    allowed = expected_readers(store, path, readers)
    observed = {}
    owners = set()
    for mask, sid in entries:
        if sid in (store.api.owner, "SY"):
            if mask != "FA" or sid in owners:
                raise _invalid("Runtime store owner permissions changed.")
            owners.add(sid)
        else:
            try:
                access = {"FRFX": READ_EXECUTE, "FXFR": READ_EXECUTE, "WP": TRAVERSE}.get(mask)
                if access is None:
                    access = int(mask, 16) if mask.startswith("0x") else -1
            except ValueError as exc:
                raise _invalid("Invalid runtime read ACE mask.") from exc
            if sid in observed or allowed.get(sid) != access:
                raise _invalid("Runtime DACL changed: unowned or excessive read grant.")
            observed[sid] = access
    if owners != {store.api.owner, "SY"}:
        raise _invalid("Runtime store owner permissions missing.")
    return observed


def _targets(store, generation):
    directories = {
        store.root,
        generation.directory,
        generation.directory / "files",
        *generation.directories,
    }
    for path in (*generation.files, *generation.directories):
        directories.update(
            parent for parent in path.parents if parent.is_relative_to(generation.directory)
        )
    return sorted(directories, key = lambda path: (len(path.parts), str(path))), generation.files


def _change(store, generation, readers, sid, *, remove):
    directories, files = _targets(store, generation)
    for path in (*directories, *files):
        handle = store.api.open(path, directory = path in directories, write_dac = True)
        try:
            observed = validate_acl(store, handle, path, readers)
            desired = dict(observed)
            if remove:
                desired.pop(sid, None)
            else:
                desired[sid] = expected_readers(store, path, readers)[sid]
            if desired != observed:
                sddl = f"O:{store.api.owner}D:P(A;;FA;;;{store.api.owner})(A;;FA;;;SY)"
                sddl += "".join(
                    f"(A;;0x{mask:x};;;{identity})" for identity, mask in sorted(desired.items())
                )
                store.api.set_owned_dacl(handle, sddl)
            if validate_acl(store, handle, path, readers) != desired:
                raise _invalid("Runtime read-grant update was not enforced.")
        finally:
            store.api.kernel.CloseHandle(handle)


def _discard_unpublished(store, name):
    path = store.root / ".readers" / (name + ".tmp")
    if os.path.lexists(path):
        if os.path.lexists(path.with_suffix(".json")):
            raise _invalid("Published and unpublished reader identities collide.")
        with PathLease() as pins:
            store.api.require_private(pins.file(path))
        path.unlink()


def _delete_journal(store, name):
    if not _READER.fullmatch(name):
        raise _invalid("Invalid runtime journal cleanup identity.")
    # Remove the lock first: a surviving journal always identifies partial
    # cleanup. Never leave an unowned lock by deleting the journal first.
    for suffix in (".lock", ".json"):
        path = store.root / ".readers" / (name + suffix)
        if os.path.lexists(path):
            with PathLease() as pins:
                store.api.require_private(pins.file(path))
            path.unlink()


class RuntimeReadLease:
    """One SID's read grants and shared content pins until its process is reaped.

    The launcher binds a suspended process before resume. Closing first kills
    and waits for the verified one-process Job member; only then are ACLs and
    file leases released. This class does not resume processes or emit records.
    """

    def __init__(
        self,
        store,
        digest,
        sid,
        *,
        name = None,
    ):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise _invalid("Invalid runtime read generation.")
        self.store, self.digest, self.sid = store, digest, appcontainer_sid(sid)
        self.store_root = store.root
        if name is not None and (type(name) is not str or not _READER.fullmatch(name)):
            raise _invalid("Invalid broker-selected runtime reader name.")
        self.name = name if name is not None else secrets.token_hex(16)
        self.pins = PathLease()
        self.generation = None
        self.process = None
        self.closed = False
        self.journaled = False
        self.cleanup_in_worker = False

    @classmethod
    def _from_handoff(cls, published, name, sid):
        """Parent owner for a validated worker lease; never reopen the store."""
        from .content import ContentGeneration
        from .identity import RuntimeReaderRecipe

        RuntimeReaderRecipe(published.store_root, name, published.content_digest).validate()
        lease = cls.__new__(cls)
        lease.store = None
        lease.store_root = Path(published.store_root)
        lease.digest, lease.name, lease.sid = published.content_digest, name, appcontainer_sid(sid)
        directory = lease.store_root / lease.digest
        lease.generation = ContentGeneration(
            lease.digest,
            directory,
            tuple(directory / "files" / item.relative_path for item in published.spec().files),
            directories = tuple(directory / "files" / name for name in published.spec().directories),
        )
        lease.pins = PathLease()
        lease.process = None
        lease.closed = False
        lease.journaled = lease.cleanup_in_worker = True
        return lease

    def __enter__(self):
        if self.generation is not None or self.closed or self.journaled:
            raise _invalid("Runtime read lease cannot be reused.")
        try:
            with self.store._mutation() as readers:
                journal = self.store.root / ".readers" / self.name
                if self.name in readers or os.path.lexists(journal.with_suffix(".tmp")):
                    raise _invalid("The runtime reader name is already owned.")
                if len(readers) >= MAX_READERS or any(
                    r["sid"] == self.sid for r in readers.values()
                ):
                    raise _invalid("Runtime reader capacity exceeded or invocation SID reused.")
                directory = self.store.root / self.digest
                self.store._lease_marker(self.pins, directory)
                self.generation = self.store._verify(self.digest, self.pins, readers)
                record = {"version": 1, "digest": self.digest, "sid": self.sid}
                # A partial write is an unpublished private temp, never a reader
                # record that permits grants. rename refuses an existing target.
                self.journaled = True
                self.store.api.create(journal.with_suffix(".tmp"), _encode(record))
                os.rename(journal.with_suffix(".tmp"), journal.with_suffix(".json"))
                self.store.api.create(journal.with_suffix(".lock"), b"")
                self.pins.file(journal.with_suffix(".lock"))
                readers[self.name] = record
                _change(self.store, self.generation, readers, self.sid, remove = False)
            return self
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                if self.cleanup_in_worker:
                    # Preserve the exact failed worker/native owners so the
                    # launcher's pending cleanup can reap them before retrying.
                    raise cleanup from original
                self.pins.close()
                raise _invalid(
                    f"Runtime read preparation failed; cleanup also failed: {cleanup}"
                ) from original
            raise

    def bind_process(self, process):
        from .native_compat import WindowsLpacProcess, _api, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION

        if self.closed or self.generation is None or self.process is not None:
            raise _invalid("Runtime read lease has no unbound invocation.")
        if not isinstance(process, WindowsLpacProcess):
            raise _invalid("Runtime lease requires a native Windows process owner.")
        api = _api()
        query = api.kernel32.QueryInformationJobObject
        query.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
        query.restype = W.BOOL
        member = api.kernel32.IsProcessInJob
        member.argtypes, member.restype = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)], W.BOOL
        info, belongs = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(), W.BOOL()
        job = process._unsloth_job._handle
        if (
            not job
            or not process._handle
            or not query(job, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        ):
            raise _invalid("Cannot verify runtime lease Job policy.")
        limits = info.BasicLimitInformation
        if (
            limits.ActiveProcessLimit != 1
            or not limits.LimitFlags & 8
            or not limits.LimitFlags & 0x2000
            or limits.LimitFlags & (0x800 | 0x1000)
            or not member(process._handle, job, ctypes.byref(belongs))
            or not belongs.value
        ):
            raise _invalid(
                "Runtime lease requires its kill-on-close, no-breakaway single-process Job."
            )
        self.process = process

    def close(self):
        if self.closed:
            return
        if self.process is not None:
            # Never release the snapshot on timeout/query/termination failure.
            # The caller retains this owner and reports a cleanup diagnostic.
            self.process.reap(timeout = 5)
            self.process.close()
            self.process = None
        if self.journaled:
            if self.cleanup_in_worker:
                # The exact process is reaped above. Release its liveness locks
                # before the fixed worker attempts exclusive scoped recovery.
                self.pins.close()
                from .preparation import cleanup_runtime_reader

                cleanup_runtime_reader(self.store_root, self.name, self.sid)
                self.closed = True
                return
            with self.store._mutation() as readers:
                record = readers.get(self.name)
                if record is None:
                    if not os.path.lexists(self.store.root / ".readers" / (self.name + ".tmp")):
                        raise _invalid("Runtime read ownership journal changed.")
                    _discard_unpublished(self.store, self.name)
                    self.pins.close()
                    self.closed = True
                    return
                if record != {"version": 1, "digest": self.digest, "sid": self.sid}:
                    raise _invalid("Runtime read ownership journal changed.")
                _change(self.store, self.generation, readers, self.sid, remove = True)
                self.pins.close()
                _delete_journal(self.store, self.name)
        self.pins.close()
        self.closed = True

    def __exit__(self, *_):
        self.close()


def recover_readers(
    store,
    *,
    owner = None,
    expected_digest = None,
):
    """Revoke abandoned read ACEs; live owners retain a non-inherited kernel lock.

    This reconciles storage resources only. The production launcher must pair
    every bound process with kill-on-close and complete parent-death testing.
    """
    if owner is not None:
        if (
            type(owner) is not tuple
            or len(owner) != 2
            or type(owner[0]) is not str
            or not _READER.fullmatch(owner[0])
        ):
            raise _invalid("Invalid scoped reader cleanup identity.")
        appcontainer_sid(owner[1])
    if expected_digest is not None and (
        owner is None
        or type(expected_digest) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
    ):
        raise _invalid("Invalid scoped reader generation.")
    recovered = 0
    with store._mutation() as readers:
        if owner is not None:
            _discard_unpublished(store, owner[0])
        else:
            for path in (store.root / ".readers").iterdir():
                if path.suffix == ".tmp":
                    _discard_unpublished(store, path.stem)
        for name, record in tuple(readers.items()):
            if owner is not None:
                if name != owner[0]:
                    continue
                if record["sid"] != owner[1]:
                    raise _invalid("The reader journal belongs to another invocation SID.")
                if expected_digest is not None and record["digest"] != expected_digest:
                    raise _invalid("The reader journal belongs to another runtime generation.")
            marker = store.root / ".readers" / (name + ".lock")
            with PathLease() as pins:
                try:
                    if os.path.lexists(marker):
                        handle = pins.file(marker, exclusive = True)
                        store.api.require_private(handle)
                        if store.api.info(handle).size:
                            raise _invalid("Unexpected runtime reader lock data.")
                except WindowsRuntimeError as exc:
                    if exc.code == "WINDOWS_SANDBOX_STORE_BUSY":
                        if owner is not None:
                            raise
                        continue
                    raise
                directory = store.root / record["digest"]
                store._lease_marker(pins, directory)
                generation = store._verify(record["digest"], pins, readers)
                _change(store, generation, readers, record["sid"], remove = True)
            _delete_journal(store, name)
            del readers[name]
            recovered += 1
    return recovered
