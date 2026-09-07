# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Validate copied Terminal content and adopt only the fixed worker's pinned lease."""

import json
from pathlib import Path
from types import SimpleNamespace

from .content import ContentGeneration
from .content_access import RuntimeReadLease
from .identity import RuntimeReaderRecipe
from .launch_transfer import _with_parents
from .preparation import _check_deadline, _decode
from .terminal_content import _snapshot_response
from . import terminal_runtime as runtime


def decode_snapshot(owner, value, process):
    request = {"input": owner.request, "store_root": owner.snapshot_store}
    snapshot = _snapshot_response(
        json.dumps(
            {
                "schema": 1,
                "nonce": owner.nonce,
                "request_digest": runtime._digest(request),
                "pid": process.pid,
                "snapshot": value["snapshot"],
                "error": None,
            }
        ).encode(),
        process.pid,
        owner.nonce,
        request,
    )
    directory = Path(owner.snapshot_store) / snapshot.digest
    generation = ContentGeneration(
        snapshot.digest,
        directory,
        tuple(directory / "files" / item.relative_path for item in snapshot.spec.files),
        directories = tuple(directory / "files" / name for name in snapshot.spec.directories),
    )
    # This computes the expected paths; it does not yet authorize use of them.
    # adopt_reader checks their actual duplicated native handles before ACK.
    selected = snapshot.relocated(generation)
    request = {
        "argv": list(selected.argv),
        "workdir": selected.workdir,
        "env": owner.request["env"],
    }
    actual = runtime._response(
        json.dumps(
            {
                "schema": 1,
                "nonce": owner.nonce,
                "request_digest": runtime._digest(request),
                "pid": process.pid,
                "result": value["result"],
                "error": None,
            }
        ).encode(),
        process.pid,
        owner.nonce,
        request,
    )
    if actual != selected:
        raise runtime._invalid("Copied Terminal handoff changed its expected runtime roots.")
    return snapshot, selected


def adopt_reader(owner, value, process):
    snapshot = owner.snapshot
    reader = RuntimeReaderRecipe(owner.snapshot_store, owner.reader_name, snapshot.digest)
    reader.validate()
    rows = _decode(tuple[tuple[str, int], ...], value["reader_pins"])
    directory = Path(reader.store_root) / reader.digest
    files = {
        directory / ".lease",
        directory / "manifest.json",
        Path(reader.store_root) / ".readers" / (reader.name + ".lock"),
        *(directory / "files" / item.relative_path for item in snapshot.spec.files),
    }
    expected = _with_parents(
        files | {directory / "files" / name for name in snapshot.spec.directories}
    )
    if len(rows) != len(expected) or {Path(path) for path, _ in rows} != expected:
        raise runtime._invalid("Terminal content pins differ from its snapshot inventory.")
    base_rows = _decode(tuple[tuple[str, int], ...], value["pins"])
    handles = [handle for _, handle in (*base_rows, *rows)]
    if len(set(handles)) != len(handles):
        raise runtime._invalid("Terminal content handoff aliases distinct native handles.")
    # Reuse only storage ACL/journal ownership. Do not call the Python reader's
    # single-process bind method: TerminalJobOwner owns the entire payload Job.
    published = SimpleNamespace(
        store_root = reader.store_root,
        content_digest = reader.digest,
        spec = lambda: snapshot.spec,
    )
    owner.access = RuntimeReadLease._from_handoff(published, reader.name, owner.identity.sid_string)
    owner.access.pins.duplicate_from(
        process._handle,
        rows,
        lambda: _check_deadline(owner.deadline, owner.cancel),
    )
    for path, handle in owner.access.pins.handles.items():
        _check_deadline(owner.deadline, owner.cancel)
        owner.access.pins.api.require_path(handle, path, directory = path not in files)
    owner.reservation.reader = reader
