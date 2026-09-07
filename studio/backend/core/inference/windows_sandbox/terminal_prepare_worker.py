# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed Terminal filesystem preparation. Never execute the shell or its code."""

from dataclasses import asdict
import ctypes
import json
import os
from pathlib import Path
import re
import sys


def main():
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 2:
        return 2
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from core.inference.windows_sandbox import terminal_native as lpac
    from core.inference.windows_sandbox import identity, terminal_runtime as runtime
    from core.inference.windows_sandbox.content_files import PathLease
    from core.inference.windows_sandbox.preparation import MAX_RESULT, _json

    request = _json(sys.argv[1])
    if (
        type(request) is not dict
        or set(request)
        != {"schema", "nonce", "input", "recipe"}
        | ({"content"} if request.get("schema") == 2 else set())
        or type(request["schema"]) is not int
        or request["schema"] not in (1, 2)
        or type(request["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", request["nonce"])
        or type(request["recipe"]) is not dict
        or set(request["recipe"]) != {"moniker", "owner_pid", "owner_created"}
    ):
        return 2
    content = request.get("content")
    if request["schema"] == 2 and (
        type(content) is not dict
        or set(content) != {"store_root", "reader"}
        or not runtime._local_path(content["store_root"])
        or type(content["reader"]) is not str
        or not re.fullmatch(r"[0-9a-f]{32}", content["reader"])
    ):
        return 2
    incoming = runtime._input(request["input"])
    recipe = identity.InvocationRecipe(**request["recipe"])
    recipe.filename()
    owner = identity.InvocationReservation(recipe)
    pins, access = PathLease(), None
    response = {
        "schema": request["schema"],
        "nonce": request["nonce"],
        "request_digest": runtime._digest(incoming),
        "pid": os.getpid(),
        "result": None,
        "identity": None,
        "pins": [],
        "collision": None,
        "error": None,
    }
    if content is not None:
        response.update(snapshot = None, reader_pins = [])
    try:
        try:
            identity.recover_identities()
            selected = runtime._inspect(incoming)
            pins.directory(selected.workdir)
            if content is not None:
                from core.inference.windows_sandbox.terminal_content import publish_terminal_content
                from core.inference.windows_sandbox.content import RuntimeContentStore

                snapshot = publish_terminal_content(selected, content["store_root"])
                reader = identity.RuntimeReaderRecipe(
                    content["store_root"], content["reader"], snapshot.digest
                )
                actual = owner.create_terminal_snapshot(selected.workdir, reader = reader)
                store = RuntimeContentStore(content["store_root"])
                access = store.read_access(snapshot.digest, actual.sid_string, name = reader.name)
                access.__enter__()
                selected = snapshot.relocated(access.generation)
                response["snapshot"] = asdict(snapshot)
            # Copied runtime files already have their store lease. Direct OS
            # runtimes need pins before identity preparation. Serviced binaries
            # have WinSxS hardlinks and receive no invocation ACE, so keep their
            # directory pin rather than imposing private-content file rules.
            if any(runtime._within(selected.argv[0], root) for root in selected.acl_roots):
                pins.file(selected.argv[0])
            pins.directory(selected.workdir)
            for root in selected.runtime_roots:
                pins.directory(root)
            if content is None:
                actual = owner.create_terminal(selected)
            pins.directory(actual.private_temp)
            lpac._grant_modify(selected.workdir, actual.sid)
            lpac._grant_modify(actual.private_temp, actual.sid)
            if content is None:
                for root in selected.acl_roots:
                    lpac._grant_read_execute(root, actual.sid)
            for root in actual.traverse_roots:
                lpac._grant_traverse(root, actual.sid)
            response["result"] = asdict(selected)
            response["identity"] = {
                "record": identity._read(owner.path),
                "path": str(owner.path),
                "traverse": list(actual.traverse_roots),
            }
        except Exception as error:
            response["error"] = str(error)[:4096] or type(error).__name__
            response["result"] = response["identity"] = None
            response["collision"] = owner.collision_record
        response["pins"] = [[str(p), h] for p, h in pins.handles.items()]
        if content is not None and access is not None:
            response["reader_pins"] = [[str(p), h] for p, h in access.pins.handles.items()]
        output = json.dumps(response, separators = (",", ":")).encode("utf-8")
        if len(output) > (MAX_RESULT if content is not None else 65536):
            return 2
        sys.stdout.buffer.write(len(output).to_bytes(4, "little") + output)
        sys.stdout.buffer.flush()
        return 0 if sys.stdin.buffer.read(33) == bytes.fromhex(request["nonce"]) else 2
    finally:
        # Broker owns durable recovery even when stdout/ACK/close fails. Release
        # only local pins and SID allocations here, never handed-off grants.
        try:
            pins.close()
        finally:
            if access is not None:
                access.pins.close()
        for sid in (owner.created_sid, owner.identity.sid if owner.identity is not None else None):
            if sid and lpac._api().advapi32.FreeSid(sid):
                raise OSError("Terminal worker could not release its local SID allocation")


if __name__ == "__main__":
    raise SystemExit(main())
