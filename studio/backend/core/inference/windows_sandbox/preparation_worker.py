# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed static worker entrypoint. Never import a candidate native library."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


def main():
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 3:
        return 2
    # These paths/request come only from the parent-owned launch, not tool data.
    sys.path.insert(0, str(Path(__file__).parents[3]))
    sys.path.append(sys.argv[1])
    from core.inference.windows_sandbox.admission import _BrokerRuntime, _admit_broker_runtime
    from core.inference.windows_sandbox.native_plan import ScanBounds
    from core.inference.windows_sandbox.preparation import (
        _decode,
        _json,
        MAX_RESULT,
        PublishedRuntime,
    )
    from core.inference.windows_sandbox.profiles import (
        PYTHON_PROFILE,
        WindowsRuntimeError,
        select_abi_adapter,
    )

    request = _json(sys.argv[2])
    if (
        type(request) is not dict
        or set(request)
        not in (
            {"broker", "bounds", "nonce", "profile", "store_root"},
            {"broker", "bounds", "nonce", "profile", "store_root", "pins"},
            {"broker", "bounds", "nonce", "profile", "store_root", "pins", "launch"},
        )
        or request["profile"] != PYTHON_PROFILE.digest
        or type(request.get("pins", False)) is not bool
        or (request.get("pins", False) and request["store_root"] is None)
        or ("launch" in request and not request.get("pins", False))
    ):
        return 2
    nonce = request["nonce"]
    if (
        type(nonce) is not str
        or len(nonce) != 64
        or any(c not in "0123456789abcdef" for c in nonce)
    ):
        return 2
    broker = _decode(_BrokerRuntime, request["broker"])
    bounds = _decode(ScanBounds, request["bounds"])
    store_root = request["store_root"]
    if store_root is not None and (
        type(store_root) is not str
        or not Path(store_root).is_absolute()
        or len(store_root) > 32768
        or "\0" in store_root
    ):
        return 2
    response = {
        "nonce": nonce,
        "profile": PYTHON_PROFILE.digest,
        "pid": os.getpid(),
        "core": None,
        "error": None,
        "publication": None,
    }
    from core.inference.windows_sandbox.content_files import PathLease

    pins = PathLease() if request.get("pins", False) else None
    owner = None
    from core.inference.windows_sandbox import launch_transfer

    try:
        if "launch" in request:
            owner = launch_transfer.worker_owner(request["launch"], broker, store_root)
        core = _admit_broker_runtime(broker.executable, broker, bounds)
        if store_root is not None:
            from core.inference.windows_sandbox.artifacts import admit_installed_artifacts
            from core.inference.windows_sandbox.content import RuntimeContentStore

            adapter = select_abi_adapter(
                implementation = core.runtime.implementation,
                version = core.runtime.version,
                architecture = core.runtime.architecture,
            )
            artifacts = admit_installed_artifacts(broker.prefix, adapter)
            store = RuntimeContentStore(store_root)
            candidate = PublishedRuntime(core, artifacts, str(store.root), "")
            if owner is not None and owner.probe_executable is None:
                from core.inference.windows_sandbox.launch import _overlap, _invalid
                if any(
                    _overlap(owner.workdir, item.source.path) for item in candidate.spec().files
                ):
                    raise _invalid("A contributor-writable path supplied a startup runtime file.")
            digest = store.publish(candidate.spec())
            published = PublishedRuntime(core, artifacts, str(store.root), digest)
            response["publication"] = asdict(published)
            if pins is not None:
                with store._mutation() as readers:
                    store._lease_marker(pins, store.root / digest)
                    store._verify(digest, pins, readers)
            if owner is not None:
                owner.published = published
                owner.prepare_files()
        response["core"] = asdict(core)
    except WindowsRuntimeError as error:
        response["core"] = response["publication"] = None
        response["error"] = {"code": error.code, "message": str(error)[:4096]}
    except OSError as error:
        # Filesystem recovery can fail outside the Win32 wrapper. Preserve the
        # bounded diagnostic in the data channel, never retry or report success.
        response["core"] = response["publication"] = None
        response["error"] = {
            "code": "WINDOWS_SANDBOX_PREPARATION_IO_FAILED",
            "message": str(error)[:4096],
        }
    try:
        if "launch" in request:
            response["launch"] = launch_transfer.worker_response(
                owner,
                failed = response["error"] is not None,
            )
        if pins is not None:
            response["pins"] = [[str(path), handle] for path, handle in pins.handles.items()]
        output = json.dumps(response, separators = (",", ":")).encode("utf-8")
        if len(output) > MAX_RESULT:
            return 2
        if pins is not None:
            sys.stdout.buffer.write(len(output).to_bytes(4, "little"))
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        if pins is not None and sys.stdin.buffer.read(33) != bytes.fromhex(nonce):
            return 2
        return 0
    finally:
        launch_transfer.release_worker_pins(owner)
        if pins is not None:
            pins.close()


if __name__ == "__main__":
    raise SystemExit(main())
