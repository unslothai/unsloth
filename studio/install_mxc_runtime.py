#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit lifecycle commands for Studio's packaged Windows MXC runtime.

The artifact source and installation destination are application-owned constants.
This interface intentionally accepts neither paths nor manifest contents.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from backend.core.inference import mxc_runtime


def _status_payload(status: mxc_runtime.RuntimeStatus) -> dict:
    payload = asdict(status)
    payload["state"] = status.state.value
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("install", "verify", "repair", "update", "rollback", "gc", "uninstall"),
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        result: dict
        if args.command == "verify":
            status = mxc_runtime.runtime_status()
            result = _status_payload(status)
            exit_code = 0 if status.state is mxc_runtime.RuntimeState.READY else 2
        elif args.command == "install":
            info = mxc_runtime.install_approved_runtime(operation="install")
            result = {"state": "ready", "generation": info.generation, "sha256": info.runner_sha256}
            exit_code = 0
        elif args.command == "repair":
            info = mxc_runtime.repair_runtime()
            result = {"state": "ready", "generation": info.generation, "sha256": info.runner_sha256}
            exit_code = 0
        elif args.command == "update":
            info = mxc_runtime.update_runtime()
            result = {"state": "ready", "generation": info.generation, "sha256": info.runner_sha256}
            exit_code = 0
        elif args.command == "rollback":
            info = mxc_runtime.rollback_runtime()
            result = {"state": "ready", "generation": info.generation, "sha256": info.runner_sha256}
            exit_code = 0
        elif args.command == "gc":
            result = {"state": "ready", "retired": mxc_runtime.garbage_collect()}
            exit_code = 0
        else:
            complete = mxc_runtime.uninstall_runtime()
            result = {"state": "not_installed" if complete else "retirement_pending"}
            exit_code = 0 if complete else 3
    except mxc_runtime.MxcRuntimeUnavailable as exc:
        result = {"state": exc.code, "reason": str(exc)}
        exit_code = 2

    if args.json_output:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        if result.get("state") == "ready":
            generation = result.get("generation")
            print(f"MXC runtime ready{f': {generation}' if generation else ''}")
        else:
            print(f"MXC runtime {result.get('state')}: {result.get('reason', '')}".rstrip())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
