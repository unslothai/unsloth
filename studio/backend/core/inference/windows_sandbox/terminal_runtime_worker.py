# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed static Terminal scanner. Never launch the selected shell."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys


def main():
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 2:
        return 2
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from core.inference.windows_sandbox import terminal_runtime as runtime

    request = runtime._json(sys.argv[1])
    if (
        type(request) is not dict
        or set(request) != {"schema", "nonce", "input"}
        or type(request["schema"]) is not int
        or request["schema"] != 1
        or type(request["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", request["nonce"])
    ):
        return 2
    value = runtime._input(request["input"])
    response = dict(
        schema = 1,
        nonce = request["nonce"],
        request_digest = runtime._digest(value),
        pid = os.getpid(),
        result = None,
        error = None,
    )
    try:
        response["result"] = asdict(runtime._inspect(value))
    except (OSError, runtime.WindowsRuntimeError, runtime.lpac.SandboxUnavailableError) as error:
        response["error"] = str(error)[:4096] or type(error).__name__
    encoded = json.dumps(response, separators = (",", ":")).encode("utf-8")
    if len(encoded) > runtime._MAX_RESPONSE:
        return 2
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
