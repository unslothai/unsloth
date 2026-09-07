# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed shell-file copying entrypoint; never executes a shell or package."""

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
    from core.inference.windows_sandbox import terminal_content as content
    from core.inference.windows_sandbox import terminal_runtime as runtime

    value = content._json(sys.argv[1])
    if (
        type(value) is not dict
        or set(value) != {"schema", "nonce", "request"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or type(value["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", value["nonce"])
        or type(value["request"]) is not dict
        or set(value["request"]) != {"input", "store_root"}
        or not runtime._local_path(value["request"]["store_root"])
    ):
        return 2
    request = value["request"]
    incoming = runtime._input(request["input"])
    response = {
        "schema": 1,
        "nonce": value["nonce"],
        "request_digest": runtime._digest(request),
        "pid": os.getpid(),
        "snapshot": None,
        "error": None,
    }
    try:
        selected = runtime._inspect(incoming)
        response["snapshot"] = asdict(
            content.publish_terminal_content(selected, request["store_root"])
        )
    except Exception as error:
        response["error"] = str(error)[:4096] or type(error).__name__
    encoded = json.dumps(response, separators = (",", ":")).encode("utf-8")
    if len(encoded) > content.MAX_RESULT:
        return 2
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
