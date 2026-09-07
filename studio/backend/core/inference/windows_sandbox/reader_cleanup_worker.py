# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One fixed scoped reader cleanup, never a payload or operation dispatcher."""

import json
import os
from pathlib import Path
import re
import sys


def main():
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 2:
        return 2
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from core.inference.windows_sandbox.preparation import _json
    from core.inference.windows_sandbox.content import RuntimeContentStore
    from core.inference.windows_sandbox.content_access import appcontainer_sid, recover_readers

    request = _json(sys.argv[1])
    if (
        type(request) is not dict
        or set(request) != {"root", "name", "sid", "nonce"}
        or type(request["root"]) is not str
        or len(request["root"]) > 32768
        or "\0" in request["root"]
        or not Path(request["root"]).is_absolute()
        or Path(request["root"]).parent == Path(request["root"])
        or type(request["name"]) is not str
        or not re.fullmatch(r"[0-9a-f]{32}", request["name"])
        or type(request["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", request["nonce"])
    ):
        return 2
    appcontainer_sid(request["sid"])
    response = {"nonce": request["nonce"], "pid": os.getpid(), "error": None}
    try:
        store = RuntimeContentStore(request["root"], _existing_only = True)
        recover_readers(store, owner = (request["name"], request["sid"]))
    except Exception as error:
        response["error"] = str(error)[:4096]
    sys.stdout.buffer.write(json.dumps(response, separators = (",", ":")).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
