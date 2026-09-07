# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One fixed profile cleanup, not a generic filesystem-operation worker."""

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
    from core.inference.windows_sandbox.identity import (
        InvocationRecipe,
        _local_path,
        cleanup_recipe,
    )

    request = _json(sys.argv[1])
    if (
        type(request) is not dict
        or set(request) != {"recipe", "path", "collision", "nonce"}
        or type(request["recipe"]) is not dict
        or set(request["recipe"]) != {"moniker", "owner_pid", "owner_created"}
        or (request["path"] is not None and not _local_path(request["path"]))
        or type(request["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", request["nonce"])
        or (request["collision"] is not None and type(request["collision"]) is not dict)
    ):
        return 2
    recipe = InvocationRecipe(**request["recipe"])
    if request["path"] is not None and Path(request["path"]).name != recipe.filename():
        return 2
    response = {"nonce": request["nonce"], "pid": os.getpid(), "error": None}
    try:
        cleanup_recipe(recipe, request["path"], request["collision"])
    except Exception as error:
        response["error"] = str(error)[:4096]
    sys.stdout.buffer.write(json.dumps(response, separators = (",", ":")).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
