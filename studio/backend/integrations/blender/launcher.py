# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


"""Launch the cached Blender MCP runtime over stdio without downloading."""

import sys
from pathlib import Path


def main() -> int:
    if sys.version_info < (3, 10):
        print("Blender MCP requires Python 3.10 or newer.", file = sys.stderr)
        return 1
    if len(sys.argv) != 2:
        print("Expected the Blender MCP runtime path.", file = sys.stderr)
        return 2
    runtime = Path(sys.argv[1])
    if not (runtime / ".ready").is_file():
        print("Open Blender MCP in Studio and enable it to download the runtime.", file = sys.stderr)
        return 1
    sys.path.insert(0, str(runtime))
    from blmcp import main as upstream_main

    sys.argv = [str(Path(__file__).resolve()), "--transport", "stdio"]
    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
