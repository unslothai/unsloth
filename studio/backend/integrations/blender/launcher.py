# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


"""Launch the cached Blender MCP runtime over stdio without downloading."""

import sys
import types
from pathlib import Path


def _alias_fastmcp() -> None:
    # mcp 2 renamed FastMCP to MCPServer; the pinned runtime still imports the v1 module.
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ModuleNotFoundError:
        from mcp.server import mcpserver

        shim = types.ModuleType("mcp.server.fastmcp")
        shim.FastMCP = mcpserver.MCPServer
        shim.__getattr__ = lambda name: getattr(mcpserver, name)
        sys.modules["mcp.server.fastmcp"] = shim


def main() -> int:
    if sys.version_info < (3, 10):
        print("Blender MCP requires Python 3.10 or newer.", file = sys.stderr)
        return 1
    if len(sys.argv) != 2:
        print("Expected the Blender MCP runtime path.", file = sys.stderr)
        return 2
    runtime = Path(sys.argv[1])
    if not (runtime / ".ready").is_file():
        print("Open Blender MCP in Unsloth and enable it to download the runtime.", file = sys.stderr)
        return 1
    sys.path.insert(0, str(runtime))
    _alias_fastmcp()
    from blmcp import main as upstream_main

    sys.argv = [str(Path(__file__).resolve()), "--transport", "stdio"]
    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
