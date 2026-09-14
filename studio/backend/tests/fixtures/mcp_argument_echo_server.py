# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import sys

from fastmcp import FastMCP


launch_log = os.environ.get("UNSLOTH_MCP_ARGUMENT_LOG")
if launch_log:
    with open(launch_log, "a", encoding = "utf-8") as log:
        log.write(
            json.dumps(
                {
                    "arguments": sys.argv[1:],
                    "marker": os.environ.get("UNSLOTH_MCP_ARGUMENT_MARKER"),
                }
            )
            + "\n"
        )


server = FastMCP("argument echo")


@server.tool
def launch_state() -> str:
    return json.dumps(
        {
            "arguments": sys.argv[1:],
            "marker": os.environ.get("UNSLOTH_MCP_ARGUMENT_MARKER"),
        }
    )


if __name__ == "__main__":
    server.run(transport = "stdio", show_banner = False)
