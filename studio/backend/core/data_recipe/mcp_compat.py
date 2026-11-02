# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.abc
import sys
from contextlib import asynccontextmanager

_DATA_DESIGNER_MCP_IO = "data_designer.engine.mcp.io"


@asynccontextmanager
async def _streamablehttp_client(url, headers = None):
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    async with create_mcp_http_client(headers = headers) as client:
        async with streamable_http_client(url, http_client = client) as streams:
            yield streams


class _StreamableHttpAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
        if fullname == _DATA_DESIGNER_MCP_IO:
            import mcp.client.streamable_http as streamable_http
            if not hasattr(streamable_http, "streamablehttp_client"):
                streamable_http.streamablehttp_client = _streamablehttp_client
        return None


# data-designer imports a name mcp 2 removed; a hook avoids importing mcp at startup.
def install_streamablehttp_alias() -> None:
    sys.meta_path.insert(0, _StreamableHttpAliasFinder())
