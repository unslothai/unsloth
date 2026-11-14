# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.abc
import importlib.machinery
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


def _negotiating_session(session_cls):
    try:
        from mcp.client._probe import negotiate_auto
    except ImportError:
        return session_cls

    class _NegotiatingClientSession(session_cls):
        # data-designer only calls initialize(), which a modern-only server rejects.
        async def initialize(self):
            if getattr(self, "_negotiating", False):
                return await super().initialize()
            self._negotiating = True
            try:
                await negotiate_auto(self)
            finally:
                self._negotiating = False
            return self.initialize_result

    return _NegotiatingClientSession


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        module.ClientSession = _negotiating_session(module.ClientSession)


class _DataDesignerMcpFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
        if fullname != _DATA_DESIGNER_MCP_IO:
            return None
        import mcp.client.streamable_http as streamable_http

        alias = getattr(streamable_http, "streamablehttp_client", _streamablehttp_client)
        mcp2 = alias is _streamablehttp_client
        streamable_http.streamablehttp_client = alias
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if mcp2 and spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader)
        return spec


# data-designer targets mcp 1; a hook avoids importing mcp at startup.
def install_streamablehttp_alias() -> None:
    if not any(isinstance(f, _DataDesignerMcpFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _DataDesignerMcpFinder())
