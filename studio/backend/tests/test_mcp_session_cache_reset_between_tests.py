# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A scoped MCP session one test leaves cached does not outlive that test.

``mcp_client._mcp_sessions`` holds a scoped HTTP session until something closes it. The scoped case of
test_tool_call_outlives_sdk_read_timeout left one behind, and test_unscoped_http_stays_one_shot, later on
the same xdist worker, found it in the cache. The two tests below run in order on one worker: the first
leaves a session cached the way a scoped call does, the second fails unless conftest closed it in between.
"""

from __future__ import annotations

import pytest

from core.inference import mcp_client

pytestmark = pytest.mark.xdist_group("mcp_session_cache_reset_between_tests")


class _Session:
    closed = []

    def close(self):
        _Session.closed.append(self)


def test_a_scoped_call_leaves_its_session_cached():
    # Assigned, not monkeypatched: a scoped call stores its session the same way.
    session = _Session()
    mcp_client._mcp_sessions[("http://127.0.0.1:1/mcp", (), "chat-1", "owner")] = session
    _Session.closed.clear()
    assert mcp_client._mcp_sessions


def test_the_next_test_starts_with_an_empty_cache_and_the_session_closed():
    assert mcp_client._mcp_sessions == {}
    assert len(_Session.closed) == 1
