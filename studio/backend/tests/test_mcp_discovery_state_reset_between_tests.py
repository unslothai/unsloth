# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""MCP discovery state one test leaves behind does not decide the next test's discovery.

``mcp_client._tool_cache`` and ``_probe_cooloff_until`` are module state keyed by server id. A failed
probe in test_mcp_apps.py left "s1" cooling off, and test_get_enabled_mcp_tools_caches_discovery, later on
the same xdist worker, then skipped discovery for its own "s1" (``assert 0 == 1``). The two tests below
run in order on one worker: the first leaves both behind the way a real probe does, the second fails
unless conftest forgot them in between.
"""

from __future__ import annotations

import pytest

from core.inference import mcp_client

pytestmark = pytest.mark.xdist_group("mcp_discovery_state_reset_between_tests")


def test_a_probe_leaves_a_cooloff_and_a_cache_entry_behind():
    # Recorded the way the client does it, not monkeypatched.
    mcp_client.record_probe_failure("s1")
    mcp_client._tool_cache[mcp_client._account_key("s2")] = [{"name": "echo"}]
    assert mcp_client.in_failure_cooloff("s1")
    assert mcp_client.get_cached_tools("s2") is not None


def test_the_next_test_starts_with_neither():
    assert not mcp_client.in_failure_cooloff("s1")
    assert mcp_client.get_cached_tools("s2") is None
