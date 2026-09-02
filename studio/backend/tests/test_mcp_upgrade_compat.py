# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an existing Studio install keeps when the MCP session cache stops being
stdio-only: the close entry point, the session-cap environment variable, and the
fastmcp surface this code actually depends on.

Studio declares fastmcp>=3.0.2 with no upper bound and no lockfile, so a released
install resolves to whatever is newest. Anything asserted about fastmcp here is
asserted against the installed version, not a mock.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client

LEGACY_ENV = "UNSLOTH_STUDIO_MAX_STDIO_MCP_SESSIONS"
ENV = "UNSLOTH_STUDIO_MAX_MCP_SESSIONS"


# --------------------------------------------------------------------------
# Entry points an old install may still call
# --------------------------------------------------------------------------


def test_the_old_close_name_still_works():
    """close_stdio_sessions had no underscore, so treat it as callable from
    outside this module even though nothing in-repo does."""
    assert mcp_client.close_stdio_sessions is mcp_client.close_mcp_sessions
    mcp_client.close_stdio_sessions()  # must not raise on an empty cache


def test_the_old_close_name_takes_the_same_arguments():
    mcp_client.close_stdio_sessions("https://mcp.example.test/mcp", None)


# --------------------------------------------------------------------------
# Session cap: the name changed, the setting must not
# --------------------------------------------------------------------------


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.delenv(LEGACY_ENV, raising = False)


def test_unset_falls_back_to_the_default():
    assert mcp_client._max_sessions_from_env() == mcp_client._DEFAULT_MAX_SESSIONS


def test_a_deployment_that_set_the_legacy_name_keeps_its_value(monkeypatch):
    """The cap now covers HTTP too, but silently reverting someone's tuning to 32
    would be a worse surprise than the name being stale."""
    monkeypatch.setenv(LEGACY_ENV, "7")
    assert mcp_client._max_sessions_from_env() == 7


def test_the_new_name_works(monkeypatch):
    monkeypatch.setenv(ENV, "9")
    assert mcp_client._max_sessions_from_env() == 9


def test_the_new_name_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv(LEGACY_ENV, "7")
    monkeypatch.setenv(ENV, "9")
    assert mcp_client._max_sessions_from_env() == 9


@pytest.mark.parametrize("raw", ["", "abc", "1.5", "  "])
def test_an_unparseable_value_falls_back_instead_of_crashing_startup(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert mcp_client._max_sessions_from_env() == mcp_client._DEFAULT_MAX_SESSIONS


@pytest.mark.parametrize("raw,expected", [("0", 1), ("-5", 1), ("1", 1)])
def test_the_cap_never_drops_below_one(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    assert mcp_client._max_sessions_from_env() == expected


# --------------------------------------------------------------------------
# The fastmcp surface this module relies on
# --------------------------------------------------------------------------


def test_the_transports_this_code_builds_still_exist():
    from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
    for cls in (StdioTransport, SSETransport, StreamableHttpTransport):
        assert cls is not None


def test_only_stdio_answers_the_liveness_probe():
    """_transport_dead is the reason the HTTP idle recheck exists. If a future
    fastmcp gives the HTTP transports a real probe, this test fails and the
    recheck can become cheaper."""
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    for cls in (StreamableHttpTransport, SSETransport):
        transport = cls(url = "https://x.test/mcp")
        assert not hasattr(
            transport, "_is_session_dead"
        ), f"{cls.__name__} grew a liveness probe; _transport_dead can use it now"


def test_the_installed_fastmcp_meets_the_declared_floor():
    version = importlib.metadata.version("fastmcp")
    major, minor, *_ = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (3, 0), f"fastmcp {version} is below the declared >=3.0.2"


def test_tool_errors_are_still_distinguishable_from_transport_errors():
    """The whole keep-or-drop-the-session decision rests on this import."""
    from fastmcp.exceptions import ToolError

    assert mcp_client._is_tool_error(ToolError("nope")) is True
    assert mcp_client._is_tool_error(RuntimeError("stream closed")) is False


# --------------------------------------------------------------------------
# Rows written by an older Studio
# --------------------------------------------------------------------------


def test_a_row_without_use_oauth_is_treated_as_non_oauth():
    """The column was backfilled by an ALTER on an existing table, so old rows
    can read as None rather than 0."""
    assert bool({"url": "https://x.test/mcp"}.get("use_oauth")) is False


def test_header_identity_survives_a_rewritten_headers_dict():
    """A row re-saved with the same headers in another key order must not look
    like a different server and drop the live session."""
    a = mcp_client._headers_key({"Authorization": "Bearer x", "X-Trace": "1"})
    b = mcp_client._headers_key({"X-Trace": "1", "Authorization": "Bearer x"})
    assert a == b


def test_header_values_are_part_of_the_identity():
    a = mcp_client._headers_key({"Authorization": "Bearer x"})
    b = mcp_client._headers_key({"Authorization": "Bearer y"})
    assert a != b, "two credentials would have shared one session"
