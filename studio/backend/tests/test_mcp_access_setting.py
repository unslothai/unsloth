# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

import storage.studio_db as studio_db
from utils import mcp_access
from utils.account_context import OWNER, AccountContext, run_as
from utils.mcp_access import (
    ENV_FORCE,
    MCP_DISABLED_TOOLS_SETTING_KEY,
    MCP_ENABLED_SETTING_KEY,
    UnknownMcpToolError,
    disabled_tools,
    forced_by_env,
    get_mcp_enabled,
    is_mcp_enabled,
    set_mcp_enabled,
    set_mcp_settings,
    tool_enabled,
)

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv(ENV_FORCE, raising = False)
    mcp_access._reset_cache()
    yield
    mcp_access._reset_cache()


@pytest.mark.parametrize("value", [None, "1", "0", "true", "yes", "", " 1", "2"])
def test_only_env_one_forces_it_on_and_never_writes_the_setting(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV_FORCE, value)
    assert forced_by_env() is (value == "1")
    assert is_mcp_enabled() is (value == "1")
    assert get_mcp_enabled() is False


@pytest.mark.parametrize(
    "call",
    [
        lambda: run_as(ALICE, set_mcp_enabled, True),
        *(lambda v = v: set_mcp_enabled(v) for v in ["true", 1, None]),
    ],
)
def test_setter_refuses_a_managed_account_or_a_non_bool(call):
    with pytest.raises(ValueError):
        call()
    assert get_mcp_enabled() is False


def test_set_invalidates_the_cache():
    assert get_mcp_enabled() is False
    assert set_mcp_enabled(True) is True
    assert get_mcp_enabled() is True
    assert is_mcp_enabled() is True
    set_mcp_enabled(False)
    assert get_mcp_enabled() is False


def test_a_cached_value_is_held_for_the_ttl():
    assert get_mcp_enabled() is False
    studio_db.upsert_app_settings({MCP_ENABLED_SETTING_KEY: True})
    assert get_mcp_enabled() is False
    mcp_access._reset_cache()
    assert get_mcp_enabled() is True


def test_a_db_error_reads_false(monkeypatch):
    set_mcp_enabled(True)
    mcp_access._reset_cache()

    def broken(*args, **kwargs):
        raise RuntimeError("database is locked")

    real = studio_db.get_app_settings
    monkeypatch.setattr(studio_db, "get_app_settings", broken)
    assert get_mcp_enabled() is False
    assert is_mcp_enabled() is False
    assert disabled_tools() is None
    assert tool_enabled("studio_status") is False
    # Nothing was cached, so the next good read sees the real value.
    monkeypatch.setattr(studio_db, "get_app_settings", real)
    assert get_mcp_enabled() is True
    assert disabled_tools() == frozenset()


def test_a_non_bool_stored_value_reads_false():
    studio_db.upsert_app_settings({MCP_ENABLED_SETTING_KEY: "true"})
    assert get_mcp_enabled() is False


def test_a_write_during_a_read_is_not_published(monkeypatch):
    set_mcp_enabled(True)
    real = mcp_access._read_from_db

    def read_then_disable():
        value = real()
        set_mcp_enabled(False)
        return value

    monkeypatch.setattr(mcp_access, "_read_from_db", read_then_disable)
    assert get_mcp_enabled() is False
    monkeypatch.setattr(mcp_access, "_read_from_db", real)
    assert get_mcp_enabled() is False


def test_a_managed_account_reads_the_owners_value():
    run_as(OWNER, set_mcp_enabled, True)
    mcp_access._reset_cache()
    assert run_as(ALICE, studio_db.get_app_setting, MCP_ENABLED_SETTING_KEY, None) is None
    assert run_as(ALICE, get_mcp_enabled) is True


def _stored():
    return studio_db.get_app_settings([MCP_ENABLED_SETTING_KEY, MCP_DISABLED_TOOLS_SETTING_KEY])


def test_every_tool_is_on_by_default():
    assert disabled_tools() == frozenset()
    assert tool_enabled("cancel") is True


def test_the_tool_list_is_stored_sorted_and_unique_and_read_live():
    set_mcp_settings(disabled_tools = ["voices", "cancel", "cancel"])
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel", "voices"]}
    assert disabled_tools() == {"cancel", "voices"}
    assert tool_enabled("cancel") is False and tool_enabled("chat") is True
    set_mcp_settings(disabled_tools = [])
    assert tool_enabled("cancel") is True


def test_both_values_go_in_one_upsert(monkeypatch):
    calls = []
    real = studio_db.upsert_app_settings

    def spy(settings, **kwargs):
        calls.append(dict(settings))
        return real(settings, **kwargs)

    monkeypatch.setattr(studio_db, "upsert_app_settings", spy)
    set_mcp_settings(enabled = True, disabled_tools = ["chat"])
    assert calls == [{MCP_ENABLED_SETTING_KEY: True, MCP_DISABLED_TOOLS_SETTING_KEY: ["chat"]}]
    assert get_mcp_enabled() is True and disabled_tools() == {"chat"}


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda: set_mcp_settings(disabled_tools = ["cancel", "nope"]), UnknownMcpToolError),
        (lambda: set_mcp_settings(disabled_tools = "cancel"), ValueError),
        (lambda: set_mcp_settings(disabled_tools = ["cancel", 1]), ValueError),
        (lambda: set_mcp_settings(), ValueError),
        (lambda: set_mcp_settings(enabled = "true", disabled_tools = ["cancel"]), ValueError),
        (lambda: run_as(ALICE, set_mcp_settings, disabled_tools = ["cancel"]), ValueError),
    ],
    ids = ["unknown", "string", "non-string", "nothing", "bad-enabled", "managed-account"],
)
def test_a_bad_tool_write_stores_nothing(call, error):
    with pytest.raises(error):
        call()
    assert _stored() == {}
    assert disabled_tools() == frozenset()


def test_an_unknown_tool_names_itself():
    with pytest.raises(UnknownMcpToolError, match = "^Unknown Unsloth Studio MCP tool: nope$"):
        set_mcp_settings(disabled_tools = ["nope"])


@pytest.mark.parametrize("stored", ["x", {}, None, 3, {"cancel": True}])
def test_an_unreadable_tool_list_turns_every_tool_off(stored):
    studio_db.upsert_app_settings(
        {MCP_ENABLED_SETTING_KEY: True, MCP_DISABLED_TOOLS_SETTING_KEY: stored}
    )
    assert disabled_tools() is None
    assert tool_enabled("studio_status") is False
    # The switch is a separate value and still reads.
    assert get_mcp_enabled() is True


def test_non_string_and_stale_names_are_ignored_on_read():
    studio_db.upsert_app_settings(
        {MCP_DISABLED_TOOLS_SETTING_KEY: [1, None, "gone_tool", "cancel"]}
    )
    assert disabled_tools() == {"gone_tool", "cancel"}
    assert tool_enabled("cancel") is False and tool_enabled("chat") is True


def test_a_managed_account_reads_the_owners_tool_list():
    run_as(OWNER, set_mcp_settings, disabled_tools = ["cancel"])
    mcp_access._reset_cache()
    assert run_as(ALICE, studio_db.get_app_settings, [MCP_DISABLED_TOOLS_SETTING_KEY]) == {}
    assert run_as(ALICE, disabled_tools) == {"cancel"}
    assert run_as(ALICE, tool_enabled, "cancel") is False


def test_a_read_that_overlaps_a_write_is_retried_not_reported_off(monkeypatch):
    set_mcp_settings(enabled = True, disabled_tools = ["cancel"])
    mcp_access._reset_cache()
    real = mcp_access._read_from_db
    calls = []

    def racing_read():
        calls.append(1)
        if len(calls) == 1:
            # Another thread's write lands while this read is in flight.
            with mcp_access._cache_lock:
                mcp_access._generation += 1
        return real()

    monkeypatch.setattr(mcp_access, "_read_from_db", racing_read)
    assert disabled_tools() == frozenset({"cancel"})
    assert get_mcp_enabled() is True
    assert len(calls) == 2


def test_a_read_that_keeps_racing_writes_still_fails_closed(monkeypatch):
    set_mcp_settings(enabled = True)
    mcp_access._reset_cache()
    real = mcp_access._read_from_db

    def always_racing():
        with mcp_access._cache_lock:
            mcp_access._generation += 1
        return real()

    monkeypatch.setattr(mcp_access, "_read_from_db", always_racing)
    assert mcp_access.owner_record() is None
    assert disabled_tools() is None
    assert get_mcp_enabled() is False
