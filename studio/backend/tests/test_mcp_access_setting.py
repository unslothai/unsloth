# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

import storage.studio_db as studio_db
from utils import mcp_access
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as
from utils.mcp_access import (
    ENV_FORCE,
    MCP_ENABLED_SETTING_KEY,
    forced_by_env,
    get_mcp_enabled,
    is_mcp_enabled,
    set_mcp_enabled,
)

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv(ENV_FORCE, raising = False)
    mcp_access._reset_cache()
    yield
    mcp_access._reset_cache()


def test_off_by_default():
    assert get_mcp_enabled() is False
    assert forced_by_env() is False
    assert is_mcp_enabled() is False


def test_env_one_forces_it_on(monkeypatch):
    monkeypatch.setenv(ENV_FORCE, "1")
    assert forced_by_env() is True
    assert is_mcp_enabled() is True
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("value", ["0", "true", "yes", "", " 1", "2"])
def test_other_env_values_do_not_force_it(monkeypatch, value):
    monkeypatch.setenv(ENV_FORCE, value)
    assert forced_by_env() is False
    assert is_mcp_enabled() is False


def test_setter_refuses_a_managed_account():
    token = bind_account(ALICE)
    try:
        with pytest.raises(ValueError):
            set_mcp_enabled(True)
    finally:
        reset_account(token)
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_setter_needs_a_bool(value):
    with pytest.raises(ValueError):
        set_mcp_enabled(value)


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

    monkeypatch.setattr(studio_db, "get_app_setting", broken)
    assert get_mcp_enabled() is False
    assert is_mcp_enabled() is False


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
    token = bind_account(ALICE)
    try:
        assert studio_db.get_app_setting(MCP_ENABLED_SETTING_KEY, None) is None
        assert get_mcp_enabled() is True
    finally:
        reset_account(token)
