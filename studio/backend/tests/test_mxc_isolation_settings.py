# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox switches for Windows MXC: env first, then the owner's saved choice."""

from pathlib import Path
import subprocess
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mxc_policy, mxc_probe, mxc_read_grants, tools
from storage import studio_db
from utils import mxc_isolation_settings as settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(mxc_policy.DACL_FALLBACK_ENV, raising = False)
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    settings.forget_cached_setting()
    yield
    settings.forget_cached_setting()


def test_defaults_are_off_for_the_opt_in_and_on_for_grants():
    assert mxc_policy.dacl_fallback_enabled() is False
    assert mxc_read_grants.enabled() is True


def test_saved_choices_apply_when_the_environment_is_silent():
    settings.set_dacl_fallback_setting(True)
    settings.set_persistent_grants_setting(False)
    assert mxc_policy.dacl_fallback_enabled() is True
    assert mxc_read_grants.enabled() is False


@pytest.mark.parametrize(
    "value, expected", [("1", True), (" 1 ", True), ("0", False), ("", False), ("yes", False)]
)
def test_a_present_environment_variable_decides_as_before(monkeypatch, value, expected):
    settings.set_dacl_fallback_setting(not expected)
    monkeypatch.setenv(mxc_policy.DACL_FALLBACK_ENV, value)
    assert mxc_policy.dacl_fallback_enabled() is expected
    assert settings.locked_by_environment(mxc_policy.DACL_FALLBACK_ENV) is True


@pytest.mark.parametrize("value, expected", [("0", False), ("1", True), ("", True)])
def test_the_grants_environment_variable_decides_as_before(monkeypatch, value, expected):
    settings.set_persistent_grants_setting(not expected)
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, value)
    assert mxc_read_grants.enabled() is expected


def test_an_unreadable_store_keeps_the_shipped_defaults(monkeypatch):
    settings.set_dacl_fallback_setting(True)
    settings.set_persistent_grants_setting(False)
    settings.forget_cached_setting()

    def boom(*_args, **_kwargs):
        raise OSError("unable to open database file")

    monkeypatch.setattr(studio_db, "get_app_settings", boom)
    assert mxc_policy.dacl_fallback_enabled() is False
    assert mxc_read_grants.enabled() is True
    assert settings._cached is None  # a failure is never held


def test_a_missing_settings_module_means_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.mxc_isolation_settings", None)
    assert mxc_policy.dacl_fallback_enabled() is False
    assert mxc_read_grants.enabled() is True


@pytest.mark.parametrize(
    "stored, expected", [(True, True), ("true", True), ("off", False), (7, False), (None, False)]
)
def test_stored_value_coercion(monkeypatch, stored, expected):
    monkeypatch.setattr(
        studio_db, "get_app_settings", lambda _keys: {settings.DACL_SETTING_KEY: stored}
    )
    assert settings.dacl_fallback_setting() is expected


def test_setters_reject_non_booleans():
    with pytest.raises(ValueError):
        settings.set_dacl_fallback_setting("yes")


def test_the_owner_choice_is_what_a_managed_account_reads():
    settings.set_dacl_fallback_setting(True)
    settings.forget_cached_setting()
    token = bind_account(ALICE)
    try:
        assert mxc_policy.dacl_fallback_enabled() is True
        settings.set_dacl_fallback_setting(False)  # still lands in the owner's store
    finally:
        reset_account(token)
    token = bind_account(OWNER)
    try:
        assert settings.dacl_fallback_setting() is False
    finally:
        reset_account(token)


def test_a_read_in_flight_cannot_republish_what_a_write_replaced(monkeypatch):
    settings.set_dacl_fallback_setting(True)
    settings.forget_cached_setting()
    real = studio_db.get_app_settings

    def write_lands_mid_read(keys):
        stored = real(keys)
        monkeypatch.setattr(studio_db, "get_app_settings", real)
        settings.set_dacl_fallback_setting(False)
        return stored

    monkeypatch.setattr(studio_db, "get_app_settings", write_lands_mid_read)
    assert settings.dacl_fallback_setting() is True  # the answer this call was committed to
    assert settings._cached is None
    assert settings.dacl_fallback_setting() is False


def test_repeated_reads_hit_the_store_once(monkeypatch):
    calls = []
    real = studio_db.get_app_settings
    monkeypatch.setattr(
        studio_db, "get_app_settings", lambda keys: calls.append(keys) or real(keys)
    )
    for _ in range(5):
        mxc_policy.dacl_fallback_enabled()
        mxc_read_grants.enabled()
    assert len(calls) == 1


def test_reset_terminal_profile_cache_forgets_the_advertised_profile():
    tools._request_profile[:] = ["cmd_isolated", 123.0]
    tools.reset_terminal_profile_cache()
    assert tools._request_profile == [None, 0.0]


def test_the_remediation_names_the_same_host_prep_command(monkeypatch):
    monkeypatch.setattr(mxc_probe, "_host_prep_cache", {})
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: "id")
    monkeypatch.setattr(
        mxc_probe.mxc_runtime, "probe_host_prep_steps", lambda **_kwargs: ("prepare-null-device",)
    )
    monkeypatch.setattr(mxc_probe.mxc_adapter, "_control_environment", lambda: {})
    command = mxc_probe.host_prep_command()
    assert command[2:4] == ["--prepare-host", "--install-dir"]
    assert command[1].endswith("install_mxc_prebuilt.py")
    advice = mxc_probe.host_prep_remediation()
    assert subprocess.list2cmdline(command) in advice
    assert "Settings > Sandbox" in advice
