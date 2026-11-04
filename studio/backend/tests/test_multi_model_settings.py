# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep multiple models loaded: the off / fit / always mode, its migration and its route."""

from __future__ import annotations

import pytest

import utils.multi_model_settings as mms
from utils.account_context import OWNER, AccountContext, ROLE_USER, run_as


@pytest.fixture
def store(monkeypatch):
    """One fake studio.db per account, so a read from the wrong account shows up."""
    import storage.studio_db as studio_db
    from utils.account_context import current_account_id

    dbs: dict[str, dict] = {}
    writes: list[dict] = []

    def get_app_settings(keys):
        db = dbs.get(current_account_id(), {})
        return {key: db[key] for key in keys if key in db}

    def upsert_app_settings(settings, **_kwargs):
        writes.append(dict(settings))
        dbs.setdefault(current_account_id(), {}).update(settings)
        return dict(settings)

    monkeypatch.setattr(studio_db, "get_app_settings", get_app_settings)
    monkeypatch.setattr(studio_db, "upsert_app_settings", upsert_app_settings)
    mms.invalidate_multi_model_cache()
    yield dbs, writes
    mms.invalidate_multi_model_cache()


@pytest.mark.parametrize(
    ("stored", "mode"),
    [
        ({}, "off"),
        ({"keep_multiple_models_loaded": False}, "off"),
        ({"keep_multiple_models_loaded": False, "multi_model_mode": "fit"}, "off"),
        ({"keep_multiple_models_loaded": False, "multi_model_mode": "always"}, "off"),
        ({"multi_model_mode": "always"}, "off"),
        ({"keep_multiple_models_loaded": True}, "fit"),
        ({"keep_multiple_models_loaded": True, "multi_model_mode": "fit"}, "fit"),
        ({"keep_multiple_models_loaded": True, "multi_model_mode": "off"}, "fit"),
        ({"keep_multiple_models_loaded": True, "multi_model_mode": "bogus"}, "fit"),
        ({"keep_multiple_models_loaded": True, "multi_model_mode": "always"}, "always"),
        ({"keep_multiple_models_loaded": "yes", "multi_model_mode": "always"}, "off"),
    ],
)
def test_the_bool_decides_on_or_off_and_the_mode_picks_fit_or_always(store, stored, mode):
    dbs, _ = store
    dbs[OWNER.account_id] = dict(stored)
    assert mms.get_multi_model_mode() == mode
    assert mms.get_multi_model_enabled() is (mode != "off")


def test_both_keys_are_written_in_one_call(store):
    dbs, writes = store
    assert mms.set_multi_model_mode("always") is True
    assert writes == [{"multi_model_mode": "always", "keep_multiple_models_loaded": True}]
    assert mms.set_multi_model_mode("always") is False
    assert mms.set_multi_model_mode("off") is True
    assert writes[-1] == {"multi_model_mode": "off", "keep_multiple_models_loaded": False}
    assert dbs[OWNER.account_id]["keep_multiple_models_loaded"] is False


@pytest.mark.parametrize("bad", ["on", "", None, True, "Always"])
def test_a_bad_mode_is_refused_with_the_setting_name(store, bad):
    _, writes = store
    with pytest.raises(
        ValueError, match = "^Keep multiple models loaded must be off, fit or always.$"
    ):
        mms.set_multi_model_mode(bad)
    assert writes == []


def test_a_write_invalidates_the_cached_mode(store):
    dbs, _ = store
    assert mms.get_multi_model_mode() == "off"
    # A write made behind the module's back stays hidden for the TTL...
    dbs[OWNER.account_id] = {"keep_multiple_models_loaded": True}
    assert mms.get_multi_model_mode() == "off"
    # ...but one made through it shows at once.
    mms.set_multi_model_mode("always")
    assert mms.get_multi_model_mode() == "always"


def test_a_managed_account_reads_and_writes_the_owner_database(store):
    dbs, _ = store
    alice = AccountContext("alice", "alice", ROLE_USER)
    dbs[OWNER.account_id] = {"keep_multiple_models_loaded": True, "multi_model_mode": "always"}
    dbs["alice"] = {"keep_multiple_models_loaded": False}
    assert run_as(alice, mms.get_multi_model_mode) == "always"
    run_as(alice, mms.set_multi_model_mode, "fit")
    assert dbs[OWNER.account_id]["multi_model_mode"] == "fit"
    assert dbs["alice"] == {"keep_multiple_models_loaded": False}


def test_an_unreadable_database_reads_as_off(monkeypatch):
    import storage.studio_db as studio_db

    def broken(_keys):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_settings", broken)
    mms.invalidate_multi_model_cache()
    assert mms.get_multi_model_mode() == "off"


def _put(payload):
    from fastapi import BackgroundTasks
    import routes.settings as settings_routes

    tasks = BackgroundTasks()
    response = settings_routes.update_multi_model(
        settings_routes.MultiModelPayload(**payload), tasks, "s"
    )
    return response, tasks


def test_the_route_takes_a_mode_and_reports_both_shapes(store):
    response, tasks = _put({"mode": "always"})
    assert (response.mode, response.enabled, response.default_mode) == ("always", True, "off")
    assert response.default_enabled is False and not tasks.tasks
    import routes.settings as settings_routes

    current = settings_routes.get_multi_model("s")
    assert (current.mode, current.enabled) == ("always", True)


def test_a_legacy_enabled_true_keeps_always(store):
    _put({"mode": "always"})
    response, _ = _put({"enabled": True})
    assert response.mode == "always"


def test_a_legacy_enabled_true_turns_off_into_fit(store):
    response, _ = _put({"enabled": True})
    assert (response.mode, response.enabled) == ("fit", True)


def test_turning_it_off_queues_the_idle_unload(store):
    _put({"mode": "fit"})
    response, tasks = _put({"enabled": False})
    assert response.mode == "off" and len(tasks.tasks) == 1
    response, tasks = _put({"mode": "off"})
    assert len(tasks.tasks) == 1


@pytest.mark.parametrize(
    "payload", [{}, {"mode": "fit", "enabled": True}, {"mode": "sometimes"}, {"enabled": "yes"}]
)
def test_the_payload_needs_exactly_one_valid_field(payload):
    import routes.settings as settings_routes
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        settings_routes.MultiModelPayload(**payload)


def test_an_unreadable_setting_keeps_the_last_mode_read(monkeypatch):
    def broken():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mms, "_read_mode", broken)
    mms.invalidate_multi_model_cache()
    assert mms.get_multi_model_mode() == "off"
    monkeypatch.setattr(mms, "_cache", (-1e9, "always"))
    assert mms.get_multi_model_mode() == "always"
    mms.invalidate_multi_model_cache()
