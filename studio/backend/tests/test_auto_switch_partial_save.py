# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The auto-switch PUT without ``enabled`` (a row saved from Model memory) and the origin-aware
KV purge that save runs."""

from __future__ import annotations

import pytest

import routes.settings as settings_route
import storage.studio_db as db
from core.inference import llama_keepwarm as kw
from utils import openai_auto_switch_settings as settings


@pytest.fixture
def store(monkeypatch):
    stored = {settings.OPENAI_AUTO_SWITCH_SETTING_KEY: True}
    writes: list[dict] = []

    def _upsert(updates):
        writes.append(dict(updates))
        stored.update(updates)

    monkeypatch.setattr(db, "upsert_app_settings", _upsert)
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None, *a: stored.get(k, d))
    monkeypatch.setattr(kw, "_kv_resume", None)
    return stored, writes


def test_a_save_without_enabled_keeps_the_stored_switch(store):
    stored, writes = store
    payload = settings_route.OpenAIAutoSwitchPayload(media_auto_unload_idle_seconds = 300)
    assert payload.enabled is None
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.enabled is True
    assert resp.media_auto_unload_idle_seconds == 300
    assert stored[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] is True
    assert writes == [{settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY: 300}]


def test_the_setter_treats_a_missing_enabled_as_untouched(store):
    stored, writes = store
    assert settings.set_openai_auto_switch(None, None)[0] is True
    assert writes == [{}]  # upsert_app_settings({}) is a no-op on the real store
    assert settings.set_openai_auto_switch(False, None)[0] is False
    assert writes[-1] == {settings.OPENAI_AUTO_SWITCH_SETTING_KEY: False}
    with pytest.raises(ValueError) as exc:
        settings.set_openai_auto_switch("garbage", None)
    assert str(exc.value).startswith("OpenAI auto-switch must be")


@pytest.mark.parametrize(
    "origin, keep_kv, configured, kept",
    [
        ("swap", True, False, True),  # a media TTL save keeps the swap stash
        ("swap", False, False, False),  # keep-KV off purges every stash
        ("idle", True, False, False),
        (None, True, False, False),  # an untagged (older) stash counts as idle
        ("idle", True, True, True),
    ],
)
def test_the_purge_rule_follows_the_stash_origin(
    store, monkeypatch, tmp_path, origin, keep_kv, configured, kept
):
    saved = tmp_path / "resume-x-slot0.bin"
    saved.write_bytes(b"kv")
    stash = {
        "identity": ("m", None, "m"),
        "dir": str(tmp_path),
        "slots": [{"filename": saved.name}],
    }
    if origin:
        stash["origin"] = origin
    kw._kv_resume = stash
    monkeypatch.setattr(
        settings_route,
        "set_openai_auto_switch",
        lambda *a: (True, 0, keep_kv, False, False, 300, False),
    )
    monkeypatch.setattr(settings_route, "idle_unload_is_configured", lambda: configured)
    payload = settings_route.OpenAIAutoSwitchPayload(media_auto_unload_idle_seconds = 300)
    settings_route.update_openai_auto_switch(payload, "tester")
    assert (kw._kv_resume is stash) is kept
    assert saved.exists() is kept


def test_purge_with_an_origin_leaves_another_origin_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(kw, "_kv_resume", {"origin": "swap", "dir": str(tmp_path), "slots": []})
    kw.purge_kv_resume(origin = "idle")
    assert kw._kv_resume is not None
    kw.purge_kv_resume(origin = "swap")
    assert kw._kv_resume is None
