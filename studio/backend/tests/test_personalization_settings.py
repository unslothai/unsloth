# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server-side personalization validation + persistence roundtrip."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.personalization_settings as pers  # noqa: E402
from routes.settings import PersonalizationPayload  # noqa: E402


def test_defaults_fill_missing_fields():
    p = PersonalizationPayload.model_validate({})
    assert p.version == pers.PERSONALIZATION_VERSION
    assert p.appearance.theme == "system"
    assert p.profile.avatarShape == "circle"
    assert p.profile.displayName == ""


def test_unknown_keys_are_ignored():
    p = PersonalizationPayload.model_validate(
        {"profile": {"displayName": "Mike", "bogus": 1}, "extra": True}
    )
    assert p.profile.displayName == "Mike"


def test_invalid_theme_rejected():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate({"appearance": {"theme": "neon"}})


def test_avatar_must_be_image_data_url():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"profile": {"avatarDataUrl": "http://example.com/a.png"}}
        )


def test_avatar_size_is_capped():
    big = "data:image/png;base64," + "A" * (pers.MAX_AVATAR_DATA_URL_BYTES + 1)
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate({"profile": {"avatarDataUrl": big}})


def test_valid_avatar_and_shape():
    p = PersonalizationPayload.model_validate(
        {
            "profile": {
                "avatarDataUrl": "data:image/png;base64,AAAA",
                "avatarShape": "rounded",
            }
        }
    )
    assert p.profile.avatarShape == "rounded"


def test_get_set_roundtrip(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", lambda d: store.update(d))

    assert pers.get_personalization() == {}
    data = {
        "version": 1,
        "profile": {"displayName": "Mike"},
        "appearance": {"theme": "dark"},
    }
    pers.set_personalization(data)
    assert store[pers.PERSONALIZATION_SETTING_KEY]["profile"]["displayName"] == "Mike"
    assert pers.get_personalization()["appearance"]["theme"] == "dark"
