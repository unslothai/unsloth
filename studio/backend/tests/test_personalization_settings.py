# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.personalization_settings as pers  # noqa: E402
from auth.authentication import get_current_subject  # noqa: E402
from routes import settings as settings_routes  # noqa: E402
from routes.settings import PersonalizationPayload  # noqa: E402


def test_defaults_fill_missing_fields():
    p = PersonalizationPayload.model_validate({})
    assert p.version == pers.PERSONALIZATION_VERSION
    assert p.appearance.theme == "system"
    assert p.appearance.palette == "standard"
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


def test_invalid_palette_rejected():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate({"appearance": {"palette": "neon"}})


def test_customization_defaults():
    p = PersonalizationPayload.model_validate({})
    c = p.appearance.customization
    assert c.contrast == 50
    assert c.reduceMotion == "system"
    assert c.fontSmoothing is True
    assert c.pointerCursors is False
    assert c.colors.light.accent is None
    assert c.headingFont is None
    assert c.chatFont is None
    assert c.uiFontSize is None


def test_customization_invalid_values_rejected():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"colors": {"light": {"accent": "red"}}}}}
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate({"appearance": {"customization": {"uiFontSize": 99}}})
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate({"appearance": {"customization": {"contrast": 500}}})
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"reduceMotion": "sometimes"}}}
        )


def test_customization_imported_fonts_validated():
    ok = PersonalizationPayload.model_validate(
        {
            "appearance": {
                "customization": {
                    "importedFonts": [{"name": "My Font", "dataUrl": "data:font/woff2;base64,AAAA"}]
                }
            }
        }
    )
    assert ok.appearance.customization.importedFonts[0].name == "My Font"
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {
                "appearance": {
                    "customization": {
                        "importedFonts": [
                            {"name": "Evil", "dataUrl": "https://example.com/font.woff2"}
                        ]
                    }
                }
            }
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {
                "appearance": {
                    "customization": {
                        "importedFonts": [
                            {"name": f"Font {i}", "dataUrl": "data:font/ttf;base64,AAAA"}
                            for i in range(4)
                        ]
                    }
                }
            }
        )


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


def test_bundled_avatar_url_allowed():
    p = PersonalizationPayload.model_validate(
        {"profile": {"avatarDataUrl": "/Sloth%20emojis/large%20sloth%20yay.png"}}
    )
    assert p.profile.avatarDataUrl == "/Sloth%20emojis/large%20sloth%20yay.png"


def test_bundled_avatar_subpath_allowed():
    p = PersonalizationPayload.model_validate(
        {"profile": {"avatarDataUrl": "/studio/Sloth%20emojis/large%20sloth%20yay.png"}}
    )
    assert "Sloth%20emojis" in p.profile.avatarDataUrl


def test_bundled_avatar_traversal_rejected():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"profile": {"avatarDataUrl": "/Sloth%20emojis/../secret.png"}}
        )


def test_get_read_errors_propagate(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("read failed")

    monkeypatch.setattr("storage.studio_db.get_app_setting", fail)

    with pytest.raises(RuntimeError, match = "read failed"):
        pers.get_personalization()


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


def test_personalization_route_roundtrip_real_shape(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", lambda d: store.update(d))

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    client = TestClient(app)

    initial = client.get("/api/settings/personalization")
    assert initial.status_code == 200
    assert initial.json()["saved"] is False

    payload = {
        "version": 1,
        "profile": {
            "displayName": "Mike",
            "nickname": "M",
            "avatarDataUrl": "/Sloth%20emojis/large%20sloth%20yay.png",
            "avatarShape": "rounded",
        },
        "appearance": {
            "theme": "dark",
            "palette": "classic",
            "language": "en",
            "customization": {
                "colors": {
                    "light": {"accent": "#339cff", "background": None, "foreground": None},
                    "dark": {"accent": None, "background": "#111111", "foreground": None},
                },
                "uiFont": "SF Pro Text",
                "headingFont": "Avenir Next",
                "chatFont": "Georgia",
                "codeFont": None,
                "importedFonts": [
                    {"name": "SF Pro Text", "dataUrl": "data:font/woff2;base64,AAAA"}
                ],
                "uiFontSize": 14,
                "codeFontSize": 13,
                "contrast": 60,
                "pointerCursors": True,
                "reduceMotion": "off",
                "fontSmoothing": True,
            },
        },
    }
    put = client.put("/api/settings/personalization", json = payload)
    assert put.status_code == 200

    saved = client.get("/api/settings/personalization")
    assert saved.status_code == 200
    body = saved.json()
    assert body["saved"] is True
    assert body["profile"] == payload["profile"]
    assert body["appearance"] == payload["appearance"]
