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
from routes.settings import (  # noqa: E402
    MAX_SIDEBAR_MENU_INPUT_ITEMS,
    PersonalizationPayload,
    SIDEBAR_MENU_ITEM_DEFAULTS,
)


def test_defaults_fill_missing_fields():
    p = PersonalizationPayload.model_validate({})
    assert p.version == pers.PERSONALIZATION_VERSION
    assert p.appearance.theme == "system"
    assert p.appearance.palette == "standard"
    assert p.profile.avatarShape == "circle"
    assert p.profile.displayName == ""
    assert p.profile.showGreetingSloth is True


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
    assert [(i.id, i.visible) for i in c.sidebarMenu] == [
        ("api", True),
        ("darkMode", True),
        ("guidedTour", True),
        ("profile", False),
        ("appearance", False),
        ("resources", False),
        ("chat", False),
        ("connections", False),
    ]


def test_customization_invalid_values_rejected():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"colors": {"light": {"accent": "red"}}}}}
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"uiFontSize": 99}}}
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"contrast": 500}}}
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"reduceMotion": "sometimes"}}}
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            {"appearance": {"customization": {"sidebarMenu": [{"id": "chats"}]}}}
        )


def test_customization_sidebar_menu_normalized():
    p = PersonalizationPayload.model_validate(
        {
            "appearance": {
                "customization": {
                    "sidebarMenu": [
                        {"id": "guidedTour", "visible": False},
                        {"id": "guidedTour", "visible": True},
                        {"id": "api"},
                    ]
                }
            }
        }
    )
    # Duplicates keep the first entry; missing ids are appended with their
    # default visibility.
    assert [(i.id, i.visible) for i in p.appearance.customization.sidebarMenu] == [
        ("guidedTour", False),
        ("api", True),
        ("darkMode", True),
        ("profile", False),
        ("appearance", False),
        ("resources", False),
        ("chat", False),
        ("connections", False),
    ]


def _sidebar(items):
    return {"appearance": {"customization": {"sidebarMenu": items}}}


def test_customization_sidebar_menu_dedupes_oversized_payload():
    # A stale/duplicated payload carries more items than there are distinct ids.
    # It must reach the dedupe validator and normalize to exactly one entry per
    # id, not be rejected by the length cap before dedupe runs.
    ids = list(SIDEBAR_MENU_ITEM_DEFAULTS)
    doubled = [{"id": i} for i in ids] + [{"id": i} for i in ids]
    assert len(doubled) > len(SIDEBAR_MENU_ITEM_DEFAULTS)
    p = PersonalizationPayload.model_validate(_sidebar(doubled))
    result = [i.id for i in p.appearance.customization.sidebarMenu]
    assert result == ids
    assert len(result) == len(SIDEBAR_MENU_ITEM_DEFAULTS)


def test_customization_sidebar_menu_rejects_pathological_length():
    # The generous input cap still refuses an absurdly long list outright.
    huge = [{"id": "api"} for _ in range(MAX_SIDEBAR_MENU_INPUT_ITEMS + 1)]
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(_sidebar(huge))


def test_customization_imported_fonts_validated():
    ok = PersonalizationPayload.model_validate(
        {
            "appearance": {
                "customization": {
                    "importedFonts": [
                        {"name": "My Font", "dataUrl": "data:font/woff2;base64,AAAA"}
                    ]
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
                            {
                                "name": "Evil",
                                "dataUrl": "https://example.com/font.woff2",
                            }
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
                            {
                                "name": f"Font {i}",
                                "dataUrl": "data:font/ttf;base64,AAAA",
                            }
                            for i in range(4)
                        ]
                    }
                }
            }
        )


def _imported(fonts):
    return {"appearance": {"customization": {"importedFonts": fonts}}}


def test_imported_font_name_rejects_css_characters():
    # Includes backslash (escapes the quoted family), comma/slash (extra
    # fallbacks / comment start), and a control character.
    for bad in [
        'Ev"il',
        "Ev;il",
        "Ev{il",
        "Ev<il",
        "Ev'il",
        "Ev\\il",
        "Ev,il",
        "Ev/il",
        "Ev\til",
    ]:
        with pytest.raises(ValidationError):
            PersonalizationPayload.model_validate(
                _imported([{"name": bad, "dataUrl": "data:font/woff2;base64,AAAA"}])
            )


def test_selected_font_names_validated():
    for field in ("uiFont", "headingFont", "chatFont", "codeFont"):
        for bad in ["Ev\\il", "Ev,il", "Ev/il", "Ev;il", "Ev\nil"]:
            with pytest.raises(ValidationError):
                PersonalizationPayload.model_validate(
                    {"appearance": {"customization": {field: bad}}}
                )
    # A normal family name (spaces + digits) is still accepted.
    p = PersonalizationPayload.model_validate(
        {"appearance": {"customization": {"uiFont": "Source Serif 4"}}}
    )
    assert p.appearance.customization.uiFont == "Source Serif 4"


def test_imported_font_data_url_must_be_base64():
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            _imported([{"name": "F", "dataUrl": "data:font/woff2;base64,?not base64?"}])
        )
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            _imported([{"name": "F", "dataUrl": "data:application/json;base64,AAAA"}])
        )


def test_imported_font_data_url_rejects_newline():
    # re's ``$`` also matches just before a trailing newline, so a data URL
    # ending in "\n" (or with an embedded newline) must be rejected the same way
    # the frontend JS pattern rejects it; otherwise the backend accepts a value
    # the client would never have produced.
    for bad in [
        "data:font/woff2;base64,AAAA\n",
        "data:font/woff2;base64,AAAA\nBBBB",
        "\ndata:font/woff2;base64,AAAA",
    ]:
        with pytest.raises(ValidationError):
            PersonalizationPayload.model_validate(
                _imported([{"name": "F", "dataUrl": bad}])
            )
    # The same URL without the newline is still accepted.
    ok = PersonalizationPayload.model_validate(
        _imported([{"name": "F", "dataUrl": "data:font/woff2;base64,AAAA"}])
    )
    assert (
        ok.appearance.customization.importedFonts[0].dataUrl
        == "data:font/woff2;base64,AAAA"
    )


def test_imported_fonts_total_size_capped():
    big = "data:font/woff2;base64," + "A" * 1_600_000
    with pytest.raises(ValidationError):
        PersonalizationPayload.model_validate(
            _imported([{"name": f"F{i}", "dataUrl": big} for i in range(3)])
        )
    # Two fit under the aggregate cap.
    PersonalizationPayload.model_validate(
        _imported([{"name": f"F{i}", "dataUrl": big} for i in range(2)])
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
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d)
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda d: store.update(d)
    )

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
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d)
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda d: store.update(d)
    )

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
            "showGreetingSloth": False,
        },
        "appearance": {
            "theme": "dark",
            "palette": "classic",
            "language": "en",
            "customization": {
                "colors": {
                    "light": {
                        "accent": "#339cff",
                        "background": None,
                        "foreground": None,
                    },
                    "dark": {
                        "accent": None,
                        "background": "#111111",
                        "foreground": None,
                    },
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
                "sidebarMenu": [
                    {"id": "darkMode", "visible": True},
                    {"id": "api", "visible": False},
                    {"id": "guidedTour", "visible": True},
                    {"id": "profile", "visible": True},
                    {"id": "appearance", "visible": False},
                    {"id": "resources", "visible": False},
                    {"id": "chat", "visible": False},
                    {"id": "connections", "visible": False},
                ],
            },
        },
    }
    put = client.put("/api/settings/personalization", json = payload)
    assert put.status_code == 200

    saved = client.get("/api/settings/personalization")
    assert saved.status_code == 200
    body = saved.json()
    assert body["saved"] is True
    assert body["customizationSaved"] is True
    assert body["paletteSaved"] is True
    assert body["greetingSlothSaved"] is True
    assert body["profile"] == payload["profile"]
    assert body["appearance"] == payload["appearance"]


def test_personalization_get_flags_legacy_fields(monkeypatch):
    # A record written before these fields existed must report them as unsaved
    # so the client keeps local overrides instead of the server-filled defaults.
    store: dict = {
        pers.PERSONALIZATION_SETTING_KEY: {
            "version": 1,
            "profile": {"displayName": "Mike"},
            "appearance": {"theme": "dark"},
        }
    }
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d)
    )

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    body = TestClient(app).get("/api/settings/personalization").json()
    assert body["saved"] is True
    assert body["customizationSaved"] is False
    assert body["paletteSaved"] is False
    assert body["greetingSlothSaved"] is False


def test_personalization_put_preserves_absent_fields(monkeypatch):
    # A stale client that omits palette/customization must not materialize them,
    # so the record stays legacy and GET keeps reporting those fields unsaved.
    store: dict = {}
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d)
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda d: store.update(d)
    )

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    client = TestClient(app)

    put = client.put(
        "/api/settings/personalization",
        json = {
            "version": 1,
            "profile": {"displayName": "Mike"},
            "appearance": {"theme": "dark", "language": "en"},
        },
    )
    assert put.status_code == 200

    stored_appearance = store[pers.PERSONALIZATION_SETTING_KEY]["appearance"]
    assert "palette" not in stored_appearance
    assert "customization" not in stored_appearance

    body = client.get("/api/settings/personalization").json()
    assert body["saved"] is True
    assert body["paletteSaved"] is False
    assert body["customizationSaved"] is False
    assert body["greetingSlothSaved"] is False


def test_personalization_put_preserves_existing_fields_on_stale_write(monkeypatch):
    # A stale client that omits palette/customization must not clobber values a
    # newer client already stored; the merge keeps them and only updates theme.
    store: dict = {
        pers.PERSONALIZATION_SETTING_KEY: {
            "version": 1,
            "profile": {"displayName": "Mike", "showGreetingSloth": False},
            "appearance": {
                "theme": "light",
                "palette": "classic",
                "customization": {"uiFont": "Georgia"},
            },
        }
    }
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d)
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda d: store.update(d)
    )

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    client = TestClient(app)

    put = client.put(
        "/api/settings/personalization",
        json = {
            "version": 1,
            "profile": {"displayName": "Mike"},
            "appearance": {"theme": "dark"},
        },
    )
    assert put.status_code == 200

    # The PUT response reflects the stored record, not the defaults-filled request:
    # a stale write that omits palette/customization still echoes the preserved values.
    put_body = put.json()
    assert put_body["appearance"]["theme"] == "dark"
    assert put_body["appearance"]["palette"] == "classic"
    assert put_body["appearance"]["customization"]["uiFont"] == "Georgia"

    stored_appearance = store[pers.PERSONALIZATION_SETTING_KEY]["appearance"]
    assert stored_appearance["theme"] == "dark"
    assert stored_appearance["palette"] == "classic"
    assert stored_appearance["customization"]["uiFont"] == "Georgia"

    body = client.get("/api/settings/personalization").json()
    assert body["paletteSaved"] is True
    assert body["customizationSaved"] is True
    assert body["greetingSlothSaved"] is True
