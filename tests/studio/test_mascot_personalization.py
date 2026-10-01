# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account personalization keeps global mascot choices across old and new clients."""

import sys
from pathlib import Path

import pytest

BACKEND = str(Path(__file__).resolve().parents[2] / "studio" / "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
def settings(monkeypatch):
    from routes import settings

    stored = {}
    monkeypatch.setattr(settings, "get_personalization", lambda: stored)

    def save(value):
        stored.clear()
        stored.update(value)

    monkeypatch.setattr(settings, "set_personalization", save)
    return settings


def test_legacy_default_does_not_claim_an_explicit_mascot_choice(settings):
    response = settings.get_personalization_settings(current_subject = "test")
    assert response.appearance.customization.showMascots is True
    assert response.mascotsSaved is False
    settings.update_personalization_settings(
        settings.PersonalizationPayload.model_validate(
            {
                "appearance": {"customization": {"pointerCursors": True}},
                "profile": {"showGreetingSloth": False},
            }
        ),
        current_subject = "test",
    )
    response = settings.get_personalization_settings(current_subject = "test")
    assert response.appearance.customization.showMascots is True
    assert response.mascotsSaved is False
    assert response.profile.showGreetingSloth is False


def test_mascot_choice_roundtrips_and_survives_older_client_updates(settings):
    for enabled in (False, True):
        payload = settings.PersonalizationPayload.model_validate(
            {
                "appearance": {"customization": {"showMascots": enabled}},
            }
        )
        updated = settings.update_personalization_settings(payload, current_subject = "test")
        assert updated.appearance.customization.showMascots is enabled
        settings.update_personalization_settings(
            settings.PersonalizationPayload.model_validate(
                {
                    "appearance": {"customization": {"pointerCursors": True}},
                    "profile": {"showGreetingSloth": False},
                }
            ),
            current_subject = "test",
        )
        response = settings.get_personalization_settings(current_subject = "test")
        assert response.appearance.customization.showMascots is enabled
        assert response.mascotsSaved is True
        assert response.profile.showGreetingSloth is False
