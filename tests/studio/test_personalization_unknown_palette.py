# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stored palette this build does not know must read as unset, not fail the response."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[2] / "studio" / "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from utils.personalization_settings import drop_unknown_palette  # noqa: E402

PALETTES = frozenset({"standard", "matcha"})


def test_unknown_palette_is_dropped_and_the_rest_kept():
    stored = {"version": 1, "appearance": {"palette": "neon-ramen", "theme": "dark"}}
    cleaned = drop_unknown_palette(stored, PALETTES)
    assert cleaned["appearance"] == {"theme": "dark"}
    assert cleaned["version"] == 1
    # The stored record itself is left alone.
    assert stored["appearance"]["palette"] == "neon-ramen"


def test_known_or_missing_palette_is_untouched():
    for stored in (
        {"appearance": {"palette": "matcha"}},
        {"appearance": {"theme": "light"}},
        {"profile": {}},
        {},
    ):
        assert drop_unknown_palette(stored, PALETTES) is stored


def test_routes_read_and_answer_through_the_filter():
    src = (Path(BACKEND) / "routes" / "settings.py").read_text()
    assert "drop_unknown_palette(get_personalization(), _PALETTE_IDS)" in src
    assert (
        "PersonalizationPayload.model_validate(drop_unknown_palette(merged, _PALETTE_IDS))" in src
    )
