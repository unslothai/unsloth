# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Starter characters: original cards bundled with Studio, given to each account once.

Each is a V2 card (``<id>.json``) with its portrait (``<id>.webp``) in assets/characters. A starter the
user deletes is not given again; a starter added in a later release reaches existing accounts too.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.characters import cards
from storage import characters_db as db

logger = logging.getLogger(__name__)

STARTER_DIR = Path(__file__).resolve().parents[2] / "assets" / "characters"
STARTERS = ("rhys-calloway", "rin-takeda", "elara-voss")


def seed_starters() -> None:
    """Adds any starter this account has not received yet. Never raises: the library must still list."""
    try:
        seeded = db.seeded_starters()
    except Exception:  # noqa: BLE001
        logger.warning("Could not read which starter characters were seeded", exc_info = True)
        return
    for starter in (s for s in STARTERS if s not in seeded):
        try:
            parsed = cards.parse_card_file(
                (STARTER_DIR / f"{starter}.json").read_bytes(), f"{starter}.json"
            )
            portrait_path = STARTER_DIR / f"{starter}.webp"
            portrait = portrait_path.read_bytes() if portrait_path.exists() else None
            db.create_character(
                cards.character_fields(parsed),
                card = parsed.card,
                avatar = portrait,
                avatar_type = cards.sniff_image_type(portrait) if portrait else None,
                starter_id = starter,
            )
        except db.LimitReached:
            return
        except Exception:  # noqa: BLE001
            logger.warning("Could not add the starter character %s", starter, exc_info = True)
