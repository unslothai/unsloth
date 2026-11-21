# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bundled starter cards (``<id>.json`` + ``<id>.webp``): each reaches an account once, even one added in a later release."""

from __future__ import annotations

import logging
from pathlib import Path

from core.characters import cards
from storage import characters_db as db
from storage.studio_db import studio_db_path
from utils.image_files import sniff_image_type

logger = logging.getLogger(__name__)

STARTER_DIR = Path(__file__).resolve().parents[2] / "assets" / "characters"
STARTERS = ("leo", "rin-takeda")
# Accounts already holding every starter, so a list GET skips the query.
_fully_seeded_db_paths: set = set()


def seed_starters() -> None:
    """Adds any starter this account has not received yet. Never raises: the library must still list."""
    db_path = studio_db_path()
    if db_path in _fully_seeded_db_paths:
        return
    try:
        seeded = db.seeded_starters()
    except Exception:  # noqa: BLE001
        logger.warning("Could not read which starter characters were seeded", exc_info = True)
        return
    if all(s in seeded for s in STARTERS):
        _fully_seeded_db_paths.add(db_path)
        return
    for starter in (s for s in STARTERS if s not in seeded):
        try:
            parsed = cards.parse_card_file(
                (STARTER_DIR / f"{starter}.json").read_bytes(), f"{starter}.json"
            )
            avatar_path = STARTER_DIR / f"{starter}.webp"
            avatar = avatar_path.read_bytes() if avatar_path.exists() else None
            db.create_character(
                cards.character_fields(parsed),
                card = parsed.card,
                avatar = avatar,
                avatar_type = sniff_image_type(avatar) if avatar else None,
                starter_id = starter,
            )
        except db.CharacterLimitReached:
            return
        except Exception:  # noqa: BLE001
            logger.warning("Could not add the starter character %s", starter, exc_info = True)
