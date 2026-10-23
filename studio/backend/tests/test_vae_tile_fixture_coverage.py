# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each family needs a pinned VAE config in the seam fixture; this copy runs without torch."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import diffusion_families as DF
from core.inference import video_families as VF

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vae_tile_configs.json"


def _families() -> dict:
    return json.loads(FIXTURE.read_text(encoding = "utf-8"))["families"]


def test_every_registry_family_has_a_pinned_vae_config():
    fixture = _families()
    missing = [f.name for f in (*DF._FAMILIES, *VF._FAMILIES) if f.name not in fixture]
    assert not missing, (
        f"families without a VAE config in {FIXTURE.name}: {missing}. Add each one's vae/config.json "
        "(base repo, pinned revision) so the seam guard checks the tile geometry Studio decodes it with."
    )


@pytest.mark.parametrize("name", sorted(_families()))
def test_each_entry_is_pinned_and_complete(name):
    entry = _families()[name]
    assert entry.get("kind") in ("image", "video"), f"{name}: kind {entry.get('kind')!r}"
    assert entry.get("repo"), f"{name}: no base repo"
    assert re.fullmatch(
        r"[0-9a-f]{40}", entry.get("revision", "")
    ), f"{name}: revision is not a commit sha"
    assert isinstance(entry.get("config"), dict) and entry["config"].get(
        "_class_name"
    ), f"{name}: config is not a diffusers vae/config.json"
