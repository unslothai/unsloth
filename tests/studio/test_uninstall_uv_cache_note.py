# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uninstall must say how to free a leftover shared uv cache."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UNINSTALLERS = (
    REPO / "scripts/uninstall.sh",
    REPO / "scripts/uninstall.ps1",
)


def test_both_uninstallers_name_uv_cache_prune_and_clean():
    # #9651: after uninstall the shared uv cache can still fill the disk. The
    # Hugging Face leftover already has a note; uv had none.
    for path in UNINSTALLERS:
        text = path.read_text(encoding = "utf-8")
        assert "uv cache prune" in text, f"{path.name} never mentions uv cache prune"
        assert "uv cache clean" in text, f"{path.name} never mentions uv cache clean"
