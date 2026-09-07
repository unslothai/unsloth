# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Switch Back looks up remembered config through the resident alias (#10338)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CHAT_PAGE = REPO / "studio/frontend/src/features/chat/chat-page.tsx"


def test_remembered_config_for_uses_the_resident_lookup() -> None:
    source = CHAT_PAGE.read_text(encoding = "utf-8")
    assert "resolveResidentInitialConfig(selection.id, selection.ggufVariant)" in source
    assert "resolveInitialConfig(selection.id, selection.ggufVariant)" not in source
