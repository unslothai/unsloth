# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for API usage-example agent detection scope."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
USAGE_EXAMPLES_TSX = (
    REPO / "studio/frontend/src/features/settings/components/usage-examples.tsx"
)


def test_agent_detection_requires_desktop_scope():
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert 'import { isTauri } from "@/lib/api-base"' in src
    assert "function canUseLocalAgentDetection(base: string): boolean" in src
    helper = src[
        src.find("function canUseLocalAgentDetection") : src.find("const SHIKI_THEMES")
    ]
    assert "if (!isTauri) return false" in helper
    assert "isLoopbackHost(normalizeHost(new URL(base).hostname))" in helper
    assert "const localAgentDetection = canUseLocalAgentDetection(base)" in src
    assert "if (!localAgentDetection)" in src
    assert "}, [localAgentDetection]);" in src
