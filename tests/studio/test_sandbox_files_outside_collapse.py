# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Created-file downloads must stay outside the collapsed tool card (#10425)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON_TOOL_UI = REPO / "studio/frontend/src/components/assistant-ui/tool-ui-python.tsx"
TERMINAL_TOOL_UI = REPO / "studio/frontend/src/components/assistant-ui/tool-ui-terminal.tsx"


def test_python_and_terminal_sandbox_files_are_outside_the_collapsible() -> None:
    for path in (PYTHON_TOOL_UI, TERMINAL_TOOL_UI):
        source = path.read_text(encoding = "utf-8")
        assert "<SandboxFiles" in source
        assert source.index("</ToolFallbackContent>") < source.index("<SandboxFiles")
