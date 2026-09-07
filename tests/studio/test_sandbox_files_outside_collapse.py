# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Created-file downloads must stay outside the collapsed tool card (#10425)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON_TOOL_UI = REPO / "studio/frontend/src/components/assistant-ui/tool-ui-python.tsx"
TERMINAL_TOOL_UI = REPO / "studio/frontend/src/components/assistant-ui/tool-ui-terminal.tsx"


def _jsx_inner(source: str, name: str) -> str:
    open_tag = f"<{name}"
    close_tag = f"</{name}>"
    start = source.index(open_tag)
    inner_start = source.index(">", start) + 1
    depth = 1
    i = inner_start
    while i < len(source) and depth:
        next_open = source.find(open_tag, i)
        next_close = source.find(close_tag, i)
        assert next_close >= 0, f"unclosed <{name}>"
        if next_open >= 0 and next_open < next_close:
            depth += 1
            i = next_open + len(open_tag)
            continue
        depth -= 1
        if depth == 0:
            return source[inner_start:next_close]
        i = next_close + len(close_tag)
    raise AssertionError(f"unclosed <{name}>")


def test_python_and_terminal_sandbox_files_are_outside_the_collapsible() -> None:
    for path in (PYTHON_TOOL_UI, TERMINAL_TOOL_UI):
        source = path.read_text(encoding = "utf-8")
        assert "<SandboxFiles" in source
        inner = _jsx_inner(source, "ToolFallbackContent")
        assert "<SandboxFiles" not in inner, (
            f"{path.name} hid SandboxFiles inside ToolFallbackContent"
        )
