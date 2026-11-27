# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth-cli.py formats every dataset row with its own Alpaca prompt, which must be the standard one."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _string_assigned(path, name):
    for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _cli_prompt():
    return _string_assigned(_REPO_ROOT / "unsloth-cli.py", "alpaca_prompt")


def test_alpaca_prompt_lines_are_not_indented():
    indented = [line for line in _cli_prompt().splitlines() if line != line.lstrip()]
    assert indented == []


def test_alpaca_prompt_matches_studio_default():
    studio = _string_assigned(
        _REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "chat_templates.py",
        "DEFAULT_ALPACA_TEMPLATE",
    )
    assert _cli_prompt() == studio


def test_formatted_row_starts_each_section_at_column_zero():
    text = _cli_prompt().format("Add 2 and 3.", "", "5")
    assert "\n### Instruction:\nAdd 2 and 3.\n\n### Input:\n\n\n### Response:\n5" in text
