# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Chat UI permission step finds the Full access consent dialog by structure, not by its copy.

#12630 reworded the dialog ("Enable Full access?" became "Turn on Full access?", "I understand"
became "Turn on"). The consent flow was unchanged, but the step looked the heading and the confirm
button up by their old English names, so Chat UI went red on main. The step now reaches the title,
Cancel and confirm through the data-slot attributes the shared AlertDialog parts render. These
checks run in the fast shard and pin both halves of that contract: the step names no dialog copy,
and the dialog still renders those slots with a title naming the mode and a body warning about the
sandbox.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_DRIVER = _HERE / "playwright_chat_ui.py"
_ALERT_DIALOG = _ROOT / "studio" / "frontend" / "src" / "components" / "ui" / "alert-dialog.tsx"
_PERMISSION_SELECT = (
    _ROOT / "studio" / "frontend" / "src" / "features" / "chat" / "permission-mode-select.tsx"
)

_STEP = "exercise_permission_mode_controls"
_SLOTS = {
    "FULL_ACCESS_TITLE": ("alert-dialog-title", "AlertDialogTitle"),
    "FULL_ACCESS_CANCEL": ("alert-dialog-cancel", "AlertDialogCancel"),
    "FULL_ACCESS_CONFIRM": ("alert-dialog-action", "AlertDialogAction"),
}


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def _module() -> ast.Module:
    return ast.parse(_read(_DRIVER))


def _step() -> ast.FunctionDef:
    for node in _module().body:
        if isinstance(node, ast.FunctionDef) and node.name == _STEP:
            return node
    raise AssertionError(f"{_DRIVER.name} no longer defines {_STEP}")


def _constants() -> dict[str, str]:
    found = {}
    for node in _module().body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    found[target.id] = node.value.value
    return found


def _full_access_dialog_source() -> str:
    """The JSX of the component that renders the Full access consent card."""
    source = _read(_PERMISSION_SELECT)
    start = source.find("<AlertDialogContent")
    assert start != -1, f"{_PERMISSION_SELECT.name} no longer renders an AlertDialogContent"
    end = source.find("</AlertDialogContent>", start)
    assert end != -1, f"{_PERMISSION_SELECT.name}: AlertDialogContent is never closed"
    return source[start:end]


def test_the_consent_step_names_no_dialog_copy():
    """No dialog.get_by_role("heading" | "button", name = <literal>) in the step. Lookups on the
    permission menu itself are left alone: they take mode names, which are the product's terms."""
    pinned = []
    for node in ast.walk(_step()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get_by_role" or not node.args:
            continue
        role = node.args[0]
        if not (isinstance(role, ast.Constant) and role.value in ("heading", "button")):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id == "dialog":
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    pinned.append(f"line {node.lineno}: {role.value} {keyword.value.value!r}")
    assert not pinned, (
        "the Full access step looks the consent dialog up by its wording again, which breaks on any "
        f"copy edit; use the FULL_ACCESS_* slot selectors instead: {pinned}"
    )


@pytest.mark.parametrize("constant", sorted(_SLOTS))
def test_each_slot_selector_is_one_the_shared_dialog_renders(constant):
    slot, _component = _SLOTS[constant]
    value = _constants().get(constant)
    assert value == f'[data-slot="{slot}"]', f"{constant} is {value!r}"
    assert f'data-slot="{slot}"' in _read(
        _ALERT_DIALOG
    ), f"{_ALERT_DIALOG.name} no longer renders data-slot={slot!r}; {constant} would match nothing"
    assert constant in ast.unparse(_step()), f"{_STEP} no longer uses {constant}"


@pytest.mark.parametrize("constant", sorted(_SLOTS))
def test_the_full_access_dialog_is_built_from_those_parts(constant):
    _slot, component = _SLOTS[constant]
    assert (
        f"<{component}" in _full_access_dialog_source()
    ), f"the Full access consent dialog no longer uses {component}, so {constant} finds nothing"


def test_the_dialog_still_says_what_it_asks_consent_for():
    dialog = _full_access_dialog_source()
    title = re.search(r"<AlertDialogTitle\b[^>]*>(.*?)</AlertDialogTitle>", dialog, re.S)
    assert title and "Full access" in title.group(
        1
    ), "the consent title no longer names Full access"
    assert "sandbox" in dialog, "the consent dialog no longer warns that the sandbox is turned off"
