"""
Regression guard: Studio's shadcn/Radix menu items, sub-triggers, and the
chat search command row all use `cursor-pointer`, sub-trigger variants
carry `data-[disabled]:pointer-events-none data-[disabled]:opacity-50`,
and `index.css` keeps the base-layer `:where(...)` interactive-element
cursor rule (with disabled exclusions) plus the label:has(...) rule
that restores pointer cursor on labels wrapping enabled checkbox/radio
inputs.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[2]
UI = WORKDIR / "studio" / "frontend" / "src" / "components" / "ui"
COMMAND = UI / "command.tsx"
CONTEXT_MENU = UI / "context-menu.tsx"
DROPDOWN_MENU = UI / "dropdown-menu.tsx"
MENUBAR = UI / "menubar.tsx"
INDEX_CSS = WORKDIR / "studio" / "frontend" / "src" / "index.css"
CHAT_SEARCH = (
    WORKDIR
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "chat"
    / "components"
    / "chat-search-dialog.tsx"
)


def _read(path: Path) -> str:
    assert path.exists(), f"missing source file: {path}"
    return path.read_text()


def _component_class_string(src: str, fn_name: str) -> str:
    """Return the first className string literal inside the named function."""
    fn_match = re.search(rf"function\s+{re.escape(fn_name)}\s*\(", src)
    assert fn_match, f"could not locate {fn_name} in source"
    tail = src[fn_match.end():]
    cls_match = re.search(r'"([^"]*?cursor-[^"]*?)"', tail)
    assert cls_match, f"could not locate className for {fn_name}"
    return cls_match.group(1)


def test_command_item_has_cursor_pointer():
    src = _read(COMMAND)
    cls = _component_class_string(src, "CommandItem")
    assert "cursor-pointer" in cls
    assert "cursor-default" not in cls


def test_context_menu_item_variants_have_cursor_pointer_and_disabled_guards():
    src = _read(CONTEXT_MENU)
    for fn in (
        "ContextMenuItem",
        "ContextMenuSubTrigger",
        "ContextMenuCheckboxItem",
        "ContextMenuRadioItem",
    ):
        cls = _component_class_string(src, fn)
        assert "cursor-pointer" in cls, f"{fn}: missing cursor-pointer"
        assert "cursor-default" not in cls, f"{fn}: still uses cursor-default"
        assert (
            "data-[disabled]:pointer-events-none" in cls
        ), f"{fn}: missing data-[disabled]:pointer-events-none"
        assert (
            "data-[disabled]:opacity-50" in cls
        ), f"{fn}: missing data-[disabled]:opacity-50"


def test_dropdown_menu_item_variants_have_cursor_pointer_and_disabled_guards():
    src = _read(DROPDOWN_MENU)
    for fn in (
        "DropdownMenuItem",
        "DropdownMenuSubTrigger",
        "DropdownMenuCheckboxItem",
        "DropdownMenuRadioItem",
    ):
        cls = _component_class_string(src, fn)
        assert "cursor-pointer" in cls, f"{fn}: missing cursor-pointer"
        assert "cursor-default" not in cls, f"{fn}: still uses cursor-default"
        assert (
            "data-[disabled]:pointer-events-none" in cls
        ), f"{fn}: missing data-[disabled]:pointer-events-none"
        assert (
            "data-[disabled]:opacity-50" in cls
        ), f"{fn}: missing data-[disabled]:opacity-50"


def test_menubar_item_variants_have_cursor_pointer_and_bracketed_disabled_guards():
    src = _read(MENUBAR)
    for fn in (
        "MenubarItem",
        "MenubarSubTrigger",
        "MenubarCheckboxItem",
        "MenubarRadioItem",
    ):
        cls = _component_class_string(src, fn)
        assert "cursor-pointer" in cls, f"{fn}: missing cursor-pointer"
        assert "cursor-default" not in cls, f"{fn}: still uses cursor-default"
        assert (
            "data-[disabled]:pointer-events-none" in cls
        ), f"{fn}: missing data-[disabled]:pointer-events-none"
        assert (
            "data-[disabled]:opacity-50" in cls
        ), f"{fn}: missing data-[disabled]:opacity-50"
        assert (
            re.search(r"data-disabled:(?!\[)", cls) is None
        ), f"{fn}: still uses non-bracketed data-disabled:* (Tailwind v4 generates no CSS for it)"


def test_subtriggers_share_svg_pointer_events_and_shrink_utilities():
    for path, fn in (
        (CONTEXT_MENU, "ContextMenuSubTrigger"),
        (DROPDOWN_MENU, "DropdownMenuSubTrigger"),
        (MENUBAR, "MenubarSubTrigger"),
    ):
        cls = _component_class_string(_read(path), fn)
        assert (
            "[&_svg]:pointer-events-none" in cls
        ), f"{fn}: missing [&_svg]:pointer-events-none"
        assert "[&_svg]:shrink-0" in cls, f"{fn}: missing [&_svg]:shrink-0"


def test_chat_search_dialog_command_item_uses_cursor_pointer():
    src = _read(CHAT_SEARCH)
    item_match = re.search(
        r"<CommandPrimitive\.Item\b[\s\S]*?className=\"([^\"]+)\"",
        src,
    )
    assert item_match, "could not locate CommandPrimitive.Item className in chat search dialog"
    cls = item_match.group(1)
    assert "cursor-pointer" in cls
    assert "cursor-default" not in cls


def test_index_css_has_base_interactive_cursor_rule_with_disabled_exclusions():
    css = _read(INDEX_CSS)
    where_block = re.search(
        r":where\(\s*([^)]*?)\)\s*:not\(([^)]*?)\)\s*\{\s*cursor:\s*pointer;\s*\}",
        css,
    )
    assert where_block, "expected base-layer :where(...):not(...) cursor:pointer rule in index.css"
    selectors = where_block.group(1)
    exclusions = where_block.group(2)
    for token in (
        "button",
        '[role="button"]',
        '[role="switch"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="menuitem"]',
        '[role="menuitemcheckbox"]',
        '[role="menuitemradio"]',
        '[role="option"]',
        '[role="tab"]',
        "summary",
    ):
        assert token in selectors, f"base :where(...) is missing selector: {token}"
    for token in (":disabled", '[aria-disabled="true"]', "[data-disabled]"):
        assert token in exclusions, f":where(...):not(...) is missing exclusion: {token}"


def test_index_css_label_has_rule_excludes_disabled_inputs():
    css = _read(INDEX_CSS)
    label_block = re.search(
        r"label:has\([^{]*?\)\s*,\s*label:has\([^{]*?\)\s*\{\s*cursor:\s*pointer;\s*\}",
        css,
    )
    assert label_block, "expected label:has(...) cursor:pointer rule in index.css"
    block_text = label_block.group(0)
    assert 'input[type="checkbox"]:not(:disabled)' in block_text
    assert 'input[type="radio"]:not(:disabled)' in block_text
