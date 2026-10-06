# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every action-bar control the thread browser harnesses ask for is still rendered there.

#12735 took Delete off the assistant action bar and made it the last item of the More menu.
playwright_heavy_thread.py kept asking for a "Delete message" button, so its delete action could
not run, and probe_dismiss_guard.py kept attacking that button, so every one of its cases would
have errored. Both sit behind earlier steps of the same CI job, which hid the breakage until those
steps went green. This reads the labels the harnesses ask for and requires each to be rendered by
the assistant action bar in thread.tsx, so the next move of a control fails here, with its name.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio" / "frontend" / "src" / "components" / "assistant-ui" / "thread.tsx"
HARNESSES = (
    "tests/studio/playwright_heavy_thread.py",
    "tests/studio/playwright_thread_weight.py",
    "tests/studio/probe_dismiss_guard.py",
)


def _component_body(tsx: str, name: str) -> str | None:
    """The source of a component defined in thread.tsx, `const X: FC = ...` or `function X(`."""
    match = re.search(rf"^(?:const {name}\b[^\n]*=>\s*\{{|function {name}\()", tsx, re.M)
    if not match:
        return None
    end = tsx.find("\n}", match.end())
    return tsx[match.start() : end if end != -1 else len(tsx)]


def _assistant_bar_source(tsx: str) -> str:
    """AssistantActionBar and every thread.tsx component it renders, transitively.

    The harnesses act on the last ASSISTANT message, so a control only counts if the assistant
    bar renders it: a whole-file search would still find a label in the user message's bar.
    """
    seen: dict[str, str] = {}
    queue = ["AssistantActionBar"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        body = _component_body(tsx, name)
        if body is None:
            continue
        seen[name] = body
        queue.extend(re.findall(r"<([A-Z][A-Za-z0-9]*)\b(?!\.)", body))
    assert "AssistantActionBar" in seen, "could not find AssistantActionBar in thread.tsx"
    return "\n".join(seen.values())


def _asked_for() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Bar buttons (`actionButton("X")`) and More-menu items (`.trim() === "X"`) per harness."""
    buttons: dict[str, set[str]] = {}
    items: dict[str, set[str]] = {}
    for rel in HARNESSES:
        text = (REPO / rel).read_text(encoding = "utf-8")
        buttons[rel] = set(re.findall(r"\bactionButton\(\s*[\"']([^\"']+)[\"']\s*\)", text))
        items[rel] = set(re.findall(r"\.trim\(\)\s*===\s*\"([^\"]+)\"", text))
    return buttons, items


def _rendered(tsx: str) -> tuple[set[str], set[str]]:
    bar = _assistant_bar_source(tsx)
    tooltips = set(re.findall(r"tooltip=\"([^\"]+)\"", bar))
    # An item's label is its last line of text before the closing tag. Its attributes can hold
    # arrow functions, so the opening tag cannot be skipped with a `[^>]*` match.
    menu_items = set()
    for body in re.findall(
        r"<ActionBarMorePrimitive\.Item\b([\s\S]*?)</ActionBarMorePrimitive\.Item>", bar
    ):
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if lines and not lines[-1].startswith(("<", "{", "/")):
            menu_items.add(lines[-1])
    return tooltips, menu_items


def test_the_scan_reads_the_controls_the_harnesses_use():
    buttons, items = _asked_for()
    assert {"More"} <= buttons["tests/studio/playwright_heavy_thread.py"]
    assert {"More", "Copy"} <= buttons["tests/studio/probe_dismiss_guard.py"]
    assert "Delete" in items["tests/studio/playwright_heavy_thread.py"]
    assert "Delete" in items["tests/studio/playwright_thread_weight.py"]
    tooltips, menu_items = _rendered(THREAD_TSX.read_text(encoding = "utf-8"))
    assert {"Copy", "More"} <= tooltips and "Delete" in menu_items, (
        "parsed fewer controls out of thread.tsx than the assistant bar renders; "
        "the check below would pass on nothing"
    )


def test_every_control_a_harness_asks_for_is_still_rendered():
    tooltips, menu_items = _rendered(THREAD_TSX.read_text(encoding = "utf-8"))
    buttons, items = _asked_for()
    missing = {
        rel: sorted(buttons[rel] - tooltips) + sorted(items[rel] - menu_items)
        for rel in HARNESSES
        if (buttons[rel] - tooltips) or (items[rel] - menu_items)
    }
    assert not missing, (
        f"these harnesses ask for controls the assistant action bar in thread.tsx no longer "
        f"renders as a tooltip or a More-menu item, so the action would not run: {missing}"
    )
