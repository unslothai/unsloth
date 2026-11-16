# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import html_to_markdown
from core.inference.tools import _render_page

_INTRO = " ".join(f"Install sentence {i} about setting up the project." for i in range(8))
_COMMANDS = {
    "npm": "npm install --save-dev jest",
    "Yarn": "yarn add --dev jest",
    "pnpm": "pnpm add --save-dev jest",
}


def _tabbed_page(hidden: str) -> str:
    tabs = "".join(
        f'<li role="tab" aria-selected="{i == 0}">{label}</li>' for i, label in enumerate(_COMMANDS)
    )
    panels = "".join(
        f'<div role="tabpanel"{"" if i == 0 else hidden}><pre><code>{cmd}</code></pre></div>'
        for i, cmd in enumerate(_COMMANDS.values())
    )
    return (
        "<html><head><title>Getting Started</title></head><body><main>"
        f"<h1>Getting Started</h1><p>{_INTRO}</p>"
        f'<div class="tabs-container"><ul role="tablist">{tabs}</ul><div>{panels}</div></div>'
        "</main></body></html>"
    )


@pytest.mark.parametrize("hidden", [" hidden", ' hidden=""', ' hidden="true"'])
@pytest.mark.parametrize("main_content", [True, False])
def test_every_tab_of_a_tabbed_docs_page_is_read(hidden, main_content):
    out = html_to_markdown(_tabbed_page(hidden), main_content = main_content)
    for command in _COMMANDS.values():
        assert command in out


def test_fetched_tabbed_page_reaches_the_model_with_every_tab():
    out = _render_page(
        ("html", "https://docs.example/getting-started", _tabbed_page(" hidden")), 20000
    )
    for command in _COMMANDS.values():
        assert command in out


def test_until_found_content_is_read():
    page = f"<main><h1>FAQ</h1><p>{_INTRO}</p><div hidden='until-found'><p>Searchable answer.</p></div></main>"
    assert "Searchable answer." in html_to_markdown(page, main_content = True)


@pytest.mark.parametrize(
    "element",
    [
        "<div hidden><p>Old answer.</p></div>",
        "<div role='dialog' hidden><p>Old answer.</p></div>",
        "<div role='tabpanel' aria-hidden='true'><p>Old answer.</p></div>",
        "<div role='tabpanel' style='display:none'><p>Old answer.</p></div>",
    ],
)
def test_other_hidden_content_stays_hidden(element):
    page = f"<main><h1>FAQ</h1><p>{_INTRO}</p>{element}</main>"
    assert "Old answer." not in html_to_markdown(page, main_content = True)
