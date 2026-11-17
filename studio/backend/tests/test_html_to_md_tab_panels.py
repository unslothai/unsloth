# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import _TabLabelScan, html_to_markdown
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
        "<div hidden><a role='tab' id='h'>Old answer.</a></div><div role='tabpanel' aria-labelledby='h'>x</div>",
        "<a role='tab' id='h'>x<template>Old answer.</template></a><div role='tabpanel' aria-labelledby='h'>x</div>",
        "<head><title role='tab' id='h'>Old answer.</title></head><div role='tabpanel' aria-labelledby='h'>x</div>",
    ],
)
def test_other_hidden_content_stays_hidden(element):
    page = f"<main><h1>FAQ</h1><p>{_INTRO}</p>{element}</main>"
    assert "Old answer." not in html_to_markdown(page, main_content = True)


_LINK_TABS = "".join(
    f'<a role="tab" id="t{i}"><svg><title>icon</title></svg> {label}</a>'
    for i, label in enumerate(_COMMANDS)
)
# end tags omitted, the last one too; self-closing void tags; nested same-tag spans; text that is not shown
_LIST_TABS = (
    '<ul role="tablist"><li role="tab" id="t0"><img src="npm.svg" alt=""/><span aria-hidden="true">box</span>npm'
    '<li role="tab" id="t1"><span><span>Yarn</span></span><span hidden>(current)</span><br/>'
    '<li role="tab" id="t2">pnpm</ul>'
)


def _labelled_panels(i: int, cmd: str) -> str:
    return f'<div role="tabpanel" aria-labelledby="t{i}"{" hidden" if i else ""}><pre><code>{cmd}</code></pre></div>'


_BUTTON_TABS = "".join(
    f'<button role="tab" id="t{i}">{label}</button>' for i, label in enumerate(_COMMANDS)
)


@pytest.mark.parametrize(
    "tabs",
    [
        _LINK_TABS,
        _BUTTON_TABS,
        _LIST_TABS,
        _LIST_TABS.replace("</ul>", "<li>Help</ul>"),
        f"<ul><li hidden>Old tabs<li>{_LINK_TABS}</ul>",  # the next <li> ends the hidden one
        # the next <li> ends the open <p> and the <li> holding it
        "<ul>"
        + "".join(f'<li role="tab" id="t{i}"><p>{label}' for i, label in enumerate(_COMMANDS))
        + "</ul>",
    ],
)
def test_each_panel_is_named_by_its_labelling_tab(tabs):
    # panels in reverse tab order, so only aria-labelledby can pair them
    panels = "".join(
        _labelled_panels(i, cmd) for i, cmd in reversed(list(enumerate(_COMMANDS.values())))
    )
    page = f"<main><h1>Install</h1><p>{_INTRO}</p><div>{tabs}</div>{panels}</main>"
    out = html_to_markdown(page, main_content = True)
    labels = [line for line in out.splitlines() if line.startswith("**")]
    assert labels == [f"**{label}**" for label in reversed(_COMMANDS)]
    for label, command in _COMMANDS.items():
        start = out.index(f"**{label}**") + len(label) + 4
        assert start < out.index(command)
        assert "**" not in out[start : out.index(command)]


def test_panel_label_does_not_break_a_table_row():
    page = (
        f"<main><h1>Install</h1><p>{_INTRO}</p>{_LINK_TABS}<table><tr><th>Tool</th><th>Command</th></tr>"
        f'<tr><td>npm</td><td><div role="tabpanel" aria-labelledby="t0">npm i</div></td></tr></table></main>'
    )
    assert "| npm | npm i |" in html_to_markdown(page, main_content = True)


def test_tabpanel_role_matches_any_token_case():
    page = f"<main><h1>FAQ</h1><p>{_INTRO}</p><div role='TabPanel region' hidden><p>Answer.</p></div></main>"
    assert "Answer." in html_to_markdown(page, main_content = True)


def test_a_panel_label_stays_short_however_many_ids_it_names():
    tabs = "".join(f'<a role="tab" id="t{i}">Tab number {i}</a>' for i in range(50))
    panel = f'<div role="tabpanel" aria-labelledby="{"t0 " * 1000}{" ".join(f"t{i}" for i in range(50))}">x</div>'
    out = html_to_markdown(
        f"<main><h1>Install</h1><p>{_INTRO}</p>{tabs}{panel}</main>", main_content = True
    )
    (label,) = [line for line in out.splitlines() if line.startswith("**")]
    assert label.startswith("**Tab number 0 Tab number 1 ") and len(label) <= 84


def test_a_long_tab_label_is_stored_capped():
    scan = _TabLabelScan()
    scan.feed(f'<a role="tab" id="t">{"x" * 100000}</a>')
    assert scan.labels == {"t": "x" * 80}
