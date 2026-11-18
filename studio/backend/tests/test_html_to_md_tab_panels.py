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


@pytest.mark.parametrize(
    "hidden", [" hidden", ' hidden=""', ' hidden="true"', ' hidden="" style="display:none"']
)
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
        "<div role='tabpanel' hidden aria-hidden='true' style='display:none'><p>Old answer.</p></div>",
        "<div style='display:none'><div role='tabpanel' hidden style='display:none'><p>Old answer.</p></div></div>",
        "<div hidden='until-found' style='display:none'><p>Old answer.</p></div>",
        "<div role='menu' hidden style='display:none'><p>Old answer.</p></div>",
        "<div role='dialog' hidden style='display:none'><p>Old answer.</p></div>",
        "<ul role='listbox' hidden style='display:none'><li>Old answer.</li></ul>",
        "<span hidden style='display:none'>Old answer.</span>",
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


def _headless_ui_tabs(prefix: str, panels: dict[str, str]) -> str:
    # Headless UI's server render: button tabs, panels with ids but no aria-labelledby, the inactive ones hidden
    tabs = "".join(
        f'<button role="tab" id="{prefix}t{i}">{label}</button>' for i, label in enumerate(panels)
    )
    bodies = "".join(
        f'<div role="tabpanel" id="{prefix}p{i}"{" hidden style=display:none" if i else ""}>{body}</div>'
        for i, body in enumerate(panels.values())
    )
    return f'<div><div role="tablist">{tabs}</div><div>{bodies}</div></div>'


def test_headless_ui_panels_are_named_by_their_tab_in_order():
    commands = {label: f"<pre><code>{cmd}</code></pre>" for label, cmd in _COMMANDS.items()}
    frameworks = {
        name: _headless_ui_tabs(
            f"{name}-", {label: body.replace("jest", name) for label, body in commands.items()}
        )
        for name in ("Next", "Svelte")
    }
    page = (
        f"<main><h1>Install</h1><p>{_INTRO}</p>{_headless_ui_tabs('f-', frameworks)}"
        # a closed group's leftover tab must not name a later panel at the same depth
        "<div><div role='tablist'><button role='tab' id='g'>Ghost</button></div></div>"
        "<div><div role='tabpanel' id='stray'>Stray panel.</div></div></main>"
    )
    out = html_to_markdown(page, main_content = True)
    labels = [line for line in out.splitlines() if line.startswith("**")]
    assert labels == [f"**{label}**" for name in frameworks for label in (name, *_COMMANDS)]
    for name in frameworks:
        for label, command in _COMMANDS.items():
            named = out.index(f"**{name}**")
            assert out.index(f"**{label}**", named) < out.index(command.replace("jest", name))
    assert "Stray panel." in out


@pytest.mark.parametrize(
    "body, labels",
    [
        # aria-controls outranks order, so a missing first panel shifts nothing
        (
            '<div><div role="tablist"><button role="tab" id="a" aria-controls="p0">npm</button>'
            '<button role="tab" id="b" aria-controls="p1">pnpm</button></div>'
            '<div><div role="tabpanel" id="p1">pnpm add x</div></div></div>',
            ["**pnpm**"],
        ),
        # an aria-controls later in the page still outranks order
        (
            '<div><div role="tablist"><button role="tab" id="a">npm</button>'
            '<button role="tab" id="b">yarn</button></div>'
            '<div><div role="tabpanel" id="p0">npm i x</div><div role="tabpanel" id="p1">pnpm add x</div>'
            '</div></div><div role="tablist"><button role="tab" id="c" aria-controls="p1">pnpm</button></div>',
            ["**npm**", "**pnpm**"],
        ),
        # only the element right after the tablist holds its panels, a void one included
        (
            '<div role="tablist"><button role="tab" id="a">npm</button></div><hr>'
            '<div><div role="tabpanel" id="q">pip i</div></div>',
            [],
        ),
        # order pairs only when every tab has its panel, so a missing one names nothing
        (
            '<div><div role="tablist"><button role="tab" id="a">npm</button>'
            '<button role="tab" id="b">pnpm</button></div>'
            '<div><div role="tabpanel" id="p1">pnpm add x</div></div></div>',
            [],
        ),
        (
            '<div><div role="tablist"><button role="tab" id="a">npm</button>'
            '<button role="tab" id="b">pnpm</button></div>'
            '<div><div role="tabpanel">npm i x</div><div role="tabpanel" id="p1">pnpm add x</div></div></div>',
            ["**pnpm**"],
        ),
        # one panel several tabs control takes the selected tab's name, and none without one
        (
            '<div role="tablist"><button role="tab" id="a" aria-controls="p">npm</button>'
            '<button role="tab" id="b" aria-controls="p" aria-selected="true">pnpm</button>'
            '<button role="tab" id="c" aria-controls="p">yarn</button></div>'
            '<div role="tabpanel" id="p">pnpm add x</div>',
            ["**pnpm**"],
        ),
        (
            '<div role="tablist"><button role="tab" id="a" aria-controls="p">npm</button>'
            '<button role="tab" id="b" aria-controls="p">pnpm</button></div>'
            '<div role="tabpanel" id="p">pnpm add x</div>',
            [],
        ),
        # a tablist with any aria-controls is never paired by order, which could contradict it
        (
            '<div><div role="tablist"><button role="tab" id="a" aria-controls="p1">npm</button>'
            '<button role="tab" id="b">pnpm</button></div>'
            '<div><div role="tabpanel" id="p0">pnpm add x</div><div role="tabpanel" id="p1">npm i x</div>'
            "</div></div>",
            ["**npm**"],
        ),
        # an id-less tab still takes its slot, so the one panel of two tabs is not guessed
        (
            '<div><div role="tablist"><button role="tab">npm</button>'
            '<button role="tab" id="b">pnpm</button></div>'
            '<div><div role="tabpanel" id="p0">npm i x</div></div></div>',
            [],
        ),
        # a panel naming its tab by aria-labelledby never takes an order guess, even when that tab is unread
        (
            '<div><div role="tablist"><button role="tab" id="a">npm</button>'
            '<button role="tab" id="b" hidden>pnpm</button></div>'
            '<div><div role="tabpanel" id="p0" aria-labelledby="b">pnpm add x</div>'
            '<div role="tabpanel" id="p1" aria-labelledby="a">npm i x</div></div></div>',
            ["**npm**"],
        ),
        # a panel id reused by a later tab leaves that tab's own label intact
        (
            '<div><div role="tablist"><button role="tab" id="t0">npm</button></div>'
            '<div><div role="tabpanel" id="same">npm i</div></div></div>'
            '<a role="tab" id="same">Python</a><div role="tabpanel" aria-labelledby="same">pip i</div>',
            ["**npm**", "**Python**"],
        ),
    ],
)
def test_explicit_links_win_over_order(body, labels):
    out = html_to_markdown(f"<main><h1>Install</h1><p>{_INTRO}</p>{body}</main>", main_content = True)
    assert [line for line in out.splitlines() if line.startswith("**")] == labels


def test_a_page_cut_off_inside_its_panels_still_names_them():
    page = (
        f"<main><h1>Install</h1><p>{_INTRO}</p>"
        '<div><div role="tablist"><button role="tab" id="a">npm</button></div>'
        '<div><div role="tabpanel" id="p0"><pre><code>npm i x'
    )
    assert "**npm**" in html_to_markdown(page, main_content = True)


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
