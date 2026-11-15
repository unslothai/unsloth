# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A nested <a> start tag implicitly closes the open anchor; its buffered text must emit first."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import html_to_markdown


def test_nested_anchor_keeps_outer_link():
    html = '<p><a href="/o">outer <a href="/i">inner</a> tail</a></p>'
    assert html_to_markdown(html) == "[outer](/o)[inner](/i) tail"


def test_nested_anchor_inside_heading():
    html = '<h2><a href="/o">outer <a href="/i">inner</a></a></h2>'
    assert html_to_markdown(html) == "## [outer](/o)[inner](/i)"


def test_nested_anchor_inside_table_cell():
    html = '<table><tr><td><a href="/o">outer <a href="/i">inner</a> tail</a></td></tr></table>'
    out = html_to_markdown(html)
    assert "[outer](/o)[inner](/i) tail" in out
    assert out.startswith("| ") and out.rstrip().endswith("| --- |")


def test_nested_anchor_outer_never_closed():
    html = '<p><a href="/o">outer <a href="/i">inner</a> tail'
    assert html_to_markdown(html) == "[outer](/o)[inner](/i) tail"


def test_nested_anchor_outer_without_href_keeps_text():
    html = '<p><a>plain <a href="/i">inner</a></a></p>'
    assert html_to_markdown(html) == "plain[inner](/i)"


def test_nested_anchor_empty_outer_emits_nothing():
    html = '<p><a href="/o"><a href="/i">inner</a></a></p>'
    assert html_to_markdown(html) == "[inner](/i)"


def test_hidden_nested_anchor_still_closes_outer():
    nested = '<p><a href="/o">outer <a hidden href="/i">inner</a> tail</p>'
    closed = '<p><a href="/o">outer </a><a hidden href="/i">inner</a> tail</p>'
    assert html_to_markdown(nested) == html_to_markdown(closed) == "[outer](/o) tail"


def test_nested_anchor_under_hidden_ancestor_closes_outer():
    html = '<p><a href="/o">outer <span hidden><a href="/i">inner</a></span> tail</p>'
    assert html_to_markdown(html) == "[outer](/o) tail"


def test_sibling_anchors_unchanged():
    html = '<p><a href="/o">outer</a> mid <a href="/i">inner</a></p>'
    assert html_to_markdown(html) == "[outer](/o) mid [inner](/i)"


def test_nested_anchor_inside_header_scope():
    html = '<main><header><a href="/o">nav <a href="/i">inner</a></header><article><p>body text</p></article></main>'
    assert html_to_markdown(html, main_content = True) == "[nav](/o)[inner](/i)\n\nbody text"


def test_nested_anchor_inside_pre_keeps_code_literal():
    html = '<pre><a href="/o">x<b>y</b><a href="/i">z</a></pre>'
    assert html_to_markdown(html) == "```\nxyz\n```"


def test_outer_text_before_a_header_survives_the_header_strip():
    nav = "navigation " * 30
    html = f'<a href="/o">Intro. <header>{nav}<a hidden href="/i">skip</a></header>Conclusion.</a>'
    assert html_to_markdown(html, main_content = True) == "[Intro. Conclusion.](/o)"


def test_outer_text_before_a_header_blockquote_survives_the_header_strip():
    nav = "navigation " * 30
    html = (
        f'<a href="/o">Intro. <header><blockquote>{nav}<a hidden href="/i">skip</a>'
        "</blockquote></header>Conclusion.</a>"
    )
    assert html_to_markdown(html, main_content = True) == "[Intro. Conclusion.](/o)"


def test_byline_in_a_header_opened_inside_the_link_survives():
    nav = "OuterNav " * 35
    byline = "Jane wrote this article. " * 8
    inner = f'<header>InnerNav <a href="/i">Inner</a><p>{byline}</p></header>'
    html = f'<main><header><a href="/o">{nav}{inner}</a></header><p>BODY</p></main>'
    assert "Jane wrote this article." in html_to_markdown(html, main_content = True)


def test_nested_anchor_in_long_header_still_strips_as_furniture():
    nav = "x" * 160
    html = f'<main><header><a href="/o">{nav} <a href="/i">inner</a></header><article><p>body text</p></article></main>'
    assert html_to_markdown(html, main_content = True) == "body text"


def test_heading_inside_the_outer_link_is_not_teed_twice():
    nav = "Navigation link " * 12
    html = f"<header><a href='/o'>{nav}<h1>Title<a href='/i'>Inner</a></h1></a></header>"
    out = html_to_markdown(html, main_content = True)
    assert out.count("Title") <= 1 and "Navigation" not in out


def test_title_only_card_does_not_win_main_content():
    title = "Related title words " * 16
    card = f"<article><a href='/o'>Card <div role='heading'>{title}<a href='/i'>Read</a></div></a></article>"
    body = "<article><p>" + "Real article body text. " * 10 + "</p></article>"
    assert "Real article body text." in html_to_markdown(card + body, main_content = True)


def test_closed_heading_card_does_not_win_main_content():
    title = "Related title words " * 16
    card = f"<article><a href='/o'>Card <h2>{title}</h2><a href='/i'>Read</a></a></article>"
    body = "<article><p>" + "Real article body text. " * 10 + "</p></article>"
    assert "Real article body text." in html_to_markdown(card + body, main_content = True)


def test_nested_anchor_inside_inline_code_keeps_the_inner_link():
    html = "<p><a href='/o'><code>git <a href='/i'>clone</a></code></a></p>"
    assert html_to_markdown(html).startswith("[clone](/i)")


def test_outer_link_text_stays_out_of_a_table_opened_inside_it():
    html = '<a href="/o">Intro<table><caption><a href="/i"></a></caption><tr><td>DATA</td></tr></table>After</a>'
    assert html_to_markdown(html).startswith("| DATA |")


def test_outer_link_text_does_not_move_into_a_sibling_quote():
    html = '<blockquote><a href="/o">FIRST</blockquote><blockquote><a href="/i">SECOND</a></blockquote>'
    assert "FIRST" not in html_to_markdown(html)


def test_role_heading_anchor_closing_the_outer_link_keeps_its_title():
    nav = "".join(f'<a href="/n{i}">Nav item {i}</a> ' for i in range(20))
    html = f'<header>{nav}<a href="/o"><a role="heading" href="/i">Page Title</header><p>tail</p>'
    assert "Page Title" in html_to_markdown(html, main_content = True)


def test_hidden_role_heading_anchor_does_not_mark_the_next_link_as_heading():
    first = "First article body sentence here. " * 18
    second = "Second article text. " * 12
    html = (
        f'<article><a href="/o"><a hidden role="heading">Hidden</a><a href="/body">{first}</a></article>'
        f"<article><p>{second}</p></article>"
    )
    assert "First article body" in html_to_markdown(html, main_content = True)
