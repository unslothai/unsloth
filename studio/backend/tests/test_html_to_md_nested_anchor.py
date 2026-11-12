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


def test_nested_anchor_inside_pre_matches_a_real_close():
    nested = '<pre><a href="/o">x<b>y</b><a href="/i">z</a></pre>'
    closed = '<pre><a href="/o">x<b>y</b></a><a href="/i">z</a></pre>'
    assert html_to_markdown(nested) == html_to_markdown(closed)
