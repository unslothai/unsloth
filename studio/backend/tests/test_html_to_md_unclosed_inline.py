# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An omitted inline end tag must not leave its marker open past the block that held it."""

import pytest

from core.inference._html_to_md import html_to_markdown


@pytest.mark.parametrize("tag, mark", [("strong", "**"), ("b", "**"), ("em", "*"), ("i", "*")])
@pytest.mark.parametrize(
    "before,after",
    [
        ("<p>", "</p><p>Second paragraph.</p>"),
        ("<ul><li>", "</li><li>Second item.</li></ul>"),
        ("<table><tr><td>", "</td><td>Second cell.</td></tr></table>"),
        ("<blockquote>", "</blockquote><p>Second paragraph.</p>"),
        ("<div>", "</div><p>Second paragraph.</p>"),
        ("<p>", "<p>Second paragraph.</p>"),
        ("<ul><li>", "<li>Second item.</li></ul>"),
        ("<table><tr><td>", "<td>Second cell.</td></tr></table>"),
    ],
)
def test_emphasis_closes_with_its_containing_block(tag, mark, before, after):
    opened = before + f"<{tag}>First"
    omitted = opened + after
    closed = opened + f"</{tag}>" + after
    expected = html_to_markdown(closed)
    assert mark in expected
    assert html_to_markdown(omitted) == expected


@pytest.mark.parametrize("tag", ["strong", "em"])
def test_an_explicit_block_end_closes_the_open_mark(tag):
    mark = "**" if tag == "strong" else "*"
    omitted = f"<p><{tag}>First</p><p>Second</p>"
    closed = f"<p><{tag}>First</{tag}></p><p>Second</p>"
    result = html_to_markdown(omitted)
    assert result == html_to_markdown(closed)
    assert result == f"{mark}First{mark}\n\nSecond"


def test_nested_emphasis_close_innermost_first():
    omitted = "<p><strong>One <em>two<p>Second</p>"
    closed = "<p><strong>One <em>two</em></strong><p>Second</p>"
    assert html_to_markdown(omitted) == html_to_markdown(closed)


def test_inline_code_closes_with_its_containing_block():
    omitted = "<p><code>call()<p>Second paragraph.</p>"
    closed = "<p><code>call()</code><p>Second paragraph.</p>"
    assert html_to_markdown(omitted) == html_to_markdown(closed)


def test_an_unclosed_code_span_does_not_stay_open_for_the_rest_of_the_page():
    # _inline_code_depth counts the open <code>; a dropped tag left it set, so every
    # later text node took the code path that preserves whitespace verbatim.
    html = "<p><code>x<p>a   b    c</p><p>d     e</p>"
    result = html_to_markdown(html)
    assert "a b c" in result
    assert "d e" in result
    assert not result.rstrip("\n").endswith("`")


def test_emphasis_inside_hidden_content_emits_no_marker():
    html = "<p><span hidden><strong>secret</span>shown</p>"
    assert html_to_markdown(html) == "shown"


def test_emphasis_inside_a_still_open_link_closes_in_the_link_text():
    omitted = '<p><a href="/h"><strong>label</p><p>after</p>'
    result = html_to_markdown(omitted)
    assert "**label**" in result
    assert "after" in result


def test_emphasis_surrounding_an_unclosed_link_closes_after_the_link():
    result = html_to_markdown('<p><strong><a href="https://example.com">label</p><p>after</p>')
    assert result == "**[label](https://example.com)**\n\nafter"


def test_emphasis_inside_a_link_in_a_heading_closes_in_the_link():
    result = html_to_markdown('<h2><a href="https://example.com"><em>label</a></h2>')
    assert result == "## [*label*](https://example.com)"


def test_late_emphasis_end_does_not_duplicate_recovered_delimiter():
    assert html_to_markdown("<p><em>first<p>second</em>") == "*first*\n\nsecond"


def test_hidden_inline_tag_itself_needs_no_delimiter():
    assert html_to_markdown("<p><em hidden>secret</p><p>shown</p>") == "shown"
