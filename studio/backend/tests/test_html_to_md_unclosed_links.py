# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An omitted </a> must not move a block's link text into a later output buffer."""

import pytest

from core.inference._html_to_md import SiteLinks, html_to_markdown


@pytest.mark.parametrize("main_content", [False, True])
@pytest.mark.parametrize(
    "before,after",
    [
        ("<p>", "</p><p>Second paragraph.</p>"),
        ("<ul><li>", "</li><li>Second item.</li></ul>"),
        ("<ol><li>", "</li><li>Second item.</li></ol>"),
        ("<table><tr><td>", "</td><td>Second cell.</td></tr></table>"),
        ("<table><tr><th>", "</th><th>Second heading.</th></tr></table>"),
        ("<blockquote>", "</blockquote><p>Second paragraph.</p>"),
        ("<div>", "</div><p>Second paragraph.</p>"),
        ("<p>", "<p>Second paragraph.</p>"),
        ("<ul><li>", "<li>Second item.</li></ul>"),
        ("<table><tr><td>", "<td>Second cell.</td></tr></table>"),
    ],
)
def test_link_closes_with_its_containing_block(before, after, main_content):
    opened = before + '<a href="/guide">First link'
    omitted = "<main>" + opened + after + "</main>"
    closed = "<main>" + opened + "</a>" + after + "</main>"
    expected = html_to_markdown(closed, main_content = main_content)
    assert "[First link](/guide)" in expected
    assert html_to_markdown(omitted, main_content = main_content) == expected


@pytest.mark.parametrize(
    "body",
    [
        "First <strong>bold</strong> link",
        "First <span hidden>hidden text</span>link",
        "First <code>code</code> link",
    ],
)
def test_link_keeps_its_inline_content_and_ignores_a_late_end_tag(body):
    omitted = f'<p><a href="/guide">{body}</p><p>Second paragraph.</a></p>'
    closed = f'<p><a href="/guide">{body}</a></p><p>Second paragraph.</p>'
    assert html_to_markdown(omitted) == html_to_markdown(closed)


def test_link_text_stays_in_its_original_quote():
    omitted = '<blockquote><a href="/first">FIRST</blockquote><blockquote><a href="/second">SECOND</a></blockquote>'
    closed = omitted.replace("FIRST</blockquote>", "FIRST</a></blockquote>")
    result = html_to_markdown(omitted)
    assert result == html_to_markdown(closed)
    assert "[FIRST](/first)" in result
    assert "[SECOND](/second)" in result


@pytest.mark.parametrize("main_content", [False, True])
def test_an_anchor_that_contains_blocks_still_spans_them(main_content):
    html = '<main><a href="/card"><div>Card title</div><p>Card description.</p></a><p>Body text.</p></main>'
    result = html_to_markdown(html, main_content = main_content)
    assert "[Card title Card description.](/card)" in result
    assert "Body text." in result


def test_same_site_link_collection_keeps_the_original_label():
    links = SiteLinks("https://example.com/page")
    result = html_to_markdown(
        '<p><a href="/guide">First link</p><p>Second paragraph.</p>', site_links = links
    )
    assert "[First link]" in result
    assert "[First link Second paragraph.]" not in result


@pytest.mark.parametrize("main_content", [False, True])
@pytest.mark.parametrize(
    "html,expected",
    [
        (
            '<p><a href="/card"><div>Card title</div>Card description.</a></p>',
            "[Card title Card description.](/card)",
        ),
        ('<p><a href="/post"><h3>Post title</h3></a></p>', "[### Post title](/post)"),
        (
            '<p><a href="/x"><figure><img src="a.png"><figcaption>Caption</figcaption></figure></a></p>',
            "[Caption](/x)",
        ),
        ('<p><a href="/o">text<div>block</div></a></p>', "[text block](/o)"),
    ],
)
def test_a_block_opened_inside_a_paragraph_link_stays_linked(html, expected, main_content):
    assert html_to_markdown(html, main_content = main_content) == expected


@pytest.mark.parametrize("main_content", [False, True])
@pytest.mark.parametrize(
    "html,expected",
    [
        (
            '<main><a href="/x"><header>Card title</a><p>Body text.</p></main>',
            "[Card title](/x)\n\nBody text.",
        ),
        (
            '<ul><li><a href="/post">Card link<header>Tag</li><li>Two</li></ul>',
            "* [Card linkTag](/post)\n* Two",
        ),
        ('<header><a href="/x">A<header>B</a></header></header>', "[AB](/x)"),
    ],
)
def test_a_header_left_open_in_a_link_keeps_the_link(html, expected, main_content):
    assert html_to_markdown(html, main_content = main_content) == expected
