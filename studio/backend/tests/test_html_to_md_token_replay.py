# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""html_to_markdown(main_content=True) tokenizes a page once and replays the tokenizer's handler
calls to its later scope / scoring passes. Output must equal feeding every pass the HTML again."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference._html_to_md as h2m
from core.inference._html_to_md import SiteLinks, html_to_markdown

_PROSE = " ".join(
    f"Sentence {i} of the body text, long enough to count as prose." for i in range(12)
)

PAGES = {
    "no_article": f"<html><body><nav><a href='/a'>Home</a></nav><main><h1>T</h1><p>{_PROSE}</p></main></body></html>",
    "one_article": f"<article><h2>Title</h2><p>{_PROSE}</p></article><footer>f</footer>",
    "many_articles": "".join(
        f"<article><h3>Card {i}</h3><p>{_PROSE[:120]}</p></article>" for i in range(5)
    )
    + f"<main><p>{_PROSE}</p></main>",
    "many_articles_no_main": "".join(
        f"<ARTICLE><p>card {i} &amp; more</p></ARTICLE>" for i in range(4)
    )
    + f"<div><p>{_PROSE}</p></div>",
    "accordion_titles": "<main>"
    + "".join(
        f"<h3><button type='button' aria-expanded='false'>Question {i}?</button></h3><div><p>{_PROSE[:200]}</p></div>"
        for i in range(4)
    )
    + "</main>",
    "accordion_multiline": "<article>\n<h3>\n  <button>Q\none</button>\n</h3>\n<p>"
    + _PROSE
    + "</p>\n<h3><button>Q two</button> extra</h3></article>",
    "span_table": "<main><table><tr><td colspan='40'>wide</td><td rowspan='300'>tall</td></tr>"
    "<tr><td>a</td><td>b</td></tr></table>" + f"<p>{_PROSE}</p></main>",
    "charrefs": f"<main><p>&lt;tag&gt; &#65;&#x42; &#0; &#xD800; &#99999999; &nbsp; &notanentity; &amp</p><p>{_PROSE}</p></main>",
    "script_style": f"<main><script>if (a < b) {{ x = '</div>' }}</script><style>p{{}}</style><p>{_PROSE}</p>"
    "<textarea><b>raw</b></textarea><title>T &amp; t</title></main>",
    "comments_cdata": f"<main><!-- c --><!--->x<!-- a -- b --!><![CDATA[ d ]]><!DOCTYPE x><?pi ?><p>{_PROSE}</p></main>",
    "selfclosing": f"<main><p/>text<br/><img src='i.png' alt='A &lt; B'/><div/><a href='/x'/>{_PROSE}</main>",
    "unclosed": f"<article><div><p>{_PROSE}<b>bold<i>both",
    "truncated_tag": f"<main><p>{_PROSE}</p><a href='/x",
    "svg_math": f"<main><svg><title>s</title><foreignObject><p>in</p></foreignObject></svg><math><mi>x</mi></math><p>{_PROSE}</p></main>",
    "crlf": f"<main>\r\n<p>{_PROSE}</p>\r<pre>a\r\nb</pre></main>",
    "div_only": f"<div><h1>T</h1><p>{_PROSE}</p><div><table><tr><td>x</td></tr></table></div></div>",
    "empty": "",
    "text_only": _PROSE,
}


class _NeverReplay:
    """Stands in for _TokenLog: every pass feeds the HTML again, as before the replay."""

    def __init__(self, armed: bool) -> None:
        pass

    events = property(lambda self: None, lambda self, value: None)
    armed = property(lambda self: False, lambda self, value: None)


def _reference(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(h2m, "_TokenLog", _NeverReplay, raising = False)
        h2m._TITLE_BUTTON_CACHE.clear()
        return html_to_markdown(*args, **kwargs)


def _feeds(monkeypatch):
    calls = []
    real = h2m._MarkdownRenderer.feed

    def counted(self, data):
        calls.append(1)
        return real(self, data)

    monkeypatch.setattr(h2m._MarkdownRenderer, "feed", counted)
    return calls


@pytest.mark.parametrize("name", sorted(PAGES))
@pytest.mark.parametrize("span", [None, 50])
def test_replayed_passes_render_what_fed_passes_render(monkeypatch, name, span):
    html = PAGES[name]
    expected = _reference(monkeypatch, html, main_content = True, max_span_chars = span)
    h2m._TITLE_BUTTON_CACHE.clear()
    assert html_to_markdown(html, main_content = True, max_span_chars = span) == expected
    # warm title-button cache path
    assert html_to_markdown(html, main_content = True, max_span_chars = span) == expected


@pytest.mark.parametrize("name", sorted(PAGES))
def test_site_links_record_the_same_links(monkeypatch, name):
    html = PAGES[name]
    ref_links = SiteLinks("https://example.com/page")
    with monkeypatch.context() as m:
        m.setattr(h2m, "_TokenLog", _NeverReplay, raising = False)
        m.setattr(h2m.secrets, "token_hex", lambda n: "0" * (2 * n))
        h2m._TITLE_BUTTON_CACHE.clear()
        expected = html_to_markdown(html, main_content = True, site_links = ref_links)
    with monkeypatch.context() as m:
        m.setattr(h2m.secrets, "token_hex", lambda n: "0" * (2 * n))
        links = SiteLinks("https://example.com/page")
        h2m._TITLE_BUTTON_CACHE.clear()
        got = html_to_markdown(html, main_content = True, site_links = links)
    assert got == expected
    assert links.strip(got) == ref_links.strip(expected)
    assert links._found == ref_links._found


def test_a_multi_pass_page_is_tokenized_once(monkeypatch):
    feeds = _feeds(monkeypatch)
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(PAGES["div_only"], main_content = True)
    assert len(feeds) == 1


def test_a_whole_page_pass_known_to_follow_the_main_scope_replays(monkeypatch):
    # four articles and no <main>: article scope and main scope are fed, the whole-page pass replays
    feeds = _feeds(monkeypatch)
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(PAGES["many_articles_no_main"], main_content = True)
    assert len(feeds) == 2


def test_passes_that_cannot_be_foreseen_are_fed_without_recording(monkeypatch):
    recorded = []
    real = h2m._MarkdownRenderer.feed

    def spy(self, data):
        recorded.append(self._record is not None)
        return real(self, data)

    monkeypatch.setattr(h2m._MarkdownRenderer, "feed", spy)
    for name in ("one_article", "many_articles"):
        h2m._TITLE_BUTTON_CACHE.clear()
        recorded.clear()
        html_to_markdown(PAGES[name], main_content = True)
        assert recorded and not any(recorded), name


def test_plain_conversion_is_untouched(monkeypatch):
    feeds = _feeds(monkeypatch)
    for name in sorted(PAGES):
        h2m._TITLE_BUTTON_CACHE.clear()
        before = len(feeds)
        html_to_markdown(PAGES[name])
        assert len(feeds) - before in (1, 2)  # 2 only for the accordion title rescan


def test_a_renderer_error_surfaces_from_the_first_pass(monkeypatch):
    real = h2m._MarkdownRenderer.handle_data

    def boom(self, data):
        if "Sentence 3" in data:
            raise RuntimeError("renderer failed")
        return real(self, data)

    monkeypatch.setattr(h2m._MarkdownRenderer, "handle_data", boom)
    h2m._TITLE_BUTTON_CACHE.clear()
    with pytest.raises(RuntimeError, match = "renderer failed"):
        html_to_markdown(PAGES["no_article"], main_content = True)
