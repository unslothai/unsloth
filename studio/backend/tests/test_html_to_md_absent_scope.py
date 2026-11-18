# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scope pass whose tag never appears is skipped; output must equal running it."""

from __future__ import annotations

import re
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
    "div_only": f"<div><h1>T</h1><p>{_PROSE}</p><p><a href='/next'>next</a></p></div>",
    "text_only": _PROSE,
    "main_only": f"<nav><a href='/a'>Home</a></nav><main><p>{_PROSE}</p></main>",
    "short_main": f"<main><p>tiny</p></main><div><p>{_PROSE}</p></div>",
    "article_only": f"<article><p>{_PROSE}</p></article>",
    "many_articles_no_main": "".join(f"<article><p>card {i}</p></article>" for i in range(4))
    + f"<div><p>{_PROSE}</p></div>",
    "tag_in_comment": f"<!-- <article> <main> --><div><p>{_PROSE}</p></div>",
    "tag_in_script": f"<script>var a = '<article><main>';</script><div><p>{_PROSE}</p></div>",
    "span_table": f"<div><table><tr><td colspan='40'>wide</td><td rowspan='30'>tall</td></tr></table><p>{_PROSE}</p></div>",
    "accordion": "".join(
        f"<h3><button>Question {i}?</button></h3><p>{_PROSE[:220]}</p>" for i in range(3)
    ),
    "empty": "",
}


def _convert(html, **kwargs):
    links = SiteLinks("https://example.com/page")
    h2m._TITLE_BUTTON_CACHE.clear()
    out = html_to_markdown(html, main_content = True, site_links = links, **kwargs)
    return out, links.strip(out)


@pytest.mark.parametrize("name", sorted(PAGES))
@pytest.mark.parametrize("span", [None, 50])
def test_skipping_an_absent_scope_renders_what_running_it_renders(monkeypatch, name, span):
    got = _convert(PAGES[name], max_span_chars = span)
    with monkeypatch.context() as m:
        m.setattr(h2m, "_ARTICLE_START", re.compile(""))
        m.setattr(h2m, "_MAIN_START", re.compile(""))
        expected = _convert(PAGES[name], max_span_chars = span)
    assert got == expected


def _scopes_run(monkeypatch, html):
    seen = []
    real = h2m._select_main_scope_render

    def spy(source_html, tag, *args, **kwargs):
        seen.append(tag)
        return real(source_html, tag, *args, **kwargs)

    monkeypatch.setattr(h2m, "_select_main_scope_render", spy)
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(html, main_content = True)
    return seen


def test_a_page_without_article_or_main_runs_no_scope_pass(monkeypatch):
    assert _scopes_run(monkeypatch, PAGES["div_only"]) == []


def test_a_page_without_article_runs_only_the_main_pass(monkeypatch):
    assert _scopes_run(monkeypatch, PAGES["short_main"]) == ["main"]


def test_a_page_without_main_runs_only_the_article_pass(monkeypatch):
    assert _scopes_run(monkeypatch, PAGES["many_articles_no_main"]) == ["article"]


@pytest.mark.parametrize("name", ["short_main", "accordion", "div_only"])
def test_a_page_without_article_is_tokenized_once(monkeypatch, name):
    # a short main scope leaves the whole-page pass, an accordion page a title rescan: both replay
    feeds = []
    real = h2m._MarkdownRenderer.feed

    def counted(self, data):
        feeds.append(1)
        return real(self, data)

    monkeypatch.setattr(h2m._MarkdownRenderer, "feed", counted)
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(PAGES[name], main_content = True)
    assert len(feeds) == 1


def test_an_article_page_without_main_replays_its_span_scoring_pass(monkeypatch):
    # article pass, then the whole-page pass and its span-free scoring render: the last replays
    html = f"<article><p>tiny</p></article><table><tr><td colspan='30'>c</td></tr></table><p>{_PROSE}</p>"
    feeds = []
    real = h2m._MarkdownRenderer.feed

    def counted(self, data):
        feeds.append(1)
        return real(self, data)

    monkeypatch.setattr(h2m._MarkdownRenderer, "feed", counted)
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(html, main_content = True)
    assert len(feeds) == 2
