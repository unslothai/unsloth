# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import SiteLinks, html_to_markdown
from core.inference.tools import _render_page

_POST = " ".join(f"Reply sentence {i} about loading part of a checkpoint." for i in range(8))
_SECOND_POST = " ".join(f"Follow-up sentence {i} with the working answer." for i in range(8))
_CARD = "Link preview card text that is long enough to pass for an article on its own. " * 4
_SHELL = "<html><head><title>Topic</title></head><body><div id='app'></div>{}</body></html>"
_TRACKING = (
    '<noscript><img height="1" width="1" src="https://t.example/tr?ev=PageView"></noscript>'
    "<noscript>You need to enable JavaScript to run this app.</noscript>"
    '<noscript><iframe src="https://tags.example/ns.html" height="0" width="0"></iframe></noscript>'
)
_FORUM_TOPIC = _SHELL.format(
    '<noscript data-path="/t/topic/1">'
    "<header><a href='/'>Forum</a></header>"
    "<div id='main-outlet' role='main'><h1>How to load part of a model</h1>"
    f"<div class='post'><p>{_POST}</p>"
    f"<aside class='onebox'><article class='onebox-body'><p>{_CARD}</p></article></aside></div>"
    f"<div class='post'><p>{_SECOND_POST}</p></div></div>"
    "</noscript>"
)


@pytest.mark.parametrize("main_content", [True, False])
def test_page_whose_text_sits_in_noscript_is_read(main_content):
    out = html_to_markdown(_FORUM_TOPIC, main_content = main_content)
    assert "How to load part of a model" in out
    assert _POST in out
    assert _SECOND_POST in out


def test_fetched_noscript_page_reaches_the_model_as_text():
    out = _render_page(("html", "https://forum.example/t/topic/1", _FORUM_TOPIC), 20000)
    assert _POST in out
    assert "no readable text" not in out


def test_same_site_links_in_noscript_are_kept_and_strippable():
    links = SiteLinks("https://forum.example/t/topic/1")
    page = _FORUM_TOPIC.replace("</h1>", "</h1><p><a href='/t/topic/2'>next topic</a></p>")
    out = html_to_markdown(page, main_content = True, site_links = links)
    assert "[next topic](/t/topic/2)" in out
    assert "\x00" not in out
    assert "(/t/topic/2)" not in links.strip(out)


@pytest.mark.parametrize("main_content", [True, False])
def test_noscript_is_ignored_when_the_page_has_text(main_content):
    body = f"<main><h1>Guide</h1><p>{_POST}</p></main>"
    plain = f"<html><body>{body}</body></html>"
    with_noscript = (
        f"<html><body>{body}{_TRACKING}<noscript><p>{_SECOND_POST}</p></noscript></body></html>"
    )
    assert html_to_markdown(with_noscript, main_content = main_content) == html_to_markdown(
        plain, main_content = main_content
    )


def test_enable_javascript_notice_alone_is_not_readable_text():
    page = _SHELL.format(_TRACKING)
    assert html_to_markdown(page, main_content = True) == ""
    assert _render_page(("html", "https://app.example/", page), 20000) == (
        "(page returned no readable text)"
    )


@pytest.mark.parametrize("attr", ["hidden", 'aria-hidden="true"', 'style="display:none"'])
def test_hidden_noscript_stays_hidden(attr):
    page = _SHELL.format(f"<noscript {attr}><p>{_POST}</p></noscript>")
    assert html_to_markdown(page, main_content = True) == ""


def test_unterminated_noscript_tags_convert_in_linear_time():
    page = "<head>" + "<noscript " * 40000
    start = time.perf_counter()
    assert html_to_markdown(page, main_content = True) == ""
    assert time.perf_counter() - start < 3


@pytest.mark.parametrize("extra", [_TRACKING, ""])
def test_header_only_shell_with_noscript_stays_empty(extra):
    nav = "".join(f"<a href='/s{i}'>Documentation section {i}</a> " for i in range(15))
    page = _SHELL.format(f"<header>{nav}</header>{extra}")
    assert html_to_markdown(page, main_content = True) == ""
