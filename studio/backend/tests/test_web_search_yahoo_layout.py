# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Yahoo's section layout, which ddgs parses to nothing, must reach the model as results."""

from unittest.mock import MagicMock

import pytest

from ddgs.base import BaseSearchEngine
from ddgs.engines import ENGINES
from ddgs.engines.yahoo import Yahoo

from core.inference import tools


def _redirect(url):
    return (
        "https://r.search.yahoo.com/_ylt=Awr;_ylu=Y29s/RV=2/RE=1792609327/RO=10/RU="
        + url.replace(":", "%3a").replace("/", "%2f")
        + "/RK=2/RS=abc-"
    )


def _section(
    kind,
    url,
    title,
    desc = None,
):
    # Trimmed from a live page: grp-talgo-ext is never closed, so lxml nests every later section here.
    text = (
        f'<div class="grp grp-talgo-ext"><div class="compText"><p class="s-desc lh-20">'
        f'<span class="s-time">Jul 23, 2025 · </span>{desc}</p></div>'
        if desc
        else ""
    )
    return (
        f'<section class="dd algo s-algo {kind}"><div class="grp grp-talgo"><div class="compTitle p-r">'
        f'<h3 class="title d-ib mt-42"><a class="s-title fz-m" href="{_redirect(url)}" aria-label="{title}">'
        f'<span class="title-url"><span class="fw-500">Site</span>{url}</span>{title}</a></h3>'
        f'<div><a class="thmb algo-favicon" href="{_redirect(url)}"><img src="x.png"/></a></div>'
        f"</div></div>{text}</section>"
    )


SECTION_LAYOUT = (
    '<html><body><section class="reg searchCenterMiddle">'
    + _section(
        "algo-sr richAlgo",
        "https://docs.python.org/3/library/asyncio.html",
        "asyncio: Asynchronous I/O",
        "asyncio is a library to write <b>concurrent</b> code.",
    )
    + _section(
        "imageThumbnail image small Sr",
        "https://www.datacamp.com/tutorial/python-async",
        "Python Async Programming - DataCamp",
        "Speed up your code with async.",
    )
    # A titled block that is not a result: it must not become one.
    + '<section class="dd AdTop"><div><h3><a class="s-title" href="https://ads.example/x">'
    + "Sponsored</a></h3></div></section>"
    + _section(
        "videoThumbnail richAlgo", "https://m.youtube.com/watch?v=Qb9s3UiMSTA", "Asyncio in Python"
    )
    + _section(
        "algo-sr Sr",
        "https://realpython.com/async-io-python/",
        "Python's asyncio: A Hands-On Walkthrough",
        "In this tutorial, you'll learn how asyncio works.",
    )
    + "</section></body></html>"
)

RELSRCH_LAYOUT = (
    '<html><body><ol><li><div class="dd algo algo-sr relsrch Sr"><div class="compTitle">'
    f'<div class="Title"><h3>GeeksforGeeks asyncio</h3><a href="{_redirect("https://www.geeksforgeeks.org/asyncio/")}">x</a></div>'
    '</div><div class="compText"><div class="Text">Older layout body.</div></div></div></li></ol></body></html>'
)

NO_MATCH = (
    '<html><body><section class="reg searchCenterMiddle"><p>We did not find results for: zzqxv. '
    "Check spelling or type a new query.</p></section></body></html>"
)


@pytest.fixture
def yahoo():
    text_engines = dict(ENGINES["text"])
    tools._install_yahoo_layout_parser(text_engines)
    return text_engines["yahoo"].__new__(text_engines["yahoo"])


def _parse(engine, html_text):
    return [
        (r.title, r.href, r.body)
        for r in engine.post_extract_results(engine.extract_results(html_text))
    ]


def test_ddgs_alone_finds_nothing_in_the_section_layout():
    assert Yahoo.__new__(Yahoo).extract_results(SECTION_LAYOUT) == []


def test_every_organic_section_becomes_a_result_with_its_own_snippet(yahoo):
    assert _parse(yahoo, SECTION_LAYOUT) == [
        (
            "asyncio: Asynchronous I/O",
            "https://docs.python.org/3/library/asyncio.html",
            "Jul 23, 2025 · asyncio is a library to write concurrent code.",
        ),
        (
            "Python Async Programming - DataCamp",
            "https://www.datacamp.com/tutorial/python-async",
            "Jul 23, 2025 · Speed up your code with async.",
        ),
        # No snippet on the page: the next result's must not be borrowed.
        ("Asyncio in Python", "https://m.youtube.com/watch?v=Qb9s3UiMSTA", ""),
        (
            "Python's asyncio: A Hands-On Walkthrough",
            "https://realpython.com/async-io-python/",
            "Jul 23, 2025 · In this tutorial, you'll learn how asyncio works.",
        ),
    ]


def test_the_older_layout_still_goes_through_ddgs(yahoo):
    assert _parse(yahoo, RELSRCH_LAYOUT) == [
        ("GeeksforGeeks asyncio", "https://www.geeksforgeeks.org/asyncio/", "Older layout body.")
    ]


def test_an_honest_no_match_page_stays_empty(yahoo):
    assert _parse(yahoo, NO_MATCH) == []


def test_install_is_idempotent_and_ignores_registry_stubs():
    text_engines = dict(ENGINES["text"])
    tools._install_yahoo_layout_parser(text_engines)
    installed = text_engines["yahoo"]
    tools._install_yahoo_layout_parser(text_engines)
    assert text_engines["yahoo"] is installed

    stub = object()
    stubbed = {"yahoo": stub}
    tools._install_yahoo_layout_parser(stubbed)
    assert stubbed["yahoo"] is stub
    tools._install_yahoo_layout_parser(None)


def _offline_init(self, *args, **kwargs):
    self.http_client = MagicMock()


def test_web_search_returns_yahoo_section_results_when_every_other_engine_is_blocked(monkeypatch):
    """The reported failure: only Yahoo answers, in the layout ddgs cannot read."""
    from ddgs.ddgs import DDGS

    if callable(getattr(DDGS, "_get_network_client", None)):
        monkeypatch.setattr(DDGS, "_get_network_client", lambda self: None)
    monkeypatch.setattr(tools, "_wikipedia_search", lambda *args: [])
    for cls in ENGINES["text"].values():
        # A real client cannot be built on the py<3.10 pin (see test_web_search_tiers); Google's
        # build_payload still sets a cookie on it.
        monkeypatch.setattr(cls, "__init__", _offline_init, raising = False)
    # _web_search swaps its Yahoo subclass into the shared registry; put the entry back afterwards.
    monkeypatch.setitem(ENGINES["text"], "yahoo", ENGINES["text"]["yahoo"])
    contacted = []

    def request(engine, *args, **kwargs):
        contacted.append(engine.name)
        return SECTION_LAYOUT if engine.name == "yahoo" else None  # None is how ddgs reports a 429

    monkeypatch.setattr(BaseSearchEngine, "request", request)

    result = tools.execute_tool("web_search", {"query": "python asyncio tutorial"})

    assert "yahoo" in contacted
    assert "URL: https://docs.python.org/3/library/asyncio.html" in result
    assert "URL: https://realpython.com/async-io-python/" in result
    assert "Wikipedia-only" not in result
