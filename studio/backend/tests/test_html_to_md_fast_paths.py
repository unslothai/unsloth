# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fast paths in html_to_markdown's renderer and title-button scan (shared attribute checks, guarded
helper calls, C-speed open-tag lookups) must render exactly what the plain per-token code did.
Expected strings were produced by the converter before those fast paths."""

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
    f"Sentence {i} of the body text, long enough to count as prose." for i in range(6)
)

# name: (html, main_content Markdown with site links, its stripped form, whole-page Markdown)
CASES = {
    "aria_heading_title_button": (
        "<main><div role='presentation Heading'><button>Is it fast?</button></div><p>"
        + _PROSE
        + "</p><h3 aria-hidden=' TRUE '><button>Hidden</button></h3><h4><button>Kept title</button></h4><p>"
        + _PROSE
        + "</p></main>",
        "Is it fast?\n\n" + _PROSE + "\n\n#### Kept title\n\n" + _PROSE + "",
        "Is it fast?\n\n" + _PROSE + "\n\n#### Kept title\n\n" + _PROSE + "",
        "Is it fast?\n\n" + _PROSE + "\n\n#### Kept title\n\n" + _PROSE + "",
    ),
    "hidden_variants": (
        "<main><p hidden=false>gone</p><p style='color:red'>shown red</p><p style='display : none'>gone too</p><p aria-hidden='false'>shown</p><img hidden src=x><p>"
        + _PROSE
        + "</p></main>",
        "shown red\n\nshown\n\n" + _PROSE + "",
        "shown red\n\nshown\n\n" + _PROSE + "",
        "shown red\n\nshown\n\n" + _PROSE + "",
    ),
    "implicit_closers": (
        "<main><p>one<div>two</div><ul><li>a<li>b<ul><li>c</ul><li>d</ul><table><tr><td>x<td>y<tr><th>z</table><dl><dt>t<dd>u<dt>v</dl><p>"
        + _PROSE
        + "</main>",
        "one\n\ntwo\n\n* a\n* b\n\n  * c\n\n* d\n\n| x | y |\n| --- | --- |\n| z |\n\nt\n\nu\n\nv\n\n"
        + _PROSE
        + "",
        "one\n\ntwo\n\n* a\n* b\n\n  * c\n\n* d\n\n| x | y |\n| --- | --- |\n| z |\n\nt\n\nu\n\nv\n\n"
        + _PROSE
        + "",
        "one\n\ntwo\n\n* a\n* b\n\n  * c\n\n* d\n\n| x | y |\n| --- | --- |\n| z |\n\nt\n\nu\n\nv\n\n"
        + _PROSE
        + "",
    ),
    "header_frames": (
        "<main><header><a href='/home'>Home</a> <a href='/about'>About</a><h1>Big <a href='/t'>title</a></h1></header><p>"
        + _PROSE
        + " <a href='/more'>more</a></p><header><p>Byline by someone with a longer line of text</p></header></main>",
        "[Home](/home) [About](/about)\n\n# Big [title](/t)\n\n"
        + _PROSE
        + " [more](/more)\n\nByline by someone with a longer line of text",
        "Home About\n\n# Big title\n\n"
        + _PROSE
        + " more\n\nByline by someone with a longer line of text",
        "[Home](/home) [About](/about)\n\n# Big [title](/t)\n\n"
        + _PROSE
        + " [more](/more)\n\nByline by someone with a longer line of text",
    ),
    "foreign_content": (
        "<main><h2><svg><title>icon</title><foreignObject><button>in svg</button></foreignObject></svg>Text</h2><math><mi>x</mi><annotation-xml encoding='text/html'><p>html in math</p></annotation-xml></math><svg><p>breakout</p></svg><p>"
        + _PROSE
        + "</p></main>",
        "## Text\n\nbreakout\n\n" + _PROSE + "",
        "## Text\n\nbreakout\n\n" + _PROSE + "",
        "## Text\n\nbreakout\n\n" + _PROSE + "",
    ),
    "skipped_subtrees": (
        "<main><nav><h3><button>Nav q</button></h3></nav><h3><button>Real q</button></h3><div><p>"
        + _PROSE
        + "</p></div><footer><p>foot</p></footer><script>var a='</h3>';</script></main>",
        "### Real q\n\n" + _PROSE + "",
        "### Real q\n\n" + _PROSE + "",
        "### Real q\n\n" + _PROSE + "",
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_output_matches_pre_fast_path_render(name):
    html, main_md, stripped, full_md = CASES[name]
    h2m._TITLE_BUTTON_CACHE.clear()
    site_links = SiteLinks("https://example.com/page")
    rendered = html_to_markdown(html, main_content = True, site_links = site_links)
    assert rendered == main_md
    assert site_links.strip(rendered) == stripped
    assert html_to_markdown(html) == full_md


def test_title_buttons_found_through_shared_attribute_checks():
    html = CASES["aria_heading_title_button"][0]
    h2m._TITLE_BUTTON_CACHE.clear()
    html_to_markdown(html, main_content = True)
    (keep,) = h2m._TITLE_BUTTON_CACHE.values()
    # role="presentation Heading" and the plain h4 keep their buttons; the aria-hidden h3 does not
    assert len(keep) == 2


def _old_is_hidden(attr_dict):
    if "hidden" in attr_dict:
        return True
    if (attr_dict.get("aria-hidden") or "").strip().lower() == "true":
        return True
    return h2m._style_hides_element(attr_dict.get("style") or "")


def _old_is_aria_heading(attr_dict):
    return "heading" in (attr_dict.get("role") or "").lower().split()


_ATTR_VALUES = [
    None,
    "",
    " ",
    "true",
    " TRUE ",
    "false",
    "heading",
    "Heading tab",
    "display:none",
    "visibility: hidden !important",
    "color: red",
    "none",
]


@pytest.mark.parametrize("name", ["hidden", "aria-hidden", "style", "role", "class"])
def test_attribute_checks_match_previous_formulas(name):
    for value in _ATTR_VALUES:
        for attr_dict in ({}, {name: value}, {name: value, "style": "display:none"}):
            assert h2m._is_hidden_element(attr_dict) is _old_is_hidden(attr_dict)
            assert h2m._is_aria_heading(attr_dict) is _old_is_aria_heading(attr_dict)


def test_any_implicit_closer_is_the_union_of_closer_sets():
    union = set()
    for closers in h2m._IMPLICIT_CLOSERS.values():
        union |= closers
    assert h2m._ANY_IMPLICIT_CLOSER == union
    # inline elements never close anything, so they skip the recovery scan
    assert not {"span", "a", "img", "b", "svg"} & h2m._ANY_IMPLICIT_CLOSER
