# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long-chat gate's source hints: selector text -> CSS rule, and the :has() shape check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _css_culprit as culprit  # noqa: E402

# Chromium's serialization of the #16603 rule, as its selector statistics report it.
REPORTED = (
    ":is([data-wallpaper-workspace], [data-chat-content-root]):has(> .chat-wallpaper[data-ready] "
    '.chat-wallpaper-image[data-fade="false"]) .thread-bottom-fade'
)
SOURCE = """\
/* a comment { with braces } */
.unrelated { color: red; }

:is([data-wallpaper-workspace],[data-chat-content-root]):has(>.chat-wallpaper[data-ready] .chat-wallpaper-image[data-fade=false]) .thread-bottom-fade,
.other { display: none; }
@media (min-width: 1px) {
  .aui-thread-viewport:has(.aui-thread-welcome-center) { gap: 0; }
}
"""


def _use(tmp_path, monkeypatch):
    src = tmp_path / "studio" / "frontend" / "src"
    src.mkdir(parents = True)
    (src / "index.css").write_text(SOURCE)
    (src / "root.tsx").write_text('<SidebarProvider data-wallpaper-workspace="" />\n')
    monkeypatch.setattr(culprit, "REPO", tmp_path)
    monkeypatch.setattr(culprit, "SRC", src)
    culprit.css_rules.cache_clear()


def test_reported_selector_maps_to_its_source_line(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch)
    [(path, line, raw, how)] = culprit.locate(REPORTED)
    assert path.endswith("index.css") and line == 4 and how == "exact text match"
    assert raw.startswith(":is([data-wallpaper-workspace]")
    text = "\n".join(culprit.explain(REPORTED))
    assert "It is possible this section caused the issue, but we cannot be sure" in text
    assert "index.css:4" in text and "root.tsx:1" in text and "holds the whole thread" in text


def test_nested_rule_is_found_and_classified(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch)
    [(path, line, _, _)] = culprit.locate(".aui-thread-viewport:has(.aui-thread-welcome-center)")
    assert line == 7


def test_descendant_shapes():
    walks = [
        ".a:has(.b)",
        ".a:has(> .b .c)",
        ".a:has(> .b, .c .d)",
        ":is([x]):not(:has(.welcome)) > .w",
    ]
    bounded = [".a:has(> .b)", ".a:has(> .b > .c)", ".a:has(+ .b)", ".a:has(~ .b)", ".a:not(.b)"]
    for sel in walks:
        assert culprit.has_descendant(sel), sel
    for sel in bounded:
        assert not culprit.has_descendant(sel), sel
    [(anchor, _)] = culprit.has_descendant(":is([x]):not(:has(.welcome)) > .w")
    assert anchor == ":is([x])"


def test_no_match_says_so(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch)
    text = "\n".join(culprit.explain(".nothing-like-this"))
    assert "No CSS rule or class under studio/frontend/src matched" in text
