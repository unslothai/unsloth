# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long-chat gate's static lint for non-subject :has() rules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _has_lint as lint  # noqa: E402


@pytest.mark.parametrize(
    "selector, bad",
    [
        (".a:has(.b)", False),
        (".a .b:has(.c)", False),
        (".a:has(.b)::after", False),
        (".a:has(.b), .c", False),
        ('[data-a="b c"]:has(.d)', False),
        (".x:not(:has(.y))", False),
        (".a:has(.b) .c", True),
        (".a:has(> .b) > .c", True),
        (".a:has(.b) + .c", True),
        (".x:not(:has(.y)) .z", True),
        (".c, .a:has(.b) .d", True),
        # Tailwind group-has-* compiles to this
        (r".group-has-\[\:checked\]\:x:is(:where(.group):has(:checked) *)", True),
        (
            ":is([data-wallpaper-workspace], [data-chat-content-root]):has(> .chat-wallpaper[data-ready]) .thread-bottom-fade",
            True,
        ),
        (".hub-readme-prose :is(p, div):has(> a:nth-of-type(2) > img) img", True),
        (".hub-readme-prose :is(p, div):has(> a:nth-of-type(2) > img)", False),
        # arithmetic and attribute values are not combinators
        (".a:has(.b):nth-child(2n+1)", False),
        (".a:has(.b):nth-of-type(3n + 2)", False),
        (".a:has(.b):nth-last-child(-n+3)", False),
        (".a:has(.b) :nth-child(2n+1)", True),
        ('[data-x="a>b"]:has(.c)', False),
        ('[data-x="a ~ b"]:has(.c)', False),
        ('.a:has(.b)[data-x="1+2"]', False),
        ('.a:has([data-x="p > q"]) > .c', True),
        (".a:has(.b):is(:nth-child(2n+1), .c)", False),
    ],
)
def test_non_subject_has(selector, bad):
    assert lint.non_subject_has(selector) is bad


def test_new_rules_fail_and_gone_rules_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(lint, "ALLOWLIST", tmp_path / "allow.txt")
    lint.write_allowlist([".a:has(.b) .c", ".old:has(.x) .y"])
    assert lint.allowlist() == {".a:has(.b) .c", ".old:has(.x) .y"}
    found = lint.offenders([".a:has(.b) .c", ".a:has(.b) .c", ".n:has(.m) > .k", ".ok:has(.z)"])
    assert found == [".a:has(.b) .c", ".n:has(.m) > .k"]
    assert lint.check(found) == ([".n:has(.m) > .k"], [".old:has(.x) .y"])


def test_allowlist_is_well_formed():
    allow = lint.allowlist()
    assert allow, "has_subject_allowlist.txt is empty or missing"
    assert all(
        lint.non_subject_has(s) for s in allow
    ), "allowlist holds a selector the lint would not flag"
