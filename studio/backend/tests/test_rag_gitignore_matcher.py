# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compiled .gitignore matcher a linked-folder scan uses gives the rule-by-rule answer, with
one regex match per run of rules instead of one per rule."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from core.rag import folder_sync, gitignore

_ATOMS = [
    "a",
    "b",
    "foo",
    "x.py",
    "*.py",
    "build",
    "*",
    "**",
    "?",
    "[ab]",
    "[!a]",
    "[^b]",
    "[]a]",
    "[a-c]",
    "\\*",
    "\\ ",
    "/",
    "\\!",
    "\\#",
    "#c",
    "!",
    " ",
    "\\",
    "[",
    "(?P<g0>",
    ")",
    "$",
    "^",
    ".",
    "\n",
]
_PATH_ATOMS = ["a", "b", "foo", "x.py", "build", "ab", "*", "(?P<g0>", "$", "é", "\n"]


def _rules(rng: random.Random) -> tuple[gitignore.Rule, ...]:
    rules: list[gitignore.Rule] = []
    for _ in range(rng.randint(1, 4)):
        base = rng.choice(["", "", "a", "foo/bar", "a.b", "x*y"])
        text = "\n".join(
            "".join(rng.choice(_ATOMS) for _ in range(rng.randint(1, 5)))
            for _ in range(rng.randint(0, 30))
        )
        rules += gitignore.parse(text, base)
    return tuple(rules)


def _path(rng: random.Random, rules) -> str:
    rel = "/".join(
        "".join(rng.choice(_PATH_ATOMS) for _ in range(rng.randint(1, 2)))
        for _ in range(rng.randint(1, 5))
    )
    if rules and rng.random() < 0.3 and rng.choice(rules).base:
        rel = rng.choice(rules).base + "/" + rel
    return rel


def test_matcher_agrees_with_rule_by_rule_matching():
    rng = random.Random(0)
    checked = ignored = 0
    for _ in range(400):
        rules = _rules(rng)
        matcher = gitignore.Matcher(rules)
        for _ in range(40):
            rel, is_dir = _path(rng, rules), rng.random() < 0.5
            expected = gitignore.is_ignored(rel, is_dir, rules)
            assert matcher.is_ignored(rel, is_dir) is expected, (rel, is_dir)
            checked += 1
            ignored += expected
    # Both answers are exercised.
    assert 0.1 < ignored / checked < 0.9


@pytest.mark.parametrize(
    "rules, rel, is_dir",
    [
        ("*.md\n!keep.md", "keep.md", False),
        ("*.md\n!keep.md", "drop.md", False),
        ("generated/", "generated", True),
        ("generated/", "generated", False),
        ("*a*b\n*.log", "xaab", False),
        ("a*b*c\n!acbc", "acbc", False),
        ("logs/**\n!logs/keep/\nlogs/keep/*.tmp", "logs/keep/x.tmp", False),
        ("logs/**\n!logs/keep/", "logs/keep", True),
        ("x\n", "x\n", False),
    ],
)
def test_matcher_keeps_last_rule_wins(rules, rel, is_dir):
    for base in ("", "sub"):
        parsed = tuple(gitignore.parse(rules, base))
        target = f"{base}/{rel}" if base else rel
        assert gitignore.Matcher(parsed).is_ignored(target, is_dir) is gitignore.is_ignored(
            target, is_dir, parsed
        )


def test_adjacent_rules_share_one_regex():
    text = "\n".join(f"*.ext{i}" for i in range(100)) + "\n!keep.ext1\ncache/\n*.tmp\n"
    matcher = gitignore.Matcher(tuple(gitignore.parse(text, "")))
    # Files: one run of 100 rules, the negation, then *.tmp (cache/ is directory-only).
    assert len(matcher._file_plan) == 3
    # Directories: the trailing cache/ and *.tmp share a run.
    assert len(matcher._dir_plan) == 3
    assert all(isinstance(m, re.Pattern) for _, m in matcher._file_plan)


def test_a_rule_with_named_groups_keeps_its_own_regex():
    parsed = tuple(gitignore.parse("*a*b\n*c*d\n", ""))
    assert all(rule.regex.groupindex for rule in parsed)
    matcher = gitignore.Matcher(parsed)
    assert len(matcher._file_plan) == 2
    assert matcher.is_ignored("xaab", False) and matcher.is_ignored("q/ccd", False)
    assert not matcher.is_ignored("xaa/b", False)


def _write(
    root: Path,
    rel: str,
    text: str = "words",
) -> None:
    path = root / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def test_scan_matches_through_one_matcher_per_gitignore(linkable_temp_base, tmp_path, monkeypatch):
    source = linkable_temp_base / f"scan-{tmp_path.name}"
    source.mkdir(parents = True)
    _write(source, ".gitignore", "*.log\ngenerated/\n!important.log\n")
    _write(source, "notes.md")
    _write(source, "debug.log")
    _write(source, "important.log")
    _write(source, "generated/out.md")
    _write(source, "pkg/.gitignore", "local.md\n")
    _write(source, "pkg/local.md")
    _write(source, "pkg/kept.md")
    _write(source, "pkg/deep/more.md")
    _write(source, "local.md")
    built = []
    real_matcher = gitignore.Matcher

    def counting_matcher(rules):
        built.append(len(rules))
        return real_matcher(rules)

    def per_rule(*_args, **_kwargs):
        raise AssertionError("the scan matched rule by rule")

    monkeypatch.setattr(gitignore, "Matcher", counting_matcher)
    monkeypatch.setattr(gitignore, "is_ignored", per_rule)
    report: dict[str, int] = {}
    found, _ = folder_sync._scan(str(source), report = report)
    assert set(found) == {
        "notes.md",
        "important.log",
        "pkg/kept.md",
        "pkg/deep/more.md",
        "local.md",
    }
    assert report["gitignored"] == 3
    # The root and pkg/ hold a .gitignore; pkg/deep reuses pkg/'s matcher.
    assert sorted(built) == [3, 4]
