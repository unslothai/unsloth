# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A sibling combinator (`+`, `~`, or `:has(+ ...)` / `:has(~ ...)`) must not end in a compound with no
class, id or attribute.

Chromium matches a rule whose last compound is a bare tag or a bare `:has()` against every such
element, and evaluating the sibling part marks the element's parent as having children styled by
their siblings. With KaTeX's MathML on the page, Chromium's MathML stylesheet
(`mprescripts ~ :nth-child(...)`) makes every insertion or removal of a child of a marked parent
restyle that parent's whole subtree. For <body> that is the whole page, once per modal open and
once per close: Radix inserts its focus guards before #root and removes them again.
`.composer-run-settings :has(+ [data-pinned-footer])` and `.browser-web-toolbar > form ~ div` did
exactly that, and opening Settings restyled the whole thread (the area matrix's `length:settings`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_culprit as C  # noqa: E402

REPO = C.REPO
SRC = C.SRC


def _compounds(selector: str) -> list[tuple[str, str]]:
    """(combinator before, compound) pairs of a normalized selector; ' ' is the descendant combinator."""
    out, cur, depth, quote, comb = [], [], 0, "", ""
    for ch in selector:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch in " >+~":
            if cur:
                out.append((comb, "".join(cur)))
                cur = []
            comb = ch
            continue
        cur.append(ch)
    if cur:
        out.append((comb, "".join(cur)))
    return out


def _top_level(compound: str) -> str:
    """The compound with everything inside (), [] removed, so only its own simple selectors remain."""
    out, depth = [], 0
    for ch in compound:
        if ch in "([":
            if depth == 0:
                out.append(ch)
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                out.append(ch)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _pseudo_args(compound: str, name: str) -> list[list[str]]:
    """The argument lists of each top-level `:name(...)` in a compound."""
    found, i = [], compound.find(f":{name}(")
    while i >= 0:
        depth, j = 0, i + len(name) + 2
        start = j
        while j < len(compound):
            if compound[j] in "([":
                depth += 1
            elif compound[j] in ")]":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        found.append(C.split_top(compound[start:j]))
        i = compound.find(f":{name}(", j)
    return found


def _keyed(selector: str) -> bool:
    top = _top_level(selector)
    return "&" in top or any(mark in top for mark in (".", "#", "["))


def keyless_sibling(selector: str) -> bool:
    """True when the rule's last compound has no class / id / attribute and a sibling relation is
    evaluated for it (a `+` / `~` right before it, or a `:has(+ ...)` / `:has(~ ...)` in it)."""
    parts = _compounds(C.norm(selector))
    if not parts:
        return False
    comb, compound = parts[-1]
    compound = compound.split("::", 1)[0]
    # `:is(.a, [b])` keys the compound when every argument does: it is checked before what follows it.
    if _keyed(compound) or any(
        args and all(_keyed(a) for a in args)
        for name in ("is", "where")
        for args in _pseudo_args(compound, name)
    ):
        return False
    has_sibling = any(
        a.startswith(("+", "~")) for args in _pseudo_args(compound, "has") for a in args
    )
    return comb in ("+", "~") or has_sibling


@pytest.mark.parametrize(
    "selector, flagged",
    [
        (".composer-run-settings :has(+ [data-pinned-footer])", True),
        (".browser-web-toolbar > form ~ div", True),
        ("p + p", True),
        (".a :has(~ .b, > .c)", True),
        (".composer-run-settings [data-pinned-footer-above]", False),
        (".browser-web-toolbar > .browser-web-toolbar-trailing", False),
        (".a:has(~ .b)", False),
        (".peer:checked ~ .x", False),
        ("[data-variant=legend] + .y", False),
        (':root[data-palette="x"] [class~="bg-primary"]', False),
        (".x > :last-child:nth-child(2n+1)", False),
        (".a ~ div::before", True),
        ("& + div", True),
        (':is([data-slot="a"], [data-slot="b"]):has(~ [data-x])', False),
        (":is(.a, div):has(+ .b)", True),
    ],
)
def test_the_check_tells_keyed_from_keyless(selector, flagged):
    assert keyless_sibling(selector) is flagged


def test_no_frontend_rule_ends_a_sibling_relation_on_a_keyless_compound():
    C.css_rules.cache_clear()  # another test may have cached a synthetic tree in this worker
    rules = C.css_rules()
    assert len(rules) > 1000, "found too few CSS rules under studio/frontend/src to trust the scan"
    assert all(path.startswith("studio/frontend/src/") for path, *_ in rules)
    offenders = [f"{path}:{line}: {raw}" for path, line, raw, _ in rules if keyless_sibling(raw)]
    assert offenders == [], (
        "These rules end a sibling relation on a compound with no class, id or attribute, so Chromium "
        "evaluates the relation for every element of that kind and marks each parent; with KaTeX's "
        "MathML on the page, inserting any child of a marked parent restyles its whole subtree (the "
        "whole page for <body>, on every modal open and close). Put a class or attribute on the "
        "element instead:\n  " + "\n  ".join(offenders)
    )


def test_the_two_rewritten_rules_still_find_their_elements():
    """The markup the replacement selectors rely on is still rendered where the old ones matched."""
    config = (SRC / "features" / "model-picker" / "components" / "model-config-page.tsx").read_text(
        encoding = "utf-8"
    )
    above = config.find('data-pinned-footer-above={variant === "page" || undefined}')
    footer = config.find('data-pinned-footer={variant === "page" || undefined}')
    assert above >= 0 and footer > above, "the row above the pinned footer is no longer marked"
    browser = (SRC / "features" / "browser" / "browser-panel.tsx").read_text(encoding = "utf-8")
    web_toolbar = browser[browser.index("function WebToolbar(") :]
    web_toolbar = web_toolbar[: web_toolbar.index("\nfunction ")]
    assert web_toolbar.index("<AddressBar") < web_toolbar.index(
        "browser-web-toolbar-trailing"
    ), "the web toolbar's trailing actions must stay marked and after the address bar"
    css = (SRC / "index.css").read_text(encoding = "utf-8")
    assert ".composer-run-settings [data-pinned-footer-above]" in css
    assert ".browser-web-toolbar > .browser-web-toolbar-trailing" in css
