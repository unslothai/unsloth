# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static check of the built stylesheet: no new non-subject :has() rules.

A rule like `X:has(...) Y` or `X:has(...) > Y` (the :has() is not in the last compound; Tailwind's
group-has-* variants compile to `:is(:where(.group):has(...) *)`) makes Chromium and WebKit keep one
invalidation set for the :has() pseudo: any child insertion or attribute change under an ancestor
of the thread re-runs it and restyles every element that matches the right-hand side, so opening
the sidebar, run settings, the model picker or the browser pane restyles the whole chat (9-10k
elements at 150 turns; deleting these rules takes a sidebar toggle to 141). `:has()` in the subject
(`X:has(...)` with nothing after it) does not do this.

The rules main ships today are listed in has_subject_allowlist.txt (being rewritten elsewhere); a
rule not in it fails the gate. LONG_THREAD_HAS_LINT_WRITE=1 rewrites the list from the build.
"""

from __future__ import annotations

from pathlib import Path

ALLOWLIST = Path(__file__).resolve().parent / "has_subject_allowlist.txt"
RECURSE = (":is(", ":where(", ":not(", ":matches(", ":-webkit-any(", ":-moz-any(")

# Every style rule's selector in a stylesheet, as the browser serializes it (nested rules included).
SELECTORS_JS = """(text) => {
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(text);
  const out = [];
  const walk = (list) => {
    for (const r of list) {
      if (typeof r.selectorText === 'string' && r.selectorText.includes(':has(')) out.push(r.selectorText);
      if (r.cssRules && r.cssRules.length) walk(r.cssRules);
    }
  };
  walk(sheet.cssRules);
  return out;
}"""


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, bracket, quote, cur, i = [], 0, 0, "", [], 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            cur.append(text[i : i + 2])
            i += 2
            continue
        if quote:
            quote = "" if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket -= 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == sep and depth == 0 and bracket == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _matching(text: str, start: int) -> int:
    """Index of the ')' closing the '(' at text[start - 1]."""
    depth, quote, i = 1, "", start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if quote:
            quote = "" if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _scan(selector: str) -> tuple[bool, bool]:
    """(a :has() is followed by a combinator, the selector contains a :has()) for one complex selector."""
    bad, has_here, any_has = False, False, False
    i, bracket, quote = 0, 0, ""
    s = selector.strip()
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if quote:
            quote = "" if c == quote else quote
            i += 1
            continue
        if c in "\"'":
            quote = c
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket -= 1
        elif bracket == 0 and s.startswith(":has(", i):
            has_here = any_has = True
            i = _matching(s, i + 5) + 1
            continue
        elif bracket == 0 and any(s.startswith(p, i) for p in RECURSE):
            open_at = s.index("(", i) + 1
            close = _matching(s, open_at)
            for inner in _split_top(s[open_at:close]):
                inner_bad, inner_has = _scan(inner)
                bad = bad or inner_bad
                has_here = has_here or inner_has
                any_has = any_has or inner_has
            i = close + 1
            continue
        elif bracket == 0 and c == "(":
            # any other functional pseudo-class (:nth-child(2n+1), :lang(), :dir()): its argument is
            # not a selector at this level, so nothing inside it is a combinator
            i = _matching(s, i + 1) + 1
            continue
        elif bracket == 0 and (c in ">+~" or c.isspace()):
            # a combinator: the compound before it ends here
            j = i
            while j < len(s) and (s[j].isspace() or s[j] in ">+~"):
                j += 1
            if j < len(s) and has_here:
                bad = True
            has_here = False
            i = j
            continue
        i += 1
    return bad, any_has


def non_subject_has(selector_list: str) -> bool:
    """True when any selector in the list has a :has() that is not in its last compound."""
    return any(_scan(sel)[0] for sel in _split_top(selector_list))


def offenders(selectors: list[str]) -> list[str]:
    seen, out = set(), []
    for sel in selectors:
        if sel not in seen and non_subject_has(sel):
            seen.add(sel)
            out.append(sel)
    return out


def allowlist() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    return {
        ln.rstrip("\n")
        for ln in ALLOWLIST.read_text(encoding = "utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def write_allowlist(found: list[str]) -> None:
    head = (
        "# Non-subject :has() selectors main shipped when the gate's :has() lint was added (see _has_lint.py).\n"
        "# Being rewritten as data attributes; delete lines as they go. One selector per line, as Chromium\n"
        "# serializes it from the production build. Regenerate: LONG_THREAD_HAS_LINT_WRITE=1.\n"
    )
    ALLOWLIST.write_text(head + "\n".join(sorted(set(found))) + "\n", encoding = "utf-8")


def check(found: list[str]) -> tuple[list[str], list[str]]:
    """(new offenders not in the allowlist, allowlisted selectors no longer in the build)."""
    allow = allowlist()
    new = [s for s in found if s not in allow]
    stale = sorted(allow - set(found))
    return new, stale
