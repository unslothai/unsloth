# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort pointer from a slow selector (as Chromium's selector statistics name it) to the
source that probably wrote it. Text matching only: a guess to start diagnosis from, not proof.

  locate(selector)       -> [(path, line, source text, how it matched)] for CSS rules under src/
  has_descendant(sel)    -> the :has() arguments that contain a descendant combinator
  anchor_sites(sel)      -> where the attributes / classes a :has() is anchored on are set in TSX
  changed_in_branch(path, line) -> True / False / None (None: git could not tell)
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "studio" / "frontend" / "src"
# Elements that hold the whole thread: a :has() anchored on one walks every message.
THREAD_ANCESTORS = (
    "[data-wallpaper-workspace]",
    "[data-chat-content-root]",
    "[data-windows-wallpaper-page]",
    ".aui-thread-viewport",
    "[data-chat-page-root]",
    '[data-slot="sidebar-inset"]',
)


def _strip_comments(text: str) -> str:
    # Keep newlines so line numbers survive.
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags = re.S)


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside (), [] and quotes."""
    out, depth, quote, cur = [], 0, "", []
    for ch in text:
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
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def norm(selector: str) -> str:
    s = selector.replace("'", '"')
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([>+~,()])\s*", r"\1", s)
    s = re.sub(r"\[\s*([\w-]+)\s*([~|^$*]?=)\s*", r"[\1\2", s)
    s = re.sub(r"=([\w-]+)\]", r'="\1"]', s)  # [a=b] -> [a="b"]
    return s


@lru_cache(maxsize = 1)
def css_rules() -> tuple:
    rules = []
    for path in sorted(SRC.rglob("*.css")):
        text = _strip_comments(path.read_text(encoding = "utf-8", errors = "replace"))
        start = 0
        for i, ch in enumerate(text):
            if ch in "{};":
                if ch == "{":
                    prelude = text[start:i]
                    stripped = prelude.lstrip()
                    if stripped and not stripped.startswith("@"):
                        line = text.count("\n", 0, start + len(prelude) - len(stripped)) + 1
                        for sel in split_top(stripped.strip()):
                            rules.append((str(path.relative_to(REPO)), line, sel, norm(sel)))
                start = i + 1
    return tuple(rules)


def _tokens(selector: str) -> set[str]:
    return set(re.findall(r"\.[\w-]+|\[[\w-]+|:[\w-]+\(", selector))


def locate(selector: str, limit: int = 3) -> list[tuple[str, int, str, str]]:
    want = norm(selector)
    rules = css_rules()
    exact = [(p, ln, raw, "exact text match") for p, ln, raw, n in rules if n == want]
    if exact:
        return _dedupe(exact)[:limit]
    # Nested CSS (`&`) and serialization differences: rank by shared classes / attributes.
    toks = _tokens(selector)
    if not toks:
        return []
    scored = []
    for p, ln, raw, n in rules:
        common = toks & _tokens(raw)
        if common:
            scored.append((len(common) / len(toks | _tokens(raw)), p, ln, raw, len(common)))
    scored.sort(reverse = True)
    return _dedupe(
        [
            (p, ln, raw, f"partial match, {k} of {len(toks)} classes / attributes in common")
            for score, p, ln, raw, k in scored
            if score >= 0.4
        ]
    )[:limit]


def _dedupe(rows: list) -> list:
    seen, out = set(), []
    for row in rows:
        if (row[0], row[1]) not in seen:
            seen.add((row[0], row[1]))
            out.append(row)
    return out


def _has_args(selector: str) -> list[tuple[str, str]]:
    """(anchor compound, argument) for each :has( in the selector."""
    out = []
    for m in re.finditer(r":has\(", selector):
        depth, j = 1, m.end()
        while j < len(selector) and depth:
            depth += {"(": 1, ")": -1}.get(selector[j], 0)
            j += 1
        arg = selector[m.end() : j - 1]
        out.append((_compound_before(selector[: m.start()]), arg))
    return out


def _compound_before(head: str) -> str:
    """The compound a pseudo-class at the end of `head` belongs to; climbs out of :not( / :is(."""
    depth, k = 0, len(head)
    while k > 0:
        c = head[k - 1]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and c in " >+~,":
            break
        k -= 1
    if head[k:] or k == 0 or head[k - 1] != "(":
        return head[k:]
    # inside :not( ... ): the anchor is the compound that :not() belongs to
    return _compound_before(re.sub(r":[\w-]+$", "", head[: k - 1]))


def has_descendant(selector: str) -> list[tuple[str, str]]:
    """:has() arguments with a descendant combinator: Chromium then searches the anchor's subtree."""
    hits = []
    for anchor, arg in _has_args(selector):
        for part in split_top(arg):
            # No leading combinator means "a descendant anywhere": the whole subtree is searched.
            if part.strip()[:1] not in (">", "+", "~"):
                hits.append((anchor, part.strip()))
                break
            depth, quote = 0, ""
            p = re.sub(r"\s*([>+~])\s*", r"\1", part.strip())
            for i, c in enumerate(p):
                if quote:
                    quote = "" if c == quote else quote
                elif c in "\"'":
                    quote = c
                elif c in "([":
                    depth += 1
                elif c in ")]":
                    depth -= 1
                elif c == " " and depth == 0:
                    hits.append((anchor, part.strip()))
                    break
            else:
                continue
            break
    return hits


def anchor_sites(anchor: str, limit: int = 3) -> list[tuple[str, int, str]]:
    names = re.findall(r"\[([\w-]+)", anchor) + [c[1:] for c in re.findall(r"\.[\w-]+", anchor)]
    out = []
    for name in names:
        pat = re.compile(re.escape(name) + r"(?![\w-])")
        for path in sorted(SRC.rglob("*.tsx")):
            for i, line in enumerate(
                path.read_text(encoding = "utf-8", errors = "replace").splitlines(), 1
            ):
                if pat.search(line) and not line.lstrip().startswith(("//", "*", "/*")):
                    if "querySelector" in line or "closest(" in line:
                        continue
                    out.append((str(path.relative_to(REPO)), i, line.strip()[:140]))
                    if len(out) >= limit:
                        return out
    return out


def utility_uses(selector: str, limit: int = 3) -> list[tuple[str, int, str]]:
    """TSX lines using the escaped class names (Tailwind utilities) in `selector`."""
    classes = [re.sub(r"\\(.)", r"\1", c) for c in re.findall(r"\.((?:\\.|[\w-])+)", selector)]
    classes = [
        c for c in classes if not c.replace("-", "").isalpha() or ":" in c or "[" in c
    ] or classes
    out = []
    for cls in classes:
        pat = re.compile(r"(?<![\w:-])" + re.escape(cls) + r"(?![\w-])")
        for path in sorted(SRC.rglob("*.tsx")):
            for i, line in enumerate(
                path.read_text(encoding = "utf-8", errors = "replace").splitlines(), 1
            ):
                if pat.search(line):
                    out.append((str(path.relative_to(REPO)), i, line.strip()[:140]))
                    if len(out) >= limit:
                        return out
    return out


def thread_anchor(anchor: str) -> str | None:
    n = norm(anchor)
    for a in THREAD_ANCESTORS:
        if norm(a) in n:
            return a
    return None


@lru_cache(maxsize = 64)
def _changed_ranges(path: str) -> tuple | None:
    base = os.environ.get("GITHUB_BASE_REF") or "main"
    for ref in (f"origin/{base}...HEAD", "HEAD^1"):
        try:
            diff = subprocess.run(
                ["git", "-C", str(REPO), "diff", "--unified=0", ref, "--", path],
                capture_output = True,
                text = True,
                timeout = 30,
            )
        except Exception:  # noqa: BLE001
            return None
        if diff.returncode != 0:
            continue
        return tuple(
            (int(m.group(1)), max(int(m.group(2) or 1), 1))
            for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff.stdout, re.M)
        )
    return None


def changed_in_branch(path: str, line: int) -> bool | None:
    ranges = _changed_ranges(path)
    if ranges is None:
        return None
    return any(start <= line < start + count for start, count in ranges)


STATE = re.compile(r":(?:hover|focus|focus-visible|focus-within|active|checked)\b|\[data-state")


def suspect_rules(limit: int = 5) -> list[tuple[int, str, int, str, list[str]]]:
    """:has() rules ranked by how likely they are to restyle much of the page on each keystroke."""
    rows = []
    for path, line, raw, _ in css_rules():
        if ":has(" not in raw:
            continue
        reasons, score = [], 0
        for anchor, _arg in has_descendant(raw)[:1]:
            where = thread_anchor(anchor)
            if where:
                reasons.append(f"searches all descendants of {where}, which holds the thread")
                score += 3
            else:
                reasons.append(f"searches all descendants of `{anchor[:60]}`")
                score += 1
        if any(STATE.search(arg) for _, arg in _has_args(raw)):
            reasons.append(
                "its :has() argument depends on hover / focus / open state, which input flips"
            )
            score += 2
        tail = raw[raw.rfind(")") + 1 :].strip()
        if " " in tail or re.search(r":not\(|\[class\*=|\*", tail):
            reasons.append(
                "its subject is a broad descendant of the :has() element, so one change re-matches that subtree"
            )
            score += 1
        if changed_in_branch(path, line):
            reasons.insert(0, "changed on this branch")
            score += 4
        if score:
            rows.append((score, path, line, raw, reasons))
    rows.sort(key = lambda r: (-r[0], r[1], r[2]))
    return rows[:limit]


def explain(selector: str, indent: str = "    ") -> list[str]:
    """Human-readable guesses for one slow selector."""
    lines = []
    found = locate(selector)
    if found:
        lines.append(
            f"{indent}It is possible this section caused the issue, but we cannot be sure (a text match):"
        )
        for path, line, raw, how in found:
            changed = changed_in_branch(path, line)
            tag = {True: ", changed on this branch", False: ", unchanged on this branch", None: ""}[
                changed
            ]
            lines.append(f"{indent}  {path}:{line} ({how}{tag})")
            lines.append(f"{indent}    {raw[:220]}")
    else:
        uses = utility_uses(selector)
        if uses:
            lines.append(
                f"{indent}It is possible these class names caused the issue, but we cannot be sure "
                "(a Tailwind utility; text match on its class):"
            )
            lines += [f"{indent}  {p}:{ln}  {txt}" for p, ln, txt in uses]
        else:
            lines.append(
                f"{indent}No CSS rule or class under studio/frontend/src matched this selector's text; "
                "it may come from a dependency's stylesheet."
            )
    for anchor, arg in has_descendant(selector):
        where = thread_anchor(anchor)
        lines.append(
            f"{indent}Pattern: `:has({arg[:120]})` on `{anchor[:160]}` searches all descendants (no leading `>`, "
            "or a space inside), so Chromium re-walks everything under that element after each DOM change, "
            "whether or not anything matches."
        )
        if where:
            sites = anchor_sites(where)
            site = f" (first found in {sites[0][0]}:{sites[0][1]})" if sites else ""
            lines.append(
                f"{indent}  `{where}`{site} holds the whole thread, so every keystroke walks every message."
            )
    return lines
