# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate: typing in a long chat must not cost more as the chat grows.

The production build of smoke-thread-weight.html (vite.long-thread.config.ts) is seeded with an
agent-style history (open reasoning, tool cards, web sources, compaction markers, code, KaTeX,
tables) inside the app's own ancestors (`?shell=1`). While keys are typed, Chromium's selector
statistics give the time spent matching each selector. Two checks, both blind to how fast the
runner is:

  1. `:has()` matching per keystroke in the large chat stays under MAX_HAS_MS. A descendant
     `:has()` on an element that holds the thread (#16603) walks every message on every key:
     74-84 ms per key at 200 turns, against 0.2 ms once fixed.
  2. Total selector matching per keystroke does not grow with the chat: large / small at most
     MAX_MATCH_RATIO for LARGE / SMALL times the turns (healthy: 1.00x).

Fails fast: the large chat is measured first and a blown `:has()` budget stops the run there; an
untraced pre-check shortens the traced window to 2 keys on a slow page; a second repeat runs only
when the ratio is close to its limit. A failure prints the measured numbers, how far over budget
they are, and the selectors responsible with their cost per key; selectors tried on most of the
DOM are marked as likely victims of a whole-page restyle. It then pins the trigger down by
removing :has() rules from the stylesheet before fresh page loads (a bisection, time-boxed) and
names the rule whose removal makes typing cheap, with the CSS file and line it probably comes from.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _css_culprit import explain, has_descendant, thread_anchor  # noqa: E402
from _playwright_robust import chromium_launch_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
DIST = Path(os.environ.get("LONG_THREAD_DIST", REPO / ".playwright-cli" / "long-thread-production"))
SMALL = int(os.environ.get("LONG_THREAD_SMALL", "40"))
LARGE = int(os.environ.get("LONG_THREAD_LARGE", "200"))
# Measured (40 -> 200 turns, per keystroke): with the #16603 rule, selector matching 79-80 ->
# 423-472 ms (5.3-6.0x) and :has() 74-84 ms; healthy, 0.7-0.9 -> 0.7-1.0 ms (1.00x) and 0.2 ms.
MAX_MATCH_RATIO = float(os.environ.get("LONG_THREAD_MAX_MATCH_RATIO", "2.0"))
MAX_HAS_MS = float(os.environ.get("LONG_THREAD_MAX_HAS_MS", "5.0"))
# Style recalc for one open or close of a modal layer (Radix writes pointer-events: none on <body>,
# an inherited property). Healthy: 1-5 ms at 200 turns; when the thread inherits it, every message
# restyles and this is hundreds of ms in Chromium and seconds in WebKit.
MAX_LOCK_MS = float(os.environ.get("LONG_THREAD_MAX_LOCK_MS", "30.0"))
LOCK_TOGGLES = 4
# A ratio inside this band gets a second interleaved repeat before the verdict; outside it, one
# measurement is decisive (healthy runs read 1.00x, the regression 5.3-6.0x).
RECHECK_BAND = (1.5, 3.0)
# Keeps sub-millisecond noise from becoming a large ratio.
MATCH_FLOOR_MS = 2.0
HEALTHY = "healthy reference: 0.7-1.0 ms/key selector matching, 1.00x, 0.2 ms/key :has()"
KEYS = "the quick "
PRECHECK_KEYS = "ab"
# Untraced style recalc per key above this on the large chat means a regressed build (healthy:
# 1-15 ms; #16603: 140-1000 ms); its traced window is then 2 keys, not 10. Above EXTREME the
# page is hundreds of times a healthy one and fails on the untraced number alone.
SLOW_STYLE_MS = 60.0
EXTREME_STYLE_MS = 300.0
# Rule bisection on a failure: a smaller chat, 3 keys per probe, a time budget.
PIN_TURNS = 60
PIN_KEYS = "abc"
PIN_BUDGET_S = 240.0
TOP = 5
SEED_OPTIONS = {"agent": {"rounds": 2, "compactionEvery": 10, "sources": True, "rich": True}}
HINT = (
    "Typical cause: a :has() whose argument has a descendant combinator (a space), anchored on an "
    "element that holds the thread ([data-wallpaper-workspace], [data-chat-content-root], "
    "[data-windows-wallpaper-page], .aui-thread-viewport). Chromium walks every message on each "
    "keystroke even when nothing matches. Use a child-only path (`:has(> a > b)`) or a flag on "
    "<html> set from a layout effect (see useMarkChatWelcome in features/chat-background)."
)


def build() -> None:
    if os.environ.get("LONG_THREAD_DIST"):
        return
    # npx is npx.cmd on Windows, which CreateProcess only finds through the shell.
    subprocess.run(
        ["npx", "vite", "build", "--config", "vite.long-thread.config.ts"],
        cwd = FRONTEND,
        check = True,
        shell = os.name == "nt",
    )


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(_QuietHandler, directory = str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: D401
        pass


def metrics(cdp) -> dict:
    return {m["name"]: m["value"] for m in cdp.send("Performance.getMetrics")["metrics"]}


# The stylesheet is edited as text before the page loads: rules switched off in a page that has
# already rendered leave Chromium's :has() flags on its elements, so the cost stays (measured).
SHEET_RULES_JS = """(text) => {
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(text);
  const out = [];
  const walk = (list, path) => {
    for (let i = 0; i < list.length; i++) {
      const r = list[i];
      const p = path.concat(i);
      if (typeof r.selectorText === 'string' && r.selectorText.includes(':has(')) out.push({p, s: r.selectorText});
      if (r.cssRules && r.cssRules.length) walk(r.cssRules, p);
    }
  };
  walk(sheet.cssRules, []);
  return out;
}"""
SHEET_WITHOUT_JS = """([text, paths]) => {
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(text);
  for (const p of paths) {
    let list = sheet.cssRules, rule = null;
    for (const i of p) { rule = list[i]; list = rule.cssRules; }
    if (rule) rule.selectorText = '.__long_chat_gate_disabled__';
  }
  return [...sheet.cssRules].map((r) => r.cssText).join('\\n');
}"""


def main_stylesheet(dist: Path) -> tuple[str, str] | None:
    """(URL path, text) of the built stylesheet that carries the :has() rules."""
    best = None
    for css in (dist / "assets").glob("*.css"):
        text = css.read_text(encoding = "utf-8", errors = "replace")
        if ":has(" in text and (best is None or len(text) > len(best[1])):
            best = (f"/assets/{css.name}", text)
    return best


def open_thread(
    browser,
    base: str,
    turns: int,
    css: tuple[str, str] | None = None,
):
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
    if css is not None:
        path, body = css
        ctx.route(
            f"{base}{path}",
            lambda route: route.fulfill(status = 200, body = body, content_type = "text/css"),
        )
    page = ctx.new_page()
    page.goto(f"{base}/smoke-thread-weight.html?shell=1", wait_until = "load")
    page.wait_for_function("() => !!window.__threadWeight", timeout = 60_000)
    count = turns * 2
    page.evaluate("([n, o]) => window.__threadWeight.seed(n, o)", [count, SEED_OPTIONS])
    page.wait_for_function(
        "(n) => window.__threadWeight.messageCount() >= n", arg = count, timeout = 120_000
    )
    page.wait_for_timeout(1500 if css is not None else 2500)
    cdp = ctx.new_cdp_session(page)
    cdp.send("Performance.enable")
    box = page.locator(".aui-composer-input").first
    box.click()
    # Focusing the composer restyles focus-dependent rules once, across the thread; that one-off
    # is not a per-keystroke cost, so let it and a warm-up key land before the window opens.
    page.wait_for_timeout(800)
    page.keyboard.press("x")
    page.keyboard.press("Backspace")
    page.wait_for_timeout(400)
    return ctx, page, cdp


def style_per_key(cdp, page, keys: str) -> float:
    m0 = metrics(cdp)
    for ch in keys:
        page.keyboard.press("Space" if ch == " " else ch)
        page.wait_for_timeout(60)
    page.wait_for_timeout(200)
    return 1000 * (metrics(cdp)["RecalcStyleDuration"] - m0["RecalcStyleDuration"]) / len(keys)


def lock_style_ms(cdp, page) -> float:
    """Untraced style recalc per open/close of a modal layer's body pointer-events lock."""
    m0 = metrics(cdp)
    for _ in range(LOCK_TOGGLES):
        for value in ("none", ""):
            page.evaluate(
                "(v) => { document.body.style.pointerEvents = v; return getComputedStyle(document.body).color; }",
                value,
            )
            page.wait_for_timeout(30)
    return (
        1000
        * (metrics(cdp)["RecalcStyleDuration"] - m0["RecalcStyleDuration"])
        / (2 * LOCK_TOGGLES)
    )


def measure(browser, base: str, turns: int) -> dict:
    ctx, page, cdp = open_thread(browser, base, turns)
    shape = page.evaluate("""() => ({
      messages: document.querySelectorAll('[data-role]').length,
      nodes: document.getElementsByTagName('*').length,
      sources: document.querySelectorAll('[data-role="assistant"] a[href^="https://example"]').length,
      reasoning: document.querySelectorAll('.aui-reasoning-root, [data-reasoning]').length,
      code: document.querySelectorAll('pre').length,
      katex: document.querySelectorAll('.katex').length,
      tables: document.querySelectorAll('table').length,
      workspace: !!document.querySelector('[data-wallpaper-workspace] .thread-bottom-fade'),
    })""")
    # Untraced pre-check: selector statistics slow a regressed build ~50x, so a slow page gets a
    # short traced window; the verdict still comes from the traced numbers.
    pre = style_per_key(cdp, page, PRECHECK_KEYS)
    if pre >= EXTREME_STYLE_MS:
        # Hundreds of times a healthy keystroke: one traced key would take minutes and could not
        # change the verdict, so skip tracing and let the pin-down find the rule.
        typed_ok = page.evaluate("() => window.__threadWeight.composerText()") == PRECHECK_KEYS
        ctx.close()
        return {
            "turns": turns,
            "shape": shape,
            "typed_ok": typed_ok,
            "keys": 0,
            "extreme": True,
            "precheck_style_ms": round(pre, 1),
            "typing_s": 0.0,
            "match_ms": None,
            "has_ms": None,
            "style_ms": None,
            "selectors": {},
        }
    keys = KEYS if pre < SLOW_STYLE_MS else KEYS[:2]
    events: list = []
    done: list = []
    cdp.on("Tracing.dataCollected", lambda d: events.extend(d["value"]))
    cdp.on("Tracing.tracingComplete", lambda d: done.append(True))
    m0 = metrics(cdp)
    t0 = time.perf_counter()
    cdp.send(
        "Tracing.start",
        {
            "traceConfig": {
                "includedCategories": ["devtools.timeline", "disabled-by-default-blink.debug"]
            },
            "transferMode": "ReportEvents",
        },
    )
    for ch in keys:
        page.keyboard.press("Space" if ch == " " else ch)
        page.wait_for_timeout(60)
    page.wait_for_timeout(300)
    cdp.send("Tracing.end")
    while not done:
        page.wait_for_timeout(50)
    typing_s = time.perf_counter() - t0
    m1 = metrics(cdp)
    typed_ok = page.evaluate("() => window.__threadWeight.composerText()") == PRECHECK_KEYS + keys
    lock_ms = lock_style_ms(cdp, page)
    ctx.close()
    n = len(keys)
    selectors: dict[str, dict] = {}
    for event in events:
        stats = (event.get("args") or {}).get("selector_stats")
        for timing in (stats or {}).get("selector_timings", []):
            row = selectors.setdefault(timing["selector"], {"ms": 0.0, "attempts": 0, "matches": 0})
            row["ms"] += timing.get("elapsed (us)", 0) / 1000 / n
            row["attempts"] += timing.get("match_attempts", 0) / n
            row["matches"] += timing.get("match_count", 0) / n
    return {
        "turns": turns,
        "shape": shape,
        "typed_ok": typed_ok,
        "keys": n,
        "precheck_style_ms": round(pre, 1),
        "typing_s": round(typing_s, 1),
        "match_ms": sum(r["ms"] for r in selectors.values()),
        "has_ms": sum(r["ms"] for s, r in selectors.items() if ":has(" in s),
        "style_ms": 1000 * (m1["RecalcStyleDuration"] - m0["RecalcStyleDuration"]) / n,
        "lock_ms": lock_ms,
        "selectors": selectors,
    }


def pin_down(
    browser,
    base: str,
    budget_s: float = PIN_BUDGET_S,
) -> list[str]:
    """Find the :has() rules whose removal makes the keystroke cheap, one fresh page per probe.

    Untraced style recalc per key with every :has() rule on (E) and off (F). The cost is often not
    additive: one rule makes the browser restyle a whole subtree and the other rules then pay for
    it. So the search removes rules: bisect for a set whose removal brings the page back near F,
    narrowed to single rules. For each rule found it reports the page without that rule and the
    rule on its own. Up to three rules, time-boxed. Measurements, not a guess.
    """
    started = time.time()
    sheet = main_stylesheet(DIST)
    if sheet is None:
        return ["  pin-down: the built stylesheet has no :has() rules to test"]
    css_path, css_text = sheet
    scratch = browser.new_context()
    page = scratch.new_page()
    rules = page.evaluate(SHEET_RULES_JS, css_text)
    paths = [r["p"] for r in rules]
    every = list(range(len(paths)))
    probes = [0]

    def cost(enabled) -> float:
        on = set(enabled)
        probes[0] += 1
        body = page.evaluate(
            SHEET_WITHOUT_JS, [css_text, [p for i, p in enumerate(paths) if i not in on]]
        )
        c, pg, cd = open_thread(browser, base, PIN_TURNS, (css_path, body))
        try:
            return style_per_key(cd, pg, PIN_KEYS)
        finally:
            c.close()

    def without(removed) -> float:
        gone = set(removed)
        return cost([i for i in every if i not in gone])

    try:
        e_all, f_none = cost(every), cost([])
        out = [
            f"  pin-down at {PIN_TURNS} turns, untraced style recalc per keystroke: {e_all:.1f} ms with all "
            f"{len(paths)} :has() rules, {f_none:.1f} ms with none of them"
        ]
        if e_all < max(2 * f_none, f_none + 10):
            out.append(
                "    switching every :has() rule off does not remove the cost, so the cause is probably "
                "not a :has() rule; see the selectors above"
            )
            return out
        fixed = lambda ms: ms <= max(2 * f_none, f_none + 0.15 * (e_all - f_none))  # noqa: E731
        found: list[int] = []

        def search(cands: list[int], also_removed: list[int]) -> int | None:
            """One rule in `cands` whose removal (with `also_removed`) fixes the page."""
            extra = list(also_removed)
            while len(cands) > 1:
                if time.time() - started > budget_s:
                    return None
                first, second = cands[: len(cands) // 2], cands[len(cands) // 2 :]
                if fixed(without(found + extra + first)):
                    cands = first
                elif fixed(without(found + extra + second)):
                    cands = second
                else:
                    # needs something from both halves: keep the second out, find the first's part
                    extra += second
                    cands = first
            return cands[0] if cands else None

        # Rules that search all descendants of an element holding the thread go first: when
        # removing them fixes the page, the bisection only has to look among them.
        likely = [
            i
            for i, r in enumerate(rules)
            if any(thread_anchor(a) for a, _ in has_descendant(r["s"]))
        ]
        while len(found) < 3 and time.time() - started < budget_s:
            rest = [i for i in every if i not in found]
            if found and fixed(without(found)):
                break
            group = [i for i in likely if i in rest]
            if group and fixed(without(found + group)):
                i = search(group, [])
            else:
                i = search(rest, [])
            if i is None:
                break
            found.append(i)
        for i in found:
            minus, alone = without([i]), cost([i])
            out.append(
                f"  Measured: removing only this rule takes the keystroke from {e_all:.1f} to {minus:.1f} ms "
                f"(no :has() rules: {f_none:.1f} ms); this rule on its own costs {alone:.1f} ms:"
            )
            out.append(f"    {short(rules[i]['s'], 240)}")
            out += suspects_for(rules[i]["s"])
        if len(found) > 1:
            out.append(
                f"  removing all {len(found)} together: {without(found):.1f} ms per keystroke"
            )
        if not found:
            out.append(f"    no rule found within the {budget_s:.0f} s budget")
        out.append(f"  (pin-down: {probes[0]} probes, {time.time() - started:.0f} s)")
        return out
    finally:
        scratch.close()


def shape_problems(result: dict) -> list[str]:
    shape, turns = result["shape"], result["turns"]
    out = []
    missing = [k for k in ("reasoning", "sources", "code", "katex", "tables") if not shape[k]]
    if shape["messages"] < turns * 2 or not shape["workspace"] or missing:
        out.append(
            f"fixture did not render what the gate measures at {turns} turns: {shape['messages']} of "
            f"{turns * 2} messages, workspace fade {'present' if shape['workspace'] else 'MISSING'}, "
            f"missing content: {', '.join(missing) or 'none'}"
        )
    if not result["typed_ok"]:
        out.append(f"typed text was lost at {turns} turns")
    return out


def short(selector: str, width: int = 170) -> str:
    return selector if len(selector) <= width else selector[: width - 3] + "..."


def top_selectors(result: dict, only_has: bool = False) -> list[str]:
    nodes = max(1, result["shape"]["nodes"])
    rows = [(s, r) for s, r in result["selectors"].items() if not only_has or ":has(" in s]
    rows.sort(key = lambda x: x[1]["ms"], reverse = True)
    out = []
    for s, r in rows[:TOP]:
        share = r["attempts"] / nodes
        tag = (
            f"  [tried on {share:.0%} of the DOM: likely restyled because of another rule]"
            if share >= 0.5
            else ""
        )
        out.append(
            f"    {r['ms']:8.2f} ms/key  {r['attempts']:>10,.0f} match attempts/key  {short(s)}{tag}"
        )
    wide = sum(1 for r in result["selectors"].values() if r["attempts"] >= 0.5 * nodes)
    if wide >= 3:
        out.append(
            f"  {wide} selectors were tried against at least half of the {nodes:,} elements on every "
            "keystroke: the whole page is being restyled, so the slowest selectors above are probably "
            "victims rather than the cause. The pin-down below switches rules off to find the trigger."
        )
        return out
    return out + suspects([s for s, r in rows[:2] if r["ms"] > 1.0 and r["attempts"] < 0.5 * nodes])


def suspects_for(selector: str) -> list[str]:
    try:
        return explain(selector, indent = "    ")
    except Exception as exc:  # noqa: BLE001 - diagnosis must never hide the failure itself
        return [f"    (source lookup failed: {exc!r})"]


def suspects(selectors: list[str]) -> list[str]:
    """Where the slowest selectors probably come from: a text match, so a hint, never a verdict."""
    out = []
    for s in selectors:
        out.append(f"  where `{short(s, 120)}` may come from:")
        out += suspects_for(s)
    return out


def growth(small: dict, large: dict) -> list[str]:
    names = set(small["selectors"]) | set(large["selectors"])
    rows = []
    for s in names:
        a = small["selectors"].get(s, {"ms": 0.0, "attempts": 0})
        b = large["selectors"].get(s, {"ms": 0.0, "attempts": 0})
        rows.append((b["ms"] - a["ms"], s, a, b))
    rows.sort(reverse = True)
    out = [
        f"    +{d:7.2f} ms/key  {a['ms']:.2f} -> {b['ms']:.2f} ms/key, "
        f"{a['attempts']:,.0f} -> {b['attempts']:,.0f} attempts/key  {short(s)}"
        for d, s, a, b in rows[:TOP]
        if d > 0
    ]
    return out + suspects([s for d, s, a, b in rows[:2] if d > 1.0])


def report(
    runs: dict[int, list[dict]], failures: list[str], details: list[str], started: float
) -> int:
    def med(turns: int, key: str):
        vals = [r[key] for r in runs.get(turns, [])]
        return statistics.median(vals) if vals else None

    lines = [f"long-chat typing gate ({time.time() - started:.0f} s)"]
    for turns in (SMALL, LARGE):
        if runs.get(turns) and runs[turns][-1].get("extreme"):
            r = runs[turns][-1]
            lines.append(
                f"  {turns:>3} turns: style recalc {r['precheck_style_ms']:.1f} ms/key untraced, "
                f"{r['shape']['nodes']:,} DOM nodes (tracing skipped)"
            )
        elif runs.get(turns):
            lines.append(
                f"  {turns:>3} turns: selector matching {med(turns, 'match_ms'):.2f} ms/key, :has() "
                f"{med(turns, 'has_ms'):.2f} ms/key, style recalc {med(turns, 'style_ms'):.1f} ms/key "
                f"(tracing on; {runs[turns][-1]['precheck_style_ms']:.1f} ms/key untraced), "
                f"{runs[turns][0]['shape']['nodes']:,} DOM nodes, {runs[turns][-1]['keys']} keys traced "
                f"in {runs[turns][-1]['typing_s']} s, dialog/menu open or close "
                f"{med(turns, 'lock_ms'):.1f} ms style recalc"
            )
    print("\n".join(lines))
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        for d in details:
            print(d)
        print(f"  {HEALTHY}")
        return 1
    print(
        "PASS: selector work per keystroke does not grow with the chat, and opening a dialog or menu does not restyle it"
    )
    return 0


def main() -> int:
    started = time.time()
    build()
    if not (DIST / "smoke-thread-weight.html").exists():
        print(f"FAIL: no production build at {DIST}")
        return 1
    server, base = serve(DIST)
    runs: dict[int, list[dict]] = {SMALL: [], LARGE: []}
    failures: list[str] = []
    details: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args = chromium_launch_args())
            # Large first: a blown :has() budget is decisive on its own and stops the run here.
            large = measure(browser, base, LARGE)
            runs[LARGE].append(large)
            failures += shape_problems(large)
            if large.get("extreme"):
                pre = large["precheck_style_ms"]
                failures.append(
                    f"a keystroke costs {pre:.0f} ms of style recalc at {LARGE} turns (untraced); healthy "
                    f"builds measure 1-15 ms, so this is at least {pre / 15:.0f}x over. Selector tracing "
                    "was skipped: one traced key would take minutes"
                )
                print(f"FAIL (diagnosis follows): {failures[-1]}", flush = True)
                details += pin_down(browser, base)
                details.append("  " + HINT)
            elif large["has_ms"] > MAX_HAS_MS:
                print(
                    f"FAIL (diagnosis follows): :has() matching costs {large['has_ms']:.1f} ms per keystroke",
                    flush = True,
                )
                failures.append(
                    f":has() matching costs {large['has_ms']:.1f} ms per keystroke at {LARGE} turns, "
                    f"budget {MAX_HAS_MS:.1f} ms ({large['has_ms'] / MAX_HAS_MS:.1f}x over); "
                    f"total selector matching {large['match_ms']:.1f} ms/key"
                )
                details.append(f"  most expensive :has() selectors at {LARGE} turns:")
                details += top_selectors(large, only_has = True)
                details += pin_down(browser, base)
                details.append("  " + HINT)
            if not failures and large.get("lock_ms", 0.0) > MAX_LOCK_MS:
                failures.append(
                    f"opening or closing a dialog or menu costs {large['lock_ms']:.0f} ms of style recalc at "
                    f"{LARGE} turns, budget {MAX_LOCK_MS:.0f} ms ({large['lock_ms'] / MAX_LOCK_MS:.1f}x over). "
                    "Radix writes pointer-events: none on <body> and the thread inherits it, so every message "
                    "restyles. It is possible the explicit pointer-events on .aui-thread-viewport in index.css "
                    "(the modal-layer cover rules) was removed or overridden, but we cannot be sure"
                )
            if not failures:
                small = measure(browser, base, SMALL)
                runs[SMALL].append(small)
                failures += shape_problems(small)
            if not failures:

                def ratio() -> float:
                    a = statistics.median(r["match_ms"] for r in runs[SMALL])
                    b = statistics.median(r["match_ms"] for r in runs[LARGE])
                    return max(b, MATCH_FLOOR_MS) / max(a, MATCH_FLOOR_MS)

                if RECHECK_BAND[0] <= ratio() <= RECHECK_BAND[1]:
                    print(
                        f"ratio {ratio():.2f}x is near the {MAX_MATCH_RATIO}x limit; repeating",
                        flush = True,
                    )
                    runs[SMALL].append(measure(browser, base, SMALL))
                    runs[LARGE].append(measure(browser, base, LARGE))
                r = ratio()
                if r > MAX_MATCH_RATIO:
                    a = statistics.median(x["match_ms"] for x in runs[SMALL])
                    b = statistics.median(x["match_ms"] for x in runs[LARGE])
                    failures.append(
                        f"selector matching per keystroke grows with the chat: {a:.2f} ms/key at "
                        f"{SMALL} turns -> {b:.2f} ms/key at {LARGE} turns = {r:.2f}x for "
                        f"{LARGE / SMALL:.0f}x the turns (limit {MAX_MATCH_RATIO}x, +{b - a:.1f} ms/key)"
                    )
                    details.append(
                        f"  selectors whose cost grew most from {SMALL} to {LARGE} turns:"
                    )
                    details += growth(runs[SMALL][-1], runs[LARGE][-1])
                    details += pin_down(browser, base)
                    details.append("  " + HINT)
            browser.close()
    finally:
        server.shutdown()
    return report(runs, failures, details, started)


if __name__ == "__main__":
    raise SystemExit(main())
