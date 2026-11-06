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
switching :has() rules off in fresh pages (a bisection, time-boxed) and names the rule that costs
the time on its own, with the CSS file and line it probably comes from.
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
from _css_culprit import explain  # noqa: E402
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
# A ratio inside this band gets a second interleaved repeat before the verdict; outside it, one
# measurement is decisive (healthy runs read 1.00x, the regression 5.3-6.0x).
RECHECK_BAND = (1.5, 3.0)
# Keeps sub-millisecond noise from becoming a large ratio.
MATCH_FLOOR_MS = 2.0
HEALTHY = "healthy reference: 0.7-1.0 ms/key selector matching, 1.00x, 0.2 ms/key :has()"
KEYS = "the quick "
PRECHECK_KEYS = "ab"
# Untraced style recalc per key above this on the large chat means a regressed build (healthy:
# 1-15 ms; #16603: 140-600 ms); its traced window is then 2 keys, not 10, to fail fast.
SLOW_STYLE_MS = 60.0
# Rule bisection on a failure: a smaller chat, 3 keys per probe, a time budget.
PIN_TURNS = 100
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


RULES_JS = """() => {
  const out = [];
  const walk = (list, path) => {
    for (let i = 0; i < list.length; i++) {
      const r = list[i];
      const p = path.concat(i);
      if (typeof r.selectorText === 'string' && r.selectorText.includes(':has(')) out.push({p, s: r.selectorText});
      if (r.cssRules && r.cssRules.length) walk(r.cssRules, p);
    }
  };
  [...document.styleSheets].forEach((sheet, i) => { try { walk(sheet.cssRules, [i]); } catch (e) {} });
  return out;
}"""
DISABLE_JS = """(paths) => {
  for (const p of paths) {
    let list = document.styleSheets[p[0]].cssRules, rule = null;
    for (let k = 1; k < p.length; k++) { rule = list[p[k]]; list = rule.cssRules; }
    if (rule) rule.selectorText = '.__long_chat_gate_disabled__';
  }
  return paths.length;
}"""


def open_thread(
    browser,
    base: str,
    turns: int,
    disable: list | None = None,
):
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(f"{base}/smoke-thread-weight.html?shell=1", wait_until = "load")
    page.wait_for_function("() => !!window.__threadWeight", timeout = 60_000)
    if disable:
        # Before the thread mounts, so no element carries :has() state from the rule.
        page.evaluate(DISABLE_JS, disable)
    count = turns * 2
    page.evaluate("([n, o]) => window.__threadWeight.seed(n, o)", [count, SEED_OPTIONS])
    page.wait_for_function(
        "(n) => window.__threadWeight.messageCount() >= n", arg = count, timeout = 120_000
    )
    page.wait_for_timeout(1500 if disable is not None else 2500)
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
        "selectors": selectors,
    }


def pin_down(
    browser,
    base: str,
    budget_s: float = PIN_BUDGET_S,
) -> list[str]:
    """Find the :has() rules behind a slow keystroke by switching them off in fresh pages.

    Style recalc per key (untraced) with every :has() rule on (E) and off (F); then bisect: a
    group is guilty when switching on only that group (the rest off) costs well above F. Each
    rule found is switched off for the next search, up to three. Evidence, not a guess: the
    numbers say how much each rule costs on its own.
    """
    started = time.time()
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(f"{base}/smoke-thread-weight.html?shell=1", wait_until = "load")
    page.wait_for_function("() => !!window.__threadWeight", timeout = 60_000)
    rules = page.evaluate(RULES_JS)
    ctx.close()
    if not rules:
        return ["  pin-down: the built stylesheet has no :has() rules to test"]
    paths = [r["p"] for r in rules]

    def cost(enabled_idx: list[int]) -> float:
        on = set(enabled_idx)
        c, pg, cd = open_thread(
            browser, base, PIN_TURNS, [p for i, p in enumerate(paths) if i not in on]
        )
        try:
            return style_per_key(cd, pg, PIN_KEYS)
        finally:
            c.close()

    every = list(range(len(paths)))
    e_all, f_none = cost(every), cost([])
    out = [
        f"  pin-down at {PIN_TURNS} turns, untraced style recalc per keystroke: {e_all:.1f} ms with all "
        f"{len(paths)} :has() rules, {f_none:.1f} ms with none of them"
    ]
    if e_all < max(2 * f_none, f_none + 10):
        out.append(
            "    switching every :has() rule off does not remove the cost, so the cause is probably not "
            "a :has() rule; see the selectors above"
        )
        return out
    guilty = lambda ms: ms > max(2 * f_none, f_none + 0.25 * (e_all - f_none))  # noqa: E731
    found: list[int] = []
    remaining = every
    while len(found) < 3 and time.time() - started < budget_s:
        cand = list(remaining)
        while len(cand) > 1 and time.time() - started < budget_s:
            first, second = cand[: len(cand) // 2], cand[len(cand) // 2 :]
            if guilty(cost(first)):
                cand = first
            elif guilty(cost(second)):
                cand = second
            else:
                out.append(
                    f"    {len(cand)} rules are slow together but no half is slow alone; stopping there"
                )
                cand = []
        if len(cand) != 1:
            break
        i = cand[0]
        alone = cost([i])
        found.append(i)
        remaining = [j for j in remaining if j != i]
        rest = cost(remaining)
        out.append(
            f"  This rule on its own adds {alone - f_none:+.1f} ms per keystroke ({f_none:.1f} -> "
            f"{alone:.1f} ms); every other :has() rule together costs {rest:.1f} ms (all on: {e_all:.1f} ms):"
        )
        out.append(f"    {short(rules[i]['s'], 240)}")
        out += suspects_for(rules[i]["s"])
        if not guilty(rest):
            break
    if not found:
        out.append(f"    no single rule found within the {budget_s:.0f} s budget")
    out.append(f"  (pin-down took {time.time() - started:.0f} s)")
    return out


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
        if runs.get(turns):
            lines.append(
                f"  {turns:>3} turns: selector matching {med(turns, 'match_ms'):.2f} ms/key, :has() "
                f"{med(turns, 'has_ms'):.2f} ms/key, style recalc {med(turns, 'style_ms'):.1f} ms/key "
                f"(tracing on; {runs[turns][-1]['precheck_style_ms']:.1f} ms/key untraced), "
                f"{runs[turns][0]['shape']['nodes']:,} DOM nodes, {runs[turns][-1]['keys']} keys traced "
                f"in {runs[turns][-1]['typing_s']} s"
            )
    print("\n".join(lines))
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        for d in details:
            print(d)
        print(f"  {HEALTHY}")
        return 1
    print("PASS: selector work per keystroke does not grow with the chat")
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
            if large["has_ms"] > MAX_HAS_MS:
                failures.append(
                    f":has() matching costs {large['has_ms']:.1f} ms per keystroke at {LARGE} turns, "
                    f"budget {MAX_HAS_MS:.1f} ms ({large['has_ms'] / MAX_HAS_MS:.1f}x over); "
                    f"total selector matching {large['match_ms']:.1f} ms/key"
                )
                details.append(f"  most expensive :has() selectors at {LARGE} turns:")
                details += top_selectors(large, only_has = True)
                details += pin_down(browser, base)
                details.append("  " + HINT)
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
