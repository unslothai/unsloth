# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""WebKit leg of the long-chat gate: open/close time and longest frame per surface, short vs long chat.

The Chromium area matrix (_area_matrix.py) counts restyled elements and CPU from a trace, which WebKit
cannot give. Here each surface is timed in the page itself: from just before the trigger to the first
frame after the surface is visible (and back for closing), with the longest rAF frame inside that
window. Playwright's WebKit runs twice: with a macOS Safari user agent and platform, so the app takes
the macOS paths (the WebKitGTK long-thread windowing of #16854 stays off, as on a Mac), and with its
own Linux user agent (the Linux desktop app's WebKitGTK paths). Same seeded chat as the matrix, served
from the same production build with the API answered in-process.

Per (engine, appearance, surface): the median of REPS runs on one page at SHORT and at LONG turns.
Fails when the long chat is over RATIO x the short one AND over FLOOR_MS, or when its longest frame is
over FRAME_MS AND over FRAME_GROWTH x the short chat's (WebKit on a CI runner already draws 300-600 ms
frames mounting Settings at 1 turn: a mount cost, not a long-chat one); a failing cell is measured again
on fresh pages and the lower reading kept. The two engines run in parallel, one browser each. Today's
known WebKit costs are pinned in KNOWN (measured on main, with margin) so the gate holds that level
and fails on anything worse; each names the fix work, and tightening is a one-line change.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import _area_matrix as am

SHORT = int(os.environ.get("LONG_THREAD_WEBKIT_SHORT", "1"))
LONG = int(os.environ.get("LONG_THREAD_WEBKIT_LONG", "150"))
REPS = int(os.environ.get("LONG_THREAD_WEBKIT_REPS", "3"))
RATIO = float(os.environ.get("LONG_THREAD_WEBKIT_RATIO", "2.5"))
FLOOR_MS = float(os.environ.get("LONG_THREAD_WEBKIT_FLOOR_MS", "200"))
FRAME_MS = float(os.environ.get("LONG_THREAD_WEBKIT_FRAME_MS", "250"))
FRAME_GROWTH = float(os.environ.get("LONG_THREAD_WEBKIT_FRAME_GROWTH", "2.5"))
# A short-chat time below this counts as this, so a near-instant surface cannot make a large ratio.
SHORT_FLOOR_MS = 100.0
TIMEOUT_MS = 8_000
SETTLE_MS = 150

MAC_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/18.0 Safari/605.1.15"
)
ENGINES = {
    "webkit_mac": {"user_agent": MAC_UA, "platform": "MacIntel"},
    "webkit_linux": {},
}
MODES = ("default", am.GLASS)
# The surfaces Round 11 measured on real runners (plans/xplat_lag_status3.md), as _area_matrix AREAS.
SURFACES = [
    "browser_pane",
    "settings",
    "shortcuts_dialog",
    "command_palette",
    "composer_menu",
    "sidebar_toggle",
    "thread_scroll",
]

# Known WebKit long-chat costs on main, pinned so the gate holds today's level. Keys: engine:mode:surface:
# metric (open / close = long/short ratio of the time; frame = the long chat's longest frame, ms).
# Set from main readings with margin; lower each when its fix lands.
KNOWN_DEFAULT = {
    # Browser pane: opening or closing it narrows / widens the thread and WebKit re-lays out and repaints
    # the rows still inside the #16986 content-visibility window (macos-15 Safari engine, Round 11:
    # 460-562 ms at 150 turns vs 144-165 at 1). GitHub ubuntu runners, 6 readings: macOS UA open 1.5-3.0x
    # (frame 267-335 ms), close 3.6-7.2x / 3.6x over the 100 ms floor (frame 235-300); Linux UA open up
    # to 3.9x (Glass 6.4x, frame 327-453), close 2.9x (Glass 4.8x, frame 191-297). Fix work: branch
    # studio-browser-pane-open.
    "webkit_mac:default:browser_pane:open": 4.0,
    "webkit_mac:default:browser_pane:close": 5.0,
    "webkit_mac:default:browser_pane:open_frame": 450,
    "webkit_mac:default:browser_pane:close_frame": 400,
    "webkit_mac:liquid-3d:browser_pane:close": 7.0,
    "webkit_mac:liquid-3d:browser_pane:close_frame": 520,
    "webkit_linux:default:browser_pane:open": 5.0,
    "webkit_linux:default:browser_pane:close": 4.0,
    "webkit_linux:liquid-3d:browser_pane:open": 8.0,
    "webkit_linux:liquid-3d:browser_pane:close": 6.5,
    "webkit_linux:liquid-3d:browser_pane:open_frame": 600,
    "webkit_linux:liquid-3d:browser_pane:close_frame": 400,
    # Thread scroll: 20 scroll steps take 3.0-3.3x as long at 150 turns (macOS UA solid; Linux UA with
    # Glass, the WebKitGTK Glass scroll of Round 11: 12-38 of 67-132 frames over 50 ms on Linux).
    # Fix work: Linux WebKitGTK Glass scroll (GOAL_DETAILS open todo); macOS thread scroll paint.
    "webkit_mac:default:thread_scroll:open": 4.5,
    "webkit_linux:liquid-3d:thread_scroll:open": 4.5,
    # Composer menu and command palette (macOS UA, solid): a 237-297 / 182-268 ms frame opening them at
    # 150 turns: the paint behind the overlay grows with the thread (Round 11's Chromium probe: composer
    # menu paint 10 -> 80 ms, Settings 15 -> 104). Fix work: Settings / overlay paint (GOAL_DETAILS todo).
    "webkit_mac:default:composer_menu:open_frame": 400,
    "webkit_mac:default:command_palette:open_frame": 350,
}
KNOWN = {
    **KNOWN_DEFAULT,
    **{
        k: float(v)
        for k, _, v in (
            p.strip().partition("=")
            for p in os.environ.get("LONG_THREAD_WEBKIT_KNOWN", "").split(",")
            if p.strip()
        )
    },
}


@dataclass
class Leg:
    """One browser engine family's leg: what it runs, its limits and how its failures read. The WebKit
    leg is WEBKIT below; _firefox_leg.py builds FIREFOX the same way."""

    name: str  # in reports: "WebKit"
    browser: str  # Playwright browser type: "webkit", "firefox"
    env: str  # environment prefix: LONG_THREAD_WEBKIT (=1 runs it, _KNOWN, _REPS, ...)
    engines: dict  # engine name -> {"user_agent": ..., "platform": ...} (empty: the browser's own)
    surfaces: list
    known: dict
    who: str  # who pays a failure: "WebKit (Safari, the macOS app, the Linux desktop app)"
    modes: tuple = MODES
    short: int = SHORT
    long: int = LONG
    reps: int = REPS
    ratio: float = RATIO
    floor_ms: float = FLOOR_MS
    frame_ms: float = FRAME_MS
    frame_growth: float = FRAME_GROWTH


def leg_known(env: str, default: dict) -> dict:
    """KNOWN ceilings: the pinned defaults plus <env>_KNOWN=engine:mode:surface:metric=value,..."""
    return {
        **default,
        **{
            k: float(v)
            for k, _, v in (
                p.strip().partition("=")
                for p in os.environ.get(f"{env}_KNOWN", "").split(",")
                if p.strip()
            )
        },
    }


WEBKIT = Leg(
    name = "WebKit",
    browser = "webkit",
    env = "LONG_THREAD_WEBKIT",
    engines = ENGINES,
    surfaces = SURFACES,
    known = KNOWN,
    who = "WebKit (Safari, the macOS app, the Linux desktop app)",
)

ARM = """() => {
  const w = window.__wk = {t0: performance.now(), frames: [], last: performance.now(), stop: false};
  const tick = (t) => { w.frames.push([t, t - w.last]); w.last = t; if (!w.stop) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}"""
# Wait (in rAF steps) until `pred` holds, then one more frame: the time is that frame's, so it
# includes painting the surface. Returns [ms since ARM, longest frame in the window] or null.
UNTIL = """async ([pred, timeoutMs]) => {
  const w = window.__wk; const test = (0, eval)(pred);
  const frame = () => new Promise((r) => requestAnimationFrame(r));
  while (!test()) { if (performance.now() - w.t0 > timeoutMs) { w.stop = true; return null; } await frame(); }
  const t = await frame(); w.stop = true;
  const long = w.frames.filter(([ts]) => ts <= t + 1).reduce((m, [, d]) => Math.max(m, d), 0);
  return [t - w.t0, long];
}"""
# Surfaces judged on "settle" (typing): the frame clock plus a MutationObserver armed before the action,
# ignoring the autoscroll follow flag and the Glass motion hold flags (state, not content). The time
# is the last content mutation, once the page has been quiet for QUIET_MS.
QUIET_MS = 300
SETTLE_IGNORED = ["data-aui-following", "data-glass-moving", "data-glass-settling"]
ARM_SETTLE = """(ignored) => {
  const w = window.__wk = {t0: performance.now(), frames: [], last: performance.now(), stop: false, lastMut: null};
  const tick = (t) => { w.frames.push([t, t - w.last]); w.last = t; if (!w.stop) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  w.mo = new MutationObserver((recs) => { for (const r of recs) {
    if (r.type === 'attributes' && ignored.includes(r.attributeName)) continue; w.lastMut = performance.now(); } });
  w.mo.observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
}"""
SETTLED = """async ([quietMs, timeoutMs]) => {
  const w = window.__wk; const frame = () => new Promise((r) => requestAnimationFrame(r));
  while (performance.now() - (w.lastMut ?? w.t0) < quietMs) {
    if (performance.now() - w.t0 > timeoutMs) { w.mo.disconnect(); w.stop = true; return null; }
    await frame();
  }
  w.mo.disconnect(); w.stop = true; const end = w.lastMut ?? w.t0;
  // Frames that started by the last mutation, so the one rendering it counts even when it ends long after.
  const long = w.frames.filter(([ts, d]) => ts - d <= end + 20).reduce((m, [, d]) => Math.max(m, d), 0);
  return [end - w.t0, long];
}"""
# Thread scroll: the matrix's 20-step scroll; the time is the whole scroll, the frame its longest.
SCROLLED = """() => { const w = window.__wk; w.stop = true; const t = performance.now();
  return [t - w.t0, w.frames.reduce((m, [, d]) => Math.max(m, d), 0)]; }"""


def pred_js(pred, base_overlays: int, attr_before) -> str:
    kind = pred[0]
    if kind == "overlay":
        return f"() => window.__gateOverlays() > {base_overlays}"
    if kind == "overlay_gone":
        return f"() => window.__gateOverlays() <= {max(0, base_overlays - 1)} || window.__gateOverlays() === 0"
    if kind == "sel":
        return f"() => window.__gateVisible({json.dumps(pred[1])})"
    if kind == "gone":
        return f"() => !window.__gateVisible({json.dumps(pred[1])})"
    if kind == "changed":
        return (
            f"() => {{ const v = window.__gateAttr({json.dumps(pred[1])}, {json.dumps(pred[2])}); "
            f"return v !== null && v !== {json.dumps(attr_before)}; }}"
        )
    if kind == "restored":
        return f"() => window.__gateAttr({json.dumps(pred[1])}, {json.dumps(pred[2])}) === {json.dumps(attr_before)}"
    raise ValueError(f"no timing predicate for {kind}")


class WebKitPage(am.Page):
    """The matrix's page (same thread, API and controls) in a non-Chromium browser, without tracing."""

    def __init__(
        self,
        browser,
        base: str,
        turns: int,
        mode: dict,
        engine: str,
        engines = None,
    ):
        self.browser, self.turns, self.mode, self.tabs = browser, turns, mode, []
        self.api = am.Api(turns)
        spec = (engines or ENGINES).get(engine, {})
        extra = {"user_agent": spec["user_agent"]} if "user_agent" in spec else {}
        self.ctx = browser.new_context(viewport = {"width": 1440, "height": 900}, **extra)
        if spec.get("platform"):
            self.ctx.add_init_script(
                "Object.defineProperty(Navigator.prototype, 'platform', {get: () => %s});"
                % json.dumps(spec["platform"])
            )
        init = {
            "unsloth_auth_token": am.auth_token(),
            "unsloth_auth_refresh_token": "r",
            "unsloth_window_material": am.material_storage(mode["set"]),
            "theme": mode["set"]["theme"],
        }
        self.ctx.add_init_script(
            "(() => { const v = %s; for (const k in v) localStorage.setItem(k, v[k]); })()"
            % json.dumps(init)
        )
        self.ctx.route(lambda url: urlparse(url).path.startswith("/api/"), self.api.handle)
        self.ctx.route(
            lambda url: not url.startswith(base), lambda route: route.fulfill(status = 204, body = "")
        )
        self.page = page = self.ctx.new_page()
        self.errors, self.last_error = [], ""
        self.attr_before, self.focusins_before, self.hovered = None, 0, False
        page.goto(f"{base}/chat?thread={self.api.rec['id']}", wait_until = "load", timeout = 180_000)
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length >= n",
            arg = 2 * turns,
            timeout = 180_000,
        )
        page.evaluate(am.PROBE)
        page.wait_for_timeout(1_500)
        self.mac = page.evaluate(
            "() => /mac|iphone|ipad|ipod/i.test(`${navigator.platform || ''} ${navigator.userAgent || ''}`)"
        )
        page.mouse.move(1435, 450)

    def timed(self, action, pred, base_overlays: int):
        settle = pred is not None and pred[0] == "settle"
        if settle:
            self.page.evaluate(ARM_SETTLE, SETTLE_IGNORED)
        else:
            self.page.evaluate(ARM)
        if not self.act(action):
            self.page.evaluate("() => { window.__wk.stop = true; window.__wk.mo?.disconnect(); }")
            return None
        if pred is None:
            return self.page.evaluate(SCROLLED)
        if settle:
            return self.page.evaluate(SETTLED, [QUIET_MS, TIMEOUT_MS])
        return self.page.evaluate(
            UNTIL, [pred_js(pred, base_overlays, self.attr_before), TIMEOUT_MS]
        )

    def surface(self, name: str) -> dict:
        """One open (and close) of a surface: {open_ms, open_frame, close_ms, close_frame} or {error}."""
        spec = am.AREAS[name]
        page = self.page
        self.reset()
        page.wait_for_timeout(SETTLE_MS)
        if spec.get("setup"):  # typing: focus the composer first, untimed
            if not self.act(spec["setup"]):
                return {"error": "setup did not run"}
            page.wait_for_timeout(SETTLE_MS)
        base = page.evaluate("() => window.__gateOverlays()")
        ready = spec["ready"]
        self.attr_before = (
            page.evaluate("([s, a]) => window.__gateAttr(s, a)", [ready[1], ready[2]])
            if ready[0] == "changed"
            else None
        )
        scroll = spec["open"][0] == "scroll"
        got = self.timed(spec["open"], None if scroll else ready, base)
        if got is None:
            return {"error": f"never reached {ready[0]}"}
        out = {"open_ms": got[0], "open_frame": got[1]}
        if spec.get("close"):
            page.wait_for_timeout(SETTLE_MS)
            gone = self.timed(spec["close"], spec["gone"], base + 1)
            if gone is None:
                return {**out, "error": f"opened but never reached {spec['gone'][0]}"}
            out.update(close_ms = gone[0], close_frame = gone[1])
        if spec.get("teardown"):
            self.act(spec["teardown"])
        return out

    def measure(self, surfaces: list[str], reps: int) -> dict[str, dict]:
        cells: dict[str, dict] = {}
        for name in surfaces:
            runs, error = [], None
            for _ in range(reps):
                try:
                    r = self.surface(name)
                except Exception as exc:  # noqa: BLE001 - reported as a NOT_RUN surface
                    r = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                if "error" in r:
                    error = r["error"]
                    continue
                runs.append(r)
            if not runs:
                cells[name] = {"status": f"NOT_RUN: {error}"}
                continue
            cell = {"status": "OK", "reps": len(runs)}
            for key in ("open_ms", "open_frame", "close_ms", "close_frame"):
                vals = [r[key] for r in runs if key in r]
                if vals:
                    cell[key] = round(statistics.median(vals), 1)
            cells[name] = cell
        return cells


def _engine_worker(
    engine,
    dist_base,
    modes,
    surfaces,
    reps,
    out,
    log,
    leg = None,
) -> None:
    from playwright.sync_api import sync_playwright
    leg = leg or WEBKIT
    with sync_playwright() as pw:
        browser = getattr(pw, leg.browser).launch()
        try:
            for mode in modes:
                for turns in (leg.short, leg.long):
                    t = time.time()
                    try:
                        page = WebKitPage(browser, dist_base, turns, mode, engine, leg.engines)
                    except Exception as exc:  # noqa: BLE001
                        out[(engine, mode["name"], turns)] = {
                            "error": f"did not load: {str(exc)[:160]}"
                        }
                        continue
                    try:
                        out[(engine, mode["name"], turns)] = {"cells": page.measure(surfaces, reps)}
                    finally:
                        page.close()
                    log(
                        f"  {leg.name.lower()}: {engine} {mode['name']} {turns} turns ({time.time() - t:.0f} s)",
                        flush = True,
                    )
        finally:
            browser.close()


def measure(
    dist: Path,
    surfaces = None,
    modes = None,
    engines = None,
    reps = None,
    log = print,
    leg = None,
) -> dict:
    leg = leg or WEBKIT
    surfaces = surfaces or leg.surfaces
    modes = modes or leg.modes
    reps = reps or leg.reps
    by_name = {m["name"]: m for m in am.app_modes()}
    chosen = [by_name[m] for m in modes]
    server, base = am.serve_app(dist)
    out: dict = {}
    try:
        # One browser per engine (a browser per appearance as well doubles the frames on a 4-core runner).
        crashed: dict = {}

        def work(engine):
            try:
                _engine_worker(engine, base, chosen, surfaces, reps, out, log, leg = leg)
            except Exception as exc:  # noqa: BLE001  (recorded below: a crash must never read as a pass)
                crashed[engine] = f"{type(exc).__name__}: {str(exc)[:160]}"

        engines = engines or leg.engines
        threads = [threading.Thread(target = work, args = (e,)) for e in engines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.shutdown()
    # Every planned page gets a result: one a worker never recorded is a load failure.
    for e in engines:
        for m in chosen:
            for turns in (leg.short, leg.long):
                out.setdefault(
                    (e, m["name"], turns),
                    {
                        "error": f"not measured ({crashed.get(e, 'the engine worker stopped early')})"
                    },
                )
    return out


def judge(
    result: dict,
    known: dict | None = None,
    leg = None,
) -> list[dict]:
    leg = leg or WEBKIT
    known = leg.known if known is None else known
    SHORT, LONG = leg.short, leg.long  # noqa: N806 - the leg's lengths
    out = []
    for (engine, mode, turns), page in sorted(result.items()):
        if "error" in page:
            out.append(
                {
                    "engine": engine,
                    "mode": mode,
                    "surface": "*",
                    "metric": "load",
                    "fail": True,
                    "why": page["error"],
                }
            )
    for (engine, mode, turns), page in sorted(result.items()):
        if turns != LONG or "cells" not in page:
            continue
        short_page = result.get((engine, mode, SHORT)) or {}
        if "cells" not in short_page:
            continue  # its load failure is reported above
        short = short_page["cells"]
        for name in sorted(set(page["cells"]) | set(short)):
            lc, sc = page["cells"].get(name, {}), short.get(name, {})
            if lc.get("status") != "OK" or sc.get("status") != "OK":
                # A surface missing at either length is a gap in the gate, never a silent pass.
                out.append(
                    {
                        "engine": engine,
                        "mode": mode,
                        "surface": name,
                        "metric": "run",
                        "fail": True,
                        "why": f"{lc.get('status', 'missing')} at {LONG} turns, {sc.get('status', 'missing')} at {SHORT}",
                    }
                )
                continue
            for edge in ("open", "close"):
                lv, sv = lc.get(f"{edge}_ms"), sc.get(f"{edge}_ms")
                if lv is None or sv is None:
                    continue
                ratio = lv / max(sv, SHORT_FLOOR_MS)
                limit = known.get(f"{engine}:{mode}:{name}:{edge}", leg.ratio)
                out.append(
                    {
                        "engine": engine,
                        "mode": mode,
                        "surface": name,
                        "metric": edge,
                        "value": lv,
                        "base_value": sv,
                        "ratio": ratio,
                        "limit": limit,
                        "fail": ratio > limit and lv > leg.floor_ms,
                    }
                )
                fv = lc.get(f"{edge}_frame")
                if fv is not None:
                    flimit = known.get(f"{engine}:{mode}:{name}:{edge}_frame", leg.frame_ms)
                    sf = sc.get(f"{edge}_frame") or 0.0
                    out.append(
                        {
                            "engine": engine,
                            "mode": mode,
                            "surface": name,
                            "metric": f"{edge}_frame",
                            "value": fv,
                            "base_value": sf,
                            "limit": flimit,
                            "fail": fv > flimit and fv > leg.frame_growth * sf,
                        }
                    )
    return out


def lower(first: dict, again: dict) -> dict:
    """Merge a recheck: per cell, the lower of each reading (a recheck only ever lowers a reading)."""
    merged = dict(first)
    for key, page in again.items():
        if "cells" not in page:
            continue
        if "cells" not in merged.get(key, {}):
            merged[key] = page
            continue
        cells = dict(merged[key]["cells"])
        for name, cell in page["cells"].items():
            old = cells.get(name, {})
            if cell.get("status") != "OK":
                continue
            if old.get("status") != "OK":
                cells[name] = cell
                continue
            cells[name] = {
                **old,
                **{
                    k: min(old[k], v)
                    for k, v in cell.items()
                    if k.endswith(("_ms", "_frame")) and k in old
                },
            }
        merged[key] = {"cells": cells}
    return merged


def run(
    dist: Path,
    log = print,
    leg = None,
) -> tuple[list[str], list[str], dict]:
    """(report lines, failures, result)."""
    leg = leg or WEBKIT
    SHORT, LONG = leg.short, leg.long  # noqa: N806 - the leg's lengths
    started = time.time()
    result = measure(dist, log = log, leg = leg)
    verdicts = judge(result, leg = leg)
    failing = {
        (v["engine"], v["mode"], v["surface"])
        for v in verdicts
        if v["fail"] and v["surface"] != "*"
    }
    if failing:
        log(
            f"  {leg.name.lower()}: {len(failing)} cell(s) over their limit; measuring them again on fresh pages",
            flush = True,
        )
        for engine in sorted({f[0] for f in failing}):
            mode_names = sorted({f[1] for f in failing if f[0] == engine})
            surfaces = sorted({f[2] for f in failing if f[0] == engine})
            again = measure(
                dist, surfaces = surfaces, modes = mode_names, engines = [engine], log = log, leg = leg
            )
            result = lower(result, again)
        verdicts = judge(result, leg = leg)
    lines = table(result, verdicts, leg = leg)
    lines.append(
        f"  {leg.name.lower()} leg: {len(leg.engines)} engine(s) x {len(leg.modes)} appearances x {SHORT}/{LONG} turns, {time.time() - started:.0f} s"
    )
    failures = []
    for v in verdicts:
        if not v["fail"]:
            continue
        where = f"{v['engine']}, {v['mode']}"
        if v["metric"] in ("load", "run"):
            failures.append(f"{leg.name} {where}: {v['surface']} {v['why']}")
        elif v["metric"].endswith("frame"):
            failures.append(
                f"{leg.name} {where}: {v['surface']} ({v['metric'].replace('_', ' ')}) has a {v['value']:.0f} ms frame at "
                f"{LONG} turns against {v['base_value']:.0f} ms at {SHORT} (limit {v['limit']:.0f} ms and {leg.frame_growth:g}x; {leg.env}_KNOWN {v['engine']}:{v['mode']}:{v['surface']}:{v['metric']}=...)"
            )
        else:
            failures.append(
                f"{leg.name} {where}: {'opening' if v['metric'] == 'open' else 'closing'} {v['surface']} takes {v['base_value']:.0f} ms at {SHORT} turns -> "
                f"{v['value']:.0f} ms at {LONG} turns = {v['ratio']:.1f}x (limit {v['limit']:.1f}x over {leg.floor_ms:.0f} ms; "
                f"{leg.env}_KNOWN {v['engine']}:{v['mode']}:{v['surface']}:{v['metric']}=...). It costs more "
                f"as the chat grows in {leg.who}: usually re-layout or "
                "repaint of the whole thread behind the surface"
            )
    return lines, failures, {"result": result, "verdicts": verdicts}


def table(
    result: dict,
    verdicts: list[dict],
    leg = None,
) -> list[str]:
    leg = leg or WEBKIT
    SHORT, LONG = leg.short, leg.long  # noqa: N806 - the leg's lengths
    by = {(v["engine"], v["mode"], v["surface"], v["metric"]): v for v in verdicts}
    lines = [
        f"  {leg.name}: open / close ms at {SHORT} -> {LONG} turns, longest frame at {LONG} (! = over its limit)"
    ]
    for (engine, mode, turns), page in sorted(result.items()):
        if turns != LONG:
            continue
        if "error" in page:
            lines.append(f"    {engine:<13}{mode:<10} {page['error']}")
            continue
        short = (result.get((engine, mode, SHORT)) or {}).get("cells", {})
        for name, lc in page["cells"].items():
            sc = short.get(name, {})
            if lc.get("status") != "OK":
                lines.append(f"    {engine:<13}{mode:<10}{name:<18}{lc.get('status')}")
                continue
            parts = []
            for edge in ("open", "close"):
                if f"{edge}_ms" not in lc:
                    continue
                v = by.get((engine, mode, name, edge))
                f = by.get((engine, mode, name, f"{edge}_frame"))
                parts.append(
                    f"{edge} {sc.get(f'{edge}_ms', float('nan')):>5.0f} -> {lc[f'{edge}_ms']:>5.0f}"
                    + (f" ({v['ratio']:.1f}x{'!' if v['fail'] else ''})" if v else "")
                    + f" frame {lc.get(f'{edge}_frame', 0):>4.0f}{'!' if f and f['fail'] else ''}"
                )
            lines.append(f"    {engine:<13}{mode:<10}{name:<18}" + "   ".join(parts))
    return lines


def main(leg = None) -> int:
    """Frontend CI's WebKit (or Firefox) job: the leg alone, on the long-thread production build."""
    leg = leg or WEBKIT
    if os.environ.get(leg.env, "0") != "1":
        print(f"{leg.name} leg: off (set {leg.env}=1)")
        return 0
    repo = Path(__file__).resolve().parents[2]
    dist = Path(
        os.environ.get("LONG_THREAD_DIST", repo / ".playwright-cli" / "long-thread-production")
    )
    if not (dist / "index.html").exists():
        print(f"FAIL: no app build at {dist / 'index.html'} for the {leg.name} leg")
        return 1
    lines, fails, _ = run(dist, leg = leg)
    for line in lines:
        print(line)
    for f in fails:
        print(f"FAIL: {f}")
    print(f"{leg.name} leg: " + (f"{len(fails)} failure(s)" if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
