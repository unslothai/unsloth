# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The overlay rail's shadow gutter must cost the cards nothing.

The rail reserves room around its cards so its `overflow-y-auto` clip does not cut their
shadows off, and pays for it by dropping its box and growing its cap by the same amounts. All
of that lives in the browser -- an inline style, arithmetic on the hook's output, and a
`getComputedStyle` read -- so the node suite cannot reach it, and the suites that boot the app
assert the horizontal gutter and the cap but never the block one.

Drives the real `useStackGeometry` against a bare vite entry. What it holds:

  - the reserved room is in pixels at any root font size, so a rem cannot move the cards;
  - the rail's edge and cap carry the gutter, and the cards' band is unchanged by it;
  - the clip really has that much room around the cards afterwards;
  - both `scrollHeight` probes see the padding exactly once;
  - a rail sized against a cap it overflows still says so, even where an engine leaves the
    block-end padding out of its scrollable overflow;
  - the taller pointer box takes no input that was not already the rail's.

Run: SMOKE_PORT, SMOKE_BASE_URL, PW_ART_DIR and STUDIO_PLAYWRIGHT_BROWSER, as the other
smokes take them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    echo_browser_errors,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5197"))
BASE_URL = os.environ.get("SMOKE_BASE_URL", "").rstrip("/")
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-overlay-rail"))
ENTRY = "smoke-overlay-rail.html"
# Vite's SPA fallback answers 200 for a path it cannot resolve, so readiness is matched on
# the module specifier, not the status.
ENTRY_MODULE = "smoke-overlay-rail-main.tsx"
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
    raise SystemExit(f"unknown STUDIO_PLAYWRIGHT_BROWSER: {PLAYWRIGHT_BROWSER!r}")

# `right-4` and `gap-2` resolve through `--spacing` in rem, so the driver reads both off the
# node rather than mirroring them here: a root size other than 16 is measured, not assumed.
STORE_STACK_GAP = 8
# `STACK_GAP`, the clearance the placement leaves between the cards and a box it dodges.
CARD_TO_MONITOR_GAP = 8
# Layout is fractional under zoom and at DPR != 1, so equality is to within a CSS pixel.
EPS = 1.0

failures: list[str] = []


def check(ok: bool, message: str) -> bool:
    if not ok:
        failures.append(message)
    return ok


def info(message: str) -> None:
    print(f"  {message}", flush = True)


MEASURE = """
() => {
  const rail = document.querySelector('[data-testid="overlay-rail"]');
  if (!rail) return null;
  const style = getComputedStyle(rail);
  const num = (v) => Number.parseFloat(v) || 0;
  // The hook counts only the cards still in the rail's flow; a dragged one is not one.
  const flow = [...rail.children].filter((c) => {
    const p = getComputedStyle(c).position;
    return p !== "fixed" && p !== "absolute";
  });
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      top: r.top, bottom: r.bottom, left: r.left, right: r.right,
      width: r.width, height: r.height,
    };
  };
  const surfaces = flow.map((c) => c.querySelector("[data-card-surface]") || c);
  return {
    padding: {
      top: num(style.paddingTop), bottom: num(style.paddingBottom),
      left: num(style.paddingLeft), right: num(style.paddingRight),
    },
    margin: { left: num(style.marginLeft), right: num(style.marginRight) },
    // Both spacing utilities, read off the node rather than mirrored from a constant.
    rowGap: num(style.rowGap),
    rightInset: num(style.right),
    maxHeight: num(style.maxHeight),
    styleBottom: num(rail.style.bottom),
    pointerEvents: style.pointerEvents,
    rail: box(rail),
    first: box(flow[0]),
    last: box(flow[flow.length - 1]),
    firstSurface: box(surfaces[0]),
    lastSurface: box(surfaces[surfaces.length - 1]),
    cardCount: flow.length,
    cardHeights: flow.map((c) => c.offsetHeight),
    scrollHeight: rail.scrollHeight,
    clientHeight: rail.clientHeight,
    scrollTop: rail.scrollTop,
    geometry: window.__railSmoke.geometry(),
    gutter: window.__railSmoke.gutter(),
    innerHeight: window.innerHeight,
    innerWidth: window.innerWidth,
    dpr: window.devicePixelRatio,
    rootFontSize: num(getComputedStyle(document.documentElement).fontSize),
    dir: document.documentElement.dir || "ltr",
    errors: window.__railSmoke.errors(),
  };
}
"""

# The rail measures itself and changes its own cap, so read after the observers go quiet
# rather than after a fixed wait.
SETTLE = """
async () => {
  const rail = document.querySelector('[data-testid="overlay-rail"]');
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  let last = "";
  for (let i = 0; i < 40; i += 1) {
    await frame();
    const now = [rail.style.bottom, rail.style.maxHeight, rail.scrollHeight,
                 rail.clientHeight, rail.children.length].join("|");
    if (now === last) return { settledAfter: i, state: now };
    last = now;
  }
  return { settledAfter: -1, state: last };
}
"""


def configure(
    page,
    *,
    cards,
    obstacles = None,
    root_font = 16,
    direction = "ltr",
    theme = "dark",
) -> dict:
    """Put the page in one state of the matrix and wait for the rail to stop moving."""
    page.evaluate(
        """([opts]) => {
      document.documentElement.dir = opts.direction;
      document.documentElement.style.fontSize = opts.rootFont + "px";
      document.documentElement.classList.toggle("dark", opts.theme === "dark");
      document.documentElement.classList.toggle("light", opts.theme !== "dark");
      for (const who of ["monitor", "composer"]) window.__railSmoke.clear(who);
      for (const [who, frame] of Object.entries(opts.obstacles || {})) {
        window.__railSmoke.publish(who, frame);
      }
      window.__railSmoke.setCards(opts.cards);
    }""",
        [
            {
                "cards": cards,
                "obstacles": obstacles or {},
                "rootFont": root_font,
                "direction": direction,
                "theme": theme,
            }
        ],
    )
    page.evaluate(SETTLE)
    return page.evaluate(MEASURE)


def report_gap(m: dict) -> None:
    """The rail's own gap against the one the store's floor arithmetic assumes.

    `gap-2` is a rem and `STACK_GAP` a literal 8, so they agree only at a 16px root. That
    predates the gutter and is not gated here, but it is the drift the gutter was pinned in
    pixels to avoid, so it is printed rather than left to be rediscovered.
    """
    if abs(m["rowGap"] - STORE_STACK_GAP) > 0.01:
        info(
            f"note: at a {m['rootFontSize']:.0f}px root the rail's gap is {m['rowGap']:.1f}px "
            f"while STACK_GAP is {STORE_STACK_GAP}; pre-existing, unrelated to the gutter"
        )


def assert_invariants(m: dict, label: str, *, fitting: bool) -> None:
    """The properties that must hold in every state, whatever else the case is testing."""
    if not check(m is not None, f"{label}: the rail is not in the page"):
        return
    top, bottom = m["gutter"]["top"], m["gutter"]["bottom"]

    # In pixels: a spacing utility would be a rem and disagree with the compensation.
    check(
        abs(m["padding"]["top"] - top) < 0.01 and abs(m["padding"]["bottom"] - bottom) < 0.01,
        f"{label}: block padding is {m['padding']['top']}/{m['padding']['bottom']}, "
        f"reserved {top}/{bottom}; a rem has got into the gutter",
    )
    # Across, the utility and its negative margin cancel whatever a rem is worth.
    check(
        abs(m["padding"]["left"] + m["margin"]["left"]) < 0.01
        and abs(m["padding"]["right"] + m["margin"]["right"]) < 0.01,
        f"{label}: the horizontal gutter no longer cancels: "
        f"{m['padding']['left']}/{m['margin']['left']}",
    )
    # The rail gives up exactly what it takes back.
    check(
        abs((m["styleBottom"] + bottom) - m["geometry"]["bottom"]) < 0.01,
        f"{label}: rail bottom {m['styleBottom']} + {bottom} is not the placed inset "
        f"{m['geometry']['bottom']}; the cards have moved",
    )
    check(
        abs((m["maxHeight"] - top - bottom) - m["geometry"]["maxHeight"]) < 0.01,
        f"{label}: cap {m['maxHeight']} less the gutter is not the placed band "
        f"{m['geometry']['maxHeight']}; the gutter is being taken out of the cards",
    )
    # The clip is at the padding box, so this is the room the shadows have. Only at rest: a
    # scrolling rail has cards below the fold, and run_observer_and_scroll covers that end.
    if fitting and m["cardCount"] > 0:
        check(
            m["rail"]["bottom"] - m["last"]["bottom"] >= bottom - 0.6,
            f"{label}: {m['rail']['bottom'] - m['last']['bottom']:.2f}px under the bottom "
            f"card, reserved {bottom}",
        )
        check(
            m["first"]["top"] - m["rail"]["top"] >= top - 0.6,
            f"{label}: {m['first']['top'] - m['rail']['top']:.2f}px over the top card, "
            f"reserved {top}",
        )
    # A cap that grew by the gutter must not push the rail off the viewport.
    check(
        m["rail"]["top"] >= -0.5 and m["rail"]["bottom"] <= m["innerHeight"] + 0.5,
        f"{label}: the rail is off screen: {m['rail']['top']:.1f}..{m['rail']['bottom']:.1f} "
        f"in {m['innerHeight']}",
    )
    # Both probes read the padding box, so the cards' own total is what is left after it.
    if m["cardCount"] > 0:
        cards_total = sum(m["cardHeights"]) + m["rowGap"] * (m["cardCount"] - 1)
        measured = m["scrollHeight"] - m["padding"]["top"] - m["padding"]["bottom"]
        check(
            abs(measured - cards_total) <= EPS,
            f"{label}: discounted scrollHeight {measured} is not the cards' {cards_total}; "
            f"the padding is counted a number of times other than once",
        )
    if fitting and m["cardCount"] > 0 and m["dir"] == "ltr":
        # The block gutter may not touch where the cards sit across.
        check(
            abs((m["innerWidth"] - m["last"]["right"]) - m["rightInset"]) <= EPS,
            f"{label}: the cards' right inset is "
            f"{m['innerWidth'] - m['last']['right']:.2f}, but the rail's is "
            f"{m['rightInset']:.2f}",
        )
    check(not m["errors"], f"{label}: page errors {m['errors']}")


BANNER = {"height": 160, "floor": 160, "dismissible": True}
PANEL = {"height": 120, "floor": 60}
INDICATOR = {"height": 64, "floor": 64}

VIEWPORTS = [
    (1280, 800),
    (1440, 900),
    (1920, 1080),
    (1024, 768),
    (921, 534),
    (768, 500),
    (390, 844),
    (640, 480),
]
ROOT_FONTS = [12, 14, 16, 20, 24]
DECKS = {
    "empty": [],
    "one": [INDICATOR],
    "banner+indicator": [BANNER, INDICATOR],
    "full": [BANNER, dict(BANNER), PANEL, INDICATOR],
    "dragged": [BANNER, PANEL, {**INDICATOR, "dragged": True}],
}


def obstacles_for(name: str, width: int, height: int) -> dict:
    """The three shapes that publish a frame, in the stack's own column."""
    if name == "none":
        return {}
    if name == "composer":
        return {
            "composer": {
                "left": width // 2 - 300,
                "right": width // 2 + 300,
                "top": height - 180,
                "bottom": height - 24,
                "coverable": True,
            }
        }
    if name == "monitor-short":
        return {
            "monitor": {
                "left": width - 288,
                "right": width - 16,
                "top": height - 320,
                "bottom": height - 200,
            }
        }
    if name == "monitor-tall":
        # Too tall to lift over, so the stack seats inside the panel instead.
        return {
            "monitor": {
                "left": width - 288,
                "right": width - 16,
                "top": 20,
                "bottom": height - 20,
            }
        }
    if name == "both":
        out = obstacles_for("monitor-short", width, height)
        out.update(obstacles_for("composer", width, height))
        return out
    raise AssertionError(name)


OBSTACLES = ["none", "composer", "monitor-short", "monitor-tall", "both"]


def run_geometry_matrix(page) -> None:
    """Every combination of viewport, root size, deck, obstacle and writing direction."""
    print("[matrix] geometry", flush = True)
    seen = 0
    noted_gaps: set[int] = set()
    for width, height in VIEWPORTS:
        page.set_viewport_size({"width": width, "height": height})
        for root_font in ROOT_FONTS:
            for deck_name, deck in DECKS.items():
                for obstacle in OBSTACLES:
                    for direction in ("ltr", "rtl"):
                        label = (
                            f"{width}x{height} root{root_font} {deck_name} "
                            f"{obstacle} {direction}"
                        )
                        m = configure(
                            page,
                            cards = deck,
                            obstacles = obstacles_for(obstacle, width, height),
                            root_font = root_font,
                            direction = direction,
                        )
                        fitting = bool(m) and not m["geometry"]["overflowing"]
                        assert_invariants(m, label, fitting = fitting)
                        if m and m["cardCount"] > 1 and root_font not in noted_gaps:
                            noted_gaps.add(root_font)
                            report_gap(m)
                        seen += 1
    info(f"{seen} states")


def run_cap_boundary(page) -> None:
    """Crossing the cap must flip the rail's overflow state once, not oscillate over it."""
    print("[matrix] cap boundary", flush = True)
    page.set_viewport_size({"width": 1280, "height": 800})
    probe = configure(page, cards = [INDICATOR])
    room = probe["geometry"]["maxHeight"]
    transitions = 0
    previous = None
    for step in range(-16, 17):
        height = room + step * 0.25
        m = configure(page, cards = [{"height": height, "floor": height}])
        state = m["geometry"]["overflowing"]
        if previous is not None and state != previous:
            transitions += 1
        previous = state
        assert_invariants(m, f"cap{step * 0.25:+.2f}", fitting = not state)
    check(
        transitions <= 1,
        f"the rail changed its mind about overflowing {transitions} times across the "
        f"boundary; a fractional reading is flapping",
    )
    # And well past it, where it must certainly scroll and take input.
    for over in (8, 10, 16, 24, 40):
        height = room + over
        m = configure(page, cards = [{"height": height, "floor": height}])
        check(
            m["geometry"]["overflowing"],
            f"a card {over}px past the cap does not make the rail scroll",
        )
        check(
            m["pointerEvents"] == "auto",
            f"a scrolling rail is {m['pointerEvents']}, so its scrollbar cannot be reached",
        )
    info(f"{transitions} transition(s) across the boundary")


def run_short_cap(page) -> None:
    """A cap that lands a little short of a card must show it whole, not shear its corners.

    The clip is at the padding box, so the gutter is slack: a card overrunning the band by
    less than the room reserved under it is still painted in full. The sheared-corner half of
    the report.
    """
    print("[matrix] a cap short of the card", flush = True)
    # No obstacle, so the band is the viewport less both insets and the shortfall is exact.
    height = 400
    page.set_viewport_size({"width": 1280, "height": height})
    band = height - 2 * 16
    for short in (0, 4, 10, 16, 24, 40):
        card = band + short
        m = configure(page, cards = [{"height": card, "floor": card}])
        sliced = page.evaluate(
            """() => {
          const rail = document.querySelector('[data-testid="overlay-rail"]');
          const card = rail.querySelector("[data-card-index]");
          const c = card.getBoundingClientRect();
          const r = rail.getBoundingClientRect();
          return {
            sliced: Math.round((Math.max(0, r.top - c.top)
                              + Math.max(0, c.bottom - r.bottom)) * 100) / 100,
            // What is left for the shadow once the card has spent the slack.
            room: Math.round((r.bottom - c.bottom) * 100) / 100,
            band: r.bottom - r.top,
          };
        }"""
        )
        # Cards sit against the top of the band, so the slack is the room under them only.
        reserved = m["gutter"]["bottom"]
        info(
            f"cap short by {short:2d}: {sliced['sliced']:5.2f}px of the card is clipped, "
            f"{sliced['room']:5.2f}px left under it "
            f"(cap {m['geometry']['maxHeight']}, card {card}, {reserved}px reserved)"
        )
        # Spent, not free: overrunning the band by k eats k of the room under the card.
        check(
            abs(sliced["room"] - (reserved - short)) <= 0.6,
            f"a cap {short}px short leaves {sliced['room']:.2f}px under the card, "
            f"expected {reserved - short}",
        )
        if short <= reserved:
            check(
                sliced["sliced"] <= 0.6,
                f"a cap {short}px short of the card slices {sliced['sliced']:.2f}px off it, "
                f"with {reserved}px reserved under it",
            )
        else:
            check(
                abs(sliced["sliced"] - (short - reserved)) <= 0.6,
                f"a cap {short}px short slices {sliced['sliced']:.2f}px, expected "
                f"{short - reserved}",
            )


def run_stale_block_end_padding(page) -> None:
    """An engine that leaves the block-end padding out of its scrollable overflow.

    Chromium always reserved it and Gecko now does, but the CSSWG settled it after both had
    shipped and a Linux build runs whatever WebKitGTK the distribution carries. Under that
    reading `collapsed` is 16px short and the rail would stop scrolling, so the live
    `scrollHeight > clientHeight` fallback beside it is what has to hold.
    """
    print("[matrix] block-end padding left out of the overflow region", flush = True)
    page.set_viewport_size({"width": 1280, "height": 800})
    room = configure(page, cards = [INDICATOR])["geometry"]["maxHeight"]
    tall = room + 120
    page.evaluate(
        """() => {
      const proto = Element.prototype;
      const original = Object.getOwnPropertyDescriptor(proto, "scrollHeight");
      window.__restoreScrollHeight = () =>
        Object.defineProperty(proto, "scrollHeight", original);
      Object.defineProperty(proto, "scrollHeight", {
        configurable: true,
        get() {
          const raw = original.get.call(this);
          if (this.dataset && this.dataset.testid === "overlay-rail") {
            const pad = Number.parseFloat(getComputedStyle(this).paddingBottom) || 0;
            return Math.max(0, raw - pad);
          }
          return raw;
        },
      });
    }"""
    )
    try:
        m = configure(page, cards = [{"height": tall, "floor": tall}])
        check(
            m["geometry"]["overflowing"],
            "with the block-end padding out of scrollHeight the rail stops scrolling, so "
            "the cards below the fold cannot be reached",
        )
        check(
            m["pointerEvents"] == "auto",
            "the rail lost its scrollbar under the older overflow reading",
        )
    finally:
        page.evaluate("() => window.__restoreScrollHeight()")


HIT = """
([spec]) => {
  const rail = document.querySelector('[data-testid="overlay-rail"]');
  const r = rail.getBoundingClientRect();
  const name = (el) => {
    if (!el) return null;
    if (el === rail) return "rail";
    if (el.closest && el.closest('[data-testid="overlay-rail"]')) return "card";
    if (el.dataset && el.dataset.testid) return el.dataset.testid;
    return el.tagName;
  };
  const x = Math.round((r.left + r.right) / 2);
  const rows = {};
  // Every pixel row of both gutters, and the two rows just outside them.
  for (let dy = -2; dy <= spec.top + 1; dy += 1) {
    rows["top" + dy] = name(document.elementFromPoint(x, Math.round(r.top) + dy));
  }
  for (let dy = -1; dy <= spec.bottom + 2; dy += 1) {
    rows["bottom" + dy] = name(document.elementFromPoint(x, Math.round(r.bottom) - dy));
  }
  return { x, rows, rail: {top: r.top, bottom: r.bottom, left: r.left, right: r.right} };
}
"""


def run_hit_testing(page) -> None:
    """The taller box must take no input a click-through rail would have let past."""
    print("[matrix] hit testing", flush = True)
    page.set_viewport_size({"width": 1280, "height": 800})
    gutter = page.evaluate("() => window.__railSmoke.gutter()")

    m = configure(page, cards = [INDICATOR])
    check(
        m["pointerEvents"] == "none",
        f"a rail that fits is {m['pointerEvents']}, so it is taking clicks it should pass on",
    )
    hit = page.evaluate(HIT, [gutter])
    check(
        "rail" not in hit["rows"].values(),
        f"a fitting rail answers a click in its gutter: {hit['rows']}",
    )

    # Seated inside a tall monitor: the panel outranks the rail, so however far down the
    # box goes the gutter cannot reach the panel's grip.
    room = configure(page, cards = [INDICATOR])["geometry"]["maxHeight"]
    m = configure(
        page,
        cards = [{"height": room + 200, "floor": room + 200}],
        obstacles = obstacles_for("monitor-tall", 1280, 800),
    )
    check(m["geometry"]["overflowing"], "the deck meant to overflow does not")
    grip = page.evaluate(
        """([grip]) => {
      const rail = document.querySelector('[data-testid="overlay-rail"]');
      const panel = document.querySelector('[data-testid="obstacle-monitor"]');
      const p = panel.getBoundingClientRect();
      const r = rail.getBoundingClientRect();
      const x = Math.round(p.right - grip / 2);
      const answers = [];
      for (let y = Math.round(p.bottom) - grip; y < Math.round(p.bottom); y += 1) {
        const el = document.elementFromPoint(x, y);
        answers.push(el === rail ? "rail"
          : el && el.dataset && el.dataset.testid ? el.dataset.testid
          : el ? el.tagName : null);
      }
      return {
        answers,
        railOverlapsBand: r.bottom > Math.round(p.bottom) - grip && r.right > p.left,
      };
    }""",
        [16],
    )
    check(
        "rail" not in grip["answers"],
        f"the rail's gutter answers over the monitor's resize grip: {grip['answers']}",
    )
    info(
        f"gutter overlaps the grip band: {grip['railOverlapsBand']}, "
        f"answers: {sorted(set(grip['answers']))}"
    )

    # Emptied after scrolling: its box is the gutter now, and must not hold the corner.
    m = configure(page, cards = [])
    hit = page.evaluate(HIT, [gutter])
    check(
        "rail" not in hit["rows"].values() and "card" not in hit["rows"].values(),
        f"an emptied rail still answers clicks in the corner: {hit['rows']}",
    )

    run_reach_and_capture(page)


# What the rail's box can land on unprotected: the eight resize targets and the composer.
CAPTURE_TARGETS = [
    "resize-north",
    "resize-south",
    "resize-west",
    "resize-east",
    "resize-northwest",
    "resize-northeast",
    "resize-southwest",
    "resize-southeast",
    "obstacle-composer",
]
# Measured the same way, but against the grips rather than against the rail.
CONTROLS = ["control-minimize", "control-maximize", "control-close"]

CONTROL_LOSS = """
([ids]) => {
  // Only pixels taken by a grip or by the rail count. A rounded button's own corners
  // answer as its container, which is the button's shape rather than a lost hit area.
  const stolen = (el) =>
    Boolean(el) && (
      el.closest('[data-testid="overlay-rail"]') !== null ||
      (el.dataset && typeof el.dataset.testid === "string"
        && el.dataset.testid.startsWith("resize-"))
    );
  const out = {};
  for (const id of ids) {
    const node = document.querySelector(`[data-testid="${id}"]`);
    if (!node) continue;
    const b = node.getBoundingClientRect();
    let lost = 0, total = 0;
    for (let x = Math.ceil(b.left); x < b.right; x += 1) {
      for (let y = Math.ceil(b.top); y < b.bottom; y += 1) {
        total += 1;
        if (stolen(document.elementFromPoint(x, y))) lost += 1;
      }
    }
    out[id] = { lost, total };
  }
  return out;
}
"""

REACH = """
([ids]) => {
  const rail = document.querySelector('[data-testid="overlay-rail"]');
  const r = rail.getBoundingClientRect();
  const mine = (el) =>
    el === rail || Boolean(el && el.closest && el.closest('[data-testid="overlay-rail"]'));
  const out = {};
  for (const id of ids) {
    const node = document.querySelector(`[data-testid="${id}"]`);
    if (!node) continue;
    const g = node.getBoundingClientRect();
    // Geometric reach first, so a target the rail lands on is visible in the report even
    // when a layer above the rail is what stops it taking the pixels.
    const overlapW = Math.max(0, Math.min(g.right, r.right) - Math.max(g.left, r.left));
    const overlapH = Math.max(0, Math.min(g.bottom, r.bottom) - Math.max(g.top, r.top));
    let taken = 0, total = 0;
    for (let x = Math.ceil(g.left); x < g.right; x += 1) {
      for (let y = Math.ceil(g.top); y < g.bottom; y += 1) {
        total += 1;
        if (mine(document.elementFromPoint(x, y))) taken += 1;
      }
    }
    out[id] = {
      reach: Math.round(overlapW * overlapH),
      taken, total,
      box: {top: g.top, bottom: g.bottom, left: g.left, right: g.right},
    };
  }
  out.__rail = {top: r.top, bottom: r.bottom, left: r.left, right: r.right};
  return out;
}
"""


def report_capture(page, label: str) -> dict:
    """Which unprotected surfaces the rail's box lands on, and which it actually takes."""
    seen = page.evaluate(REACH, [CAPTURE_TARGETS])
    rail = seen.pop("__rail")
    reached = {k: v for k, v in seen.items() if v["reach"] > 0}
    info(
        f"[{label}] rail box "
        f"{rail['left']:.0f},{rail['top']:.0f}..{rail['right']:.0f},{rail['bottom']:.0f}; "
        f"reaches {sorted(reached) or 'nothing'}"
    )
    for name, v in sorted(reached.items()):
        info(f"    {name}: reach {v['reach']}px2, takes {v['taken']} of {v['total']}")
        check(
            v["taken"] == 0,
            f"{label}: a scrolling rail takes {v['taken']} of the {name} target's "
            f"{v['total']}px, so input started there lands on the rail",
        )
    return seen


def run_reach_and_capture(page) -> None:
    """The taller, pointer-active box must take nothing that is not already the rail's.

    Three states, because reach depends on width and placement: a card is
    `w-[calc(100vw-2rem)]` up to its max, so a narrow window spans the rail across nearly
    everything, and a lifted placement puts its lower gutter over the box it just dodged.
    """
    room = configure(page, cards = [INDICATOR])["geometry"]["maxHeight"]
    tall = {"height": room + 200, "floor": room + 200}

    page.set_viewport_size({"width": 1280, "height": 800})
    configure(page, cards = [tall])
    report_capture(page, "1280x800, scrolling")

    page.set_viewport_size({"width": 420, "height": 760})
    narrow_room = configure(page, cards = [INDICATOR])["geometry"]["maxHeight"]
    configure(
        page,
        cards = [{"height": narrow_room + 200, "floor": narrow_room + 200}],
    )
    report_capture(page, "420x760, scrolling")

    # Lifted over the composer, so the gutter hangs below the cards and over its top edge.
    page.set_viewport_size({"width": 1280, "height": 800})
    lifted = configure(
        page,
        cards = [{"height": 300, "floor": 300}] * 3,
        obstacles = obstacles_for("composer", 1280, 800),
    )
    check(
        lifted["geometry"]["overflowing"],
        "the lifted deck does not overflow, so the rail is click-through and the "
        "composer check proves nothing",
    )
    info(
        f"    lifted: bottom={lifted['geometry']['bottom']} "
        f"overflowing={lifted['geometry']['overflowing']}"
    )
    report_capture(page, "lifted over the composer")

    # The controls lose nothing to the rail. What they lose to a grip belongs to the
    # titlebar, and run_titlebar_stacking measures that against the pre-PR shape.
    taken = page.evaluate(CONTROL_LOSS, [CONTROLS])
    for name, v in sorted(taken.items()):
        info(f"    {name}: {v['lost']} of {v['total']}px taken by a grip or the rail")


def run_titlebar_stacking(page, base: str) -> None:
    """Whether the controls lose anything to the grips, and to whom the answer belongs.

    The controls sit inside a positioned, numbered header, which by CSS is a stacking
    context: every z-index inside it, the toolbar's included, is compared there and never
    against the grips outside. A number on the toolbar therefore reads as protection and
    gives none.

    Three shapes, so the report can say who owns the overlap. The grips are later siblings of
    the header, so the pre-PR pair at equal z-index already resolved in their favour by
    document order; if the three agree, the Close corner is inherited, not moved by this PR.
    """
    print("[matrix] titlebar stacking", flush = True)
    # Document order decides between equal z-indexes, so the comparison below holds only
    # while the harness keeps the app's order: header first, grips after.
    order = page.evaluate(
        """() => {
      const header = document.querySelector('[data-testid="titlebar-header"]');
      const grip = document.querySelector('[data-testid="resize-northeast"]');
      if (!header || !grip) return null;
      // DOCUMENT_POSITION_FOLLOWING is 4: the grip comes after the header.
      return Boolean(header.compareDocumentPosition(grip) & 4);
    }"""
    )
    check(
        order is True,
        "the harness renders the resize grips before the titlebar header, which is not the "
        "order window-titlebar.tsx uses; every equal-z result below would be backwards",
    )
    shapes = [
        ("before this PR (grips 70, header 70)", "gripz=70&headerz=70"),
        (
            "toolbar number added (grips 9050, header 70, toolbar 9060)",
            "gripz=9050&headerz=70&toolbarz=9060",
        ),
        ("at head (grips 9050, header 70, no toolbar number)", "gripz=9050&headerz=70"),
    ]
    losses = {}
    for label, query in shapes:
        page.goto(f"{base}/{ENTRY}?nostrict&{query}", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__railSmoke)", timeout = 30_000)
        configure(page, cards = [INDICATOR])
        taken = page.evaluate(CONTROL_LOSS, [CONTROLS])
        losses[label] = {k: v["lost"] for k, v in taken.items()}
        for name, v in sorted(taken.items()):
            info(f"    {label}: {name} loses {v['lost']} of {v['total']}px to a grip")

    before = losses[shapes[0][0]]
    numbered = losses[shapes[1][0]]
    head = losses[shapes[2][0]]
    check(
        numbered == before,
        f"a z-index on the toolbar changed what the controls keep, so it is not trapped "
        f"after all: {before} -> {numbered}",
    )
    check(
        head == before,
        f"raising the grips cost the window controls hit area they had before this PR: "
        f"{before} -> {head}",
    )

    page.goto(f"{base}/{ENTRY}?nostrict", wait_until = "domcontentloaded")
    page.wait_for_function("() => Boolean(window.__railSmoke)", timeout = 30_000)


def run_observer_and_scroll(page) -> None:
    """The rail measures itself; that must terminate, and must not lose the scroll."""
    print("[matrix] observers and scroll", flush = True)
    page.set_viewport_size({"width": 1280, "height": 800})
    loops: list[str] = []
    page.on(
        "console",
        lambda msg: (loops.append(msg.text) if "ResizeObserver loop" in msg.text else None),
    )
    room = configure(page, cards = [INDICATOR])["geometry"]["maxHeight"]

    for step in (-4, 4, -1, 1, -12, 12):
        height = room + step
        settle = page.evaluate(SETTLE)
        configure(page, cards = [{"height": height, "floor": height}])
        settle = page.evaluate(SETTLE)
        check(
            settle["settledAfter"] >= 0,
            f"the rail never stopped moving at cap{step:+d}: {settle['state']}",
        )
    check(not loops, f"ResizeObserver loop errors: {loops[:3]}")

    # The scroll position and the last card's reach must survive the none -> 0px -> cap
    # probe.
    tall = room + 240
    configure(page, cards = [{"height": tall, "floor": tall}, INDICATOR])
    state = page.evaluate(
        """async () => {
      const rail = document.querySelector('[data-testid="overlay-rail"]');
      const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
      rail.scrollTop = rail.scrollHeight;
      await frame();
      const before = rail.scrollTop;
      const max = rail.scrollHeight - rail.clientHeight;
      // Anything that makes the hook remeasure without changing the deck.
      window.dispatchEvent(new Event("resize"));
      for (let i = 0; i < 10; i += 1) await frame();
      const cards = [...rail.children];
      const last = cards[cards.length - 1].getBoundingClientRect();
      const r = rail.getBoundingClientRect();
      return {
        before, after: rail.scrollTop, max,
        lastFullyInside: last.bottom <= r.bottom + 0.6 && last.top >= r.top - 0.6,
      };
    }"""
    )
    check(
        abs(state["after"] - state["before"]) <= EPS,
        f"the measurement moved the scroll from {state['before']} to {state['after']}",
    )
    check(
        state["lastFullyInside"],
        "scrolled to the end, the last card is not wholly inside the clip; the gutter is "
        "not reachable",
    )
    info(f"scroll {state['before']:.1f} -> {state['after']:.1f} of {state['max']:.1f}")


def run_shadow_pixels(page) -> None:
    """The clip must no longer cut the card's shadow into a straight line."""
    print("[matrix] shadow", flush = True)
    page.set_viewport_size({"width": 1280, "height": 800})
    m = configure(page, cards = [INDICATOR])
    if not check(m is not None and m["cardCount"] == 1, "the shadow deck did not render"):
        return
    profile = page.evaluate(
        """() => {
      const rail = document.querySelector('[data-testid="overlay-rail"]');
      const surface = rail.querySelector("[data-card-surface]");
      const s = surface.getBoundingClientRect();
      const r = rail.getBoundingClientRect();
      return {
        // How far below the card's own edge the clip lets anything paint at all.
        room: r.bottom - s.bottom,
        shadow: getComputedStyle(surface).boxShadow,
      };
    }"""
    )
    ART.mkdir(parents = True, exist_ok = True)
    page.screenshot(
        path = str(ART / f"rail-{PLAYWRIGHT_BROWSER}.png"),
        clip = {
            "x": max(0, m["rail"]["left"] - 8),
            "y": m["rail"]["top"] - 8,
            "width": min(m["rail"]["width"] + 16, m["innerWidth"] - m["rail"]["left"]),
            "height": m["rail"]["height"] + 16,
        },
    )
    check(
        profile["room"] >= m["gutter"]["bottom"] - 0.6,
        f"only {profile['room']:.2f}px of paintable room under the card, reserved "
        f"{m['gutter']['bottom']}",
    )
    check(
        "rgba" in profile["shadow"] or "rgb" in profile["shadow"],
        f"the card carries no shadow to clip: {profile['shadow']}",
    )
    info(f"{profile['room']:.1f}px under the card, shadow {profile['shadow']}")


def main() -> int:
    server = None
    base = BASE_URL
    if not base:
        server = start_vite(PORT)
        base = f"http://127.0.0.1:{PORT}"
    url = f"{base}/{ENTRY}?nostrict"
    try:
        with sync_playwright() as pw:
            launcher = getattr(pw, PLAYWRIGHT_BROWSER)
            browser = launcher.launch(
                args = chromium_launch_args() if PLAYWRIGHT_BROWSER == "chromium" else [],
            )
            page = browser.new_page(viewport = {"width": 1280, "height": 800})
            echo_browser_errors(page, info)
            wait_for_smoke_page(url, ENTRY_MODULE, proc = server, info = info)
            page.goto(url, wait_until = "domcontentloaded")
            # Attached, not visible: an empty rail is a zero-height box, so waiting on
            # visibility would hang.
            page.wait_for_selector(
                '[data-testid="overlay-rail"]',
                state = "attached",
                timeout = 30_000,
            )
            page.wait_for_function("() => Boolean(window.__railSmoke)", timeout = 30_000)

            print(f"=== overlay rail, {PLAYWRIGHT_BROWSER} ===", flush = True)
            run_geometry_matrix(page)
            run_cap_boundary(page)
            run_short_cap(page)
            run_stale_block_end_padding(page)
            run_hit_testing(page)
            run_titlebar_stacking(page, base)
            run_observer_and_scroll(page)
            run_shadow_pixels(page)
            browser.close()
    finally:
        if server is not None:
            stop_process(server)

    if failures:
        print(f"\n{len(failures)} failure(s):", flush = True)
        for line in failures[:40]:
            print(f"  - {line}", flush = True)
        ART.mkdir(parents = True, exist_ok = True)
        # Named, not the platform default: read back on the Windows shard, where a
        # non-ASCII byte would otherwise not round-trip.
        (ART / f"failures-{PLAYWRIGHT_BROWSER}.json").write_text(
            json.dumps(failures, indent = 2),
            encoding = "utf-8",
        )
        return 1
    print("\noverlay rail: all checks passed", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
