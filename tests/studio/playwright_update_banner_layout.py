# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The overlay rail's update banners must never print over each other.

The reported failure: in a window short enough that the rail hits its cap, the
app-update card's release notes were painted over its own row of buttons.

The second thing checked here is where the rail is. It was placed from JS for a
while, dodging the boxes the composer and the floating panels publish, and every
input to that placement moved on its own, so the rail drifted out of its corner
into the middle and the top of the window. It is anchored in CSS again, and the
indicator pass at the end asserts it stays there while cards come and go.

The node suite cannot catch either: one is a flex shrink across a capped column
and the other is where a fixed box actually lands, so both need a real layout,
a real ResizeObserver and the real route, and they show only at some viewport
heights. Rects are intersected with whatever clips them, so anything an
overflow-hidden ancestor hides does not count as visible.

Both update endpoints are stubbed with page.route, so this runs on any host: no
GPU, no pypi release, no llama.cpp build.

Run: BASE_URL, STUDIO_OLD_PW and STUDIO_NEW_PW as the other suites take them.
STUDIO_PLAYWRIGHT_BROWSER selects chromium (default), firefox or webkit.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_wall_clock_watchdog,
    wait_for_health,
)

# The wall this suite did not have, matching playwright_chat_ui.py and playwright_extra_ui.py. It has six raw
# `page.evaluate` calls, which take no `timeout=` at all, and it runs mid-lane on Windows sharing a server with the
# suite before it -- so a wedge here stalled the lane with nothing printed.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-update-banner"))
ART.mkdir(parents = True, exist_ok = True)

PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# The web check fires 5s after mount and the llama.cpp one after 1s; this is the ceiling on waiting for them, not the
# wait itself.
SETTLE_MS = int(os.environ.get("STUDIO_UI_BANNER_SETTLE_MS", "9000"))
# What the cards need after they mount, to animate in and lay out.
SETTLED_MS = int(os.environ.get("STUDIO_UI_BANNER_SETTLED_MS", "900"))

# Must match the name use-web-update-check.ts reads. Kept short rather than zero so the card still arrives after
# first paint, which is the situation the layout checks exist for.
E2E_DELAY_GLOBAL = "__unslothE2EWebUpdateDelayMs"
E2E_DELAY_MS = int(os.environ.get("STUDIO_UI_BANNER_UPDATE_DELAY_MS", "150"))

LATEST = "2099.1.0"

# Long enough that the collapsed preview alone overflows a capped card.
NOTES_MARKDOWN = "\n".join(
    [
        f"## {LATEST}",
        "",
        "### What's Changed",
        "",
        "- DeepSeek-V4 0731 DSpark 2x faster inference support, with a lead "
        "sentence long enough to wrap onto a second line in a 448px card.",
        "- Many bug fixes across training, inference and the model hub.",
        "- Training page full rework.",
        "- Unsloth desktop update flow reworked.",
    ]
)

UPDATE_STATUS = {
    "current_version": "2026.8.7",
    "latest_version": LATEST,
    "update_available": True,
    "install_source": "pypi",
    "can_show_web_notification": True,
    "release_notes_url": "https://unsloth.ai/docs/new/changelog",
    "checked_at": "2099-01-01T00:00:00Z",
    "reason": None,
    "error": None,
}
RELEASE_NOTES = {
    "version": LATEST,
    "markdown": NOTES_MARKDOWN,
    "matched": True,
    "truncated": False,
    "source": "test",
    "release_notes_url": "https://unsloth.ai/docs/new/changelog",
    "error": None,
}
# A successful lookup with no previewable body.
RELEASE_NOTES_NONE = dict(RELEASE_NOTES, markdown = "", matched = False)
LLAMA_STATUS = {
    "supported": True,
    "update_available": True,
    "llama_update_available": True,
    "update_component": "llama",
    "installed_tag": "b10333",
    "latest_tag": "b10333-mix-e34b418",
    "update_size_bytes": 28 * 1024 * 1024,
    "source_build": False,
    "component": "llama.cpp",
    "whisper": {
        "update_available": False,
        "installed_tag": "v1.9.1",
        "latest_tag": "v1.9.1",
        "update_size_bytes": None,
        "skip_reason": "up_to_date",
    },
    "job": {
        "state": "idle",
        "message": "",
        "from_tag": None,
        "to_tag": None,
        "reload_required": None,
        "error": None,
        "progress": None,
        "finished_at": None,
    },
}
LLAMA_CHANGELOG = {
    "matched": True,
    "installed_tag": "b10333",
    "latest_tag": "b10333-mix-e34b418",
    "changes": [
        {
            "summary": "model: add GLM-5-Next (GLM-5.3-Flash)",
            "links": [
                {
                    "label": "#27754",
                    "url": "https://github.com/ggml-org/llama.cpp/pull/27754",
                },
                {
                    "label": "commit 949f7ef",
                    "url": "https://github.com/ggml-org/llama.cpp/pull/27754/commits/949f7ef",
                },
            ],
        },
        {
            "summary": "llama: batched readahead for lazily read gather tables",
            "links": [
                {
                    "label": "unslothai/llama.cpp#137",
                    "url": "https://github.com/unslothai/llama.cpp/pull/137",
                }
            ],
        },
        {"summary": "MTP for Qwen3.8-Flash-Next", "links": []},
    ],
    "total_changes": 3,
    "truncated": False,
    "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/b10333-mix-e34b418",
    "error": None,
}
# The same card, renamed: whisper.cpp is not a second banner.
WHISPER_STATUS = dict(
    LLAMA_STATUS,
    llama_update_available = False,
    update_component = "whisper",
    component = "whisper.cpp",
    whisper = {
        "update_available": True,
        "installed_tag": "v1.9.1",
        "latest_tag": "v1.9.2",
        "update_size_bytes": 11 * 1024 * 1024,
        "skip_reason": None,
    },
)

# 921x534 and 768x500 are where the report reproduces; the taller ones prove the fix costs nothing when there is room.
VIEWPORTS = [
    (1440, 900),
    (1280, 830),
    (921, 534),
    (768, 500),
    (390, 844),
    # Narrow enough to wrap the action row, short enough to hit the card's floor.
    (390, 500),
]
ROUTES = [("new chat", "/"), ("train", "/train"), ("model hub", "/model-hub")]

# Real display and window sizes, walked by RESIZING one already-loaded page rather than booting one each: a resize is
# what a maximise, an unmaximise, a restore from the dock and a drag of the window edge all are, so this is the cheap
# sweep and the realistic one at the same time. Ordered largest first so the run starts from the roomiest layout and
# squeezes.
#
# 3840x2160 and 2560x1440 are maximised on a 4K and a QHD display; 1920x1080 is the commonest desktop there is;
# 1512x982 and 1440x900 are the default scaled resolutions of a 14in MacBook Pro and a MacBook Air; 1366x768 is the
# commonest Windows laptop; 900x600 is the desktop app's own minimum window; the rest are ordinary small windows down
# to a phone in portrait.
RESIZE_SWEEP = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1680, 1050),
    (1600, 900),
    (1512, 982),
    (1440, 900),
    (1366, 768),
    (1280, 800),
    (1280, 720),
    (1152, 720),
    (1024, 768),
    (1024, 600),
    (960, 640),
    (900, 600),
    (800, 600),
    (768, 500),
    (720, 480),
    (640, 480),
    (430, 932),
    (390, 844),
    (360, 640),
    (320, 568),
]

# What a resize needs before it has settled: the ResizeObserver, the placement it feeds, and the reflow after that.
# Nothing is fetched, so this is short.
RESIZE_SETTLE_MS = int(os.environ.get("STUDIO_UI_BANNER_RESIZE_MS", "700"))

# The four runtime status endpoints the loaded models card reads, shaped as
# tests/studio/playwright_loaded_models_indicator.py has them. Only chat holds anything: one loaded model is all it
# takes to put the card in the rail.
CHAT_LOADED = {
    "active_model": "unsloth/Qwen3-4B",
    "loaded": ["unsloth/Qwen3-4B"],
    "is_gguf": False,
    "is_mlx": False,
    "is_vision": False,
    "is_audio": False,
    "audio_type": None,
    "gguf_variant": None,
}
NOTHING_DIFFUSION = {
    "loaded": False,
    "repo_id": None,
    "family": None,
    "device": None,
    "dtype": None,
    "model_kind": None,
}
NOTHING_VIDEO = dict(NOTHING_DIFFUSION, transformer_quant = None)
NOTHING_STT = {
    "available": True,
    "loaded_model": None,
    "device": None,
    "transformers": {"loaded_model": None, "device": None},
    "mtmd": {"loaded_model": None, "device": None},
    "gguf": {"loaded_model": None, "device": None},
}

# Short windows on the chat route: the ones where the rail used to leave its corner. Two is enough, since what is
# being checked does not vary with size.
INDICATOR_VIEWPORTS = [(921, 534), (768, 500)]

# One roomy viewport and one capped viewport.
NO_PREVIEW_VIEWPORTS = [(1440, 900), (921, 534)]

# Where the rail actually is, against the corner it is anchored to. It was placed from JS for a while, lifting clear
# of the boxes in the frame store, and drifted to the middle and the top of the window as those boxes and its own
# cards changed. Nothing on the page may move it now.
RAIL_CORNER = """
() => {
  const card = document.querySelector('[data-testid="web-update-banner"]');
  if (!card) return null;
  const rail = card.parentElement;
  const a = rail.getBoundingClientRect();
  // Cards in the rail's own flow. A dragged loaded models card is `fixed`
  // somewhere else and says nothing about the rail.
  const flowed = Array.from(rail.children).filter(
    (kid) => getComputedStyle(kid).position === 'static',
  );
  return {
    rail: {top: Math.round(a.top), bottom: Math.round(a.bottom),
           right: Math.round(a.right)},
    viewport: {width: window.innerWidth, height: window.innerHeight},
    // Where the CARDS sit, 16px off both edges. Neither reading comes off the
    // rail's border box: it carries the shadow gutter on all four sides, so it
    // sits 4px from the right and flush with the floor while the cards it pads
    // sit at 16. Measuring the border box reported 4 and 0.
    //
    // The bottom off the padding box, which is the cards' own floor and stays
    // put however far they are scrolled. The right off a card directly, since
    // the horizontal gutter is a negative margin the padding cancels.
    fromBottom: Math.round(
      window.innerHeight - a.bottom
        + parseFloat(getComputedStyle(rail).paddingBottom || '0'),
    ),
    fromRight: flowed.length
      ? Math.max(...flowed.map(
          (kid) => Math.round(
            window.innerWidth - kid.getBoundingClientRect().right,
          ),
        ))
      : null,
    flowedCards: flowed.length,
    // Nothing inline may set either: an offset or a cap written by JS is the
    // placement coming back, whatever value it happens to have landed on.
    railStyle: {bottom: rail.style.bottom, maxHeight: rail.style.maxHeight,
                kids: rail.childElementCount,
                height: getComputedStyle(rail).height,
                cappedTo: getComputedStyle(rail).maxHeight},
    // In the RAIL, not merely on the page: the card is draggable, and one
    // parked elsewhere would satisfy a page-wide search while telling us
    // nothing about the stack under test.
    indicator: (() => {
      const label = Array.from(document.querySelectorAll('*')).find(
        (el) => el.childElementCount === 0
          && el.textContent.trim() === 'Loaded models',
      );
      return Boolean(label && rail.contains(label));
    })(),
  };
}
"""

# A minimised window has no layout to photograph, so what is actually testable is the RESTORE: the geometry is
# measured by a ResizeObserver and cached in React state, and a window that goes away and comes back is exactly how a
# stale measurement would survive. Each pair is (parked, restored).
RESTORE_CYCLES = [((320, 400), (1920, 1080)), ((320, 400), (900, 600))]

# `spot` is the same suite cut down to what a SECOND browser engine is worth running: the viewports that reproduce,
# one route, no resize sweep and no type pass. Chromium runs `full`; Firefox and WebKit run this, because the job they
# share has minutes rather than tens of minutes to spare and a third full pass would mostly re-answer questions the
# first one already answered.
SCOPE = os.environ.get("STUDIO_UI_BANNER_SCOPE", "full").lower()
SPOT = SCOPE == "spot"

# The two that squeeze the rail hardest, re-run at the largest UI font size.
# The whole matrix again would not fit the job's budget and would say the same thing three times over.
# 320x480 is not a spare small size, it is the one that bites.
# Below 384px the card's action pair wraps onto a row of its own on top of the notes toggle's, and at the 20px setting
# that is a whole extra row: 259px where the wide card needs 209.
# The height matters as much as the width, because the harm needs a rail cap BETWEEN the two.
# At 320x480 the cap is 293px, so a 209px floor let the card shrink to 209 and its own overflow-hidden surface cut 34px
# off Copy command;
# at 320x568 the cap is 528px and nothing is squeezed at all.
FONT_SCALE_VIEWPORTS = [(921, 534), (390, 500), (320, 480)]
# appearance-custom-store.ts: UI_FONT_SIZE_RANGE, UI_FONT_SIZE_CSS_BASE and the persist version.
# Kept in step by test_update_release_notes.py.
UI_FONT_SIZE_MAX = 20
UI_FONT_SIZE_DEFAULT = 15
UI_FONT_SIZE_CSS_BASE = 16
APPEARANCE_STORE_VERSION = 5

failures: list[str] = []
checks = [0]


def info(s: str) -> None:
    print(f"[banner] {s}", flush = True)


def check(
    name: str,
    ok: bool,
    detail: str = "",
) -> None:
    checks[0] += 1
    if ok:
        info(f"PASS {name}")
        return
    failures.append(f"{name} ({detail})" if detail else name)
    info(f"FAIL {name} {detail}")


def api(
    path: str,
    payload: dict | None = None,
    token: str | None = None,
    method: str | None = None,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data = data,
        method = method or ("POST" if data else "GET"),
        headers = {"Content-Type": "application/json"}
        | ({"Authorization": f"Bearer {token}"} if token else {}),
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        return json.loads(response.read().decode())


def read_ui_font_size(token: str) -> int | None:
    """The Appearance type size this install is on before the suite touches it."""
    current = api("/api/settings/personalization", token = token)
    return current["appearance"]["customization"].get("uiFontSize")


def set_ui_font_size(token: str, size: int | None) -> None:
    """Set, or clear, the Appearance type size on the SERVER.

    Seeding it into localStorage is not enough and is actively harmful: the
    appearance store syncs up to `/api/settings/personalization`, so a browser
    that starts at 20px leaves the install at 20px for everything that runs
    after it. That is not hypothetical, it is how a whole afternoon of local
    runs came to be measured at 20px while reporting themselves as default,
    and how a later suite in the same CI job would inherit it.
    """
    current = api("/api/settings/personalization", token = token)
    current["appearance"]["customization"]["uiFontSize"] = size
    api("/api/settings/personalization", current, token = token, method = "PUT")


# Every box the fix is about, already intersected with whatever clips it.
MEASURE = """
() => {
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, left: r.left, right: r.right,
            width: r.width, height: r.height};
  };
  // Takes an element or an already-clipped box, so clips can be chained.
  const clip = (el, clipper) => {
    if (el === null) return null;
    const a = el.top !== undefined ? el : rect(el);
    const b = rect(clipper);
    if (!a || !b) return a;
    // Only a scroll or hidden ancestor hides anything; clipping to a visible
    // one would erase the very overflow this is looking for.
    if (getComputedStyle(clipper).overflow === 'visible') return a;
    const top = Math.max(a.top, b.top), bottom = Math.min(a.bottom, b.bottom);
    const left = Math.max(a.left, b.left), right = Math.min(a.right, b.right);
    if (bottom <= top || right <= left) return null;
    return {top, bottom, left, right, width: right - left, height: bottom - top};
  };
  // Measure any height the slot reserves but its surface does not paint.
  const dead = (el) => {
    if (!el || !el.firstElementChild) return null;
    const a = el.getBoundingClientRect();
    const b = el.firstElementChild.getBoundingClientRect();
    const round = (n) => Math.round(n * 10) / 10;
    return {above: round(b.top - a.top), below: round(a.bottom - b.bottom),
            slot: round(a.height), painted: round(b.height),
            minHeight: getComputedStyle(el).minHeight};
  };
  const q = (sel) => document.querySelector(sel);
  const card = q('[data-testid="web-update-banner"]');
  const llama = q('[data-testid="llama-update-banner"]');
  const notes = q('[data-testid="update-release-notes-panel"]');
  const body = q('[data-testid="update-release-notes-summary"]')
            || q('[data-testid="update-release-notes-scroll"]');
  const toggle = q('[data-testid="web-update-release-notes-toggle"]');
  const snooze = q('[data-testid="web-update-snooze-button"]');
  const copy = q('[data-testid="web-update-copy-button"]');
  const footer = snooze ? snooze.closest('div').parentElement : null;
  const rail = card ? card.parentElement : (llama ? llama.parentElement : null);
  // The clipper is the card's inner surface, the one with overflow-hidden; the
  // rail-facing root above it is overflow-visible and clips nothing.
  const surface = card ? card.firstElementChild : null;
  return {
    viewport: {width: innerWidth, height: innerHeight},
    // Clipped by the card, not by the rail. What the rail hides is under a
    // fold the reader can scroll to; what the card hides is gone for good.
    card: rect(card), llama: rect(llama),
    cardDead: dead(card), llamaDead: dead(llama),
    // The same two, as much of them as the rail is SHOWING. Containment is
    // asked of these: a card the rail has folded away is not on screen at all,
    // and judging its unclipped rect against the viewport fails it for being
    // scrolled out of sight, which is what the reach check below is for.
    cardShown: clip(card, rail), llamaShown: clip(llama, rail),
    notesBody: clip(body, notes),
    toggle: clip(toggle, surface),
    snooze: clip(snooze, surface),
    copy: clip(copy, surface),
    footer: rect(footer),
    llamaText: llama ? (llama.innerText || '') : '',
    // pointer-events-none costs the rail its scrollbar, so it may only be
    // click-through while there is nothing under the fold to scroll to.
    railScrolls: rail ? rail.scrollHeight > rail.clientHeight : null,
    // A classic scrollbar (Windows, Linux) takes width out of the box it is
    // on; an overlay one (macOS, and WebKit generally) does not. The rail
    // reserves a gutter for exactly this, so the card's width must not depend
    // on which platform it is or on whether the rail happens to be scrolling.
    railGutterPx: rail ? Math.round(rail.offsetWidth - rail.clientWidth) : null,
    cardWidth: card ? Math.round(card.getBoundingClientRect().width) : null,
    // The same card, measured off the LAYOUT box instead of the painted one.
    // offsetWidth is the border box with no transform applied (CSSOM-View), so
    // it answers the scrollbar question the assertion below is actually asking
    // and cannot be moved by the card's enter animation. `cardWidth` stays,
    // reported alongside, because the gap between the two is the diagnosis.
    cardLayoutWidth: card ? card.offsetWidth : null,
    railPointerEvents: rail ? getComputedStyle(rail).pointerEvents : null,
    // Everything needed to say WHY a card came out narrow, reported with the
    // failure instead of being guessed at afterwards from two numbers. A card
    // that lost width to a scrollbar and a card that lost it to an unfinished
    // transform read identically as `cardWidth`, and the engine that reports
    // offsetWidth === clientWidth while still taking the width out of the
    // content box (Playwright WebKit on Linux) makes railGutterPx no help on
    // its own. `borderBox` is the layout width with no transform applied.
    widthWhy: card && rail ? {
      transform: getComputedStyle(card).transform,
      borderBox: card.offsetWidth,
      cssWidth: getComputedStyle(card).width,
      maxWidth: getComputedStyle(card).maxWidth,
      innerWidth: innerWidth,
      docClientWidth: document.documentElement.clientWidth,
      railOffsetW: rail.offsetWidth,
      railClientW: rail.clientWidth,
      railContentW: rail.clientWidth
        - parseFloat(getComputedStyle(rail).paddingLeft || '0')
        - parseFloat(getComputedStyle(rail).paddingRight || '0'),
      railScrollH: rail.scrollHeight,
      railClientH: rail.clientHeight,
      railMaxHeight: getComputedStyle(rail).maxHeight,
      kids: [...rail.children].map((c) => Math.round(c.getBoundingClientRect().height)),
    } : null,
    // What a click on the rail's own gutter lands on when it is click-through.
    gutterIsRail: rail ? (() => {
      const r = rail.getBoundingClientRect();
      return document.elementFromPoint(
        Math.round(r.right - 2), Math.round(r.top + r.height / 2)) === rail;
    })() : null,
  };
}
"""


# Scroll each box into the rail's view and report what is still hidden. The
# rail is the last resort the cards fall back on, so "below the fold" is a pass
# and "cannot be brought into view" is the failure.
REACH = """
(selectors) => {
  const q = (sel) => document.querySelector(sel);
  const card = q('[data-testid="web-update-banner"]');
  const llama = q('[data-testid="llama-update-banner"]');
  const rail = card ? card.parentElement : (llama ? llama.parentElement : null);
  if (!rail) return null;
  const was = rail.scrollTop;
  const out = {};
  for (const [name, sel] of Object.entries(selectors)) {
    const el = q(sel);
    if (!el) { out[name] = null; continue; }
    el.scrollIntoView({block: 'nearest', inline: 'nearest'});
    const r = el.getBoundingClientRect();
    const b = rail.getBoundingClientRect();
    out[name] = {
      hidden: Math.round(Math.max(0,
        Math.max(b.top - r.top, 0) + Math.max(r.bottom - b.bottom, 0)) * 10) / 10,
      offscreen: r.top < -0.5 || r.bottom > innerHeight + 0.5,
      // Taller than the fold itself: no scroll position shows all of it, and
      // none has to, since every part of it can be scrolled to.
      taller: r.height > b.height + 1,
      scrollable: rail.scrollHeight > rail.clientHeight,
    };
  }
  rail.scrollTop = was;
  return out;
}
"""

REACHABLE = {
    "card": '[data-testid="web-update-banner"]',
    "llama": '[data-testid="llama-update-banner"]',
    "toggle": '[data-testid="web-update-release-notes-toggle"]',
    "snooze": '[data-testid="web-update-snooze-button"]',
    "copy": '[data-testid="web-update-copy-button"]',
}


def overlap(a: dict | None, b: dict | None) -> float:
    """Pixels of the smaller intersecting side, 0 if they do not intersect."""
    if not a or not b:
        return 0.0
    dy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    dx = min(a["right"], b["right"]) - max(a["left"], b["left"])
    # Half a pixel of touching is a rounded edge, not an overlap.
    return round(min(dy, dx), 1) if dy > 0.5 and dx > 0.5 else 0.0


def inside(box: dict | None, viewport: dict) -> bool:
    # A missing box is not "inside": clip() returns None for an element that is entirely hidden, and reading that as a
    # pass would make this whole suite green on the one failure it exists to catch.
    if not box:
        return False
    return (
        box["top"] >= -0.5
        and box["left"] >= -0.5
        and box["bottom"] <= viewport["height"] + 0.5
        and box["right"] <= viewport["width"] + 0.5
    )


def measure(page, label: str) -> dict:
    facts = page.evaluate(MEASURE)
    view = facts["viewport"]
    check(
        f"{label}: the notes do not print over the buttons",
        overlap(facts["notesBody"], facts["footer"]) == 0.0
        and overlap(facts["notesBody"], facts["toggle"]) == 0.0,
        f"notes={facts['notesBody']} footer={facts['footer']}",
    )
    check(
        f"{label}: the two banners do not print over each other",
        overlap(facts["card"], facts["llama"]) == 0.0,
        f"card={facts['card']} llama={facts['llama']}",
    )
    for name in ("card", "llama", "footer", "toggle"):
        # As shown, where the rail can hide it; as measured otherwise. A card entirely under the fold is None here and
        # is the reach check's to make.
        shown = facts.get(f"{name}Shown", facts[name]) if name in ("card", "llama") else facts[name]
        if name in ("card", "llama") and shown is None:
            continue
        check(
            f"{label}: the {name} stays inside the viewport",
            inside(shown, view),
            f"{name}={shown} viewport={view}",
        )
    # Anything the rail is holding under its fold has to come back when the rail is scrolled to it, and land on screen
    # when it does.
    reach = page.evaluate(REACH, REACHABLE)
    for name, seen in (reach or {}).items():
        if seen is None:
            continue
        reached = seen["scrollable"] if seen["taller"] else seen["hidden"] <= 1.0
        check(
            f"{label}: the {name} can be scrolled into the rail's view",
            reached and not seen["offscreen"],
            f"{name}={seen}",
        )
    if facts["cardLayoutWidth"] is not None:
        # 448px is the card's max width and 2rem the viewport inset it keeps.
        want = min(448, view["width"] - 32)
        # Asked of the layout box, not the painted one. A scrollbar that takes its width out of the rail's content box
        # shrinks the card's layout width, which is the whole subject here; the card's enter animation (opacity 0,
        # y 12, scale .96 -- see components/*/update-banner.tsx) shrinks only the painted one, and measuring that
        # raced the animation rather than the scrollbar.
        # 448 * 0.96 = 430.08, which is exactly the 430 this reported on the WebKit leg while its own borderBox read 448
        # and railGutter read 0.
        check(
            f"{label}: the card keeps its full width whatever the scrollbar does",
            abs(facts["cardLayoutWidth"] - want) <= 1,
            f"cardLayoutWidth={facts['cardLayoutWidth']} want={want} "
            f"cardPaintedWidth={facts['cardWidth']} "
            f"railGutter={facts['railGutterPx']} scrolls={facts['railScrolls']} "
            f"why={json.dumps(facts['widthWhy'], sort_keys = True)}",
        )
    if facts["railScrolls"] is not None:
        scrolls = facts["railScrolls"]
        # Click-through in every state, scrolling or not. It used to take pointer input while it scrolled, which needed
        # the JS that also placed it. The fold is reached by wheeling over a card, whose nearest scrollable ancestor is
        # the rail, or by focus, which scrolls it into view.
        check(
            f"{label}: the rail stays click-through",
            facts["railPointerEvents"] == "none",
            f"scrolls={scrolls} pointerEvents={facts['railPointerEvents']} "
            f"why={json.dumps(facts['widthWhy'], sort_keys = True)}",
        )
        # The gutter is the widest part of the rail that no card covers, so it is where a swallowed click would show up
        # first.
        check(
            f"{label}: the rail's gutter never swallows a click",
            facts["gutterIsRail"] is False,
            f"scrolls={scrolls} gutterIsRail={facts['gutterIsRail']}",
        )
    # A floor must not leave unpainted space around a compact card.
    for name in ("card", "llama"):
        hole = facts[f"{name}Dead"]
        if hole is None:
            continue
        check(
            f"{label}: the {name}'s slot reserves no height it does not paint",
            hole["above"] <= 1.0 and hole["below"] <= 1.0,
            f"{name}Dead={hole}",
        )
    # The notes are allowed to yield all of their height, and do; the controls are not, and a card clipped to nothing is
    # the failure being tested for.
    for name in ("card", "llama", "toggle", "snooze", "copy"):
        box = facts[name]
        check(
            f"{label}: the card does not clip its own {name} away",
            box is not None and box["height"] > 1.0 and box["width"] > 1.0,
            f"{name}={box}",
        )
    return facts


def stub(payload: dict):
    """A route handler that answers with `payload`."""

    def handler(route) -> None:
        route.fulfill(
            status = 200,
            content_type = "application/json",
            body = json.dumps(payload),
        )

    return handler


RAIL_BOX = """
() => {
  const card = document.querySelector('[data-testid="web-update-banner"]')
            || document.querySelector('[data-testid="llama-update-banner"]');
  if (!card) return null;
  const r = card.parentElement.getBoundingClientRect();
  return [Math.round(r.top), Math.round(r.bottom), Math.round(r.height)].join(',');
}
"""


def settle_stack(
    page,
    tries: int = 24,
    gap_ms: int = 250,
) -> None:
    """Wait until the rail's box stops moving.

    Waits for STABILITY, not for the answer the checks want: a card mounting
    late changes the stack's height, which re-measures the placement, which
    moves the rail on the frame after that. Measuring in the middle of that is
    how the models indicator pass reported the cards below the viewport when a
    probe watching the same page settled correctly a second later. Waiting for
    `card.bottom <= innerHeight` instead would be waiting for the assertion,
    and would pass on a layout that never settled at all.
    """
    seen = None
    stable = 0
    for _ in range(tries):
        now = page.evaluate(RAIL_BOX)
        stable = stable + 1 if now is not None and now == seen else 0
        seen = now
        if stable >= 2:
            return
        page.wait_for_timeout(gap_ms)


def boot(page, path: str) -> None:
    page.goto(f"{BASE}{path}", wait_until = "domcontentloaded")
    # Both cards are on a timer, so wait for them rather than for the worst case: this step runs 24 times and the job it
    # shares has minutes, not tens of minutes, to spare. The app card's 5s is shortened to E2E_DELAY_MS by the seed
    # script, llama.cpp keeps its 1s, and both still mount after first paint.
    for testid in ("web-update-banner", "llama-update-banner"):
        try:
            page.wait_for_selector(f'[data-testid="{testid}"]', state = "attached", timeout = SETTLE_MS)
        except PlaywrightTimeoutError:
            # Let the caller's own checks report the missing card; a bare timeout here would say nothing about which one
            # or where.
            pass
    # The banners animate in, and a box measured mid-transition is not the box.
    page.wait_for_timeout(SETTLED_MS)
    settle_stack(page)
    landed = page.evaluate("location.pathname")
    if landed.startswith(("/login", "/change-password")):
        raise AssertionError(f"not authenticated: landed on {landed}")


LLAMA_CHANGELOG_GEOMETRY = """
() => {
  const q = (selector) => document.querySelector(selector);
  const rect = (element) => {
    if (!element) return null;
    const box = element.getBoundingClientRect();
    return {top: box.top, bottom: box.bottom, left: box.left, right: box.right,
            width: box.width, height: box.height};
  };
  const banner = q('[data-testid="llama-update-banner"]');
  const surface = banner ? banner.firstElementChild : null;
  const list = q('[data-testid="llama-update-changelog-list"]');
  const toggle = q('[data-testid="llama-update-changelog-toggle"]');
  const update = q('[data-testid="llama-update-button"]');
  const footer = update ? update.closest('div').parentElement : null;
  return {
    surface: rect(surface), list: rect(list), toggle: rect(toggle),
    update: rect(update), footer: rect(footer),
    listScrolls: list ? list.scrollHeight > list.clientHeight : null,
  };
}
"""


def exercise_llama_changelog(page, label: str) -> None:
    toggle = page.locator('[data-testid="llama-update-changelog-toggle"]')
    check(
        f"{label}: the llama.cpp changelog starts collapsed",
        toggle.count() == 1 and toggle.get_attribute("aria-expanded") == "false",
    )
    if toggle.count() != 1:
        return
    with page.expect_response("**/api/llama/update-changelog*", timeout = 10_000):
        toggle.click()
    listing = page.locator('[data-testid="llama-update-changelog-list"]')
    listing.wait_for(state = "visible", timeout = 10_000)
    text = listing.inner_text()
    check(
        f"{label}: expansion shows only the new carried changes",
        "GLM-5-Next" in text
        and "MTP for Qwen3.8-Flash-Next" in text
        and "Add TML Inkling" not in text,
        f"list={text!r}",
    )
    check(
        f"{label}: expansion exposes its state to assistive technology",
        toggle.get_attribute("aria-expanded") == "true",
    )
    pull = listing.locator('a[href="https://github.com/ggml-org/llama.cpp/pull/27754"]')
    check(
        f"{label}: change references are safe external links",
        pull.count() == 1
        and pull.get_attribute("target") == "_blank"
        and pull.get_attribute("rel") == "noopener noreferrer",
    )
    geometry = page.evaluate(LLAMA_CHANGELOG_GEOMETRY)
    surface, body, footer = geometry["surface"], geometry["list"], geometry["footer"]
    check(
        f"{label}: the changelog stays inside its card and above its actions",
        surface is not None
        and body is not None
        and footer is not None
        and body["left"] >= surface["left"] - 1
        and body["right"] <= surface["right"] + 1
        and body["bottom"] <= footer["top"] + 1,
        f"geometry={geometry}",
    )
    page.screenshot(path = str(ART / f"{label.replace(' ', '-')}-llama-expanded.png"))
    toggle.click()
    check(
        f"{label}: the changelog collapses without dismissing the update",
        toggle.get_attribute("aria-expanded") == "false"
        and page.locator('[data-testid="llama-update-banner"]').count() == 1,
    )


def main() -> int:
    wait_for_health(BASE, timeout = 60.0, info = info)
    # OLD is already NEW on a rerun, or when an earlier suite in the same job rotated it, and that login fails before
    # the rotation below can be skipped.
    try:
        token = api("/api/auth/login", {"username": "unsloth", "password": OLD})["access_token"]
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 401, 403):
            raise
        token = None
    if token is not None:
        try:
            api(
                "/api/auth/change-password",
                {"current_password": OLD, "new_password": NEW},
                token,
            )
        except urllib.error.HTTPError as exc:
            # Already rotated by a previous run on the same install.
            if exc.code not in (400, 401, 403):
                raise
    session = api("/api/auth/login", {"username": "unsloth", "password": NEW})

    # add_init_script takes raw source, not a function to call.
    seed_js = (
        "(() => {"
        # The app arms its update check on a 5s timer so the request stays off the critical path at launch. This suite
        # boots a fresh page for every case it measures and waits out that timer each time before the card it is
        # measuring exists, which was over two and a half minutes of a five minute step. The override is read at mount
        # from a global that exists only here, so the shortened delay reaches no build and no browser but this one, and
        # it stays a timer rather than becoming synchronous, because a card that mounts on the first frame would not
        # exercise the late-mount reflow this file is about.
        f"  window.{E2E_DELAY_GLOBAL} = {E2E_DELAY_MS};"
        f"  localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"  localStorage.setItem('unsloth_refresh_token', {json.dumps(session.get('refresh_token', ''))});"
        "  localStorage.setItem('unsloth_show_llama_update_banner', 'true');"
        # A dismissal from an earlier run would hide the very card under test.
        "  for (const k of Object.keys(localStorage))"
        "    if (k.startsWith('unsloth_web_update_dismissed')) localStorage.removeItem(k);"
        "})();"
    )

    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        info(f"FAIL unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
        return 1

    with sync_playwright() as p:
        install_wall_clock_watchdog(
            WALL_TIMEOUT_S,
            label = "ui-update-banner",
            info = info,
        )
        launch_kwargs: dict = {"headless": True}
        if PLAYWRIGHT_BROWSER == "chromium":
            launch_kwargs["args"] = chromium_launch_args()
            if PLAYWRIGHT_CHANNEL:
                launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL
        elif PLAYWRIGHT_CHANNEL:
            info("FAIL STUDIO_PLAYWRIGHT_CHANNEL requires chromium")
            return 1
        browser = getattr(p, PLAYWRIGHT_BROWSER).launch(**launch_kwargs)

        llama_payload = [LLAMA_STATUS]
        for width, height in VIEWPORTS[2:5] if SPOT else VIEWPORTS:
            context = browser.new_context(
                viewport = {"width": width, "height": height},
                reduced_motion = "reduce",
            )
            context.add_init_script(seed_js)
            context.route(
                "**/api/studio/update-status*",
                lambda route: route.fulfill(
                    status = 200,
                    content_type = "application/json",
                    body = json.dumps(UPDATE_STATUS),
                ),
            )
            context.route(
                "**/api/studio/release-notes*",
                lambda route: route.fulfill(
                    status = 200,
                    content_type = "application/json",
                    body = json.dumps(RELEASE_NOTES),
                ),
            )
            context.route(
                "**/api/llama/update-status*",
                lambda route: route.fulfill(
                    status = 200,
                    content_type = "application/json",
                    body = json.dumps(llama_payload[0]),
                ),
            )
            context.route("**/api/llama/update-changelog*", stub(LLAMA_CHANGELOG))
            page = context.new_page()
            for name, path in ROUTES[:1] if SPOT else ROUTES:
                size = f"{width}x{height}"
                boot(page, path)
                if path == "/":
                    check(
                        f"{size} {name}: the app update card is on screen",
                        page.locator('[data-testid="web-update-banner"]').count() == 1,
                        "nothing to measure, so every other check is vacuous",
                    )
                measure(page, f"{size} {name} collapsed")
                page.screenshot(path = str(ART / f"{size}-{path.strip('/') or 'new-chat'}.png"))

                toggle = page.locator('[data-testid="web-update-release-notes-toggle"]')
                if toggle.count() == 1:
                    toggle.click()
                    page.wait_for_timeout(1500)
                    measure(page, f"{size} {name} expanded")
                    toggle.click()
                if path == "/":
                    exercise_llama_changelog(page, f"{size} {name}")

            # whisper.cpp renames the same card rather than adding a second one.
            llama_payload[0] = WHISPER_STATUS
            boot(page, "/")
            facts = measure(page, f"{width}x{height} new chat whisper")
            check(
                f"{width}x{height}: whisper.cpp reuses the one runtime card",
                page.locator('[data-testid="llama-update-banner"]').count() <= 1
                and "whisper.cpp" in facts["llamaText"],
                f"text={facts['llamaText']!r}",
            )
            llama_payload[0] = LLAMA_STATUS
            context.close()

        # Exercise the compact app card omitted by the notes-bearing fixtures.
        for width, height in NO_PREVIEW_VIEWPORTS[:1] if SPOT else NO_PREVIEW_VIEWPORTS:
            context = browser.new_context(
                viewport = {"width": width, "height": height},
                reduced_motion = "reduce",
            )
            context.add_init_script(seed_js)
            for pattern, payload in (
                ("**/api/studio/update-status*", UPDATE_STATUS),
                ("**/api/studio/release-notes*", RELEASE_NOTES_NONE),
                ("**/api/llama/update-status*", LLAMA_STATUS),
                ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
            ):
                context.route(pattern, stub(payload))
            page = context.new_page()
            boot(page, "/")
            panel = page.locator('[data-testid="update-release-notes-panel"]')
            check(
                f"{width}x{height} with no preview: the collapsed card shows no notes",
                panel.count() == 0,
                "the card is at its full height, so this pass proves nothing",
            )
            measure(page, f"{width}x{height} with no preview")
            page.screenshot(path = str(ART / f"{width}x{height}-no-preview.png"))
            # Expanded, the panel is back and so is the floor it pays for.
            toggle = page.locator('[data-testid="web-update-release-notes-toggle"]')
            if toggle.count() == 1:
                toggle.click()
                panel.wait_for(state = "visible", timeout = 10_000)
                page.wait_for_timeout(SETTLED_MS)
                measure(page, f"{width}x{height} with no preview, expanded")
            context.close()

        # The loaded models indicator, switched on. It is the last child of the
        # rail, so it lands on the corner, and its arrival used to re-measure
        # the rail and move the whole stack. That is the case to check.
        # #8346 ships it off by default, so nothing above this point sees it.
        for width, height in INDICATOR_VIEWPORTS:
            context = browser.new_context(
                viewport = {"width": width, "height": height},
                reduced_motion = "reduce",
            )
            context.add_init_script(seed_js)
            context.add_init_script(
                "localStorage.setItem('unsloth_show_loaded_models_indicator', 'true');"
            )
            # The card only exists when something is loaded, so the preference alone would leave the rail exactly as it
            # was and this whole pass would agree with itself about nothing.
            for pattern, payload in (
                ("**/api/studio/update-status*", UPDATE_STATUS),
                ("**/api/studio/release-notes*", RELEASE_NOTES),
                ("**/api/llama/update-status*", LLAMA_STATUS),
                ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
                ("**/api/inference/status", CHAT_LOADED),
                ("**/api/inference/images/status", NOTHING_DIFFUSION),
                ("**/api/inference/video/status", NOTHING_VIDEO),
                ("**/api/inference/audio/stt/status", NOTHING_STT),
            ):
                context.route(pattern, stub(payload))
            page = context.new_page()
            boot(page, "/")
            page.wait_for_selector("text=Loaded models", timeout = 30_000)
            # A third card changes the rail's height, which is what used to move it; give the layout a frame to prove
            # it does not.
            settle_stack(page)
            measure(page, f"{width}x{height} with the models indicator")
            seen = page.evaluate(RAIL_CORNER)
            check(
                f"{width}x{height}: the models indicator is actually up",
                seen is not None and seen["indicator"],
                f"{seen}, so the corner checks below prove nothing",
            )
            check(
                f"{width}x{height}: the rail is still in its bottom-right corner",
                seen is not None and seen["fromBottom"] == 16 and seen["fromRight"] == 16,
                f"{seen}, so the rail has left the corner it is anchored to",
            )
            check(
                f"{width}x{height}: nothing places the rail from JS",
                seen is not None
                and not seen["railStyle"]["bottom"]
                and not seen["railStyle"]["maxHeight"],
                f"{seen['railStyle'] if seen else seen}, so an inline offset or"
                " cap is back on the rail",
            )
            page.screenshot(path = str(ART / f"{width}x{height}-indicator.png"))
            context.close()

        if SPOT:
            info(f"{checks[0]} checks, {len(failures)} failed")
            for failure in failures:
                info(f"  {failure}")
            browser.close()
            return 1 if failures else 0

        # One page, many window sizes.
        # Every check the core matrix runs, at every resolution in RESIZE_SWEEP, for the price of one boot: the cards
        # are already mounted and a resize is all a maximise or a restore ever is. It also exercises the path a fresh
        # load never does, where the placement has to re-measure rather than measure once.
        context = browser.new_context(
            viewport = {"width": RESIZE_SWEEP[0][0], "height": RESIZE_SWEEP[0][1]},
            reduced_motion = "reduce",
        )
        context.add_init_script(seed_js)
        for pattern, payload in (
            ("**/api/studio/update-status*", UPDATE_STATUS),
            ("**/api/studio/release-notes*", RELEASE_NOTES),
            ("**/api/llama/update-status*", LLAMA_STATUS),
            ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
        ):
            context.route(pattern, stub(payload))
        page = context.new_page()
        boot(page, "/")
        for width, height in RESIZE_SWEEP:
            page.set_viewport_size({"width": width, "height": height})
            # Long enough for the ResizeObserver, the placement it feeds and the reflow that follows. Short because
            # nothing is being fetched.
            page.wait_for_timeout(RESIZE_SETTLE_MS)
            settle_stack(page)
            measure(page, f"{width}x{height} resized")
        page.set_viewport_size({"width": 1280, "height": 830})
        page.wait_for_timeout(RESIZE_SETTLE_MS)
        page.screenshot(path = str(ART / "resize-sweep-end.png"))

        # Parked small and brought back. A minimised window cannot be photographed, but the restore is where a cached
        # measurement would show, and the claim is that it lands where a fresh load of the same size does rather than
        # merely looking tidy.
        for (small_w, small_h), (back_w, back_h) in RESTORE_CYCLES:
            page.set_viewport_size({"width": small_w, "height": small_h})
            page.wait_for_timeout(RESIZE_SETTLE_MS)
            page.set_viewport_size({"width": back_w, "height": back_h})
            page.wait_for_timeout(RESIZE_SETTLE_MS)
            restored = measure(page, f"{back_w}x{back_h} restored from {small_w}x{small_h}")
            fresh_context = browser.new_context(
                viewport = {"width": back_w, "height": back_h},
                reduced_motion = "reduce",
            )
            fresh_context.add_init_script(seed_js)
            for pattern, payload in (
                ("**/api/studio/update-status*", UPDATE_STATUS),
                ("**/api/studio/release-notes*", RELEASE_NOTES),
                ("**/api/llama/update-status*", LLAMA_STATUS),
                ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
            ):
                fresh_context.route(pattern, stub(payload))
            fresh_page = fresh_context.new_page()
            boot(fresh_page, "/")
            fresh = measure(fresh_page, f"{back_w}x{back_h} fresh")
            for name in ("card", "llama", "footer"):
                a, b = restored[name], fresh[name]
                check(
                    f"{back_w}x{back_h}: the {name} restores to where a fresh load puts it",
                    a is not None
                    and b is not None
                    and abs(a["top"] - b["top"]) <= 1.0
                    and abs(a["height"] - b["height"]) <= 1.0,
                    f"restored={a} fresh={b}",
                )
            fresh_context.close()
        context.close()

        # Settings > Appearance scales the type, and the card's floor is written against that scale rather than measured
        # once at the default. At the 20px maximum the action row wraps at every card width, so a default-font floor
        # left the buttons clipped inside the card.
        #
        # Set on the server and put back in a finally, because the appearance store syncs up: leaving it at 20px hands
        # every later suite in this job an Unsloth whose type is not the default, and they will not notice.
        # Put BACK what was there, which is not always the default: run this against your own Unsloth and an
        # unconditional reset would take your Appearance setting with it.
        was = read_ui_font_size(session["access_token"])
        set_ui_font_size(session["access_token"], UI_FONT_SIZE_MAX)
        try:
            for width, height in FONT_SCALE_VIEWPORTS:
                context = browser.new_context(
                    viewport = {"width": width, "height": height},
                    reduced_motion = "reduce",
                )
                context.add_init_script(seed_js)
                for pattern, payload in (
                    ("**/api/studio/update-status*", UPDATE_STATUS),
                    ("**/api/studio/release-notes*", RELEASE_NOTES),
                    ("**/api/llama/update-status*", LLAMA_STATUS),
                    ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
                ):
                    context.route(pattern, stub(payload))
                page = context.new_page()
                boot(page, "/")
                scale = page.evaluate(
                    "() => getComputedStyle(document.documentElement)"
                    ".getPropertyValue('--ui-font-scale').trim()"
                )
                check(
                    f"{width}x{height} at {UI_FONT_SIZE_MAX}px: the type is actually scaled",
                    scale not in ("", str(UI_FONT_SIZE_DEFAULT / UI_FONT_SIZE_CSS_BASE)),
                    f"--ui-font-scale={scale!r}, so the rest of this pass proves nothing",
                )
                measure(page, f"{width}x{height} at {UI_FONT_SIZE_MAX}px")
                page.screenshot(path = str(ART / f"{width}x{height}-font{UI_FONT_SIZE_MAX}.png"))
                context.close()
        finally:
            set_ui_font_size(session["access_token"], was)
        browser.close()

    info(f"{checks[0]} checks, {len(failures)} failed")
    for failure in failures:
        info(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
