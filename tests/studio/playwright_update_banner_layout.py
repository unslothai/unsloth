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
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    goto_with_socket_backoff,
    install_wall_clock_watchdog,
    report_failing_step,
    step_budget_s,
    wait_for_health,
    wait_for_settled,
)

# page.evaluate takes no timeout, and this suite runs mid-lane on Windows.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-update-banner"))
ART.mkdir(parents = True, exist_ok = True)

PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# The web check fires 5s after mount and llama.cpp's after 1s; this is the ceiling.
SETTLE_MS = int(os.environ.get("STUDIO_UI_BANNER_SETTLE_MS", "9000"))
SETTLED_MS = int(os.environ.get("STUDIO_UI_BANNER_SETTLED_MS", "10000"))

# Must match the name use-web-update-check.ts reads. Short but nonzero so the card still mounts late.
E2E_DELAY_GLOBAL = "__unslothE2EWebUpdateDelayMs"
E2E_DELAY_MS = int(os.environ.get("STUDIO_UI_BANNER_UPDATE_DELAY_MS", "150"))

LATEST = "2099.1.0"

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
RELEASE_NOTES_NONE = dict(RELEASE_NOTES, markdown = "", matched = False)
# Inherited overflow-wrap:anywhere split "Download" in the Link column.
_DL = "https://github.com/unslothai/unsloth/releases/latest/download"
RELEASE_NOTES_TABLE = dict(
    RELEASE_NOTES,
    markdown = "\n".join(
        [
            NOTES_MARKDOWN,
            "",
            "## Download Unsloth Desktop",
            "",
            "| Platform | Link |",
            "|---|---|",
            f"| macOS (Apple Silicon, M1 or newer, macOS 12 and later) | [Download]({_DL}/Unsloth-Desktop-MacOS.dmg) |",
            f"| Windows 10 / 11 (x64 installer with automatic updates) | [Download]({_DL}/Unsloth-Desktop-Windows.exe) |",
            f"| Linux AppImage (x86_64, runs on most distributions) | [Download]({_DL}/Unsloth-Desktop-Linux.AppImage) |",
            "",
            "| Platform | URL |",
            "|---|---|",
            f"| macOS | {_DL}/Unsloth-Desktop-MacOS.dmg |",
        ]
    ),
)
TABLE_VIEWPORTS = [(1440, 900), (390, 844)]
NOTES_TABLES = """
() => {
  const scroll = document.querySelector('[data-testid="update-release-notes-scroll"]');
  const card = document.querySelector('[data-testid="web-update-banner"]');
  if (!scroll || !card) return null;
  const right = card.getBoundingClientRect().right;
  return [...scroll.querySelectorAll('table')].map((table) => ({
    links: [...table.querySelectorAll('td a')].map((a) => [a.textContent, a.getClientRects().length]),
    pastCard: Math.max(0, table.parentElement.getBoundingClientRect().right - right),
  }));
}
"""
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

# 921x534 and 768x500 reproduce the report; taller ones prove the fix costs nothing.
VIEWPORTS = [
    (1440, 900),
    (1280, 830),
    (921, 534),
    (768, 500),
    (390, 844),
    (390, 500),
]
ROUTES = [("new chat", "/"), ("train", "/train"), ("model hub", "/model-hub")]

# Resize one loaded page rather than booting one per size; a resize is what maximise and restore are.
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

RESIZE_SETTLE_MS = int(os.environ.get("STUDIO_UI_BANNER_RESIZE_MS", "10000"))

PHASE_BUDGET_S = step_budget_s(180)
SWEEP_BUDGET_S = step_budget_s(420)

# Shaped as in playwright_loaded_models_indicator.py; one loaded chat model puts the card in the rail.
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

INDICATOR_VIEWPORTS = [(921, 534), (768, 500)]

NO_PREVIEW_VIEWPORTS = [(1440, 900), (921, 534)]

# The rail carries a shadow gutter, so this is the cards' inset; see RAIL_CORNER.
CORNER_INSET_PX = 16

# Nothing on the page may move the rail off its corner.
RAIL_CORNER = """
() => {
  const card = document.querySelector('[data-testid="web-update-banner"]');
  if (!card) return null;
  const rail = card.parentElement;
  const a = rail.getBoundingClientRect();
  // Cards in the rail's own flow. A dragged loaded models card is `fixed`
  // somewhere else and says nothing about the rail; `relative` and `sticky`
  // still take part in layout, so only the out-of-flow pair is excluded.
  const flowed = Array.from(rail.children).filter(
    (kid) => !['absolute', 'fixed'].includes(getComputedStyle(kid).position),
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

# A ResizeObserver measurement cached in state would survive a park-and-restore.
RESTORE_CYCLES = [((320, 400), (1920, 1080)), ((320, 400), (900, 600))]

# `spot` is the cut-down pass for Firefox/WebKit, which share a job with little time to spare.
SCOPE = os.environ.get("STUDIO_UI_BANNER_SCOPE", "full").lower()
SPOT = SCOPE == "spot"

# At the 20px font size below 384px the action row wraps; 320x480 has a rail cap between the two
# card heights, which is what clipped Copy command.
FONT_SCALE_VIEWPORTS = [(921, 534), (390, 500), (320, 480)]
# Mirrors appearance-custom-store.ts; kept in step by test_update_release_notes.py.
UI_FONT_SIZE_MAX = 20
UI_FONT_SIZE_DEFAULT = 15
UI_FONT_SIZE_CSS_BASE = 16
# Read through a length since the properties are calc()s (--ui-space-scale since #11648).
UI_SPACE_SCALE_JS = "(() => { const probe = document.createElement('div'); probe.style.cssText = 'position:absolute;visibility:hidden;width:calc(10000px * var(--ui-space-scale, 1))'; document.body.appendChild(probe); const px = parseFloat(getComputedStyle(probe).width); probe.remove(); return String(px / 10000); })()"
UI_FONT_SCALE_JS = "(() => { const probe = document.createElement('div'); probe.style.cssText = 'position:absolute;visibility:hidden;width:calc(10000px * var(--ui-font-scale, 1))'; document.body.appendChild(probe); const px = parseFloat(getComputedStyle(probe).width); probe.remove(); return String(px / 10000); })()"
APPEARANCE_STORE_VERSION = 5

failures: list[str] = []
checks = [0]
_watchdog = [None]


def info(s: str) -> None:
    print(f"[banner] {s}", flush = True)


def phase(name: str, budget_s: float = PHASE_BUDGET_S) -> None:
    """Start phase `name`; it may run `budget_s` before the run stops, naming it."""
    info(f"STEP {name}")
    if _watchdog[0] is not None:
        _watchdog[0].begin_step(name, budget_s)


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
  // By its own handle, not through whichever banner happens to be up: reaching
  // the rail via a card finds nothing when no card is up, and "no cards" is a
  // state the download panel and the loaded models card have to be judged in.
  const rail = document.querySelector('[data-testid="overlay-rail"]')
            || (card ? card.parentElement : (llama ? llama.parentElement : null));
  // Cards in the rail's own flow. A dragged loaded models card is `fixed`
  // somewhere else and says nothing about the rail; `relative` and `sticky`
  // still take part in layout, so only the out-of-flow pair is excluded.
  const flowed = rail
    ? [...rail.children].filter(
        (kid) => !['absolute', 'fixed'].includes(getComputedStyle(kid).position))
    : [];
  // The clipper is the card's inner surface, the one with overflow-hidden; the
  // rail-facing root above it is overflow-visible and clips nothing.
  const surface = card ? card.firstElementChild : null;
  return {
    viewport: {width: innerWidth, height: innerHeight},
    // Clipped by the card, not by the rail. What the rail hides is under a
    // fold the reader can scroll to; what the card hides is gone for good.
    card: rect(card), llama: rect(llama),
    cardDead: dead(card), llamaDead: dead(llama),
    // Every card in the rail, not only the two update banners. #10117 put an
    // ungated floor on a card that had none the day before, so naming the cards
    // to check is naming the ones that have already gone wrong.
    railCards: flowed.map((kid) => ({
      // Enough to name the offender in the failure without opening a browser.
      tag: kid.tagName.toLowerCase(),
      testid: kid.getAttribute('data-testid'),
      cls: (kid.getAttribute('class') || '').slice(0, 120),
      painted: kid.firstElementChild
        ? (kid.firstElementChild.getAttribute('class') || '').slice(0, 80)
        : null,
      dead: dead(kid),
    })),
    // Where the bottom-most card ACTUALLY paints, against the corner it is
    // anchored to. `fromBottom` in RAIL_CORNER comes off the rail's padding
    // box, which is `fixed bottom-0` and so reports the intended offset however
    // much dead air sits inside the last card. That is why the rail passed its
    // own corner check all through #10117. This reads the painted surface.
    lastPaintedFromBottom: (() => {
      const last = flowed[flowed.length - 1];
      const paint = last && (last.firstElementChild || last);
      if (!paint) return null;
      return Math.round((innerHeight - paint.getBoundingClientRect().bottom) * 10) / 10;
    })(),
    // The same two, as much of them as the rail is SHOWING. Containment is
    // asked of these: a card the rail has folded away is not on screen at all,
    // and judging its unclipped rect against the viewport fails it for being
    // scrolled out of sight, which is what the reach check below is for.
    cardShown: clip(card, rail), llamaShown: clip(llama, rail),
    notesBody: clip(body, notes),
    toggle: clip(toggle, surface),
    snooze: clip(snooze, surface),
    copy: clip(copy, surface),
    // The same three unclipped, so a control the card has cut DOWN is as visible
    // here as one it cut away. Half a button is not the button the card promises.
    toggleWhole: rect(toggle),
    snoozeWhole: rect(snooze),
    copyWhole: rect(copy),
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


# "Below the fold" is a pass; "cannot be brought into view" is the failure.
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
    return round(min(dy, dx), 1) if dy > 0.5 and dx > 0.5 else 0.0


def inside(box: dict | None, viewport: dict) -> bool:
    # clip() returns None for an entirely hidden element; that must not read as inside.
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
        shown = facts.get(f"{name}Shown", facts[name]) if name in ("card", "llama") else facts[name]
        if name in ("card", "llama") and shown is None:
            continue
        check(
            f"{label}: the {name} stays inside the viewport",
            inside(shown, view),
            f"{name}={shown} viewport={view}",
        )
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
        # 448px is the card's max width, scaled with the UI since #11648; 2rem is its viewport inset.
        space = float(page.evaluate("() => " + UI_SPACE_SCALE_JS))
        want = min(448 * space, view["width"] - 32)
        # Measure the layout box: the enter animation (scale .96) shrinks only the painted one.
        check(
            f"{label}: the card keeps its full width whatever the scrollbar does",
            abs(facts["cardLayoutWidth"] - want) <= 1,
            f"cardLayoutWidth={facts['cardLayoutWidth']} want={want} spaceScale={space} "
            f"cardPaintedWidth={facts['cardWidth']} "
            f"railGutter={facts['railGutterPx']} scrolls={facts['railScrolls']} "
            f"why={json.dumps(facts['widthWhy'], sort_keys = True)}",
        )
    if facts["railScrolls"] is not None:
        scrolls = facts["railScrolls"]
        # Click-through in every state; the fold is reached by wheel over a card or by focus.
        check(
            f"{label}: the rail stays click-through",
            facts["railPointerEvents"] == "none",
            f"scrolls={scrolls} pointerEvents={facts['railPointerEvents']} "
            f"why={json.dumps(facts['widthWhy'], sort_keys = True)}",
        )
        check(
            f"{label}: the rail's gutter never swallows a click",
            facts["gutterIsRail"] is False,
            f"scrolls={scrolls} gutterIsRail={facts['gutterIsRail']}",
        )
    # The rail is bottom-anchored, so one card's dead space lifts the rest.
    cards = facts["railCards"]
    check(
        f"{label}: the rail has cards to judge",
        len(cards) > 0,
        f"railCards={cards}, so every dead-space check below passed vacuously",
    )
    for index, kid in enumerate(cards):
        hole = kid["dead"]
        if hole is None:
            continue
        who = kid["testid"] or kid["cls"] or f"{kid['tag']}#{index}"
        check(
            f"{label}: {who} reserves no height it does not paint",
            hole["above"] <= 1.0 and hole["below"] <= 1.0,
            f"dead={hole} painted-child={kid['painted']}",
        )
    if cards and not facts["railScrolls"]:
        check(
            f"{label}: the bottom card is on the corner, not floating above it",
            facts["lastPaintedFromBottom"] == CORNER_INSET_PX,
            f"lastPaintedFromBottom={facts['lastPaintedFromBottom']} "
            f"(want {CORNER_INSET_PX}) last={cards[-1]}",
        )
    for name in ("card", "llama", "toggle", "snooze", "copy"):
        box = facts[name]
        check(
            f"{label}: the card does not clip its own {name} away",
            box is not None and box["height"] > 1.0 and box["width"] > 1.0,
            f"{name}={box}",
        )
    # A control cut down, not just clipped to nothing, also means the card's floor is wrong.
    for name in ("toggle", "snooze", "copy"):
        box, whole = facts[name], facts[f"{name}Whole"]
        check(
            f"{label}: the card shows all of its own {name}",
            box is not None
            and whole is not None
            and box["height"] >= whole["height"] - 1.0
            and box["width"] >= whole["width"] - 1.0,
            f"{name}={box} whole={whole}",
        )
    return facts


# #9849 made the Downloads overlay always mount, and was reverted by #10298.
DOWNLOADS = """
() => {
  const rail = document.querySelector('[data-testid="overlay-rail"]');
  if (!rail) return null;
  return {
    // The panel and its collapsed FAB. Neither may exist with no jobs.
    panels: document.querySelectorAll('.hub-download-panel, .hub-download-fab').length,
    // Slots, not pixels: a panel that renders a wrapper around nothing is
    // invisible but still takes a flex slot and its gap-2, which pushes the
    // card that is meant to hold the corner up off it.
    cards: [...rail.children].filter(
      (kid) => !['absolute', 'fixed'].includes(getComputedStyle(kid).position)).length,
  };
}
"""


def check_downloads_absent(page, label: str, expected_cards: int) -> None:
    """With no transfers, the Downloads overlay must not be in the rail at all."""
    seen = page.evaluate(DOWNLOADS)
    check(
        f"{label}: the rail is reachable by its own handle",
        seen is not None,
        "no [data-testid=overlay-rail] on the page, so the two checks below prove nothing",
    )
    if seen is None:
        return
    check(
        f"{label}: no Downloads overlay while there is nothing downloading",
        seen["panels"] == 0,
        f"{seen['panels']} download panel(s) mounted with an empty job list, "
        "which is the permanent corner FAB from #9849",
    )
    # An exact count: a bare absence also holds when the selector has rotted.
    check(
        f"{label}: the rail holds exactly the cards that are up",
        seen["cards"] == expected_cards,
        f"cards={seen['cards']} want={expected_cards}; an extra slot is an "
        "overlay mounting when it has nothing to show",
    )


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


def settle_cards(page, timeout_ms: int = SETTLED_MS) -> None:
    """Wait for the rail and each card in it to hold one box with nothing animating.

    Replaces a fixed pause after mount or resize: it ends as soon as the stack is still, and on a runner slow enough
    that the pause was not enough it keeps waiting instead of measuring mid-transition. A stack that never settles is
    reported and measured anyway, so the checks, not this wait, say what is wrong with it.
    """
    for selector in (
        '[data-testid="overlay-rail"]',
        '[data-testid="web-update-banner"]',
        '[data-testid="llama-update-banner"]',
    ):
        target = page.locator(selector)
        if target.count() == 0:
            continue
        # Re-resolve each slice: a card React remounts leaves a held handle detached.
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                info(f"WARN {selector} did not settle within {timeout_ms}ms; measuring anyway")
                break
            if target.count() == 0:
                break
            try:
                wait_for_settled(target, timeout_ms = min(2_000, remaining_ms))
                break
            except PlaywrightTimeoutError:
                continue


def boot(page, path: str) -> None:
    goto_with_socket_backoff(page, f"{BASE}{path}", wait_until = "domcontentloaded")
    # The seed script shortens the app card's 5s to E2E_DELAY_MS; llama.cpp keeps its 1s.
    for testid in ("web-update-banner", "llama-update-banner"):
        try:
            page.wait_for_selector(f'[data-testid="{testid}"]', state = "attached", timeout = SETTLE_MS)
        except PlaywrightTimeoutError:
            pass
    settle_cards(page)
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


LLAMA_CHANGELOG_ROWS_IN_VIEW = """
() => {
  const list = document.querySelector('[data-testid="llama-update-changelog-list"]');
  if (!list) return null;
  const box = list.getBoundingClientRect();
  const style = getComputedStyle(list);
  const top = box.top + list.clientTop + parseFloat(style.paddingTop);
  const bottom = box.top + list.clientTop + list.clientHeight - parseFloat(style.paddingBottom);
  const items = [...list.children];
  const whole = items.filter((item) => {
    const r = item.getBoundingClientRect();
    return r.height > 0 && r.top >= top - 0.5 && r.bottom <= bottom + 0.5;
  }).length;
  return {whole, items: items.length, viewport: Math.round(bottom - top), list: Math.round(box.height)};
}
"""


def settle_llama_changelog(page, timeout_s: float = 3.0) -> None:
    """Return once the open changelog's height has held for three frames in a row."""
    deadline = time.monotonic() + timeout_s
    last, steady = None, 0
    while time.monotonic() < deadline:
        height = page.evaluate(
            """() => new Promise((resolve) => requestAnimationFrame(() => {
              const list = document.querySelector('[data-testid="llama-update-changelog-list"]');
              resolve(list ? list.getBoundingClientRect().height : null);
            }))"""
        )
        steady = steady + 1 if height == last else 0
        if steady >= 3:
            return
        last = height
    info(f"WARN the open llama.cpp changelog never held one height for 3 frames (last={last})")


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
    # The open card lays out full height, then shrinks a frame or two later; read the settled one.
    settle_llama_changelog(page)
    # textContent, not innerText: WebKit's innerText drops text clipped out of its scroller.
    text = listing.text_content() or ""
    check(
        f"{label}: expansion shows only the new carried changes",
        "GLM-5-Next" in text
        and "MTP for Qwen3.8-Flash-Next" in text
        and "Add TML Inkling" not in text,
        f"list={text!r}",
    )
    # Reported, not gated: at 768x500 with both cards up no whole change fits on any engine.
    rows = page.evaluate(LLAMA_CHANGELOG_ROWS_IN_VIEW)
    info(f"{label}: open llama.cpp changelog shows {rows}")
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
    # OLD may already be NEW on a rerun or after an earlier suite rotated it.
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
            if exc.code not in (400, 401, 403):
                raise
    session = api("/api/auth/login", {"username": "unsloth", "password": NEW})

    # add_init_script takes raw source, not a function to call.
    seed_js = (
        "(() => {"
        # Shortens the 5s update-check timer only in this page, but keeps it a timer so the late-mount
        # reflow is still exercised.
        f"  window.{E2E_DELAY_GLOBAL} = {E2E_DELAY_MS};"
        f"  localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"  localStorage.setItem('unsloth_refresh_token', {json.dumps(session.get('refresh_token', ''))});"
        "  localStorage.setItem('unsloth_show_llama_update_banner', 'true');"
        # A dismissal from an earlier run would hide the card under test.
        "  for (const k of Object.keys(localStorage))"
        "    if (k.startsWith('unsloth_web_update_dismissed')) localStorage.removeItem(k);"
        "})();"
    )

    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        info(f"FAIL unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
        return 1

    with sync_playwright() as p:
        # begin_step() restarts the inactivity budget, so the same number is also the whole-run total.
        _watchdog[0] = install_wall_clock_watchdog(
            WALL_TIMEOUT_S,
            label = "ui-update-banner",
            info = info,
            total_deadline_s = WALL_TIMEOUT_S,
        )
        report_failing_step(_watchdog[0], label = "ui-update-banner")
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
            phase(f"update cards at {width}x{height}")
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
                if path == "/":
                    # Indicator is off by default (#8346).
                    check_downloads_absent(page, f"{size} {name}", 2)
                page.screenshot(path = str(ART / f"{size}-{path.strip('/') or 'new-chat'}.png"))

                toggle = page.locator('[data-testid="web-update-release-notes-toggle"]')
                if toggle.count() == 1:
                    toggle.click()
                    try:
                        page.wait_for_selector(
                            '[data-testid="web-update-release-notes-toggle"][aria-expanded="true"]',
                            state = "attached",
                            timeout = 10_000,
                        )
                    except PlaywrightTimeoutError:
                        info(
                            f"WARN {size} {name}: the notes toggle never reported aria-expanded=true"
                        )
                    settle_cards(page)
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

        for width, height in NO_PREVIEW_VIEWPORTS[:1] if SPOT else NO_PREVIEW_VIEWPORTS:
            phase(f"app card with no notes preview at {width}x{height}")
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
            toggle = page.locator('[data-testid="web-update-release-notes-toggle"]')
            if toggle.count() == 1:
                toggle.click()
                panel.wait_for(state = "visible", timeout = 10_000)
                settle_cards(page)
                measure(page, f"{width}x{height} with no preview, expanded")
            context.close()

        for width, height in TABLE_VIEWPORTS[:1] if SPOT else TABLE_VIEWPORTS:
            size = f"{width}x{height}"
            phase(f"release-notes tables at {size}")
            context = browser.new_context(
                viewport = {"width": width, "height": height},
                reduced_motion = "reduce",
            )
            context.add_init_script(seed_js)
            for pattern, payload in (
                ("**/api/studio/update-status*", UPDATE_STATUS),
                ("**/api/studio/release-notes*", RELEASE_NOTES_TABLE),
                ("**/api/llama/update-status*", LLAMA_STATUS),
                ("**/api/llama/update-changelog*", LLAMA_CHANGELOG),
            ):
                context.route(pattern, stub(payload))
            page = context.new_page()
            boot(page, "/")
            page.locator('[data-testid="web-update-release-notes-toggle"]').click()
            page.wait_for_selector(
                '[data-testid="update-release-notes-scroll"] table', timeout = 10_000
            )
            settle_cards(page)
            tables = page.evaluate(NOTES_TABLES) or []
            words = tables[0]["links"] if tables else []
            check(
                f"{size}: every Download link in the notes table is on one line",
                len(words) == 3 and all(text == "Download" and lines == 1 for text, lines in words),
                f"links={words}",
            )
            check(
                f"{size}: no notes table paints past the card",
                len(tables) == 2 and all(t["pastCard"] <= 0.5 for t in tables),
                f"tables={tables}",
            )
            page.screenshot(path = str(ART / f"{size}-notes-tables.png"))
            context.close()

        # The indicator is the rail's last child, and its arrival used to re-measure and move the stack.
        for width, height in INDICATOR_VIEWPORTS:
            phase(f"loaded models indicator in the rail at {width}x{height}")
            context = browser.new_context(
                viewport = {"width": width, "height": height},
                reduced_motion = "reduce",
            )
            context.add_init_script(seed_js)
            context.add_init_script(
                "localStorage.setItem('unsloth_show_loaded_models_indicator', 'true');"
            )
            # The card only exists when something is loaded.
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
                seen is not None
                and seen["fromBottom"] == CORNER_INSET_PX
                and seen["fromRight"] == CORNER_INSET_PX,
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

        phase("resize sweep and restore cycles", SWEEP_BUDGET_S)
        # Resize one page through every size; this also exercises re-measurement, which a fresh load never does.
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
            settle_cards(page, RESIZE_SETTLE_MS)
            settle_stack(page)
            measure(page, f"{width}x{height} resized")
        page.set_viewport_size({"width": 1280, "height": 830})
        settle_cards(page, RESIZE_SETTLE_MS)
        page.screenshot(path = str(ART / "resize-sweep-end.png"))

        for (small_w, small_h), (back_w, back_h) in RESTORE_CYCLES:
            page.set_viewport_size({"width": small_w, "height": small_h})
            # Park long enough to lay out small, or the restore restores nothing.
            settle_cards(page, RESIZE_SETTLE_MS)
            page.set_viewport_size({"width": back_w, "height": back_h})
            settle_cards(page, RESIZE_SETTLE_MS)
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

        # The appearance store syncs to the server, so restore the previous value (not the default) in finally.
        was = read_ui_font_size(session["access_token"])
        set_ui_font_size(session["access_token"], UI_FONT_SIZE_MAX)
        try:
            for width, height in FONT_SCALE_VIEWPORTS:
                phase(f"{UI_FONT_SIZE_MAX}px type at {width}x{height}")
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
                # Resolve through a length: --ui-font-scale is a calc() and the raw string never equals the default.
                scale = float(page.evaluate("() => " + UI_FONT_SCALE_JS))
                check(
                    f"{width}x{height} at {UI_FONT_SIZE_MAX}px: the type is actually scaled",
                    abs(scale - UI_FONT_SIZE_DEFAULT / UI_FONT_SIZE_CSS_BASE) > 1e-3,
                    f"--ui-font-scale resolved to {scale!r}, the default, so the rest of this pass proves nothing",
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
