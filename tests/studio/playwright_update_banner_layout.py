# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The overlay rail's update banners must never print over each other.

The reported failure: on New chat, in a window short enough that the composer
crowds the rail, the app-update card's release notes were painted over its own
row of buttons. Train and Model hub were fine, because only the chat routes
publish a composer box into the frame store and only that box caps the rail.

The node suite cannot catch this: it is a flex shrink across a capped column, so
it needs a real layout, a real ResizeObserver and the real route, and it shows
only at some viewport heights. Rects are intersected with whatever clips them,
so anything an overflow-hidden ancestor hides does not count as visible.

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

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-update-banner"))
ART.mkdir(parents = True, exist_ok = True)

PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# The web check fires 5s after mount and the llama.cpp one after 1s.
SETTLE_MS = int(os.environ.get("STUDIO_UI_BANNER_SETTLE_MS", "9000"))

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
        "- Studio desktop update flow reworked.",
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
LLAMA_STATUS = {
    "supported": True,
    "update_available": True,
    "llama_update_available": True,
    "update_component": "llama",
    "installed_tag": "b10333",
    "latest_tag": "b10333-mix-e34b418",
    "update_size_bytes": 28 * 1024 * 1024,
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

# 921x534 and 768x500 are where the report reproduces; the taller ones prove the
# fix costs nothing when there is room.
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
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data = data,
        method = "POST" if data else "GET",
        headers = {"Content-Type": "application/json"}
        | ({"Authorization": f"Bearer {token}"} if token else {}),
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        return json.loads(response.read().decode())


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
    card: clip(card, rail), llama: clip(llama, rail),
    notesBody: clip(body, notes),
    toggle: clip(clip(toggle, surface), rail),
    snooze: clip(clip(snooze, surface), rail),
    copy: clip(clip(copy, surface), rail),
    footer: rect(footer),
    llamaText: llama ? (llama.innerText || '') : '',
    // pointer-events-none costs the rail its scrollbar, so it may only be
    // click-through while there is nothing under the fold to scroll to.
    railScrolls: rail ? rail.scrollHeight > rail.clientHeight : null,
    railPointerEvents: rail ? getComputedStyle(rail).pointerEvents : null,
    // What a click on the rail's own gutter lands on when it is click-through.
    gutterIsRail: rail ? (() => {
      const r = rail.getBoundingClientRect();
      return document.elementFromPoint(
        Math.round(r.right - 2), Math.round(r.top + r.height / 2)) === rail;
    })() : null,
  };
}
"""


def overlap(a: dict | None, b: dict | None) -> float:
    """Pixels of the smaller intersecting side, 0 if they do not intersect."""
    if not a or not b:
        return 0.0
    dy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    dx = min(a["right"], b["right"]) - max(a["left"], b["left"])
    # Half a pixel of touching is a rounded edge, not an overlap.
    return round(min(dy, dx), 1) if dy > 0.5 and dx > 0.5 else 0.0


def inside(box: dict | None, viewport: dict) -> bool:
    # A missing box is not "inside": clip() returns None for an element that is
    # entirely hidden, and reading that as a pass would make this whole suite
    # green on the one failure it exists to catch.
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
        check(
            f"{label}: the {name} stays inside the viewport",
            inside(facts[name], view),
            f"{name}={facts[name]} viewport={view}",
        )
    if facts["railScrolls"] is not None:
        scrolls = facts["railScrolls"]
        check(
            f"{label}: the rail takes pointer input exactly when it scrolls",
            facts["railPointerEvents"] == ("auto" if scrolls else "none"),
            f"scrolls={scrolls} pointerEvents={facts['railPointerEvents']}",
        )
        # Click-through is what pointer-events-none is for, and the gutter is
        # the widest part of the rail that no card covers.
        check(
            f"{label}: a rail with nothing to scroll to stays click-through",
            scrolls or facts["gutterIsRail"] is False,
            f"scrolls={scrolls} gutterIsRail={facts['gutterIsRail']}",
        )
    # The notes are allowed to yield all of their height, and do; the controls
    # are not, and a card clipped to nothing is the failure being tested for.
    for name in ("card", "llama", "toggle", "snooze", "copy"):
        box = facts[name]
        check(
            f"{label}: the {name} is not clipped away",
            box is not None and box["height"] > 1.0 and box["width"] > 1.0,
            f"{name}={box}",
        )
    return facts


def boot(page, path: str) -> None:
    page.goto(f"{BASE}{path}", wait_until = "domcontentloaded")
    page.wait_for_timeout(SETTLE_MS)
    landed = page.evaluate("location.pathname")
    if landed.startswith(("/login", "/onboarding", "/change-password")):
        raise AssertionError(f"not authenticated: landed on {landed}")


def main() -> int:
    wait_for_health(BASE, timeout = 60.0, info = info)
    # OLD is already NEW on a rerun, or when an earlier suite in the same job
    # rotated it, and that login fails before the rotation below can be skipped.
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
        for width, height in VIEWPORTS:
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
            page = context.new_page()
            for name, path in ROUTES:
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
        browser.close()

    info(f"{checks[0]} checks, {len(failures)} failed")
    for failure in failures:
        info(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
