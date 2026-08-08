# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-browser checks for the loaded-models indicator.

The four /status endpoints are stubbed with page.route, so this runs on any
host: no GPU, no model download, no llama.cpp build. That is the point -- the
payload shapes the card has to survive come from hardware most CI runners do not
have (AMD reporting itself as "cuda", Apple's "mps", the sd.cpp engine that omits
model_kind), and stubbing is the only way to exercise all of them anywhere.

What genuinely needs a real engine, and is therefore checked here rather than in
the node suite:

  * the position restore. The card is position:fixed and stores absolute
    viewport coordinates, so one saved on a wide monitor lands off screen on a
    laptop -- taking its own drag handle and collapse button with it. That needs
    a real ResizeObserver, a real layout and a real localStorage.
  * pointer capture during a drag.
  * that polling actually stops when the preference is off.

Run: BASE_URL, STUDIO_OLD_PW and STUDIO_NEW_PW as the other suites take them.
STUDIO_PLAYWRIGHT_BROWSER selects chromium (also Chrome/Edge/WebView2),
firefox, or webkit (also Safari and the Linux WebKitGTK Tauri embeds).
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
    install_view_transition_killer,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-loaded-models"))
ART.mkdir(parents = True, exist_ok = True)

PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# The card polls every 5s; two ticks plus slack is enough to see a change land.
SETTLE_MS = int(os.environ.get("STUDIO_UI_INDICATOR_SETTLE_MS", "12000"))

CARD = 'text="Loaded models"'
EJECT = '[aria-label^="Eject "]'
HANDLE = '[aria-label="Drag to move"]'
POSITION_KEY = "unsloth_loaded_models_position"
COLLAPSED_KEY = "unsloth_loaded_models_collapsed"
SHOW_KEY = "unsloth_show_loaded_models_indicator"

failures: list[str] = []
checks = [0]


def info(s: str) -> None:
    print(f"[indicator] {s}", flush = True)


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


# ── Stub payloads, straight from the backend's own response models ───────

NOTHING_CHAT = {
    "active_model": None,
    "loaded": [],
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


def chat(**overrides) -> dict:
    return dict(NOTHING_CHAT, **overrides)


class Runtime:
    """Mutable stub state, so a scenario can change what a runtime holds
    between polls without tearing the routes down."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.chat = dict(NOTHING_CHAT)
        self.diffusion = dict(NOTHING_DIFFUSION)
        self.video = dict(NOTHING_VIDEO)
        self.stt = json.loads(json.dumps(NOTHING_STT))
        self.hang: set[str] = set()
        self.status_reads = 0
        self.unloads: list[str] = []
        # Routes deliberately left unanswered, kept so teardown can settle them
        # instead of cancelling them out from under the handler.
        self.parked: list = []


def install_routes(context, state: Runtime) -> None:
    def stub(key: str, body):
        def handler(route):
            state.status_reads += 1
            if key in state.hang:
                # Accept the connection and never answer: the read must time
                # out rather than wedge the card forever.
                state.parked.append(route)
                return
            route.fulfill(
                status = 200,
                content_type = "application/json",
                body = json.dumps(body() if callable(body) else body),
            )

        return handler

    context.route("**/api/inference/status", stub("chat", lambda: state.chat))
    context.route("**/api/inference/images/status", stub("image", lambda: state.diffusion))
    context.route("**/api/inference/video/status", stub("video", lambda: state.video))
    context.route("**/api/inference/audio/stt/status", stub("stt", lambda: state.stt))

    def unload_chat(route):
        state.unloads.append("chat")
        state.chat = dict(NOTHING_CHAT)
        route.fulfill(
            status = 200, content_type = "application/json", body = json.dumps({"status": "unloaded"})
        )

    def unload_images(route):
        state.unloads.append("image")
        state.diffusion = dict(NOTHING_DIFFUSION)
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(state.diffusion))

    def unload_video(route):
        state.unloads.append("video")
        state.video = dict(NOTHING_VIDEO)
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(state.video))

    def unload_stt(route):
        state.unloads.append("stt")
        state.stt["transformers"] = {"loaded_model": None, "device": None}
        state.stt["loaded_model"] = None
        route.fulfill(
            status = 200,
            content_type = "application/json",
            body = json.dumps({"loaded_model": None, "device": None}),
        )

    context.route("**/api/inference/unload", unload_chat)
    context.route("**/api/inference/images/unload", unload_images)
    context.route("**/api/inference/video/unload", unload_video)
    context.route("**/api/inference/audio/stt/unload**", unload_stt)


def rows(page) -> list[str]:
    labels = page.locator(EJECT)
    return [labels.nth(i).get_attribute("aria-label") or "" for i in range(labels.count())]


def card_text(page) -> str:
    if page.locator(CARD).count() == 0:
        return ""
    return page.locator(CARD).locator("xpath=ancestor::div[3]").first.inner_text()


def boot(
    page,
    state: Runtime,
    *,
    seed: dict | None = None,
) -> None:
    """Reload with a known localStorage, then wait for the card to settle."""
    page.goto(BASE, wait_until = "domcontentloaded")
    page.evaluate(
        """([seed, keys]) => {
            for (const k of keys) localStorage.removeItem(k);
            for (const [k, v] of Object.entries(seed || {}))
                localStorage.setItem(k, v);
        }""",
        [seed or {}, [POSITION_KEY, COLLAPSED_KEY, SHOW_KEY]],
    )
    page.reload(wait_until = "domcontentloaded")
    page.wait_for_timeout(SETTLE_MS // 2)
    # The card is deliberately hidden on /login, so an auth slip would make
    # every "no card" check pass for the wrong reason.
    path = page.evaluate("location.pathname")
    if path.startswith(("/login", "/onboarding", "/change-password")):
        raise AssertionError(f"not authenticated: landed on {path}")


def main() -> int:
    wait_for_health(BASE, timeout = 60.0, info = info)
    # Bootstrap exactly as the other suites do: the first login forces a change.
    token = api("/api/auth/login", {"username": "unsloth", "password": OLD})["access_token"]
    try:
        api("/api/auth/change-password", {"current_password": OLD, "new_password": NEW}, token)
    except urllib.error.HTTPError as exc:
        # Already rotated by a previous run on the same install.
        if exc.code not in (400, 401, 403):
            raise
    session = api("/api/auth/login", {"username": "unsloth", "password": NEW})
    if session.get("must_change_password"):
        info("FAIL bootstrap left must_change_password set")
        return 1

    # add_init_script takes raw source, not a function to call: an arrow
    # expression here would evaluate to a function nobody invokes, the SPA would
    # find no token, and every check would silently run against /login.
    seed_js = (
        "(() => {"
        f"  localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"  localStorage.setItem('unsloth_refresh_token', {json.dumps(session.get('refresh_token', ''))});"
        "})();"
    )

    state = Runtime()
    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        info(f"FAIL unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
        return 1

    with sync_playwright() as p:
        browser_type = getattr(p, PLAYWRIGHT_BROWSER)
        launch_kwargs: dict = {"headless": True}
        if PLAYWRIGHT_BROWSER == "chromium":
            launch_kwargs["args"] = chromium_launch_args()
            if PLAYWRIGHT_CHANNEL:
                launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL
        elif PLAYWRIGHT_CHANNEL:
            info("FAIL STUDIO_PLAYWRIGHT_CHANNEL requires chromium")
            return 1
        browser = browser_type.launch(**launch_kwargs)
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900},
            reduced_motion = "reduce",
        )
        install_view_transition_killer(context)
        context.add_init_script(seed_js)
        install_routes(context, state)
        page = context.new_page()
        page.set_default_timeout(60_000)
        try:
            run(page, state)
        finally:
            page.screenshot(path = str(ART / f"final-{PLAYWRIGHT_BROWSER}.png"))
            # Settle the deliberately-hung routes before tearing down: closing
            # over a parked one dumps a CancelledError traceback that reads
            # like a failure.
            for parked in state.parked:
                try:
                    parked.abort()
                except Exception:
                    pass
            state.parked.clear()
            try:
                page.goto("about:blank", wait_until = "domcontentloaded")
            except Exception:
                pass
            context.unroute_all(behavior = "ignoreErrors")
            context.close()
            browser.close()

    info(f"{checks[0] - len(failures)}/{checks[0]} checks passed")
    for failure in failures:
        info(f"  FAILED: {failure}")
    return 1 if failures else 0


def run(page, state: Runtime) -> None:
    # ── Nothing loaded ──────────────────────────────────────────────────
    state.reset()
    boot(page, state)
    check("no card when nothing is loaded", page.locator(CARD).count() == 0)

    # ── The common two-runtime host ─────────────────────────────────────
    state.chat = chat(
        active_model = "unsloth/Qwen3-4B-GGUF",
        loaded = ["unsloth/Qwen3-4B-GGUF"],
        is_gguf = True,
        gguf_variant = "Q4_K_M",
    )
    state.stt["transformers"] = {"loaded_model": "large-v3", "device": "cuda"}
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check("card lists both runtimes", len(rows(page)) == 2, str(rows(page)))
    text = card_text(page)
    check("chat row names its quant", "Q4_K_M" in text)
    check("dictation row is distinguished", "Dictation" in text)

    # The card mounts from the root, so it must survive a route change.
    for route in ("/hub", "/train", "/images"):
        page.goto(BASE + route, wait_until = "domcontentloaded")
        page.wait_for_timeout(3000)
        check(f"card survives {route}", page.locator(CARD).count() > 0)

    # ── Hardware shapes a CUDA runner never produces ────────────────────
    matrix = [
        (
            "AMD ROCm reports cuda",
            dict(
                loaded = True,
                repo_id = "black-forest-labs/FLUX.1-dev",
                family = "flux",
                device = "cuda",
                dtype = "bfloat16",
                model_kind = "pipeline",
            ),
            "flux · BF16 · cuda",
        ),
        (
            "Apple Silicon reports mps",
            dict(
                loaded = True,
                repo_id = "black-forest-labs/FLUX.1-dev",
                family = "flux",
                device = "mps",
                dtype = "bfloat16",
                model_kind = "pipeline",
            ),
            "flux · BF16 · mps",
        ),
        (
            "Intel XPU",
            dict(
                loaded = True,
                repo_id = "black-forest-labs/FLUX.1-dev",
                family = "flux",
                device = "xpu",
                dtype = "float16",
                model_kind = "pipeline",
            ),
            "flux · FP16 · xpu",
        ),
        # sd.cpp has no model_kind key at all and puts "gguf" in dtype.
        (
            "the sd.cpp engine on a CPU-only host",
            dict(
                loaded = True,
                repo_id = "unsloth/FLUX.1-dev-GGUF",
                family = "flux",
                device = "cpu",
                dtype = "gguf",
            ),
            "flux · GGUF · cpu",
        ),
    ]
    for name, payload, expected in matrix:
        state.chat = dict(NOTHING_CHAT)
        state.stt = json.loads(json.dumps(NOTHING_STT))
        state.diffusion = dict(NOTHING_DIFFUSION, **payload)
        boot(page, state)
        page.wait_for_selector(CARD, timeout = 30_000)
        check(name, expected in card_text(page), card_text(page).replace("\n", " | "))

    # An audio-input VLM listens; it must not be filed under Speech.
    state.diffusion = dict(NOTHING_DIFFUSION)
    state.chat = chat(active_model = "unsloth/gemma-3n-E4B-it", is_audio = True, audio_type = "audio_vlm")
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check(
        "an audio-input VLM is Dictation, not Speech",
        "Dictation" in card_text(page),
        card_text(page).replace("\n", " | "),
    )

    # ── A backend too old to have the video route ───────────────────────
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    page.context.route(
        "**/api/inference/video/status",
        lambda route: route.fulfill(
            status = 404, content_type = "application/json", body = json.dumps({"detail": "Not Found"})
        ),
    )
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check("a 404 video route does not blank the other rows", len(rows(page)) == 1, str(rows(page)))
    install_routes(page.context, state)  # restore the stub

    # ── A runtime that accepts the connection and never answers ─────────
    state.hang = {"video"}
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check("a hung runtime still lets the other rows render", len(rows(page)) == 1, str(rows(page)))
    state.hang = set()

    # ── The position restore: the bug this suite exists for ─────────────
    state.chat = chat(active_model = "unsloth/Qwen3-4B-GGUF", is_gguf = True, gguf_variant = "Q4_K_M")
    # As if dragged to the corner of a 2560x1440 monitor, then reopened here.
    boot(page, state, seed = {POSITION_KEY: json.dumps({"left": 2300, "top": 1300})})
    page.wait_for_selector(CARD, timeout = 30_000)
    box = page.locator(HANDLE).first.bounding_box()
    check(
        "a position saved on a bigger screen is pulled back into view",
        box is not None and 0 <= box["x"] < 1440 and 0 <= box["y"] < 900,
        f"handle={box}",
    )
    page.screenshot(path = str(ART / f"restore-{PLAYWRIGHT_BROWSER}.png"))

    # And it keeps up with a window that shrinks under it.
    page.set_viewport_size({"width": 720, "height": 560})
    page.wait_for_timeout(3000)
    box = page.locator(HANDLE).first.bounding_box()
    check(
        "a shrinking window drags the card back with it",
        box is not None and 0 <= box["x"] < 720 and 0 <= box["y"] < 560,
        f"handle={box}",
    )
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(1500)

    # ── Drag, and the pointer release the window never sees ─────────────
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    box = page.locator(HANDLE).first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] - 400, box["y"] - 300, steps = 20)
    page.mouse.up()
    page.wait_for_timeout(1500)
    stored = page.evaluate(f"localStorage.getItem({json.dumps(POSITION_KEY)})")
    check("a drag is persisted", stored is not None, str(stored))
    page.reload(wait_until = "domcontentloaded")
    page.wait_for_selector(CARD, timeout = 30_000)
    page.wait_for_timeout(2000)
    check(
        "the dragged position survives a reload",
        page.evaluate(f"localStorage.getItem({json.dumps(POSITION_KEY)})") == stored,
    )

    # A move with no button held must not keep dragging the card.
    before = page.locator(HANDLE).first.bounding_box()
    page.mouse.move(before["x"] + 200, before["y"] + 200, steps = 10)
    page.wait_for_timeout(500)
    after = page.locator(HANDLE).first.bounding_box()
    check(
        "the card does not follow a released pointer",
        abs(after["x"] - before["x"]) < 2 and abs(after["y"] - before["y"]) < 2,
        f"{before} -> {after}",
    )

    # ── Collapse ────────────────────────────────────────────────────────
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    page.locator('[aria-label="Collapse loaded models"]').first.click()
    page.wait_for_timeout(1500)
    pill = 'button[aria-label*="Show details"]'
    check("collapses to a pill", page.locator(pill).count() > 0)
    page.reload(wait_until = "domcontentloaded")
    page.wait_for_timeout(SETTLE_MS // 2)
    check("the collapsed state survives a reload", page.locator(pill).count() > 0)

    # ── Eject ───────────────────────────────────────────────────────────
    state.chat = chat(active_model = "unsloth/Qwen3-4B-GGUF", is_gguf = True, gguf_variant = "Q4_K_M")
    state.diffusion = dict(
        NOTHING_DIFFUSION,
        loaded = True,
        repo_id = "black-forest-labs/FLUX.1-dev",
        family = "flux",
        device = "cuda",
        dtype = "bfloat16",
    )
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    labels = rows(page)
    index = next((i for i, label in enumerate(labels) if "Qwen3" in label), None)
    check("the chat row is present before ejecting", index is not None, str(labels))
    if index is not None:
        page.locator(EJECT).nth(index).click()
        # Watch across more than one poll: a read already in flight when the
        # eject lands used to put the row straight back.
        reappeared = False
        gone = False
        for _ in range(60):
            present = any("Qwen3" in label for label in rows(page))
            if not present:
                gone = True
            elif gone:
                reappeared = True
            page.wait_for_timeout(200)
        check("the ejected row disappears", gone)
        check("the ejected row does not come back", not reappeared)
        check(
            "the other runtime is untouched",
            "chat" in state.unloads and "image" not in state.unloads,
            str(state.unloads),
        )

    # A row the runtime has already replaced must not unload the replacement.
    state.reset()
    state.diffusion = dict(
        NOTHING_DIFFUSION,
        loaded = True,
        repo_id = "black-forest-labs/FLUX.1-dev",
        family = "flux",
        device = "cuda",
        dtype = "bfloat16",
    )
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    # Swap the model behind the card's back, as a load from another tab would.
    state.diffusion = dict(state.diffusion, repo_id = "Qwen/Qwen-Image")
    page.locator(EJECT).first.click()
    page.wait_for_timeout(4000)
    check(
        "a replaced image row is not ejected on the replacement's behalf",
        "image" not in state.unloads,
        str(state.unloads),
    )

    # ── The preference ──────────────────────────────────────────────────
    state.reset()
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    boot(page, state, seed = {SHOW_KEY: "false"})
    check("the preference hides the card", page.locator(CARD).count() == 0)
    state.status_reads = 0
    page.wait_for_timeout(SETTLE_MS)
    check(
        "the preference stops the poll",
        state.status_reads <= 1,
        f"{state.status_reads} status reads while off",
    )


if __name__ == "__main__":
    sys.exit(main())
