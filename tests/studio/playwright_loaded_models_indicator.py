# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loaded-models card on stubbed /status; position restore, drag and polling need a real engine."""

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
    install_wall_clock_watchdog,
    report_failing_step,
    step_budget_s,
    wait_for_health,
    wait_for_settled,
    wait_until,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-loaded-models"))
ART.mkdir(parents = True, exist_ok = True)

PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# page.evaluate takes no timeout, and this suite runs last on the Windows chat lane.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))

# The card polls every 5s; two ticks plus slack.
SETTLE_MS = int(os.environ.get("STUDIO_UI_INDICATOR_SETTLE_MS", "12000"))
READS_PER_POLL = 4
STEP_BUDGET_S = step_budget_s(240)

CARD = 'text="Loaded models"'
EJECT = '[aria-label^="Eject "]'
HANDLE = '[aria-label="Drag to move"]'
# Named here so a failed presence check can say whether it found a pill instead of the card.
PILL = 'button[aria-label*="Show details"]'
POSITION_KEY = "unsloth_loaded_models_position"
COLLAPSED_KEY = "unsloth_loaded_models_collapsed"
SHOW_KEY = "unsloth_show_loaded_models_indicator"

failures: list[str] = []
checks = [0]
_watchdog = None
console_errors: list[str] = []


def info(s: str) -> None:
    print(f"[indicator] {s}", flush = True)


def step(s: str) -> None:
    """Start section `s`: it may run STEP_BUDGET_S before the run stops, naming it."""
    print(f"[indicator] STEP {s}", flush = True)
    if _watchdog is not None:
        _watchdog.begin_step(s, STEP_BUDGET_S)


def appears_within(page, selector: str, window_ms: int) -> bool:
    """Returns True as soon as selector attaches; a False after the full window means it stayed absent."""
    try:
        page.wait_for_selector(selector, state = "attached", timeout = window_ms)
    except Exception:
        return False
    return True


def watch(page, predicate, window_s: float, what: str) -> None:
    """Polls through the page so stubbed routes keep running; a timeout here is not a verdict."""
    try:
        wait_until(predicate, timeout_s = window_s, what = what, interval_s = 0.1, page = page)
    except TimeoutError:
        pass


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
        # Kept so teardown can settle them instead of cancelling them under the handler.
        self.parked: list = []


def install_routes(context, state: Runtime) -> None:
    def stub(key: str, body):
        def handler(route):
            state.status_reads += 1
            if key in state.hang:
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
    # One round trip: count() then nth(i) races the row disappearing and blocks for the full timeout.
    return page.locator(EJECT).evaluate_all(
        "els => els.map((el) => el.getAttribute('aria-label') || '')"
    )


def card_text(page) -> str:
    try:
        return page.locator(CARD).locator("xpath=ancestor::div[3]").first.inner_text(timeout = 5000)
    except Exception:
        return ""


def why_no_card(
    page,
    state: Runtime,
    waited: str = "",
    reads_before: int | None = None,
) -> str:
    """Explains a failed presence check; status reads count only from the caller's pre-navigation mark."""

    def probe(expression: str):
        try:
            return page.evaluate(expression)
        except Exception:
            return "<unreadable>"

    def nodes(selector: str):
        # Attached, not visible: an off-screen card is a position bug, not a missing card.
        count = counted(page, selector)
        return "<unreadable>" if count is None else count

    pathname = probe("location.pathname")
    mounted = probe("document.getElementById('root')?.childElementCount ?? -1")
    token = probe("Boolean(localStorage.getItem('unsloth_auth_token'))")
    shown = probe(f"localStorage.getItem({json.dumps(SHOW_KEY)})")
    reads = (
        f"{state.status_reads}"
        if reads_before is None
        else f"{state.status_reads - reads_before} (of {state.status_reads} since boot)"
    )
    return (
        f"wait={waited or 'returned'} pathname={pathname!r} card_nodes={nodes(CARD)} "
        f"collapsed_pill={nodes(PILL)} root_children={mounted} auth_token={token} "
        f"show_pref={shown!r} status_reads={reads} console={console_errors[-4:]}"
    )


def await_selector(page, selector: str, timeout: int) -> str:
    """Returns what ended a failed wait, so a timeout is told apart from a closed page or other error."""
    try:
        page.wait_for_selector(selector, timeout = timeout)
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        return f"{type(exc).__name__}: {first_line[:120]}"
    return ""


def await_selector_state(page, selector: str, state: str, timeout: int) -> str:
    """await_selector for a state other than visible (for example "detached")."""
    try:
        page.wait_for_selector(selector, state = state, timeout = timeout)
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        return f"{type(exc).__name__}: {first_line[:120]}"
    return ""


def counted(page, selector: str) -> int | None:
    """Returns None, not zero, when the page cannot answer; zero means it answered and had no match."""
    try:
        return page.locator(selector).count()
    except Exception:
        return None


def boot(
    page,
    state: Runtime,
    *,
    seed: dict | None = None,
    show: bool = True,
) -> None:
    """Reload with a known localStorage, then wait for the card to settle."""
    page.goto(BASE, wait_until = "domcontentloaded")
    # The indicator ships off, so checks that want the card switch it on.
    seeded = dict(seed or {})
    if show:
        seeded.setdefault(SHOW_KEY, "true")
    page.evaluate(
        """([seed, keys]) => {
            for (const k of keys) localStorage.removeItem(k);
            for (const [k, v] of Object.entries(seed || {}))
                localStorage.setItem(k, v);
        }""",
        [seeded, [POSITION_KEY, COLLAPSED_KEY, SHOW_KEY]],
    )
    page.reload(wait_until = "domcontentloaded")
    try:
        page.wait_for_function(
            """() => /^\\/(login|change-password)/.test(location.pathname)
                || !!document.querySelector('textarea[aria-label="Message input"]')""",
            timeout = 30_000,
        )
    except Exception:
        pass
    # The card is hidden on /login, so an auth slip would make every "no card" check pass.
    path = page.evaluate("location.pathname")
    if path.startswith(("/login", "/change-password")):
        raise AssertionError(f"not authenticated: landed on {path}")


def main() -> int:
    wait_for_health(BASE, timeout = 60.0, info = info)
    token = api("/api/auth/login", {"username": "unsloth", "password": OLD})["access_token"]
    try:
        api("/api/auth/change-password", {"current_password": OLD, "new_password": NEW}, token)
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 401, 403):
            raise
    session = api("/api/auth/login", {"username": "unsloth", "password": NEW})
    if session.get("must_change_password"):
        info("FAIL bootstrap left must_change_password set")
        return 1

    # add_init_script takes raw source: an arrow expression would never be invoked.
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
        global _watchdog
        # total_deadline_s keeps 720s an absolute wall; begin_step() would otherwise restart it.
        _watchdog = install_wall_clock_watchdog(
            WALL_TIMEOUT_S,
            label = "ui-indicator",
            info = info,
            total_deadline_s = WALL_TIMEOUT_S,
        )
        report_failing_step(_watchdog, label = "ui-indicator")
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
        page.on(
            "console",
            lambda message: console_errors.append(f"{message.type}: {message.text}"[:200])
            if message.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda error: console_errors.append(f"pageerror: {error}"[:200]))
        try:
            run(page, state)
        finally:
            page.screenshot(path = str(ART / f"final-{PLAYWRIGHT_BROWSER}.png"))
            # Closing over a parked route dumps a CancelledError traceback that reads like a failure.
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
    step("no card when nothing is loaded")
    state.reset()
    boot(page, state)
    check("no card when nothing is loaded", not appears_within(page, CARD, SETTLE_MS // 2))

    step("two runtimes, and the card across routes")
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

    for route in ("/hub", "/train", "/images"):
        console_errors.clear()
        reads_before = state.status_reads
        page.goto(BASE + route, wait_until = "domcontentloaded")
        # Hard navigation: wait for the card rather than a fixed budget.
        waited = await_selector(page, CARD, SETTLE_MS)
        nodes = counted(page, CARD)
        present = bool(nodes)
        # `count` is attached nodes and the wait is visible ones; present-but-invisible is a stacking bug.
        if waited and present:
            info(
                f"NOTE card survives {route}: attached but not visible in {SETTLE_MS}ms ({waited})"
            )
        check(
            f"card survives {route}",
            present,
            "" if present else why_no_card(page, state, waited, reads_before),
        )

    step("hardware shapes, audio VLM, 404 and hung runtimes")
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
        # sd.cpp has no model_kind key and puts "gguf" in dtype.
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

    # An audio VLM is a chat model, neither Speech nor Dictation.
    state.diffusion = dict(NOTHING_DIFFUSION)
    state.chat = chat(active_model = "unsloth/gemma-3n-E4B-it", is_audio = True, audio_type = "audio_vlm")
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    text = card_text(page)
    check(
        "an audio VLM stays a Chat row",
        "Chat" in text and "Speech" not in text and "Dictation" not in text,
        text.replace("\n", " | "),
    )

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
    install_routes(page.context, state)

    state.hang = {"video"}
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check("a hung runtime still lets the other rows render", len(rows(page)) == 1, str(rows(page)))
    state.hang = set()

    # A failed read is not evidence the runtime is empty; the row must survive it.
    step("a failed status read keeps the row, a readable empty one retires it")
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    check("the chat row is up before the blip", len(rows(page)) == 1, str(rows(page)))
    failing = {"count": 0}

    def fail_chat_status(route):
        failing["count"] += 1
        route.fulfill(
            status = 503, content_type = "application/json", body = json.dumps({"detail": "upstream"})
        )

    page.context.route("**/api/inference/status", fail_chat_status)
    watch(
        page,
        lambda: failing["count"] >= 2 or len(rows(page)) != 1,
        30.0,
        "two failed chat status reads",
    )
    check(
        "a failing status read keeps the row it cannot confirm",
        failing["count"] > 0 and len(rows(page)) == 1,
        f"{failing['count']} failed reads, rows={rows(page)}",
    )
    install_routes(page.context, state)

    state.chat = chat()
    watch(page, lambda: len(rows(page)) == 0, SETTLE_MS / 1000, "the chat row to retire")
    check(
        "a readable empty status still retires the row",
        len(rows(page)) == 0,
        str(rows(page)),
    )
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    state.hang = set()

    # The position restore: the bug this suite exists for.
    step("position restore, drag, released pointer")
    state.chat = chat(active_model = "unsloth/Qwen3-4B-GGUF", is_gguf = True, gguf_variant = "Q4_K_M")
    boot(page, state, seed = {POSITION_KEY: json.dumps({"left": 2300, "top": 1300})})
    page.wait_for_selector(CARD, timeout = 30_000)
    box = page.locator(HANDLE).first.bounding_box()
    check(
        "a position saved on a bigger screen is pulled back into view",
        box is not None and 0 <= box["x"] < 1440 and 0 <= box["y"] < 900,
        f"handle={box}",
    )
    page.screenshot(path = str(ART / f"restore-{PLAYWRIGHT_BROWSER}.png"))

    page.set_viewport_size({"width": 720, "height": 560})

    def handle_inside(width, height):
        b = page.locator(HANDLE).first.bounding_box()
        return b is not None and 0 <= b["x"] < width and 0 <= b["y"] < height

    watch(page, lambda: handle_inside(720, 560), SETTLE_MS / 1000, "the card inside 720x560")
    box = page.locator(HANDLE).first.bounding_box()
    check(
        "a shrinking window drags the card back with it",
        box is not None and 0 <= box["x"] < 720 and 0 <= box["y"] < 560,
        f"handle={box}",
    )
    page.set_viewport_size({"width": 1440, "height": 900})

    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    box = page.locator(HANDLE).first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] - 400, box["y"] - 300, steps = 20)
    page.mouse.up()

    def stored_position():
        return page.evaluate(f"localStorage.getItem({json.dumps(POSITION_KEY)})")

    watch(page, lambda: stored_position() is not None, SETTLE_MS / 1000, "the drag to be stored")
    stored = stored_position()
    check("a drag is persisted", stored is not None, str(stored))
    page.reload(wait_until = "domcontentloaded")
    page.wait_for_selector(CARD, timeout = 30_000)
    watch(page, lambda: stored_position() != stored, 2.0, "a rewrite of the stored position")
    check(
        "the dragged position survives a reload",
        stored_position() == stored,
    )

    before = page.locator(HANDLE).first.bounding_box()
    page.mouse.move(before["x"] + 200, before["y"] + 200, steps = 10)
    page.wait_for_timeout(500)
    after = page.locator(HANDLE).first.bounding_box()
    check(
        "the card does not follow a released pointer",
        abs(after["x"] - before["x"]) < 2 and abs(after["y"] - before["y"]) < 2,
        f"{before} -> {after}",
    )

    step("collapse, close, and a load nobody announced")
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    page.locator('[aria-label="Collapse loaded models"]').first.click()
    await_selector(page, PILL, SETTLE_MS)
    check("collapses to a pill", page.locator(PILL).count() > 0)
    console_errors.clear()
    reads_before = state.status_reads
    page.reload(wait_until = "domcontentloaded")
    waited = await_selector(page, PILL, SETTLE_MS)
    restored = bool(counted(page, PILL))
    check(
        "the collapsed state survives a reload",
        restored,
        "" if restored else why_no_card(page, state, waited, reads_before),
    )

    # A load via the OpenAI API or auto-switch raises no lifecycle event; the poll is the only witness.
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    page.locator('[aria-label="Close loaded models"]').first.click()
    await_selector_state(page, CARD, "detached", SETTLE_MS)
    check(
        "closing hides the card while a model is still resident",
        page.locator(CARD).count() == 0,
    )
    reads_before = state.status_reads
    watch(
        page,
        lambda: counted(page, CARD) or state.status_reads >= reads_before + 2 * READS_PER_POLL,
        30.0,
        "two polls with the card closed",
    )
    check(
        "a closed card stays closed over what was already loaded",
        page.locator(CARD).count() == 0,
    )
    state.diffusion = dict(
        NOTHING_DIFFUSION,
        loaded = True,
        repo_id = "black-forest-labs/FLUX.1-dev",
        family = "flux",
        device = "cuda",
        dtype = "bfloat16",
    )
    await_selector(page, CARD, SETTLE_MS)
    check(
        "a load nobody announced reopens the closed card",
        page.locator(CARD).count() > 0,
        "the poll is the only witness for a load started outside the frontend",
    )
    state.diffusion = NOTHING_DIFFUSION

    # Grip and pill share one drag sentinel; without dropping it after a click-less drag, the first pill
    # click is swallowed. No reload, which would clear the flag.
    step("grip drag, collapse, one click reopens")
    boot(page, state)
    page.wait_for_selector(CARD, timeout = 30_000)
    grip = page.locator(HANDLE).first.bounding_box()
    page.mouse.move(grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2)
    page.mouse.down()
    page.mouse.move(grip["x"] - 120, grip["y"] - 80, steps = 12)
    page.mouse.up()
    try:
        wait_for_settled(page.locator(HANDLE), timeout_ms = SETTLE_MS)
    except Exception:
        pass
    page.locator('[aria-label="Collapse loaded models"]').first.click()
    await_selector(page, PILL, SETTLE_MS)
    collapsed_ok = page.locator(CARD).count() == 0 and page.locator(PILL).count() > 0
    check("the grip drag still collapses to a pill", collapsed_ok)
    page.locator(PILL).first.click()
    await_selector(page, CARD, SETTLE_MS)
    check(
        "one click reopens the pill after dragging by the grip",
        collapsed_ok and page.locator(CARD).count() > 0,
        "the grip's drag was still held against the pill's first click",
    )

    step("eject, replaced and stale rows")
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
        # A read in flight when the eject lands used to put the row back.
        reappeared = False
        gone = False
        for _ in range(60):
            present = any("Qwen3" in label for label in rows(page))
            if not present:
                gone = True
            elif gone:
                reappeared = True
                break
            page.wait_for_timeout(200)
        check("the ejected row disappears", gone)
        check("the ejected row does not come back", not reappeared)
        check(
            "the other runtime is untouched",
            "chat" in state.unloads and "image" not in state.unloads,
            str(state.unloads),
        )

    # A row the runtime already replaced must not unload the replacement.
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
    state.diffusion = dict(state.diffusion, repo_id = "Qwen/Qwen-Image")
    page.locator(EJECT).first.click()
    watch(page, lambda: "image" in state.unloads, 4.0, "an image unload")
    check(
        "a replaced image row is not ejected on the replacement's behalf",
        "image" not in state.unloads,
        str(state.unloads),
    )

    # An already-idle runtime: nothing is unloaded, so the toast must not report an eject.
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
    state.diffusion = dict(NOTHING_DIFFUSION)
    page.locator(EJECT).first.click()
    watch(
        page,
        lambda: "image" in state.unloads or counted(page, "[data-sonner-toast]"),
        4.0,
        "the eject to answer",
    )
    said = page.locator("[data-sonner-toast]").evaluate_all(
        "els => els.map((el) => el.innerText || '').join(' | ')"
    )
    check(
        "a stale row does not claim an eject it never performed",
        "image" not in state.unloads and "Ejected" not in said,
        f"unloads={state.unloads} toasts={said!r}",
    )

    step("the preference: off by default, and off stops the poll")
    state.reset()
    state.chat = chat(active_model = "unsloth/Qwen3-4B", loaded = ["unsloth/Qwen3-4B"])
    # A fresh install shows no card even with a model resident.
    boot(page, state, show = False)
    check("the card is off by default", not appears_within(page, CARD, SETTLE_MS // 2))
    state.status_reads = 0
    watch(page, lambda: state.status_reads > 1, SETTLE_MS / 1000, "status reads while off")
    check(
        "the default stops the poll",
        state.status_reads <= 1,
        f"{state.status_reads} status reads while off",
    )
    # What the old default wrote when turned down; still off.
    boot(page, state, seed = {SHOW_KEY: "false"}, show = False)
    check(
        "an older explicit false still hides the card",
        not appears_within(page, CARD, SETTLE_MS // 2),
    )


if __name__ == "__main__":
    sys.exit(main())
