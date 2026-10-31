# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Estimated Memory Usage row against a stubbed estimate API; the row must hide when unavailable."""

import json
import re
import sys
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    dump_diagnostics,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    recover_or_replace_page,
    report_failing_step,
    step_budget_s,
    wait_for_first,
    wait_for_health,
    wait_until,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ.get("STUDIO_NEW_PW", "MemEst-NEW-2026!")
# Attach mode: the model-config scene already rotated the bootstrap password on this server.
LOGIN_PW = os.environ.get("STUDIO_LOGIN_PW")
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
# From GGUF_REPO: a family-name hint can match a non-GGUF sibling first, which has no memory row.
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT") or GGUF_REPO.rsplit("/", 1)[-1]
# Typing the value already displayed commits no onChange and no re-price, so choose a value that
# differs from the box at typing time.
CTX_CANDIDATES = [
    int(part)
    for part in os.environ.get("STUDIO_CTX_CANDIDATES", "6144,5120,4096,3072,2048,1536").split(",")
    if part.strip()
]
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_memory_estimate")
# An edit in the panel's first moments is discarded when it re-derives its baseline.
CONFIG_SETTLE_MS = int(os.environ.get("STUDIO_CONFIG_SETTLE_MS", "1000"))
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "600"))
# The hook debounces at 250ms; this is the "never coming" bound.
ESTIMATE_WAIT_MS = int(os.environ.get("STUDIO_UI_ESTIMATE_WAIT_MS", "20000"))
STEP_BUDGET_S = step_budget_s(max(180.0, 4 * ESTIMATE_WAIT_MS / 1000 + 60))
NO_STEP_CEILING = 0

TRANSCRIPT_NAME = "memory-estimate-exchanges.json"

GIB = 1024**3
# Quarter-GiB values are exact, so formatBytesGiB's toFixed(2) output is predictable.
STUB_WEIGHTS_BYTES = int(3.25 * GIB)
STUB_KV_BYTES = int(1.50 * GIB)
STUB_COMPUTE_BYTES = int(0.75 * GIB)
STUB_TOTAL_BYTES = STUB_WEIGHTS_BYTES + STUB_KV_BYTES + STUB_COMPUTE_BYTES
STUB_GPU_BYTES = int(4.75 * GIB)
STUB_LAYER_COUNT = 27
STUB_GPU_LAYERS = 12


def _gib(num_bytes: int) -> str:
    """Mirrors formatBytesGiB in lib/memory/format.ts; its label is GiB, not GB, and is asserted."""
    return f"{num_bytes / GIB:.2f} GiB"


_n = [0]
_failed: list[str] = []
_watchdog = None


def step(s: str, budget_s: float | None = None) -> None:
    """Start step `s`; it may run `budget_s` (default STEP_BUDGET_S) before the run stops."""
    print(f"[ui-memest] STEP {s}", flush = True)
    if _watchdog is not None:
        _watchdog.begin_step(s, STEP_BUDGET_S if budget_s is None else budget_s)


def info(s: str) -> None:
    print(f"[ui-memest] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[ui-memest] FAIL: {m}", flush = True)
    _failed.append(m)


def soft_fail(m: str) -> None:
    if STRICT:
        fail(m)
    else:
        info(f"WARN (strict-off): {m}")


def runtime_warn(m: str) -> None:
    """Warn about a genuinely-optional check that STRICT does not gate."""
    info(f"WARN (runtime): {m}")


def _count(loc) -> int:
    """Returns 0 on no match; a raise is reported apart, since a closed page is not a missing selector."""
    try:
        return loc.count()
    except Exception as exc:
        info(f"WARN: locator raised (not a missing element): {type(exc).__name__}: {exc}")
        return 0


def _login_token_via_api(base: str, user: str, pw: str) -> str:
    """POST /api/auth/login -> access_token (attach-mode helper, stdlib only)."""
    import urllib.request

    req = urllib.request.Request(
        f"{base}/api/auth/login",
        data = json.dumps({"username": user, "password": pw}).encode(),
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )
    with urllib.request.urlopen(req, timeout = 15) as r:
        return json.loads(r.read().decode())["access_token"]


# The stubbed endpoint: `_mode` decides what the NEXT call answers; every call is recorded.
_mode = ["available"]
exchanges: list[dict] = []

UNAVAILABLE_BODY = {
    "available": False,
    "reason": "not_downloaded",
    "weights_bytes": 0,
    "kv_bytes": 0,
    "compute_bytes": 0,
    "drafter_runtime_bytes": 0,
    "drafter_runtime_gpu_bytes": 0,
    "projector_runtime_bytes": 0,
    "drafter_kv_unsized": False,
    "total_bytes": 0,
    "gpu_bytes": 0,
    "kv_estimable": False,
    "kv_on_gpu": True,
    "n_ctx": 0,
    "cache_type_kv": None,
    "n_parallel": 1,
    "layer_count": None,
    "gpu_layers": None,
    "moe_offload_unmodelled": False,
}


def _available_body(request_payload: dict) -> dict:
    """Echoes n_ctx, cache dtype and slot count from the request, so the KV note proves a round trip."""
    raw_ctx = request_payload.get("n_ctx")
    n_ctx = int(raw_ctx) if isinstance(raw_ctx, (int, float)) and raw_ctx else 0
    raw_parallel = request_payload.get("n_parallel")
    n_parallel = int(raw_parallel) if isinstance(raw_parallel, (int, float)) and raw_parallel else 1
    return {
        "available": True,
        "reason": None,
        "weights_bytes": STUB_WEIGHTS_BYTES,
        "kv_bytes": STUB_KV_BYTES,
        "compute_bytes": STUB_COMPUTE_BYTES,
        "drafter_runtime_bytes": 0,
        "drafter_runtime_gpu_bytes": 0,
        "projector_runtime_bytes": 0,
        "drafter_kv_unsized": False,
        "total_bytes": STUB_TOTAL_BYTES,
        "gpu_bytes": STUB_GPU_BYTES,
        "kv_estimable": True,
        "kv_on_gpu": True,
        "n_ctx": n_ctx,
        "cache_type_kv": request_payload.get("cache_type_kv") or "f16",
        "n_parallel": n_parallel,
        "layer_count": STUB_LAYER_COUNT,
        "gpu_layers": STUB_GPU_LAYERS,
        "moe_offload_unmodelled": False,
    }


def _handle_estimate(route) -> None:
    request = route.request
    raw = ""
    try:
        raw = request.post_data or ""
    except Exception:
        raw = ""
    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    mode = _mode[0]
    record: dict = {
        "ts": time.time(),
        "mode": mode,
        "method": request.method,
        "url": request.url,
        "request": payload,
        "raw_request": raw[:2000],
    }
    exchanges.append(record)
    if mode == "http404":
        body = {"detail": "Not Found"}
        record["response_status"] = 404
        record["response"] = body
        route.fulfill(
            status = 404,
            content_type = "application/json",
            body = json.dumps(body),
        )
        return
    body = dict(UNAVAILABLE_BODY) if mode == "unavailable" else _available_body(payload)
    record["response_status"] = 200
    record["response"] = body
    route.fulfill(status = 200, content_type = "application/json", body = json.dumps(body))


def write_transcript() -> None:
    """The recorded API exchange, in the artifact directory the sibling scenes use."""
    try:
        (ART / TRANSCRIPT_NAME).write_text(
            json.dumps(
                {
                    "base_url": BASE,
                    "browser": PLAYWRIGHT_BROWSER,
                    "model": {"repo": GGUF_REPO, "variant": GGUF_VARIANT},
                    "stub": {
                        "weights_bytes": STUB_WEIGHTS_BYTES,
                        "kv_bytes": STUB_KV_BYTES,
                        "compute_bytes": STUB_COMPUTE_BYTES,
                        "total_bytes": STUB_TOTAL_BYTES,
                        "gpu_bytes": STUB_GPU_BYTES,
                    },
                    "exchanges": exchanges,
                },
                indent = 2,
                default = str,
            ),
            encoding = "utf-8",
        )
        info(f"wrote {len(exchanges)} estimate exchange(s) to {ART / TRANSCRIPT_NAME}")
    except Exception as exc:
        info(f"WARN: could not write the exchange transcript: {exc}")


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S, label = "ui-memest", info = info, total_deadline_s = WALL_TIMEOUT_S
    )
    report_failing_step(_watchdog, label = "ui-memest")
    # A shell health wait can pass before the auth DB migrates.
    wait_for_health(BASE, timeout = 30.0, info = info)
    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        fail(f"unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
        sys.exit(1)
    browser_type = getattr(p, PLAYWRIGHT_BROWSER)
    launch_kwargs = {"headless": True}
    if PLAYWRIGHT_BROWSER == "chromium":
        launch_kwargs["args"] = chromium_launch_args()
        if PLAYWRIGHT_CHANNEL:
            launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL
    elif PLAYWRIGHT_CHANNEL:
        fail("STUDIO_PLAYWRIGHT_CHANNEL requires chromium")
        sys.exit(1)
    browser = browser_type.launch(**launch_kwargs)
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
        # Pin the locale: toLocaleString group separators vary by runner.
        locale = "en-US",
    )
    install_view_transition_killer(ctx)
    # On the context, not the page, so a replaced page keeps the interception.
    ctx.route("**/api/inference/estimate-memory*", _handle_estimate)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    page_errors: list[str] = []

    def _on_pageerror(e):
        msg = str(e)
        if is_benign_page_error(msg):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    def shoot(name: str) -> None:
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as exc:
            info(f"WARN: screenshot {name} failed: {exc}")

    def diagnose(name: str, missed: str) -> None:
        rows = []
        try:
            opts = page.locator("[data-model-picker-option]")
            rows = [
                (opts.nth(i).inner_text() or "").strip()[:60] for i in range(min(opts.count(), 12))
            ]
        except Exception:
            pass
        dump_diagnostics(
            page,
            ART,
            name,
            info = info,
            extra = {
                "missed_selector": missed,
                "option_rows": rows,
                "estimate_exchanges": exchanges[-4:],
                "mode": _mode[0],
            },
        )

    if LOGIN_PW:
        step("setup: API login + token seed (attach to running Unsloth)", NO_STEP_CEILING)
        _tok = _login_token_via_api(BASE, LOGIN_USER, LOGIN_PW)
        ctx.add_init_script(
            f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(_tok)});}}catch(e){{}}"
        )
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)
    else:
        step("setup: change-password", NO_STEP_CEILING)
        form_err: Exception | None = None
        for _attempt in range(3):
            try:
                page.goto(f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000)
                try:
                    page.wait_for_load_state("networkidle", timeout = 30_000)
                except Exception:
                    pass
                pw_field = page.locator("#new-password")
                pw_field.wait_for(state = "visible", timeout = 60_000)
                pw_field.fill(NEW, timeout = 60_000)
                page.fill("#confirm-password", NEW, timeout = 60_000)
                status, _ = click_and_wait_for_response(
                    page,
                    url_substr = "/api/auth/change-password",
                    method = "POST",
                    do_click = lambda: page.locator('button[type="submit"]').click(),
                    timeout_ms = 30_000,
                    info = lambda m: print(f"[ui-memest]   {m}", flush = True),
                )
                if status is not None and status >= 400:
                    raise AssertionError(
                        f"change-password POST returned {status}; page_errors={page_errors[:1]!r}"
                    )
                form_err = None
                break
            except Exception as e:
                form_err = e
                try:
                    cur_url = page.url
                except Exception:
                    cur_url = "<page closed>"
                print(
                    f"[ui-memest]   change-password attempt {_attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}",
                    flush = True,
                )
                if _attempt < 2:
                    page = recover_or_replace_page(
                        page,
                        ctx,
                        default_timeout_ms = 60_000,
                        info = lambda m: print(f"[ui-memest]   recovery: {m}", flush = True),
                    )
                    page.on("pageerror", _on_pageerror)
        if form_err is not None:
            raise form_err

    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
    composer = page.locator('textarea[aria-label="Message input"]')
    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            composer.wait_for(state = "visible", timeout = 60_000)
            last_err = None
            break
        except Exception as e:
            last_err = e
            shoot(f"00-composer-wait-attempt-{_attempt + 1}-fail")
            if _attempt == 0:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    goto_url = BASE,
                    settle_networkidle = True,
                    info = lambda m: print(f"[ui-memest]   recovery: {m}", flush = True),
                )
                page.on("pageerror", _on_pageerror)
                composer = page.locator('textarea[aria-label="Message input"]')
    if last_err is not None:
        raise last_err
    shoot("01-chat-loaded")

    POPOVER = '[data-tour="chat-model-selector-popover"]'
    TRIGGER = '[data-tour="chat-model-selector"]'
    SOLE_QUANT_SETTLE_MS = 30_000
    QUANT_GEAR_MS = 2_000

    def open_picker():
        popover = page.locator(POPOVER).first
        if _count(popover) == 0 or not popover.is_visible():
            page.locator(TRIGGER).first.click()
            popover = page.locator(POPOVER).first
        popover.wait_for(state = "visible", timeout = 30_000)
        return popover

    def close_picker():
        try:
            page.keyboard.press("Escape")
            page.locator(POPOVER).first.wait_for(state = "hidden", timeout = 10_000)
        except Exception:
            pass

    def reveal_on_device_row(popover, hint):
        """Bring the row into view without clicking it: a single-quant row loads its
        quant on click and closes the picker, taking the gear with it."""
        od = page.get_by_role("tab", name = "On Device").first
        if _count(od):
            od.click()
        try:
            popover.locator("[data-model-picker-option]").first.wait_for(
                state = "attached", timeout = 20_000
            )
        except Exception:
            pass
        row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(hint)
                wait_for_first(
                    popover.locator("[data-model-picker-option]", has_text = hint),
                    timeout_ms = 10_000,
                )
                row = popover.locator("[data-model-picker-option]", has_text = hint).first
        return row if _count(row) else None

    def select_on_device_row(popover, hint):
        row = reveal_on_device_row(popover, hint)
        if row is None:
            return None
        row.click()
        try:
            wait_until(
                lambda: not popover.is_visible()
                or _count(popover.locator('button[aria-label^="Inference settings for" i]')) > 0,
                timeout_s = 10,
                what = "the row click to close the picker or show its gears",
                interval_s = 0.1,
                page = page,
            )
        except TimeoutError as exc:
            info(f"WARN {exc}")
        return row

    def row_gear(
        popover,
        hint,
        quant = None,
        timeout_ms = SOLE_QUANT_SETTLE_MS,
    ):
        # The gear is a sibling of the row. Anchor the quant at the end, or F16 matches BF16.
        pattern = f"^Inference settings for .*{re.escape(hint)}"
        if quant:
            pattern += f".* {re.escape(quant)}$"
        gear = popover.get_by_role("button", name = re.compile(pattern, re.IGNORECASE)).first
        try:
            gear.wait_for(state = "visible", timeout = timeout_ms)
        except Exception:
            return None
        return gear

    def config_is_open(popover):
        """Back is unique to the config page and always rendered inside the picker."""
        return _count(popover.get_by_role("button", name = "Back to model list")) > 0

    def open_config(popover, hint):
        if reveal_on_device_row(popover, hint) is None:
            diagnose("no-picker-row", f"[data-model-picker-option] has_text={hint!r}")
            return None
        # Quant first: with "Expand quantizations" on, a repo-only lookup picks an arbitrary gear.
        gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS)
        if gear is None:
            gear = row_gear(popover, hint)
        if gear is None:
            # Clicking a collapsed sole-quant row loads it and closes the picker, so reopen and retry.
            if select_on_device_row(popover, hint) is None:
                diagnose("no-row-gear", f"Inference settings for ...{hint}")
                return None
            if not popover.is_visible():
                popover = open_picker()
                if reveal_on_device_row(popover, hint) is None:
                    diagnose("no-row-gear-after-reopen", f"[data-model-picker-option] {hint!r}")
                    return None
            gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS) or (
                row_gear(popover, hint)
            )
        if gear is None:
            diagnose("no-row-gear", f"Inference settings for ...{hint}")
            return None
        gear.click()
        if (
            wait_for_first(
                popover.get_by_role("button", name = "Back to model list"), timeout_ms = 10_000
            )
            is not None
        ):
            # The panel exposes no readiness signal to poll.
            page.wait_for_timeout(CONFIG_SETTLE_MS)
            return popover
        diagnose("config-not-open", 'button[name="Back to model list"]')
        return None

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    # Row helpers anchor on the toggle button: its parent is the header, aria-controls names the breakdown.
    ESTIMATE_LABEL = re.compile(r"Estimated Memory Usage", re.I)
    # The picker keeps a hidden copy of the panel mounted, so take the first match actually on screen.
    _match_note: list[str] = []

    def _first_visible(loc, label: str):
        """The first on-screen match of `loc`, or None. Never raises."""
        try:
            total = loc.count()
        except Exception:
            return None
        for i in range(total):
            candidate = loc.nth(i)
            try:
                if candidate.is_visible():
                    if label not in _match_note:
                        _match_note.append(label)
                        info(f"row located by {label} ({total} match(es) in the document)")
                    return candidate
            except Exception:
                continue
        return None

    def estimate_button():
        """Finds the toggle by role first, which proves a screen reader can reach it; markup is a
        fallback."""
        found = _first_visible(page.get_by_role("button", name = ESTIMATE_LABEL), "role=button")
        if found is not None:
            return found
        return _first_visible(page.locator("button").filter(has_text = ESTIMATE_LABEL), "button+text")

    def estimate_visible() -> bool:
        return estimate_button() is not None

    # After one miss, later waits answer at once instead of spending the same 20 s again.
    _row_never_rendered = [False]

    def wait_for_row(present: bool, timeout_ms: int = ESTIMATE_WAIT_MS) -> bool:
        if present and _row_never_rendered[0] and not estimate_visible():
            info("not waiting for the row again: it never rendered in the first step")
            return False
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            if estimate_visible() == present:
                return True
            page.wait_for_timeout(200)
        return estimate_visible() == present

    def _readable(raw: str | None) -> str:
        """Maps U+00A0 to plain spaces so each caption matches how it looks, not how it is encoded."""
        return (raw or "").replace(" ", " ").strip()

    def header_text() -> str:
        button = estimate_button()
        if button is None:
            return ""
        try:
            return _readable(button.locator("xpath=..").inner_text())
        except Exception:
            return ""

    def breakdown_text() -> str:
        button = estimate_button()
        if button is None:
            return ""
        try:
            content_id = button.get_attribute("aria-controls")
        except Exception:
            content_id = None
        if not content_id:
            return ""
        # Attribute selector: React useId ids contain ':', invalid in a CSS id selector.
        panel = page.locator(f'[id="{content_id}"]').first
        if _count(panel) == 0:
            return ""
        try:
            return _readable(panel.inner_text())
        except Exception:
            return ""

    def wait_for_estimate_post(
        predicate,
        *,
        since: int,
        timeout_ms: int = ESTIMATE_WAIT_MS,
    ):
        """Reads the recorded transcript rather than expect_request, since the gates check what was
        asked."""
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            for record in exchanges[since:]:
                if predicate(record):
                    return record
            page.wait_for_timeout(200)
        for record in exchanges[since:]:
            if predicate(record):
                return record
        return None

    _used_contexts: set[int] = set()

    def reprice(popover, label: str):
        """Picks a Context Length different from the shown one; the panel does not re-price the same
        value."""
        box = context_input(popover)
        if box is None:
            fail(f"{label}: the Context Length control is not in the run-settings panel")
            return None, None
        attempted: list[int] = []
        for _try in range(3):
            try:
                box.click()
                try:
                    wait_until(
                        lambda: re.fullmatch(r"[\d,\s]+", box.input_value() or "") is not None,
                        timeout_s = 5,
                        what = "Context Length to show a number once focused",
                        interval_s = 0.05,
                        page = page,
                    )
                except TimeoutError as exc:
                    info(f"WARN {label}: {exc}")
                shown = box.input_value()
            except Exception as exc:
                fail(f"{label}: could not focus the Context Length control: {exc}")
                return None, None
            try:
                shown_int = int(str(shown).replace(",", "").strip())
            except Exception:
                shown_int = None
            value = next(
                (
                    candidate
                    for candidate in CTX_CANDIDATES
                    if candidate != shown_int
                    and candidate not in _used_contexts
                    and candidate not in attempted
                ),
                None,
            )
            if value is None:
                fail(
                    f"{label}: ran out of Context Length values to type; the control shows "
                    f"{shown!r} and {sorted(_used_contexts)} are spent"
                )
                return None, None
            attempted.append(value)
            before = len(exchanges)
            try:
                box.fill(str(value))
                # A value left focused mid-edit can stay a draft on slower engines.
                box.press("Tab")
            except Exception as exc:
                fail(f"{label}: could not type {value} into the Context Length control: {exc}")
                return None, None
            record = wait_for_estimate_post(
                lambda rec: rec["request"].get("n_ctx") == value, since = before
            )
            if record is not None:
                _used_contexts.add(value)
                info(f"{label}: Context Length {shown!r} -> {value}, priced")
                return value, record
            info(
                f"{label}: typing {value} over {shown!r} produced no estimate request; "
                f"retrying with another value"
            )
        fail(
            f"{label}: the panel never re-priced after the Context Length was changed "
            f"(tried {attempted}); every n_ctx asked for so far="
            f"{[r['request'].get('n_ctx') for r in exchanges]!r}"
        )
        return None, None

    # 1. Open run-settings for the GGUF and prove the row is there (HARD).
    step("open run-settings for the GGUF target")
    popover = open_picker()
    shoot("02-picker-open")
    if open_config(popover, MODEL_HINT) is None:
        fail(f"could not open run-settings for a model matching {MODEL_HINT!r}")
        write_transcript()
        shoot("03-config-failed")
        browser.close()
        print(f"[ui-memest] RESULT: FAIL ({len(_failed)} issue(s))", flush = True)
        for m in _failed:
            print(f"[ui-memest]   - {m}", flush = True)
        sys.exit(1)
    shoot("03-config-open")

    if not wait_for_row(True):
        _row_never_rendered[0] = True
        fail(
            "the Estimated Memory Usage row never appeared for a GGUF target whose "
            f"estimate was stubbed available (exchanges={len(exchanges)}, "
            f"last={exchanges[-1:]!r})"
        )
        diagnose("row-missing", "button[name=/Estimated Memory Usage/]")
    else:
        info("OK row: Estimated Memory Usage rendered for the GGUF target")
        if "role=button" not in _match_note:
            # Found only by the structural locator: not reachable by role, hidden or misnamed; cause unproven.
            runtime_warn(
                "the row was found only by markup, not by role=button: a screen "
                "reader would not announce this toggle"
            )

    if not exchanges:
        fail(
            "the panel never called POST /api/inference/estimate-memory, so nothing on "
            "screen can be attributed to the estimate at all"
        )

    # 2. The request carries the load settings and a changed control re-prices (HARD).
    step("the Context Length on screen reaches the estimate request")
    priced_ctx, priced = reprice(popover, "context reaches the request")
    if priced is not None:
        info(f"OK request: estimate re-priced at n_ctx={priced_ctx}")
        body = priced["request"]
        required = ("model_path", "n_ctx", "cache_type_kv", "n_parallel", "gpu_memory_mode")
        missing = [key for key in required if key not in body]
        if missing:
            fail(f"the estimate request no longer carries {missing}; body keys={sorted(body)}")
        else:
            info(f"OK request: carries {list(required)}")
        model_path = str(body.get("model_path") or "")
        if MODEL_HINT.lower() not in model_path.lower():
            fail(
                f"the estimate priced {model_path!r}, which is not the model whose run "
                f"settings are open ({MODEL_HINT!r})"
            )
        else:
            info(f"OK request: priced model_path={model_path!r}")
        variant = body.get("gguf_variant")
        if (
            variant is not None
            and GGUF_VARIANT
            and str(variant).strip().lower() != (GGUF_VARIANT.strip().lower())
        ):
            runtime_warn(
                f"the estimate priced quant {variant!r}, not {GGUF_VARIANT!r}; the picker row "
                f"may have collapsed onto a different variant"
            )

    # 3. The stubbed numbers are on screen (HARD).
    step("the row displays the numbers the endpoint returned")
    if not wait_for_row(True):
        fail("the row is gone after the re-price, so its figures cannot be read")
    else:
        head = header_text()
        info(f"row header text: {head!r}")
        # CSS `uppercase` renders "BETA" in inner_text, so match case-insensitively.
        if not re.search(r"\bbeta\b", head, re.I):
            soft_fail(f"the row lost its Beta pill (header text={head!r})")
        total_gib = _gib(STUB_TOTAL_BYTES)
        gpu_gib = _gib(STUB_GPU_BYTES)
        # The GPU figure only exists when GPU and host memory are separate pools.
        if total_gib not in head:
            fail(
                f"the row does not show the returned total {total_gib!r} "
                f"(total_bytes={STUB_TOTAL_BYTES}); header text={head!r}"
            )
        else:
            info(f"OK figures: total {total_gib} is on screen")
        if re.search(r"\bGPU\b", head):
            if gpu_gib not in head:
                fail(
                    f"the row shows a GPU figure but not the returned gpu_bytes {gpu_gib!r}; "
                    f"header text={head!r}"
                )
            else:
                info(f"OK figures: GPU {gpu_gib} is on screen")
        else:
            info("single-pool layout: no separate GPU figure to check")
        shoot("04-row-collapsed")

        try:
            expander = estimate_button()
            if expander is None:
                raise RuntimeError("the row is no longer on screen")
            expander.click()
            try:
                wait_until(breakdown_text, timeout_s = 5, what = "the breakdown panel", page = page)
            except TimeoutError as exc:
                info(f"WARN {exc}")
        except Exception as exc:
            fail(f"could not expand the Estimated Memory Usage row: {exc}")
        detail = breakdown_text()
        info(f"row breakdown text: {detail!r}")
        if not detail:
            fail(
                "expanding the row produced no breakdown panel (the button's aria-controls "
                "names nothing on the page)"
            )
        else:
            for label, value in (
                ("Weights", _gib(STUB_WEIGHTS_BYTES)),
                ("KV cache", _gib(STUB_KV_BYTES)),
                ("Compute buffers", _gib(STUB_COMPUTE_BYTES)),
            ):
                if label not in detail:
                    fail(f"the breakdown has no {label!r} line; text={detail!r}")
                elif value not in detail:
                    fail(
                        f"the {label!r} line does not show the returned {value!r}; "
                        f"text={detail!r}"
                    )
                else:
                    info(f"OK breakdown: {label} = {value}")
            # The KV note is built from the response's context and cache dtype; the stub echoes the request.
            if priced is not None and priced_ctx is not None:
                echoed = f"{priced_ctx:,} tokens"
                if echoed not in detail:
                    fail(
                        f"the KV note does not quote the priced context {echoed!r}, so the "
                        f"response's n_ctx is not what the row is displaying; text={detail!r}"
                    )
                else:
                    info(f"OK round trip: the KV note quotes {echoed}")
            layers_note = f"{STUB_GPU_LAYERS} of {STUB_LAYER_COUNT + 1} layers on GPU"
            if layers_note not in detail:
                runtime_warn(
                    f"the Weights line does not carry the placement note {layers_note!r}; "
                    f"text={detail!r}"
                )
            else:
                info(f"OK breakdown: placement note {layers_note!r}")
        shoot("05-row-expanded")

    # 4. An unavailable estimate hides the row (HARD).
    step("available:false hides the row")
    _mode[0] = "unavailable"
    _unavailable_ctx, unavailable_record = reprice(popover, "available:false hides the row")
    if unavailable_record is not None and unavailable_record["mode"] != "unavailable":
        fail(
            "the re-price was served before the endpoint was switched to unavailable, so "
            "the hide below would be measuring the wrong response"
        )
    if wait_for_row(False):
        info("OK hide: available:false removed the row")
    else:
        fail(
            "the row is still on screen after the estimate came back available:false; "
            f"header={header_text()!r}"
        )
    shoot("06-unavailable")

    # 5. Restoring the available response brings the row back (HARD).
    step("restoring an available estimate brings the row back")
    _mode[0] = "available"
    restored_ctx, restored = reprice(popover, "restoring brings the row back")
    if restored is not None and restored["mode"] != "available":
        fail(
            "the restoring re-price was served before the endpoint was switched back, so "
            "the row coming back below would not be attributable to it"
        )
    if wait_for_row(True):
        head = header_text()
        if _gib(STUB_TOTAL_BYTES) in head:
            info("OK restore: the row came back with the returned total")
        else:
            fail(f"the row came back without the returned total; header={head!r}")
        detail = breakdown_text()
        echoed = f"{restored_ctx:,} tokens" if restored_ctx is not None else ""
        if detail and echoed and echoed not in detail:
            soft_fail(
                f"the restored row still quotes an older context; expected {echoed!r} in "
                f"{detail!r}"
            )
    else:
        fail(
            "the row did not come back once the estimate was available again, so the hide "
            "gate above proves nothing about the response and everything about the panel"
        )
    shoot("07-restored")

    # 6. A 404 also hides it (HARD). Must be last: after one non-OK answer the panel never re-prices again.
    step("HTTP 404 hides the row")
    _mode[0] = "http404"
    _not_found_ctx, not_found_record = reprice(popover, "404 hides the row")
    if not_found_record is not None and not_found_record["mode"] != "http404":
        fail(
            "the re-price was served before the endpoint was switched to 404, so the hide "
            "below would be measuring the wrong response"
        )
    if wait_for_row(False):
        info("OK hide: a 404 removed the row rather than surfacing an error")
    else:
        fail(f"the row survived a 404 from the estimate endpoint; header={header_text()!r}")
    if page_errors:
        fail(f"a 404 estimate produced page errors instead of a hidden row: {page_errors[:3]!r}")
    shoot("08-http404")

    close_picker()

    if page_errors:
        fail(f"page errors during run: {page_errors[:3]!r}")

    write_transcript()
    browser.close()

if _failed:
    print(f"[ui-memest] RESULT: FAIL ({len(_failed)} issue(s))", flush = True)
    for m in _failed:
        print(f"[ui-memest]   - {m}", flush = True)
    sys.exit(1)
print(f"[ui-memest] RESULT: PASS ({len(exchanges)} estimate exchange(s) recorded)", flush = True)
sys.exit(0)
