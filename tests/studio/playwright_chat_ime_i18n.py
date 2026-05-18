# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio chat composer IME + multilingual regression smoke.

Covers two surfaces:
  A. Stuck IME composition (issue #5318 / PR #5327): duplicate
     compositionstart with no compositionend left isComposing=true,
     dropping all subsequent keystrokes including ASCII.
  B. Multilingual paste round-trip across 31 scripts -- guards the
     controlled-textarea / React state plumbing against Unicode mangling.

Model-free; the bug surface is the composer, not inference.

Env contract matches playwright_chat_ui.py:
  BASE_URL, STUDIO_NEW_PW, PW_ART_DIR, STUDIO_UI_STRICT.
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_console_error,
    is_benign_page_error,
    recover_or_replace_page,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ["STUDIO_NEW_PW"]
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_ime")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"

# Wall-clock cap. Realistic run is 30-60s; 5 min leaves cold-launch headroom.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_IME_WALL_TIMEOUT_S", "300"))


# One short greeting + arithmetic per script (ordered by speaker count) --
# each entry catches a distinct class of Unicode regression.
I18N_SAMPLES = [
    ("en", "English", "Hello, 1+1=2"),
    ("zh-CN", "Chinese (Simplified)", "你好，1+1=2"),
    ("es", "Spanish", "Hola, 1+1=2"),
    ("hi", "Hindi (Devanagari)", "नमस्ते, 1+1=2"),
    ("ar", "Arabic (RTL)", "مرحبا، ١+١=٢"),
    ("bn", "Bengali", "নমস্কার, ১+১=২"),
    ("pt", "Portuguese", "Olá, 1+1=2"),
    ("ru", "Russian (Cyrillic)", "Привет, 1+1=2"),
    ("ja", "Japanese", "こんにちは、1+1=2"),
    ("pa", "Punjabi (Gurmukhi)", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, 1+1=2"),
    ("de", "German", "Hallo, 1+1=2"),
    ("jv", "Javanese", "Halo, 1+1=2"),
    ("ko", "Korean (Hangul)", "안녕하세요, 1+1=2"),
    ("fr", "French", "Bonjour, 1+1=2"),
    ("tr", "Turkish", "Merhaba, 1+1=2"),
    ("vi", "Vietnamese (diacritics)", "Xin chào, 1+1=2"),
    ("ur", "Urdu (Arabic-Naskh)", "ہیلو، 1+1=2"),
    ("ta", "Tamil", "வணக்கம், 1+1=2"),
    ("te", "Telugu", "నమస్తే, 1+1=2"),
    ("mr", "Marathi (Devanagari)", "नमस्कार, 1+1=2"),
    ("it", "Italian", "Ciao, 1+1=2"),
    ("th", "Thai", "สวัสดี, ๑+๑=๒"),
    ("pl", "Polish", "Cześć, 1+1=2"),
    ("uk", "Ukrainian (Cyrillic)", "Привіт, 1+1=2"),
    ("fa", "Persian (RTL)", "سلام، ۱+۱=۲"),
    ("nl", "Dutch", "Hallo, 1+1=2"),
    ("he", "Hebrew (RTL)", "שלום, 1+1=2"),
    ("el", "Greek", "Γειά, 1+1=2"),
    ("id", "Indonesian", "Halo, 1+1=2"),
    ("sw", "Swahili", "Habari, 1+1=2"),
    ("emoji", "Emoji + ZWJ + flag", "👋 🇺🇳 👨‍👩‍👧‍👦 1+1=2"),
]


_n = [0]


def step(s):
    print(f"[ime] STEP {s}", flush = True)


def info(s):
    print(f"[ime] {s}", flush = True)


def fail(m):
    raise AssertionError(f"[ime] FAIL: {m}")


def soft_fail(m):
    """Hard fail in STRICT mode, info-warn otherwise. Mirrors playwright_chat_ui.py."""
    if STRICT:
        fail(m)
    info(f"WARN (strict-off): {m}")


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ime",
        info = info,
    )
    wait_for_health(BASE, timeout = 30.0, info = info)
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    install_view_transition_killer(ctx)
    page = ctx.new_page()
    page.set_default_timeout(60_000)

    page_errors: list[str] = []
    console_errors: list[str] = []

    def _on_console(m):
        if m.type != "error":
            return
        try:
            console_errors.append(m.text)
        except Exception:
            return

    def _attach_listeners(target):
        target.on("pageerror", lambda e: page_errors.append(str(e)))
        target.on("console", _on_console)

    _attach_listeners(page)

    def shoot(name):
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as _shoot_err:
            info(f"WARN: screenshot {name} failed: {_shoot_err}")

    # 1. Bootstrap auth via /change-password (mirrors playwright_chat_ui.py
    #    retry-on-rerender to absorb React form-detach races).
    step("change-password through UI (Setup your account)")
    form_err: Exception | None = None
    for _form_attempt in range(3):
        try:
            page.goto(
                f"{BASE}/change-password",
                wait_until = "domcontentloaded",
                timeout = 60_000,
            )
            try:
                page.wait_for_load_state("networkidle", timeout = 30_000)
            except Exception:
                pass
            pw_field = page.locator("#new-password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            shoot("01-change-password-filled")
            status, _ = click_and_wait_for_response(
                page,
                url_substr = "/api/auth/change-password",
                method = "POST",
                do_click = lambda: page.locator('button[type="submit"]').click(),
                timeout_ms = 30_000,
                info = lambda m: print(f"[ime]   {m}", flush = True),
            )
            if status is not None and status >= 400:
                raise AssertionError(f"change-password POST returned {status}")
            form_err = None
            break
        except Exception as e:
            form_err = e
            info(
                f"change-password attempt {_form_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            if _form_attempt < 2:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ime]   recovery: {m}", flush = True),
                )
                _attach_listeners(page)
    if form_err is not None:
        raise form_err

    # 2. Wait for composer mount. No GGUF: the bug surface is React state, not inference.
    step("wait for composer to mount")
    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
    composer = page.locator('textarea[aria-label="Message input"]')
    _mount_err: Exception | None = None
    for _mount_attempt in range(2):
        try:
            composer.wait_for(state = "visible", timeout = 60_000)
            _mount_err = None
            break
        except Exception as e:
            _mount_err = e
            info(
                f"composer.wait_for attempt {_mount_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            try:
                shoot(f"02-composer-wait-attempt-{_mount_attempt + 1}-fail")
            except Exception:
                pass
            if _mount_attempt == 0:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ime]   recovery: {m}", flush = True),
                )
                _attach_listeners(page)
                composer = page.locator('textarea[aria-label="Message input"]')
    if _mount_err is not None:
        raise _mount_err
    composer.click()
    shoot("02-composer-focused")

    # Main composer must carry dir="auto" so RTL flows right-to-left.
    dir_attr = composer.evaluate("(el) => el.getAttribute('dir')")
    if dir_attr != "auto":
        soft_fail(
            f'composer is missing dir="auto" (got {dir_attr!r}); RTL '
            "languages will render LTR."
        )
    else:
        info('composer dir="auto" present')

    # Source-level guard for the edit and compare composers (neither
    # is mounted here): grep the JSX for dir="auto" inside each block.
    _repo_root = Path(__file__).resolve().parents[2]
    _thread_src = (
        _repo_root / "studio/frontend/src/components/assistant-ui/thread.tsx"
    ).read_text()
    _shared_src = (
        _repo_root / "studio/frontend/src/features/chat/shared-composer.tsx"
    ).read_text()
    _edit_idx = _thread_src.find("aui-edit-composer-input")
    if _edit_idx == -1 or 'dir="auto"' not in _thread_src[_edit_idx : _edit_idx + 600]:
        soft_fail('edit composer source is missing dir="auto"')
    else:
        info('edit composer dir="auto" present (source)')
    _compare_idx = _shared_src.find("Send to both models")
    if (
        _compare_idx == -1
        or 'dir="auto"'
        not in _shared_src[max(_compare_idx - 400, 0) : _compare_idx + 400]
    ):
        soft_fail('compare composer source is missing dir="auto"')
    else:
        info('compare composer dir="auto" present (source)')

    def read_value() -> str:
        return composer.evaluate("(el) => el.value")

    def set_value_via_setter(s: str) -> str:
        """Write via React's monkey-patched setter + paste input event,
        then await two rAFs so the controlled value is committed before
        readback (plain `.value=s` would be overwritten on next render)."""
        return composer.evaluate(
            """async (el, v) => {
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                setter.call(el, v);
                el.dispatchEvent(new InputEvent('input', {
                    bubbles: true,
                    inputType: 'insertFromPaste',
                    data: v,
                }));
                await new Promise((r) => requestAnimationFrame(r));
                await new Promise((r) => requestAnimationFrame(r));
                return el.value;
            }""",
            s,
        )

    def clear() -> None:
        set_value_via_setter("")

    # 3. Baseline: ASCII keyboard typing works. Bail fast if not.
    step("baseline ASCII keyboard typing")
    clear()
    composer.click()
    for ch in "hello world":
        page.keyboard.type(ch)
    got = read_value()
    if got != "hello world":
        fail(f"ASCII typing readback {got!r} != 'hello world'")
    info("baseline ASCII OK")
    shoot("03-baseline-ascii")
    clear()

    # 4. Multilingual paste round-trip; byte-for-byte readback required.
    step(f"multilingual paste round-trip ({len(I18N_SAMPLES)} samples)")
    paste_failures: list[tuple[str, str, str, str]] = []
    for code, label, text in I18N_SAMPLES:
        got = set_value_via_setter(text)
        if got != text:
            paste_failures.append((code, label, text, got))
            info(f"  {code:>6} ({label}): FAIL -- got {got!r}")
        else:
            info(f"  {code:>6} ({label}): OK")
        clear()
    if paste_failures:
        shoot("04-paste-failures")
        lines = [
            f"  {code} ({label}): want={want!r} got={got!r}"
            for code, label, want, got in paste_failures
        ]
        fail(
            f"{len(paste_failures)}/{len(I18N_SAMPLES)} languages failed paste round-trip:\n"
            + "\n".join(lines)
        )
    info(f"all {len(I18N_SAMPLES)} multilingual paste samples OK")
    shoot("04-paste-all-ok")

    # 5. Healthy IME composition (compositionstart/update/end + insert events).
    step("normal IME composition (compose 你好)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'你'}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'你好'}));
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, el.value + '你好');
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertCompositionText',
                data:'你好', isComposing:true,
            }));
            el.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true, data:'你好'}));
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertFromComposition', data:'你好',
            }));
        }"""
    )
    got = read_value()
    if "你好" not in got:
        shoot("05-normal-composition-FAIL")
        fail(f"normal composition readback {got!r} missing '你好'")
    info(f"normal composition OK: ta.value={got!r}")
    shoot("05-normal-composition")
    clear()

    # 6. Stuck IME repro for issue #5318: duplicate compositionstart with
    #    no compositionend wedged isComposing=true and dropped ASCII keys.
    #    PR #5327 cleared the stale state on non-composing input.
    step("BUG REPRO: stuck IME composition recovery (issue #5318)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            // Duplicate compositionstart with NO matching compositionend.
            // This is exactly the event sequence observed from the IMEs
            // in issue #5318 (kei-yamazaki / langxiaopiao030 / PapyrusNotes).
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
        }"""
    )
    # Drive the real keyboard path; on the broken build React drops
    # 'abcd' and reconciles el.value back to ''. wait_for_function
    # crosses the microtask boundary so we see committed React state.
    page.keyboard.type("abcd")
    try:
        page.wait_for_function(
            """(el) => el.value === 'abcd'""",
            composer.element_handle(),
            timeout = 5_000,
        )
    except Exception:
        pass
    after_key = read_value()
    info(f"after_key='abcd' readback={after_key!r}")
    shoot("06-stuck-composition-recovery")
    if after_key != "abcd":
        fail(
            "stuck-composition repro: keyboard 'abcd' was not preserved after "
            f"duplicate compositionstart; readback {after_key!r}. React state "
            "likely still stuck in isComposing=true (issue #5318 / before "
            "PR #5327)."
        )
    # Cross-check React's view of isComposing via the Send button:
    # ComposerAction stays disabled while isComposing is true (PR #5327).
    send_btn = page.locator('button[aria-label="Send message"]')
    if send_btn.count() == 0:
        soft_fail("Send button not found after stuck-composition recovery")
    else:
        try:
            expect(send_btn).not_to_be_disabled(timeout = 5_000)
            info("Send button correctly enabled after stuck-composition recovery")
        except Exception:
            soft_fail(
                "Send button still disabled after stuck-composition recovery -- "
                "React isComposing state likely never cleared"
            )
    info("stuck-composition recovery PASS")
    clear()

    # 7. Final state. The change-password redirect emits benign 401 noise,
    #    so we filter via is_benign_* and only fail on real errors.
    shoot("07-final")
    real_page_errors = [e for e in page_errors if not is_benign_page_error(e)]
    real_console_errors = [e for e in console_errors if not is_benign_console_error(e)]
    info(
        f"page_errors={len(page_errors)} ({len(real_page_errors)} non-benign); "
        f"console_errors={len(console_errors)} "
        f"({len(real_console_errors)} non-benign)"
    )
    if page_errors:
        info(f"first page error: {page_errors[0][:200]!r}")
    if console_errors:
        info(f"first console error: {console_errors[0][:200]!r}")
    if real_page_errors:
        fail(
            f"{len(real_page_errors)} non-benign pageerror events; "
            f"first={real_page_errors[0][:200]!r}"
        )
    if real_console_errors:
        fail(
            f"{len(real_console_errors)} non-benign console.error events; "
            f"first={real_console_errors[0][:200]!r}"
        )

    info(
        f"DONE: ascii=OK paste={len(I18N_SAMPLES)}/{len(I18N_SAMPLES)} "
        f"normal_composition=OK stuck_recovery=OK"
    )
    _watchdog.cancel()
    browser.close()
