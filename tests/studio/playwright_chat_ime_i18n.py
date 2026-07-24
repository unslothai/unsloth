# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth chat composer IME + multilingual regression smoke.

Covers: stuck IME composition (#5318 / PR #5327), multilingual paste round-trip,
stuck compositionend (#5546), and Mac input-method switch recovery (keydown/blur).
Model-free; the bug surface is the composer, not inference. Env contract matches
playwright_chat_ui.py: BASE_URL, STUDIO_NEW_PW, PW_ART_DIR, STUDIO_UI_STRICT.
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

# Wall-clock cap: 5 min leaves cold-launch headroom over the 30-60s run.
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_IME_WALL_TIMEOUT_S", "300"))


# One greeting + arithmetic per script; each catches a distinct Unicode class.
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
    """Hard fail in STRICT mode, else info-warn (mirrors playwright_chat_ui.py)."""
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
    expected_probe_cancel_500s = [0]

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

    # 1. Bootstrap auth via /change-password; retry absorbs React form-detach races.
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

    # 2. Wait for composer mount (no GGUF; the bug is React state, not inference).
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
            f'composer is missing dir="auto" (got {dir_attr!r}); RTL ' "languages will render LTR."
        )
    else:
        info('composer dir="auto" present')

    # Source-level guard: grep the unmounted edit/compare composers' JSX for dir="auto".
    _repo_root = Path(__file__).resolve().parents[2]
    _thread_src = (
        _repo_root / "studio/frontend/src/components/assistant-ui/thread.tsx"
    ).read_text()
    _shared_src = (_repo_root / "studio/frontend/src/features/chat/shared-composer.tsx").read_text()
    _edit_idx = _thread_src.find("aui-edit-composer-input")
    if _edit_idx == -1 or 'dir="auto"' not in _thread_src[_edit_idx : _edit_idx + 600]:
        soft_fail('edit composer source is missing dir="auto"')
    else:
        info('edit composer dir="auto" present (source)')
    _compare_idx = _shared_src.find("Send to both models")
    if (
        _compare_idx == -1
        or 'dir="auto"' not in _shared_src[max(_compare_idx - 400, 0) : _compare_idx + 400]
    ):
        soft_fail('compare composer source is missing dir="auto"')
    else:
        info('compare composer dir="auto" present (source)')

    def read_value() -> str:
        return composer.evaluate("(el) => el.value")

    def set_value_via_setter(s: str) -> str:
        """Write via React's setter + paste event, await two rAFs so the controlled
        value commits before readback (plain `.value=s` is overwritten on render)."""
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

    def restore_idle_composer_after_probe(label: str) -> None:
        """Cancel a real run started by a submit probe before the next case."""
        stop_btn = page.locator('button[aria-label="Stop generating"]')
        send_btn = page.locator('button[aria-label="Send message"]')
        allowed_cancel_500 = False
        try:
            stop_btn.wait_for(state = "visible", timeout = 5_000)
            expected_probe_cancel_500s[0] += 1
            allowed_cancel_500 = True
            stop_btn.click(timeout = 5_000)
            info(f"{label}: stopped generation started by submit probe")
        except Exception:
            if allowed_cancel_500:
                expected_probe_cancel_500s[0] = max(0, expected_probe_cancel_500s[0] - 1)
        try:
            expect(send_btn).to_be_visible(timeout = 15_000)
        except Exception:
            shoot(f"{label}-idle-restore-FAIL")
            fail(
                "Composer did not return to idle after the submit probe; "
                "Send button is unavailable for the next IME regression case."
            )

    # 3. Baseline: ASCII keyboard typing works.
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

    # 6. Stuck IME repro (#5318): duplicate compositionstart wedges
    #    isComposing=true; PR #5327 clears it on non-composing input.
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
    # On the broken build React drops 'abcd' and reconciles el.value to ''.
    # wait_for_function crosses the microtask boundary to see committed state.
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
    # Cross-check isComposing via the Send button: it stays disabled while
    # isComposing is true (PR #5327).
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

    # 6b. WSL+Chrome repro (#5546): Chrome never emits compositionend after the
    #     IME commit, so the watchdog must release the composing flag once events
    #     go silent.
    step("BUG REPRO: stuck compositionend recovery (issue #5546)")
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
            // Deliberately omit compositionend; that is the WSL/Chrome
            // bug surface. The watchdog in useImeComposerInputHandlers
            // should reset isComposing after IME_STUCK_TIMEOUT_MS.
        }"""
    )
    send_btn_5546 = page.locator('button[aria-label="Send message"]')
    if send_btn_5546.count() == 0:
        soft_fail("Send button not found for #5546 repro")
    else:
        # Watchdog is 2500ms; allow generous slack for slow CI.
        try:
            expect(send_btn_5546).not_to_be_disabled(timeout = 8_000)
            info("Send button enabled after compositionend never fired")
        except Exception:
            shoot("06b-compositionend-watchdog-FAIL")
            fail(
                "Send button stayed disabled with no compositionend; "
                "watchdog did not release the composing flag (issue #5546)."
            )
    after_value = read_value()
    if "你好" not in after_value:
        soft_fail(f"compositionend-watchdog repro lost committed text: {after_value!r}")
    shoot("06b-compositionend-watchdog")
    info("compositionend watchdog recovery PASS")
    clear()

    # 6c. Watchdog-race repro: after the watchdog clears composingRef, a later
    #     IME keydown (keyCode 229) must not slip preedit text through submit.
    #     The onKeyDown gate re-pins composingRef so handleSubmit refuses.
    step("BUG REPRO: keydown re-pin after watchdog cleared composing (issue #5546 follow-up)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'半'}));
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, el.value + '半角');
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertCompositionText',
                data:'半角', isComposing:true,
            }));
        }"""
    )
    send_btn_keydown = page.locator('button[aria-label="Send message"]')
    # Wait past the watchdog so composingRef has cleared.
    try:
        expect(send_btn_keydown).not_to_be_disabled(timeout = 8_000)
    except Exception:
        soft_fail("watchdog did not clear before keydown re-pin test")
    # Fire IME-confirm Enter (keyCode 229) then submit synchronously: the keydown
    # gate re-pins composingRef before handleSubmit, so preedit text is retained.
    submit_probe = composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles:true, key:'Enter', code:'Enter', keyCode:229,
                isComposing:true,
            }));
            const form = el.closest('form');
            const before = el.value;
            try { form && form.requestSubmit(); } catch (e) {}
            return {before, after: el.value, cleared: before !== '' && el.value === ''};
        }"""
    )
    if submit_probe.get("cleared"):
        shoot("06c-keydown-repin-FAIL")
        fail(
            "Form submitted after an IME keydown -- preedit text leaked "
            "through the watchdog gap (#5546 follow-up regression)."
        )
    info(f"Form submit refused after IME keydown; textarea retained {submit_probe.get('after')!r}")
    shoot("06c-keydown-repin")
    info("keydown re-pin gate PASS")
    clear()

    # 6d. Keydown re-pin must also re-arm the watchdog: on the WSL+Chrome
    #     stuck-compositionend path no follow-up event arrives, so after re-pin
    #     the watchdog must clear composingRef again or Send re-locks forever.
    step("BUG REPRO: keydown re-pin re-arms watchdog (#5546 follow-up regression)")
    clear()
    composer.click()
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
            el.dispatchEvent(new CompositionEvent('compositionupdate', {bubbles:true, data:'你'}));
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, el.value + '你好');
            el.dispatchEvent(new InputEvent('input', {
                bubbles:true, inputType:'insertCompositionText',
                data:'你好', isComposing:true,
            }));
        }"""
    )
    send_btn_rearm = page.locator('button[aria-label="Send message"]')
    # First watchdog cycle: wait for it to clear composingRef.
    try:
        expect(send_btn_rearm).not_to_be_disabled(timeout = 8_000)
    except Exception:
        soft_fail("watchdog did not clear before re-arm test (first cycle)")
    # IME-confirm keydown re-pins composingRef; without the re-arm fix the
    # watchdog never runs again and Send stays blocked forever.
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles:true, key:'Enter', code:'Enter', keyCode:229,
                isComposing:true,
            }));
        }"""
    )
    # Second watchdog cycle: requestSubmit() after the re-armed window must be
    # allowed; the buggy build stays gated forever.
    rearm_probe = page.evaluate(
        """async (selector) => {
            const ta = document.querySelector(selector);
            const form = ta && ta.closest('form');
            if (!form || !ta) return {ok: false, reason: 'composer missing'};
            const before = ta.value;
            // Wait past the 2500ms watchdog + slack so the re-armed timer
            // fires. If the fix is missing this still resolves but the
            // submit will not flush the textarea.
            await new Promise(r => setTimeout(r, 3500));
            try { form.requestSubmit(); } catch (e) {}
            // Give the submit handler a tick to flush state.
            await new Promise(r => setTimeout(r, 250));
            return {ok: true, before, after: ta.value};
        }""",
        'textarea[aria-label="Message input"]',
    )
    if rearm_probe.get("ok") and rearm_probe.get("after") == rearm_probe.get("before"):
        shoot("06d-keydown-rearm-FAIL")
        fail(
            "After the keydown re-pin the watchdog never re-armed; Send "
            "stayed permanently locked on the WSL+Chrome stuck-end path "
            "(#5546 follow-up regression)."
        )
    info(
        "watchdog re-armed after keydown re-pin: textarea flushed from "
        f"{rearm_probe.get('before')!r} to {rearm_probe.get('after')!r}"
    )
    shoot("06d-keydown-rearm")
    info("keydown re-pin re-arm PASS")
    restore_idle_composer_after_probe("06d-keydown-rearm")
    clear()

    # 6e. Mac input-method switch - onKeyDown immediate recovery.
    #     A Mac IME switch fires compositionstart but never compositionend; the
    #     first English keydown (isComposing=false) must clear composingRef via
    #     the onKeyDown else-if branch, before the 2500ms watchdog fires.
    #     To isolate that path (not onChange) we dispatch a synthetic keydown with
    #     NO follow-up input event, so onChange never fires.
    step("BUG REPRO: Mac IME switch - onKeyDown immediate recovery")
    clear()
    composer.click()
    # Seed sendable content so Send's state reflects composition only, not
    # empty-content gating (insertFromPaste leaves composingRef false).
    set_value_via_setter("hello")
    # Switch TO Chinese: compositionstart fires but compositionend never arrives.
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
        }"""
    )
    # Let React process compositionstart and update isComposing.
    page.wait_for_timeout(200)
    send_btn_mac_kd = page.locator('button[aria-label="Send message"]')
    # Dispatch ONLY a keydown (no input event) so onChange never fires and the
    # onKeyDown else-if branch is the only path that can clear composingRef.
    # page.keyboard.type() would fire onChange too and mask a regression.
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles: true, key: 'a', code: 'KeyA', keyCode: 65,
                isComposing: false,
            }));
        }"""
    )
    if send_btn_mac_kd.count() == 0:
        soft_fail("Send button not found for Mac IME switch (onKeyDown) repro")
    else:
        try:
            # 1500ms < 2500ms watchdog: only the onKeyDown else-if path can
            # clear composingRef this quickly.
            expect(send_btn_mac_kd).not_to_be_disabled(timeout = 1_500)
            info(
                "Send button enabled within 1500ms after Mac IME switch + "
                "English keydown (onKeyDown else-if branch fired, not watchdog)"
            )
        except Exception:
            shoot("06e-mac-ime-keydown-FAIL")
            fail(
                "Send button stayed disabled after Mac input-method switch + "
                "English keydown; the onKeyDown else-if branch did not clear "
                "composingRef immediately (expected recovery in < 1500ms)."
            )
    shoot("06e-mac-ime-keydown")
    info("Mac IME switch onKeyDown recovery PASS")
    clear()

    # 6f. Candidate-confirming Enter must not unblock submit.
    #     Some IMEs report it as isComposing=false/keyCode=13 while composingRef
    #     is still pinned; it must be swallowed and keep Send disabled, else it
    #     submits before the candidate is committed.
    step("BUG REPRO: Mac IME switch - Enter must not unblock submit")
    clear()
    composer.click()
    set_value_via_setter("hello")
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
        }"""
    )
    page.wait_for_timeout(200)
    send_btn_mac_enter = page.locator('button[aria-label="Send message"]')
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles: true, cancelable: true, key: 'Enter', code: 'Enter',
                keyCode: 13, isComposing: false,
            }));
        }"""
    )
    try:
        expect(send_btn_mac_enter).to_be_disabled(timeout = 500)
        info("Enter while composingRef is stuck kept Send disabled")
    except Exception:
        shoot("06f-mac-ime-enter-guard-FAIL")
        fail(
            "Enter cleared a stuck composingRef immediately; candidate-confirming "
            "Enter can fall through to submit."
        )
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles: true, key: 'a', code: 'KeyA', keyCode: 65,
                isComposing: false,
            }));
        }"""
    )
    try:
        expect(send_btn_mac_enter).not_to_be_disabled(timeout = 1_500)
        info("non-Enter key after Enter guard still recovers immediately")
    except Exception:
        shoot("06f-mac-ime-enter-guard-recovery-FAIL")
        fail("After guarding Enter, a later non-IME key did not clear composingRef immediately.")
    shoot("06f-mac-ime-enter-guard")
    info("Mac IME switch Enter guard PASS")
    clear()

    # 6g. Mac input-method switch - onBlur immediate recovery.
    #     Some Mac IME switches steal textarea focus; onBlur resets composingRef
    #     unconditionally. Safe because the OS commits/cancels composition before
    #     surrendering focus, so blur is a reliable reset point.
    step("BUG REPRO: Mac IME switch - onBlur immediate recovery")
    clear()
    composer.click()
    # Seed sendable content so Send's state reflects composition only.
    set_value_via_setter("hello")
    # Switch TO Chinese: compositionstart fires, compositionend never comes.
    composer.evaluate(
        """(el) => {
            el.focus();
            el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true, data:''}));
        }"""
    )
    page.wait_for_timeout(200)
    send_btn_mac_blur = page.locator('button[aria-label="Send message"]')
    # Blur to simulate the OS stealing focus during an IME switch.
    composer.evaluate("(el) => el.blur()")
    # onBlur clears composition; re-focus so React renders the updated Send state.
    composer.click()
    if send_btn_mac_blur.count() == 0:
        soft_fail("Send button not found for Mac IME switch (onBlur) repro")
    else:
        try:
            # 1500ms < 2500ms watchdog: only onBlur can clear composingRef this
            # quickly when no keydown is fired.
            expect(send_btn_mac_blur).not_to_be_disabled(timeout = 1_500)
            info(
                "Send button enabled within 1500ms after Mac IME switch + "
                "textarea blur (onBlur handler fired, not watchdog)"
            )
        except Exception:
            shoot("06f-mac-ime-blur-FAIL")
            fail(
                "Send button stayed disabled after Mac input-method switch + "
                "textarea blur; the onBlur handler did not reset composingRef "
                "(expected recovery in < 1500ms)."
            )
    shoot("06g-mac-ime-blur")
    info("Mac IME switch onBlur recovery PASS")
    clear()

    # 7. Final state: filter benign 401 noise via is_benign_*; fail on real errors.
    shoot("07-final")
    real_page_errors = [e for e in page_errors if not is_benign_page_error(e)]
    probe_cancel_500_allowance = expected_probe_cancel_500s[0]
    real_console_errors = []
    for error in console_errors:
        if is_benign_console_error(error):
            continue
        if (
            probe_cancel_500_allowance > 0
            and "Failed to load resource: the server responded with a status of 500" in error
            and "Internal Server Error" in error
        ):
            probe_cancel_500_allowance -= 1
            continue
        real_console_errors.append(error)
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
        f"normal_composition=OK stuck_recovery=OK "
        f"compositionend_watchdog=OK keydown_repin=OK "
        f"keydown_repin_rearm=OK "
        f"mac_ime_switch_keydown=OK mac_ime_switch_enter_guard=OK "
        f"mac_ime_switch_blur=OK"
    )
    _watchdog.cancel()
    browser.close()
