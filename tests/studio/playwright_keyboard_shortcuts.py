# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keyboard shortcut behaviour, in a real browser, on three emulated platforms.

Drives smoke-shortcuts.html: the real registry, the real store and the real
useShortcut against a browser's own keyboard. The node suite reaches the pure
functions, but a listener is not a pure function, so what a chord does to a
focused button, to a text field, on auto-repeat, under AltGr, or to a stored
binding carried over from another platform is only answerable here.

    SMOKE_ENGINES=chromium,firefox,webkit python3 tests/studio/playwright_keyboard_shortcuts.py

Platform is emulated the way the app reads it, through navigator.platform and
the user agent, because isMacPlatform() is memoised on first call.
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
)

PORT = int(os.environ.get("SMOKE_PORT", "5407"))
ENGINES = [e for e in os.environ.get("SMOKE_ENGINES", "chromium").split(",") if e]
URL = f"http://127.0.0.1:{PORT}/smoke-shortcuts.html"

# navigator.platform, and a user agent to match.
PLATFORMS = {
    "macOS": ("MacIntel", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SmokeUA"),
    "Windows": ("Win32", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmokeUA"),
    "Linux": ("Linux x86_64", "Mozilla/5.0 (X11; Linux x86_64) SmokeUA"),
}
MOD = {"macOS": "Meta", "Windows": "Control", "Linux": "Control"}

failures: list[str] = []
passed = 0


def check(
    engine: str,
    platform: str,
    name: str,
    ok: bool,
    detail: str = "",
) -> None:
    global passed
    if ok:
        passed += 1
        return
    failures.append(f"{engine}/{platform}: {name}\n      {detail}")


def to_press(value: str, platform: str) -> str:
    """A stored binding, in the key syntax Playwright presses."""
    parts = value.split("+")
    mods = [
        MOD[platform] if token == "Mod" else {"Ctrl": "Control"}.get(token, token)
        for token in parts[:-1]
    ]
    return "+".join([*mods, parts[-1]])


def actions(page) -> list[str]:
    return [e["action"] for e in page.evaluate("window.__shortcutsSmoke.fired()")]


def details(page) -> list[str]:
    return [e.get("detail") for e in page.evaluate("window.__shortcutsSmoke.fired()")]


def reset(page) -> None:
    page.evaluate("window.__shortcutsSmoke.reset()")


def reload_with(page, raw: str | None) -> None:
    """Seed storage the way an install of any age would have, then restart."""
    page.evaluate(
        """([raw]) => {
            if (raw === null) localStorage.removeItem("unsloth_keyboard_shortcuts");
            else localStorage.setItem("unsloth_keyboard_shortcuts", raw);
        }""",
        [raw],
    )
    page.goto(URL)
    page.wait_for_selector("#smoke-ready")


def check_defaults(page, engine: str, platform: str) -> None:
    check(
        engine,
        platform,
        "isMacPlatform follows the emulated platform",
        page.evaluate("window.__shortcutsSmoke.registry.isMacPlatform()") == (platform == "macOS"),
    )
    rows = page.evaluate(
        """() => {
            const r = window.__shortcutsSmoke.registry;
            const mac = r.isMacPlatform();
            const out = [];
            for (const def of r.SHORTCUT_DEFS) {
                for (const slot of r.SHORTCUT_SLOTS) {
                    const value = r.defaultBindingFor(def, slot, mac);
                    if (value == null) continue;
                    const parsed = r.parseBinding(value);
                    out.push({ id: def.id, value, parsed: parsed &&
                        `${parsed.code}|${parsed.mod}|${parsed.ctrl}|${parsed.shift}|${parsed.alt}`,
                        canon: parsed && r.formatBindingValue(parsed) });
                }
            }
            return out;
        }"""
    )
    check(
        engine,
        platform,
        "every shipped default parses and is canonical",
        all(r["parsed"] and r["canon"] == r["value"] for r in rows),
    )
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(row["parsed"], set()).add(row["id"])
    clash = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    check(
        engine,
        platform,
        "no two actions claim one physical chord",
        not clash,
        json.dumps(clash)[:200],
    )
    if platform != "macOS":
        ctrl = [r["id"] for r in rows if r["parsed"].split("|")[2] == "true"]
        check(engine, platform, "no unreachable Ctrl default off macOS", not ctrl, str(ctrl))


def check_dispatch(page, engine: str, platform: str) -> None:
    reset(page)
    page.keyboard.press(to_press("Mod+Comma", platform))
    check(
        engine,
        platform,
        "a shipped chord fires from a real keypress",
        "openSettings" in actions(page),
        str(actions(page)),
    )

    # Exact match: the chord is this set of modifiers, not at least this set.
    reset(page)
    page.keyboard.press(f"{MOD[platform]}+Alt+Comma")
    check(
        engine,
        platform,
        "an extra modifier does not satisfy a chord",
        "openSettings" not in actions(page),
        str(actions(page)),
    )

    # Both slots answer. newChat is the only action that ships a pair.
    reset(page)
    page.keyboard.press(to_press("Mod+Shift+KeyO", platform))
    page.keyboard.press(to_press("Mod+KeyN", platform))
    check(
        engine,
        platform,
        "both slots of an action fire",
        actions(page).count("newChat") == 2,
        str(actions(page)),
    )


def check_text_fields(page, engine: str, platform: str) -> None:
    reset(page)
    page.focus("#smoke-input")
    page.keyboard.press(to_press("Mod+KeyK", platform))
    guarded = actions(page)
    page.evaluate("document.activeElement && document.activeElement.blur()")
    reset(page)
    page.keyboard.press(to_press("Mod+KeyK", platform))
    check(
        engine,
        platform,
        "a text-field chord stands aside while typing",
        "searchChats" not in guarded and "searchChats" in actions(page),
        f"in field {guarded}, outside {actions(page)}",
    )

    reset(page)
    page.focus("#smoke-editable")
    page.keyboard.press(to_press("Mod+KeyK", platform))
    check(
        engine,
        platform,
        "the gate covers contenteditable too",
        "searchChats" not in actions(page),
        str(actions(page)),
    )

    reset(page)
    page.focus("#smoke-composer")
    page.keyboard.press("Escape")
    declined = actions(page)
    # Escape types nothing in the composer, so declining keeps working there, and Enter, which sends, does not.
    reset(page)
    page.focus("#smoke-composer")
    page.keyboard.press("Enter")
    check(
        engine,
        platform,
        "the composer excepts Escape but not Enter",
        "decline" in declined and "approve" not in actions(page),
        f"escape {declined}, enter {actions(page)}",
    )

    reset(page)
    page.focus("#smoke-input")
    page.keyboard.press("Escape")
    check(
        engine,
        platform,
        "an ordinary field keeps its own Escape",
        "decline" not in actions(page),
        str(actions(page)),
    )
    page.evaluate("document.activeElement && document.activeElement.blur()")


def check_bare_keys(page, engine: str, platform: str) -> None:
    reset(page)
    page.keyboard.press("Enter")
    check(
        engine,
        platform,
        "bare Enter answers a waiting request",
        "approve" in actions(page),
        str(actions(page)),
    )

    # The whole reason the stand-aside exists:
    reset(page)
    page.focus("#smoke-button")
    page.keyboard.press("Enter")
    got = actions(page)
    check(
        engine,
        platform,
        "Enter on a focused button clicks it and does not approve",
        "buttonClick" in got and "approve" not in got,
        str(got),
    )

    reset(page)
    page.focus("#smoke-link")
    page.keyboard.press("Enter")
    check(
        engine,
        platform,
        "Enter on a focused link does not approve",
        "approve" not in actions(page),
        str(actions(page)),
    )

    # Escape activates nothing, so it is deliberately not stood aside.
    reset(page)
    page.focus("#smoke-button")
    page.keyboard.press("Escape")
    check(
        engine,
        platform,
        "Escape still declines from a focused button",
        "decline" in actions(page),
        str(actions(page)),
    )
    page.evaluate("document.activeElement && document.activeElement.blur()")


def check_repeat(page, engine: str, platform: str) -> None:
    for code, key, name, expected, label in (
        ("BracketRight", "]", "nextChat", 3, "a walking chord runs on auto-repeat"),
        ("Comma", ",", "openSettings", 1, "a one-shot chord ignores auto-repeat"),
    ):
        reset(page)
        page.evaluate(
            """([mac, code, key, shift]) => {
                for (let i = 0; i < 3; i++) {
                    window.dispatchEvent(new KeyboardEvent('keydown', {
                        key, code, bubbles: true, cancelable: true, shiftKey: shift,
                        metaKey: mac, ctrlKey: !mac, repeat: i > 0,
                    }));
                }
            }""",
            [platform == "macOS", code, key, code == "BracketRight"],
        )
        check(engine, platform, label, actions(page).count(name) == expected, str(actions(page)))


def check_altgr(page, engine: str, platform: str) -> None:
    # AltGr reports itself as Ctrl+Alt.
    reset(page)
    prevented = page.evaluate(
        """([mac]) => {
            const ev = new KeyboardEvent('keydown', {
                key: 'e', code: 'KeyE', bubbles: true, cancelable: true,
                metaKey: mac, ctrlKey: !mac, altKey: true,
            });
            ev.getModifierState = (m) => m === 'AltGraph';
            window.dispatchEvent(ev);
            return ev.defaultPrevented;
        }""",
        [platform == "macOS"],
    )
    got = actions(page)
    if platform == "macOS":
        check(
            engine,
            platform,
            "an Option chord still fires on macOS",
            "archiveActive" in got or "archiveSelected" in got,
            str(got),
        )
    else:
        check(
            engine,
            platform,
            "AltGr typing neither fires a chord nor loses the character",
            not got and not prevented,
            f"{got} prevented={prevented}",
        )


def check_foreign_binding(page, engine: str, platform: str) -> None:
    # A binding stored on a Mac must not fire on the bare key elsewhere. Before
    reload_with(page, json.dumps({"copySessionId": {"primary": "Ctrl+KeyG"}}))
    reset(page)
    page.evaluate(
        """() => window.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'g', code: 'KeyG', bubbles: true, cancelable: true }))"""
    )
    bare = actions(page)
    reset(page)
    page.keyboard.press("Control+KeyG")
    held = actions(page)
    if platform == "macOS":
        check(
            engine,
            platform,
            "a Ctrl chord fires on Ctrl, not on the bare key",
            "copySessionId" in held and "copySessionId" not in bare,
            f"bare {bare}, ctrl {held}",
        )
    else:
        check(
            engine,
            platform,
            "a Mac Ctrl chord is inert off macOS",
            "copySessionId" not in bare and "copySessionId" not in held,
            f"bare {bare}, ctrl {held}",
        )
    reload_with(page, None)


def check_storage(page, engine: str, platform: str) -> None:
    cases = [
        (
            "a pre-alternate rebind becomes a primary override",
            json.dumps({"newChat": "Mod+KeyJ"}),
            lambda a: a["newChat"]["primary"] == "Mod+KeyJ"
            and a["newChat"]["alternate"] == "Mod+KeyN",
        ),
        (
            "an action cleared before alternates existed stays cleared",
            json.dumps({"newChat": None}),
            lambda a: a["newChat"]["primary"] is None and a["newChat"]["alternate"] is None,
        ),
        (
            "an id from an older build is dropped",
            json.dumps({"aRemovedAction": "Mod+KeyJ", "newChat": "Mod+KeyJ"}),
            lambda a: a["newChat"]["primary"] == "Mod+KeyJ",
        ),
        (
            "the current shape round-trips",
            json.dumps({"toggleSidebar": {"primary": "Mod+Alt+KeyW"}}),
            lambda a: a["toggleSidebar"]["primary"] == "Mod+Alt+KeyW",
        ),
        (
            "corrupt JSON falls back to the defaults",
            "{not json",
            lambda a: a["openSettings"]["primary"] == "Mod+Comma",
        ),
        (
            "a non-object payload falls back to the defaults",
            "42",
            lambda a: a["openSettings"]["primary"] == "Mod+Comma",
        ),
        (
            "an array payload falls back to the defaults",
            "[1,2]",
            lambda a: a["openSettings"]["primary"] == "Mod+Comma",
        ),
        (
            "a junk slot type reverts that action",
            json.dumps({"toggleSidebar": {"primary": 5}}),
            lambda a: a["toggleSidebar"]["primary"] == "Mod+KeyB",
        ),
        (
            "nothing stored takes the defaults",
            None,
            lambda a: a["openSettings"]["primary"] == "Mod+Comma",
        ),
    ]
    for name, raw, predicate in cases:
        reload_with(page, raw)
        resolved = page.evaluate(
            """() => {
                const s = window.__shortcutsSmoke.store;
                return s.resolveAllBindings(s.useKeyboardShortcutsStore.getState().overrides);
            }"""
        )
        try:
            ok, detail = predicate(resolved), ""
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(engine, platform, f"storage: {name}", ok, detail)

    # A rebind has to reach the live listener, and survive a restart.
    reload_with(page, None)
    page.evaluate(
        """() => window.__shortcutsSmoke.store.useKeyboardShortcutsStore
            .getState().setBinding('toggleSidebar', 'primary', 'Mod+Alt+KeyW')"""
    )
    reset(page)
    page.keyboard.press(to_press("Mod+Alt+KeyW", platform))
    live = actions(page)
    page.goto(URL)
    page.wait_for_selector("#smoke-ready")
    reset(page)
    page.keyboard.press(to_press("Mod+Alt+KeyW", platform))
    check(
        engine,
        platform,
        "a rebind takes effect live and survives a restart",
        "toggleSidebar" in live and "toggleSidebar" in actions(page),
        f"live {live}, after restart {actions(page)}",
    )

    page.evaluate(
        "() => window.__shortcutsSmoke.store.useKeyboardShortcutsStore.getState().resetAll()"
    )
    check(
        engine,
        platform,
        "reset all leaves no storage residue",
        page.evaluate("localStorage.getItem('unsloth_keyboard_shortcuts')") is None,
    )


def check_selection_latch(page, engine: str, platform: str) -> None:
    archive = to_press("Mod+Alt+KeyE", platform)
    pin = to_press("Ctrl+Shift+KeyP" if platform == "macOS" else "Mod+Alt+KeyP", platform)

    def with_selection() -> None:
        page.click("#smoke-select")
        page.wait_for_function("document.getElementById('smoke-selection').textContent === '3'")
        reset(page)

    with_selection()
    page.keyboard.press(archive)
    page.wait_for_function("document.getElementById('smoke-selection').textContent === '0'")
    page.keyboard.press(archive)
    check(
        engine,
        platform,
        "a second fast press does not reach the open chat",
        actions(page) == ["archiveSelected", "archiveSuppressed"],
        str(actions(page)),
    )

    # otherwise land on the open chat, which was never selected.
    # A selection chord clears the selection, so an immediate second press would otherwise land on the open chat, which
    with_selection()
    page.keyboard.press(archive)
    page.wait_for_function("document.getElementById('smoke-selection').textContent === '0'")
    page.keyboard.press(pin)
    check(
        engine,
        platform,
        "a sibling chord inside the window still acts on the open chat",
        actions(page) == ["archiveSelected", "pinActive"],
        str(actions(page)),
    )

    # and swallowing it would trade one silent wrong action for another.
    # A different command straight after is not the repeat this guards against, and swallowing it would trade one
    # And a deliberate press afterwards still works, or the latch has traded one silent wrong action for a silent dead
    with_selection()
    page.keyboard.press(archive)
    page.wait_for_function("document.getElementById('smoke-selection').textContent === '0'")
    page.wait_for_timeout(900)
    page.keyboard.press(archive)
    check(
        engine,
        platform,
        "a deliberate press after the window still acts",
        actions(page) == ["archiveSelected", "archiveActive"],
        str(actions(page)),
    )

    reset(page)
    page.keyboard.press(archive)
    page.keyboard.press(archive)
    check(
        engine,
        platform,
        "with no selection nothing is swallowed",
        actions(page) == ["archiveActive", "archiveActive"],
        str(actions(page)),
    )

    # With no selection the latch is never stamped, so nothing is swallowed.
    # Deleting needs no latch: it has no open-chat branch to fall through to.
    reset(page)
    page.evaluate(
        """() => window.__shortcutsSmoke.store.useKeyboardShortcutsStore
            .getState().setBinding('deleteSelectedChats', 'primary', 'Mod+Alt+KeyY')"""
    )
    page.keyboard.press(to_press("Mod+Alt+KeyY", platform))
    check(
        engine,
        platform,
        "delete selected does nothing without a selection",
        actions(page) == [],
        str(actions(page)),
    )
    page.evaluate(
        "() => window.__shortcutsSmoke.store.useKeyboardShortcutsStore.getState().resetAll()"
    )


def check_unreads(page, engine: str, platform: str) -> None:
    chord = "Shift+Escape" if platform == "macOS" else to_press("Mod+Alt+Shift+KeyU", platform)
    for size, expected in (
        (0, "No unread chats"),
        (1, "Cleared 1 unread chat"),
        (3, "Cleared 3 unread chats"),
    ):
        reset(page)
        page.evaluate(
            """([size]) => window.__shortcutsSmoke.nav.setState({
                unreadThreadIds: new Set(Array.from({ length: size }, (_, i) => `t${i}`)),
            })""",
            [size],
        )
        page.keyboard.press(chord)
        left = page.evaluate("window.__shortcutsSmoke.nav.getState().unreadThreadIds.size")
        check(
            engine,
            platform,
            f"clearing {size} unread reports it and empties the set",
            details(page)[:1] == [expected] and left == 0,
            f"{details(page)} left={left}",
        )


def run_engine(pw, engine: str) -> None:
    launch = {"args": chromium_launch_args()} if engine == "chromium" else {}
    browser = getattr(pw, engine).launch(**launch)
    try:
        for platform, (nav_platform, agent) in PLATFORMS.items():
            context = browser.new_context(user_agent = agent)
            context.add_init_script(
                "Object.defineProperty(navigator, 'platform', "
                f"{{ get: () => {json.dumps(nav_platform)} }});"
            )
            page = context.new_page()
            for attempt in range(30):
                try:
                    page.goto(URL, wait_until = "domcontentloaded", timeout = 30000)
                    break
                except Exception:
                    if attempt == 29:
                        raise
                    time.sleep(2)
            page.wait_for_selector("#smoke-ready", timeout = 120000)
            try:
                check_defaults(page, engine, platform)
                check_dispatch(page, engine, platform)
                check_text_fields(page, engine, platform)
                check_bare_keys(page, engine, platform)
                check_repeat(page, engine, platform)
                check_altgr(page, engine, platform)
                check_foreign_binding(page, engine, platform)
                check_storage(page, engine, platform)
                check_selection_latch(page, engine, platform)
                check_unreads(page, engine, platform)
            except Exception as exc:  # noqa: BLE001
                check(
                    engine,
                    platform,
                    "the harness ran to completion",
                    False,
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            context.close()
    finally:
        browser.close()


def main() -> int:
    server = start_vite(PORT)
    try:
        with sync_playwright() as pw:
            for engine in ENGINES:
                run_engine(pw, engine)
    finally:
        stop_process(server)

    total = passed + len(failures)
    print(
        f"[keyboard-shortcuts] {passed}/{total} checks passed "
        f"across {', '.join(ENGINES)} on {', '.join(PLATFORMS)}"
    )
    for failure in failures:
        print(f"[keyboard-shortcuts] FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
