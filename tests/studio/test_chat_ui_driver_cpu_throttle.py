# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`STUDIO_UI_CPU_THROTTLE` is delivered over CDP, so it is Chromium only.

firefox and webkit are documented as supported, so the pairing is reachable
from the environment alone and `new_cdp_session` raises a CDP error that never
names the option. The driver refuses it before launch, as it already does for
`STUDIO_PLAYWRIGHT_CHANNEL`.

Comments are stripped before every match: this file's own prose says
"chromium", and a guard its neighbouring comment satisfies passes while the
code is gone.
"""

import ast
import re
from pathlib import Path


DRIVER_PATH = Path(__file__).resolve().parent / "playwright_chat_ui.py"
DRIVER = DRIVER_PATH.read_text(encoding = "utf-8")


def _code_only(source: str) -> str:
    """The source with comments and string literals removed."""
    stripped = re.sub(r"(?m)#[^\n]*", "", source)
    return re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", stripped)


CODE = _code_only(DRIVER)


def test_the_throttle_is_read_once_into_a_module_constant():
    """Read at the launch site instead and the refusal cannot see it."""
    assert "STUDIO_UI_CPU_THROTTLE" in CODE
    assert re.search(r"(?m)^CPU_THROTTLE\s*=", CODE), (
        "the throttle is no longer a module constant, so the pre-launch "
        "refusal below cannot be reading the same value the driver applies"
    )


def test_a_non_chromium_browser_refuses_the_throttle():
    refusal = re.search(
        r"if\s+CPU_THROTTLE\s*>\s*1\s+and\s+PLAYWRIGHT_BROWSER\s*!=\s*"
        r"[\"']chromium[\"']\s*:(?P<body>(?:\n[ \t]+.*)+)",
        CODE,
    )
    assert refusal, (
        "nothing refuses STUDIO_UI_CPU_THROTTLE on firefox/webkit, so the run "
        "aborts inside new_cdp_session with a message about CDP"
    )
    assert "fail(" in refusal.group("body"), (
        "the combination is detected and not refused; a warning would let the "
        "run continue unthrottled and report a pass that proves nothing"
    )


def test_the_refusal_precedes_the_first_page():
    """Order is the point: after a page exists it is not a refusal.

    Read against the launch function, not the file: the CDP call lives in a
    helper near the top, so file order says nothing about what runs first.
    """
    launch = CODE[CODE.index("browser_type = getattr(p, PLAYWRIGHT_BROWSER)") :]
    guard = launch.index("CPU_THROTTLE > 1 and PLAYWRIGHT_BROWSER")
    first_page = launch.index("new_throttled_page(ctx)")
    assert guard < first_page, (
        "a page is created before the unsupported-browser check, so firefox "
        "and webkit reach the CDP call and raise"
    )


def test_every_page_the_driver_opens_goes_through_one_factory():
    """A page opened directly is a page that is never throttled.

    The relogin path's own fresh page ran at full speed, so the steps after it
    passed under a driver reporting itself throttled -- a false pass in
    precisely the slow case the option exists to create.
    """
    factory = re.search(
        r"def new_throttled_page\([^)]*\):(?P<body>(?:\n(?:[ \t].*)?)+?)(?=\ndef |\nclass |\Z)",
        CODE,
    )
    assert factory, "no new_throttled_page factory"
    body = factory.group("body")
    assert "ctx.new_page()" in body and "apply_cpu_throttle" in body
    assert "set_default_timeout(60_000)" in body, (
        "the factory drops the 60s default the direct call sites carried, so "
        "the macos-14 runners go back to timing out on ordinary renders"
    )
    outside = CODE.replace(body, "", 1)
    assert "new_page()" not in outside, (
        "a page is still opened outside new_throttled_page, so it runs "
        "unthrottled for every step that follows it"
    )


def test_the_cdp_call_has_exactly_one_owner():
    assert CODE.count("new_cdp_session") == 1, (
        "a second CDP session would sit outside apply_cpu_throttle, so the "
        "refusal and the recovery path would both miss it"
    )
    owner = re.search(
        r"def apply_cpu_throttle\([^)]*\):(?P<body>(?:\n(?:[ \t].*)?)+?)(?=\ndef |\nclass |\Z)",
        CODE,
    )
    assert owner and "new_cdp_session" in owner.group(
        "body"
    ), "the throttle is applied somewhere other than apply_cpu_throttle"


def test_a_replacement_page_is_throttled_again():
    """A page replaced after a renderer death must be throttled again."""
    assert "recover_or_replace_page as _robust_recover_or_replace_page" in CODE, (
        "the shared helper is imported under its own name, so the call sites "
        "bypass the wrapper that re-throttles a replacement page"
    )
    wrapper = re.search(
        r"def recover_or_replace_page\([^)]*\):(?P<body>(?:\n(?:[ \t].*)?)+?)"
        r"(?=\ndef |\nclass |\Z)",
        CODE,
    )
    assert wrapper, "no local recover_or_replace_page wrapper"
    body = wrapper.group("body")
    assert "_robust_recover_or_replace_page" in body
    assert "apply_cpu_throttle" in body, (
        "the wrapper does not re-apply the throttle, so a replaced page runs "
        "unthrottled for the rest of the driver"
    )
    assert "is not page" in body, (
        "the throttle is re-applied unconditionally; only a REPLACEMENT page "
        "needs it, and a fresh CDP session per recovery is not free"
    )
    assert "_robust_recover_or_replace_page(" not in CODE.replace(body, "", 1).replace(
        "recover_or_replace_page as _robust_recover_or_replace_page", "", 1
    ), "a call site reaches the shared helper directly, skipping the wrapper"


def test_the_driver_still_parses():
    """The regexes above are read against real source, not a fixture."""
    ast.parse(DRIVER)
