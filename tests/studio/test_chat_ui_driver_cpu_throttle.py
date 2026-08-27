# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`STUDIO_UI_CPU_THROTTLE` is delivered over CDP, so it is Chromium only.

The driver documents firefox and webkit as supported browsers, so the pairing
is reachable from the documented environment alone, and `new_cdp_session` on
either raises a Playwright error about CDP that never names the option that
caused it. The driver refuses the combination before launch instead, which is
what it already does for `STUDIO_PLAYWRIGHT_CHANNEL`.

Comments are stripped before every match: this file's own prose says
"chromium", and a guard satisfied by its neighbouring comment is a guard that
passes while the code is gone.
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


def test_the_refusal_precedes_the_cdp_session():
    """Order is the whole point: after the launch it is not a refusal."""
    guard = CODE.index("CPU_THROTTLE > 1 and PLAYWRIGHT_BROWSER")
    cdp = CODE.index("new_cdp_session")
    assert guard < cdp, (
        "the CDP session is opened before the unsupported-browser check, so "
        "firefox and webkit still raise"
    )


def test_the_cdp_call_is_the_only_place_the_throttle_is_applied():
    assert CODE.count("new_cdp_session") == 1, (
        "a second CDP session would need its own refusal"
    )


def test_the_driver_still_parses():
    """The regexes above are read against real source, not a fixture."""
    ast.parse(DRIVER)
