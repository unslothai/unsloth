# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The permission step's reload must settle on the pill, never on "networkidle".

The step reloads with a page.route on /api/chat/settings in place. Once /api reads went out as
Cache-Control: no-store (#12148), Playwright's `wait_for_load_state("networkidle", timeout = 30_000)`
after that reload stopped returning at all: the 30 s timeout never fired and the step hung until
the 180 s watchdog ended the job, 3 runs out of 3 locally and on main at 1dddc1437. The same step
passes 3 out of 3 with the header in place once the reload settles on the pill instead, so this
pins the helper to the bounded wait.
"""

from __future__ import annotations

import ast
from pathlib import Path


DRIVER = (Path(__file__).resolve().parent / "playwright_chat_ui.py").read_text(encoding = "utf-8")
STEP = "exercise_permission_mode_controls"
HELPER = "reload_and_wait_for_pill"


def _helper() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(DRIVER)):
        if isinstance(node, ast.FunctionDef) and node.name == STEP:
            for inner in ast.walk(node):
                if isinstance(inner, ast.FunctionDef) and inner.name == HELPER:
                    return inner
    raise AssertionError(f"{HELPER} is no longer defined inside {STEP}")


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_the_permission_reload_never_waits_for_networkidle():
    helper = _helper()
    idle = [
        call
        for call in _calls(helper, "wait_for_load_state")
        if any(isinstance(arg, ast.Constant) and arg.value == "networkidle" for arg in call.args)
    ]
    assert not idle, (
        f"{HELPER} waits for networkidle again; with the step's page.route and no-store /api reads "
        "that wait never returns and the watchdog kills the job"
    )
    for call in _calls(helper, "reload"):
        for keyword in call.keywords:
            if keyword.arg == "wait_until":
                assert getattr(keyword.value, "value", None) != "networkidle", ast.unparse(call)


def test_the_permission_reload_settles_on_a_bounded_pill_wait():
    helper = _helper()
    reloads = _calls(helper, "reload")
    visible = _calls(helper, "to_be_visible")
    assert reloads and visible, ast.unparse(helper)
    assert reloads[0].lineno < visible[0].lineno, "the pill wait must follow the reload"
    timeout = next((kw.value for kw in visible[0].keywords if kw.arg == "timeout"), None)
    assert isinstance(timeout, ast.Constant) and 0 < timeout.value <= 60_000, ast.unparse(
        visible[0]
    )
