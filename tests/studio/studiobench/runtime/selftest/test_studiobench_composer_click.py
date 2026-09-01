# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`composer_click_ms` must be the click, and only the click.

The session opens every instrument before the window's body and closes them after it, and at
instrument level 1-3 those hooks stop a CPU profile, collect coverage and write and analyse a
trace. Timed around the `with` rather than inside it, this reading would grow with the instrument
level while still being labelled a `page.click` duration -- and it is the number the slow-click
warning is compared against.
"""

from __future__ import annotations

import contextlib
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.session import CellRunner  # noqa: E402

TEARDOWN_S = 0.4
CLICK_S = 0.1


class _Session:
    """A session whose instrument hooks are expensive, which is the whole point."""

    def __init__(self) -> None:
        self.opened: list[tuple] = []

    @contextlib.contextmanager
    def window(
        self,
        name,
        kind = "action",
    ):
        self.opened.append((name, kind))
        time.sleep(TEARDOWN_S)  # the `open` hooks
        try:
            yield types.SimpleNamespace(note = lambda *a: None)
        finally:
            time.sleep(TEARDOWN_S)  # the `close` hooks


class _Page:
    def wait_for_selector(
        self,
        selector,
        timeout = None,
    ):
        return None

    def click(
        self,
        selector,
        timeout = None,
    ):
        time.sleep(CLICK_S)

    def fill(self, selector, value):
        return None

    def wait_for_timeout(self, ms):
        return None

    def query_selector(self, selector):
        return types.SimpleNamespace(click = lambda: None)


def _run():
    runner = types.SimpleNamespace(
        session = _Session(),
        click_probe = False,
        log = lambda *a: None,
        _composer_click_ms = None,
        _click_attribution_result = None,
    )
    CellRunner._press_send(runner, _Page())
    return runner


def test_composer_click_ms_excludes_the_instrument_hooks():
    runner = _run()
    got = runner._composer_click_ms
    assert got is not None
    # The click is 100 ms and the hooks are 800 ms between them. Timed around the window this
    # came back near 900.
    assert CLICK_S * 1000 <= got < CLICK_S * 1000 + TEARDOWN_S * 1000


def test_the_click_is_filed_as_setup_and_not_as_an_action():
    """`action` would pool an 11 s driver stall into the cell's frame metrics. See
    `scoring/from_payload.UNSCORED_WINDOW_KINDS`."""
    runner = _run()
    assert runner.session.opened == [("setup:composer_click", "setup")]


# ── the probe's own output has to survive the payload schema ─────────────────────────────────


class _ProbePage(_Page):
    """Every in-page reading comes back 0, which is the case that matters: an unseeded rung has no
    code blocks, and `performance.now()` is coarsened to 100 us in a page that is not
    cross-origin isolated, so a sub-100 us operation genuinely reads 0."""

    def evaluate(
        self,
        expr,
        arg = None,
    ):
        return 0

    def query_selector(self, selector):
        return types.SimpleNamespace(
            click = lambda: None,
            bounding_box = lambda: {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
        )

    def dispatch_event(self, selector, event):
        return None

    def eval_on_selector(self, selector, expr):
        return None

    @property
    def mouse(self):
        return types.SimpleNamespace(click = lambda *a: None, move = lambda *a: None)


def test_the_probe_block_validates_against_the_payload_schema():
    from studiobench.scoring.schema import validate_payload

    runner = types.SimpleNamespace(session = _Session(), log = lambda *a: None)
    out = CellRunner._click_attribution(runner, _ProbePage(), "textarea")

    assert out["code_token_spans"] == 0
    cell = {
        "row_type": "cell",
        "cell_id": "r1K.A0.rep0",
        "target_tokens": 1000,
        "completed": True,
        "click_attribution": out,
    }
    validate_payload(
        {
            "schema": "studiobench/payload/1",
            "source": "recorder_rows",
            "complete": True,
            "truncated_records": 0,
            "record_counts": {"cells": 1},
            "header": {},
            "selfcheck": [],
            "windows": [],
            "actions": [],
            "cells": [cell],
            "samples": [],
            "surfaces": [],
            "crashes": [],
            "arms": [],
            "unknown_rows": [],
            "footer": None,
            "excluded_cells": [],
        }
    )
