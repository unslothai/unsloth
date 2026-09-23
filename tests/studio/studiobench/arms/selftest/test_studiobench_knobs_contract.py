# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The contract between `knobs.py` and `knobs.js`, pinned so it cannot drift silently.

Every arm's manifest names two strings that live on the other side of a language boundary: a
potency counter and, for an EQUIVALENT arm, the diff keys it is allowed to produce. Neither is
checked by any compiler, and both fail in the worst possible direction.

A misspelled potency counter reads as `NOT RUN` for an arm that fired perfectly. A mismatched diff
key reads as `VOIDED` for an arm whose output never changed. Both are conservative, so neither
produces a wrong NUMBER, and that is exactly why they can survive for weeks: the report simply
says less than it could, forever, and nobody can tell the difference between a knob that is
broken and a mechanism that is not there.

This was not hypothetical. Arm B declared its allowed diff as `style` while the browser side
emits `attr:style` (namespaced so that an attribute called `text` cannot collide with the
structural text key). B would have been voided on every run of every batch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.arms.knobs import (  # noqa: E402
    KNOBS_JS_PATH,
    PREBOOT_ARM_IDS,
    RUNTIME_ARMS,
    init_scripts_for,
    load_knobs_js,
)


@pytest.fixture(scope = "module")
def knobs_js() -> str:
    return load_knobs_js()


def test_the_injected_file_exists_and_is_not_a_stub(knobs_js: str):
    assert KNOBS_JS_PATH.is_file()
    assert len(knobs_js) > 10_000


def test_every_potency_counter_name_exists_on_the_browser_side(knobs_js: str):
    """A counter the browser never writes reads as NOT RUN for an arm that fired."""

    for arm in RUNTIME_ARMS:
        assert arm.potency.name in knobs_js, (
            f"arm {arm.arm_id} declares potency counter {arm.potency.name!r}, which does not "
            "appear in knobs.js. That arm will read NOT RUN on every run"
        )


def test_every_declared_diff_key_exists_on_the_browser_side(knobs_js: str):
    """A key the browser never emits voids the arm on every run."""

    for arm in RUNTIME_ARMS:
        if arm.declared_diff is None:
            continue
        for key in arm.declared_diff.keys:
            assert key in knobs_js, (
                f"arm {arm.arm_id} declares diff key {key!r}, which knobs.js never emits. That "
                "arm will be VOIDED on every run"
            )


def test_the_public_api_the_python_side_calls_is_present(knobs_js: str):
    for symbol in (
        "__sbArms",
        "__sbArmConfig",
        "apply",
        "revert",
        "potency",
        "counts",
        "digest",
        "diffKeys",
        "available",
        "unavailable",
    ):
        assert symbol in knobs_js, f"knobs.js does not expose {symbol}"


def test_the_selectors_the_arms_depend_on_are_the_shipped_ones(knobs_js: str):
    for selector in (
        "aui-stream-viewport",
        "data-message-id",
        'data-status="running"',
        'data-streamdown="code-block"',
        "--aui-scroll-stabilizer",
        "aria-expanded",
    ):
        assert selector in knobs_js, f"knobs.js does not reference {selector}"


def test_every_arm_id_is_reachable_from_the_browser_side(knobs_js: str):
    for arm in RUNTIME_ARMS:
        assert f'"{arm.arm_id}"' in knobs_js or f"'{arm.arm_id}'" in knobs_js


def test_init_scripts_are_ordered_config_then_knobs():
    scripts = init_scripts_for(["A", "D"])
    assert scripts[0].startswith("window.__sbArmConfig")
    assert "__sbArms" in scripts[1]
    assert len(scripts) == 2


def test_preboot_arms_are_the_ones_that_patch_a_prototype(knobs_js: str):
    """D, E and F patch something the app captures during boot; the rest do not."""

    assert PREBOOT_ARM_IDS == frozenset({"D", "E", "F"})
    for needle in (
        "MutationObserver.prototype",
        "CSSStyleDeclaration.prototype",
        "MessageChannel",
    ):
        assert needle in knobs_js


@pytest.mark.skipif(shutil.which("node") is None, reason = "node is not installed")
def test_knobs_js_parses():
    result = subprocess.run(
        ["node", "--check", str(KNOBS_JS_PATH)],
        capture_output = True,
        text = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr
