# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The update-check delay override is one name shared by two languages.

The banner layout suite boots a fresh page per case and, before this, waited out the
app's 5s update-check timer on each of them: 33 boots, over two and a half minutes of a
five minute step, spent waiting for a delay whose purpose is to keep a request off the
critical path at launch. That is not a property the layout suite tests.

The override is a global set by the suite's init script and read by the hook at mount.
Nothing connects the two but the spelling, and a typo on either side fails silently in
the worst possible way: the timer simply stays at 5s and the step goes back to being slow
while every assertion still passes. So the spelling is asserted here.

Deliberately NOT asserted: that the delay is short. A value is a judgement, and the
suite can raise it with an env var when a browser needs longer. What must hold is that
the two sides mean the same thing, and that production is untouched when nobody sets it.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "studio" / "frontend" / "src" / "hooks" / "use-web-update-check.ts"
SUITE = REPO / "tests" / "studio" / "playwright_update_banner_layout.py"


def test_both_sides_spell_the_override_the_same_way():
    hook = HOOK.read_text(encoding = "utf-8")
    suite = SUITE.read_text(encoding = "utf-8")

    in_hook = re.search(r'E2E_DELAY_GLOBAL\s*=\s*"([^"]+)"', hook)
    assert in_hook, f"{HOOK.name} no longer names the override global"
    in_suite = re.search(r'E2E_DELAY_GLOBAL\s*=\s*"([^"]+)"', suite)
    assert in_suite, f"{SUITE.name} no longer names the override global"

    assert in_hook.group(1) == in_suite.group(1), (
        f"{HOOK.name} reads {in_hook.group(1)!r} and {SUITE.name} sets "
        f"{in_suite.group(1)!r}. Nothing else connects them, so this drift does not fail "
        f"a single assertion: the timer stays at 5s and the step silently costs the "
        f"minutes this override was added to give back."
    )


def test_the_suite_actually_sets_it_before_the_app_runs():
    """An init script, not a page.evaluate after navigation.

    Setting it after load would be too late by definition: the hook reads it when the
    effect mounts, which has already happened.
    """
    suite = SUITE.read_text(encoding = "utf-8")
    assert "add_init_script" in suite, (
        f"{SUITE.name} no longer installs an init script, so the override cannot be in "
        f"place before the hook mounts and reads it"
    )
    seed = suite[suite.index("seed_js = (") : suite.index("if PLAYWRIGHT_BROWSER")]
    assert "E2E_DELAY_GLOBAL" in seed, (
        f"the override moved out of seed_js in {SUITE.name}. seed_js is what is handed to "
        f"add_init_script; anywhere else runs after the app has already armed the timer."
    )


def test_production_keeps_the_five_second_delay():
    """The override is absent in a real browser, and the constant is what remains.

    A default of anything but 5000, or a read at module load rather than at mount, would
    change what users get. This is the assertion that keeps the optimisation confined to
    CI.
    """
    hook = HOOK.read_text(encoding = "utf-8")
    assert re.search(r"const WEB_UPDATE_CHECK_DELAY_MS = 5000;", hook), (
        "the production update-check delay is no longer 5000ms. Shortening the CI wait "
        "must not shorten the real one: that delay exists to keep the request off the "
        "critical path while the app is starting."
    )
    assert 'typeof window === "undefined"' in hook, (
        "the override no longer guards against a server-side render, where there is no "
        "window to read it from"
    )
