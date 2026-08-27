# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Guard the model-config UI smoke's first-boot password contract."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKFLOW_YML = REPO / ".github/workflows/studio-ui-smoke.yml"
PLAYWRIGHT_SCRIPT = REPO / "tests/studio/playwright_model_config.py"


def _step_block(source: str, name: str) -> str:
    start = source.find(f"- name: {name}")
    assert start != -1, f"workflow step {name!r} not found"
    end = source.find("- name:", start + 1)
    return source[start : end if end != -1 else None]


def test_model_config_smoke_passes_the_bootstrap_password():
    workflow = WORKFLOW_YML.read_text(encoding = "utf-8")
    password_step = _step_block(workflow, "Pass bootstrap pw for model-config test")
    assert "cat ~/.unsloth/studio/auth/.bootstrap_password" in password_step
    assert "STUDIO_MODELCFG_OLD_PW" in password_step

    drive_step = _step_block(workflow, "Drive model-picker per-model-config with Playwright")
    assert "STUDIO_OLD_PW: ${{ env.STUDIO_MODELCFG_OLD_PW }}" in drive_step


def test_model_config_playwright_fills_current_password_on_first_boot():
    script = PLAYWRIGHT_SCRIPT.read_text(encoding = "utf-8")
    assert 'BOOTSTRAP_PW = os.environ.get("STUDIO_OLD_PW")' in script
    assert "STUDIO_OLD_PW is required for the first-boot change-password flow" in script
    assert "fill_bootstrap_current_password(page, BOOTSTRAP_PW)" in script


def test_every_first_boot_driver_asserts_the_seed_never_reached_the_browser():
    """The four drivers must go through the shared helper, not a bare fill.

    An `if page.locator("#current-password").count():` guard passes whether or not
    the page still injects the seed, so it cannot detect the injection coming back.
    `fill_bootstrap_current_password` asserts `window.__UNSLOTH_BOOTSTRAP__` is absent
    and the field starts empty before it types, which is the invariant worth locking.
    """
    helper = (REPO / "tests/studio/_playwright_robust.py").read_text(encoding = "utf-8")
    assert "def fill_bootstrap_current_password(" in helper
    assert "window.__UNSLOTH_BOOTSTRAP__ ?? null" in helper

    for name in (
        "playwright_chat_ui.py",
        "playwright_extra_ui.py",
        "playwright_chat_ime_i18n.py",
        "playwright_model_config.py",
    ):
        src = (REPO / "tests/studio" / name).read_text(encoding = "utf-8")
        assert (
            "fill_bootstrap_current_password" in src
        ), f"{name} does not use the shared first-boot helper"
        assert "#current-password" not in src, (
            f"{name} still locates #current-password directly; route it through "
            "fill_bootstrap_current_password so the no-injection assertion runs"
        )
