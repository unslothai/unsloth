# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Every appearance setting is measured by the long-chat gate or allowlisted with a reason (no browser)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _appearance_coverage as cov  # noqa: E402
import _area_matrix as am  # noqa: E402

MODES = {m["name"] for m in am.app_modes()}


def test_every_appearance_setting_is_measured_or_allowlisted():
    problems = cov.uncovered(MODES)
    assert not problems, "\n".join(problems)


def test_the_enumeration_reads_the_real_stores():
    custom = cov.CUSTOM_TS.read_text(encoding = "utf-8")
    keys = cov.object_keys(custom, "DEFAULT_CUSTOMIZATION")
    assert {"contrast", "elementOutline", "reduceMotion", "chatWidth"} <= set(keys)
    assert set(cov.enum_settings(custom)["chatWidth"]) == {"standard", "wide", "full"}
    assert {"blur", "effect", "scope"} <= set(
        cov.object_keys(cov.MODEL_TS.read_text(encoding = "utf-8"), "DEFAULT_ADJUSTMENTS")
    )
    flags = cov.css_root_flags()
    assert "data-glass" in flags and "data-chat-wallpaper-workspace-ready" in flags
    ids, shared = cov.palettes()
    assert cov.PALETTE_MEASURED in ids and len(shared) >= 10


def test_each_measured_setting_is_a_gate_mode_with_an_applied_check():
    by_name = {m["name"]: m for m in am.app_modes()}
    for name in (
        "solid+custom",
        f"{am.GLASS}+custom",
        "solid+wallpaper",
        f"{am.GLASS}+wallpaper",
        "solid+palette",
        f"{am.GLASS}+strength-max",
    ):
        mode = by_name[name]
        assert am.is_variant(mode), name
        assert mode.get("expect"), f"{name} must check that its setting reached <html>"
        assert by_name[mode["base"]]["base"] in (None, "default")


def test_a_new_costly_root_flag_or_setting_fails(monkeypatch):
    css = "html[data-new-sparkle] .thread { backdrop-filter: blur(30px); }\nhtml[data-quiet] { color: red; }"
    flags = cov.root_flags(css)
    assert "data-new-sparkle" in flags and "data-quiet" not in flags
    nested = "html[data-glow] { .x { & .y { filter: blur(2px); } } }"
    assert "data-glow" in cov.root_flags(nested)
    monkeypatch.setattr(
        cov,
        "css_root_flags",
        lambda: {"data-new-sparkle": ["index.css: html[data-new-sparkle] .thread"]},
    )
    assert any("data-new-sparkle" in p for p in cov.uncovered(MODES))
    monkeypatch.setattr(
        cov, "CUSTOM_ALLOW", {k: v for k, v in cov.CUSTOM_ALLOW.items() if k != "colors"}
    )
    assert any("'colors'" in p for p in cov.uncovered(MODES))


def test_new_modes_run_only_at_the_short_length_on_the_variant_areas():
    jobs = am.plan(am.app_modes(), long_modes = f"default,{am.GLASS}", full = False)
    long_names = {m["name"] for turns, m, _ in jobs if turns == am.LONG}
    assert long_names == {"default", am.GLASS}
    for turns, m, areas in jobs:
        if m["name"].endswith(("+custom", "+wallpaper", "+palette", "+strength-max")):
            assert turns == am.SHORT and set(areas) <= set(am.with_deps(am.VARIANT_AREAS))
