# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol canary for unsloth-zoo save_pretrained_merged guards
(unslothai/unsloth-zoo#647 / unslothai/unsloth#5410). Skips until #647
lands, then becomes a hard gate. CPU-only static fetch."""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text


ZOO_TAG = "main"


def _fetch_saving_utils() -> str:
    src = fetch_text("unslothai/unsloth-zoo", ZOO_TAG, "unsloth_zoo/saving_utils.py")
    if src is None:
        pytest.skip("unsloth_zoo/saving_utils.py not fetchable")
    return src


def _fetch_merge_tests() -> str:
    src = fetch_text(
        "unslothai/unsloth-zoo",
        ZOO_TAG,
        "tests/test_unsloth_zoo_lora_merge.py",
    )
    if src is None:
        pytest.skip("tests/test_unsloth_zoo_lora_merge.py not fetchable")
    return src


def _skip_until_pr_647_lands(src: str) -> None:
    if not any(
        m in src
        for m in (
            "_MOE_MERGE_STATE",
            "_detect_moe_lora_layout",
            "_resolve_num_experts_from_lora_stats",
        )
    ):
        pytest.skip(
            "unslothai/unsloth-zoo#647 has not yet merged into main; "
            "tests auto-promote to hard gates once it lands."
        )


def test_zoo_saving_utils_has_moe_merge_state():
    src = _fetch_saving_utils()
    _skip_until_pr_647_lands(src)
    for sym in (
        "_MOE_MERGE_STATE",
        "_reset_moe_merge_state",
        "_record_moe_merge_fallback",
    ):
        assert sym in src, f"{sym} missing from saving_utils.py (issue #5410 guard)."
    # zoo#647 wraps the fallback guard's message onto a second line;
    # allow the regex to span newlines via re.DOTALL.
    assert re.search(
        r"raise\s+RuntimeError\b.*?MoE", src, re.IGNORECASE | re.DOTALL
    ), "no `raise RuntimeError(...MoE...)`; post-loop guard weakened."


def test_zoo_saving_utils_has_layout_detector():
    src = _fetch_saving_utils()
    _skip_until_pr_647_lands(src)
    assert (
        "_detect_moe_lora_layout" in src
    ), "_detect_moe_lora_layout removed (issue #5410)."
    assert (
        '"swapped"' in src and '"standard"' in src
    ), "one of the layout labels removed."


def test_zoo_saving_utils_has_num_experts_resolver():
    src = _fetch_saving_utils()
    _skip_until_pr_647_lands(src)
    assert "_resolve_num_experts_from_lora_stats" in src, "resolver removed (#5410)."
    assert re.search(
        r"for\s+_\s+in\s+range\s*\(\s*\d+\s*\)", src
    ), "resolver walk no longer bounded by `for _ in range(N):`."


def test_zoo_saving_utils_writes_generation_config():
    src = _fetch_saving_utils()
    _skip_until_pr_647_lands(src)
    # zoo#647 binds the generation_config attr to a local var
    # (`gen_cfg = getattr(model, "generation_config", ...); ...
    # gen_cfg.save_pretrained(save_directory)`) so an exact
    # `generation_config.save_pretrained(` substring no longer
    # matches. Anchor on the conceptual operation: a `generation_config`
    # mention plus a `.save_pretrained(` call nearby, which is what
    # the canary actually cares about.
    assert re.search(
        r"generation_config[\s\S]{0,400}?\.save_pretrained\s*\(", src
    ), "generation_config.json no longer saved (#5410)."


def test_zoo_lora_merge_tests_have_standard_layout_coverage():
    src = _fetch_merge_tests()
    if "test_merge_moe_gate_expert_standard_layout" not in src:
        pytest.skip("unslothai/unsloth-zoo#647 not yet merged; coverage appears later.")
    for name in (
        "test_merge_moe_gate_expert_standard_layout",
        "test_merge_moe_up_expert_standard_layout",
        "test_merge_moe_down_proj_expert_standard_layout",
        "test_detect_moe_lora_layout_classifies_both_conventions",
        "test_moe_merge_fallback_counter_records_bad_layout",
        "test_resolve_num_experts_walks_base_layer_chain",
    ):
        assert name in src, f"regression test `{name}` removed."


def test_unsloth_save_pretrained_merged_entry_point_exists():
    import pathlib

    save_py = pathlib.Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
    if not save_py.is_file():
        pytest.skip(f"{save_py} not present")
    text = save_py.read_text(encoding = "utf-8", errors = "replace")
    assert "save_pretrained_merged" in text, "entry point removed from unsloth/save.py."
    assert (
        "merge_and_overwrite_lora" in text
    ), "no dispatch into unsloth_zoo merge; #647 bypassed."
