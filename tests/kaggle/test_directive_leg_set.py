# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pins the leg set as spec properties, so a leg quietly losing its export flag fails a test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github" / "scripts" / "kaggle_t4_ci"))

import legs  # noqa: E402

DIRECTIVE = {
    "default": "unsloth/Qwen3-0.6B",
    "latest_compile": "unsloth/gemma-4-E2B-it",
    "vision_fla_compile": "unsloth/Qwen3.5-2B",
    "gptoss": None,  # the leg has its own loader and does not take --model
}


def test_every_directive_leg_exists():
    for name in DIRECTIVE:
        assert name in legs.LEGS, f"{name} is not defined at all"


def test_every_directive_leg_is_pointed_at_the_model_it_was_specified_for():
    for name, model in DIRECTIVE.items():
        if model is None:
            continue
        args = legs.LEGS[name].args
        assert "--model" in args, f"{name} names no model"
        assert (
            args[args.index("--model") + 1] == model
        ), f"{name} trains {args[args.index('--model') + 1]!r}, not {model!r}"


# Only the two cheapest checkpoints carry the export; latest_compile and gptoss were dropped
# on cost (~660s) since they re-run llama.cpp without asking anything new.
EXPORTING = ("default", "vision_fla_compile")
NOT_EXPORTING = ("latest_compile", "gptoss")


def test_the_cheap_legs_export_a_gguf_and_run_it():
    """The flag is what turns on the export AND the inference against the
    exported file, so a leg without it makes neither claim. At least one leg
    has to keep it or the directive's llama.cpp item is uncovered."""
    missing = [n for n in EXPORTING if "--export-gguf" not in legs.LEGS[n].args]
    assert missing == [], f"these legs never export a GGUF: {missing}"


def test_the_expensive_exports_stay_off():
    """Wall-clock removals stay off: a re-added export slows the suite without making it go red."""
    back = [n for n in NOT_EXPORTING if "--export-gguf" in legs.LEGS[n].args]
    assert back == [], (
        f"these legs export again: {back}. They were dropped at 310.8s and "
        f"348.1s for a claim `default` makes in 40.6s; if a checkpoint-specific "
        f"conversion is genuinely in doubt, dispatch with the flag rather than "
        f"putting it back on every PR"
    )


def test_at_least_one_wired_leg_still_exports():
    """The rule above must not be satisfiable by removing every export. Read
    off KERNELS, not off the list here, so dropping the exporting leg from the
    wired set fails too."""
    wired = {name for kernel in legs.KERNELS for name in kernel}
    exporting = [n for n in wired if "--export-gguf" in legs.LEGS[n].args]
    assert exporting, (
        "no leg in the wired set exports a GGUF, so nothing covers the "
        "prebuilt llama.cpp binaries or inference on an exported file"
    )


def test_the_grpo_leg_keeps_the_settings_it_was_measured_with():
    """0.95 utilisation and sleep mode were asked for and both were measured
    passing on Kaggle AND Colab; the flashinfer uninstall stays because
    removing it broke the leg. None of the three is a default."""
    grpo = legs.LEGS["grpo"]
    args = grpo.args
    assert "--gpu-memory-utilization" in args
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
    assert grpo.env.get("UNSLOTH_VLLM_STANDBY") == "1", "sleep mode"
    assert grpo.uninstall, "the flashinfer uninstall is what makes the leg run at all"


def test_the_grpo_leg_cannot_share_a_card():
    """GRPO measured at 13.39-13.40 GB of 14.56, so its vram_gb must block any co-tenant."""
    assert legs.LEGS["grpo"].vram_gb >= 13.0


def test_no_directive_leg_is_scheduled_beside_a_co_tenant_it_cannot_fit():
    """A vram_gb over the budget means exclusive; two legs that each need most
    of a card must not both claim to fit beside something."""
    budget = 13.0
    for name in DIRECTIVE:
        vram = legs.LEGS[name].vram_gb
        assert vram > 0, f"{name} declares no VRAM at all"
        assert vram <= 14.5, f"{name} declares {vram}, which is more than a T4 has"
        # This rule only applies above half the budget; the UNWIRED note guards placeholders below it.
        if vram > budget / 2:
            # Round numbers have always been placeholders, never measurements.
            assert vram != round(vram), (
                f"{name} declares {vram}, a round number over half the budget, "
                f"which is what every placeholder in this file's history looked "
                f"like. Measure it on one card or say why."
            )


def test_a_wired_leg_never_carries_a_round_placeholder_over_a_gigabyte():
    """Wired legs with a round VRAM figure of 1 GB or more are flagged; measured figures are not round."""
    offenders = []
    for name in legs.KERNELS[0]:
        vram = legs.LEGS[name].vram_gb
        if vram >= 1.0 and vram == round(vram):
            offenders.append(f"{name}={vram}")
    assert offenders == [], (
        f"these WIRED legs schedule against a round whole-gigabyte number, "
        f"which is what every placeholder here has looked like: {offenders}"
    )
