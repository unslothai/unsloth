# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The leg set the rebuild is specified against, encoded so it cannot drift.

Four legs, each covering a distinct surface, plus GRPO which is deliberately
not per-PR. These are properties of the SPEC rather than of the code, and they
are written down here because the alternative is that they live only in a
conversation: a leg quietly losing its export flag, or being pointed at a
different model, is invisible in a diff that touches one line of a tuple.

The one that matters most is the export rule. "Export to GGUF / llama.cpp Q8_0
and run inference on the result" applies to EVERY leg, and Latest_compile was
missing it for the whole rebuild without anything noticing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github" / "scripts" / "kaggle_t4_ci"))

import legs  # noqa: E402

# leg name -> the model it is specified to exercise.
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


# Which legs carry the export, and why it is these two rather than all four.
# The claim is that the PREBUILT llama.cpp binaries convert a trained adapter
# and that the result runs -- `run_failures` in gguf_export.py rules on the
# second half. Two legs make it, on the two cheapest checkpoints:
#
#   default             609.8 MB Q8_0            40.6s
#   vision_fla_compile  1980.5 MB Q8_0 + mmproj  99.3s
#
# The two that no longer carry it were measured and dropped on cost:
#
#   latest_compile      4725.1 MB Q8_0          310.8s
#   gptoss             13153.7 MB MXFP4         348.1s   (never a Q8_0 at all)
#
# 659s for a third and fourth conversion that re-run llama.cpp rather than ask
# a new question.
EXPORTING = ("default", "vision_fla_compile")
NOT_EXPORTING = ("latest_compile", "gptoss")


def test_the_cheap_legs_export_a_gguf_and_run_it():
    """The flag is what turns on the export AND the inference against the
    exported file, so a leg without it makes neither claim. At least one leg
    has to keep it or the directive's llama.cpp item is uncovered."""
    missing = [n for n in EXPORTING if "--export-gguf" not in legs.LEGS[n].args]
    assert missing == [], f"these legs never export a GGUF: {missing}"


def test_the_expensive_exports_stay_off():
    """A removal made for wall-clock reasons has to be visible, or it comes
    back by accident on the next edit and nobody notices 659 seconds.

    Stated as a rule rather than a comment because the cost is invisible in a
    green run: an export that reappears makes the suite slower and no redder.
    """
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
    """Measured at 13.39-13.40 GB of 14.56 across nine sessions. The directive
    says the GRPO run must be standalone on one GPU, and the way that is
    expressed here is a vram_gb the admission scheduler cannot fit a co-tenant
    beside."""
    assert legs.LEGS["grpo"].vram_gb >= 13.0


def test_no_directive_leg_is_scheduled_beside_a_co_tenant_it_cannot_fit():
    """A vram_gb over the budget means exclusive; two legs that each need most
    of a card must not both claim to fit beside something."""
    budget = 13.0
    for name in DIRECTIVE:
        vram = legs.LEGS[name].vram_gb
        assert vram > 0, f"{name} declares no VRAM at all"
        assert vram <= 14.5, f"{name} declares {vram}, which is more than a T4 has"
        # The rule only bites above half the budget, and that is a real limit
        # worth stating: Vision_FLA_compile sat at a round 4.0 placeholder
        # under this threshold and nothing here objected. The measured figure
        # is 2.84; what stops a placeholder shipping is the UNWIRED note plus
        # the rule below, not this one.
        if vram > budget / 2:
            # Not a failure, but it must be a MEASURED number rather than a
            # round placeholder: every placeholder in this file's history was
            # a round number and every measured one was not.
            assert vram != round(vram), (
                f"{name} declares {vram}, a round number over half the budget, "
                f"which is what every placeholder in this file's history looked "
                f"like. Measure it on one card or say why."
            )


def test_a_wired_leg_never_carries_a_round_placeholder_over_a_gigabyte():
    """Every placeholder in this file's history was a round number and every
    measured figure was not: 6.0 against a measured 12.73, 4.0 against 2.84.
    A leg small enough to co-tenant freely (0.7) is not the risk; a leg that
    claims whole gigabytes on a 14.56 GB card is.

    UNWIRED legs are exempt by construction -- they are not scheduled, and the
    note is where the missing measurement is recorded.
    """
    offenders = []
    for name in legs.KERNELS[0]:
        vram = legs.LEGS[name].vram_gb
        if vram >= 1.0 and vram == round(vram):
            offenders.append(f"{name}={vram}")
    assert offenders == [], (
        f"these WIRED legs schedule against a round whole-gigabyte number, "
        f"which is what every placeholder here has looked like: {offenders}"
    )
