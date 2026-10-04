# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pins the logit transforms the GRPO path applies, per model family.

The GRPO loss recomputes log-probs from hidden states, so it has to reproduce what
the model's own ``forward`` does to its logits. A wrong factor does not raise: it
silently shifts every log-prob, and with it the importance ratio.

``detect_logit_transforms`` owns the field names, this test owns the expected
*values*, so a change in its coverage shows up here as a diff rather than a quiet
numerics shift.

The loss applies them in the order multiply, divide, soft cap
(``unsloth_zoo/rl_replacements.py``), matching what each family's ``modeling_*.py``
does on the line after ``lm_head``.
"""

from __future__ import annotations

import pytest


detect = pytest.importorskip(
    "unsloth_zoo.device_map_planner",
    reason = "unsloth_zoo without the planner: the fallback branch is in use",
).__dict__.get("detect_logit_transforms")

pytestmark = pytest.mark.skipif(
    detect is None,
    reason = "installed unsloth_zoo predates detect_logit_transforms",
)


class _Cfg:
    """Stand-in for a PretrainedConfig, built by hand because the newest families
    here only exist on transformers 5.9+ and this test runs on the pinned version."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, _Cfg(**value) if isinstance(value, dict) else value)


def _grpo_transforms(config):
    """Exactly what rl_replacements derives before handing off to the loss."""
    t = detect(config)
    return (
        t["logit_softcapping"],
        t["logit_scale_multiply"],
        t["logit_scale_divide"],
    )


# (name, config, (softcapping, multiply, divide), why)
_COVERED = [
    (
        "granite",
        _Cfg(model_type = "granite", logits_scaling = 8.0),
        (0.0, 0.0, 8.0),
        "modeling_granite.py divides: logits = logits / config.logits_scaling",
    ),
    (
        "cohere",
        _Cfg(model_type = "cohere", logit_scale = 0.0625),
        (0.0, 0.0625, 0.0),
        "modeling_cohere.py multiplies: logits = logits * self.logit_scale",
    ),
    (
        "falcon_h1",
        _Cfg(model_type = "falcon_h1", lm_head_multiplier = 0.01953125),
        (0.0, 0.01953125, 0.0),
        "modeling_falcon_h1.py multiplies by model.lm_head_multiplier",
    ),
    (
        "gemma2",
        _Cfg(model_type = "gemma2", final_logit_softcapping = 30.0),
        (30.0, 0.0, 0.0),
        "Gemma-style tanh soft cap, no scale",
    ),
]


@pytest.mark.parametrize("name, config, expected, why", _COVERED)
def test_stable_families(name, config, expected, why):
    assert _grpo_transforms(config) == expected, why


# Families the helper only bucketed recently. Each entry records what the GRPO path
# used to apply, so the behaviour change stays legible.
_RECENT = [
    (
        "muse_glimmer",
        _Cfg(
            model_type = "muse_glimmer",
            text_config = {
                "final_logit_softcapping": 20.0,
                "output_multiplier": 0.19611613513818404,
            },
        ),
        (20.0, 0.19611613513818404, 0.0),
        (20.0, 0.0, 0.0),
        "transformers applies logits * output_multiplier before the soft cap",
    ),
    (
        "hyperclovax",
        _Cfg(model_type = "hyperclovax", logits_scaling = 8.0),
        (0.0, 8.0, 0.0),
        (0.0, 0.0, 8.0),
        "MuP multiplies by logits_scaling, unlike Granite which divides",
    ),
    (
        "minicpm3",
        _Cfg(model_type = "minicpm3", logits_scaling = 8.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 8.0),
        "logits_scaling scales hidden states before the head, not the logits",
    ),
]


@pytest.mark.parametrize("name, config, expected, previously, why", _RECENT)
def test_recently_rebucketed_families(name, config, expected, previously, why):
    """These three changed what the GRPO path applies. Deliberate, see the PR body.

    Skips rather than fails on an unsloth_zoo predating the coverage, so the suite
    stays green across the version window instead of pinning one release.
    """
    got = _grpo_transforms(config)
    if got == previously:
        pytest.skip(f"installed unsloth_zoo predates {name} coverage")
    assert got == expected, why
