# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the (lr_scheduler, lr_warmup_steps) coupling.

A family that advertises lr_warmup_steps must also advertise a scheduler that
realizes it: diffusers' "constant" schedule never reads num_warmup_steps, so
the pair ("constant", warmup > 0) is a silent no-op. Covers the defaults-table
invariant, the normalized() promotion backstop for direct callers, and that
no-warmup behaviour is unchanged."""

from __future__ import annotations

import pytest

from core.training.diffusion_train_common import (
    FAMILY_TRAIN_DEFAULTS,
    DiffusionLoraConfig,
    train_defaults,
)

WARMUP_FAMILIES = ["flux.1", "qwen-image", "flux.2-klein", "flux.2-dev"]
NO_WARMUP_FAMILIES = ["sdxl", "z-image", "krea-2"]


def _cfg(family, **overrides):
    d = train_defaults(family)
    kwargs = {k: v for k, v in d.items() if k in DiffusionLoraConfig.__dataclass_fields__}
    kwargs.update(overrides)
    return DiffusionLoraConfig(
        base_model = f"stub/{family}",
        data_dir = "/tmp/data",
        output_dir = "/tmp/out",
        model_family = family,
        **kwargs,
    )


def test_warmup_families_advertise_a_warmup_capable_scheduler():
    for family in WARMUP_FAMILIES:
        d = FAMILY_TRAIN_DEFAULTS[family]
        assert d.get("lr_warmup_steps", 0) > 0
        assert d.get("lr_scheduler") == "constant_with_warmup", (
            f"{family} advertises lr_warmup_steps but not a scheduler that realizes it"
        )


def test_no_warmup_families_are_untouched():
    for family in NO_WARMUP_FAMILIES:
        d = FAMILY_TRAIN_DEFAULTS[family]
        assert "lr_warmup_steps" not in d
        assert "lr_scheduler" not in d
        cfg = _cfg(family).normalized()
        assert cfg.lr_scheduler == "constant"
        assert cfg.lr_warmup_steps == 0


def test_normalized_promotes_constant_with_positive_warmup():
    cfg = _cfg("sdxl", lr_scheduler = "constant", lr_warmup_steps = 20).normalized()
    assert cfg.lr_scheduler == "constant_with_warmup"
    assert cfg.lr_warmup_steps == 20


def test_normalized_leaves_explicit_schedulers_alone():
    for scheduler in ("constant_with_warmup", "cosine", "linear"):
        cfg = _cfg("sdxl", lr_scheduler = scheduler, lr_warmup_steps = 20).normalized()
        assert cfg.lr_scheduler == scheduler
    cfg = _cfg("sdxl", lr_scheduler = "constant", lr_warmup_steps = 0).normalized()
    assert cfg.lr_scheduler == "constant"


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match = "lr_warmup_steps"):
        _cfg("sdxl", lr_warmup_steps = -1).normalized()
