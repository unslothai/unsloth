# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the (lr_scheduler, lr_warmup_steps) coupling.

A family that advertises lr_warmup_steps must also advertise a scheduler that
realizes it: diffusers' "constant" schedule never reads num_warmup_steps, so
the pair ("constant", warmup > 0) is a silent no-op. Covers the defaults-table
invariant, that normalized() validates the warmup value without rewriting the
requested schedule, and that no-warmup behaviour is unchanged."""

from __future__ import annotations

import pytest

from core.training.diffusion_train_common import (
    FAMILY_TRAIN_DEFAULTS,
    DiffusionLoraConfig,
    train_defaults,
)

# Derived from the table, not listed by hand: a family added with lr_warmup_steps and no
# scheduler must fail this, which a hand-written list would silently skip.
WARMUP_FAMILIES = [
    family
    for family, defaults in FAMILY_TRAIN_DEFAULTS.items()
    if int(defaults.get("lr_warmup_steps", 0) or 0) > 0
]
NO_WARMUP_FAMILIES = [family for family in FAMILY_TRAIN_DEFAULTS if family not in WARMUP_FAMILIES]


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
    assert WARMUP_FAMILIES, "no family advertises lr_warmup_steps; the invariant tests nothing"
    for family in WARMUP_FAMILIES:
        d = FAMILY_TRAIN_DEFAULTS[family]
        assert (
            d.get("lr_scheduler") == "constant_with_warmup"
        ), f"{family} advertises lr_warmup_steps but not a scheduler that realizes it"


def test_no_warmup_families_are_untouched():
    for family in NO_WARMUP_FAMILIES:
        d = FAMILY_TRAIN_DEFAULTS[family]
        assert "lr_warmup_steps" not in d
        assert "lr_scheduler" not in d
        cfg = _cfg(family).normalized()
        assert cfg.lr_scheduler == "constant"
        assert cfg.lr_warmup_steps == 0


def test_normalized_leaves_the_requested_scheduler_alone():
    for scheduler in ("constant", "constant_with_warmup", "cosine", "linear"):
        cfg = _cfg("sdxl", lr_scheduler = scheduler, lr_warmup_steps = 20).normalized()
        assert cfg.lr_scheduler == scheduler
        assert cfg.lr_warmup_steps == 20
    cfg = _cfg("sdxl", lr_scheduler = "constant", lr_warmup_steps = 0).normalized()
    assert cfg.lr_scheduler == "constant"


def test_a_constant_schedule_with_warmup_still_resumes_its_own_bundle():
    """lr_scheduler is a checkpoint identity field, so rewriting the requested pair inside
    normalized() would strand every bundle written before the rewrite: its manifest records
    "constant" while replaying that same config would now normalise to something else, and
    resume preflight rejects the difference as a changed learning-rate schedule."""
    from core.training.diffusion_checkpoint import CheckpointIdentity, identity_for_config

    cfg = _cfg("sdxl", lr_scheduler = "constant", lr_warmup_steps = 20).normalized()
    incoming = identity_for_config(cfg)
    assert incoming.lr_scheduler == "constant"
    stored = CheckpointIdentity.from_dict(
        {**incoming.as_dict(), "lr_scheduler": "constant", "lr_warmup_steps": 20}
    )
    assert stored is not None
    assert stored.mismatch_reason(incoming) is None


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match = "lr_warmup_steps"):
        _cfg("sdxl", lr_warmup_steps = -1).normalized()
