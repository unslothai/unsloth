# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import importlib.util

import pytest


def _spec(present):
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "mamba_ssm":
            return object() if present else None
        return real(name, *a, **k)

    return fake


@pytest.mark.parametrize("mt", ["mamba", "falcon_mamba"])
def test_fused_kernel_weights_dense_when_mamba_ssm_importable(monkeypatch, mt):
    from unsloth.models.vision import _architecture_skip_modules

    monkeypatch.setattr(importlib.util, "find_spec", _spec(True))
    skip = _architecture_skip_modules([mt])
    assert {"dt_proj", "x_proj", "out_proj"} <= set(skip) and "in_proj" not in skip
    assert len(skip) == len(set(skip))


def test_without_mamba_ssm_only_dt_proj_stays_dense(monkeypatch):
    from unsloth.models.vision import _architecture_skip_modules

    monkeypatch.setattr(importlib.util, "find_spec", _spec(False))
    assert _architecture_skip_modules(["mamba"]) == ["dt_proj"]
    assert _architecture_skip_modules(["falcon_mamba"]) == []
    assert _architecture_skip_modules(["nemotron_h"]) == ["out_proj"]
