# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPTQ bases (packed qweight, no dense .weight) train and decode through their own forward (#1573)."""

from __future__ import annotations

import types

import pytest
import torch
import unsloth  # noqa: F401
import transformers.utils

from unsloth.kernels.utils import fast_linear_forward
from unsloth.models.llama import _base_weight_dtype, _has_packed_base
from unsloth.models.loader_utils import gptq_trainable_quantization_config


class _PackedLinear(torch.nn.Module):
    """GPTQ-shaped linear: int32 qweight buffer, no dense weight, optional metadata-only `weight`."""

    def __init__(
        self,
        in_features = 16,
        out_features = 8,
        weight_metadata = False,
    ):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            "qweight", torch.zeros(in_features // 8, out_features, dtype = torch.int32)
        )
        self.register_buffer("dense", torch.randn(out_features, in_features))
        self.bias = None
        if weight_metadata:
            self.weight = types.SimpleNamespace(
                dtype = torch.float16, shape = (out_features, in_features)
            )

    def forward(self, X):
        return X @ self.dense.t()


@pytest.mark.parametrize("weight_metadata", [False, True])
def test_packed_base_detected(weight_metadata):
    proj = _PackedLinear(weight_metadata = weight_metadata)
    assert _base_weight_dtype(proj) is None
    assert _has_packed_base(proj)
    assert _has_packed_base(torch.nn.Linear(4, 4), proj)


def test_dense_and_mxfp4_bases_not_packed():
    assert not _has_packed_base(torch.nn.Linear(4, 4, dtype = torch.bfloat16))

    class _Mxfp4(torch.nn.Module):
        _unsloth_mxfp4_packed_linear = True

    assert not _has_packed_base(_Mxfp4())


@pytest.mark.parametrize("weight_metadata", [False, True])
def test_fast_linear_forward_runs_packed_module(weight_metadata):
    proj = _PackedLinear(weight_metadata = weight_metadata)
    X = torch.randn(1, 1, 16)
    torch.testing.assert_close(fast_linear_forward(proj, X), proj(X))
    out = torch.empty(1, 1, 8)
    assert fast_linear_forward(proj, X, out = out) is out
    torch.testing.assert_close(out, proj(X))


def test_fast_linear_forward_dense_unchanged():
    proj = torch.nn.Linear(16, 8, bias = False)
    X = torch.randn(1, 1, 16)
    torch.testing.assert_close(fast_linear_forward(proj, X), proj(X))


def _config(quantization_config):
    return types.SimpleNamespace(quantization_config = quantization_config)


_GPTQ8 = {"quant_method": "gptq", "bits": 8, "group_size": 128, "desc_act": False, "sym": True}


def test_gptq_checkpoint_gets_trainable_backend(monkeypatch):
    monkeypatch.setattr(transformers.utils, "is_gptqmodel_available", lambda: True)
    cfg = gptq_trainable_quantization_config(_config(dict(_GPTQ8)), None)
    assert cfg.backend == "auto_trainable" and cfg.bits == 8
    assert cfg.get_loading_attributes()["backend"] == "auto_trainable"


def test_gptq_backend_left_alone(monkeypatch):
    monkeypatch.setattr(transformers.utils, "is_gptqmodel_available", lambda: True)
    assert gptq_trainable_quantization_config(_config(dict(_GPTQ8)), object()) is None
    assert (
        gptq_trainable_quantization_config(_config({**_GPTQ8, "backend": "gptq_torch"}), None)
        is None
    )
    assert (
        gptq_trainable_quantization_config(_config({"quant_method": "awq", "bits": 4}), None)
        is None
    )
    assert (
        gptq_trainable_quantization_config(_config({"quant_method": "bitsandbytes"}), None) is None
    )
    assert gptq_trainable_quantization_config(_config(None), None) is None
    monkeypatch.setattr(transformers.utils, "is_gptqmodel_available", lambda: False)
    assert gptq_trainable_quantization_config(_config(dict(_GPTQ8)), None) is None
