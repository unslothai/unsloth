# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

_spec = importlib.util.spec_from_file_location(
    "unsloth_lora_init", Path(__file__).resolve().parents[1] / "unsloth" / "models" / "lora_init.py"
)
lora_init = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lora_init)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _weight(
    out_features,
    in_features,
    seed = 0,
):
    # Slowly decaying spectrum, like LLM projections.
    g = torch.Generator().manual_seed(seed)
    k = min(out_features, in_features)
    U = torch.linalg.qr(torch.randn(out_features, k, generator = g, dtype = torch.float64)).Q
    V = torch.linalg.qr(torch.randn(in_features, k, generator = g, dtype = torch.float64)).Q
    S = torch.logspace(0, -2, k, dtype = torch.float64) + 0.01 * torch.rand(
        k, generator = g, dtype = torch.float64
    )
    return (U * S) @ V.T


@pytest.mark.parametrize("shape", [(96, 48), (48, 96), (64, 64)])
@pytest.mark.parametrize("rank", [1, 8])
def test_randomized_svd_near_optimal(shape, rank):
    W = _weight(*shape)
    U, S, Vh = lora_init.randomized_svd(W.float().to(DEVICE), rank)
    Ue, Se, Vhe = torch.linalg.svd(W, full_matrices = False)
    best = (W - (Ue[:, :rank] * Se[:rank]) @ Vhe[:rank]).norm()
    err = (W - (U.double().cpu() * S.double().cpu()) @ Vh.double().cpu()).norm()
    assert err / best < 1.001
    assert torch.allclose(S.double().cpu(), Se[:rank], rtol = 1e-3)
    assert U.dtype == torch.float32 and U.shape == (shape[0], rank) and Vh.shape == (rank, shape[1])


def test_randomized_svd_rank_deficient_and_zero():
    W = torch.zeros(40, 30, device = DEVICE)
    U, S, Vh = lora_init.randomized_svd(W, 4)
    assert torch.isfinite(U).all() and torch.isfinite(Vh).all() and (S == 0).all()
    W = torch.randn(40, 2, device = DEVICE) @ torch.randn(2, 30, device = DEVICE)
    U, S, Vh = lora_init.randomized_svd(W, 4)
    assert torch.isfinite(U).all() and torch.allclose((U * S) @ Vh, W, atol = 1e-4)


@pytest.mark.parametrize("shape", [(96, 48), (48, 96), (64, 64)])
def test_mica_basis_matches_fp64_svd(shape):
    W = _weight(*shape)
    r = 8
    B = lora_init.mica_basis(W.float(), r)
    U = torch.linalg.svd(W, full_matrices = False)[0][:, -r:]
    assert torch.linalg.matrix_norm(U @ U.T - B.double() @ B.double().T, ord = 2) < 1e-4
    # Same column order as PEFT's U[:, -r:].
    assert torch.nn.functional.cosine_similarity(B.double(), U, dim = 0).abs().min() > 0.999


class _Model(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.q = torch.nn.Linear(W.shape[1], W.shape[0], bias = False)
        self.q.weight.data = W.float()

    def forward(self, x):
        return self.q(x)


@pytest.mark.parametrize("init", ["pissa", "pissa_niter_4"])
def test_fast_pissa_preserves_output(init):
    W = _weight(96, 64)
    base = _Model(W).to(DEVICE)
    x = torch.randn(3, 64, device = DEVICE)
    with lora_init.fast_lora_init():
        assert LoraLayer.pissa_init is lora_init._pissa_init
        model = get_peft_model(
            copy.deepcopy(base),
            LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q"], init_lora_weights = init),
        )
    assert LoraLayer.pissa_init is not lora_init._pissa_init
    assert torch.allclose(model(x), base(x), atol = 1e-5)
    layer = model.base_model.model.q
    BA = (
        layer.scaling["default"]
        * layer.lora_B["default"].weight.double().cpu()
        @ layer.lora_A["default"].weight.double().cpu()
    )
    Ue, Se, Vhe = torch.linalg.svd(W, full_matrices = False)
    assert (W - BA).norm() / (W - (Ue[:, :8] * Se[:8]) @ Vhe[:8]).norm() < 1.001


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FAST_LORA_INIT", "0")
    original = LoraLayer.pissa_init
    with lora_init.fast_lora_init():
        assert LoraLayer.pissa_init is original
