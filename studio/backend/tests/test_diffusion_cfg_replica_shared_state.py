# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a CFG replica shares with its source (FP8 step policy) or rebuilds on its card (FLUX.3 QKV). CPU only."""

import contextlib
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_modelopt_fp8 as mo  # noqa: E402
from core.inference import diffusion_multigpu as mg  # noqa: E402


class _Fp8Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cls = mo.fp8_linear_class()
        g = torch.Generator().manual_seed(0)
        w = torch.randn(16, 32, generator = g)
        scale = float(w.abs().max()) / 448
        codes = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.policy = mo.StepPolicy(first_steps = 3, last_steps = 3)
        self.proj = cls(
            32,
            16,
            codes,
            scale,
            0.05,
            compute_dtype = torch.float32,
            path = "generation",
            policy = self.policy,
        )
        mo.install_step_policy(self, self.policy)

    @contextlib.contextmanager
    def cache_context(self, name, **kwargs):
        yield

    def forward(self, x):
        return self.proj(x)


def test_replica_follows_the_source_fp8_step_policy():
    model = _Fp8Model()
    with model.cache_context("cond", step_index = 0, num_inference_steps = 20):
        replica = mg.build_replica(model, torch.device("cpu"))  # built on a high-precision step
    x = torch.randn(4, 32, generator = torch.Generator().manual_seed(1))
    for step in (0, 3, 10, 17, 19):
        with model.cache_context("cond", step_index = step, num_inference_steps = 20):
            a16 = model.proj.policy.high_precision("generation")
            assert replica.proj.policy.high_precision("generation") is a16, step
            assert torch.equal(model(x), replica(x)), step
    assert replica.proj.policy is model.proj.policy


def test_shared_fp8_policy_keeps_warm_signatures_stable():
    """A per-call compile key would keep the two cards from ever overlapping."""
    policy = mo.StepPolicy(first_steps = 3, last_steps = 3)
    policy.set_step(5, 20)
    assert mg._shared_state_key((policy,)) == mg._shared_state_key((policy,))
    before = mg._shared_state_key((policy,))
    policy.set_step(0, 20)
    assert mg._shared_state_key((policy,)) != before


def _int8_attention():
    pytest.importorskip("torchao")
    from torchao.quantization import Int8Tensor
    from torchao.quantization.granularity import PerRow

    from core.inference import diffusion_flux_convrot_fused as F
    from core.inference import diffusion_int8_gemm as G

    class Flux3Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(2)
            for name in ("to_q", "to_k", "to_v"):
                lin = torch.nn.Linear(64, 48, bias = False, dtype = torch.bfloat16)
                w = Int8Tensor.from_hp(
                    torch.randn(48, 64, generator = g).to(torch.bfloat16), PerRow()
                )
                lin.weight = torch.nn.Parameter(w, requires_grad = False)
                lin.__dict__[G._REC] = ("v2", 256, True, lin.weight, True)
                lin.__dict__[G._MARK] = None
                lin.__dict__["forward"] = types.MethodType(G._linear_forward, lin)
                setattr(self, name, lin)

    class Flux3Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Flux3Attention(), Flux3Attention()])

    return F, G, Flux3Attention, Flux3Transformer2DModel


def test_replica_merges_its_own_qkv_instead_of_copying_the_source_buffer(monkeypatch):
    F, G, attn_cls, model_cls = _int8_attention()
    from core.inference import diffusion_int8_fused as I

    monkeypatch.setitem(F._CLS, "Flux3Attention", attn_cls)
    monkeypatch.setattr(I, "resident_cuda_cards", lambda m: (0,))
    monkeypatch.delenv(F.FLUX3_FUSED_QKV_ENV, raising = False)
    source = model_cls()
    assert F.merge_qkv(source) == 2
    replica = mg._copy_module(source, torch.device("meta"))
    for attn in (m for m in replica.modules() if type(m) is attn_cls):
        record = attn.__dict__.get(F._QKV_ATTR)
        assert record is not None, "the replica lost the merged Q / K / V GEMM"
        q, k, v, (wq, ws, n, _fused) = record
        assert (q, k, v) == (attn.to_q, attn.to_k, attn.to_v)
        assert wq.device.type == "meta" and ws.device.type == "meta" and n == 3
        assert tuple(wq.shape) == (144, 64)
        for i, lin in enumerate((attn.to_q, attn.to_k, attn.to_v)):
            assert lin.weight.qdata.device.type == "meta"
            assert lin.weight.qdata.storage_offset() == 48 * 64 * i


def test_replica_of_an_unmerged_flux3_stays_unmerged(monkeypatch):
    F, G, attn_cls, model_cls = _int8_attention()
    from core.inference import diffusion_int8_fused as I

    monkeypatch.setitem(F._CLS, "Flux3Attention", attn_cls)
    monkeypatch.setattr(I, "resident_cuda_cards", lambda m: (0,))
    replica = mg._copy_module(model_cls(), torch.device("meta"))
    assert F.fused_qkv_count(replica) == 0
