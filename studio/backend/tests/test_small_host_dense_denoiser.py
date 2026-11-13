# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host transformer_quant none keeps the denoiser memory-mapped; status never reports 'off' over int8."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_small_host as sh

QWEN21 = {
    "transformer": sh.StoredComponent("transformer", 13900, "bfloat16"),
    "vae": sh.StoredComponent("vae", 600, "bfloat16"),
}
T4_TOTAL = 12 * 1024 + 700
T4_AVAILABLE = 8680


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV, sh.DENSE_DENOISER_ENV, sh.SPILL_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(**kw):
    return sh.decide_small_host(
        QWEN21,
        torch.float16,
        device = "cuda",
        host_total_mib = T4_TOTAL,
        host_available_mib = T4_AVAILABLE,
        spill_free_mib = 200_000,
        **kw,
    )


def test_none_keeps_the_denoiser_mapped_and_never_spills_int8():
    auto = _decide()
    dense = _decide(dense_denoiser = True)
    assert auto.engaged and dense.engaged
    assert auto.int8_spill and not auto.dense_denoiser  # the auto route: int8 in a scratch file
    assert dense.dense_denoiser and not dense.int8_spill
    assert (
        dense.storage_dtypes
        == auto.storage_dtypes
        == {"transformer": "bfloat16", "vae": "bfloat16"}
    )
    # memory-mapped like an encoder: only the converted vae lands in host RAM
    assert dense.route_host_mib == 600 * 2 // 2
    assert dense.refuse is None


def test_the_kill_switch_reads_off(monkeypatch):
    assert sh.dense_denoiser_enabled()
    monkeypatch.setenv(sh.DENSE_DENOISER_ENV, "0")
    assert not sh.dense_denoiser_enabled()


def _dit():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(64)
            self.proj = torch.nn.Linear(64, 256)
            self.out = torch.nn.Linear(256, 64)

        def forward(self, x):
            return x + self.out(torch.nn.functional.gelu(self.proj(self.norm(x))))

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = torch.nn.Linear(16, 64)
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
            self.proj_out = torch.nn.Linear(64, 16)

        def forward(self, x):
            h = self.x_embedder(x)
            for block in self.blocks:
                h = block(h)
            return self.proj_out(h)

    return DiT


def test_a_mapped_denoiser_matches_the_dense_conversion_bit_for_bit():
    pytest.importorskip("diffusers.hooks")
    DiT = _dit()
    torch.manual_seed(0)
    stored = DiT().to(torch.bfloat16)
    state = {k: v.clone() for k, v in stored.state_dict().items()}
    dense = DiT()
    dense.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})
    dense = dense.to(torch.float32)  # the dense load: bf16 converted once
    mapped = DiT().to(torch.bfloat16)
    mapped.load_state_dict(state)
    sh.prepare_streamed_encoder_(mapped, torch.float32)
    assert mapped.blocks[1].proj.weight.dtype == torch.bfloat16  # still as stored
    x = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.equal(mapped(x), dense(x))
        assert torch.equal(mapped(x), dense(x))  # the per-call cast back to storage is lossless
    assert mapped.blocks[1].proj.weight.dtype == torch.bfloat16


def test_mapped_denoiser_blocks_lora_and_reports_dense():
    from core.inference.diffusion import _lora_load_block_reason, _small_host_int8

    pipe = types.SimpleNamespace(
        _unsloth_small_host = {
            "components": {
                "transformer": "memory-mapped bfloat16, layerwise cast to float16 (13000 MiB)"
            }
        },
        transformer = None,
    )
    assert sh.mapped_denoiser(pipe)
    assert not _small_host_int8(pipe)
    assert "casts them per layer" in (_lora_load_block_reason(pipe) or "")
    enc = types.SimpleNamespace(
        _unsloth_small_host = {
            "components": {"text_encoder": "memory-mapped, layerwise cast (9 MiB)"}
        }
    )
    assert not sh.mapped_denoiser(enc)


def test_status_reports_int8_over_an_unengaged_small_host_int8_denoiser():
    from core.inference.diffusion import _reported_transformer_quant

    int8_pipe = types.SimpleNamespace(
        _unsloth_small_host = {"components": {"transformer": "int8 weights (6784 MiB, 229 linears)"}}
    )
    assert (
        _reported_transformer_quant(types.SimpleNamespace(transformer_quant = None, pipe = int8_pipe))
        == "int8"
    )
    assert (
        _reported_transformer_quant(types.SimpleNamespace(transformer_quant = "fp8", pipe = int8_pipe))
        == "fp8"
    )
    plain = types.SimpleNamespace(transformer_quant = None, pipe = types.SimpleNamespace())
    assert _reported_transformer_quant(plain) is None


def test_the_load_passes_an_explicit_none_to_the_small_host_decision():
    from pathlib import Path

    import core.inference.diffusion as d

    src = Path(d.__file__).read_text(encoding = "utf-8")
    flag = src.index("small_host_dense = (")
    assert src.count("dense_denoiser = small_host_dense,") == 2
    assert all(i > flag for i in [src.index("dense_denoiser = small_host_dense,")])
    body = src[flag : flag + 400]
    assert "not transformer_quant_auto" in body and "transformer_quant_pinned is None" in body


@pytest.mark.parametrize("mapped", [False, True])
def test_qkv_fusion_skips_a_mapped_denoiser(mapped):
    import torch

    from core.inference import diffusion_speed as ds

    # the first Linear converts (it sets the model dtype), the second stays mapped with a cast hook
    transformer = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(
        torch.bfloat16
    )
    if mapped:
        sh.prepare_streamed_encoder_(transformer, torch.float16)
    calls = []

    class _Pipe:
        def __init__(self):
            self.transformer = transformer

        def fuse_qkv_projections(self):
            calls.append(1)

    assert ds._fuse_qkv(_Pipe(), None) is not mapped
    assert calls == ([] if mapped else [1])
