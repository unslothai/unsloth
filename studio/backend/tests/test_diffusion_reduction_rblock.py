# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion compiles pin inductor's dynamic_scale_rblock off, so one seed renders one image in every server process.

With it on, a register-heavy looped reduction (FLUX's LayerNorm + int8 activation-quant kernel) gets a second launcher
with half the R0_BLOCK, benchmarked on first use in every process and never cached; the two block sizes give different
bits, so FLUX.1-schnell renders differed between Studio servers on a B200 (4 of 15). The CUDA test reproduces that
kernel and checks no runtime autotune is left once the knob is set."""

from __future__ import annotations

import hashlib
import types

import pytest

torch = pytest.importorskip("torch")
import torch._inductor.config  # noqa: E402

from core.inference import diffusion_compile_cache as cache  # noqa: E402
from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_INDUCTOR = torch._inductor.config
_KILL_SWITCH = "UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    before = (_INDUCTOR.dynamic_scale_rblock, _INDUCTOR.emulate_precision_casts)
    monkeypatch.delenv(_KILL_SWITCH, raising = False)
    cc._reset_for_tests()
    yield
    cc._reset_for_tests()
    _INDUCTOR.dynamic_scale_rblock, _INDUCTOR.emulate_precision_casts = before


class _Transformer:
    _repeated_blocks = ()

    def compile_repeated_blocks(self, **kwargs):
        self.kwargs = kwargs

    def named_modules(self):
        return iter(())

    def modules(self):
        return iter(())

    def parameters(self):
        return iter(())


def _compile(max_autotune = False):
    pipe = types.SimpleNamespace(transformer = _Transformer())
    assert ds._compile_repeated_blocks(pipe, None, max_autotune = max_autotune) is True


@pytest.mark.parametrize("max_autotune", [False, True])
def test_regional_compile_pins_dynamic_scale_rblock_off(max_autotune):
    _INDUCTOR.dynamic_scale_rblock = True
    _compile(max_autotune)
    assert _INDUCTOR.dynamic_scale_rblock is False
    assert cc.get_knob("torch._inductor.config", "dynamic_scale_rblock") is False


def test_kill_switch_keeps_inductor_default(monkeypatch):
    monkeypatch.setenv(_KILL_SWITCH, "1")
    _INDUCTOR.dynamic_scale_rblock = True
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is True


def test_unload_restores_the_process_value():
    _INDUCTOR.dynamic_scale_rblock = True
    snap = ds.snapshot_backend_flags()
    assert snap["inductor_dynamic_scale_rblock"] is True
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is False
    ds.restore_backend_flags(snap)
    assert _INDUCTOR.dynamic_scale_rblock is True


def _fingerprint():
    return cache.model_fingerprint(
        family = "flux.1",
        transformer = None,
        dtype = "torch.bfloat16",
        quant = "int8",
        attention_backend = "_native_cudnn",
        compile_kwargs = {"fullgraph": True, "dynamic": None, "mode": "default"},
    )


def test_bundle_key_tracks_the_pin(monkeypatch):
    pinned = _fingerprint()
    assert pinned["inductor"] == {"dynamic_scale_rblock": False}
    monkeypatch.setenv(_KILL_SWITCH, "1")
    unpinned = _fingerprint()
    # Kill switch = the pre-pin key, so bundles compiled with inductor's default still hit under it.
    assert "inductor" not in unpinned
    env = cache.environment_fingerprint()
    assert cache.cache_key(env, pinned) != cache.cache_key(env, unpinned)


def _flux_like_block():
    """Single-stream FLUX block head: LayerNorm + AdaLN modulation feeding four torchao int8 dynamic-activation
    Linears (q, k, v, mlp). Inductor fuses the norm and the four activation quantisations into one looped reduction."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(3072, elementwise_affine = False, eps = 1e-6)
            self.q = torch.nn.Linear(3072, 3072)
            self.k = torch.nn.Linear(3072, 3072)
            self.v = torch.nn.Linear(3072, 3072)
            self.mlp = torch.nn.Linear(3072, 12288)

        def forward(self, x, shift, scale):
            h = self.norm(x) * (1 + scale) + shift
            return self.q(h), self.k(h), self.v(h), self.mlp(h)

    torch.manual_seed(0)
    block = Block().cuda().to(torch.bfloat16)
    quantize_(block, Int8DynamicActivationInt8WeightConfig(set_inductor_config = False))
    x = torch.randn(4096, 3072, device = "cuda", dtype = torch.bfloat16)
    shift = torch.randn(3072, device = "cuda", dtype = torch.bfloat16)
    scale = torch.randn(3072, device = "cuda", dtype = torch.bfloat16)
    return block, (x, shift, scale)


def _run_counting_reduction_autotunes(
    monkeypatch,
    tmp_path,
    *,
    force_r0_block = None,
):
    """Compile + run the block in a fresh inductor cache. Returns (R0_BLOCK sets benchmarked at runtime, output sha).
    ``force_r0_block`` makes a two-launcher reduction take that block size instead of benchmarking."""
    from torch._inductor.runtime import triton_heuristics

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    from torch._inductor.codecache import PyCodeCache

    torch._dynamo.reset()
    PyCodeCache.cache_clear()
    seen = []
    real = triton_heuristics.CachingAutotuner.autotune_to_one_config

    def autotune(self, *args, **kwargs):
        blocks = sorted(
            {launcher.config.kwargs.get("R0_BLOCK") for launcher in self.launchers} - {None}
        )
        if blocks:
            seen.append(blocks)
            if force_r0_block in blocks:
                self.launchers = [
                    l for l in self.launchers if l.config.kwargs.get("R0_BLOCK") == force_r0_block
                ]
                return None
        return real(self, *args, **kwargs)

    monkeypatch.setattr(triton_heuristics.CachingAutotuner, "autotune_to_one_config", autotune)
    block, inputs = _flux_like_block()
    with torch.no_grad():
        out = torch.compile(block, fullgraph = True, dynamic = False)(*inputs)
    torch.cuda.synchronize()
    digest = hashlib.sha256()
    for t in out:
        digest.update(t.float().cpu().numpy().tobytes())
    torch._dynamo.reset()
    return seen, digest.hexdigest()


def _cuda_torchao_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("inductor's dynamic_scale_rblock is sm80+ only")
    pytest.importorskip("torchao")
    pytest.importorskip("triton")


@pytest.mark.gpu
def test_pinned_reduction_has_one_block_size_and_the_default_has_two(monkeypatch, tmp_path):
    _cuda_torchao_or_skip()
    _INDUCTOR.emulate_precision_casts = True
    _INDUCTOR.dynamic_scale_rblock = True
    default_seen, _ = _run_counting_reduction_autotunes(monkeypatch, tmp_path / "default")
    if not default_seen:
        pytest.skip(
            "this GPU does not give the fused norm kernel a second R0_BLOCK (register budget not exceeded)"
        )
    # The second launcher is the same tile with half the reduction block.
    assert any(len(b) == 2 and b[1] == 2 * b[0] for b in default_seen), default_seen
    small, large = next(b for b in default_seen if len(b) == 2)
    _, sha_small = _run_counting_reduction_autotunes(
        monkeypatch, tmp_path / "small", force_r0_block = small
    )
    _, sha_large = _run_counting_reduction_autotunes(
        monkeypatch, tmp_path / "large", force_r0_block = large
    )
    # Which one wins the per-process benchmark changes the bits: that is the cross-server drift.
    assert sha_small != sha_large
    _compile()
    pinned_seen, sha_pinned = _run_counting_reduction_autotunes(monkeypatch, tmp_path / "pinned")
    assert pinned_seen == []
    assert sha_pinned in (sha_small, sha_large)
