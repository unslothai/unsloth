# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A VAE declaring ``_repeated_blocks`` (MiniMax-H3) compiles its repeated block, never the tiled ``decode``."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_speed as ds_mod  # noqa: E402


class _Blk(torch.nn.Module):
    def forward(self, x):
        return x + 1


class _RegionalVae(torch.nn.Module):
    """Mirrors diffusers ModelMixin.compile_repeated_blocks, with a stand-in compiled callable per block."""

    _repeated_blocks = ["_Blk"]

    def __init__(self, compiled_factory = None):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Blk(), _Blk()])
        self.compile_kwargs = None
        self.compiled_factory = compiled_factory
        self.use_tiling = True

    def compile_repeated_blocks(self, **kwargs):
        self.compile_kwargs = kwargs
        for m in self.modules():
            if type(m).__name__ in self._repeated_blocks:
                m._compiled_call_impl = (
                    self.compiled_factory(m) if self.compiled_factory else m._call_impl
                )

    def decode(self, z):
        for b in self.blocks:
            z = b(z)
        return z


def _inductor_error(msg):
    try:
        from torch._inductor.exc import InductorError  # noqa: PLC0415
    except ImportError:  # torch < 2.7 raises BackendCompilerFailed instead
        from torch._dynamo.exc import BackendCompilerFailed  # noqa: PLC0415

        return BackendCompilerFailed("inductor", RuntimeError(msg))
    try:
        return InductorError(RuntimeError(msg), None)
    except TypeError:  # constructor signature varies by torch version
        return InductorError(msg)


def test_regional_vae_compiles_blocks_static_and_keeps_decode():
    vae = _RegionalVae()
    pipe = types.SimpleNamespace(vae = vae)
    assert ds_mod._compile_vae_decode(pipe, None) is True
    assert vae.compile_kwargs == {"fullgraph": False, "dynamic": False}
    # the tiled decode itself is never wrapped: the tile loop stays in Python
    assert "decode" not in vae.__dict__
    assert vae.decode(torch.zeros(1)).item() == 2
    assert vae._unsloth_compiled_decode is True
    # idempotent across the dual-DiT second call
    vae.compile_kwargs = None
    assert ds_mod._compile_vae_decode(pipe, None) is True
    assert vae.compile_kwargs is None


def test_regional_vae_max_tier_autotunes_without_cudagraphs():
    vae = _RegionalVae()
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None, max_autotune = True)
    assert vae.compile_kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert vae.compile_kwargs["dynamic"] is False


def test_regional_vae_tiling_does_not_force_eager():
    # eager_when_tiled exists because a WHOLE-decode compile unrolls the tile loop; a block compile does not.
    calls = {"compiled": 0}

    def factory(m):
        def compiled(*a, **k):
            calls["compiled"] += 1
            return m._call_impl(*a, **k)

        return compiled

    vae = _RegionalVae(factory)
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None, eager_when_tiled = True)
    vae.decode(torch.zeros(1))
    assert calls["compiled"] == 2


def test_regional_vae_compile_failure_falls_back_and_settles_status():
    def factory(m):
        def compiled(*a, **k):
            raise _inductor_error("RecursionError: maximum recursion depth exceeded")

        return compiled

    vae = _RegionalVae(factory)
    pipe = types.SimpleNamespace(vae = vae)
    assert ds_mod._compile_vae_decode(pipe, None) is True
    assert vae.decode(torch.zeros(1)).item() == 2  # eager answer, same value
    state = types.SimpleNamespace(speed_optims = ("compiled", "compiled_vae_decode"))
    reason = ds_mod.settle_compile_fallback(state, pipe)
    assert "RecursionError" in reason or "BackendCompilerFailed" in reason  # torch < 2.7 wraps it
    assert "compiled_vae_decode" not in state.speed_optims
    assert "compile_fallback_eager" in state.speed_optims
    # a later apply_speed_optims pass does not retry the broken lowering
    assert ds_mod._compile_vae_decode(pipe, None) is False
    assert all(b._compiled_call_impl is None for b in vae.blocks)


def test_vae_without_repeated_blocks_keeps_the_whole_decode_path(monkeypatch):
    seen = {}

    def fake_compile(fn, **kwargs):
        seen.update(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class _PlainVae:
        def decode(self, z):
            return z

    vae = _PlainVae()
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None) is True
    assert seen == {"fullgraph": False, "dynamic": True}
    assert "decode" in vae.__dict__


def test_empty_repeated_blocks_is_not_regional():
    class _Empty(_RegionalVae):
        _repeated_blocks = []

    assert ds_mod._vae_declares_repeated_blocks(_Empty()) is False
    assert ds_mod._vae_declares_repeated_blocks(_RegionalVae()) is True
    assert ds_mod._vae_declares_repeated_blocks(None) is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or not ds_mod.torch_compile_runtime_available(),
    reason = "needs a GPU and a working inductor (Windows ROCm has no Triton)",
)
def test_tiny_minimax_h3_vae_decode_compiles_once_per_tile_shape():
    """Real diffusers H3 VAE, tiny config: the tiled decode used to be one unrolled graph; now one block graph."""
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLMiniMaxH3", None)
    if cls is None:
        pytest.skip("diffusers without MiniMax-H3")
    from torch._dynamo.utils import counters

    torch.manual_seed(0)
    vae = (
        cls(
            block_out_channels = (8, 8, 8, 8, 8, 8),
            decoder_num_layers = 2,
            decoder_num_attention_heads = 2,
            decoder_attention_head_dim = 16,
            norm_num_groups = 4,
        )
        .cuda()
        .eval()
    )
    # 2 x 2 tiles, 2 temporal chunks
    z = torch.randn(1, 24, 7, 24, 24, device = "cuda")
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0]
    torch._dynamo.reset()
    counters.clear()
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None) is True
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]
        out2 = vae.decode(z, return_dict = False)[0]
    assert ds_mod._vae_compile_error(vae) is None
    assert counters["stats"]["unique_graphs"] == 1
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol = 2e-3, rtol = 2e-3)
    torch.testing.assert_close(out2, out)
    torch._dynamo.reset()


def test_a_vae_whose_blocks_the_fast_decoder_bypasses_is_not_compiled_even_when_forced(monkeypatch):
    # MiniMax-H3's fused decoder reads block weights and never calls the blocks: compiling them would report
    # compiled_vae_decode for a compile that never runs.
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    vae = _RegionalVae()
    pipe = types.SimpleNamespace(vae = vae)
    assert ds_mod._vae_decode_compile_allowed(pipe, ds_mod.SPEED_DEFAULT) is True
    vae._unsloth_decode_blocks_bypassed = True
    for tier in (ds_mod.SPEED_DEFAULT, ds_mod.SPEED_MAX):
        assert ds_mod._vae_decode_compile_allowed(pipe, tier) is False
        assert ds_mod.vae_decode_compile_allowed(pipe, tier) is False
