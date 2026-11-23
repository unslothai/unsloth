# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-Block-Cache on FLUX.2, whose dual-stream head feeds a single-stream tail.

Uses the REAL diffusers hooks on tiny random transformers (CPU, seconds). Stock diffusers raises
"unsupported operand type(s) for -: 'Tensor' and 'list'" in the tail hook on the first cached step.
"""

from __future__ import annotations

import pytest

from core.inference import diffusion_cache as dc


def _diffusers():
    # Importing diffusers drags in optional integrations that can raise non-ImportErrors.
    try:
        import torch
        from diffusers.hooks import FirstBlockCacheConfig
        from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
        from diffusers.hooks.first_block_cache import FBCBlockHook
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    return (
        torch,
        FirstBlockCacheConfig,
        TransformerBlockMetadata,
        TransformerBlockRegistry,
        FBCBlockHook,
    )


@pytest.fixture
def stock_tail_hook(monkeypatch):
    """Start from diffusers' own tail hook and put back whatever was there afterwards."""
    *_, FBCBlockHook = _diffusers()
    current = FBCBlockHook.new_forward
    stock = getattr(current, "_unsloth_original", current)
    monkeypatch.setattr(FBCBlockHook, "new_forward", stock)
    return stock


def _tiny_flux2():
    torch, _, TransformerBlockMetadata, TransformerBlockRegistry, _ = _diffusers()
    try:
        from diffusers import Flux2Transformer2DModel
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2SingleTransformerBlock,
            Flux2TransformerBlock,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this diffusers has no FLUX.2: {type(exc).__name__}")
    # The metadata diffusers main registers; older releases ship FLUX.2 without it.
    for cls, hidden, encoder in (
        (Flux2TransformerBlock, 1, 0),
        (Flux2SingleTransformerBlock, 0, None),
    ):
        try:
            TransformerBlockRegistry.get(cls)
        except Exception:  # noqa: BLE001
            TransformerBlockRegistry.register(
                model_class = cls,
                metadata = TransformerBlockMetadata(
                    return_hidden_states_index = hidden,
                    return_encoder_hidden_states_index = encoder,
                ),
            )
    torch.manual_seed(0)
    model = Flux2Transformer2DModel(
        in_channels = 8,
        num_layers = 2,
        num_single_layers = 3,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        timestep_guidance_channels = 32,
        axes_dims_rope = (4, 4, 4, 4),
        guidance_embeds = False,
    ).eval()
    img_ids = torch.zeros(16, 4)
    img_ids[:, 1] = torch.arange(16) // 4
    img_ids[:, 2] = torch.arange(16) % 4
    txt_ids = torch.zeros(6, 4)
    txt_ids[:, 3] = torch.arange(6)
    inputs = dict(
        hidden_states = torch.randn(1, 16, 8),
        encoder_hidden_states = torch.randn(1, 6, 24),
        img_ids = img_ids,
        txt_ids = txt_ids,
    )
    return model, inputs


def _tiny_flux1():
    torch, *_ = _diffusers()
    try:
        from diffusers import FluxTransformer2DModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this diffusers has no FLUX.1: {type(exc).__name__}")
    torch.manual_seed(0)
    model = FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 8,
        num_layers = 2,
        num_single_layers = 3,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        pooled_projection_dim = 12,
        axes_dims_rope = (4, 6, 6),
    ).eval()
    inputs = dict(
        hidden_states = torch.randn(1, 16, 8),
        encoder_hidden_states = torch.randn(1, 6, 24),
        pooled_projections = torch.randn(1, 12),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(6, 3),
    )
    return model, inputs


def _render(
    model,
    inputs,
    timesteps,
    *,
    threshold = None,
):
    """One forward per timestep inside a cache context, FBCache on when ``threshold`` is set."""
    torch, FirstBlockCacheConfig, *_ = _diffusers()
    if threshold is not None:
        model.enable_cache(FirstBlockCacheConfig(threshold = threshold))
        # diffusers 0.39/0.40 cache the child-registry list across renders, as apply_step_cache does
        dc._invalidate_child_registry_cache(model)
    tail_attn = model.single_transformer_blocks[-1].attn
    calls = []
    handle = tail_attn.register_forward_hook(lambda *_: calls.append(1))
    outs = []
    try:
        with torch.no_grad(), model.cache_context("cond"):
            for t in timesteps:
                outs.append(model(**inputs, timestep = torch.tensor([t]), return_dict = False)[0])
    finally:
        handle.remove()
        if threshold is not None:
            model.disable_cache()
    return outs, len(calls)


def test_stock_fbcache_on_flux2_fails_with_tensor_minus_list(stock_tail_hook):
    """The bug itself, pinned so a diffusers that fixes it upstream is noticed."""
    model, inputs = _tiny_flux2()
    with pytest.raises(TypeError, match = "'Tensor' and 'list'"):
        _render(model, inputs, (1.0, 1.0), threshold = 0.5)


def test_fbcache_on_flux2_skips_and_matches_the_full_pass(stock_tail_hook):
    torch, *_ = _diffusers()
    assert dc.install_fbcache_mixed_stream_tail() is True
    assert dc.install_fbcache_mixed_stream_tail() is True  # idempotent
    model, inputs = _tiny_flux2()
    steps = (1.0, 1.0, 1.0)
    full, full_calls = _render(model, inputs, steps)
    cached, cached_calls = _render(model, inputs, steps, threshold = 0.5)
    assert full_calls == 3
    # Same input every step: the head residual does not move, so steps 2 and 3 skip the stack.
    assert cached_calls == 1
    assert torch.equal(full[0], cached[0])
    # A skip reapplies the stored tail residual, which must rebuild the full pass's output.
    for a, b in zip(full[1:], cached[1:]):
        torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)


def test_fbcache_on_flux2_recomputes_when_the_residual_moves(stock_tail_hook):
    dc.install_fbcache_mixed_stream_tail()
    model, inputs = _tiny_flux2()
    _, calls = _render(model, inputs, (1.0, 0.5, 0.1), threshold = 1e-9)
    assert calls == 3


def test_unmappable_tail_never_skips(stock_tail_hook):
    """A tail output that is not the joint [encoder, hidden] sequence cannot seed a skip."""
    torch, *_ = _diffusers()
    hidden, encoder = torch.zeros(1, 4, 8), torch.zeros(1, 3, 8)
    assert dc._split_mixed_stream_tail_residuals(torch.zeros(1, 9, 8), [hidden, encoder]) is None
    assert dc._split_mixed_stream_tail_residuals(torch.zeros(1, 7, 8), [hidden, None]) is None
    res_hidden, res_encoder = dc._split_mixed_stream_tail_residuals(
        torch.arange(56.0).reshape(1, 7, 8), (hidden, encoder)
    )
    assert torch.equal(res_hidden, torch.arange(24.0, 56.0).reshape(1, 4, 8))
    assert torch.equal(res_encoder, torch.arange(24.0).reshape(1, 3, 8))


def test_working_family_output_is_byte_identical(stock_tail_hook):
    """FLUX.1 (dual-stream tuple head and tail) must not see the patch at all."""
    torch, *_ = _diffusers()
    steps = (1.0, 1.0, 0.5)  # step 2 skips, step 3 computes
    model, inputs = _tiny_flux1()
    before, before_calls = _render(model, inputs, steps, threshold = 0.05)
    dc.install_fbcache_mixed_stream_tail()
    model, inputs = _tiny_flux1()
    after, after_calls = _render(model, inputs, steps, threshold = 0.05)
    assert before_calls == after_calls == 2
    for a, b in zip(before, after):
        assert torch.equal(a, b)


class _Flux2Pipe:
    """Stands in for Flux2KleinPipeline: a denoise loop that opens ``cache_context``."""

    def __init__(self, transformer, inputs):
        self.transformer = transformer
        self.inputs = inputs

    def __call__(self, timesteps):
        import torch

        outs = []
        with torch.no_grad():
            for t in timesteps:
                with self.transformer.cache_context("cond"):
                    outs.append(
                        self.transformer(
                            **self.inputs, timestep = torch.tensor([t]), return_dict = False
                        )[0]
                    )
        return outs


def test_apply_step_cache_on_flux2_renders(stock_tail_hook):
    """The Studio entry point: FBCache engages on FLUX.2 and the cached render completes."""
    torch, *_ = _diffusers()
    model, inputs = _tiny_flux2()
    pipe = _Flux2Pipe(model, inputs)
    steps = (1.0, 1.0, 1.0)
    full = pipe(steps)
    assert dc.apply_step_cache(pipe, mode = "fbcache", threshold = 0.5) == "fbcache"
    try:
        cached = pipe(steps)
    finally:
        model.disable_cache()
    assert torch.equal(full[0], cached[0])
    for a, b in zip(full[1:], cached[1:]):
        torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)


def test_apply_step_cache_adopts_a_cache_enabled_before_the_patch(stock_tail_hook):
    """An FBCache enabled with the stock tail hook (same threshold, so apply_step_cache adopts it rather than
    re-enabling) must run the patched tail too, not the bound stock method diffusers captured at registration."""
    torch, FirstBlockCacheConfig, *_ = _diffusers()
    model, inputs = _tiny_flux2()
    pipe = _Flux2Pipe(model, inputs)
    steps = (1.0, 1.0, 1.0)
    full = pipe(steps)
    model.enable_cache(FirstBlockCacheConfig(threshold = 0.5))  # hooks bound to the stock tail
    try:
        assert dc.apply_step_cache(pipe, mode = "fbcache", threshold = 0.5) == "fbcache"
        cached = pipe(steps)
    finally:
        model.disable_cache()
    assert torch.equal(full[0], cached[0])
    for a, b in zip(full[1:], cached[1:]):
        torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)
