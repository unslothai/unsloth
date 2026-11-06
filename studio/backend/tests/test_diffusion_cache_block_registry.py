# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks Unsloth's first-block-cache metadata against the real diffusers registry, not a stub."""

from __future__ import annotations

import pytest

from core.inference import diffusion_cache as dc


def _registry_and_block():
    """Skip instead of importorskip: a host whose bitsandbytes cannot find CUDA raises RuntimeError."""
    try:
        from diffusers.hooks._helpers import TransformerBlockRegistry
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21TransformerBlock,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    return TransformerBlockRegistry, QwenImage21TransformerBlock


def test_the_qwen_image_21_block_is_registered_for_step_caching():
    """No metadata for QwenImage21TransformerBlock: enable_cache raised, so the family ran uncached."""
    registry, block = _registry_and_block()

    dc.register_unregistered_transformer_blocks()
    meta = registry.get(block)
    assert meta.return_hidden_states_index == 0
    assert meta.return_encoder_hidden_states_index is None


def test_registration_is_idempotent_and_never_overwrites_diffusers_own():
    """Upstream's metadata is authoritative; ours fills a gap until it lands, so a class diffusers
    registers itself must be left exactly as it is, however many times this runs."""
    registry, _ = _registry_and_block()
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    dc.register_unregistered_transformer_blocks()
    before = registry.get(QwenImageTransformerBlock)
    dc.register_unregistered_transformer_blocks()
    assert registry.get(QwenImageTransformerBlock) is before


def test_step_cache_probe_refuses_unregistered_blocks():
    registry, _ = _registry_and_block()
    import torch
    from diffusers.hooks._helpers import TransformerBlockMetadata

    class _Block(torch.nn.Module):
        def forward(self, x):
            return x

    class _Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])

    assert not dc._transformer_blocks_registered(_Transformer())
    registry.register(model_class = _Block, metadata = TransformerBlockMetadata(0, None))
    try:
        assert dc._transformer_blocks_registered(_Transformer())
    finally:
        registry._registry.pop(_Block, None)
