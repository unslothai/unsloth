# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-block-cache metadata Unsloth registers for blocks diffusers has not.

Separate from ``test_diffusion_cache.py``, which stubs ``diffusers`` through ``sys.modules``: this
one needs the REAL registry, since the thing under test is whether a class the installed diffusers
ships is known to it.
"""

from __future__ import annotations

import pytest

from core.inference import diffusion_cache as dc


def _registry_and_block():
    """The real registry plus the 2.1 block, or a skip.

    ``importorskip`` is not enough: importing diffusers drags in optional integrations, and a host
    whose bitsandbytes cannot find CUDA raises RuntimeError rather than ImportError. That is an
    environment fact, not a result.
    """
    try:
        from diffusers.hooks._helpers import TransformerBlockRegistry
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21TransformerBlock,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    return TransformerBlockRegistry, QwenImage21TransformerBlock


def test_the_qwen_image_21_block_is_registered_for_step_caching():
    """diffusers ships no metadata for ``QwenImage21TransformerBlock``, so ``enable_cache`` raised
    "Model class ... not registered." and every load of the family rendered uncached, which is the
    whole step-cache saving gone on a 20+ step model and only visible in a log line.

    The block is single stream: it takes ``hidden_states, modulation, rotary_emb, ...`` and returns
    the hidden states alone, hence 0 and None.
    """
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
    """LTX-2 and HunyuanVideo-1.5 ship blocks diffusers never registered, so enable_cache raises
    on every attempt. The probe must say so up front, or auto keeps the per-generation toggle live
    (fullgraph dropped, a warning per render) on a model that can never cache."""
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
