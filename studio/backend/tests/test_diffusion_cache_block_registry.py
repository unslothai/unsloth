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


def test_the_qwen_image_21_block_is_left_unregistered():
    """Registering it as (0, None) reads right from the block's return alone and made every
    generation of the family fail at the first step: the transformer passes a JOINT text+image
    sequence, and the cond and uncond passes carry different text lengths, so FBCache's residual
    comparison hits a length mismatch. Uncached costs the step-cache saving; wrongly registered
    cannot generate at all."""
    registry, block = _registry_and_block()

    assert "QwenImage21TransformerBlock" not in dc.register_unregistered_transformer_blocks()
    with pytest.raises(ValueError, match = "not registered"):
        registry.get(block)


def test_registration_is_idempotent_and_never_overwrites_diffusers_own():
    """Upstream's metadata is authoritative; ours fills a gap until it lands, so a class diffusers
    registers itself must be left exactly as it is, however many times this runs."""
    registry, _ = _registry_and_block()
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    dc.register_unregistered_transformer_blocks()
    before = registry.get(QwenImageTransformerBlock)
    dc.register_unregistered_transformer_blocks()
    assert registry.get(QwenImageTransformerBlock) is before
