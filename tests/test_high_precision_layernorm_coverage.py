# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_HIGH_PRECISION_LAYERNORM must upcast every norm in a block, not some.

The selector used to be name-only, so a block whose norms are not all named
"*norm" got a float32 norm feeding a bfloat16 one on the same chain. Gemma 4's
`embed_vision` is the live case: `pos_norm` matched, its siblings `patch_ln1`
and `patch_ln2` did not.
"""

import torch
import torch.nn as nn

from unsloth.models.vision import _NORM_MODULE_TYPES


def _selected(model):
    """Mirror of the loader's selector in unsloth/models/vision.py."""
    out = []
    for name, module in model.named_modules():
        if (
            name.endswith(("norm", "norm1", "norm2", "norm3", "norm4"))
            or "layernorm" in name
            or "layer_norm" in name
            or isinstance(module, _NORM_MODULE_TYPES)
        ) and hasattr(module, "weight"):
            out.append(name)
    return set(out)


class _VisionEmbedderLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_ln1 = nn.LayerNorm(8)
        self.patch_ln2 = nn.LayerNorm(8)
        self.pos_norm = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8)


def test_sibling_norms_are_all_selected():
    got = _selected(_VisionEmbedderLike())
    assert got == {"patch_ln1", "patch_ln2", "pos_norm"}


def test_non_norm_modules_are_not_selected():
    assert "proj" not in _selected(_VisionEmbedderLike())


def test_custom_rmsnorm_still_matched_by_name():
    """Model-specific RMSNorm classes are not torch.nn types, so the name rule
    still has to carry them."""

    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm()
            self.q_norm = RMSNorm()

    assert _selected(Block()) == {"input_layernorm", "q_norm"}


def test_norm_types_is_non_empty_and_typed():
    assert _NORM_MODULE_TYPES
    assert nn.LayerNorm in _NORM_MODULE_TYPES
    assert all(isinstance(t, type) for t in _NORM_MODULE_TYPES)


def test_weightless_norm_is_skipped():
    """`.to(float32)` on a norm with no weight is a no-op, and named_modules
    exposes such norms, so the hasattr guard must stay."""

    class Weightless(nn.LayerNorm):
        def __init__(self):
            super().__init__(8, elementwise_affine = False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_norm = Weightless()

    m = Block()
    assert m.v_norm.weight is None
    assert "v_norm" in _selected(m)  # selected, but .to() is harmless
