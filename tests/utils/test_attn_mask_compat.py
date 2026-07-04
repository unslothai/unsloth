# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the GNU Lesser General Public License v3.0 or later.

"""Equivalence tests for local attention-mask compat helpers (issue #6860)."""

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPAT_PATH = _REPO_ROOT / "unsloth" / "models" / "_attn_mask_compat.py"


def _load_compat_module():
    module_name = "unsloth.models._attn_mask_compat"
    spec = importlib.util.spec_from_file_location(module_name, _COMPAT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compat = _load_compat_module()


def test_no_deprecation_warning_on_causal_mask():
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        compat.AttentionMaskConverter(is_causal = True, sliding_window = 3).to_causal_4d(
            1,
            8,
            8,
            dtype = torch.float16,
        )
    assert not any(
        issubclass(w.category, FutureWarning) and "modeling_attn_mask_utils" in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize("sliding_window", [None, 3, 5])
def test_causal_4d_matches_transformers(batch_size, query_length, sliding_window):
    try:
        legacy = importlib.import_module("transformers.modeling_attn_mask_utils")
    except ImportError:
        pytest.skip("transformers.modeling_attn_mask_utils unavailable")

    key_value_length = query_length
    dtype = torch.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = legacy.AttentionMaskConverter(
            is_causal = True,
            sliding_window = sliding_window,
        ).to_causal_4d(
            batch_size,
            query_length,
            key_value_length,
            dtype = dtype,
        )

    actual = compat.AttentionMaskConverter(
        is_causal = True,
        sliding_window = sliding_window,
    ).to_causal_4d(
        batch_size,
        query_length,
        key_value_length,
        dtype = dtype,
    )

    if expected is None:
        assert actual is None
    else:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "attention_mask,past_length",
    [
        (None, 0),
        (None, 4),
        (torch.ones(2, 5), 0),
        (torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]), 0),
    ],
)
def test_prepare_4d_causal_attention_mask_for_sdpa_matches_transformers(
    attention_mask, past_length
):
    try:
        legacy = importlib.import_module("transformers.modeling_attn_mask_utils")
    except ImportError:
        pytest.skip("transformers.modeling_attn_mask_utils unavailable")

    batch_size = 2 if attention_mask is not None else 1
    query_length = 5
    inputs_embeds = torch.zeros(batch_size, query_length, 16, dtype = torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = legacy._prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, query_length),
            inputs_embeds,
            past_length,
            sliding_window = 3,
        )

    actual = compat._prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask,
        (batch_size, query_length),
        inputs_embeds,
        past_length,
        sliding_window = 3,
    )

    if expected is None:
        assert actual is None
    else:
        assert torch.equal(actual, expected)


def test_prepare_4d_attention_mask_for_sdpa_matches_transformers():
    try:
        legacy = importlib.import_module("transformers.modeling_attn_mask_utils")
    except ImportError:
        pytest.skip("transformers.modeling_attn_mask_utils unavailable")

    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype = torch.float32)
    dtype = torch.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = legacy._prepare_4d_attention_mask_for_sdpa(mask, dtype = dtype)

    actual = compat._prepare_4d_attention_mask_for_sdpa(mask, dtype = dtype)

    if expected is None:
        assert actual is None
    else:
        assert torch.equal(actual, expected)


def test_repo_has_no_direct_deprecated_imports():
    model_dir = _REPO_ROOT / "unsloth" / "models"
    offenders = []
    for path in model_dir.glob("*.py"):
        if path.name == "_attn_mask_compat.py":
            continue
        text = path.read_text(encoding = "utf-8")
        if "transformers.modeling_attn_mask_utils" in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == []
