# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Equivalence tests for local attention-mask compat helpers (issue #6860)."""

import importlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from unittest import mock

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize(
    "attention_mask,past_length",
    [
        (None, 8),
        (None, 0),
        ("ones", 8),
        ("left_pad", 8),
    ],
)
def test_sdpa_mask_matches_transformers_on_cuda(attention_mask, past_length):
    """CUDA counterpart of the test above.

    `_unmask_unattended` is gated on ``device.type in ("cuda", "xpu")``, so a
    CPU-only comparison never reaches it. Asserts stride and storage too: an
    expanded view materialised into a dense [bsz, 1, q, kv] tensor is a memory
    regression even when every element compares equal.
    """
    try:
        legacy = importlib.import_module("transformers.modeling_attn_mask_utils")
    except ImportError:
        pytest.skip("transformers.modeling_attn_mask_utils unavailable")

    batch_size, query_length = 4, 5
    key_value_length = query_length + past_length
    inputs_embeds = torch.zeros(
        batch_size,
        query_length,
        16,
        dtype = torch.float32,
        device = "cuda",
    )
    if attention_mask == "ones":
        mask = torch.ones(batch_size, key_value_length, dtype = torch.int64, device = "cuda")
    elif attention_mask == "left_pad":
        mask = torch.ones(batch_size, key_value_length, dtype = torch.int64, device = "cuda")
        mask[:, :2] = 0
    else:
        mask = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = legacy._prepare_4d_causal_attention_mask_for_sdpa(
            mask,
            (batch_size, query_length),
            inputs_embeds,
            past_length,
            sliding_window = 3,
        )
    actual = compat._prepare_4d_causal_attention_mask_for_sdpa(
        mask,
        (batch_size, query_length),
        inputs_embeds,
        past_length,
        sliding_window = 3,
    )

    if expected is None:
        assert actual is None
        return

    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride(), (
        f"layout diverged: {actual.stride()} vs upstream {expected.stride()} — "
        "an expanded view was materialised"
    )
    assert actual.untyped_storage().nbytes() == expected.untyped_storage().nbytes(), (
        f"allocation diverged: {actual.untyped_storage().nbytes()} bytes vs "
        f"upstream {expected.untyped_storage().nbytes()}"
    )


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


def test_import_falls_back_when_is_tracing_missing():
    """Regression for Codex review on PR #6880.

    The compat module imports `is_tracing` from `transformers.utils.import_utils`,
    but that symbol is only exported from transformers >= 5.0.0. It is absent
    from every 4.x release, including the declared `transformers>=4.51.3` floor
    and the 4.57.6 pin used by tests/version_compat, so the fallback below is
    the live path across the whole 4.x half of the supported range.

    Reload the module with `is_tracing` removed from the namespace and confirm
    the local fallback is used. The fallback must mirror the legacy
    `transformers==4.51.3` inline expression
    (``torch.jit.is_tracing() or isinstance(tensor, torch.fx.Proxy) or
    is_torchdynamo_compiling()``) so the data-dependent ``torch.all(...)``
    branches in the mask helpers continue to be skipped during JIT trace,
    symbolic trace, and Dynamo compilation — otherwise tracing/exporting
    these models on transformers 4.51.x either fails on proxy control flow
    or bakes the wrong SDPA causal-mask path.
    """
    fake_import_utils = types.ModuleType("transformers.utils.import_utils")

    def _is_torchdynamo_compiling() -> bool:
        return False

    fake_import_utils.is_torchdynamo_compiling = _is_torchdynamo_compiling

    # Ensure both the leaf and the parent's `transformers.utils` package resolve to our stub so the `from ...
    # import is_tracing` inside the compat module body raises ImportError as it would on transformers < 4.52.
    # We re-use `transformers.utils` if it's already in sys.modules (so we don't disturb the rest of the test suite),
    existing_utils_pkg = sys.modules.get("transformers.utils")
    with mock.patch.dict(
        sys.modules,
        {"transformers.utils.import_utils": fake_import_utils},
    ):
        reloaded = _load_compat_module()

    assert existing_utils_pkg is not None, (
        "transformers.utils was not pre-imported; stubbing the leaf alone "
        "would not exercise the fallback path"
    )

    assert reloaded.is_tracing() is False
    assert reloaded.is_tracing(torch.zeros(1)) is False

    # ``torch.fx.Proxy`` should be detected even when Dynamo is idle, since symbolic_trace / export-only paths don't go
    # through dynamo.
    # Construct the Proxy from a real fx.Graph node (passing a Tensor directly to ``Proxy(...)`` is a common foot-gun
    fx_graph = torch.fx.Graph()
    fx_node = fx_graph.create_node("call_function", torch.zeros, (torch.zeros(1).shape,))
    proxy = torch.fx.Proxy(fx_node)
    assert reloaded.is_tracing(proxy) is True

    # ``torch.jit.is_tracing()`` should be detected via patch.
    with mock.patch("torch.jit.is_tracing", return_value = True):
        assert reloaded.is_tracing() is True

