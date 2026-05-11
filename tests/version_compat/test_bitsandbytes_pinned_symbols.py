# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across bitsandbytes PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - bnb 0.46.0 release was broken (in pyproject.toml as `!=0.46.0`).
    Don't test against it.
  - bnb 0.48.0 release was broken (also `!=0.48.0`). Same.
  - bnb 0.45 series introduced fp4 + nf4 paged optimisers; unsloth-zoo
    expects bnb.functional.dequantize_4bit + bnb.nn.Linear4bit /
    Params4bit to remain stable from this point onward.
  - vLLM bitsandbytes-loader patches in unsloth_zoo/vllm_utils.py:
    apply_bnb_4bit (line 237), is_layer_skipped_bnb (line 281),
    BitsAndBytesLinearMethod._apply_4bit_weight (line 282) — these
    live in vllm.* but they call into bnb's public surface.

Strategy: GitHub raw fetch + symbol grep. CPU-only, no install.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# pyproject pin: bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0
# Test floor + each safe minor since.
BNB_TAGS = [
    "0.45.5",
    "0.47.0",  # skip 0.46.0 (broken)
    "0.49.2",  # skip 0.48.0 (broken)
    "main",
]


# -------------------------------------------------------------------------
# bnb.functional: dequantize_4bit / quantize_4bit are the public 4-bit
# surface unsloth's compiled kernels and unsloth-zoo's vllm_utils
# bnb-loader patches all call into.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_functional_4bit(tag: str):
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    assert (
        hit is not None
    ), f"{tag}: bitsandbytes/functional[.py|/__init__.py] both missing"
    _, src = hit
    needed = ("dequantize_4bit", "quantize_4bit")
    missing = [n for n in needed if not has_def(src, n, "func") and n not in src]
    assert not missing, (
        f"{tag}: bnb.functional missing {missing}; "
        f"unsloth-zoo dequant kernels rely on these"
    )


# -------------------------------------------------------------------------
# bnb.nn.Linear4bit / Params4bit: the two classes peft and unsloth
# isinstance-check against. Renaming either silently breaks 4-bit LoRA.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_nn_linear4bit_classes(tag: str):
    candidates = [
        "bitsandbytes/nn/modules.py",
        "bitsandbytes/nn/__init__.py",
    ]
    found_linear = False
    found_params = False
    for p in candidates:
        src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, p)
        if src is None:
            continue
        if has_def(src, "Linear4bit", "class") or "Linear4bit" in src:
            found_linear = True
        if has_def(src, "Params4bit", "class") or "Params4bit" in src:
            found_params = True
        if found_linear and found_params:
            return
    pytest.fail(
        f"{tag}: Linear4bit={found_linear} Params4bit={found_params} "
        f"in {candidates}; unsloth + peft 4-bit isinstance checks fail"
    )


# =========================================================================
# Coverage extension (added 2026-05): every bnb symbol unsloth +
# unsloth-zoo touch, derived from a full grep of both repos.
# =========================================================================


# -------------------------------------------------------------------------
# Top-level convenience export. unsloth/kernels/utils.py + unsloth-zoo
# vllm_utils.py call `bnb.matmul_4bit(x, w, bias=, quant_state=)`.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_matmul_4bit_top_level(tag: str):
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/__init__.py"
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/__init__.py missing")
    assert "matmul_4bit" in src, (
        f"{tag}: bitsandbytes.matmul_4bit not exported at package root; "
        f"unsloth/kernels/utils.py + zoo/temporary_patches/moe call paths break"
    )


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_functional_4bit_kernel_path(tag: str):
    """unsloth/kernels/utils.py module-top binds the 4-bit dequantize
    and gemm primitives via one of two paths:
      - LEGACY (bnb <= 0.48.x): `bnb.functional.lib.cdequantize_blockwise_*`
        and `bnb.functional.lib.cgemm_4bit_inference_naive_*` — C
        symbols listed in functional.py source.
      - NEW (bnb >= 0.49.0): `torch.ops.bitsandbytes.dequantize_blockwise`
        and `torch.ops.bitsandbytes.dequantize_4bit` Python wrappers;
        the C symbols still live in libbitsandbytes_*.so but the
        Python source no longer references them by name.
    Either path lets unsloth resolve the kernels at runtime — we only
    fail if NEITHER signal is present."""
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: bitsandbytes/functional missing")
    _, src = hit
    legacy_path = "cdequantize_blockwise" in src and "cgemm_4bit_inference" in src
    new_path = (
        "dequantize_blockwise" in src
        and ("dequantize_4bit" in src or "dequantize_nf4" in src)
        and "torch.ops.bitsandbytes" in src
    )
    assert legacy_path or new_path, (
        f"{tag}: bnb.functional has NEITHER legacy `lib.cdequantize_*` "
        f"NOR new `torch.ops.bitsandbytes.*` kernel path; "
        f"unsloth/kernels/utils.py module-top binding will AttributeError"
    )


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_functional_get_ptr(tag: str):
    """unsloth/kernels/utils.py top-level: `get_ptr = bnb.functional.get_ptr`."""
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: functional missing")
    _, src = hit
    assert has_def(src, "get_ptr", "func") or "get_ptr" in src, (
        f"{tag}: bnb.functional.get_ptr missing; "
        f"unsloth/kernels/utils.py module-top ImportError"
    )


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_quantstate_from_dict(tag: str):
    """unsloth-zoo monkey-patches `QuantState.from_dict = ...`. Both
    the class AND the classmethod must be present for the rebinding
    to take effect."""
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: functional missing")
    _, src = hit
    assert has_def(
        src, "QuantState", "class"
    ), f"{tag}: bnb.functional.QuantState missing"
    assert "from_dict" in src, (
        f"{tag}: QuantState.from_dict missing; "
        f"unsloth-zoo monkey-patch silently no-ops"
    )


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_nn_modules_fix_4bit_weight_optional(tag: str):
    """fix_4bit_weight_quant_state_from_module added in newer bnb;
    unsloth uses getattr() with a fallback so older versions are OK."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/nn/modules.py"
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/nn/modules.py missing")
    if "fix_4bit_weight_quant_state_from_module" not in src:
        pytest.skip(f"{tag}: helper not yet added (OK; getattr fallback)")


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_nn_linear8bitlt(tag: str):
    """unsloth/__init__ probes both Linear4bit AND Linear8bitLt."""
    candidates = [
        "bitsandbytes/nn/modules.py",
        "bitsandbytes/nn/__init__.py",
    ]
    for p in candidates:
        src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, p)
        if src and (has_def(src, "Linear8bitLt", "class") or "Linear8bitLt" in src):
            return
    pytest.fail(
        f"{tag}: bnb.nn.Linear8bitLt missing in {candidates}; "
        f"legacy load_in_8bit path breaks"
    )


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_optim_optimizer2state(tag: str):
    """PagedAdamW32bit + 8bit optimisers subclass Optimizer2State."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes",
        tag,
        "bitsandbytes/optim/optimizer.py",
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/optim/optimizer.py missing")
    assert has_def(
        src, "Optimizer2State", "class"
    ), f"{tag}: bnb.optim.optimizer.Optimizer2State missing"


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_utils_pack_unpack(tag: str):
    """4bit state-dict save/load uses these two helpers."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/utils.py"
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/utils.py missing")
    for name in ("pack_dict_to_tensor", "unpack_tensor_to_dict"):
        assert (
            has_def(src, name, "func") or name in src
        ), f"{tag}: bnb.utils.{name} missing"


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_cextension_rocm_warp_size_optional(tag: str):
    """ROCM_WARP_SIZE_64 added with AMD ROCm support; pre-ROCm bnb
    builds don't have it. unsloth probes via try/except — informational."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/cextension.py"
    )
    if src is None:
        pytest.skip(f"{tag}: cextension.py missing")
    if "ROCM_WARP_SIZE_64" not in src:
        pytest.skip(f"{tag}: ROCM_WARP_SIZE_64 not yet defined (pre-ROCm bnb)")


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_autograd_functions_matmul_4bit(tag: str):
    """unsloth-zoo has a dynamo-disable patch site for
    bnb.autograd._functions.matmul_4bit. Symbol must remain so the
    probe + decision logic works."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes",
        tag,
        "bitsandbytes/autograd/_functions.py",
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/autograd/_functions.py missing")
    assert "matmul_4bit" in src, f"{tag}: bnb.autograd._functions.matmul_4bit missing"


@pytest.mark.parametrize("tag", BNB_TAGS)
def test_bnb_version_parseable(tag: str):
    """Multiple unsloth code paths read Version(bnb.__version__) for
    feature gating (floors 0.43.3, 0.46.0, 0.48.2.dev0, 0.49.0,
    0.49.2). At least one export mechanism must work."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/__init__.py"
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/__init__.py missing")
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(
        re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE)
    )
    has_metadata = bool(
        re.search(
            r"^from\s+importlib\.metadata\s+import\s+(?:[\w,\s]+,\s*)?version",
            src,
            re.MULTILINE,
        )
        and re.search(r"^\s*__version__\s*=\s*version\s*\(", src, re.MULTILINE)
    )
    has_version_attr = "__version__" in src
    assert (
        has_literal or has_subimport or has_metadata or has_version_attr
    ), f"{tag}: bnb.__version__ not exported"
