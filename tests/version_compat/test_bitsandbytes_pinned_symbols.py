# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across bitsandbytes minor versions via GitHub raw fetch + symbol grep."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


def _super_init_call(source: str, class_name: str) -> ast.Call | None:
    """The ``super().__init__(...)`` Call node inside ``class_name``'s __init__.

    Parsed rather than grepped: a substring search over the class body also sees the
    names quoted in the legacy_kwargs table, so it passes whether or not the real call
    uses keywords.
    """
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "__init__"
            ):
                return sub
    return None


def first_match_signature(src: str, class_name: str) -> str | None:
    """The text of ``class_name``'s __init__ parameter list, or None."""
    at = src.find(f"class {class_name}")
    if at == -1:
        return None
    at = src.find("def __init__", at)
    if at == -1:
        return None
    depth = 0
    for i in range(src.index("(", at), len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[src.index("(", at) : i + 1]
    return None


# pyproject pin: bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0 Test floor + each safe minor since.
BNB_TAGS = [
    "0.45.5",
    "0.47.0",  # skip 0.46.0 (broken)
    "0.49.2",  # skip 0.48.0 (broken)
    "main",
]

# Every check runs once per tag; one that cannot skips from inside so the tag stays in the report.
pytestmark = pytest.mark.parametrize("tag", BNB_TAGS)


# bnb.functional dequantize_4bit / quantize_4bit: the public 4-bit surface unsloth kernels call into.
def test_bnb_functional_4bit(tag: str):
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    assert hit is not None, f"{tag}: bitsandbytes/functional[.py|/__init__.py] both missing"
    _, src = hit
    needed = ("dequantize_4bit", "quantize_4bit")
    missing = [n for n in needed if not has_def(src, n, "func") and n not in src]
    assert not missing, (
        f"{tag}: bnb.functional missing {missing}; " f"unsloth-zoo dequant kernels rely on these"
    )


# bnb.nn.Linear4bit / Params4bit: peft + unsloth isinstance-check these; renaming breaks 4-bit LoRA.
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


# Coverage extension (2026-05):
# Top-level export: unsloth/kernels/utils.py + zoo vllm_utils.py call bnb.matmul_4bit(...).
def test_bnb_matmul_4bit_top_level(tag: str):
    src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/__init__.py")
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/__init__.py missing")
    assert "matmul_4bit" in src, (
        f"{tag}: bitsandbytes.matmul_4bit not exported at package root; "
        f"unsloth/kernels/utils.py + zoo/temporary_patches/moe call paths break"
    )


def test_bnb_functional_4bit_kernel_path(tag: str):
    """bnb.functional must expose either the legacy `lib.c*` kernels or the new `torch.ops.bitsandbytes.*` path."""
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


def test_bnb_quantstate_from_dict(tag: str):
    """unsloth-zoo rebinds QuantState.from_dict; both class and classmethod must be present."""
    candidates = [
        "bitsandbytes/functional.py",
        "bitsandbytes/functional/__init__.py",
    ]
    hit = first_match("bitsandbytes-foundation/bitsandbytes", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: functional missing")
    _, src = hit
    assert has_def(src, "QuantState", "class"), f"{tag}: bnb.functional.QuantState missing"
    assert "from_dict" in src, (
        f"{tag}: QuantState.from_dict missing; " f"unsloth-zoo monkey-patch silently no-ops"
    )


def test_bnb_nn_modules_fix_4bit_weight_optional(tag: str):
    """fix_4bit_weight_quant_state_from_module is optional; unsloth getattr-fallbacks on older bnb."""
    src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/nn/modules.py")
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/nn/modules.py missing")
    if "fix_4bit_weight_quant_state_from_module" not in src:
        pytest.skip(f"{tag}: helper not yet added (OK; getattr fallback)")


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
        f"{tag}: bnb.nn.Linear8bitLt missing in {candidates}; " f"legacy load_in_8bit path breaks"
    )


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


def test_bnb_utils_pack_unpack(tag: str):
    """4bit state-dict save/load uses these two helpers."""
    src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/utils.py")
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/utils.py missing")
    for name in ("pack_dict_to_tensor", "unpack_tensor_to_dict"):
        assert has_def(src, name, "func") or name in src, f"{tag}: bnb.utils.{name} missing"


def test_bnb_cextension_rocm_warp_size_optional(tag: str):
    """ROCM_WARP_SIZE_64 is optional (pre-ROCm bnb lacks it); unsloth probes via try/except."""
    src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/cextension.py")
    if src is None:
        pytest.skip(f"{tag}: cextension.py missing")
    if "ROCM_WARP_SIZE_64" not in src:
        pytest.skip(f"{tag}: ROCM_WARP_SIZE_64 not yet defined (pre-ROCm bnb)")


def test_bnb_autograd_functions_matmul_4bit(tag: str):
    """bnb.autograd._functions.matmul_4bit must remain (unsloth-zoo has a dynamo-disable patch site)."""
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes",
        tag,
        "bitsandbytes/autograd/_functions.py",
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/autograd/_functions.py missing")
    assert "matmul_4bit" in src, f"{tag}: bnb.autograd._functions.matmul_4bit missing"


def test_bnb_version_parseable(tag: str):
    """bnb.__version__ must be exported via at least one mechanism (unsloth feature-gates on it)."""
    src = fetch_text("bitsandbytes-foundation/bitsandbytes", tag, "bitsandbytes/__init__.py")
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/__init__.py missing")
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE))
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


def test_bnb_optimizer2state_options_are_not_passed_positionally(tag: str):
    """QGaLoreAdamW8bit must not pass percentile_clipping / block_wise by position.

    bitsandbytes 0.50.0 removed both from Optimizer2State.__init__ (PR #1871), so positions
    10 and 11 became max_unorm and skip_zeros. Arity still matched, nothing raised, and the
    optimiser silently received max_unorm=100 / skip_zeros=True. Since Q-GaLore zeroes
    p.data before the update, param_norm was 0, the unorm clip scaled every update to 0 and
    projected parameters stopped moving entirely.

    Checked here rather than at runtime because the failure is invisible on the installed
    version alone: it needs the signature from a version the test environment does not have.
    """
    src = fetch_text(
        "bitsandbytes-foundation/bitsandbytes",
        tag,
        "bitsandbytes/optim/optimizer.py",
    )
    if src is None:
        pytest.skip(f"{tag}: bitsandbytes/optim/optimizer.py missing")

    signature = first_match_signature(src, "Optimizer2State")
    if signature is None:
        pytest.skip(f"{tag}: could not read Optimizer2State.__init__ signature")

    removed = [n for n in ("percentile_clipping", "block_wise") if n not in signature]
    caller = (
        Path(__file__).resolve().parents[2] / "unsloth" / "optimizers" / "q_galore_adamw.py"
    ).read_text(encoding = "utf-8")

    call = _super_init_call(caller, "QGaLoreAdamW8bit")
    assert call is not None, "could not find QGaLoreAdamW8bit's super().__init__ call"

    # "adam" and params are positional on purpose; everything after them must be a keyword,
    # because bitsandbytes reorders the tail of this signature between minors.
    positional = [a for a in call.args if not isinstance(a, ast.Starred)]
    extra = positional[2:]
    assert not extra, (
        f"{tag}: QGaLoreAdamW8bit passes {len(extra)} argument(s) after params positionally. "
        f"{removed or 'Nothing'} was removed from Optimizer2State at this tag, so a positional "
        f"call lands values in whichever parameters now occupy those slots."
    )
    passed = {kw.arg for kw in call.keywords if kw.arg}
    assert "optim_bits" in passed, f"{tag}: optim_bits must be passed by name"
    for name in ("percentile_clipping", "block_wise"):
        if name in signature:
            continue
        assert (
            name not in passed
        ), f"{tag}: Optimizer2State no longer accepts {name}, but it is still passed."
