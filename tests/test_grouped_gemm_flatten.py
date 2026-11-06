# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The dX expert loop used the builtin range(..., flatten = FLATTEN): Triton drops keywords there."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

KERNELS_DIR = ROOT / "unsloth" / "kernels" / "moe" / "grouped_gemm" / "kernels"


@pytest.fixture(scope = "module")
def bw():
    try:
        from unsloth.kernels.moe.grouped_gemm.kernels import backward
    except Exception as exc:
        if torch.cuda.is_available():
            raise  # a real import regression, not an unsupported host
        pytest.skip(f"grouped_gemm needs a GPU build: {exc}")
    return backward


def test_no_builtin_range_receives_keywords():
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in sorted(KERNELS_DIR.glob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and node.keywords
    ]
    assert not offenders, offenders


def test_dw_kernel_has_no_dead_flatten_constexpr(bw):
    assert "FLATTEN" not in bw._grouped_gemm_dW_kernel.arg_names


# Triton < 3.4 resolves the constexpr-off tl.make_tensor_descriptor branches while hashing.
@pytest.mark.skipif(not hasattr(tl, "make_tensor_descriptor"), reason = "needs Triton >= 3.4")
@pytest.mark.parametrize("flatten", [True, False])
def test_dx_expert_loop_ir_carries_flatten(bw, flatten):
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if torch.version.hip is not None or "nvidia" not in backends:
        pytest.skip("needs Triton's CUDA backend")

    fn = bw._grouped_gemm_dX_kernel
    ptrs = {"dY_ptr": "*bf16", "w_ptr": "*bf16", "dX_ptr": "*bf16"}
    ptrs |= {"m_sizes_ptr": "*i32", "gather_indices_ptr": "*i32"}
    consts = dict(NUM_EXPERTS = 8, TOPK = 1, N = 128, K = 256, BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64)
    consts |= dict(BLOCK_SIZE_K = 64, FLATTEN = flatten)
    sig, constexprs = {}, {}
    for p in fn.params:
        if p.name in ptrs:
            sig[p.name] = ptrs[p.name]
        elif p.is_constexpr:
            constexprs[p.name] = consts.get(p.name, p.default)
            sig[p.name] = "constexpr"
        else:
            sig[p.name] = "i32"
    try:
        src = ASTSource(fn, sig, constexprs = constexprs)
        ttir = triton.compile(src, target = GPUTarget("cuda", 80, 32)).asm["ttir"]
    except Exception as exc:  # pragma: no cover
        if "driver" in str(exc).lower():
            pytest.skip(f"no Triton CUDA driver: {exc}")
        raise
    assert ttir.count("tt.flatten") == int(flatten)
