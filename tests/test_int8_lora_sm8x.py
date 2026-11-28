# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""int8 LoRA GEMMs on sm_8x compile without Triton's prefetch pass (CPU only: offline sm_80 compile)."""

import re

import pytest

triton = pytest.importorskip("triton")
IL = pytest.importorskip("unsloth.kernels.int8_lora")
if IL.triton is None:
    pytest.skip("needs Triton", allow_module_level = True)

try:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
except ImportError:  # pragma: no cover
    pytest.skip("this Triton has no offline compile API", allow_module_level = True)


def _triton_at_least(major, minor):
    m = re.match(r"(\d+)\.(\d+)", triton.__version__)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (major, minor)


_TYPES = {
    "_int8_mm_kernel": (
        "*i8",
        "*i8",
        "*fp32",
        "*fp32",
        "*bf16",
        "*bf16",
        "i32",
        "i32",
        "i32",
        "i32",
    ),
    "_int8_dgrad_kernel": (
        "*i8",
        "*i8",
        "*fp32",
        "*bf16",
        "*bf16",
        "*bf16",
        "i32",
        "i32",
        "i32",
        "i32",
    ),
}


def _compile_sm80(kernel, constexprs, warps, stages):
    names = list(kernel.arg_names)
    types = _TYPES[kernel.__name__]
    sig = dict(zip(names, types))
    sig.update({name: "constexpr" for name in constexprs})
    attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in names[: len(types)] if n != "M"}
    try:
        src = ASTSource(fn = kernel, signature = sig, constexprs = constexprs, attrs = attrs)
    except TypeError:
        pytest.skip("this Triton's ASTSource takes another signature format")
    try:
        return triton.compile(
            src,
            target = GPUTarget("cuda", 80, 32),
            options = {"num_warps": warps, "num_stages": stages},
        )
    except Exception as exc:  # no ptxas for this target in this install
        pytest.skip(f"cannot compile for sm_80 here: {type(exc).__name__}: {str(exc)[:200]}")


def _loop_loads(ttgir):
    """ttg.local_load ops in the K loop body (the first scf.for up to its scf.yield)."""
    body = ttgir.split("scf.for", 1)[1].split("scf.yield", 1)[0]
    return body.count("ttg.local_load")


def _cases(arch):
    bm, bn, bk, gm, warps, stages = IL._TILE
    mm = {"HAS_D": True, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm}
    bm, br, gm, dwarps, dstages = arch.dgrad_tile
    dg = {
        "HAS_D": True,
        "INV_SQRT_G": 1.0 / 128**0.5,
        "G": IL.GROUP,
        "BLOCK_M": bm,
        "BLOCK_R": br,
        "GROUP_M": gm,
    }
    return [
        (IL._int8_mm_kernel, mm, warps, stages),
        (IL._int8_dgrad_kernel, dg, dwarps, dstages),
    ]


@pytest.mark.skipif(
    not _triton_at_least(3, 8), reason = "Triton < 3.8 does not split 8-bit dots in the prefetch pass"
)
@pytest.mark.parametrize("cap", [(8, 0), (8, 9)], ids = ["sm80", "sm89"])
@pytest.mark.parametrize("case", [0, 1], ids = ["fwd", "dgrad"])
def test_sm8x_int8_dots_are_not_split_by_prefetch(monkeypatch, case, cap):
    """With prefetch skipped on sm_8x: one shared-memory load per operand per K step."""
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")  # judge fresh compiles, never a cached cubin
    monkeypatch.setattr(IL, "_SKIP_PREFETCH", True)
    arch = IL._arch_for(cap)
    assert arch.prefetch_off
    kernel, constexprs, warps, stages = _cases(arch)[case]
    with IL._no_prefetch(arch.prefetch_off):
        head = _compile_sm80(kernel, {**constexprs, "PREFETCH_OFF": True}, warps, stages)
    stock = _compile_sm80(kernel, {**constexprs, "PREFETCH_OFF": False}, warps, stages)
    head_loads = _loop_loads(head.asm["ttgir"])
    stock_loads = _loop_loads(stock.asm["ttgir"])
    assert head_loads == 2, head_loads
    # control: without the skip the pass splits them, so this test can fail
    assert stock_loads > head_loads, (stock_loads, head_loads)
    # the skip is scoped: nothing outside _no_prefetch loses the pass
    assert not getattr(IL._PREFETCH, "skip", False)


def test_arch_configuration(monkeypatch):
    monkeypatch.setattr(IL, "_SKIP_PREFETCH", True)
    split = _triton_at_least(3, 8)
    a100, l4, a10, g4, h100 = (IL._arch_for(c) for c in [(8, 0), (8, 9), (8, 6), (12, 0), (9, 0)])
    assert a100.prefetch_off == l4.prefetch_off == a10.prefetch_off == split
    assert a100.dgrad_tile == (IL._DGRAD_TILE_SM80 if split else IL._DGRAD_TILE)
    assert l4.dgrad_tile == a10.dgrad_tile == IL._DGRAD_TILE
    assert a100.rowq_one_pass and l4.rowq_one_pass and a10.rowq_one_pass
    assert l4.rotq_one_pass and not a100.rotq_one_pass and not a10.rotq_one_pass
    # sm_90 / sm_120 keep the configuration #19965 measured
    for other in (g4, h100):
        assert other == (False, IL._DGRAD_TILE, False, False)
    # kill switch: the prefetch pass stays, and with it the default input-gradient tile
    monkeypatch.setattr(IL, "_SKIP_PREFETCH", False)
    off = IL._arch_for((8, 0))
    assert not off.prefetch_off and off.dgrad_tile == IL._DGRAD_TILE
