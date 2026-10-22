# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU-only: narrow-row RMSNorm selection per backend, read from the kernel source (no GPU, no import).

CUDA keeps the 32-lane replay and settings it shipped with; ROCm divides by the wavefront width.
"""

import ast
import functools
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "kernels" / "rms_layernorm.py"
TREE = ast.parse(SOURCE.read_text(encoding = "utf-8"))
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}

# The CUDA replay as shipped before ROCm support; any edit changes CUDA bytes.
CUDA_REPLAY = """
def _row_dot_in_cuda_row_order(a, b, ROWS: tl.constexpr, WARPS: tl.constexpr):
    a = tl.permute(tl.reshape(a, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
    b = tl.permute(tl.reshape(b, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
    a_lo, a_hi = tl.split(a)
    b_lo, b_hi = tl.split(b)
    acc = tl.fma(a_lo, b_lo, a_hi * b_hi)
    acc = _fold_lanes(acc, ROWS, WARPS, 8)
    acc = _fold_lanes(acc, ROWS, WARPS, 4)
    acc = _fold_lanes(acc, ROWS, WARPS, 2)
    acc = _fold_lanes(acc, ROWS, WARPS, 1)
    acc = tl.reshape(acc, (ROWS, WARPS))
    if WARPS == 4:
        acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), axis = 1)
    return tl.sum(acc, axis = 1)
"""


def _calculate_settings(n):
    block = 1 << (n - 1).bit_length()
    warps = 32 if block >= 32768 else 16 if block >= 8192 else 8 if block >= 2048 else 4
    return block, warps


def _load(hip, lanes):
    """Exec _multirow_settings (+ _hip_wavefront) against a fake torch."""
    calls = []

    def wavefront(index):
        calls.append(index)
        return lanes

    torch = types.SimpleNamespace(
        version = types.SimpleNamespace(hip = hip),
        cuda = types.SimpleNamespace(current_device = lambda: 0),
    )
    ns = {
        "torch": torch,
        "functools": functools,
        "_settings": _calculate_settings,
        "_MULTIROW": True,
        "_MULTIROW_ELEMENTS": 4096,
        "_MULTIROW_NUM_WARPS": 4,
    }
    exec(compile(ast.Module([FUNCS["_multirow_settings"]], []), str(SOURCE), "exec"), ns)
    ns["_hip_wavefront"] = wavefront
    return ns["_multirow_settings"], calls


def _shipped_cuda(n):
    block, warps = _calculate_settings(n)
    if warps != 4 or not (64 <= block <= 128):
        return None
    return block, 4096 // block, block // 32, 4


def test_cuda_settings_unchanged_and_never_ask_for_a_wavefront():
    settings, calls = _load(hip = None, lanes = 64)
    for n in range(1, 4097):
        assert settings(n) == _shipped_cuda(n), n
    assert calls == []


@pytest.mark.parametrize("lanes", [32, 64])
def test_hip_settings_divide_by_the_wavefront(lanes):
    settings, calls = _load(hip = "7.0", lanes = lanes)
    for n in range(1, 4097):
        cuda = _shipped_cuda(n)
        expect = None if cuda is None else (cuda[0], cuda[1], cuda[0] // lanes, 4)
        assert settings(n) == expect, n
    assert calls and set(calls) == {0}
    assert settings(128, types.SimpleNamespace(index = 3)) is not None and calls[-1] == 3


def test_hip_unknown_wavefront_stays_one_row():
    settings, _ = _load(hip = "7.0", lanes = None)
    assert all(settings(n) is None for n in (33, 64, 100, 128))


def test_cuda_replay_body_is_the_shipped_one():
    shipped = ast.parse(CUDA_REPLAY).body[0]
    current = FUNCS["_row_dot_in_cuda_row_order"]
    assert [ast.dump(s) for s in current.body if not isinstance(s, ast.Expr)] == [
        ast.dump(s) for s in shipped.body
    ]
    assert ast.dump(current.args) == ast.dump(shipped.args)


def test_row_order_dispatch_is_a_compile_time_backend_switch():
    flag = next(
        n
        for n in TREE.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_HIP_ROW_ORDER"
    )
    assert ast.unparse(flag.value) == "tl.constexpr(bool(torch.version.hip))"
    dispatch = FUNCS["_row_dot_in_row_kernel_order"]
    branch = next(s for s in dispatch.body if isinstance(s, ast.If))
    assert ast.unparse(branch.test) == "_HIP_ROW_ORDER"
    assert "_row_dot_in_hip_row_order" in ast.unparse(branch.body[0])
    assert "_row_dot_in_cuda_row_order" in ast.unparse(branch.orelse[0])
    for name in ("_rms_layernorm_forward_rows", "_rms_backward_rows_block"):
        text = ast.unparse(FUNCS[name])
        assert "_row_dot_in_row_kernel_order(" in text
        assert (
            "_row_dot_in_cuda_row_order(" not in text and "_row_dot_in_hip_row_order(" not in text
        )


def test_multirow_is_no_longer_gated_off_on_hip():
    gate = next(
        n
        for n in TREE.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_MULTIROW"
    )
    assert "version.hip" not in ast.unparse(gate.value)
    assert "_triton_at_least(3, 6)" in ast.unparse(gate.value)
