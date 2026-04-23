# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Numerical + performance check for ``refresh_moe_lora_merge_from_pristine``.

Does what we never actually did on this branch: verify the stacked MoE
LoRA merge kernel produces the same W_inf as the textbook reference
``W_ref[e] = W_pristine[e] + sum_a scaling_a * B_a[e] @ A_a[e]`` for both
standard (E, 2I, H) and transposed (E, H, 2I) expert layouts, with 1 and
2 active adapters, then benchmarks the batched ``baddbmm`` path against
the dense-layer-style ``addmm`` loop at Qwen3-30B-A3B MoE shapes.

Usage::
    CUDA_VISIBLE_DEVICES=2 python -u tests/flex_moe_merge_parity.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Reference: the exact formula from the docstring, no tricks.
# ---------------------------------------------------------------------------

def reference_merge(W_pristine, adapters, *, transposed):
    """adapters: list of (A_stacked [E*R, in], B_stacked [out, E*R], scaling)."""
    E = W_pristine.shape[0]
    out = W_pristine.clone()
    for A_stack, B_stack, scaling in adapters:
        R = A_stack.shape[0] // E
        in_dim = A_stack.shape[1]
        out_dim = B_stack.shape[0]
        for e in range(E):
            A_e = A_stack[e * R : (e + 1) * R]           # (R, in_dim)
            B_e = B_stack[:, e * R : (e + 1) * R]         # (out_dim, R)
            delta = (B_e @ A_e).to(W_pristine.dtype)      # (out_dim, in_dim)
            if transposed:
                out[e] += scaling * delta.t()
            else:
                out[e] += scaling * delta
    return out


# ---------------------------------------------------------------------------
# Under-test: a standalone copy of the merge body from flex_moe.py so we
# can drive it without loading a 30B model. The logic is a character-for-
# character lift of lines 747-794 in unsloth/inference/flex_moe.py.
# ---------------------------------------------------------------------------

def merge_under_test(W_inf, W_pristine, adapters):
    """adapters: list of (A_stacked, B_stacked, scaling)."""
    E = W_inf.shape[0]
    # Orientation detection (mirrors flex_moe.py:752-763).
    A_w0, B_w0, _ = adapters[0]
    in_dim = A_w0.shape[1]
    out_dim = B_w0.shape[0]
    d0, d1 = W_inf.shape[1], W_inf.shape[2]
    if d0 == out_dim and d1 == in_dim:
        is_standard = True
    elif d0 == in_dim and d1 == out_dim:
        is_standard = False
    else:
        raise RuntimeError("orientation")

    W_inf.copy_(W_pristine)

    for A_w, B_w, scaling in adapters:
        R = A_w.shape[0] // E
        A_3d = A_w.view(E, R, in_dim)
        B_3d = B_w.view(out_dim, E, R).permute(1, 0, 2).contiguous()
        if is_standard:
            torch.baddbmm(
                W_inf,
                B_3d.to(W_inf.dtype),
                A_3d.to(W_inf.dtype),
                alpha=float(scaling),
                beta=1.0,
                out=W_inf,
            )
        else:
            torch.baddbmm(
                W_inf,
                A_3d.transpose(-2, -1).contiguous().to(W_inf.dtype),
                B_3d.transpose(-2, -1).contiguous().to(W_inf.dtype),
                alpha=float(scaling),
                beta=1.0,
                out=W_inf,
            )
    return W_inf


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def _make_adapter(E, R, in_dim, out_dim, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(E * R, in_dim, generator=g, device=device, dtype=dtype) * 0.02
    B = torch.randn(out_dim, E * R, generator=g, device=device, dtype=dtype) * 0.02
    return A, B


def test_correctness(E, in_dim, out_dim, R, dtype, device, *, transposed, n_adapters):
    scalings = [2.0, 0.5][:n_adapters]
    adapters = [
        (*_make_adapter(E, R, in_dim, out_dim, dtype, device, seed=11 + i), s)
        for i, s in enumerate(scalings)
    ]

    if transposed:
        W_pristine = torch.randn(E, in_dim, out_dim, device=device, dtype=dtype) * 0.02
    else:
        W_pristine = torch.randn(E, out_dim, in_dim, device=device, dtype=dtype) * 0.02
    W_inf = torch.empty_like(W_pristine)

    merge_under_test(W_inf, W_pristine, adapters)
    W_ref = reference_merge(W_pristine, adapters, transposed=transposed)

    tol = dict(atol=3e-3, rtol=3e-3) if dtype == torch.bfloat16 else dict(atol=1e-5, rtol=1e-5)
    close = torch.allclose(W_inf, W_ref, **tol)
    max_err = (W_inf - W_ref).abs().max().item()
    scale = W_ref.abs().max().item()
    rel = max_err / max(scale, 1e-8)
    label = f"E={E} in={in_dim} out={out_dim} R={R} {dtype} transposed={transposed} n_adapters={n_adapters}"
    verdict = "OK" if close else "FAIL"
    print(f"  [{verdict}] {label}  max_abs={max_err:.3e}  rel={rel:.3e}")
    return close


# ---------------------------------------------------------------------------
# Performance: batched baddbmm vs per-expert addmm loop (same total flops).
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    _sync()
    return (time.perf_counter() - t0) / iters


def perf_vs_dense_addmm(out_dim, in_dim, R, dtype, device):
    """Dense-layer LoRA refresh: a single torch.addmm call like the one
    in flex_qwen3_llama.py::refresh_lora_merge_from_pristine. Used as a
    per-matrix baseline to compare against the MoE baddbmm kernel cost."""
    W_pristine = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.02
    W = torch.empty_like(W_pristine)
    A = torch.randn(R, in_dim, device=device, dtype=dtype) * 0.02
    B = torch.randn(out_dim, R, device=device, dtype=dtype) * 0.02
    scaling = 2.0

    def run_addmm():
        torch.addmm(W_pristine, B, A, alpha=scaling, out=W)

    t = bench(run_addmm, iters=200, warmup=20)
    print(f"  dense addmm  out={out_dim:>5} in={in_dim:>5} R={R:>2} "
          f"{t * 1e3:7.4f}ms")
    return t


def perf_compare(E, in_dim, out_dim, R, dtype, device):
    A_w, B_w = _make_adapter(E, R, in_dim, out_dim, dtype, device, seed=0)
    W_pristine = torch.randn(E, out_dim, in_dim, device=device, dtype=dtype) * 0.02
    W_inf_a = torch.empty_like(W_pristine)
    W_inf_b = torch.empty_like(W_pristine)

    A_3d = A_w.view(E, R, in_dim)
    B_3d = B_w.view(out_dim, E, R).permute(1, 0, 2).contiguous()
    scaling = 2.0

    def run_baddbmm():
        W_inf_a.copy_(W_pristine)
        torch.baddbmm(W_inf_a, B_3d, A_3d, alpha=scaling, beta=1.0, out=W_inf_a)

    def run_addmm_loop():
        W_inf_b.copy_(W_pristine)
        for e in range(E):
            torch.addmm(
                W_inf_b[e],
                B_3d[e],
                A_3d[e],
                alpha=scaling,
                out=W_inf_b[e],
            )

    t_bad = bench(run_baddbmm)
    t_loop = bench(run_addmm_loop)

    # Sanity: they produce the same result.
    run_baddbmm()
    run_addmm_loop()
    max_err = (W_inf_a - W_inf_b).abs().max().item()

    print(f"  E={E:>3} out={out_dim:>5} in={in_dim:>5} R={R:>2} "
          f"baddbmm={t_bad * 1e3:7.3f}ms  addmm_loop={t_loop * 1e3:7.3f}ms  "
          f"speedup={t_loop / t_bad:5.2f}x  max_abs_diff={max_err:.1e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407)
    print(f"[merge-parity] device={device} dtype=bf16 + fp32")
    print(f"[merge-parity] torch={torch.__version__}")
    print()

    print("== Correctness (fp32 golden + bf16 realistic) ==")
    # Qwen3-30B-A3B shapes.
    #   gate_up_proj: (E=128, 2I=1536, H=2048) standard
    #   down_proj:    (E=128, H=2048, I=768)  standard
    # Transposed variant: (E, H, 2I) — exercised via the `transposed=True` test.
    cases = [
        # (E, in_dim, out_dim, R, dtype, transposed, n_adapters)
        ( 8, 128, 192, 4, torch.float32, False, 1),
        ( 8, 128, 192, 4, torch.float32, True,  1),
        ( 8, 128, 192, 4, torch.float32, False, 2),
        (16, 256, 512, 8, torch.float32, True,  2),
        # bf16 at realistic Qwen3 MoE shapes. Mild tol to allow 3e-3 bf16 noise.
        (128, 2048, 1536, 16, torch.bfloat16, False, 1),  # gate_up_proj-like
        (128, 768,  2048, 16, torch.bfloat16, False, 1),  # down_proj-like
        (128, 2048, 1536, 16, torch.bfloat16, True,  1),  # transposed gate_up
        (128, 2048, 1536, 16, torch.bfloat16, False, 2),  # two adapters
    ]
    all_ok = True
    for E, in_dim, out_dim, R, dtype, tr, na in cases:
        ok = test_correctness(E, in_dim, out_dim, R, dtype, device,
                              transposed=tr, n_adapters=na)
        all_ok = all_ok and ok
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")

    print()
    print("== Performance: batched baddbmm vs per-expert addmm loop (bf16) ==")
    print("   (same arithmetic, different dispatch pattern — baddbmm = 1 kernel,")
    print("    addmm loop = E kernels)")
    moe_cases = [
        (128, 2048, 1536, 16),   # gate_up_proj
        (128, 768,  2048, 16),   # down_proj
        (128, 2048, 1536, 64),   # larger rank
    ]
    for E, in_dim, out_dim, R in moe_cases:
        perf_compare(E, in_dim, out_dim, R, torch.bfloat16, device)

    print()
    print("== Dense baseline: torch.addmm on a single expert-sized matrix ==")
    print("   (this is what flex_qwen3_llama.py:354-370 does for dense layers)")
    dense_times = {}
    for E, in_dim, out_dim, R in moe_cases:
        t = perf_vs_dense_addmm(out_dim, in_dim, R, torch.bfloat16, device)
        dense_times[(E, in_dim, out_dim, R)] = t

    print()
    print("== Per-expert cost comparison ==")
    print("   (MoE baddbmm cost / E) vs single dense addmm for the same per-expert matrix")
    for E, in_dim, out_dim, R in moe_cases:
        # Measure baddbmm cost again for this exact shape to get the numerator.
        A_w, B_w = _make_adapter(E, R, in_dim, out_dim, torch.bfloat16, device, seed=0)
        W_pristine = torch.randn(E, out_dim, in_dim, device=device, dtype=torch.bfloat16) * 0.02
        W_inf = torch.empty_like(W_pristine)
        A_3d = A_w.view(E, R, in_dim)
        B_3d = B_w.view(out_dim, E, R).permute(1, 0, 2).contiguous()

        def run_baddbmm():
            W_inf.copy_(W_pristine)
            torch.baddbmm(W_inf, B_3d, A_3d, alpha=2.0, beta=1.0, out=W_inf)
        t_moe = bench(run_baddbmm, iters=50, warmup=10)
        t_dense = dense_times[(E, in_dim, out_dim, R)]
        per_expert = t_moe / E
        print(f"  E={E:>3} out={out_dim:>5} in={in_dim:>5} R={R:>2}  "
              f"moe_per_expert={per_expert * 1e6:7.2f}us  "
              f"dense_addmm={t_dense * 1e6:7.2f}us  "
              f"ratio(moe/dense)={per_expert / t_dense:5.2f}x")


if __name__ == "__main__":
    main()
