# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MultiLoRA_W backward: fp32 dH cannot overflow, and the scale reaches dA, dB and dX exactly once."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent(
    """
    import importlib.util, os, sys, types
    import torch
    import triton
    import triton.language as tl
    root, mode = sys.argv[1], sys.argv[2]

    @triton.jit
    def _probe(X, n):
        for i in range(0, n, 1):
            tl.store(X + i, i)

    try:  # some interpreter builds cannot take a runtime loop bound with this NumPy
        _probe[(1,)](torch.zeros(2), 2)
    except Exception as exc:  # noqa: BLE001
        print("INTERPRETER UNSUPPORTED", repr(exc)[:200])
        sys.exit(0)
    pkg = types.ModuleType("mlk"); pkg.__path__ = []; sys.modules["mlk"] = pkg
    utils = types.ModuleType("mlk.utils")
    utils._base_quant_state = lambda layer, W: None
    utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
    utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
    utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
    sys.modules["mlk.utils"] = utils
    path = os.path.join(root, "unsloth/kernels/multi_lora.py")
    spec = importlib.util.spec_from_file_location("mlk.multi_lora", path)
    ml = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ml; spec.loader.exec_module(ml)
    assert os.path.samefile(ml.__file__, path)

    def reference(X, W, A, B, scales, ids, dY):
        # fp64, per token, written out by hand: Y_t = X_t W^T + s_u (X_t A_u^T) B_u^T.
        X, A, B = [t.detach().double().requires_grad_() for t in (X, A, B)]
        rows = []
        for t in range(X.shape[0]):
            y = X[t] @ W.double().t()
            u = int(ids[t])
            if u >= 0:
                y = y + float(scales[u]) * (X[t] @ A[u].t()) @ B[u].t()
            rows.append(y)
        Y = torch.stack(rows)
        return torch.autograd.grad(Y, (X, A, B), dY.double())

    if mode.startswith("fp16_overflow"):
        # dY = 40000 does fit in fp16 (max 65504), but scale 2 * dY @ B (B = 1) and dY @ B alone with
        # B = 2 (then scale 0.5) are 80000 and do not; the true grads are about 800 and 400.
        scale, b = (2.0, 1.0) if mode == "fp16_overflow_scale_2" else (0.5, 2.0)
        X = torch.tensor([[0.01]], dtype = torch.float16, requires_grad = True)
        W = torch.zeros(1, 1, dtype = torch.float16)
        A = torch.tensor([[[0.01]]], requires_grad = True)
        B = torch.full((1, 1, 1), b, requires_grad = True)
        scales = torch.tensor([scale])
        ids = torch.tensor([0])
        batch = ml.MultiLoraBatch.from_adapter_ids(ids, 1)
        Y = ml.MultiLoRA_W.apply(X, W, None, A, B, scales, batch.sorted_idx, batch.seg_start)
        dY = torch.full_like(Y, 40000.0)
        got = torch.autograd.grad(Y, (X, A, B), dY)
        want = reference(X, W, A, B, scales, ids, dY)
        print("got", [g.item() for g in got], "fp64", [w.item() for w in want])
        for name, g, w in zip(("dX", "dA", "dB"), got, want):
            assert g.isfinite().all(), (name, g)
            err = (g.double() - w).abs().max().item()
            assert err <= 1e-2 * w.abs().max().item(), (name, g, w)
    else:
        torch.manual_seed(0)
        U, M, K, N, R = 3, 41, 24, 20, 8
        X = torch.randn(M, K, requires_grad = True)
        W = torch.randn(N, K) / 5
        A = torch.randn(U, R, K, requires_grad = True)
        B = torch.randn(U, N, R, requires_grad = True)
        scales = torch.tensor([0.5, 2.0, 4.0])
        ids = torch.arange(M) % (U + 1) - 1  # every adapter present, plus base-only tokens
        batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
        Y = ml.MultiLoRA_W.apply(X, W, None, A, B, scales, batch.sorted_idx, batch.seg_start)
        dY = torch.randn_like(Y)
        got = torch.autograd.grad(Y, (X, A, B), dY)
        want = reference(X, W, A, B, scales, ids, dY)
        # The check must be able to see a scale applied twice or not at all.
        for wrong in (scales**2, torch.ones(U)):
            bad = reference(X, W, A, B, wrong, ids, dY)
            for name, b, w in zip(("dX", "dA", "dB"), bad, want):
                assert (b - w).abs().max().item() > 100 * 1e-4 * w.abs().max().item(), name
        for name, g, w in zip(("dX", "dA", "dB"), got, want):
            err = (g.double() - w).abs().max().item()
            print(name, "max abs err", err)
            assert err <= 1e-4 * max(w.abs().max().item(), 1), (name, err)
    print("BACKWARD OK")
    """
)


@pytest.mark.parametrize(
    "mode", ["fp16_overflow_scale_2", "fp16_overflow_scale_0.5", "fp32_three_adapter_parity"]
)
def test_backward_applies_the_scale_once_after_the_low_precision_store(tmp_path, mode):
    pytest.importorskip("triton")
    script = tmp_path / "backward_scale.py"
    script.write_text(_SCRIPT, encoding = "utf-8")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    argv = [sys.executable, str(script), str(REPO), mode]
    proc = subprocess.run(argv, env = env, capture_output = True, text = True, timeout = 600)
    if "INTERPRETER UNSUPPORTED" in proc.stdout:
        # Some triton CPU interpreter builds cannot run a runtime-bounded loop with the installed NumPy, so
        # the kernels cannot execute on this host at all; the probe output names the failure.
        pytest.skip(reason = "TRITON_INTERPRET cannot run the kernels here: " + proc.stdout.strip())
    assert proc.returncode == 0 and "BACKWARD OK" in proc.stdout, proc.stdout + proc.stderr
