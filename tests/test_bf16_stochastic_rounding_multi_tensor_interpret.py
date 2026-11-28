# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU, Triton interpreter: the multi-tensor AdamW kernel writes the same bits as one _adamw_kernel launch per
weight (weights, both moments), across chunk tails, mixed steps and gradient dtypes. TRITON_INTERPRET must be set
before triton imports, so the check runs in a subprocess."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    """
    import importlib.util, sys, torch
    spec = importlib.util.spec_from_file_location("sr", sys.argv[1])
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    span = sr._BLOCK * sr._SUB
    sizes = (3, 1025, span + 7)
    group_scalars = (1 - 1e-5, 0.1, 0.999, 0.001, 1e-8)

    def run(g_dtype, swap_seeds):
        gen = torch.Generator().manual_seed(0)
        make = lambda n, s, d = torch.bfloat16: (torch.randn(n, generator = gen) * s).to(d)
        multi = [[make(n, 0.05), make(n, 1e-3, g_dtype), make(n, 1e-3), make(n, 1e-6).abs()] for n in sizes]
        single = [[x.clone() for x in w] for w in multi]
        seeds = [(1 << 40) + 977 * j + 2654435761 * t for j, t in enumerate((1, 7, 300))]
        scalars = [sr._adamw_scalars(1e-3, 0.9, 0.999, t) for t in (1, 7, 300)]
        for (p, g, m, v), seed, (neg, bc2) in zip(single, seeds, scalars):
            sr._adamw_kernel[sr._grid(p.numel())](
                p, g, m, v, p.numel(), seed, *group_scalars[:4], neg, bc2, group_scalars[4], BLOCK = sr._BLOCK
            )
        if swap_seeds:
            seeds = seeds[::-1]
        table = torch.empty((len(sizes), 7), dtype = torch.int64)
        table[:, :6] = torch.tensor(
            [[p.data_ptr(), g.data_ptr(), m.data_ptr(), v.data_ptr(), p.numel(), s]
             for (p, g, m, v), s in zip(multi, seeds)]
        )
        table[:, 6:].view(torch.float32).copy_(torch.tensor(scalars, dtype = torch.float32))
        chunks = torch.tensor(
            [(t, k) for t, n in enumerate(sizes) for k in range((n + span - 1) // span)], dtype = torch.int32
        )
        sr._adamw_multi_kernel[(chunks.shape[0],)](
            table, table.view(torch.float32), chunks, *group_scalars,
            G_DTYPE = sr._G_DTYPES[g_dtype], BLOCK = sr._BLOCK, SUB = sr._SUB,
        )
        return all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for w, s in zip(multi, single) for a, b in zip(w, s)
            if a.dtype == torch.bfloat16
        )

    for g_dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert run(g_dtype, False), g_dtype
    assert not run(torch.bfloat16, True)  # the comparison can fail: other seeds give other bits
    print("MULTI_TENSOR_BITWISE_OK")
    """
)


def test_multi_tensor_kernel_is_bitwise_per_weight_kernel_interpreted(tmp_path):
    script = tmp_path / "check.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, TRITON_INTERPRET = "1")
    env.pop("UNSLOTH_BF16_SR_MULTI_TENSOR", None)
    out = subprocess.run(
        [sys.executable, str(script), str(ROOT / "unsloth/optimizers/bf16_stochastic_rounding.py")],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert out.returncode == 0 and "MULTI_TENSOR_BITWISE_OK" in out.stdout, (
        out.stdout[-2000:] + out.stderr[-4000:]
    )
