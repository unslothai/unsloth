# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The grad-layout wrapper must skip DTensor outputs (context_parallel), else the backward copy
raises "aten.copy_.default got mixed torch.Tensor and DTensor"."""

import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

torch = pytest.importorskip("torch")

_PROBE = textwrap.dedent(
    """
    import json, os
    import unsloth  # noqa: F401  (installs the import-time SDPA wrappers)
    import torch, torch.distributed as dist, torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dist.init_process_group("nccl", rank = 0, world_size = 1)
    mesh = init_device_mesh("cuda", (1,))
    B, H, T, D = 1, 4, 256, 64
    out = {}
    # Contiguous (B, H, T, D) query, output read through transpose(1, 2): the gradient's layout
    # differs from the output's, so the wrapper's backward re-lays it out.
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        g = torch.Generator(device = "cuda").manual_seed(0)
        q, k, v = (torch.randn(B, H, T, D, device = "cuda", dtype = torch.bfloat16, generator = g) for _ in range(3))
        w = torch.randn(B, T, H, D, device = "cuda", dtype = torch.bfloat16, generator = g)
        ref_in = [t.clone().requires_grad_() for t in (q, k, v)]
        with sdpa_kernel([backend]):
            o = F.scaled_dot_product_attention(*ref_in, is_causal = True)
        ref = torch.autograd.grad((o.transpose(1, 2) * w).sum(), ref_in)
        bufs = [t.clone() for t in (q, k, v)]
        try:
            with sdpa_kernel([backend]), context_parallel(
                mesh, buffers = bufs, buffer_seq_dims = [2, 2, 2], no_restore_buffers = set(bufs)
            ):
                cp_in = [t.requires_grad_() for t in bufs]
                o = F.scaled_dot_product_attention(*cp_in, is_causal = True)
                got = torch.autograd.grad((o.transpose(1, 2) * w).sum(), cp_in)
            out[name] = max(float((a.float() - r.float()).abs().max()) for a, r in zip(got, ref))
        except Exception as e:
            out[name] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    print("PROBE_RESULT " + json.dumps(out))
    dist.destroy_process_group()
    """
)


def _context_parallel_available():
    try:
        from torch.distributed.tensor.experimental import context_parallel  # noqa: F401
    except Exception:
        return False
    return True


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_nccl_available()),
    reason = "needs NCCL",
)
@pytest.mark.skipif(not _context_parallel_available(), reason = "torch without context_parallel")
def test_context_parallel_sdpa_backward_with_the_grad_layout_wrapper():
    env = dict(
        os.environ,
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(_free_port()),
        RANK = "0",
        WORLD_SIZE = "1",
        LOCAL_RANK = "0",
    )
    env.pop("UNSLOTH_SKIP_CUDNN_SDPA_GRAD_LAYOUT_FIX", None)
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output = True, text = True, env = env, timeout = 900
    )
    line = [x for x in r.stdout.splitlines() if x.startswith("PROBE_RESULT ")]
    assert line, f"probe crashed (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    res = json.loads(line[-1][len("PROBE_RESULT ") :])
    # One rank: context parallel attention is the plain attention, gradients equal.
    assert all(isinstance(v, float) and v == 0.0 for v in res.values()), res
