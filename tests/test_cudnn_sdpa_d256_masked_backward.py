# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Masked head_dim-256 SDPA training must not get NaN gradients from cuDNN on SM100 (torch 2.14,
cuDNN 9.24). The GPU tests only `import unsloth` in a fresh interpreter, so they fail on a tree
without the fix wherever the bug exists."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _detour(*args):
    # Per test, so the GPU tests still run (and fail) on a tree without the fix.
    from unsloth.import_fixes import _sdpa_needs_cudnn_d256_detour
    return _sdpa_needs_cudnn_d256_detour(*args)


def _t(
    *shape,
    dtype = None,
    requires_grad = False,
    device = "cpu",
):
    return torch.zeros(
        *shape, dtype = dtype or torch.bfloat16, device = device, requires_grad = requires_grad
    )


class TestDetourDecision:
    """The detour is for masked, grad-building, half-precision head_dim-256 calls only."""

    def test_no_mask_is_never_detoured(self):
        q = _t(1, 2, 8, 256, requires_grad = True)
        assert not _detour(frozenset({0}), q, q, None)

    def test_cpu_query_is_never_detoured(self):
        q = _t(1, 2, 8, 256, requires_grad = True)
        m = torch.ones(8, 8, dtype = torch.bool)
        assert not _detour(frozenset({0, None}), q, q, m)

    def test_no_grad_is_never_detoured(self):
        q = _t(1, 2, 8, 256, requires_grad = True)
        m = torch.ones(8, 8, dtype = torch.bool)
        with torch.no_grad():
            assert not _detour(frozenset({0, None}), q, q, m)


def _gpu_sm100():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        return any(
            torch.cuda.get_device_capability(i)[0] == 10 for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


_PROBE = textwrap.dedent(
    """
    import json, sys
    import unsloth  # noqa: F401  (applies the import-time fixes)
    import torch
    import torch.nn.functional as F

    dev = next(i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i)[0] == 10)
    torch.cuda.set_device(dev)
    out = {}

    def grads(fn, D, mask, dtype):
        g = torch.Generator(device = "cuda").manual_seed(0)
        B, H, S = 2, 4, 24
        q, k, v = (torch.randn(B, H, S, D, device = "cuda", dtype = dtype, generator = g).requires_grad_() for _ in range(3))
        m = torch.ones(S, S, device = "cuda", dtype = torch.bool).tril().expand(B, 1, S, S) if mask else None
        o = fn(q, k, v, m)
        o.float().square().sum().backward()
        return [x.grad.float() for x in (q, k, v)]

    def sdpa(q, k, v, m):
        return F.scaled_dot_product_attention(q, k, v, attn_mask = m, is_causal = m is None)

    compiled = torch.compile(sdpa, fullgraph = True, dynamic = True)
    for name, fn in (("eager", sdpa), ("compiled", compiled)):
        for D in (256, 128):
            for mask in (True, False):
                got = grads(fn, D, mask, torch.bfloat16)
                with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                    ref = grads(sdpa, D, mask, torch.float32)
                out[f"{name}-D{D}-mask{int(mask)}"] = [
                    None if not torch.isfinite(a).all() else float((a - r).norm() / r.norm())
                    for a, r in zip(got, ref)
                ]
    # fp32 Q/K/V (DoRA promotes them) under bf16 autocast: SDPA casts to bf16 before dispatch.
    for name, fn in (("eager", sdpa), ("compiled", compiled)):
        with torch.autocast("cuda", dtype = torch.bfloat16):
            got = grads(fn, 256, True, torch.float32)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            ref = grads(sdpa, 256, True, torch.float32)
        out[f"{name}-autocast-fp32-D256-mask1"] = [
            None if not torch.isfinite(a).all() else float((a - r).norm() / r.norm())
            for a, r in zip(got, ref)
        ]
    with torch.no_grad():
        q = torch.randn(1, 2, 8, 256, device = "cuda", dtype = torch.bfloat16)
        m = torch.ones(8, 8, device = "cuda", dtype = torch.bool)
        out["inference_finite"] = bool(torch.isfinite(F.scaled_dot_product_attention(q, q, q, attn_mask = m)).all())
    print("PROBE_RESULT " + json.dumps(out))
    """
)


def _run_probe(extra_env = None):
    env = dict(os.environ)
    env.update(extra_env or {})
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output = True, text = True, env = env, timeout = 900
    )
    line = [x for x in r.stdout.splitlines() if x.startswith("PROBE_RESULT ")]
    assert line, f"probe crashed (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return json.loads(line[-1][len("PROBE_RESULT ") :])


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_sm100(), reason = "needs an SM100 (B200 / B300) GPU")
def test_masked_head_dim_256_backward_is_finite_after_import():
    res = _run_probe()
    assert res.pop("inference_finite") is True
    bad = {k: v for k, v in res.items() if any(e is None or e > 2e-2 for e in v)}
    assert not bad, f"non-finite or wrong SDPA gradients (rel err dq, dk, dv vs fp32 math): {bad}"


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_sm100(), reason = "needs an SM100 (B200 / B300) GPU")
def test_opt_out_env_leaves_torch_dispatch_alone():
    """With the opt-out set the wrapper is not installed, so torch's own dispatch runs."""
    code = (
        "import unsloth, torch.nn.functional as F\n"
        "print('WRAPPED', getattr(F.scaled_dot_product_attention, '_unsloth_avoids_cudnn_d256_masked_backward', False))\n"
    )
    for env, want in (({"UNSLOTH_ALLOW_CUDNN_SDPA_D256": "1"}, "False"), ({}, "True")):
        e = dict(os.environ)
        e.pop("UNSLOTH_ALLOW_CUDNN_SDPA_D256", None)
        e.update(env)
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output = True, text = True, env = e, timeout = 600
        )
        assert f"WRAPPED {want}" in r.stdout, (env, r.stdout[-2000:], r.stderr[-2000:])
