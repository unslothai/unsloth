# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch 2.6 - 2.13 cuDNN SDPA backward plans ignore grad_output strides, so same-shape backwards
with different grad layouts give wrong grads. The GPU test's opt-out arm proves the bug exists."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _affected(*args):
    # Per test, so the GPU test still runs (and fails) on a tree without the fix.
    from unsloth.import_fixes import _cudnn_sdpa_backward_cache_ignores_grad_layout
    return _cudnn_sdpa_backward_cache_ignores_grad_layout(*args)


class TestAffectedRange:
    @pytest.mark.parametrize(
        "torch_version, cudnn_version",
        [
            ("2.13.0+cu130", 92000),
            ("2.13.0+cu129", 92400),
            ("2.10.0+cu128", 91002),
            ("2.6.0+cu126", 90501),
            ("2.12.1", 91900),
        ],
    )
    def test_affected(self, torch_version, cudnn_version):
        assert _affected(torch_version, cudnn_version)

    @pytest.mark.parametrize(
        "torch_version, cudnn_version",
        [
            ("2.14.0+cu130", 92400),  # pytorch/pytorch#191576 keys the plan on grad_output strides
            ("2.14.1+cu130", 92400),
            ("2.15.0", 92500),
            ("2.5.1+cu124", 90501),  # always copied grad_output into the output layout
            ("2.6.0+cu124", 90100),  # cuDNN < 9.5.1
            ("2.13.0+cpu", None),
            ("not a version", 92000),
        ],
    )
    def test_not_affected(self, torch_version, cudnn_version):
        assert not _affected(torch_version, cudnn_version)


class TestGradInOutputLayout:
    def _fn(self):
        from unsloth.import_fixes import _sdpa_grad_layout_function
        return _sdpa_grad_layout_function()

    def test_gradient_takes_the_output_layout(self):
        # Output laid out (B, T, H, D) seen as (B, H, T, D), the layout SDPA gives a HF-style query.
        x = torch.randn(2, 5, 3, 4).transpose(1, 2).requires_grad_()
        y = self._fn().apply(x)
        grad = torch.randn(2, 3, 5, 4)  # contiguous (B, H, T, D): a different layout
        (got,) = torch.autograd.grad(y, x, grad)
        assert got.stride() == x.stride()
        assert torch.equal(got, grad)

    def test_matching_gradient_is_not_copied(self):
        x = torch.randn(2, 3, 5, 4, requires_grad = True)
        y = self._fn().apply(x)
        grad = torch.randn(2, 3, 5, 4)
        captured = []
        x.register_hook(lambda g: captured.append(g.data_ptr()))
        y.backward(grad)
        assert captured == [grad.data_ptr()]

    def test_forward_is_identity(self):
        x = torch.randn(2, 3, 5, 4, requires_grad = True)
        y = self._fn().apply(x)
        assert torch.equal(y, x) and y.stride() == x.stride() and y.data_ptr() == x.data_ptr()


def _stack_is_affected():
    """Checked without unsloth, so a tree without the fix still runs the GPU test."""
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    from packaging.version import Version

    v = Version(Version(torch.__version__).base_version)
    cudnn = torch.backends.cudnn.version()
    if not (Version("2.6") <= v < Version("2.14")) or cudnn is None or cudnn < 90501:
        return False
    return torch.cuda.get_device_capability()[0] >= 8


_PROBE = textwrap.dedent(
    """
    import json
    import unsloth  # noqa: F401  (applies the import-time fixes)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, T, D = 1, 8, 374, 64
    out = {}

    def loss(q, k, v, w, kw, backend):
        with sdpa_kernel([backend]):
            o = F.scaled_dot_product_attention(q, k, v, scale = 0.11, **kw)
        # "hf": output read through transpose(1, 2) like HF modeling code, grad laid out (B, T, H, D);
        # "plain": output read directly, contiguous (B, H, T, D) grad.
        return ((o.transpose(1, 2) if w.shape[1] == T else o) * w).sum()

    def hf_like(q, k, v, w, kw):
        return loss(q, k, v, w, kw, SDPBackend.CUDNN_ATTENTION)

    def err(fn, seed, kw, layout):
        g = torch.Generator(device = "cuda").manual_seed(seed)
        q, k, v = (torch.randn(B, H, T, D, device = "cuda", dtype = torch.bfloat16, generator = g).requires_grad_() for _ in range(3))
        shape = (B, T, H, D) if layout == "hf" else (B, H, T, D)
        w = torch.randn(*shape, device = "cuda", dtype = torch.bfloat16, generator = g)
        got = torch.autograd.grad(fn(q, k, v, w, kw), (q, k, v))
        qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
        ref = torch.autograd.grad(loss(qf, kf, vf, w.float(), kw, SDPBackend.MATH), (qf, kf, vf))
        return max(float((a.float() - r).abs().max()) for a, r in zip(got, ref))

    mask = torch.ones(T, T, dtype = torch.bool, device = "cuda").tril()[None, None]
    # Every case: a backward with one grad layout, then a same-shape one with the other layout.
    for name, kw in (("mask", {"attn_mask": mask}), ("causal", {"is_causal": True}), ("nomask", {})):
        out[f"eager-{name}"] = [err(hf_like, 1, kw, "hf"), err(hf_like, 0, kw, "plain")]
        out[f"eager-{name}-reverse"] = [err(hf_like, 3, kw, "plain"), err(hf_like, 2, kw, "hf")]
    torch._dynamo.reset()
    compiled = torch.compile(hf_like, fullgraph = True)
    kw = {"attn_mask": mask}
    out["compiled-mask"] = [err(compiled, 5, kw, "hf"), err(compiled, 4, kw, "plain")]
    print("PROBE_RESULT " + json.dumps(out))
    """
)


def _run_probe(extra_env = None):
    env = dict(os.environ)
    env.pop("UNSLOTH_SKIP_CUDNN_SDPA_GRAD_LAYOUT_FIX", None)
    env.update(extra_env or {})
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output = True, text = True, env = env, timeout = 900
    )
    line = [x for x in r.stdout.splitlines() if x.startswith("PROBE_RESULT ")]
    assert line, f"probe crashed (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return json.loads(line[-1][len("PROBE_RESULT ") :])


@pytest.mark.gpu
@pytest.mark.skipif(
    not _stack_is_affected(), reason = "needs CUDA torch 2.6 - 2.13 with cuDNN >= 9.5.1"
)
def test_cudnn_sdpa_backward_is_right_across_grad_layouts_after_import():
    # bf16 vs fp32 math max abs error is ~0.03 here; the stale plan gives 0.7 - 7.
    res = _run_probe()
    bad = {k: v for k, v in res.items() if max(v) > 0.1}
    assert not bad, f"wrong cuDNN SDPA gradients (max abs err vs fp32 math, first, second): {bad}"
    # Kill switch: without the fix the second backward of a pair is wrong, so the probe can fail.
    off = _run_probe({"UNSLOTH_SKIP_CUDNN_SDPA_GRAD_LAYOUT_FIX": "1"})
    assert max(off["eager-mask"][1], off["eager-causal"][1]) > 0.5, off
