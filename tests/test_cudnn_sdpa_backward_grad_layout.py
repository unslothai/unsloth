# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch 2.6 - 2.13 cuDNN SDPA backward plans ignore grad_output strides, so same-shape backwards
with different grad layouts give wrong grads. The GPU test's opt-out arm proves the bug exists."""

import importlib
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


def _install_wrapper(monkeypatch):
    """Installs the real wrapper on a CPU box made to look like CUDA torch 2.13 / cuDNN 9.20."""
    import torch.nn.functional as F

    import_fixes = importlib.import_module("unsloth.import_fixes")

    monkeypatch.delenv("UNSLOTH_SKIP_CUDNN_SDPA_GRAD_LAYOUT_FIX", raising = False)
    monkeypatch.setattr(
        F, "scaled_dot_product_attention", torch._C._nn.scaled_dot_product_attention
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 92000)
    import_fixes.fix_cudnn_sdpa_backward_grad_layout()
    sdpa = F.scaled_dot_product_attention
    assert getattr(sdpa, import_fixes._SDPA_GRAD_LAYOUT_FLAG, False)
    # CPU outputs stand in for CUDA ones from here on.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    return sdpa


def _math():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def _hf_query():
    # (B, T, H, D) seen as (B, H, T, D); math returns a contiguous output, so a gradient read back
    # through transpose(1, 2) arrives in a different layout.
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 6, 2, 8, dtype = torch.float64, generator = g).transpose(1, 2)


def _loss(
    sdpa,
    x,
    inplace = False,
):
    out = sdpa(x, x, x)
    if inplace:
        out.mul_(2.0)
    return (out.transpose(1, 2).contiguous() * 3.0).sum()


class TestNonCudnnBackendsKeepNativeBehaviour:
    def test_func_grad_through_the_wrapper(self, monkeypatch):
        sdpa = _install_wrapper(monkeypatch)
        x = _hf_query()
        with _math():
            got = torch.func.grad(lambda t: _loss(sdpa, t))(x)
            ref = torch.func.grad(lambda t: _loss(torch._C._nn.scaled_dot_product_attention, t))(x)
        assert torch.equal(got, ref)

    def test_inplace_on_the_output_then_backward(self, monkeypatch):
        sdpa = _install_wrapper(monkeypatch)
        grads = []
        for fn in (sdpa, torch._C._nn.scaled_dot_product_attention):
            x = _hf_query().requires_grad_()
            with _math():
                _loss(fn, x, inplace = True).backward()
            grads.append(x.grad)
        assert torch.equal(*grads)

    def test_math_output_is_returned_untouched(self, monkeypatch):
        sdpa = _install_wrapper(monkeypatch)
        x = _hf_query().requires_grad_()
        with _math():
            out = sdpa(x, x, x)
        assert not out._is_view() and not out._backward_hooks
        assert type(out.grad_fn).__name__ == "UnsafeViewBackward0"


class TestCudnnHookPath:
    """The eager path a cuDNN output takes, driven on CPU by treating the math node as cuDNN's."""

    @pytest.fixture
    def sdpa(self, monkeypatch):
        import_fixes = importlib.import_module("unsloth.import_fixes")

        sdpa = _install_wrapper(monkeypatch)
        monkeypatch.setattr(
            import_fixes,
            "_SDPA_GRAD_LAYOUT_NODES",
            frozenset(("UnsafeViewBackward0",)),
            raising = False,
        )
        return sdpa

    def _grad_at_sdpa_node(self, sdpa, inplace):
        x = _hf_query().requires_grad_()
        seen = []
        with _math():
            out = sdpa(x, x, x)
            assert not out._is_view() and out._backward_hooks  # the hook engaged
            out.grad_fn.register_prehook(lambda grads: seen.append(grads[0].stride()))
            stride = out.stride()
            if inplace:
                out.mul_(2.0)
            (out.transpose(1, 2).contiguous() * 3.0).sum().backward()
        return seen, stride, x.grad

    @pytest.mark.parametrize("inplace", [False, True])
    def test_grad_reaches_backward_in_the_output_layout(self, sdpa, inplace):
        seen, stride, grad = self._grad_at_sdpa_node(sdpa, inplace)
        assert seen == [stride]
        x = _hf_query().requires_grad_()
        with _math():
            _loss(torch._C._nn.scaled_dot_product_attention, x, inplace).backward()
        assert torch.equal(grad, x.grad)

    def test_func_grad(self, sdpa):
        x = _hf_query()
        with _math():
            got = torch.func.grad(lambda t: _loss(sdpa, t))(x)
            ref = torch.func.grad(lambda t: _loss(torch._C._nn.scaled_dot_product_attention, t))(x)
        assert torch.equal(got, ref)


class TestCompiledPathFunction:
    """torch.compile takes the autograd.Function (a hook on an intermediate graph-breaks)."""

    def _fn(self):
        from unsloth.import_fixes import _sdpa_grad_layout_function
        return _sdpa_grad_layout_function()

    def _f(self, wrap):
        def f(x):
            return (
                wrap(torch._C._nn.scaled_dot_product_attention(x, x, x)).transpose(1, 2) * 3.0
            ).sum()

        return f

    @pytest.mark.parametrize("transform", ["grad", "vmap_grad", "jacrev"])
    def test_func_transforms(self, transform):
        fn = self._fn()
        x = _hf_query()

        def run(wrap):
            f = self._f(wrap)
            if transform == "grad":
                return torch.func.grad(f)(x)
            if transform == "vmap_grad":
                return torch.func.vmap(torch.func.grad(f))(x[None].expand(3, *x.shape))
            return torch.func.jacrev(
                lambda t: wrap(torch._C._nn.scaled_dot_product_attention(t, t, t))
            )(x)

        with _math():
            assert torch.equal(run(fn.apply), run(lambda o: o))

    def test_fullgraph_compile_has_no_graph_break(self):
        from torch._dynamo.utils import counters

        fn = self._fn()
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(self._f(fn.apply), fullgraph = True, backend = "aot_eager")
        x = _hf_query().requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        with _math():
            compiled(x).backward()
            self._f(lambda o: o)(ref_x).backward()
        assert not counters["graph_break"]
        assert torch.equal(x.grad, ref_x.grad)
        torch._dynamo.reset()


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
