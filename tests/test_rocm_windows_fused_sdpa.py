# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""On Windows ROCm (torch 2.11.0+rocm7.14.1, gfx1151) fused SDPA raises hipErrorInvalidValue on
any backward, an explicit mask and enable_gqa. Those calls must run on the math kernel, and every
other call must keep its fused kernel. The routing tests fake the probe result, so they run on
any CUDA GPU; the last test needs the real Windows ROCm host."""

import json
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _detour(*args):
    # Per test, so the tests still run (and fail) on a tree without the fix.
    from unsloth.import_fixes import _sdpa_needs_rocm_windows_detour
    return _sdpa_needs_rocm_windows_detour(*args)


ALL = {0: frozenset({"grad", "mask", "gqa"})}


def _t(
    *shape,
    requires_grad = False,
    device = "cpu",
):
    return torch.zeros(*shape, dtype = torch.bfloat16, device = device, requires_grad = requires_grad)


class TestDetourDecision:
    def test_cpu_query_is_never_detoured(self):
        q = _t(1, 2, 8, 64, requires_grad = True)
        assert not _detour(
            {None: ALL[0], 0: ALL[0]}, q, q, q, torch.ones(8, 8, dtype = torch.bool), True
        )

    def test_device_without_failures_is_never_detoured(self):
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA tensor")
        q = _t(1, 2, 8, 64, requires_grad = True, device = "cuda:0")
        assert not _detour({}, q, q, q, torch.ones(8, 8, dtype = torch.bool, device = "cuda:0"), True)

    @pytest.mark.parametrize(
        "kinds, grad, mask, gqa, expected",
        [
            (
                {"grad", "mask", "gqa"},
                False,
                False,
                False,
                False,
            ),  # plain forward keeps the fused kernel
            ({"grad", "mask", "gqa"}, True, False, False, True),
            ({"grad", "mask", "gqa"}, False, True, False, True),
            ({"grad", "mask", "gqa"}, False, False, True, True),
            ({"mask"}, True, False, True, False),  # only the kinds that failed are routed
            ({"gqa"}, False, True, False, False),
            ({"grad"}, False, True, True, False),
        ],
    )
    def test_only_failing_kinds_are_detoured(self, kinds, grad, mask, gqa, expected):
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA tensor")
        q = _t(1, 2, 8, 64, requires_grad = grad, device = "cuda:0")
        m = torch.ones(8, 8, dtype = torch.bool, device = "cuda:0") if mask else None
        assert _detour({0: frozenset(kinds)}, q, q, q, m, gqa) is expected

    def test_no_grad_mode_is_not_a_backward(self):
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA tensor")
        q = _t(1, 2, 8, 64, requires_grad = True, device = "cuda:0")
        with torch.no_grad():
            assert not _detour({0: frozenset({"grad"})}, q, q, q, None, False)


def test_probe_is_windows_rocm_only():
    from unsloth.import_fixes import _rocm_windows_fused_sdpa_failures
    if sys.platform == "win32" and getattr(torch.version, "hip", None):
        pytest.skip("this host is the one the probe is for")
    assert _rocm_windows_fused_sdpa_failures() == {}


_ROUTING = textwrap.dedent(
    """
    import json
    import unsloth.import_fixes as fixes
    fixes._rocm_windows_fused_sdpa_failures = lambda: {0: frozenset({"grad", "mask", "gqa"})}
    fixes.fix_rocm_windows_fused_sdpa()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.cuda.set_device(0)
    g = torch.Generator(device = "cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 64, 64, device = "cuda", dtype = torch.bfloat16, generator = g) for _ in range(3))
    kv = (k[:, :2].contiguous(), v[:, :2].contiguous())
    mask = torch.ones(64, 64, device = "cuda", dtype = torch.bool).tril()
    orig = F.scaled_dot_product_attention.__wrapped__

    def ref(backend, *a, **kw):
        with sdpa_kernel([backend]):
            return orig(*a, **kw)

    def same(a, b):
        return bool(torch.equal(a, b))

    out = {"wrapped": F.scaled_dot_product_attention is not orig}
    with torch.no_grad():
        out["mask_is_math"] = same(F.scaled_dot_product_attention(q, k, v, attn_mask = mask),
                                   ref(SDPBackend.MATH, q, k, v, attn_mask = mask))
        out["gqa_is_math"] = same(F.scaled_dot_product_attention(q, *kv, is_causal = True, enable_gqa = True),
                                  ref(SDPBackend.MATH, q, *kv, is_causal = True, enable_gqa = True))
        out["plain_is_flash"] = same(F.scaled_dot_product_attention(q, k, v, is_causal = True),
                                     ref(SDPBackend.FLASH_ATTENTION, q, k, v, is_causal = True))
        out["flash_differs_from_math"] = not same(ref(SDPBackend.FLASH_ATTENTION, q, k, v, is_causal = True),
                                                  ref(SDPBackend.MATH, q, k, v, is_causal = True))
    qg = q.clone().requires_grad_()
    out["grad_is_math"] = same(F.scaled_dot_product_attention(qg, k, v, is_causal = True),
                               ref(SDPBackend.MATH, qg, k, v, is_causal = True))
    print("ROUTING " + json.dumps(out))
    """
)


def test_wrapper_routes_failing_kinds_to_math_and_keeps_the_rest():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        pytest.skip("needs an NVIDIA GPU with the flash kernel")
    p = subprocess.run(
        [sys.executable, "-c", _ROUTING], capture_output = True, text = True, timeout = 900
    )
    line = [l for l in p.stdout.splitlines() if l.startswith("ROUTING ")]
    assert line, p.stdout[-2000:] + p.stderr[-3000:]
    out = json.loads(line[-1].split(" ", 1)[1])
    if not out.pop("flash_differs_from_math"):
        pytest.skip("flash and math agree bit for bit here, so routing cannot be told apart")
    assert out == {
        "wrapped": True,
        "mask_is_math": True,
        "gqa_is_math": True,
        "plain_is_flash": True,
        "grad_is_math": True,
    }


_REAL = textwrap.dedent(
    """
    import json
    import unsloth  # noqa: F401  (applies the import-time fixes)
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device = "cuda").manual_seed(0)
    q, k, v = (torch.randn(2, 4, 24, 64, device = "cuda", dtype = torch.bfloat16, generator = g).requires_grad_() for _ in range(3))
    out = {}
    for name, kw in (("causal_bwd", {"is_causal": True}),
                     ("mask_bwd", {"attn_mask": torch.ones(24, 24, device = "cuda", dtype = torch.bool).tril()}),
                     ("gqa_bwd", {"is_causal": True, "enable_gqa": True})):
        kk, vv = (k[:, :2], v[:, :2]) if name == "gqa_bwd" else (k, v)
        try:
            F.scaled_dot_product_attention(q, kk, vv, **kw).float().sum().backward()
            torch.cuda.synchronize()
            out[name] = bool(torch.isfinite(q.grad).all())
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"[:200]
        q.grad = None
    print("REAL " + json.dumps(out))
    """
)


def test_windows_rocm_training_attention_runs():
    if not (
        sys.platform == "win32"
        and getattr(torch.version, "hip", None)
        and torch.cuda.is_available()
    ):
        pytest.skip("needs a Windows ROCm GPU")
    p = subprocess.run([sys.executable, "-c", _REAL], capture_output = True, text = True, timeout = 900)
    line = [l for l in p.stdout.splitlines() if l.startswith("REAL ")]
    assert line, p.stdout[-2000:] + p.stderr[-3000:]
    assert json.loads(line[-1].split(" ", 1)[1]) == {
        "causal_bwd": True,
        "mask_bwd": True,
        "gqa_bwd": True,
    }
