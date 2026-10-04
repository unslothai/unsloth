# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""On Windows ROCm (torch 2.11.0+rocm7.14.1, gfx1151) flash and memory-efficient SDPA fail with
hipErrorInvalidValue on every call. `import unsloth` must turn off exactly the backends that fail,
so attention runs on the math kernel. The failed launch only surfaces at the next checked kernel
launch, so the probe must not blame a later call. The last two tests need the real host."""

import json
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _on_windows_rocm():
    if sys.platform != "win32" or not torch.cuda.is_available():
        return False
    # Same detection as the fix: some AMD wheels tag only the torch version.
    from unsloth.import_fixes import _is_rocm_torch_build
    return bool(getattr(torch.version, "hip", None)) or _is_rocm_torch_build()


def _run(code, *args):
    p = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output = True, text = True, timeout = 900
    )
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, p.stdout[-2000:] + p.stderr[-3000:]
    return json.loads(line[-1].split(" ", 1)[1])


def test_probe_is_windows_rocm_only():
    if _on_windows_rocm():
        pytest.skip(reason = "this host is the Windows ROCm one the probe targets")
    from unsloth.import_fixes import _rocm_windows_broken_sdpa_backends
    assert _rocm_windows_broken_sdpa_backends() == []


@pytest.mark.parametrize(
    "broken, expected",
    [
        ([], [True, True, True]),
        (["flash"], [False, True, True]),
        (["mem_efficient"], [True, False, True]),
        (["flash", "mem_efficient"], [False, False, True]),
    ],
)
def test_only_broken_backends_are_turned_off(monkeypatch, broken, expected):
    import unsloth.import_fixes as fixes

    cuda = torch.backends.cuda
    saved = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled())
    monkeypatch.delenv("UNSLOTH_ALLOW_ROCM_FUSED_SDPA", raising = False)
    monkeypatch.setattr(fixes, "_rocm_windows_broken_sdpa_backends", lambda: list(broken))
    try:
        cuda.enable_flash_sdp(True)
        cuda.enable_mem_efficient_sdp(True)
        fixes.fix_rocm_windows_fused_sdpa()
        got = [cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled()]
    finally:
        cuda.enable_flash_sdp(saved[0])
        cuda.enable_mem_efficient_sdp(saved[1])
    assert got == expected


def test_opt_out_skips_the_probe(monkeypatch):
    import unsloth.import_fixes as fixes

    monkeypatch.setenv("UNSLOTH_ALLOW_ROCM_FUSED_SDPA", "1")
    monkeypatch.setattr(
        fixes,
        "_rocm_windows_broken_sdpa_backends",
        lambda: pytest.fail("probed despite the opt-out"),
    )
    fixes.fix_rocm_windows_fused_sdpa()


_PROBE_TWICE = textwrap.dedent(
    """
    import json
    import torch
    from unsloth.import_fixes import _rocm_windows_broken_sdpa_backends
    first = _rocm_windows_broken_sdpa_backends()
    with torch.inference_mode():  # an import inside inference_mode must probe the same
        second = _rocm_windows_broken_sdpa_backends()
    # Nothing may be left pending: a stale launch error would raise on this checked kernel.
    torch.ones(4, device = "cuda").float().sum().item()
    print("RESULT " + json.dumps([first, second]))
    """
)


def test_windows_rocm_probe_is_stable_and_leaves_no_pending_error():
    if not _on_windows_rocm():
        pytest.skip(reason = "needs a Windows ROCm GPU to reproduce the launch failure")
    first, second = _run(_PROBE_TWICE)
    assert first == second


_ATTENTION = textwrap.dedent(
    """
    import json
    import unsloth  # noqa: F401  (applies the import-time fixes)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    g = torch.Generator(device = "cuda").manual_seed(0)
    # fp16: gfx10 claims bf16 it lacks, and fp16 exercises the same fused kernels.
    q, k, v = (torch.randn(2, 4, 24, 64, device = "cuda", dtype = torch.float16, generator = g)
               for _ in range(3))
    mask = torch.ones(24, 24, device = "cuda", dtype = torch.bool).tril()
    out = {}
    for name, grad, kw, nkv in (
        ("causal_fwd", False, {"is_causal": True}, 4),
        ("mask_fwd", False, {"attn_mask": mask}, 4),
        ("causal_bwd", True, {"is_causal": True}, 4),
        ("mask_bwd", True, {"attn_mask": mask}, 4),
        ("gqa_bwd", True, {"is_causal": True, "enable_gqa": True}, 2),
    ):
        def run(math):
            qq = q.clone().requires_grad_(grad)
            if math:
                with sdpa_kernel([SDPBackend.MATH]):
                    o = F.scaled_dot_product_attention(qq, k[:, :nkv], v[:, :nkv], **kw)
            else:
                o = F.scaled_dot_product_attention(qq, k[:, :nkv], v[:, :nkv], **kw)
            o = o.float()  # a checked launch: a failed attention launch raises here
            if grad:
                o.sum().backward()
                return o.detach(), qq.grad.float()
            return o, o
        try:
            o, d = run(False)
            ro, rd = run(True)
            out[name] = bool(torch.allclose(o, ro, atol = 2e-2, rtol = 2e-2)
                             and torch.allclose(d, rd, atol = 2e-2, rtol = 2e-2))
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"[:200]
    print("RESULT " + json.dumps(out))
    """
)


def test_windows_rocm_attention_runs_after_import():
    if not _on_windows_rocm():
        pytest.skip(reason = "needs a Windows ROCm GPU to reproduce the launch failure")
    assert _run(_ATTENTION) == {
        "causal_fwd": True,
        "mask_fwd": True,
        "causal_bwd": True,
        "mask_bwd": True,
        "gqa_bwd": True,
    }
