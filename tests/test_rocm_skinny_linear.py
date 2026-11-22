# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""gfx11 ROCm decode linear (unsloth/kernels/rocm_skinny_linear.py): out = X @ W.T + s * (X @ A.T) @ B.T for
<= 16 decode rows in one Triton launch.

CPU: config heuristic, gating, the fast_linear_forward hook, and the kernel itself under TRITON_INTERPRET=1
(fp16; the interpreter has no bf16 dot). GPU (gfx1151 ROCm only): bf16 / fp16 accuracy against fp64, at least
as close as the rocBLAS three-GEMM sequence it replaces, determinism, row independence, strided outputs."""

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "unsloth" / "kernels" / "rocm_skinny_linear.py"
UTILS = ROOT / "unsloth" / "kernels" / "utils.py"


def _load(hip = "7.0", env = None):
    """The kernel module on its own (no unsloth import), as if on a ROCm build."""
    old_hip = torch.version.hip
    old_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    torch.version.hip = hip
    try:
        spec = importlib.util.spec_from_file_location("_rsl_under_test", KERNEL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        torch.version.hip = old_hip
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _have_triton():
    return importlib.util.find_spec("triton") is not None


def test_cuda_build_never_loads_triton_kernel():
    mod = _load(hip = None)
    assert mod.triton is None
    X = torch.randn(4, 1, 8)
    assert mod.skinny_decode_linear(X, torch.randn(6, 8), None, None, None, None) is None


def test_kill_switch():
    mod = _load(env = {"UNSLOTH_ROCM_SKINNY_LINEAR": "0"})
    assert mod.triton is None


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
@pytest.mark.parametrize(
    "env,arch,want",
    [
        ("1", "gfx1151", True),
        ("1", "gfx1100", False),
        ("1", "gfx1150", False),
        ("1", "gfx942", False),
        ("1", "gfx1201", False),
        ("all", "gfx1100", True),
        ("all", "gfx942", False),
    ],
)
def test_arch_gate(monkeypatch, env, arch, want):
    mod = _load(env = {"UNSLOTH_ROCM_SKINNY_LINEAR": env})

    class Props:
        gcnArchName = arch + ":sramecc-:xnack-"

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: Props())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert mod.skinny_linear_supported(torch.device("cuda", 0)) is want
    assert mod.skinny_linear_supported(torch.device("cpu")) is False


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
def test_pick_config_measured_rule():
    mod = _load()
    # Qwen3-0.6B projections: (N, K) -> split
    for (N, K), split in {
        (2048, 1024): 4,
        (1024, 1024): 4,
        (1024, 2048): 8,
        (3072, 1024): 4,
        (1024, 3072): 8,
    }.items():
        assert mod.pick_config(N, K) == (64, 64, 2, split, 2)
    # wide outputs keep the grid bounded; tiny K never splits
    bn, bk, nw, split, st = mod.pick_config(151936, 1024)
    assert split == 1
    assert mod.pick_config(4096, 128)[3] == 1
    for N in (64, 1000, 4096, 14336, 151936):
        for K in (64, 1000, 4096, 14336):
            cfg = mod.pick_config(N, K)
            assert 1 <= cfg[3] <= 8
            assert cfg[3] == 1 or -(-N // 64) * cfg[3] <= 512


def test_routing_rule_matches_measurements():
    mod = _load()
    # (N, K) -> routed, from the gfx1151 sweep (Linux and Windows agree)
    measured = {
        (2048, 1024): True,
        (1024, 1024): True,
        (1024, 2048): True,
        (3072, 1024): True,
        (1024, 3072): True,
        (4096, 2560): True,
        (1024, 2560): True,
        (9728, 2560): True,
        (1024, 4096): True,
        (2560, 4096): False,
        (4096, 4096): False,
        (14336, 4096): False,
    }
    for (N, K), want in measured.items():
        assert mod.routed(N, K) is want, (N, K)


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
def test_forced_config_env():
    mod = _load(env = {"UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG": "32,128,4,2"})
    assert mod.pick_config(1024, 1024) == (32, 128, 4, 2)
    mod = _load(env = {"UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG": "bad"})
    assert mod.pick_config(1024, 1024) == (64, 64, 2, 4, 2)


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
def test_decode_gate_refuses_non_decode_shapes_on_cpu():
    mod = _load()
    W = torch.randn(6, 8, dtype = torch.bfloat16)
    # CPU tensors: never a gfx11 device
    assert (
        mod.skinny_decode_linear(
            torch.randn(4, 1, 8, dtype = torch.bfloat16), W, None, None, None, None
        )
        is None
    )
    # prefill (q_len > 1) and > 16 rows are not decode-shaped
    assert (
        mod.skinny_decode_linear(
            torch.randn(4, 2, 8, dtype = torch.bfloat16), W, None, None, None, None
        )
        is None
    )
    assert (
        mod.skinny_decode_linear(
            torch.randn(17, 1, 8, dtype = torch.bfloat16), W, None, None, None, None
        )
        is None
    )
    # fp32 is not routed
    assert mod.skinny_decode_linear(torch.randn(4, 1, 8), W.float(), None, None, None, None) is None


def test_fast_linear_forward_hook_is_rocm_gated():
    """The hook sits before the 16-bit decode GEMM, only behind _ROCM_SKINNY_LINEAR, which needs torch.version.hip
    and the kill switch unset, and returns early only when the kernel took the call."""
    src = UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    flag = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "UNSLOTH_ROCM_SKINNY_LINEAR" in ast.unparse(n.test)
    ]
    assert flag and "torch.version.hip is not None" in ast.unparse(flag[0].test)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fast_linear_forward"
    )
    body = ast.unparse(fn)
    hook = body.index("_rocm_skinny_decode_linear(X, W, A, B, lora_S, out)")
    assert body.index("if q_len != 1:") < hook < body.index("out = torch_matmul(X, W.t(), out=out)")
    assert "if W_quant is None and _ROCM_SKINNY_LINEAR and (not _is_compiling()):" in body
    assert "if fused is not None:" in body


_INTERP = textwrap.dedent(
    """
    import importlib.util, os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    import torch
    torch.version.hip = "7.0"
    spec = importlib.util.spec_from_file_location("rsl", sys.argv[1])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0)
    for M, N, K, R, cfg, strided in [(4, 96, 160, 32, (32, 64, 4, 1), False), (1, 50, 100, 8, (16, 32, 2, 1), True),
                                     (16, 64, 300, 16, (32, 64, 4, 3), False), (7, 33, 129, 0, (16, 64, 4, 2), True),
                                     (3, 70, 200, 16, None, True)]:
        dt = torch.float16
        X = torch.randn(M, 2 * K).to(dt)[:, ::2] if strided else torch.randn(M, K).to(dt)
        W = (torch.randn(N, K) * 0.05).to(dt)
        A = (torch.randn(R, K) * 0.05).to(dt) if R else None
        B = (torch.randn(N, R) * 0.05).to(dt) if R else None
        ref = X.double() @ W.double().T
        if R:
            ref = ref + 2.0 * (X.double() @ A.double().T) @ B.double().T
        big = torch.zeros(M, N + 7, dtype = dt)
        out = big[:, 3:3 + N] if strided else None
        y = m.skinny_linear(X, W, A, B, 2.0, out, config = cfg)
        if strided:
            assert y.data_ptr() == out.data_ptr()
            assert (big[:, :3] == 0).all() and (big[:, 3 + N:] == 0).all(), "wrote outside out"
        err = ((y.double() - ref).abs() / (ref.abs() + 1e-2)).max().item()
        assert err < 5e-3, (M, N, K, R, cfg, err)
    print("INTERP_OK")
    """
)


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
def test_kernel_under_triton_interpreter(tmp_path):
    script = tmp_path / "interp.py"
    script.write_text(_INTERP, encoding = "utf-8")
    env = dict(os.environ, TRITON_INTERPRET = "1")
    p = subprocess.run(
        [sys.executable, "-I", str(script), str(KERNEL)],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
        cwd = str(tmp_path),
    )
    assert p.returncode == 0 and "INTERP_OK" in p.stdout, (p.stdout[-2000:], p.stderr[-3000:])


def _gfx1151():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"


gpu = pytest.mark.skipif(not _gfx1151(), reason = "needs a gfx1151 ROCm GPU")


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "shape", [(2048, 1024, 32), (1024, 3072, 16), (3072, 1024, 64), (1000, 1000, 0)]
)
def test_gpu_accuracy_vs_rocblas_sequence(dtype, M, shape):
    from unsloth.kernels import rocm_skinny_linear as S

    N, K, R = shape
    g = torch.Generator(device = "cuda").manual_seed(M * 7 + N)
    X = torch.randn(M, K, device = "cuda", generator = g).to(dtype)
    W = (torch.randn(N, K, device = "cuda", generator = g) * 0.02).to(dtype)
    A = (torch.randn(R, K, device = "cuda", generator = g) * 0.03).to(dtype) if R else None
    B = (torch.randn(N, R, device = "cuda", generator = g) * 0.01).to(dtype) if R else None
    ref = X.double() @ W.double().t()
    seq = torch.matmul(X, W.t())
    if R:
        ref = ref + 2.0 * (X.double() @ A.double().t()) @ B.double().t()
        seq.addmm_(torch.mm(X, A.t()), B.t(), alpha = 2.0)
    y = S.skinny_linear(X, W, A, B, 2.0)
    den = ref.abs().mean()
    err = ((y.double() - ref).abs().mean() / den).item()
    seq_err = ((seq.double() - ref).abs().mean() / den).item()
    assert err <= seq_err * 1.05 + 1e-6, (err, seq_err)
    assert torch.equal(y, S.skinny_linear(X, W, A, B, 2.0)), "not deterministic"
    assert torch.equal(
        S.skinny_linear(X[:1], W, A, B, 2.0)[0], y[0]
    ), "row 0 depends on the other rows"


@gpu
def test_gpu_decode_linear_strided_out_and_fallbacks():
    from unsloth.kernels import rocm_skinny_linear as S

    X = torch.randn(4, 1, 512, device = "cuda", dtype = torch.bfloat16)
    W = torch.randn(384, 512, device = "cuda", dtype = torch.bfloat16) * 0.02
    buf = torch.zeros(4, 1, 1024, device = "cuda", dtype = torch.bfloat16)
    out = buf[:, :, :384]
    calls = S.STATS["calls"]
    y = S.skinny_decode_linear(X, W, None, None, None, out)
    assert y is out and S.STATS["calls"] == calls + 1
    assert (buf[:, :, 384:] == 0).all()
    torch.testing.assert_close(
        y[:, 0], (X[:, 0].float() @ W.float().t()).to(y.dtype), rtol = 2e-2, atol = 2e-2
    )
    with torch.enable_grad():
        Wg = W.clone().requires_grad_(True)
        assert S.skinny_decode_linear(X, Wg, None, None, None, None) is None
    assert S.skinny_decode_linear(X.float(), W.float(), None, None, None, None) is None
