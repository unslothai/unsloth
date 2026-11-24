# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""gfx11 ROCm / NVIDIA L4 decode linear (unsloth/kernels/rocm_skinny_linear.py): out = X @ W.T + s * (X @ A.T) @ B.T
for <= 16 decode rows in one Triton launch.

CPU: config heuristic, gating, the fast_linear_forward hook, and the kernel itself under TRITON_INTERPRET=1
(fp16; the interpreter has no bf16 dot). GPU (gfx1151 ROCm, or an NVIDIA card the CUDA gate takes): bf16 / fp16
accuracy against fp64, at least as close as the BLAS three-GEMM sequence it replaces, determinism, row
independence, strided outputs."""

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
    """The kernel module on its own (no unsloth import), as if on a ROCm build (hip = None: a CUDA build)."""
    # The feature flags are set explicitly: an ambient UNSLOTH_ROCM_SKINNY_LINEAR=0 must not change the tests.
    env = {
        "UNSLOTH_ROCM_SKINNY_LINEAR": "1",
        "UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG": "",
        "UNSLOTH_CUDA_SKINNY_LINEAR": "1",
        **(env or {}),
    }
    old_hip, old_cuda = torch.version.hip, torch.version.cuda
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    torch.version.hip = hip
    torch.version.cuda = (old_cuda or "12.8") if hip is None else None
    try:
        spec = importlib.util.spec_from_file_location("_rsl_under_test", KERNEL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        torch.version.hip, torch.version.cuda = old_hip, old_cuda
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _have_triton():
    return importlib.util.find_spec("triton") is not None


def test_cuda_build_loads_triton_kernel_unless_disabled():
    assert _load(hip = None, env = {"UNSLOTH_CUDA_SKINNY_LINEAR": "0"}).triton is None
    mod = _load(hip = None)
    assert (mod.triton is not None) is _have_triton()
    X = torch.randn(4, 1, 8)
    assert mod.skinny_decode_linear(X, torch.randn(6, 8), None, None, None, None) is None


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
@pytest.mark.parametrize(
    "env,cc,want",
    [
        ("1", (8, 9), True),
        ("1", (7, 5), False),
        ("1", (8, 0), False),
        ("1", (9, 0), False),
        ("1", (10, 0), False),
        ("all", (8, 0), True),
        ("all", (12, 0), True),
        ("all", (7, 5), False),
    ],
)
def test_cuda_arch_gate(monkeypatch, env, cc, want):
    mod = _load(hip = None, env = {"UNSLOTH_CUDA_SKINNY_LINEAR": env})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: cc)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert mod.skinny_linear_supported(torch.device("cuda", 0)) is want
    assert mod.skinny_linear_supported(torch.device("cpu")) is False


def test_cuda_routing_and_config_match_l4_measurements():
    mod = _load(hip = None)
    # (N, K) -> routed, from the L4 sweep (weights streamed from DRAM, LoRA r32, M = 8 / 16)
    measured = {
        (1024, 1024): True,
        (2048, 1024): True,
        (3072, 1024): True,
        (1024, 3072): True,
        (512, 2048): True,
        (9728, 2560): True,
        (4096, 4096): True,
        (4096, 12288): False,
        (12288, 4096): False,
        (14336, 4096): False,
    }
    for (N, K), want in measured.items():
        assert mod.routed(N, K) is want, (N, K)
    assert mod.CUDA_MIN_ROWS == 2
    if _have_triton():
        assert mod.pick_config(1024, 1024) == (64, 128, 4, 2, 2)
        assert mod.pick_config(151936, 4096) == (64, 128, 4, 2, 2)
        # The ROCm tuning override is not a CUDA one.
        forced = _load(hip = None, env = {"UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG": "32,128,4,2"})
        assert forced.pick_config(1024, 1024) == (64, 128, 4, 2, 2)


def test_cuda_takes_lora_calls_of_two_or_more_rows_only(monkeypatch):
    mod, ran = _gated(monkeypatch, hip = None)
    with torch.no_grad():
        X, W, A, B = _operands(M = 2)
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
        assert len(ran) == 1
        # no active LoRA (a base layer, or a folded adapter): cuBLAS
        assert mod.skinny_decode_linear(X, W, None, None, None, None) is None
        X1, W1, A1, B1 = _operands(M = 1)
        assert mod.skinny_decode_linear(X1, W1, A1, B1, 2.0, None) is None
    assert len(ran) == 1


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
    (or a CUDA build) and the kill switch unset, and returns early only when the kernel took the call."""
    src = UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    flag = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "UNSLOTH_ROCM_SKINNY_LINEAR" in ast.unparse(n.test)
    ]
    assert flag
    guard = ast.unparse(flag[0].test)
    for term in (
        "torch.version.hip is not None",
        "UNSLOTH_ROCM_SKINNY_LINEAR",
        "torch.version.cuda is not None",
        "UNSLOTH_CUDA_SKINNY_LINEAR",
    ):
        assert term in guard, term
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fast_linear_forward"
    )
    body = ast.unparse(fn)
    hook = body.index("_rocm_skinny_decode_linear(X, W, A, B, lora_S, out)")
    assert body.index("if q_len != 1:") < hook < body.index("out = torch_matmul(X, W.t(), out=out)")
    assert (
        "if W_quant is None and temp_lora is None and _ROCM_SKINNY_LINEAR and (not _is_compiling()):"
        in body
    )
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
    # The decode wrapper past the device gate runs the same kernel into a caller's buffer (one call counted).
    m.skinny_linear_supported = lambda device: True
    X3 = torch.randn(4, 1, 160).to(dt)
    W, A, B = (torch.randn(96, 160) * 0.05).to(dt), (torch.randn(8, 160) * 0.05).to(dt), (torch.randn(96, 8) * 0.05).to(dt)
    buf = torch.zeros(2, 4, 1, 96, dtype = dt)
    out = buf[1]
    with torch.no_grad():
        y = m.skinny_decode_linear(X3, W, A, B, 2.0, out)
    assert y is out and m.STATS["calls"] == 1 and (buf[0] == 0).all()
    assert torch.equal(y[:, 0], m.skinny_linear(X3[:, 0], W, A, B, 2.0))
    # A buffer autograd saved for backward: the raw write must bump its version so backward refuses it,
    # exactly like torch.matmul(out = saved).
    a = torch.ones(4, 1, 96, dtype = dt, requires_grad = True)
    saved = torch.zeros(4, 1, 96, dtype = dt)
    loss = (a * saved).sum()
    v0 = saved._version
    with torch.no_grad():
        assert m.skinny_decode_linear(X3, W, None, None, None, saved) is saved
    assert m.STATS["calls"] == 2 and saved._version > v0 and saved.abs().sum() > 0
    try:
        loss.backward()
    except RuntimeError as e:
        assert "modified by an inplace operation" in str(e), e
    else:
        raise AssertionError("backward read an overwritten saved tensor")
    # Lazily negated views keep PyTorch's path (the raw storage holds the un-negated values).
    def neg(t):
        v = torch.complex(torch.zeros_like(t), t).conj().imag
        assert v.is_neg() and torch.equal(v, -t)
        return v

    with torch.no_grad():
        X3n = neg(X3[:, 0]).unsqueeze(1)
        for args in ((X3n, W, A, B), (X3, neg(W), A, B), (X3, W, neg(A), B), (X3, W, A, neg(B))):
            assert m.skinny_decode_linear(*args, 2.0, None) is None
        assert m.skinny_decode_linear(X3, W, None, None, None, neg(torch.zeros(4, 1, 96, dtype = dt))) is None
    assert m.STATS["calls"] == 2
    print("INTERP_OK")
    """
)


def _gfx1151():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"


def _interpreter_ok():
    # Triton 3.6's interpreter cannot run the split-K loop bound (InterpreterError on a 1-element array);
    # 3.8 can. GPU hosts run the compiled kernel tests below instead.
    if not _have_triton() or _gfx1151():
        return False
    import triton
    return tuple(int(x) for x in triton.__version__.split("+")[0].split(".")[:2]) >= (3, 8)


@pytest.mark.skipif(
    not _interpreter_ok(), reason = "needs the Triton >= 3.8 interpreter on a host without gfx1151"
)
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


def _rocm_skinny_gpu():
    # The GPU tests import the real module: UNSLOTH_ROCM_SKINNY_LINEAR=0 leaves it without Triton.
    return (
        os.environ.get("UNSLOTH_ROCM_SKINNY_LINEAR", "1") != "0" and _gfx1151() and _have_triton()
    )


def _cuda_skinny_gpu():
    # The production gate on the current device, under the ambient UNSLOTH_CUDA_SKINNY_LINEAR (=0: none).
    if torch.version.hip is not None or torch.version.cuda is None or not torch.cuda.is_available():
        return False
    if not _have_triton():
        return False
    mode = os.environ.get("UNSLOTH_CUDA_SKINNY_LINEAR", "1")
    mod = _load(hip = None, env = {"UNSLOTH_CUDA_SKINNY_LINEAR": mode})
    return mod.skinny_linear_supported(torch.device("cuda"))


gpu = pytest.mark.skipif(not _rocm_skinny_gpu(), reason = "needs an enabled gfx1151 ROCm GPU")


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
@pytest.mark.parametrize(
    "hip,mode,want", [(None, "0", False), (None, "1", True), ("7.0", "0", False)]
)
def test_gpu_selectors_honour_the_kill_switches(monkeypatch, hip, mode, want):
    # An L4 (or gfx1151) host with the backend's kill switch set must not select the raw-kernel GPU tests.
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", None if hip else (torch.version.cuda or "12.8"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    props = type("P", (), {"gcnArchName": "gfx1151:sramecc-:xnack-"})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index = None: props)
    monkeypatch.setenv("UNSLOTH_CUDA_SKINNY_LINEAR", mode)
    monkeypatch.setenv("UNSLOTH_ROCM_SKINNY_LINEAR", mode)
    assert (_rocm_skinny_gpu() if hip else _cuda_skinny_gpu()) is want


@pytest.mark.skipif(
    not (_rocm_skinny_gpu() or _cuda_skinny_gpu()),
    reason = "needs an enabled gfx1151 ROCm GPU or an NVIDIA card the CUDA gate takes",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "shape",
    [(2048, 1024, 32), (1024, 3072, 16), (3072, 1024, 64), (1024, 1024, 128), (1000, 1000, 0)],
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


class _Packed(torch.Tensor):
    """A minimal traceable wrapper subclass: float dtype and a 2D logical shape over other storage (as NF4Tensor)."""

    @staticmethod
    def __new__(cls, inner):
        return torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, dtype = inner.dtype, device = inner.device
        )

    def __init__(self, inner):
        self.inner = inner

    def __tensor_flatten__(self):
        return ["inner"], None

    @staticmethod
    def __tensor_unflatten__(tensors, meta, outer_size, outer_stride):
        return _Packed(tensors["inner"])

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        unwrap = lambda t: t.inner if isinstance(t, _Packed) else t  # noqa: E731
        return func(*[unwrap(a) for a in args], **{k: unwrap(v) for k, v in (kwargs or {}).items()})


def test_wrapper_subclass_weights_keep_their_own_matmul(monkeypatch):
    # Past every device / shape gate, a plain weight reaches the kernel and a wrapper-subclass weight does not.
    mod = _load()
    ran = []
    monkeypatch.setattr(mod, "skinny_linear_supported", lambda device: True)
    monkeypatch.setattr(mod, "routed", lambda n, k: True)
    for name in ("skinny_linear", "_launch"):
        monkeypatch.setattr(mod, name, lambda *a, **k: ran.append(1), raising = False)
    if mod.triton is None:
        monkeypatch.setattr(mod, "triton", object())
    X = torch.randn(2, 1, 8, dtype = torch.bfloat16)
    W = torch.randn(6, 8, dtype = torch.bfloat16)
    with torch.no_grad():
        assert mod.skinny_decode_linear(X, W, None, None, None, None) is not None
        assert ran == [1]
        assert mod.skinny_decode_linear(X, _Packed(W), None, None, None, None) is None
    assert ran == [1]


def _gated(monkeypatch, hip = "7.0"):
    """The kernel module past every device / shape gate on CPU, skinny_linear replaced by a recorder."""
    mod = _load(hip = hip)
    ran = []
    monkeypatch.setattr(mod, "skinny_linear_supported", lambda device: True)
    for name in ("skinny_linear", "_launch"):
        monkeypatch.setattr(mod, name, lambda *a, **k: ran.append(a), raising = False)
    if mod.triton is None:
        monkeypatch.setattr(mod, "triton", object())
    return mod, ran


def _operands(
    M = 2,
    N = 6,
    K = 8,
    R = 4,
):
    bf = torch.bfloat16
    return (
        torch.randn(M, 1, K, dtype = bf),
        torch.randn(N, K, dtype = bf),
        torch.randn(R, K, dtype = bf),
        torch.randn(N, R, dtype = bf),
    )


def test_lora_operands_get_the_raw_storage_checks_of_W(monkeypatch):
    # A / B are read through raw pointers like W: wrapper subclasses, sparse and missing operands keep the
    # dispatched path, and so does a non-scalar scale.
    mod, ran = _gated(monkeypatch)
    X, W, A, B = _operands()
    with torch.no_grad():
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
        assert len(ran) == 1
        for a, b, s in (
            (_Packed(A), B, 2.0),
            (A, _Packed(B), 2.0),
            (A.to_sparse(), B, 2.0),
            (A, B.to_sparse(), 2.0),
            (A, None, 2.0),
            (A, B, torch.tensor(2.0)),
            (A, B, None),
        ):
            assert mod.skinny_decode_linear(X, W, a, b, s, None) is None
        assert mod.skinny_decode_linear(X, W.to_sparse(), None, None, None, None) is None
    assert len(ran) == 1


def test_inner_and_empty_dimensions_fall_back(monkeypatch):
    # X's width must be W's K (else the kernel reads past W), and N, K, R must be >= 1.
    mod, ran = _gated(monkeypatch)
    X, W, A, B = _operands()
    bf = torch.bfloat16
    with torch.no_grad():
        assert (
            mod.skinny_decode_linear(torch.randn(2, 1, 16, dtype = bf), W, None, None, None, None)
            is None
        )
        assert (
            mod.skinny_decode_linear(
                torch.randn(2, 1, 0, dtype = bf), W[:, :0], None, None, None, None
            )
            is None
        )
        assert mod.skinny_decode_linear(X, W[:0], None, None, None, None) is None
        assert mod.skinny_decode_linear(X, W, A[:0], B[:, :0], 2.0, None) is None
        assert (
            mod.skinny_decode_linear(torch.randn(2, 8, dtype = bf), W, None, None, None, None) is None
        )
        assert mod.skinny_decode_linear(torch.randn(()).to(bf), W, None, None, None, None) is None
    assert ran == []


def test_unsafe_out_buffers_fall_back(monkeypatch):
    # The kernel writes out while it reads X / W / A / B: aliased, self-overlapping, inference-only (outside
    # inference mode) and grad-tracked buffers keep the PyTorch path, which raises or handles them itself.
    mod, ran = _gated(monkeypatch)
    bf = torch.bfloat16
    X, W, A, B = _operands(N = 8, K = 8)
    buf = torch.zeros(2, 2, 1, 16, dtype = bf)
    flat = torch.zeros(32, dtype = bf)
    with torch.no_grad():
        # Disjoint views of one buffer are fine (temp_mlp[0] -> down_proj -> temp_mlp[1][..., :hd]).
        out = buf[1, :, :, :8]
        assert mod.skinny_decode_linear(buf[0, :, :, :8], W, None, None, None, out) is out
        assert len(ran) == 1
        bad_cases = (
            (X, X, None),  # out is X
            (flat[:16].view(2, 1, 8), flat[8:24].view(2, 1, 8), None),  # out overlaps X
            (X, torch.zeros(1, 1, 8, dtype = bf).expand(2, 1, 8), None),  # rows alias each other
            (X, W[:2].view(2, 1, 8), None),  # out is W
            (X, A[:2].view(2, 1, 8), (A, B, 2.0)),  # out is the LoRA A
        )
        for x, bad, lora in bad_cases:
            assert mod.skinny_decode_linear(x, W, *(lora or (None, None, None)), bad) is None
        with torch.inference_mode():
            frozen = torch.zeros(2, 1, 8, dtype = bf)
        assert mod.skinny_decode_linear(X, W, None, None, None, frozen) is None
        with torch.inference_mode():
            assert mod.skinny_decode_linear(X, W, None, None, None, frozen) is frozen
    with torch.enable_grad():
        tracked = torch.zeros(2, 1, 8, dtype = bf, requires_grad = True)
        assert mod.skinny_decode_linear(X, W, None, None, None, tracked) is None
    assert len(ran) == 2


def test_supplied_out_bumps_its_version_and_negated_views_fall_back(monkeypatch):
    # The kernel writes out through a raw pointer: a buffer autograd saved for backward must still be seen as
    # modified (torch.matmul(out = out) bumps the version), and lazily negated operands / outputs keep the
    # dispatched path because the raw storage holds the un-negated values.
    mod, ran = _gated(monkeypatch)
    fp = torch.float16
    X, W, A, B = (t.to(fp) for t in _operands(N = 8, K = 8))
    w = torch.ones(2, 1, 8, dtype = fp, requires_grad = True)
    saved = torch.zeros(2, 1, 8, dtype = fp)
    loss = (w * saved).sum()
    v0 = saved._version
    with torch.no_grad():
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, saved) is saved
    assert len(ran) == 1 and saved._version > v0
    with pytest.raises(RuntimeError, match = "modified by an inplace operation"):
        loss.backward()
    # A fresh output and an inference tensor (no version counter) need no bump.
    with torch.inference_mode():
        frozen = torch.zeros(2, 1, 8, dtype = fp)
        assert mod.skinny_decode_linear(X, W, None, None, None, frozen) is frozen
    assert len(ran) == 2

    def neg(t):
        v = torch.complex(torch.zeros_like(t), t).conj().imag
        assert v.is_neg() and torch.equal(v, -t)
        return v

    with torch.no_grad():
        for args in ((neg(X), W, A, B), (X, neg(W), A, B), (X, W, neg(A), B), (X, W, A, neg(B))):
            assert mod.skinny_decode_linear(*args, 2.0, None) is None
        assert (
            mod.skinny_decode_linear(X, W, None, None, None, neg(torch.zeros(2, 1, 8, dtype = fp)))
            is None
        )
    assert len(ran) == 2


def test_forward_ad_functorch_and_autocast_fall_back(monkeypatch):
    # Tangents, vmap batching and a different autocast dtype exist only on the dispatched ops.
    import torch.autograd.forward_ad as fwAD

    mod, ran = _gated(monkeypatch)
    X, W, A, B = _operands()
    with torch.no_grad():
        with fwAD.dual_level():
            dual = fwAD.make_dual(X, torch.ones_like(X))
            assert mod.skinny_decode_linear(dual, W, A, B, 2.0, None) is None
        got = []
        torch.func.vmap(
            lambda x: got.append(mod.skinny_decode_linear(x, W, A, B, 2.0, None)) or x.sum()
        )(X[None].expand(3, -1, -1, -1).contiguous())
        assert got == [None]
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
        monkeypatch.setattr(torch, "get_autocast_dtype", lambda *a: torch.float16)
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
        monkeypatch.setattr(torch, "get_autocast_dtype", lambda *a: torch.bfloat16)
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
    assert len(ran) == 1


def test_launch_failure_falls_back_and_stays_off(monkeypatch):
    # A launch that cannot compile is reported as "not taken", never raised mid-decode, and not retried.
    mod = _load()
    monkeypatch.setattr(mod, "skinny_linear_supported", lambda device: True)
    if mod.triton is None:
        monkeypatch.setattr(mod, "triton", object())
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("PassManager::run failed")

    for name in ("skinny_linear", "_launch"):
        monkeypatch.setattr(mod, name, boom, raising = False)
    X, W, A, B = _operands()
    with torch.no_grad():
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
    assert calls == [1]


@pytest.mark.parametrize(
    "text",
    ["64,64,2,0", "64,64,3,4", "48,64,2,2", "0,64,2,2", "64,64,0,2", "64,64,2,2,0", "64,512,2,2"],
)
def test_unlaunchable_forced_config_is_ignored(text):
    mod = _load(env = {"UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG": text})
    assert mod.pick_config(1024, 1024) == (64, 64, 2, 4, 2)


@pytest.mark.skipif(not _have_triton(), reason = "triton not installed")
def test_skinny_linear_rejects_shapes_it_cannot_write():
    # Direct callers: rows past 16 would be left unwritten and a short W read out of bounds.
    mod = _load()
    W = torch.randn(6, 8, dtype = torch.float16)
    for X, w in (
        (torch.randn(17, 8, dtype = torch.float16), W),
        (torch.randn(2, 16, dtype = torch.float16), W),
    ):
        with pytest.raises(ValueError, match = "needs 1 <= M <= 16"):
            mod.skinny_linear(X, w)


def test_hook_skips_the_cast_and_kernel_for_trainable_adapters_and_other_devices(monkeypatch):
    """fast_linear_forward decides before the LoRA cast: under grad, trainable adapters keep the PyTorch path even
    when a no_grad decode cached a detached cast; an unsupported device casts once (on the path below)."""
    U = pytest.importorskip("unsloth.kernels.utils")
    seen = []
    casts = []
    real_cast = U._fast_lora_weights

    def cast(a, b, dtype):
        casts.append(1)
        return real_cast(a, b, dtype)

    lin = torch.nn.Linear(8, 6, bias = False).to(torch.bfloat16)
    lin.weight.requires_grad_(False)
    lora_A = torch.nn.Parameter(torch.randn(4, 8))
    lora_B = torch.nn.Parameter(torch.randn(6, 4))
    monkeypatch.setattr(
        U, "get_lora_parameters_bias", lambda p: (lin.weight, None, lora_A, lora_B, 2.0, None)
    )
    monkeypatch.setattr(U, "_ROCM_SKINNY_LINEAR", True)
    monkeypatch.setattr(U, "_rocm_skinny_supported", lambda device: True, raising = False)
    monkeypatch.setattr(U, "_fast_lora_weights", cast)

    def fake(X, W, A, B, s, out):
        seen.append((A.requires_grad, B.requires_grad))
        return torch.zeros(X.shape[0], 1, W.shape[0], dtype = X.dtype)

    monkeypatch.setattr(U, "_rocm_skinny_decode_linear", fake, raising = False)
    X = torch.randn(2, 1, 8, dtype = torch.bfloat16)
    with torch.no_grad():
        U.fast_linear_forward(lin, X)
    assert seen == [(False, False)]
    with torch.enable_grad():
        y = U.fast_linear_forward(lin, X)
    assert seen == [(False, False)], "trainable adapters under grad reached the kernel"
    assert y.shape == (2, 1, 6)
    monkeypatch.setattr(U, "_rocm_skinny_supported", lambda device: False)
    casts.clear()
    with torch.no_grad():
        U.fast_linear_forward(lin, X)
    assert casts == [1] and len(seen) == 1


def test_row_count_is_not_specialized():
    # M (1 to MAX_ROWS rows) varies per decode batch; a specialized M compiles M == 1 and M % 16 == 0 as extra
    # variants (three per call site). Static check: the kernels only exist on a ROCm build.
    import ast
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "unsloth"
        / "kernels"
        / "rocm_skinny_linear.py"
    )
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_skinny_linear_kernel",
            "_splitk_reduce_kernel",
        ):
            names = set()
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg == "do_not_specialize":
                            names |= {e.value for e in kw.value.elts}
            found[node.name] = names
    assert found == {"_skinny_linear_kernel": {"M"}, "_splitk_reduce_kernel": {"M"}}, found


def test_launch_oom_and_device_errors_raise_and_do_not_latch(monkeypatch):
    """OOM goes to the caller's recovery (generate()'s graph-OOM eager redo), and a HIP launch error is not
    hidden: neither returns None nor turns the kernel off. Only a compile failure falls back and latches."""
    mod = _load()
    monkeypatch.setattr(mod, "skinny_linear_supported", lambda device: True)
    if mod.triton is None:
        monkeypatch.setattr(mod, "triton", object())
    X, W, A, B = _operands()
    raised = []

    def make(error):
        def boom(*a, **k):
            raised.append(type(error))
            raise error

        return boom

    for error in (
        torch.cuda.OutOfMemoryError("HIP out of memory. Tried to allocate 64.00 KiB"),
        RuntimeError("Triton Error [HIP]: Code: 1, Messsage: invalid argument"),
        RuntimeError("HIP error: operation not permitted when stream is capturing"),
    ):
        monkeypatch.setattr(mod, "_launch", make(error))
        with torch.no_grad(), pytest.raises(type(error)):
            mod.skinny_decode_linear(X, W, A, B, 2.0, None)
        assert not mod._BROKEN, error
    assert len(raised) == 3
    ran = []
    monkeypatch.setattr(mod, "_launch", lambda *a, **k: ran.append(1))
    with torch.no_grad():
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
    assert ran == [1]
    monkeypatch.setattr(mod, "_launch", make(RuntimeError("PassManager::run failed")))
    with torch.no_grad(), pytest.warns(UserWarning, match = "failed to compile"):
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
    assert mod._BROKEN == {X.device.index}


def test_rejected_decode_casts_the_adapters_once(monkeypatch):
    """A call the kernel rejects (rows, routing, a fallback check) reuses the LoRA cast it made for the kernel:
    one _fast_lora_weights lookup per projection, as without the kernel."""
    U = pytest.importorskip("unsloth.kernels.utils")
    real_cast = U._fast_lora_weights
    casts = []

    def cast(a, b, dtype):
        casts.append(1)
        return real_cast(a, b, dtype)

    lin = torch.nn.Linear(8, 6, bias = False).to(torch.bfloat16)
    lin.weight.requires_grad_(False)
    lora_A = torch.nn.Parameter(torch.randn(4, 8))
    lora_B = torch.nn.Parameter(torch.randn(6, 4))
    monkeypatch.setattr(
        U, "get_lora_parameters_bias", lambda p: (lin.weight, None, lora_A, lora_B, 2.0, None)
    )
    monkeypatch.setattr(U, "_fast_lora_weights", cast)
    X = torch.randn(3, 1, 8, dtype = torch.bfloat16)
    monkeypatch.setattr(U, "_ROCM_SKINNY_LINEAR", False)
    with torch.no_grad():
        want = U.fast_linear_forward(lin, X)
    assert casts == [1]
    monkeypatch.setattr(U, "_ROCM_SKINNY_LINEAR", True)
    monkeypatch.setattr(U, "_rocm_skinny_supported", lambda device: True, raising = False)
    monkeypatch.setattr(U, "_rocm_skinny_decode_linear", lambda *a: None, raising = False)
    casts.clear()
    with torch.no_grad():
        got = U.fast_linear_forward(lin, X)
    assert casts == [1], "the rejected call looked the LoRA cast up twice"
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)


def test_cuda_compile_failure_names_the_cuda_switch(monkeypatch):
    # The warning names the backend's own kill switch: the CUDA import gate never reads the ROCm one.
    for hip, flag, other in (
        (None, "UNSLOTH_CUDA_SKINNY_LINEAR=0", "ROCM"),
        ("7.0", "UNSLOTH_ROCM_SKINNY_LINEAR=0", "CUDA"),
    ):
        mod, _ = _gated(monkeypatch, hip = hip)

        def boom(*a, **k):
            raise RuntimeError("PassManager::run failed")

        monkeypatch.setattr(mod, "_launch", boom)
        X, W, A, B = _operands()
        with torch.no_grad(), pytest.warns(UserWarning, match = flag) as record:
            assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
        assert all(other not in str(w.message) for w in record)


def test_zero_scale_adapter_keeps_the_blas_path(monkeypatch):
    # BLAS skips the adapters at alpha = 0; the kernel would compute 0 * (NaN or an fp16 overflow) = NaN.
    for hip in (None, "7.0"):
        mod, ran = _gated(monkeypatch, hip = hip)
        X, W, A, B = _operands()
        with torch.no_grad():
            for s in (0, 0.0, -0.0):
                assert mod.skinny_decode_linear(X, W, A, B, s, None) is None
            assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
        assert len(ran) == 1


def test_function_dispatch_modes_and_tracing_keep_the_dispatched_ops(monkeypatch):
    # A mode overriding matmul, make_fx's proxy mode or a JIT trace never sees the kernel's raw-pointer write.
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.fx.experimental.proxy_tensor import make_fx

    class FnMode(TorchFunctionMode):
        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            return func(*args, **(kwargs or {}))

    class DispMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            return func(*args, **(kwargs or {}))

    mod, ran = _gated(monkeypatch, hip = None)
    X, W, A, B = _operands()
    with torch.no_grad():
        with FnMode():
            assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
        with DispMode():
            assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is None
        got = []
        make_fx(lambda x: got.append(mod.skinny_decode_linear(x, W, A, B, 2.0, None)) or x + 1)(X)
        assert got == [None]
        torch.jit.trace(
            lambda x: got.append(mod.skinny_decode_linear(x, W, A, B, 2.0, None)) or x + 1,
            (X,),
            check_trace = False,
        )
        assert got == [None, None]
        assert mod.skinny_decode_linear(X, W, A, B, 2.0, None) is not None
    assert len(ran) == 1


def test_triton_stub_is_not_a_runtime(monkeypatch):
    # With no Triton package, `import unsloth` resolves unsloth's `import triton` to unsloth_zoo's stub.
    import types

    stub = types.ModuleType("unsloth_zoo.stubs.triton_stub")
    stub.jit = lambda *a, **k: (lambda fn: fn)
    stub.language = types.ModuleType("triton.language")
    stub.language.constexpr = int
    stub.cdiv = lambda a, b: -(-a // b)
    stub.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", stub)
    monkeypatch.setitem(sys.modules, "triton.language", stub.language)
    monkeypatch.setitem(sys.modules, "triton.errors", types.ModuleType("triton.errors"))
    for hip in (None, "7.0"):
        mod = _load(hip = hip)
        assert mod.triton is None
        with pytest.raises(RuntimeError, match = "needs Triton"):
            mod.skinny_linear(torch.randn(2, 8), torch.randn(6, 8))


def test_hook_writes_a_callers_temp_lora(monkeypatch):
    """fast_linear_forward(temp_lora = buf) fills buf with X @ A.T; the fused kernel never materializes it, so
    such calls keep the PyTorch path."""
    U = pytest.importorskip("unsloth.kernels.utils")
    lin = torch.nn.Linear(8, 6, bias = False).to(torch.bfloat16)
    lin.weight.requires_grad_(False)
    lora_A = torch.randn(4, 8, dtype = torch.bfloat16)
    lora_B = torch.randn(6, 4, dtype = torch.bfloat16)
    monkeypatch.setattr(
        U, "get_lora_parameters_bias", lambda p: (lin.weight, None, lora_A, lora_B, 2.0, None)
    )
    seen = []

    def fake(X, W, A, B, s, out):
        seen.append(1)
        return torch.zeros(X.shape[0], 1, W.shape[0], dtype = X.dtype)

    monkeypatch.setattr(U, "_ROCM_SKINNY_LINEAR", True)
    monkeypatch.setattr(U, "_rocm_skinny_supported", lambda device: True, raising = False)
    monkeypatch.setattr(U, "_rocm_skinny_decode_linear", fake, raising = False)
    X = torch.randn(3, 1, 8, dtype = torch.bfloat16)
    buf = torch.full((3, 4), 7.0, dtype = torch.bfloat16)
    with torch.no_grad():
        U.fast_linear_forward(lin, X, temp_lora = buf)
        assert seen == []
        torch.testing.assert_close(buf, X[:, 0] @ lora_A.t())
        U.fast_linear_forward(lin, X)
    assert seen == [1]
