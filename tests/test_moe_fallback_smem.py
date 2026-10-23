# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fallback MoE configs fit shared memory for float32 X / dY (default dX needed 163,840 B > L4's 101,376 B).

Byte counts are Triton metadata.shared cross-compiled on CPU (Triton 3.6-3.8)."""

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("pandas")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"

GPUS = {
    "T4": ((7, 5), 65536),
    "L4": ((8, 9), 101376),
    "RTX4090": ((8, 9), 101376),
    "A10": ((8, 6), 101376),
    "A100": ((8, 0), 166912),
    "H100": ((9, 0), 232448),
    "B200": ((10, 0), 232448),
    "RTX5090": ((12, 0), 101376),
}
F32, BF16, F16 = torch.float32, torch.bfloat16, torch.float16
DTYPES = [
    (BF16, None, BF16),
    (F16, None, F16),
    (F32, BF16, BF16),
    (F32, F16, F16),
    (F32, None, F32),
    (F32, F32, F32),
]
# (hidden, moe_intermediate)
SHAPES = [(2048, 768), (2048, 1408), (1024, 352), (768, 2048)]
OLD_HEURISTIC = ((64, 128, 128, 8, 3), (64, 64, 256, 8, 4), (64, 64, 256, 8, 4))
OLD_DEFAULT = ((128, 128, 64, 8, 3),) * 3

# Main's float32-activation configs: (capability major, kind, activation, weight, (M, N, K, num_stages)) -> bytes.
MAIN_FLOAT32_COMPILED = {
    (8, "dX", "fp32", "bf16", (128, 128, 64, 3)): 163840,  # default dX, the L4 crash
    (8, "fwd", "fp32", "bf16", (64, 128, 128, 3)): 131072,  # heuristic forward
    (8, "dX", "fp32", "bf16", (64, 64, 256, 4)): 147456,  # heuristic dX
    (8, "dX", "fp32", "fp32", (64, 64, 256, 4)): 245760,
    (8, "fwd", "fp32", "fp32", (64, 128, 128, 3)): 196608,
    (9, "dX", "fp32", "fp32", (64, 64, 256, 4)): 327680,  # over the H100 limit too
    (9, "fwd", "fp32", "fp32", (64, 128, 128, 3)): 294912,
}


@pytest.fixture
def ac(monkeypatch):
    for name, path in (
        ("unsloth", ROOT / "unsloth"),
        ("unsloth.kernels", ROOT / "unsloth" / "kernels"),
        ("unsloth.kernels.moe", MOE_DIR),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    yield module
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _device(ac, monkeypatch, capability, smem):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(ac, "_device_smem_per_block", lambda: smem)


def _gemm_dims(hidden, inter):
    return [(2 * inter, hidden), (hidden, inter)]


def _tiles(configs):
    return tuple(
        (c.BLOCK_SIZE_M, c.BLOCK_SIZE_N, c.BLOCK_SIZE_K, c.num_warps, c.num_stages) for c in configs
    )


def _pick(ac, which, n, k, act, weight):
    fn = ac._get_heuristic_configs if which == "heuristic" else ac._get_default_configs
    return fn(n_dim = n, k_dim = k, dtype = act, weight_dtype = weight)


@pytest.mark.parametrize("gpu", list(GPUS))
@pytest.mark.parametrize("act,weight,seen", DTYPES)
@pytest.mark.parametrize("which", ["heuristic", "default"])
def test_picked_configs_fit_the_device(ac, monkeypatch, gpu, act, weight, seen, which):
    capability, smem = GPUS[gpu]
    _device(ac, monkeypatch, capability, smem)
    for hidden, inter in SHAPES:
        for n, k in _gemm_dims(hidden, inter):
            for kind, c in zip(("fwd", "dX", "dW"), _pick(ac, which, n, k, act, weight)):
                need = ac._estimate_smem(kind, c, act.itemsize, seen.itemsize, capability)
                assert need <= smem, (gpu, which, kind, act, seen, n, k, _tiles([c]))
                assert n % c.BLOCK_SIZE_N == 0 and k % c.BLOCK_SIZE_K == 0
                assert min(c.BLOCK_SIZE_M, c.BLOCK_SIZE_N, c.BLOCK_SIZE_K) >= 16
                key = (
                    capability[0] if capability[0] in (8, 9) else None,
                    kind,
                    "fp32" if act == F32 else "16",
                    "fp32" if seen == F32 else ("bf16" if seen == BF16 else "16"),
                    (c.BLOCK_SIZE_M, c.BLOCK_SIZE_N, c.BLOCK_SIZE_K, c.num_stages),
                )
                compiled = MAIN_FLOAT32_COMPILED.get(key)
                assert compiled is None or compiled <= smem, (gpu, key, compiled)


@pytest.mark.parametrize("gpu", list(GPUS))
@pytest.mark.parametrize("act", [BF16, F16])
@pytest.mark.parametrize("which", ["heuristic", "default"])
def test_16bit_configs_unchanged(ac, monkeypatch, gpu, act, which):
    capability, smem = GPUS[gpu]
    for hidden, inter in SHAPES:
        for n, k in _gemm_dims(hidden, inter):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
            unfitted = _tiles(_pick(ac, which, n, k, None, None))
            _device(ac, monkeypatch, capability, smem)
            expected = list(unfitted)
            # main's only smem step-down: the 4-stage heuristic dX (122,880 B) to 3 stages on sm_8x / sm_12x.
            if which == "heuristic" and smem < 122880 and capability[0] >= 8:
                if unfitted[1] == (64, 64, 256, 8, 4):
                    expected[1] = (64, 64, 256, 8, 3)
            assert _tiles(_pick(ac, which, n, k, act, None)) == tuple(expected), (gpu, n, k)
            assert _tiles(_pick(ac, which, n, k, None, None)) == tuple(expected), (gpu, n, k)


def test_l4_float32_default_dx_no_longer_over_limit(ac, monkeypatch):
    # The configuration that crashed: autotune failed, float32 dY, bf16 W, L4.
    _device(ac, monkeypatch, (8, 9), 101376)
    _, dx, _ = _pick(ac, "default", 1536, 2048, F32, BF16)
    assert (dx.BLOCK_SIZE_M, dx.BLOCK_SIZE_N, dx.BLOCK_SIZE_K, dx.num_stages) != (128, 128, 64, 3)
    assert ac._estimate_smem("dX", dx, 4, 2, (8, 9)) <= 101376
    # Unknown weight dtype: sized for float32 W, which also fits bf16 W.
    _, dx32, _ = _pick(ac, "default", 1536, 2048, F32, None)
    assert ac._estimate_smem("dX", dx32, 4, 4, (8, 9)) <= 101376


def test_unknown_smem_or_no_cuda_keeps_configs(ac, monkeypatch):
    _device(ac, monkeypatch, (8, 9), None)
    assert _tiles(_pick(ac, "heuristic", 1536, 2048, F32, BF16)) == OLD_HEURISTIC
    assert _tiles(_pick(ac, "default", 1536, 2048, F32, BF16)) == OLD_DEFAULT
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _tiles(_pick(ac, "default", 1536, 2048, F32, BF16)) == OLD_DEFAULT


@pytest.mark.parametrize("env", ["1", "0"])
def test_get_or_autotune_passes_dtypes(ac, monkeypatch, env):
    _device(ac, monkeypatch, (8, 9), 101376)
    monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", env)
    monkeypatch.setattr(ac, "load_cached_config", lambda key: None)
    monkeypatch.setattr(ac, "save_cached_config", lambda *a, **k: None)
    seen = {}

    def tune(*args):
        seen["args"] = args
        raise RuntimeError("tuning failed")

    monkeypatch.setattr(ac, "_run_moe_autotuning", tune)
    ac.clear_cache()
    configs = ac.get_or_autotune_moe_kernels(
        num_experts = 8,
        hidden_dim = 2048,
        intermediate_dim = 768,
        top_k = 2,
        dtype = F32,
        weight_dtype = BF16,
    )
    for kind, c in zip(("fwd", "dX", "dW"), configs):
        assert ac._estimate_smem(kind, c, 4, 2, (8, 9)) <= 101376
    if env == "0":
        assert seen["args"][4] == F32 and seen["args"][-1] == BF16


def test_cache_key_unchanged_without_a_distinct_weight_dtype(ac):
    base = ac._get_cache_key(8, 2048, 768, 2, BF16, (9, 0), 8192)
    assert ac._get_cache_key(8, 2048, 768, 2, BF16, (9, 0), 8192, None) == base
    assert ac._get_cache_key(8, 2048, 768, 2, BF16, (9, 0), 8192, BF16) == base
    f32 = ac._get_cache_key(8, 2048, 768, 2, F32, (9, 0), 8192)
    assert ac._get_cache_key(8, 2048, 768, 2, F32, (9, 0), 8192, BF16) != f32


# Recompile a sample (CPU only, needs Triton's CUDA backend + ptxas).


def _compile_shared(fn, ptr_types, consts, arch, num_warps, num_stages):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    params = list(inspect.signature(fn.fn).parameters)
    signature, constexprs = {}, {}
    for i, p in enumerate(params):
        if p in ptr_types:
            signature[p] = ptr_types[p]
        elif p in ("NUM_TOKENS", "NUM_SMS"):
            signature[p] = "i32"
        else:
            signature[p] = "constexpr"
            constexprs[(i,)] = consts.get(p, inspect.signature(fn.fn).parameters[p].default)
    # A real launch specializes 16 B aligned pointers; without it cp.async is not used.
    attrs = {(i,): [["tt.divisibility", 16]] for i, p in enumerate(params) if p in ptr_types}
    kernel = triton.compile(
        ASTSource(fn, signature, constexprs, attrs),
        target = GPUTarget("cuda", arch, 32),
        options = dict(num_warps = num_warps, num_stages = num_stages),
    )
    return kernel.metadata.shared


SAMPLE_GPUS = ["L4", "A100", "H100", "B200", "RTX5090"]


@pytest.mark.parametrize("gpu", SAMPLE_GPUS)
def test_estimate_matches_compiled_kernels(ac, monkeypatch, gpu):
    try:
        from triton.backends.compiler import GPUTarget  # noqa: F401
        from triton.compiler import ASTSource  # noqa: F401
        from unsloth.kernels.moe.grouped_gemm.kernels import backward, forward
    except Exception as e:
        pytest.skip(f"Triton CUDA cross compile unavailable: {e}")
    capability, smem = GPUS[gpu]
    arch = capability[0] * 10 + capability[1]
    _device(ac, monkeypatch, capability, smem)
    names = {F32: "*fp32", BF16: "*bf16", F16: "*fp16"}
    n, k = 1536, 2048  # Qwen3-30B-A3B gate_up
    try:
        _compile_shared(
            forward._grouped_gemm_forward_kernel,
            dict(
                x_ptr = "*bf16",
                w_ptr = "*bf16",
                y_ptr = "*bf16",
                m_sizes_ptr = "*i32",
                gather_indices_ptr = "*i64",
                topk_weights_ptr = "*bf16",
            ),
            dict(
                NUM_EXPERTS = 8,
                TOPK = 2,
                N = n,
                K = k,
                BLOCK_SIZE_M = 32,
                BLOCK_SIZE_N = 32,
                BLOCK_SIZE_K = 32,
                PERMUTE_X = True,
                PERMUTE_Y = False,
            ),
            arch,
            4,
            2,
        )
    except Exception as e:
        pytest.skip(f"Triton cannot cross compile on this host: {type(e).__name__}: {e}")
    for act, weight, seen in [(BF16, None, BF16), (F32, BF16, BF16), (F32, None, F32)]:
        for which in ("heuristic", "default"):
            for kind, c in zip(("fwd", "dX", "dW"), _pick(ac, which, n, k, act, weight)):
                consts = dict(
                    NUM_EXPERTS = 8,
                    TOPK = 2,
                    N = n,
                    K = k,
                    BLOCK_SIZE_M = c.BLOCK_SIZE_M,
                    BLOCK_SIZE_N = c.BLOCK_SIZE_N,
                    BLOCK_SIZE_K = c.BLOCK_SIZE_K,
                    PERMUTE_X = True,
                    PERMUTE_Y = False,
                )
                a, w = names[act], names[seen]
                if kind == "fwd":
                    compiled = _compile_shared(
                        forward._grouped_gemm_forward_kernel,
                        dict(
                            x_ptr = a,
                            w_ptr = w,
                            y_ptr = a,
                            m_sizes_ptr = "*i32",
                            gather_indices_ptr = "*i64",
                            topk_weights_ptr = a,
                        ),
                        consts,
                        arch,
                        c.num_warps,
                        c.num_stages,
                    )
                elif kind == "dX":
                    compiled = _compile_shared(
                        backward._grouped_gemm_dX_kernel,
                        dict(
                            dY_ptr = a,
                            w_ptr = w,
                            dX_ptr = a,
                            gather_indices_ptr = "*i64",
                            m_sizes_ptr = "*i32",
                        ),
                        consts,
                        arch,
                        c.num_warps,
                        c.num_stages,
                    )
                else:
                    compiled = _compile_shared(
                        backward._grouped_gemm_dW_per_expert_kernel,
                        dict(
                            x_ptr = a,
                            dY_ptr = a,
                            dW_ptr = w,
                            m_sizes_ptr = "*i32",
                            gather_indices_ptr = "*i64",
                        ),
                        dict(consts, EXPERTS_POW2 = 8),
                        arch,
                        c.num_warps,
                        1 if act == F32 else c.num_stages,
                    )
                estimate = ac._estimate_smem(kind, c, act.itemsize, seen.itemsize, capability)
                assert compiled <= smem, (gpu, which, kind, act, seen, _tiles([c]), compiled)
                assert compiled <= estimate, (gpu, which, kind, act, seen, _tiles([c]), compiled)
