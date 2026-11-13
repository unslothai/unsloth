# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Every Triton launch passes enable_fp_fusion=True: Triton 3.9 defaults it to False.
The GPU check emulates that via TRITON_DEFAULT_FP_FUSION=0 in a fresh process."""

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

KERNELS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "kernels"

# Subscript calls that are not launches, or forward options already carrying enable_fp_fusion.
_NOT_LAUNCHES = {
    "bnb_override.py": {"_FALLBACK"},
    "int4_packed.py": {
        "compiled",
        "call",
    },  # a CompiledKernel built by warmup(enable_fp_fusion=True)
    "utils.py": {"plan"},
    "nf4.py": {"kernel"},
    "triton_launch.py": {"kernel"},
    "layernorm.py": {"kernel"},
    "rms_layernorm.py": {"wrap(kernel)"},
}
# Launch helpers forwarding options to the excluded sites above: each call is checked instead.
_LAUNCH_HELPERS = {"layernorm.py": "_launch", "rms_layernorm.py": "_launch"}


def _launch_sites():
    for path in sorted(KERNELS.rglob("*.py")):
        rel = path.relative_to(KERNELS)
        if rel.parts[0] == "moe" and rel.parts[1] in ("tests", "benchmark"):
            continue
        src = path.read_text(encoding = "utf-8")
        if "triton" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            helper = _LAUNCH_HELPERS.get(path.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == helper
            ):
                yield f"{rel}:{node.lineno}:{helper}", node
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                name = ast.get_source_segment(src, node.func.value)
                if name in _NOT_LAUNCHES.get(path.name, ()):
                    continue
                yield f"{rel}:{node.lineno}:{name}", node


def test_every_launch_requests_fp_fusion():
    sites = list(_launch_sites())
    assert len(sites) >= 25, [s for s, _ in sites]
    missing = [
        site for site, node in sites if not any(k.arg == "enable_fp_fusion" for k in node.keywords)
    ]
    assert not missing, f"Triton launches without enable_fp_fusion: {missing}"


def test_nf4_options_and_warmup_carry_fp_fusion():
    nf4 = (KERNELS / "nf4.py").read_text(encoding = "utf-8")
    assert '"enable_fp_fusion": bool(fp_fusion)' in nf4
    gemv = (KERNELS / "nf4_gemv.py").read_text(encoding = "utf-8")
    assert gemv.count("enable_fp_fusion = True") == 2
    int4 = ast.parse((KERNELS / "int4_packed.py").read_text(encoding = "utf-8"))
    warmups = [
        node
        for node in ast.walk(int4)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "warmup"
    ]
    assert warmups and all(
        any(
            k.arg == "enable_fp_fusion" and getattr(k.value, "value", None) is True
            for k in node.keywords
        )
        for node in warmups
    )


def _compiled_options(kernel):
    """enable_fp_fusion of every compiled entry of a Triton JITFunction (3.0 - 3.9 layouts)."""
    from triton.runtime.jit import JITFunction

    seen = []
    while not isinstance(kernel, JITFunction) and hasattr(kernel, "fn"):  # autotune / heuristics
        kernel = kernel.fn
    if not isinstance(kernel, JITFunction):
        return seen
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):  # Triton <= 3.2: cache[device][key]
        tables = list(cache.values())
    else:  # Triton 3.3+: device_caches[device] = (kernel_cache, ...)
        tables = [entry[0] for entry in kernel.device_caches.values()]
    for table in tables:
        for compiled in table.values():
            meta = compiled.metadata
            value = (
                meta.get("enable_fp_fusion")
                if isinstance(meta, dict)
                else getattr(meta, "enable_fp_fusion", None)
            )
            ptx = compiled.asm.get("ptx", "") if hasattr(compiled, "asm") else ""
            seen.append((value, "fma.rn" in ptx))
    return seen


def exercise(device = "cuda"):
    """Run the representative kernels once each; returns {name: output tensors}."""
    import torch
    import triton  # noqa: F401

    from unsloth.kernels import (
        cross_entropy_loss,
        fp8,
        geglu,
        nf4,
        nf4_gemv,
        rms_layernorm,
        rope_embedding,
        swiglu,
    )

    out = {}
    g = torch.Generator(device = device).manual_seed(0)
    dt = torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16
    rnd = lambda *s, dtype = dt: torch.randn(*s, generator = g, device = device, dtype = torch.float32).to(
        dtype
    )

    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(rnd(256))
    norm.variance_epsilon = 1e-6
    X = rnd(64, 256).requires_grad_()
    Y = rms_layernorm.fast_rms_layernorm(norm, X)
    Y.backward(rnd(64, 256))
    out["rmsnorm"] = (Y.detach(), X.grad, norm.weight.grad)

    Q = rnd(2, 4, 16, 64).requires_grad_()
    K = rnd(2, 2, 16, 64).requires_grad_()
    cos, sin = rnd(16, 64), rnd(16, 64)
    q, k = rope_embedding.fast_rope_embedding(Q, K, cos, sin)
    (q.float().sum() + k.float().square().sum()).backward()
    out["rope"] = (q.detach(), k.detach(), Q.grad, K.grad)
    idx = torch.arange(16, device = device).repeat(2, 1)
    q2, k2 = rope_embedding.fast_rope_embedding(Q.detach(), K.detach(), cos, sin, idx)
    out["rope_qk"] = (q2, k2)

    # Chunked path needs vocab > 65536.
    for name, vocab in (("ce", 1000), ("ce_chunked", 70000)):
        logits = rnd(2, 8, vocab).requires_grad_()
        labels = torch.randint(0, vocab, (2, 8), generator = g, device = device)
        loss = cross_entropy_loss.fast_cross_entropy_loss(logits, labels)
        loss.backward()
        out[name] = (loss.detach(), logits.grad)

    e, gg, dw = rnd(32, 512), rnd(32, 512), rnd(32, 512)
    e3, g3 = e.view(2, 16, 512), gg.view(2, 16, 512)
    out["swiglu_fwd"] = (swiglu.swiglu_fg_kernel(e3, g3),)
    out["swiglu_bwd"] = tuple(
        t.clone() for t in swiglu.swiglu_DWf_DW_dfg_kernel(dw.clone(), e.clone(), gg.clone())
    )
    out["geglu_exact_fwd"] = (geglu.geglu_exact_forward_kernel(e3, g3),)
    out["geglu_exact_bwd"] = tuple(
        t.clone() for t in geglu.geglu_exact_backward_kernel(dw.clone(), e.clone(), gg.clone())
    )
    out["geglu_approx_fwd"] = (geglu.geglu_approx_forward_kernel(e3, g3),)
    out["geglu_approx_bwd"] = tuple(
        t.clone() for t in geglu.geglu_approx_backward_kernel(dw.clone(), e.clone(), gg.clone())
    )

    # Hand-built NF4 state so bitsandbytes is not needed.
    n, kk, bs, bs2 = 256, 512, 64, 256
    W_u8 = torch.randint(
        0, 256, (n * kk // 2, 1), generator = g, device = device, dtype = torch.int32
    ).to(torch.uint8)
    n_blocks = n * kk // bs
    code = torch.linspace(-1, 1, 16, device = device)
    code2 = torch.linspace(-1, 1, 256, device = device)
    absmax_q = torch.randint(0, 256, (n_blocks,), generator = g, device = device, dtype = torch.int32).to(
        torch.uint8
    )
    absmax2 = torch.rand(n_blocks // bs2, generator = g, device = device) + 0.5
    offset = torch.tensor(0.1, device = device)
    args = (W_u8, absmax_q, code2, absmax2, offset, code, bs, bs2)
    out["nf4_dequant_eager"] = (nf4.dequantize_nf4(*args, (n, kk), dt),)
    out["nf4_dequant_op"] = (torch.ops.unsloth.dequantize_nf4(*args, [n, kk], dt),)
    Xv = rnd(1, 1, kk)
    out["nf4_gemv"] = (nf4_gemv.gemv_nf4(Xv, *args, (n, kk), dt),)

    Wf = (rnd(512, 1024, dtype = torch.float32) / 4).to(torch.float8_e4m3fn)
    if not fp8._fp8_kernel_unsupported(Wf):
        scale = torch.rand(512, 1, generator = g, device = device) + 0.5
        out["fp8_gemv"] = (fp8.fp8_rowwise_gemv(rnd(1, 1, 1024), Wf, scale),)
    torch.cuda.synchronize()

    modules = (cross_entropy_loss, fp8, geglu, nf4, nf4_gemv, rms_layernorm, rope_embedding, swiglu)
    meta = {}
    for module in modules:
        for attr, value in vars(module).items():
            options = (
                _compiled_options(value) if hasattr(value, "run") or hasattr(value, "fn") else []
            )
            if options:
                meta[f"{module.__name__.rsplit('.', 1)[-1]}.{attr}"] = options
    return out, meta


def _child():
    import torch

    import unsloth  # noqa: F401

    from unsloth.kernels import nf4

    _, meta = exercise()
    print(
        "A1META "
        + json.dumps(
            {"meta": meta, "hip": bool(torch.version.hip), "mul_rn": bool(nf4._HAS_MUL_RN)}
        )
    )


def _run_child(tmp_path, default):
    env = dict(os.environ)
    env["TRITON_DEFAULT_FP_FUSION"] = default
    env["TRITON_CACHE_DIR"] = str(tmp_path / f"triton_{default}")
    env.pop("TRITON_FORCE_DISABLE_FP_FUSION", None)
    code = f"import sys; sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r}); import test_triton_fp_fusion as t; t._child()"
    proc = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 900
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("A1META ")]
    assert proc.returncode == 0 and line, proc.stdout[-3000:] + proc.stderr[-6000:]
    return json.loads(line[-1][len("A1META ") :])


def _needs_gpu():
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).parent / "_shared"))
        from real_accelerator import has_real_cuda
        real = has_real_cuda()
    except Exception:
        import torch
        real = torch.cuda.is_available()
    if not real:
        pytest.skip("needs a CUDA / ROCm GPU")
    pytest.importorskip("triton")


@pytest.mark.parametrize("default", ["0", "1"])
def test_compiled_kernels_keep_fp_fusion(tmp_path, default):
    _needs_gpu()
    result = _run_child(tmp_path, default)
    meta = result["meta"]
    expected = {
        "rms_layernorm._rms_layernorm_forward",
        "rms_layernorm._rms_layernorm_backward",
        "rope_embedding._rope_embedding_QK",
        "cross_entropy_loss._cross_entropy_forward",
        "cross_entropy_loss._chunked_cross_entropy_forward",
        "cross_entropy_loss._cross_entropy_backward",
        "swiglu._fg_kernel",
        "swiglu._DWf_DW_dfg_kernel",
        "geglu._exact_forward_kernel",
        "geglu._exact_backward_kernel",
        "geglu._approx_forward_kernel",
        "geglu._approx_backward_kernel",
        "nf4._nf4_dequant_kernel",
    }
    # Multi-row RMSNorm kernels replace the one-row ones on CUDA for narrow rows.
    if "rms_layernorm._rms_layernorm_forward_rows" in meta:
        expected.discard("rms_layernorm._rms_layernorm_forward")
    if "rms_layernorm._rms_layernorm_backward_rows" in meta:
        expected.discard("rms_layernorm._rms_layernorm_backward")
    assert expected <= set(meta), sorted(expected - set(meta))
    assert any(name.startswith("nf4_gemv.") for name in meta), sorted(meta)

    nf4_values = {value for value, _ in meta.pop("nf4._nf4_dequant_kernel")}
    # Eager dequant stays deliberately unfused; the traced op fuses where mul.rn exists.
    assert False in nf4_values
    assert (True in nf4_values) == result["mul_rn"], nf4_values
    bad = {name: opts for name, opts in meta.items() if any(value is not True for value, _ in opts)}
    assert not bad, f"compiled without enable_fp_fusion=True: {bad}"
    if not result["hip"]:
        # Contraction really happened: fma.rn in PTX.
        for name in (
            "rope_embedding._rope_embedding_QK",
            "geglu._exact_forward_kernel",
            "swiglu._DWf_DW_dfg_kernel",
        ):
            assert all(fma for _, fma in meta[name]), (name, meta[name])
