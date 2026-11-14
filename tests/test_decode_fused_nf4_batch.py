# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

from real_accelerator import has_real_cuda

REPO = Path(__file__).resolve().parents[1]

_CHECKS = textwrap.dedent(
    r"""
    import importlib, os, sys, types
    import torch
    import bitsandbytes.functional as BF
    from packaging.version import Version

    root, device, mode = sys.argv[1:]
    zoo = types.ModuleType("unsloth_zoo"); zoo.__path__ = []
    utils = types.ModuleType("unsloth_zoo.utils"); utils.Version = Version
    package = types.ModuleType("nfk")
    package.__path__ = [os.path.join(root, "unsloth", "kernels")]
    sys.modules.update({"unsloth_zoo": zoo, "unsloth_zoo.utils": utils, "nfk": package})
    F = importlib.import_module("nfk.nf4_decode_fused")
    CAP = torch.cuda.get_device_capability() if device == "cuda" else (8, 0)
    H, I, HD, NH, NK, EPS = 64, 128, 32, 2, 1, 1e-5
    g = torch.Generator().manual_seed(3407)
    compiled = mode == "compiled"
    names = ["_batch_main_kernel", "_batch_shrink_kernel", "_batch_final_kernel"]
    child = hasattr(F, "batch_qk_norm_rope_kv")
    if child:
        names.append("_batch_qk_kernel")

    def census():
        return {name: sum(len(v[0]) if isinstance(v, (tuple, list)) else len(v)
                          for v in getattr(getattr(F, name), "device_caches", getattr(getattr(F, name), "cache", {})).values())
                for name in names}

    def checked(fn):
        if not compiled:
            return fn
        run = torch.compile(fn, fullgraph=True, dynamic=True)
        def replay(*args, **kwargs):
            saved = {}
            def visit(value):
                if isinstance(value, torch.Tensor):
                    saved[id(value)] = (value, value.clone())
                elif isinstance(value, (tuple, list)):
                    for item in value: visit(item)
                elif isinstance(value, dict):
                    for item in value.values(): visit(item)
            visit(args); visit(kwargs)
            before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            result = run(*args, **kwargs)
            assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - before <= 1, fn.__name__
            graphs = dict(torch._dynamo.utils.counters["stats"])
            kernels = census()
            for step in (1, 2, 0):
                for value, backup in saved.values(): value.copy_(backup)
                pos = kwargs.get("pos")
                slot = kwargs.get("slot")
                if fn.__name__ == "batch_qk_norm_rope_kv":
                    pos, slot = args[8:10]
                if pos is not None: pos.add_(step)
                if slot is not None: slot.add_(step)
                result = run(*args, **kwargs)
                assert dict(torch._dynamo.utils.counters["stats"]) == graphs, fn.__name__
                assert census() == kernels, fn.__name__
            return result
        return replay

    if compiled:
        torch._dynamo.config.cache_size_limit = 64
    # Separate code objects avoid https://github.com/pytorch/pytorch/issues/176211.
    def qkv_plain(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    def qkv_rope(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    def o_plain(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    def o_norm(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    def down_norm(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    def gate_up(*args, **kwargs):
        return F.batch_fused_site(*args, **kwargs)
    batch_calls = {
        ("qkv", F.EPI_PLAIN): checked(qkv_plain),
        ("qkv", F.EPI_ROPE_KV): checked(qkv_rope),
        ("o", F.EPI_PLAIN): checked(o_plain),
        ("o", F.EPI_RESNORM): checked(o_norm),
        ("down", F.EPI_RESNORM): checked(down_norm),
        ("gate_up", F.EPI_SWIGLU): checked(gate_up),
    }
    batch_final = checked(F.batch_final_norm)
    batch_qk = checked(F.batch_qk_norm_rope_kv) if child else None

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=g, device="cpu").to(device=device, dtype=dtype)

    def ideal(q, state):
        packed = q.view(torch.uint8).flatten()
        nib = torch.stack((packed >> 4, packed & 15), -1).flatten().long()
        scale = state.state2.code.double()[state.absmax.long()]
        index = torch.arange(scale.numel(), device=device) // state.state2.blocksize
        scale = scale * state.state2.absmax.double()[index] + state.offset.double()
        return (state.code.double()[nib] * scale.repeat_interleave(state.blocksize)).reshape(state.shape)

    def site(kind, sizes, k, rank, batch, dtype):
        pairs = [BF.quantize_4bit(rand(n, k, dtype=dtype), quant_type="nf4",
                                 compress_statistics=True) for n in sizes]
        q, state = pairs[0]
        offsets = [s.offset.reshape(()) for _, s in pairs]
        offsets += offsets[-1:] * (3 - len(offsets))
        lora = [(rand(rank, k, dtype=dtype) / k**0.5,
                 rand(n, rank, dtype=dtype) / rank) for n in sizes] if rank else None
        bias = [rand(n, dtype=dtype) * 0.1 for n in sizes] if child and kind == "qkv" else None
        kwargs = dict(bias=bias) if child else {}
        # Built inside generate()'s inference_mode, as plans are: workspaces must stay writable after it.
        with torch.inference_mode():
            original = F.FusedSite(kind, q, state.absmax, state.code,
                torch.tensor(sizes).cumsum(0).tolist(), k, state.blocksize,
                code2=state.state2.code, absmax2=state.state2.absmax,
                blocksize2=state.state2.blocksize, seg_offset=torch.stack(offsets),
                lora=lora, scaling=0.5,
                segments=[(w, s.absmax, s.state2.absmax) for w, s in pairs], **kwargs)
        if rank:
            assert not any(getattr(original, n).is_inference() for n in ("H", "partials", "counter")), "serial workspace"
        with torch.inference_mode():
            batched = F.BatchSite(original, batch)
        for i, (weight, quant) in enumerate(pairs):
            assert batched.W3[i] is weight and batched.scales3[i] is quant.absmax
            assert batched.state2scales3[i] is quant.state2.absmax
        if rank:
            assert batched.H.numel() == batch * 3 * batched.rank_pad
            assert batched.partials.numel() == batch * batched.n_splits * (batched.block_r + 1)
            assert batched.counter.shape == (batch,)
            for name in ("H", "partials", "counter"):
                value = getattr(batched, name)
                assert not value.is_inference() and value.data_ptr() != getattr(original, name).data_ptr()
        return original, batched, (pairs, lora, bias)

    def call(batched, st, x, out=None, **kwargs):
        if getattr(st, "bias", None) is not None:
            kwargs["bias"] = st.bias
        run = batch_calls[(st.kind, kwargs["epi"])] if batched else F.fused_site
        return run(st, x, out, capability=CAP, **kwargs)

    def plain(original, batched, data, batch, k, dtype):
        pairs, lora, bias = data
        x = rand(batch, k, dtype=dtype)
        out = torch.full((batch, original.N), float("nan"), device=device, dtype=dtype)
        call(True, batched, x, out, epi=F.EPI_PLAIN)
        assert torch.isfinite(out).all(), "batch_plain nonfinite"
        library, reference = [], []
        for i, (weight, state) in enumerate(pairs):
            stock = x @ BF.dequantize_4bit(weight, state).T
            ref = x.double() @ ideal(weight, state).T
            if lora:
                a, b = lora[i]
                stock = torch.addmm(stock, x @ a.T, b.T, alpha=0.5)
                ref += 0.5 * (x.double() @ a.double().T) @ b.double().T
            if bias is not None:
                stock += bias[i]
                ref += bias[i].double()
            library.append(stock); reference.append(ref)
        ref = torch.cat(reference, -1)
        stock = torch.cat(library, -1).double()
        bound = (stock-ref).abs().max() + 2 * torch.finfo(dtype).eps * ref.abs().max()
        assert (out.double()-ref).abs().max() <= bound, "batch_plain fp64/library bound"
        serial = torch.empty_like(out)
        for b in range(batch): call(False, original, x[b], serial[b], epi=F.EPI_PLAIN)
        assert torch.equal(out, serial), "batch_plain serial"
        if not compiled:
            # A row stride other than K (a column slice) must read the same rows.
            wide = torch.zeros(batch, 2 * k, device=device, dtype=dtype)
            wide[:, :k] = x
            strided = torch.full_like(out, float("nan"))
            call(True, batched, wide[:, :k], strided, epi=F.EPI_PLAIN)
            assert torch.equal(strided, out), "batch_plain strided input"
        if batched.lora:
            assert not torch.count_nonzero(batched.counter), "batch_plain counter"

    cells = 0
    dtypes = [torch.float16] + ([torch.bfloat16] if device == "cuda" and CAP[0] >= 8 else [])
    for dtype in dtypes:
      for rank in (0, 8):
        previous = None
        for batch in (2, 4):
            oq, q, data = site("qkv", [H, NK*HD, NK*HD], H, rank, batch, dtype)
            oo, o, _ = site("o", [H], H, rank, batch, dtype)
            og, gu, _ = site("gate_up", [I, I], H, rank, batch, dtype)
            od, down, _ = site("down", [H], I, rank, batch, dtype)
            plain(oq, q, data, batch, H, dtype)
            residual = rand(batch, H); serial_residual = residual.clone()
            gamma = 1 + 0.3 * rand(H)
            pos = torch.tensor([2, 5, 7, 3], device=device)[:batch]
            slot = torch.tensor([3], device=device)
            angles = torch.arange(16, device=device)[:, None].double() / 10000**(torch.arange(0, HD, 2, device=device).double()/HD)
            angles = torch.cat((angles, angles), -1)
            cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
            cache = torch.zeros(2, 16, batch, NK, HD, device=device, dtype=dtype)
            serial_cache = cache.clone()
            out = torch.empty(batch, H, device=device, dtype=dtype); serial = torch.empty_like(out)
            rope = dict(epi=F.EPI_ROPE_KV, pre=F.PRE_INLOOP, norm_weight=gamma, eps=EPS,
                cache_strides=(cache.stride(3), cache.stride(1)), cos=cos, sin=sin,
                pos=pos, slot=slot, head_dim=HD)
            call(True, q, residual, out, k_cache=cache[0], v_cache=cache[1], **rope)
            for b in range(batch):
                call(False, oq, serial_residual[b], serial[b], k_cache=serial_cache[0, :, b:b+1],
                     v_cache=serial_cache[1, :, b:b+1], **{**rope, "pos": pos[b:b+1]})
            assert torch.equal(out, serial) and torch.equal(cache, serial_cache), "rope"
            att = rand(batch, H, dtype=dtype)
            ss = torch.zeros_like(residual); serial_ss = torch.zeros_like(ss)
            xg = torch.empty_like(residual); serial_xg = torch.empty_like(xg)
            n = call(True, o, att, epi=F.EPI_RESNORM, residual=residual, ss_out=ss,
                     xg_out=xg, next_norm_weight=gamma)
            for b in range(batch):
                call(False, oo, att[b], epi=F.EPI_RESNORM, residual=serial_residual[b],
                     ss_out=serial_ss[b], xg_out=serial_xg[b], next_norm_weight=gamma)
            assert torch.equal(residual, serial_residual) and torch.equal(ss, serial_ss) and torch.equal(xg, serial_xg), "resnorm"
            act = torch.empty(batch, I, device=device, dtype=dtype); serial_act = torch.empty_like(act)
            call(True, gu, xg, act, epi=F.EPI_SWIGLU, pre=F.PRE_SCALED, ss_in=ss, n_ss=n, eps=EPS)
            for b in range(batch):
                call(False, og, serial_xg[b], serial_act[b], epi=F.EPI_SWIGLU,
                     pre=F.PRE_SCALED, ss_in=serial_ss[b], n_ss=n, eps=EPS)
            assert torch.equal(act, serial_act), "gate_up"
            ss.zero_(); serial_ss.zero_()
            n = call(True, down, act, epi=F.EPI_RESNORM, residual=residual, ss_out=ss,
                     xg_out=xg, next_norm_weight=gamma)
            for b in range(batch):
                call(False, od, serial_act[b], epi=F.EPI_RESNORM, residual=serial_residual[b],
                     ss_out=serial_ss[b], xg_out=serial_xg[b], next_norm_weight=gamma)
            batch_final(residual, ss, n, gamma, EPS, out)
            for b in range(batch): F.final_norm(serial_residual[b], serial_ss[b], n, gamma, EPS, serial[b])
            assert torch.equal(out, serial), "final_norm"
            if child:
                for weight_dtype in (dtype, torch.float32):
                    raw = rand(batch, H + 2*NK*HD, dtype=dtype)
                    wq, wk = 1 + 0.3*rand(HD, dtype=weight_dtype), 1 + 0.3*rand(HD, dtype=weight_dtype)
                    cache.zero_(); serial_cache.zero_()
                    batch_qk(raw, out, wq, wk, EPS, EPS, cos, sin, pos, slot, cache[0], cache[1], NH, NK, HD)
                    for b in range(batch):
                        F.qk_norm_rope_kv(raw[b], serial[b], wq, wk, EPS, EPS, cos, sin, pos[b:b+1],
                            slot, serial_cache[0, :, b:b+1], serial_cache[1, :, b:b+1], NH, NK, HD)
                    assert torch.equal(out, serial) and torch.equal(cache, serial_cache), "qk_norm"
            current = census()
            if device == "cuda":
                assert current["_batch_main_kernel"] > 0
                if previous is not None: assert current == previous, "batch-size JIT growth"
            previous = current
            if rank:
                for st in (q, o, gu, down): assert not torch.count_nonzero(st.counter), "counter must re-arm"
            cells += 1
        if rank == 8:
            split_o, split_b, split_data = site("o", [H], 512, 8, 4, dtype)
            assert split_b.n_splits > 1
            plain(split_o, split_b, split_data, 4, 512, dtype)
    if compiled:
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
        assert not torch._dynamo.utils.counters["graph_break"]
    if device == "cpu": assert not torch.cuda.is_initialized()
    print("BATCH OK", cells)
    """
)


def _run(
    device,
    compiled = False,
    negative = False,
    root = REPO,
):
    pytest.importorskip("triton")
    pytest.importorskip("bitsandbytes")
    env = dict(os.environ)
    if device == "cpu":
        env["TRITON_INTERPRET"] = "1"
    else:
        env.pop("TRITON_INTERPRET", None)
    cache = root / ".pytest_cache"
    if negative:
        cache.mkdir(exist_ok = True)
    context = (
        tempfile.TemporaryDirectory(dir = cache, prefix = "nf4-batch-")
        if negative
        else contextlib.nullcontext(None)
    )
    with context as temporary:
        source = root
        if negative:
            source = Path(temporary)
            target = source / "unsloth" / "kernels"
            target.mkdir(parents = True)
            for name in ("nf4_decode_fused.py", "nf4.py", "nf4_gemv.py", "triton_launch.py"):
                shutil.copyfile(root / "unsloth" / "kernels" / name, target / name)
            path = target / "nf4_decode_fused.py"
            code = path.read_text(encoding = "utf-8")
            start = code.index("def _batch_main_kernel(")
            offset = code.index("row = tl.program_id(1).to(tl.int64)", start)
            code = code[:offset] + code[offset:].replace(
                "row = tl.program_id(1).to(tl.int64)", "row = 0", 1
            )
            path.write_text(code, encoding = "utf-8")
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _CHECKS,
                str(source),
                device,
                "compiled" if compiled else "eager",
            ],
            env = env,
            capture_output = True,
            text = True,
            timeout = 1200,
        )


def _can_interpret():
    try:
        import triton
        return tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 7)
    except (ImportError, AttributeError, ValueError):
        return False


@pytest.mark.skipif(not _can_interpret(), reason = "Triton <3.7 cannot interpret the input loop")
@pytest.mark.parametrize("negative", [False, True])
def test_batch_decode_in_the_interpreter(negative):
    result = _run("cpu", negative = negative)
    if negative:
        assert (
            result.returncode != 0
            and "AssertionError" in result.stderr
            and "batch_plain" in result.stderr
        ), result.stdout + result.stderr
    else:
        assert result.returncode == 0 and "BATCH OK 4" in result.stdout, (
            result.stdout + result.stderr
        )


@pytest.mark.skipif(not has_real_cuda(), reason = "requires a real CUDA GPU")
@pytest.mark.skipif(torch.version.hip is not None, reason = "fused NF4 is gated off on ROCm")
@pytest.mark.parametrize("compiled", [False, True])
def test_batch_decode_on_gpu(compiled):
    result = _run("cuda", compiled = compiled)
    assert result.returncode == 0 and "BATCH OK" in result.stdout, result.stdout + result.stderr
