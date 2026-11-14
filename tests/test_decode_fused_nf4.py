# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused batch-1 NF4 decode: each call site vs an fp64 oracle, in the Triton interpreter and on GPU."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from real_accelerator import has_real_cuda

REPO = pathlib.Path(__file__).resolve().parents[1]

_CHECKS = textwrap.dedent(
    """
    import importlib, os, sys, types
    import torch
    root, device = sys.argv[1], sys.argv[2]
    negative = "break" in sys.argv[3:]
    separate = "separate" in sys.argv[3:]
    import triton
    from packaging.version import Version
    zoo = types.ModuleType("unsloth_zoo"); zoo.__path__ = []
    zoo_utils = types.ModuleType("unsloth_zoo.utils"); zoo_utils.Version = Version
    sys.modules["unsloth_zoo"] = zoo; sys.modules["unsloth_zoo.utils"] = zoo_utils
    pkg = types.ModuleType("nfk"); pkg.__path__ = [os.path.join(root, "unsloth", "kernels")]
    sys.modules["nfk"] = pkg
    F = importlib.import_module("nfk.nf4_decode_fused")
    import bitsandbytes.functional as BF

    H, NH, D, NKV, I, R, L = 256, 2, 64, 1, 512, 8, 16
    EPS = 1e-5
    CAP = torch.cuda.get_device_capability() if device == "cuda" else (8, 0)

    def quant(n, k, g):
        W = (torch.randn(n, k, generator = g) / k ** 0.5).to(device)
        q, s = BF.quantize_4bit(W.to(torch.bfloat16), quant_type = "nf4", compress_statistics = True)
        return q, s

    def dequant64(q, s):
        n, k = s.shape
        p = q.view(torch.uint8).reshape(-1)
        nib = torch.stack((p >> 4, p & 15), dim = -1).reshape(-1).long()
        a = s.state2.code[s.absmax.long()] * s.state2.absmax.repeat_interleave(s.state2.blocksize)[: s.absmax.numel()]
        a = a + s.offset
        w = s.code.double()[nib].reshape(-1, s.blocksize) * a.double()[:, None]
        return w.reshape(n, k)

    def site(kind, shapes, K, g, rank = 0):
        qs = [quant(n, K, g) for n in shapes]
        packed = torch.cat([q.reshape(-1) for q, _ in qs])
        absmax = torch.cat([s.absmax for _, s in qs])
        absmax2 = torch.cat([s.state2.absmax for _, s in qs])
        offs = [s.offset.reshape(()).float() for _, s in qs]
        offs += [offs[-1]] * (3 - len(offs))
        bounds = list(torch.tensor(shapes).cumsum(0).tolist())
        lora = None
        if rank:
            lora = [((torch.randn(rank, K, generator = g) / K ** 0.5).to(device),
                     (torch.randn(n, rank, generator = g) / rank).to(device)) for n in shapes]
        s0 = qs[0][1]
        st = F.FusedSite(kind, packed, absmax, s0.code.float(), bounds, K, s0.blocksize, code2 = s0.state2.code.float(),
                         absmax2 = absmax2, seg_offset = torch.stack(offs).contiguous(), blocksize2 = s0.state2.blocksize,
                         lora = lora, scaling = 0.5,
                         segments = [(q, s.absmax, s.state2.absmax) for q, s in qs] if separate else None)
        assert st.separate == (separate and len(shapes) > 1)
        W64 = torch.cat([dequant64(q, s) for q, s in qs])
        if rank:
            A = torch.cat([a for a, _ in lora]).double()
            B = torch.block_diag(*[b for _, b in lora]).double()
            W64 = W64 + 0.5 * B @ A
        if negative:
            st.seg_offset = st.seg_offset.flip(0).contiguous()
        return st, W64

    def rel(a, ref):
        return ((a.double() - ref).abs().max() / ref.abs().max()).item()

    def rot(x, c, s):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat((x1 * c[:h] - x2 * s[:h], x2 * c[h:] + x1 * s[h:]), dim = -1)

    def rms64(h, w):
        h = h.double()
        return h * torch.rsqrt(h.pow(2).mean() + EPS) * w.double()

    for dt, tol in ((torch.bfloat16, 2e-2), (torch.float16, 4e-3)):
        for rank in (0, R):
            g = torch.Generator().manual_seed(7 + rank)
            qkv, Wqkv = site("qkv", [NH * D, NKV * D, NKV * D], H, g, rank)
            o, Wo = site("o", [H], NH * D, g, rank)
            gu, Wgu = site("gate_up", [I, I], H, g, rank)
            down, Wd = site("down", [H], I, g, rank)
            ln1 = (1 + 0.1 * torch.randn(H, generator = g)).to(dt).to(device)
            ln2 = (1 + 0.1 * torch.randn(H, generator = g)).to(dt).to(device)
            lnf = (1 + 0.1 * torch.randn(H, generator = g)).to(dt).to(device)
            emb = torch.randn(H, generator = g).to(dt).to(device)
            ang = torch.arange(L)[:, None].double() * (1.0 / 10000 ** (torch.arange(0, D, 2).double() / D))[None]
            ang = torch.cat((ang, ang), -1)
            cos, sin = ang.cos().to(dt).to(device), ang.sin().to(dt).to(device)
            pos_i, slot_i = 5, 9
            pos = torch.tensor([pos_i], device = device)
            slot = torch.tensor([slot_i], device = device)
            paged = torch.zeros((L, 2, 1, NKV, D), dtype = dt, device = device)
            kc, vc = paged[:, 0], paged[:, 1]
            q_out = torch.empty(NH * D, dtype = dt, device = device)
            residual = emb.float().clone()
            ss = (torch.zeros(H, device = device), torch.zeros(H, device = device))
            xg = torch.empty(H, dtype = torch.float32, device = device)  # as the driver allocates it
            act = torch.empty(I, dtype = dt, device = device)

            # qkv, layer 0: norm in the loop over the embedding, RoPE, K / V into cache row `slot`.
            F.fused_site(qkv, residual, q_out, epi = F.EPI_ROPE_KV, pre = F.PRE_INLOOP, capability = CAP,
                         norm_weight = ln1, eps = EPS, k_cache = kc, v_cache = vc,
                         cache_strides = (kc.stride(2), kc.stride(0)), cos = cos, sin = sin, pos = pos, slot = slot,
                         head_dim = D)
            xn = rms64(emb, ln1)
            y = Wqkv @ xn
            c64, s64 = cos[pos_i].double(), sin[pos_i].double()
            qr = rot(y[: NH * D].view(NH, D), c64, s64).reshape(-1)
            kr = rot(y[NH * D : NH * D + NKV * D].view(NKV, D), c64, s64)
            vr = y[NH * D + NKV * D :].view(NKV, D)
            assert rel(q_out, qr) < tol, ("q", dt, rank, rel(q_out, qr))
            assert rel(kc[slot_i, 0], kr) < tol, ("k", dt, rank, rel(kc[slot_i, 0], kr))
            assert rel(vc[slot_i, 0], vr) < tol, ("v", dt, rank)
            assert kc[slot_i - 1].abs().max() == 0 and kc[pos_i].abs().max() == 0, "cache row must be `slot`"
            if rank:
                assert int(qkv.counter.item()) == 0, "the shrink counter must re-arm"

            # o: residual add in place, xg = h * ln2, sum(h^2) partials.
            attn = torch.randn(NH * D, generator = g).to(dt).to(device)
            h_before = residual.clone()
            n1 = F.fused_site(o, attn, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[1], xg_out = xg,
                              next_norm_weight = ln2, capability = CAP, eps = EPS)
            h_ref = h_before.double() + Wo @ attn.double()
            assert rel(residual, h_ref) < tol, ("o", dt, rank)
            torch.testing.assert_close(ss[1][:n1].sum().double(), residual.double().pow(2).sum(), rtol = 1e-4, atol = 0)
            # One rounding of h * ln2 (the interpreter's bf16 cast may truncate: 1 ulp).
            ulp = 2.0 ** -7 if dt == torch.bfloat16 else 2.0 ** -10
            torch.testing.assert_close(xg.float(), residual * ln2.float(), rtol = ulp, atol = 1e-6)

            # gate_up: norm from the partials, SwiGLU.
            F.fused_site(gu, xg, act, epi = F.EPI_SWIGLU, pre = F.PRE_SCALED, ss_in = ss[1], n_ss = n1,
                         capability = CAP, eps = EPS)
            gy = Wgu @ rms64(residual, ln2)
            act_ref = torch.nn.functional.silu(gy[:I]) * gy[I:]
            assert rel(act, act_ref) < tol, ("gate_up", dt, rank, rel(act, act_ref))

            # down, then the final norm from its partials.
            h_before = residual.clone()
            n2 = F.fused_site(down, act, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[0], xg_out = xg,
                              next_norm_weight = lnf, capability = CAP, eps = EPS)
            assert rel(residual, h_before.double() + Wd @ act.double()) < tol, ("down", dt, rank)
            out = torch.empty(H, dtype = dt, device = device)
            F.final_norm(residual, ss[0], n2, lnf, EPS, out)
            assert rel(out, rms64(residual, lnf)) < tol, ("final", dt, rank)

            # Layer 1's qkv over xg with the norm scaled from the down partials.
            F.fused_site(qkv, xg, q_out, epi = F.EPI_ROPE_KV, pre = F.PRE_SCALED, ss_in = ss[0], n_ss = n2,
                         capability = CAP, eps = EPS, k_cache = kc, v_cache = vc,
                         cache_strides = (kc.stride(2), kc.stride(0)), cos = cos, sin = sin, pos = pos, slot = slot,
                         head_dim = D)
            y = Wqkv @ rms64(residual, lnf)
            assert rel(q_out, rot(y[: NH * D].view(NH, D), c64, s64).reshape(-1)) < tol, ("q scaled", dt, rank)

            if dt == torch.float16:
                # h * g above the fp16 range before rstd is applied: xg must hold it (finite after the norm).
                residual.fill_(40000.0)
                big = torch.full((H,), 2.0, dtype = dt, device = device)
                n3 = F.fused_site(o, attn, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[1], xg_out = xg,
                                  next_norm_weight = big, capability = CAP, eps = EPS)
                assert torch.isfinite(xg).all(), "xg overflowed"
                F.fused_site(gu, xg, act, epi = F.EPI_SWIGLU, pre = F.PRE_SCALED, ss_in = ss[1], n_ss = n3,
                             capability = CAP, eps = EPS)
                gy = Wgu @ rms64(residual, big)
                act_ref = torch.nn.functional.silu(gy[:I]) * gy[I:]
                assert torch.isfinite(act).all() and rel(act, act_ref) < tol, ("big residual", rank)
    print("FUSED OK")
    """
)


def _run(
    device,
    negative = False,
    separate = False,
):
    pytest.importorskip("triton")
    pytest.importorskip("bitsandbytes")
    env = dict(os.environ)
    if device == "cpu":
        env["TRITON_INTERPRET"] = "1"
    else:
        env.pop("TRITON_INTERPRET", None)
    argv = [sys.executable, "-c", _CHECKS, str(REPO), device] + (["break"] if negative else [])
    if separate:
        argv.append("separate")
    return subprocess.run(argv, env = env, capture_output = True, text = True, timeout = 1200)


def _triton_release():
    try:
        import triton
    except Exception:
        return (0, 0)
    return tuple(int(x) for x in triton.__version__.split(".")[:2])


@pytest.mark.skipif(
    _triton_release() < (3, 7),
    reason = "the Triton interpreter before 3.7 cannot run the K loop (int() of a 1-element array)",
)
@pytest.mark.parametrize("negative_control", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_call_sites_match_fp64_in_the_interpreter(negative_control, separate):
    proc = _run("cpu", negative_control, separate)
    if negative_control:
        assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stdout + proc.stderr
    else:
        assert proc.returncode == 0 and "FUSED OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
@pytest.mark.skipif(torch.version.hip is not None, reason = "the fused path is gated off on ROCm")
@pytest.mark.parametrize("negative_control", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_call_sites_match_fp64_on_gpu(negative_control, separate):
    proc = _run("cuda", negative_control, separate)
    if negative_control:
        assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stdout + proc.stderr
    else:
        assert proc.returncode == 0 and "FUSED OK" in proc.stdout, proc.stdout + proc.stderr


def _driver():
    """unsloth/models/_decode_fused_nf4.py loaded by path, its kernel / utils imports stubbed."""
    root = REPO / "unsloth"
    pkg = types.ModuleType("dfn")
    pkg.__path__ = []
    models = types.ModuleType("dfn.models")
    models.__path__ = [str(root / "models")]
    kernels = types.ModuleType("dfn.kernels")
    kernels.__path__ = []
    kernels.nf4_decode_fused = types.ModuleType("dfn.kernels.nf4_decode_fused")
    kernels.utils = types.ModuleType("dfn.kernels.utils")
    kernels.utils._USE_NF4_KERNELS = True
    saved = {k: sys.modules.get(k) for k in ("dfn", "dfn.models", "dfn.kernels")}
    sys.modules.update({"dfn": pkg, "dfn.models": models, "dfn.kernels": kernels})
    sys.modules["dfn.kernels.nf4_decode_fused"] = kernels.nf4_decode_fused
    sys.modules["dfn.kernels.utils"] = kernels.utils
    spec = importlib.util.spec_from_file_location(
        "dfn.models._decode_fused_nf4", root / "models" / "_decode_fused_nf4.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
    return module


def test_off_unless_requested(monkeypatch):
    D = _driver()
    monkeypatch.delenv("UNSLOTH_DECODE_FUSED_NF4", raising = False)
    assert not D.requested()
    for value, expected in (("0", False), ("1", True), ("force", True), ("yes", False)):
        monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", value)
        assert D.requested() is expected


def test_disabled_nf4_backend_turns_the_fused_path_off(monkeypatch):
    # UNSLOTH_BNB_TRITON=0 / failed Triton NF4 kernels: no plan built, a cached plan not executed,
    # and the graph key no longer matches graphs captured while fused.
    D = _driver()
    monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", "1")
    monkeypatch.delenv("UNSLOTH_BNB_TRITON", raising = False)
    assert D.requested() and D.graph_key() == "1"

    class Cached:
        def __getattr__(self, name):
            raise AssertionError("cached fused plan used while the NF4 backend is disabled")

    model = types.SimpleNamespace(__dict__ = {D._PLAN: Cached(), D._BATCH_PLAN: Cached()})
    X = torch.zeros(1, 1, 8)
    for disable in ("env", "kernels"):
        if disable == "env":
            monkeypatch.setenv("UNSLOTH_BNB_TRITON", "0")
        else:
            monkeypatch.delenv("UNSLOTH_BNB_TRITON")
            monkeypatch.setattr(D.KU, "_USE_NF4_KERNELS", False)
        assert not D.requested() and D.graph_key() == "0"
        with torch.no_grad():
            assert D._decode_forward(model, X, None, None, None, 1, None, 256) is None


@pytest.mark.parametrize("release", ["release", "release_for_training"])
def test_decode_graph_release_frees_fused_plans(release):
    # for_training / the first training forward / unload call these: the plans' scratch buffers
    # and weight references must go with the captured graphs.
    import weakref

    D = _driver()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    held = torch.zeros(8)
    ref = weakref.ref(held)
    for name in (D._PLAN, D._BATCH_PLAN, D._HOMES):
        model[0].__dict__[name] = ("plan", held)
    del held
    getattr(D._decode_graphs, release)(model)
    assert not any(name in model[0].__dict__ for name in (D._PLAN, D._BATCH_PLAN, D._HOMES))
    assert ref() is None


def test_replaced_attention_or_mlp_blocks_make_the_plan_stale(monkeypatch):
    D, plan, _ = _norm_plan(monkeypatch)
    for name in ("self_attn", "mlp"):
        owner, _, held = next(b for b in plan.bindings if b[1] == name)
        assert not D._plan_stale(plan)
        setattr(owner, name, types.SimpleNamespace())
        assert D._plan_stale(plan)
        setattr(owner, name, held)


def test_replaced_rope_makes_the_plan_stale(monkeypatch):
    D, plan, _ = _norm_plan(monkeypatch)
    entry = plan.layers[0]
    assert not D._plan_stale(plan)
    entry.attn.rotary_emb = object()
    assert D._plan_stale(plan)
    entry.attn.rotary_emb = entry.rope
    assert not D._plan_stale(plan)


def test_a_rebuilt_plan_over_a_replaced_attention_is_not_reused(monkeypatch):
    # A replaced attention block holding the same projection tensors gives an equal signature and equal tables:
    # the old plan (its attention, cache and RoPE) must not be kept with the new bindings copied in.
    D, plan, _ = _norm_plan(monkeypatch)
    monkeypatch.setattr(D, "_same_tables", lambda a, b: True)
    plan.signature = ("same",)

    def rebuilt(attn = None, rope = None):
        built = D._Plan()
        built.signature = ("same",)
        built.layers = []
        for entry in plan.layers:
            layer = D._Layer()
            layer.attn = entry.attn if attn is None else attn
            layer.rope = entry.rope if rope is None else rope
            built.layers.append(layer)
        return built

    assert D._reusable(plan, rebuilt())
    assert not D._reusable(plan, rebuilt(attn = types.SimpleNamespace()))
    assert not D._reusable(plan, rebuilt(rope = object()))
    shorter = rebuilt()
    shorter.layers = shorter.layers[:-1]
    assert not D._reusable(plan, shorter)


def test_every_segment_must_share_the_launch_device(monkeypatch):
    # q on cuda:0 and k on cuda:1: the kernel would read k through a cuda:0 launch.
    from torch._subclasses.fake_tensor import FakeTensorMode

    D = _driver()
    with FakeTensorMode():
        parts = []
        for index in (0, 1):
            device = torch.device("cuda", index)
            qs = types.SimpleNamespace(
                state2 = None,
                blocksize = 64,
                absmax = torch.empty(4, device = device),
                code = torch.empty(16),
            )
            parts.append((torch.empty(64, 1, dtype = torch.uint8, device = device), qs, 8, None))
    monkeypatch.setattr(D, "_proj_parts", lambda proj, K: parts.pop(0))
    assert (
        D._site(types.SimpleNamespace(), (0, "qkv"), "qkv", [object(), object()], 16)
        == "several devices"
    )


def test_empty_layer_list_keeps_the_regular_path(monkeypatch):
    D = _driver()
    monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", "1")
    model = types.SimpleNamespace(model = types.SimpleNamespace(layers = []), __dict__ = {})
    model.model.layers = []
    with torch.no_grad():
        assert (
            D._decode_forward(model, torch.zeros(1, 1, 8), None, None, None, 1, None, 256) is None
        )


def test_step_runs_on_the_model_device(monkeypatch):
    # Triton launches on the current device: a model on cuda:1 while cuda:0 is current must switch.
    D = _driver()
    seen = []

    class Device:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            seen.append(("enter", self.device))

        def __exit__(self, *exc):
            seen.append(("exit", self.device))

    monkeypatch.setattr(D.torch.cuda, "device", Device)
    monkeypatch.setattr(
        D, "_decode_forward", lambda model, X, *a: seen.append(("step", X.device)) or "out"
    )
    X = types.SimpleNamespace(device = torch.device("cuda", 1))
    assert D.decode_forward(None, X, 1, 2) == "out"
    cuda1 = torch.device("cuda", 1)
    assert seen == [("enter", cuda1), ("step", cuda1), ("exit", cuda1)]
    assert D.decode_forward(None, types.SimpleNamespace(device = torch.device("cpu"))) is None


@pytest.mark.parametrize("move_child", [False, True])
def test_device_moves_release_the_fused_caches(move_child):
    # The caches hold the re-homed weight buffers: after .to() the old device's copy must be freeable.
    import weakref

    D = _driver()
    inner = torch.nn.Linear(4, 4)
    model = torch.nn.Sequential(inner)
    held = torch.zeros(8)
    ref = weakref.ref(held)
    D._track_moves(inner)
    inner.__dict__[D._PLAN] = ("plan", held)
    inner.__dict__[D._HOMES] = {0: (held,)}
    del held
    assert ref() is not None
    (inner if move_child else model).to(torch.float64)
    assert D._PLAN not in inner.__dict__ and D._HOMES not in inner.__dict__
    assert ref() is None
    assert inner.weight.dtype == torch.float64
    D._track_moves(inner)  # idempotent
    model.to(torch.float32)
    assert inner.weight.dtype == torch.float32


def test_projection_moves_release_root_caches():
    import weakref

    D = _driver()
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4, 4)))
    held = torch.zeros(8)
    ref = weakref.ref(held)
    D._track_moves(model)
    model.__dict__[D._PLAN] = held
    model.__dict__[D._HOMES] = {0: held}
    del held
    assert ref() is not None
    model[0][0].to(torch.float64)
    assert D._PLAN not in model.__dict__ and D._HOMES not in model.__dict__
    assert ref() is None
    assert model[0][0].weight.dtype == torch.float64


@pytest.mark.parametrize("serializer", ["copy", "pickle"])
def test_move_hooks_follow_restored_modules(monkeypatch, serializer):
    import copy
    import io

    D = _driver()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    D._track_moves(model)
    model.__dict__[D._PLAN] = torch.zeros(8)
    if serializer == "copy":
        cloned = copy.deepcopy(model)
    else:
        package = types.ModuleType("dfn")
        package.__path__ = []
        models = types.ModuleType("dfn.models")
        models.__path__ = []
        package.models = models
        models._decode_fused_nf4 = D
        monkeypatch.setitem(sys.modules, "dfn", package)
        monkeypatch.setitem(sys.modules, "dfn.models", models)
        monkeypatch.setitem(sys.modules, D.__name__, D)
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        cloned = torch.load(stream, weights_only = False)
    cloned.to(torch.float64)
    assert model[0].weight.dtype == torch.float32
    assert cloned[0].weight.dtype == torch.float64
    assert D._PLAN in model.__dict__ and D._PLAN not in cloned.__dict__


def test_pickled_submodule_does_not_serialize_its_owner(monkeypatch):
    import io
    import threading

    D = _driver()
    package = types.ModuleType("dfn")
    package.__path__ = []
    models = types.ModuleType("dfn.models")
    models.__path__ = []
    package.models = models
    models._decode_fused_nf4 = D
    monkeypatch.setitem(sys.modules, "dfn", package)
    monkeypatch.setitem(sys.modules, "dfn.models", models)
    monkeypatch.setitem(sys.modules, D.__name__, D)
    owner = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    D._track_moves(owner)
    owner.__dict__["_unpicklable"] = threading.Lock()  # anything only the owner holds
    stream = io.BytesIO()
    torch.save(owner[0], stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only = False)
    assert restored.__dict__["_apply"].root_ref() is None
    restored.to(torch.float64)
    assert restored.weight.dtype == torch.float64 and owner[0].weight.dtype == torch.float32


def test_vllm_shared_weights_are_refused():
    D = _driver()
    model = types.SimpleNamespace(
        config = types.SimpleNamespace(model_type = "llama"), vllm_engine = object()
    )
    assert "vLLM" in D._build(model, torch.zeros(1, 1, 8))


def test_refusal_drops_previously_fused_graphs(monkeypatch):
    D = _driver()
    monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", "1")  # _decode_forward runs only when requested
    dropped = []
    monkeypatch.setattr(D._decode_graphs, "invalidate", lambda model: dropped.append(model))

    layer = types.SimpleNamespace(self_attn = types.SimpleNamespace())
    for previous, expected in ((None, 0), (D._Plan(), 1)):
        model = types.SimpleNamespace(model = types.SimpleNamespace(layers = [layer]))
        if previous is not None:
            model.__dict__[D._PLAN] = previous
        monkeypatch.setattr(D, "_build", lambda m, X: "bias")
        with torch.no_grad():
            assert D._decode_forward(model, torch.zeros(1, 1, 8), *([None] * 6)) is None
        assert len(dropped) == expected
        assert isinstance(model.__dict__[D._PLAN], D._Refused)


def test_whole_step_graph_capture_keeps_the_regular_path(monkeypatch):
    # The whole-step capture's span / keep mask are read only by the regular attention path.
    D = _driver()
    monkeypatch.setattr(D._decode_graphs, "static", lambda: types.SimpleNamespace(span = 256))

    def build(model, X):
        raise AssertionError("fused plan built under a whole-step capture")

    monkeypatch.setattr(D, "_build", build)
    model = types.SimpleNamespace(
        model = types.SimpleNamespace(layers = [types.SimpleNamespace(self_attn = 0)])
    )
    with torch.no_grad():
        assert D._decode_forward(model, torch.zeros(1, 1, 8), *([None] * 6)) is None


def _norm_plan(
    monkeypatch,
    strided = False,
    capability = (8, 0),
    graphs = False,
    attention = None,
    batch = 1,
):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    D = _driver()
    monkeypatch.setattr(D.torch.version, "hip", None)
    monkeypatch.setattr(D.torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(D._decode_graphs, "ENABLED", graphs)
    monkeypatch.setattr(D.F, "supported_capability", lambda capability: True, raising = False)
    monkeypatch.setattr(D, "_signature", lambda plan: ())
    monkeypatch.setattr(D, "_track_moves", lambda model: None)
    device = torch.device("cuda", 0)
    with FakeTensorMode():
        weight = torch.empty_strided((8,), (2 if strided else 1,), device = device)
        assert isinstance(weight, FakeTensor)
        norm = lambda w: types.SimpleNamespace(weight = w, variance_epsilon = 1e-5)
        attn = (
            types.SimpleNamespace(
                head_dim = 8,
                rotary_emb = object(),
                q_proj = object(),
                k_proj = object(),
                v_proj = object(),
                o_proj = object(),
            )
            if attention is None
            else attention
        )
        mlp = types.SimpleNamespace(gate_proj = object(), up_proj = object(), down_proj = object())
        layer = types.SimpleNamespace(
            self_attn = attn,
            mlp = mlp,
            input_layernorm = norm(torch.empty(8, device = device)),
            post_attention_layernorm = norm(torch.empty(8, device = device)),
        )
        model = types.SimpleNamespace(
            config = types.SimpleNamespace(
                model_type = "llama",
                hidden_size = 8,
                intermediate_size = 16,
                num_attention_heads = 1,
                num_key_value_heads = 1,
            ),
            model = types.SimpleNamespace(layers = [layer], norm = norm(weight)),
        )
        packed = torch.empty(1, device = device, dtype = torch.uint8)
        bounds = {"qkv": [8, 16, 24], "o": [8], "gate_up": [16, 32], "down": [8]}
        monkeypatch.setattr(
            D,
            "_site",
            lambda model, key, kind, projs, width, **kwargs: types.SimpleNamespace(
                packed = packed, bounds = bounds[kind], N = bounds[kind][-1], lora = False
            ),
        )
        plan = D._build(model, torch.empty((batch, 1, 8), device = device, dtype = torch.float16))
    return D, plan, weight


@pytest.mark.parametrize("strided", [False, True])
def test_strided_final_norm_keeps_the_regular_path(monkeypatch, strided):
    D, plan, weight = _norm_plan(monkeypatch, strided)
    if strided:
        assert plan == "final norm"
    else:
        assert isinstance(plan, D._Plan)
        assert plan.final_norm is weight


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 9)])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("flag", ["1", "force"])
def test_t4_prefers_regular_graphs(monkeypatch, capability, graphs, flag):
    monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", flag)
    D, plan, _ = _norm_plan(monkeypatch, capability = capability, graphs = graphs)
    assert isinstance(plan, D._Plan) != (capability == (7, 5) and graphs and flag == "1")


def test_equal_refusal_keeps_regular_graphs(monkeypatch):
    D = _driver()
    monkeypatch.setenv("UNSLOTH_DECODE_FUSED_NF4", "1")  # _decode_forward runs only when requested
    initial = D._Refused("regular graph path")
    model = types.SimpleNamespace(
        model = types.SimpleNamespace(layers = [types.SimpleNamespace(self_attn = object())])
    )
    model.__dict__[D._PLAN] = initial
    invalidated = []
    monkeypatch.setattr(D._decode_graphs, "invalidate", lambda model: invalidated.append(model))
    monkeypatch.setattr(D, "_build", lambda model, X: initial.reason)
    with torch.no_grad():
        for _ in range(2):
            assert D._decode_forward(model, torch.zeros(1, 1, 8), *([None] * 6)) is None
            assert model.__dict__[D._PLAN] is initial
        assert invalidated == []
        monkeypatch.setattr(D, "_build", lambda model, X: "different refusal")
        assert D._decode_forward(model, torch.zeros(1, 1, 8), *([None] * 6)) is None
    assert invalidated == [model]
    assert model.__dict__[D._PLAN].reason == "different refusal"


class _Lora:
    def __init__(self):
        self.disable_adapters, self.merged = False, False
        self.active_adapters, self.scaling = ["default"], {"default": 2.0, "other": 0.5}


def test_plan_goes_stale_when_adapters_change():
    # Checked every step, so a toggle inside one continuation does not keep the old LoRA terms.
    D = _driver()
    layer = _Lora()
    plan = types.SimpleNamespace(watch = [layer], adapter_state = D._adapter_state([layer]))
    assert not D._plan_stale(plan)
    for change in (
        lambda: setattr(layer, "disable_adapters", True),
        lambda: setattr(layer, "merged", True),
        lambda: setattr(layer, "active_adapters", ["other"]),
        lambda: layer.scaling.update(default = 1.0),
    ):
        layer.__init__()
        assert not D._plan_stale(plan)
        change()
        assert D._plan_stale(plan)
    assert not D._plan_stale(types.SimpleNamespace(watch = [], adapter_state = ()))


def test_each_site_watches_a_projection_with_lora():
    peft = pytest.importorskip("peft")
    D = _driver()
    model = peft.get_peft_model(
        torch.nn.Sequential(*[torch.nn.Linear(8, 8, bias = False) for _ in range(3)]),
        peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["0", "1", "2"]),
    )
    projections = list(model.base_model.model)
    watch = D._site_watch([object(), *projections])
    x = torch.ones(1, 8)
    for projection in projections:
        torch.nn.init.ones_(projection.lora_A["default"].weight)
        torch.nn.init.ones_(projection.lora_B["default"].weight)
        plan = types.SimpleNamespace(watch = watch, adapter_state = D._adapter_state(watch))
        assert not D._plan_stale(plan)
        before = projection(x).detach().clone()
        projection.scale_layer(3.0)
        assert not torch.equal(before, projection(x))
        assert D._plan_stale(plan), "a projection's scale change must invalidate the fused plan"
    assert D._site_watch([object(), object()]) == []


def test_plan_goes_stale_when_peft_is_unloaded(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    peft = pytest.importorskip("peft")
    D = _driver()
    monkeypatch.setattr(D.torch.version, "hip", None)
    model = peft.get_peft_model(
        torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(8, 8, bias = False)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
        ),
        peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    owner = model.base_model.model
    owner.head_dim, owner.rotary_emb = 8, object()
    projection = owner.q_proj
    torch.nn.init.ones_(projection.lora_A["default"].weight)
    torch.nn.init.ones_(projection.lora_B["default"].weight)
    x = torch.ones(1, 8)
    monkeypatch.setattr(D.torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(D.F, "supported_capability", lambda capability: True, raising = False)
    monkeypatch.setattr(D, "_signature", lambda plan: ())
    monkeypatch.setattr(D, "_track_moves", lambda model: None)
    device = torch.device("cuda", 0)
    with FakeTensorMode():
        norm = lambda: types.SimpleNamespace(
            weight = torch.empty(8, device = device), variance_epsilon = 1e-5
        )
        layer = types.SimpleNamespace(
            self_attn = owner,
            mlp = types.SimpleNamespace(gate_proj = object(), up_proj = object(), down_proj = object()),
            input_layernorm = norm(),
            post_attention_layernorm = norm(),
        )
        core = types.SimpleNamespace(
            config = types.SimpleNamespace(
                model_type = "llama",
                hidden_size = 8,
                intermediate_size = 16,
                num_attention_heads = 1,
                num_key_value_heads = 1,
            ),
            model = types.SimpleNamespace(layers = [layer], norm = norm()),
        )
        packed = torch.empty(1, device = device, dtype = torch.uint8)
        bounds = {"qkv": [8, 16, 24], "o": [8], "gate_up": [16, 32], "down": [8]}
        monkeypatch.setattr(
            D,
            "_site",
            lambda model, key, kind, projs, width: types.SimpleNamespace(
                packed = packed, bounds = bounds[kind], N = bounds[kind][-1]
            ),
        )
        plan = D._build(core, torch.empty((1, 1, 8), device = device, dtype = torch.float16))
    assert isinstance(plan, D._Plan)
    assert not D._plan_stale(plan)
    before = projection(x).detach().clone()
    with torch.no_grad():
        projection.lora_B["default"].weight.mul_(2)
    assert not torch.equal(before, projection(x))
    assert not D._plan_stale(plan)
    before = projection(x).detach().clone()
    unwrapped = model.unload()
    assert unwrapped is owner
    assert owner.q_proj is projection.get_base_layer()
    assert not torch.equal(before, owner.q_proj(x))
    assert not projection.merged and not projection.disable_adapters
    assert D._plan_stale(plan)


@pytest.mark.parametrize("incoming_rank", [2, 4])
def test_same_scale_hotswap_tracks_rank_changes(tmp_path, incoming_rank):
    peft = pytest.importorskip("peft")
    hotswap = pytest.importorskip("peft.utils.hotswap")
    D = _driver()

    def make(rank, value):
        model = peft.get_peft_model(
            torch.nn.Sequential(torch.nn.Linear(8, 8, bias = False)),
            peft.LoraConfig(r = rank, lora_alpha = rank, target_modules = ["0"]),
        )
        projection = model.base_model.model[0]
        torch.nn.init.constant_(projection.lora_A["default"].weight, value)
        torch.nn.init.constant_(projection.lora_B["default"].weight, value)
        return model, projection

    model, projection = make(2, 0.1)
    incoming, _ = make(incoming_rank, 0.3)
    incoming.save_pretrained(tmp_path)
    watch = D._site_watch([projection])
    plan = types.SimpleNamespace(watch = watch, adapter_state = D._adapter_state(watch))
    A, B = projection.lora_A["default"].weight, projection.lora_B["default"].weight
    x = torch.ones(1, 8)
    before = projection(x).detach().clone()
    hotswap.hotswap_adapter(model, str(tmp_path), adapter_name = "default", torch_device = "cpu")
    assert projection.scaling["default"] == 1.0
    assert projection.lora_A["default"].weight.shape[0] == incoming_rank
    assert not torch.equal(before, projection(x))
    assert D._plan_stale(plan) == (incoming_rank != 2)
    if incoming_rank == 2:
        assert A is projection.lora_A["default"].weight
        assert B is projection.lora_B["default"].weight
        assert torch.all(A == 0.3) and torch.all(B == 0.3)


def test_signature_follows_saved_norm_weights():
    # A norm in modules_to_save: its weight follows the active adapter, so it must key the plan.
    peft_other = pytest.importorskip("peft.utils.other")
    D = _driver()

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))
            self.variance_epsilon = 1e-5

    wrapper = peft_other.ModulesToSaveWrapper(Norm(), "default")
    t = torch.zeros(4)
    site = types.SimpleNamespace(packed = t, absmax = t, absmax2 = t, lora = None, bounds = [4])

    def plan():
        entry = types.SimpleNamespace(ln1 = wrapper.weight, ln2 = wrapper.weight, sites = {"o": site})
        return types.SimpleNamespace(
            device = t.device, dtype = torch.bfloat16, hidden = 8, intermediate = 16, eps = 1e-5,
            final_norm = wrapper.weight, layers = [entry],
        )  # fmt: skip

    on = D._signature(plan())
    assert D._adapter_key(wrapper) == D._adapter_key(wrapper)
    before = D._adapter_key(wrapper)
    wrapper.enable_adapters(False)
    assert D._signature(plan()) != on
    assert D._adapter_key(wrapper) != before


def test_plan_goes_stale_when_a_tensor_scale_changes_in_place():
    # prepare_model_for_compiled_hotswap makes scales tensors; hotswap fill_()s them in place.
    D = _driver()
    layer = _Lora()
    layer.scaling = {"default": torch.tensor(2.0)}
    plan = types.SimpleNamespace(watch = [layer], adapter_state = D._adapter_state([layer]))
    assert not D._plan_stale(plan)
    layer.scaling["default"].fill_(6.0)
    assert D._plan_stale(plan)
    plan.adapter_state = D._adapter_state([layer])
    assert not D._plan_stale(plan)
    layer.scaling["default"] *= 0.5  # LoraLayer.scale_layer
    assert D._plan_stale(plan)


@pytest.mark.parametrize("inference_scale", [False, True])
def test_hotswap_inference_scales_keep_the_regular_path(monkeypatch, inference_scale):
    peft = pytest.importorskip("peft")
    hotswap = pytest.importorskip("peft.utils.hotswap")
    BF = pytest.importorskip("bitsandbytes.functional")
    D = _driver()
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(256, 64, bias = False)),
        peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["0"]),
    )
    projection = model.base_model.model[0]
    torch.nn.init.ones_(projection.lora_A["default"].weight)
    torch.nn.init.ones_(projection.lora_B["default"].weight)
    with torch.inference_mode(inference_scale):
        hotswap.prepare_model_for_compiled_hotswap(model)
    scale = projection.scaling["default"]
    assert scale.is_inference() == inference_scale
    packed, state = BF.quantize_4bit(
        projection.weight.detach().to(torch.float16), quant_type = "nf4", compress_statistics = True
    )
    for name in (
        "_has_non_lora_adapter",
        "_has_multiple_active_adapters",
        "_has_active_dora_adapter",
        "_has_active_lora_bias",
        "_has_packed_qweight",
    ):
        monkeypatch.setattr(D.KU, name, lambda p: False, raising = False)
    monkeypatch.setattr(D.KU, "_quant_state_dtype", lambda qs: qs.dtype, raising = False)
    monkeypatch.setattr(
        D.KU,
        "get_lora_parameters_bias",
        lambda p: (
            packed,
            state,
            p.lora_A["default"].weight,
            p.lora_B["default"].weight,
            p.scaling["default"],
            None,
        ),
        raising = False,
    )
    parts = D._proj_parts(projection, 256)
    plan = types.SimpleNamespace(watch = [projection], adapter_state = D._adapter_state([projection]))
    with torch.inference_mode():
        x = torch.ones(1, 256)
        before = projection(x).clone()
        scale.fill_(2.0)
        assert not torch.equal(before, projection(x))
    if not inference_scale:
        assert parts[3][2] == 1.0
        assert D._plan_stale(plan)
    else:
        assert parts == "inference tensor LoRA scale"
        assert D._proj_parts(projection, 256) == "inference tensor LoRA scale"


_NO_TRITON_OP = textwrap.dedent(
    """
    import importlib.util, os, sys, types
    import torch
    for name in ("triton_op", "wrap_triton"):  # absent before torch 2.6
        if hasattr(torch.library, name):
            delattr(torch.library, name)
    from packaging.version import Version
    root = os.path.join(sys.argv[1], "unsloth")
    zoo = types.ModuleType("unsloth_zoo"); zoo.__path__ = []
    zoo_utils = types.ModuleType("unsloth_zoo.utils"); zoo_utils.Version = Version
    sys.modules["unsloth_zoo"] = zoo; sys.modules["unsloth_zoo.utils"] = zoo_utils
    pkg = types.ModuleType("dfn"); pkg.__path__ = []
    models = types.ModuleType("dfn.models"); models.__path__ = [os.path.join(root, "models")]
    kernels = types.ModuleType("dfn.kernels"); kernels.__path__ = [os.path.join(root, "kernels")]
    kernels.utils = types.ModuleType("dfn.kernels.utils")
    sys.modules.update({"dfn": pkg, "dfn.models": models, "dfn.kernels": kernels})
    sys.modules["dfn.kernels.utils"] = kernels.utils
    spec = importlib.util.spec_from_file_location(
        "dfn.models._decode_fused_nf4", os.path.join(root, "models", "_decode_fused_nf4.py")
    )
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    assert D.F is None
    print("refused:", D._build(None, torch.zeros(1, 1, 8)))
    """
)


def test_kernels_unavailable_keeps_the_regular_path():
    # torch 2.4 / 2.5: kernels/nf4.py needs torch.library.triton_op; importing the driver must not fail.
    pytest.importorskip("triton")
    result = subprocess.run(
        [sys.executable, "-c", _NO_TRITON_OP, str(REPO)], capture_output = True, text = True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "refused: fused NF4 kernels unavailable" in result.stdout


@pytest.mark.parametrize("plan_key", ["_PLAN", "_BATCH_PLAN"])
def test_adapter_reload_invalidates_cached_weights(tmp_path, plan_key):
    peft = pytest.importorskip("peft")
    D = _driver()
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(8, 8, bias = False)),
        peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["0"]),
    )
    owner = model.base_model.model
    projection = owner[0]
    with torch.no_grad():
        projection.lora_A["default"].weight.fill_(1)
        projection.lora_B["default"].weight.fill_(2)
    checkpoint = tmp_path / "adapter"
    model.save_pretrained(checkpoint)
    with torch.no_grad():
        projection.lora_B["default"].weight.fill_(1)
    old_a = projection.lora_A["default"].weight
    old_b = projection.lora_B["default"].weight
    watch = D._site_watch([projection])
    plan = types.SimpleNamespace(
        watch = watch,
        adapter_state = D._adapter_state(watch),
        bindings = [(owner, "0", projection)],
        a = old_a,
        b = old_b,
    )
    key = getattr(D, plan_key)
    setattr(owner, key, plan)
    D._track_moves(owner)
    x = torch.ones(1 if plan_key == "_PLAN" else 2, 8)
    with torch.no_grad():
        before = model(x).clone()
        model.delete_adapter("default")
        model.load_adapter(checkpoint, adapter_name = "default", torch_device = "cpu")
        model.set_adapter("default")
        after = model(x)
        cached = projection.base_layer(x) + (x @ old_a.T) @ old_b.T
    assert owner[0] is projection
    assert projection.lora_A["default"].weight is not old_a
    assert projection.lora_B["default"].weight is not old_b
    torch.testing.assert_close(cached, before, rtol = 0, atol = 1e-6)
    assert (after - cached).abs().max() >= 15
    assert getattr(owner, key, None) is not plan or D._plan_stale(plan)
    current = types.SimpleNamespace(
        watch = watch,
        adapter_state = D._adapter_state(watch),
        bindings = [(owner, "0", projection)],
    )
    setattr(owner, key, current)
    D._track_moves(owner)
    projection.lora_A["default"].to(dtype = torch.float64)
    assert key not in owner.__dict__


@pytest.mark.parametrize("batch", [1, 2])
def test_first_adapter_invalidates_plain_projection_bindings(monkeypatch, batch):
    peft = pytest.importorskip("peft")
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    owner = torch.nn.ModuleDict({name: torch.nn.Linear(8, 8, bias = False) for name in names})
    owner.head_dim, owner.rotary_emb = 8, object()
    projection = owner.q_proj
    x = torch.ones(batch, 8)
    before = projection(x).detach().clone()
    D, plan, _ = _norm_plan(monkeypatch, capability = (7, 5), attention = owner, batch = batch)
    assert isinstance(plan, D._Plan)
    assert not plan.watch and not D._plan_stale(plan)
    assert any(
        parent is owner and name == "q_proj" and module is projection
        for parent, name, module in plan.bindings
    )
    model = peft.get_peft_model(owner, peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = names))
    assert model.base_model.model is owner
    assert owner.q_proj.get_base_layer() is projection
    with torch.no_grad():
        owner.q_proj.lora_A["default"].weight.fill_(1)
        owner.q_proj.lora_B["default"].weight.fill_(1)
    assert (owner.q_proj(x) - before).abs().max() >= 15
    assert D._plan_stale(plan)


@pytest.mark.parametrize("plan_key", ["_PLAN", "_BATCH_PLAN"])
def test_loaded_saved_norm_moves_release_the_current_root(tmp_path, plan_key):
    import weakref

    peft = pytest.importorskip("peft")
    D = _driver()

    def make():
        return torch.nn.ModuleDict(
            {"proj": torch.nn.Linear(8, 8, bias = False), "norm": torch.nn.LayerNorm(8)}
        )

    trained = peft.get_peft_model(
        make(),
        peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["proj"], modules_to_save = ["norm"]),
    )
    checkpoint = tmp_path / "adapter"
    trained.save_pretrained(checkpoint)
    owner = make()
    D._track_moves(owner)
    model = peft.PeftModel.from_pretrained(owner, checkpoint, torch_device = "cpu")
    assert model.base_model.model is owner
    norm = owner.norm.modules_to_save["default"]
    key = getattr(D, plan_key)
    held = torch.ones(8)
    ref = weakref.ref(held)
    setattr(owner, key, held)
    del held
    D._track_moves(owner)
    norm.to(dtype = torch.float64)
    assert norm.weight.dtype == torch.float64
    assert owner.proj.get_base_layer().weight.dtype == torch.float32
    assert key not in owner.__dict__ and ref() is None
    assert norm.__dict__["_apply"].root_ref() is owner
    assert norm.__dict__["_apply"].module_ref() is norm


def test_partial_module_copy_does_not_copy_its_root():
    import copy

    D = _driver()
    owner = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    D._track_moves(owner)
    held = torch.ones(8)
    owner.__dict__[D._PLAN] = held
    memo = {}
    cloned = copy.deepcopy(owner[0], memo)
    assert id(owner) not in memo and id(owner[1]) not in memo
    assert cloned is not owner[0]
    assert cloned.__dict__["_apply"].root_ref() is None
    cloned.to(dtype = torch.float64)
    assert cloned.weight.dtype == torch.float64
    assert owner[0].weight.dtype == torch.float32
    assert owner.__dict__[D._PLAN] is held
    assert owner[0].__dict__["_apply"].root_ref() is owner


@pytest.mark.parametrize("plan_key", ["_PLAN", "_BATCH_PLAN"])
def test_nf4_merge_unload_invalidates_restored_projection_weights(plan_key):
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    D = _driver()
    projection = bnb.nn.Linear4bit(
        64,
        64,
        bias = False,
        compute_dtype = torch.float16,
        compress_statistics = False,
        quant_type = "nf4",
    )
    projection.weight = bnb.nn.Params4bit(
        projection.weight.detach().half(),
        requires_grad = False,
        compress_statistics = False,
        quant_type = "nf4",
    )
    projection.to("cpu")
    if projection.weight.quant_state is None:
        pytest.skip("bitsandbytes cannot quantize NF4 on CPU")
    owner = torch.nn.ModuleDict({"proj": projection})
    owner.is_loaded_in_4bit = True
    cached_weight = projection.weight
    cached_bytes = cached_weight.detach().clone()
    x = torch.ones(1 if plan_key == "_PLAN" else 2, 64, dtype = torch.float16)
    before = torch.nn.functional.linear(
        x.float(),
        bnb.functional.dequantize_4bit(cached_weight.data, cached_weight.quant_state).float(),
    )
    plan = types.SimpleNamespace(
        watch = [],
        adapter_state = (),
        bindings = [(owner, "proj", projection)],
        tensor_bindings = ((projection, "weight", cached_weight),),
    )
    key = getattr(D, plan_key)
    setattr(owner, key, plan)
    D._track_moves(owner)
    adapted = peft.get_peft_model(
        owner, peft.LoraConfig(r = 2, lora_alpha = 2, target_modules = ["proj"])
    )
    with torch.no_grad():
        owner.proj.lora_A["default"].to(dtype = torch.float16)
        owner.proj.lora_B["default"].to(dtype = torch.float16)
        owner.proj.lora_A["default"].weight.fill_(1)
        owner.proj.lora_B["default"].weight.fill_(0.25)
        merged = adapted.merge_and_unload()
    assert merged is owner and owner.proj is projection
    assert getattr(owner, key) is plan
    assert projection.weight is not cached_weight
    assert torch.equal(cached_weight, cached_bytes)
    assert not torch.equal(projection.weight, cached_bytes)
    with torch.no_grad():
        after = projection(x).float()
    assert torch.isfinite(after).all() and (after - before).abs().max() > 1
    assert D._plan_stale(plan)


@pytest.mark.parametrize("changed", [None, "code", "code2", "seg_offset"])
def test_requantized_weights_at_reused_addresses_rebuild_the_plan(changed):
    # Equal pointers (the allocator reused every address) with different copied NF4 tables: the old
    # plan's code / absmax2 code / nested offsets copies are stale, so it must not be kept.
    D = _driver()
    attn, rope = object(), object()

    def plan(tables):
        site = types.SimpleNamespace(nested = True, **tables)
        layer = types.SimpleNamespace(sites = {"qkv": site}, attn = attn, rope = rope)
        return types.SimpleNamespace(signature = ("same",), layers = [layer])

    tables = {
        "code": torch.linspace(-1, 1, 16),
        "code2": torch.linspace(-1, 1, 256),
        "seg_offset": torch.tensor([0.25, 0.5, 0.5]),
    }
    new = {k: v.clone() for k, v in tables.items()}
    if changed is not None:
        new[changed][1] += 1
    assert D._reusable(plan(tables), plan(new)) is (changed is None)
    other = plan(new)
    other.signature = ("other",)
    assert not D._reusable(plan(tables), other)
