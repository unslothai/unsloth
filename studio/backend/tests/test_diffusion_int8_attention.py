# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's INT8 attention kernel (``unsloth_int8``): switches, per-card configs, the processor route and its
fallbacks on CPU; the kernel against a plain-torch model of itself, spills, CUDA graphs and torch.compile on CUDA."""

from __future__ import annotations

import contextlib
import copy
import math
import re
import types

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as K


@pytest.fixture(autouse = True)
def _no_recorded_graph():
    """A test that fakes a capture leaves no recorded graph behind (it freezes every later plan's decision)."""
    I._GRAPHED[0] = False
    yield
    I._GRAPHED[0] = False


# dynamic shared memory one block may ask for, and registers per SM, per capability
SHARED_LIMIT = {
    (7, 5): 64 * 1024,
    (8, 0): 163 * 1024,
    (8, 6): 99 * 1024,
    (8, 9): 99 * 1024,
    (12, 0): 99 * 1024,
}
SM_REGISTERS = 65536


def _cuda_config(head_dim: int = 128):
    if not torch.cuda.is_available():
        return None
    return I.kernel_config(torch.cuda.get_device_capability(), head_dim)


needs_kernel = pytest.mark.skipif(
    _cuda_config() is None, reason = "needs a CUDA card with an INT8 kernel config"
)


def _card_dtype():
    """bf16 from sm80 on this card, else fp16: sm75 has no bf16 for Inductor or fused SDPA."""
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


def _card_sdpa() -> str:
    """The native backend the H3 blocks keep: cuDNN attention needs sm80, sm75 takes the default SDPA pick."""
    return "_native_cudnn" if torch.cuda.get_device_capability() >= (8, 0) else "native"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    stats = dict(I.STATS)
    yield
    I.STATS.update(stats)


def rotate_ref(x):
    """Fixed sign flips, then a Hadamard over each ROT_GROUP channels (head dim if smaller)."""
    D = x.shape[-1]
    G = min(I.ROT_GROUP, D)
    cols = torch.arange(D, device = x.device)
    flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
    x = x * (1.0 - 2.0 * flip.to(x.dtype))
    for _ in range(G.bit_length() - 1):
        y = x.reshape(*x.shape[:-1], D // 2, 2)
        z = torch.stack((y[..., 0] + y[..., 1], y[..., 0] - y[..., 1]), dim = -1)
        x = z.reshape(*x.shape[:-1], D // G, G // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], D)
    return x


def key_of_position(n: int, device = None):
    """Key stored at each position of the V tiles (n a multiple of 16)."""
    pos = torch.arange(n, device = device)
    return (pos & -16) + 8 * ((pos & 3) >> 1) + 2 * ((pos >> 2) & 3) + (pos & 1)


def reference_quantise(
    q,
    k,
    v,
    config,
    kc = None,
    vc = None,
    vg = None,
):
    """The quantisers' math in fp64. The per-channel K mean (over every kstep-th key) and V centre / gain (over
    every key) are computed here, or passed in (what the kernels computed) to compare the rounding alone."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bn = config["bn"]
    BH = B * H
    mul = 1.0 / math.sqrt(D) * I.LOG2E / min(I.ROT_GROUP, D)

    def heads_first(t):
        return t.permute(0, 2, 1, 3).reshape(BH, t.shape[1], D).double()

    kh, vh = heads_first(k), heads_first(v)
    clip = torch.zeros(BH, dtype = torch.bool, device = q.device)
    if kc is None:
        kstep, vstep, nvc, vrows = I.stats_plan(Sk)
        kc = kh[:, ::kstep].mean(1)
        vc, vg, clip = reference_v_range(vh, vstep, nvc, vrows)
    else:
        clip = (
            ((vh - vc.double()[:, None, :]).abs() > vg.double()[:, None, :] * (1 + 1e-4))
            .flatten(1)
            .any(1)
        )
    kc, vc, vg = kc.double(), vc.double(), vg.double()
    rcnt, ridx, rres = reference_residuals(vh, vc, vg, clip)
    qr = rotate_ref(heads_first(q))
    qa = qr.abs().amax(-1).clamp_min(1e-20)
    n_kpad = (Sk + bn - 1) // bn * bn
    kr = F.pad(rotate_ref(kh - kc[:, None, :]), (0, 0, 0, n_kpad - Sk))
    # one K scale per run of 32 keys and accumulator lane group ((key % 8) // 2)
    ka = kr.abs().amax(-1).view(BH, n_kpad // 32, 4, 4, 2).amax((2, 4)).clamp_min(1e-20)
    per_key = ka[:, :, None, :, None].expand(-1, -1, 4, -1, 2).reshape(BH, n_kpad)
    xr = (vh - vc[:, None, :]) / vg[:, None, :]
    xr = torch.where(clip[:, None, None], xr.clamp(-1, 1), xr)
    vr = F.pad(xr, (0, 0, 0, n_kpad - Sk)).view(BH, n_kpad // bn, bn, D)
    va = vr.abs().amax((2, 3)).clamp_min(1e-20)
    vt = torch.round(vr * (127.0 / va[:, :, None, None]))[
        :, :, key_of_position(bn, q.device)
    ].transpose(2, 3)
    return {
        "qi": torch.round(qr * (127.0 / qa[..., None])),
        "qs": (qa / 127.0 * mul).float(),
        "ki": torch.round(kr * (127.0 / per_key[..., None])),
        "ks": (ka / 127.0).float(),
        "kc": kc.float(),
        "vt": vt,
        "vs": (va / 127.0).float(),
        "vc": vc.float(),
        "vg": vg.float(),
        "rcnt": rcnt,
        "ridx": ridx,
        "rres": rres,
        "shape": (B, Sq, Sk, H, D),
        "dtype": v.dtype,
        "config": config,
    }


def reference_v_range(vh, vstep, nvc, vrows):
    """Per-head (centre, gain, clip) in fp64: full channel range, unless at least half the channels
    exceed SINK_RATIO (2) x their range without the largest-|v| chunk; then robust range widened
    by SINK_MARGIN (2), at most full range, clipped."""
    BH, Sk, D = vh.shape
    lo, hi = vh[:, ::vstep].amin(1), vh[:, ::vstep].amax(1)
    vc, vg = 0.5 * (hi + lo), (0.5 * (hi - lo)).clamp_min(1e-12)
    pad = nvc * vrows - Sk
    live = (torch.arange(nvc * vrows, device = vh.device) < Sk).view(1, nvc, vrows, 1)
    chunks = F.pad(vh, (0, 0, 0, pad)).view(BH, nvc, vrows, D)
    cmax = torch.where(live, chunks, torch.full_like(chunks, -1e300)).amax(2)
    cmin = torch.where(live, chunks, torch.full_like(chunks, 1e300)).amin(2)
    sink = (
        torch.where(live, chunks.abs(), torch.full_like(chunks, -1.0)).amax((2, 3)).argmax(1)
    )  # first max
    keep = torch.ones(BH, nvc, dtype = torch.bool, device = vh.device)
    keep[torch.arange(BH), sink] = False
    hir = torch.where(keep[:, :, None], cmax, torch.full_like(cmax, -1e300)).amax(1)
    lor = torch.where(keep[:, :, None], cmin, torch.full_like(cmin, 1e300)).amin(1)
    far = ((hi - lo) > 2.0 * (hir - lor)).sum(-1)
    clip = (2 * far >= D) & (nvc > 1) & (vstep == 1)
    rc = 0.5 * (hir + lor)
    rg = torch.minimum(2.0 * 0.5 * (hir - lor), 0.5 * (hi - lo)).clamp_min(1e-12)
    return torch.where(clip[:, None], rc, vc), torch.where(clip[:, None], rg, vg), clip


def reference_residuals(vh, vc, vg, clip):
    """(rcnt, ridx, rres) of the residual job: per clipping head, the keys whose (v - centre) / gain leaves [-1, 1]
    by more than 1/16 (SINK_MIN) in some channel, in key order, and the part clipped off."""
    BH, Sk, D = vh.shape
    x = (vh - vc[:, None, :]) / vg[:, None, :]
    res = x - x.clamp(-1, 1)
    out = (res.abs() > 0.0625).any(-1) & clip[:, None]
    rcnt = out.sum(1)
    n = max(1, int(rcnt.max()))
    ridx = torch.zeros(BH, n, dtype = torch.long, device = vh.device)
    rres = torch.zeros(BH, n, D, dtype = torch.float64, device = vh.device)
    for bh in range(BH):
        keys = torch.nonzero(out[bh]).flatten()
        ridx[bh, : len(keys)] = keys
        rres[bh, : len(keys)] = res[bh, keys]
    return rcnt, ridx, rres


def emulate(p):
    """What the CUDA kernel computes from ``quantise``'s output, in plain torch."""
    config = p["config"]
    B, Sq, Sk, H, D = p["shape"]
    bn = config["bn"]
    BH = B * H
    n_kpad = p["ki"].shape[1]
    tiles = n_kpad // bn
    dev = p["qi"].device
    scores = p["qi"][:, :Sq].double() @ p["ki"].double().transpose(1, 2)  # exact integers
    keys = torch.arange(n_kpad, device = dev)
    k_scale = p["ks"].reshape(BH, n_kpad // 8)[:, (keys // 32) * 4 + ((keys % 8) >> 1)]
    s = (scores.float() * (p["qs"][:, :Sq, None] * k_scale[:, None, :]).float()).view(
        BH, Sq, tiles, bn
    )
    s = torch.where((keys < Sk).view(1, 1, tiles, bn), s, torch.full_like(s, -1e30))
    tile_max = s.amax(dim = -1, keepdim = True)
    probs = torch.round(torch.exp2(s - tile_max + 7.99435343685886)).double()
    inverse = torch.empty(bn, dtype = torch.long, device = dev)
    inverse[key_of_position(bn, dev)] = torch.arange(bn, device = dev)
    v_int = p["vt"][:, :, :, inverse].double().transpose(2, 3)  # [BH, tiles, bn, D] by key
    weight = torch.exp2(
        (tile_max - tile_max[:, :, :1]).double()
    )  # relative to the first tile's maximum
    denom = (probs.sum(-1, keepdim = True) * weight).sum(2)
    pv = torch.einsum("bqtn,btnd->bqtd", probs, v_int)
    num = (pv * weight * p["vs"].double()[:, None, :, None]).sum(2)
    # the keys a head clipped: their clipped-off part times their exact P, in the frame of the first tile's maximum
    flat = s.reshape(BH, Sq, n_kpad).double()
    for bh in range(BH):
        n = int(p["rcnt"][bh]) if p.get("sinks", True) else 0
        if n:
            keys_ = p["ridx"][bh, :n].long()
            pr = torch.exp2(flat[bh][:, keys_] - tile_max[bh, :, 0].double() + 7.99435343685886)
            num[bh] += pr @ p["rres"][bh, :n].double()
    out = num / denom * p["vg"].double()[:, None, :] + p["vc"].double()[:, None, :]
    return out.view(B, H, Sq, D).permute(0, 2, 1, 3).float()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("0", "off"),
        ("off", "off"),
        ("1", "force"),
        ("force", "force"),
    ],
)
def test_env_switch(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    else:
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, raw)
    assert I.mode() == expected


def test_request_by_name():
    assert I.requested("unsloth_int8") and I.requested(" Unsloth_INT8 ")
    assert not I.requested("sage") and not I.requested(None)


def test_alias_is_a_native_pick_with_the_overlay(monkeypatch):
    from core.inference import diffusion_attention as DA

    assert DA.normalize_attention_backend("unsloth_int8") == "unsloth_int8"
    monkeypatch.setattr(DA, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(DA, "_cudnn_attention_supported", lambda: True)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    assert (
        DA.select_attention_backend(target, "unsloth_int8", speed_active = False) == "_native_cudnn"
    )
    monkeypatch.setattr(DA, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(DA, "_rocm_auto_flash", lambda target, family = None: False)
    assert DA.select_attention_backend(target, "unsloth_int8", speed_active = False) is None


def _variants(config: dict) -> list:
    """The kernels a table entry compiles: the one of short calls, the one from ``long_keys`` keys on, and the
    sink-path kernel when the entry gives that one other orderings (``sinks``)."""
    out = []
    for keys in (0, 10**9):
        for sinks in (False, True):
            chosen = I.kernel_variant(I.select_config(config, keys), sinks)
            if chosen not in out:
                out.append(chosen)
    return out


def test_every_config_fits_its_card():
    assert set(I.KERNEL_CONFIGS) == set(SHARED_LIMIT)
    for capability, table in I.KERNEL_CONFIGS.items():
        assert set(table) == set(I.HEAD_DIMS)
        for head_dim, entry in table.items():
            assert ("long" in entry) == ("long_keys" in entry)
            for config in _variants(entry):
                assert (
                    config["bn"] in (64, 128) and config["nq"] >= 1 and config["stages"] in (1, 2)
                )
                assert config["bn"] == entry["bn"]  # the tile is what the output bits depend on
                assert I.shared_bytes(head_dim, config) <= SHARED_LIMIT[capability]
                cap = config.get("max_registers")
                if cap is not None:
                    assert 32 * config["wpc"] * cap <= SM_REGISTERS
                assert (
                    32 * config["wpc"] <= 1024
                )  # sm75 holds 32 warps per SM, and a block is at most 1024 threads
            assert I.kernel_config(capability, head_dim) == entry
    assert I.kernel_config((9, 0), 128) is None and I.kernel_config((12, 0), 96) is None
    assert I.kernel_config((7, 0), 128) is None and I.kernel_config((6, 1), 128) is None
    assert I.kernel_config(None, 128) is None


def test_long_overrides_apply_from_long_keys_on():
    """A table entry may name another block shape for long sequences: ``select_config`` applies ``long`` from
    ``long_keys`` keys on and returns a plain config either way (one compiled kernel). sm80 at head dim 128 and
    sm89 at head dim 64 run 4 warps per block below 4608 keys and 8 from there on; no other entry has overrides."""
    with_long = {
        (capability, head_dim)
        for capability, table in I.KERNEL_CONFIGS.items()
        for head_dim, entry in table.items()
        if "long" in entry
    }
    assert with_long == {((8, 0), 128), ((8, 9), 64)}
    entry = I.kernel_config((8, 9), 64)
    assert I.select_config(entry, 4607)["wpc"] == 4 and I.select_config(entry, 4608)["wpc"] == 8
    entry = I.kernel_config((8, 0), 128)
    assert entry["long_keys"] == 4608 and entry["long"] == {"wpc": 8}
    short, long = (
        I.select_config(entry, entry["long_keys"] - 1),
        I.select_config(entry, entry["long_keys"]),
    )
    assert short == {"bn": 128, "wpc": 4, "nq": 1, "stages": 2, "qreg": 1}
    assert long == dict(short, wpc = 8)
    assert I.select_config(short, 10**9) == short
    assert I.kernel_defines(128, entry) == I.kernel_defines(128, short)
    assert I._config_key(entry) != I._config_key(short) and hash(I._config_key(entry)) is not None
    plain = I.kernel_config((8, 9), 128)
    assert I.select_config(plain, 0) == I.select_config(plain, 10**9) == plain


def test_sink_orderings_reach_only_the_sink_kernel(monkeypatch):
    """sm89 at head dim 128 gives its sink-path kernel other product / copy orderings (L4: 1.3-1.9% per call
    sustained at 4096-4608 tokens, the plain kernel is not faster with them). They name another compiled kernel, move
    no output bit, and ``UNSLOTH_DIFFUSION_INT8_ATTENTION_SINK_ORDER=0`` keeps the shipped one; no other entry has
    them."""
    monkeypatch.delenv(I.SINK_ORDER_ENV, raising = False)
    with_sinks = {
        (capability, head_dim)
        for capability, table in I.KERNEL_CONFIGS.items()
        for head_dim, entry in table.items()
        if "sinks" in entry
    }
    assert with_sinks == {((8, 9), 128)}
    entry = I.kernel_config((8, 9), 128)
    plain, sink = I.kernel_variant(entry, False), I.kernel_variant(entry, True)
    assert "sinks" not in plain and "sinks" not in sink
    assert plain == {
        "bn": 128,
        "wpc": 12,
        "nq": 1,
        "stages": 2,
        "max_registers": 168,
        "native_p": 1,
    }
    assert sink == dict(plain, flush_tail = 1, qkw = 1, pvw = 8, spread = 5)
    assert set(sink) - set(plain) <= set(I.SINK_ORDER_KEYS)
    assert I.kernel_defines(128, sink)["QKW"] == 1 and "QKW" not in I.kernel_defines(128, plain)
    assert hash(I._config_key(entry)) is not None and I._config_key(entry) != I._config_key(plain)
    monkeypatch.setenv(I.SINK_ORDER_ENV, "0")
    assert I.kernel_variant(entry, True) == plain
    # an override never sets anything but the orderings
    odd = dict(plain, sinks = {"wpc": 4, "qkw": 1})
    monkeypatch.delenv(I.SINK_ORDER_ENV)
    assert I.kernel_variant(odd, True) == dict(plain, qkw = 1)


@needs_kernel
def test_sink_orderings_do_not_change_the_output():
    """The sink-path kernel with and without the table's orderings: the same bits (sinks and fold on)."""
    entry = _cuda_config(128)
    base = I.kernel_variant(entry, False)
    other = (
        dict(base, flush_tail = 1, qkw = 1, pvw = 8, spread = 5)
        if base.get("stages") == 2
        else dict(base, qkw = 1)
    )
    for keys in (300, 1039):
        q, k, v = _case((1, keys, 3, 128), _card_dtype(), seed = keys)
        outs = []
        for cfg in (
            dict(base, sinks = {}),
            dict(base, sinks = {x: other[x] for x in I.SINK_ORDER_KEYS if x in other}),
        ):
            I.forget_plans()
            p = I.quantise(q, k, v, cfg, mode = I.MODE_RACE)
            assert p["sinks"]
            outs.append(I.attend(p))
        assert torch.equal(outs[0], outs[1])


def test_sm75_has_its_own_config():
    """A T4 has int8 tensor cores: it gets a config (it ran native attention before), sized for sm75's 64 KB of
    shared memory per SM. Q operands read from global memory (qreg 2) take no shared memory."""
    for head_dim in I.HEAD_DIMS:
        config = I.kernel_config((7, 5), head_dim)
        assert config is not None
        assert I.shared_bytes(head_dim, config) <= 64 * 1024
        stage = config["bn"] * (head_dim + 16) + head_dim * (config["bn"] + 16)
        assert I.shared_bytes(head_dim, dict(config, qreg = 2)) == config["stages"] * stage
        assert I.shared_bytes(head_dim, dict(config, qreg = 1)) > config["stages"] * stage
    assert I.kernel_defines(128, dict(I.kernel_config((8, 0), 128), sm75 = 1))["SM75"] == 1
    assert "SM75" not in I.kernel_defines(
        128, I.kernel_config((7, 5), 128)
    )  # the target architecture sets it


def test_register_cap_goes_on_the_entry_function():
    ptx = b".version 8.5\n.visible .entry unsloth_i8attn(\n\t.param .u64 a\n)\n{\n\tret;\n}\n\x00"
    capped = K.cap_registers(ptx, 168).decode()
    assert (
        ")\n.maxnreg 168\n{" in capped and capped.endswith("\x00") and capped.count(".maxnreg") == 1
    )


def _nvrtc_ok() -> bool:
    try:
        K.nvrtc()
        return True
    except Exception:  # noqa: BLE001
        return False


def _nvrtc_archs() -> set:
    """The real architectures this NVRTC can target (NVRTC 12.6 has no sm_120)."""
    import ctypes

    lib = K.nvrtc()
    count = ctypes.c_int()
    if lib.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0:
        return set()
    archs = (ctypes.c_int * count.value)()
    if lib.nvrtcGetSupportedArchs(archs) != 0:
        return set()
    return set(archs)


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
@pytest.mark.parametrize("capability", sorted(I.KERNEL_CONFIGS), ids = lambda c: f"sm{c[0]}{c[1]}")
def test_kernel_compiles_for_every_config(capability):
    arch = 10 * capability[0] + capability[1]
    if arch not in _nvrtc_archs():
        pytest.skip(f"NVRTC {K.nvrtc_version()} cannot target sm_{arch}")
    for head_dim, entry in I.KERNEL_CONFIGS[capability].items():
        for config in _variants(entry):
            ptx = K.compile_ptx(
                K.KERNEL_SOURCE,
                f"sm_{arch}",
                I.kernel_defines(head_dim, config),
            ).decode(errors = "replace")
            products = config["nq"] * (
                (head_dim // 32) * (config["bn"] // 8) + (config["bn"] // 32) * (head_dim // 8)
            )
            if capability < (8, 0):
                assert ptx.count("mma.sync.aligned.m8n8k16") == 4 * products
                assert "m16n8k32" not in ptx and "cp.async" not in ptx
            else:
                assert ptx.count("mma.sync.aligned.m16n8k32") == products
                assert "m8n8k16" not in ptx
            assert "cvt.rn.f32.s32" not in ptx
        quant = K.compile_ptx(
            K.QUANT_SOURCE,
            f"sm_{arch}",
            {
                "HD": head_dim,
                "BN": entry["bn"],
                "RGX": min(I.ROT_GROUP, head_dim).bit_length() - 5,
            },
        ).decode(errors = "replace")
        assert all(f".entry {name}(" in quant for name in K.QUANT_KERNELS)


def test_op_off_cuda_is_native_attention():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 40, 2, 64, generator = g) for _ in range(3))
    assert I.unsupported_reason(q, k, v) == "not on one CUDA device"
    with pytest.raises(ValueError):
        I.int8_attention(q, k, v)
    before = dict(I.STATS)
    I.int8_attention_op()
    got = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
    assert torch.equal(got, I.native_attention(q, k, v))
    assert I.STATS["native"] == before["native"] + 1 and I.STATS["int8"] == before["int8"]


def _cpu_inductor_missing() -> str:
    """Why Inductor cannot compile a CPU graph on this host ("" when it can). Its CPU backend builds C++ with the
    host compiler; Windows has none unless MSVC's ``cl`` is on PATH (the gfx1151 Windows runners have none), and a
    missing compiler fails every CPU graph with "Compiler: cl is not found", whatever the graph."""
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
    except Exception:  # noqa: BLE001 -- an older layout: let the test say what happens
        return ""
    try:
        get_cpp_compiler()
    except Exception as exc:  # noqa: BLE001
        return f"Inductor has no C++ compiler for CPU graphs here: {exc}"[:300]
    return ""


@pytest.mark.skipif(bool(_cpu_inductor_missing()), reason = _cpu_inductor_missing() or "n/a")
@pytest.mark.parametrize("sdpa", ["default", "math"])
@pytest.mark.parametrize("heads_first", [False, True], ids = ["bshd", "heads_first"])
@pytest.mark.parametrize("dynamic", [False, True], ids = ["static", "dynamic"])
def test_the_native_fallback_keeps_the_fake_layout_under_inductor(sdpa, heads_first, dynamic):
    """Native fallback must return contiguous [B, S, H, D], not SDPA permuted [B, H, S, D]: Inductor
    asserts those strides and opcheck misses them; only a compiled graph catches it."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    I.int8_attention_op()
    g = torch.Generator().manual_seed(0)

    def inputs(B, S, H, D):
        if (
            heads_first
        ):  # a [B, H, S, D] buffer viewed as [B, S, H, D], as the strided processors pass it
            return tuple(torch.randn(B, H, S, D, generator = g).transpose(1, 2) for _ in range(3))
        return tuple(torch.randn(B, S, H, D, generator = g) for _ in range(3))

    def block(q, k, v):
        return torch.ops.unsloth_studio.int8_attention_nhd(q, k, v).flatten(2, 3) + 1

    torch._dynamo.reset()
    try:
        compiled = torch.compile(block, backend = "inductor", fullgraph = True, dynamic = dynamic)
        pin = sdpa_kernel(SDPBackend.MATH) if sdpa == "math" else contextlib.nullcontext()
        with torch.no_grad(), pin:
            for shape in ((2, 11, 3, 64), (2, 17, 3, 64), (1, 126, 2, 64)):
                q, k, v = inputs(*shape)
                before = I.STATS["native"]
                # the compiled call first: on a layout mismatch Inductor's stride assertion is what fails
                out = compiled(q, k, v)
                assert I.STATS["native"] == before + 1
                assert torch.allclose(out, block(q, k, v), atol = 1e-5, rtol = 1e-5)
                got = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
                assert got.shape == q.shape and got.is_contiguous()
                assert torch.equal(got, I.native_attention(q, k, v))
    finally:
        torch._dynamo.reset()


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_cudnn"


class _Attention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _fake_dit(head_dims = (128, 128, 64, 96)):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, d) for d in head_dims])
    return dit


def _fake_cuda(
    monkeypatch,
    capability = (8, 9),
    found = None,
):
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    seen = []

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        seen.append((heads, head_dim, incumbent_name, timed))
        return found(head_dim) if callable(found) else found

    monkeypatch.setattr(I, "verdict", verdict)
    return seen


def test_the_card_check_also_probes_sharp_attention(monkeypatch):
    """The on-card check runs the probe a second time with q and k scaled up (the scores of late DiT blocks) and
    refuses a kernel whose output there is not finite or not close to fp32 attention."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    sharp_q = 1.6 * I.SHARP_GAIN / 2

    def kernel(broken):
        def run(
            q,
            k,
            v,
            scale = None,
            config = None,
        ):
            out = I.reference_attention(q, k, v).to(v.dtype)
            if q.float().std() > sharp_q:
                if broken == "nan":
                    out[0, 3, 0, 0] = float("nan")
                elif broken == "far":
                    out = out.flip(1)
            return out

        return run

    for broken, ok in ((None, True), ("nan", False), ("far", False)):
        monkeypatch.setattr(I, "int8_attention", kernel(broken))
        monkeypatch.setattr(I, "_VERIFIED", set())
        found = I._check("cpu", torch.float32, 4, 64, None, timed = False)
        assert found["ok"] is ok and found["cos"] >= I.MIN_COSINE
        assert (found["sharp_cos"] >= I.MIN_SHARP_COSINE) is ok
        assert bool(I._VERIFIED) is ok


def test_no_probability_is_packed_unclamped():
    """Every probability goes through the clamp to 255 before it is rounded and packed into a byte."""
    source = K.KERNEL_SOURCE
    assert "fminf(ex2(x), 255.0f)" in source
    assert (
        source.count("prob(fmaf(") == 8
        and "pack4(ex2(" not in source
        and " ex2(fmaf(" not in source
    )
    assert source.count("prob_rn(fmaf(") == 8 and source.count("cvt.pack.sat.u8.s32.b32") == 2
    assert "cvt.rni.sat.u8.f32" in source
    assert "if (fa == inf) oa = inf;" in source and "if (fb == inf) ob = inf;" in source


def test_native_p_reaches_the_kernel_as_a_define():
    """native_p is a per-card key that becomes the NATIVE_P define; a config without it compiles the MAGIC_F path."""
    base = {"bn": 128, "wpc": 8, "nq": 1, "stages": 2}
    assert "NATIVE_P" not in I.kernel_defines(128, base)
    assert I.kernel_defines(128, dict(base, native_p = 1))["NATIVE_P"] == 1
    assert I._config_key(dict(base, native_p = 1)) != I._config_key(base)
    for capability, table in I.KERNEL_CONFIGS.items():
        for head_dim, entry in table.items():
            for config in _variants(entry):
                assert config.get("native_p", 0) in (0, 1), (capability, head_dim)
                if capability < (8, 9):
                    assert "native_p" not in config, (capability, head_dim)


@pytest.mark.parametrize("off", ("0", "off", "false", "no", " NO "))
def test_native_p_kill_switch_compiles_the_magic_path(monkeypatch, off):
    monkeypatch.delenv(I.NATIVE_P_ENV, raising = False)
    cfg = I.kernel_config((8, 9), 128)
    assert cfg.get("native_p") == 1 and I.native_p_enabled()
    on = I.kernel_variant(cfg, False)
    assert I.kernel_defines(128, on)["NATIVE_P"] == 1
    monkeypatch.setenv(I.NATIVE_P_ENV, off)
    assert not I.native_p_enabled()
    for sinks in (False, True):
        variant = I.kernel_variant(cfg, sinks)
        assert "native_p" not in variant and "NATIVE_P" not in I.kernel_defines(128, variant)
    assert I._config_key(I.kernel_variant(cfg, False)) != I._config_key(on)
    assert cfg.get("native_p") == 1
    for keep in ("1", "", "on"):
        monkeypatch.setenv(I.NATIVE_P_ENV, keep)
        assert I.kernel_variant(cfg, False) == on


def test_native_p_switch_is_part_of_the_race_namespace(monkeypatch):
    """A race decided with native_p on is not reused with it off (the persisted namespace carries the switch)."""
    monkeypatch.delenv(I.NATIVE_P_ENV, raising = False)
    assert I._library_versions()["native_p"] is True
    monkeypatch.setenv(I.NATIVE_P_ENV, "0")
    assert I._library_versions()["native_p"] is False


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
def test_native_p_keeps_the_products_and_converts_each_probability_once():
    arch = max(a for a in _nvrtc_archs() if a in (80, 86, 89, 120))
    base = {"bn": 128, "wpc": 8, "nq": 1, "stages": 2}

    def ptx(**keys):
        return K.compile_ptx(
            K.KERNEL_SOURCE, f"sm_{arch}", I.kernel_defines(128, dict(base, **keys))
        ).decode(errors = "replace")

    plain = ptx()
    assert "cvt.rni" not in plain and "cvt.pack" not in plain
    for keys in (
        {"native_p": 1},
        {"native_p": 1, "qkw": 1, "pvw": 8, "spread": 5},
        {"native_p": 2},
    ):
        got = ptx(**keys)
        for op in ("mma.sync.aligned.m16n8k32", "cp.async.cg", "ex2.approx"):
            assert got.count(op) == plain.count(op), (keys, op)
        probabilities = 4 * (base["bn"] // 8)
        if keys["native_p"] == 1:
            assert got.count("cvt.rni.s32.f32") == probabilities
            assert got.count("cvt.pack.sat.u8.s32.b32") == probabilities // 2
        else:
            assert got.count("cvt.rni.sat.u8.f32") == probabilities
        assert "cvt.rn.f32.s32" not in got


_P_SWEEP = r"""
extern "C" __global__ void p_sweep(unsigned long long* bad) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __uint_as_float(i);
  if (x != x) return;  // NaN: the clamp gives 255, the conversion 0 (no NaN reaches it from finite inputs)
  const float nb[3] = {-1.f, 1.5849625f, 9.f};  // 2^x = 0.5 (a tie, to 0), about 3, 512 (to 255)
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    float v[4];
    int k = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) v[s] = s == slot ? x : nb[k++];
    const u32 a = pack4(prob(v[0]) + MAGIC_F, prob(v[1]) + MAGIC_F, prob(v[2]) + MAGIC_F, prob(v[3]) + MAGIC_F);
    const u32 b = pack4_sat(prob_rn(v[0]), prob_rn(v[1]), prob_rn(v[2]), prob_rn(v[3]));
    if (a != b) atomicAdd(bad, 1ull);
  }
}
"""


@needs_kernel
@pytest.mark.parametrize("native_p", (1, 2))
def test_native_p_packs_the_same_bytes_for_every_float(native_p):
    """All 2^32 fp32 inputs but NaN pack the MAGIC_F bytes; a truncating sweep must disagree (the check can fail)."""
    capability = torch.cuda.get_device_capability()
    stream = torch.cuda.current_stream().cuda_stream
    bad = {}
    for name, source in (
        ("rn", K.KERNEL_SOURCE),
        ("rz", K.KERNEL_SOURCE.replace("cvt.rni.", "cvt.rzi.")),
    ):
        fn = K.load_module(
            torch.cuda.current_device(),
            capability,
            source + _P_SWEEP,
            ("p_sweep",),
            {"NATIVE_P": native_p, "HD": 128, "BN": 128, "P_SWEEP": name},
        )["p_sweep"]
        count = torch.zeros(1, dtype = torch.int64, device = "cuda")
        fn.launch((1 << 24, 1), 256, (count.data_ptr(),), 0, stream)
        torch.cuda.synchronize()
        bad[name] = int(count.item())
    assert bad["rn"] == 0 and bad["rz"] > 0, bad


@needs_kernel
def test_native_p_does_not_change_the_output():
    """The card's kernel with native_p 0 and 1, sink path (fold on) and plain: the same bits."""
    entry = _cuda_config(128)
    for mode in (I.MODE_RACE, I.MODE_MAIN):
        for keys in (300, 1039):
            q, k, v = _case((1, keys, 3, 128), _card_dtype(), seed = keys)
            outs = []
            for native_p in (0, 1):
                I.forget_plans()
                p = I.quantise(q, k, v, dict(entry, native_p = native_p), mode = mode)
                assert bool(p.get("sinks")) == (mode == I.MODE_RACE)
                outs.append(I.attend(p))
            assert torch.equal(outs[0], outs[1]), (mode, keys)


ORDERINGS = ("qkw", "pvw", "spread")


def test_the_orderings_reach_the_kernel_as_defines():
    """qkw, pvw and spread are per-card keys that become compile-time defines; configs without them
    compile the kernel as before; head dim 128 sets all three on sm120 and two on sm75."""
    tuned = {((12, 0), 128): {"QKW": 1, "PVW": 8, "SPREAD": 5}, ((7, 5), 128): {"QKW": 1, "PVW": 8}}
    for capability in sorted(SHARED_LIMIT):
        for head_dim in I.HEAD_DIMS:
            defines = I.kernel_defines(head_dim, I.kernel_config(capability, head_dim))
            got = {
                name.upper(): defines[name.upper()] for name in ORDERINGS if name.upper() in defines
            }
            assert got == tuned.get((capability, head_dim), {}), (capability, head_dim)


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
def test_the_orderings_keep_the_products_and_refuse_bad_combinations():
    """Each ordering compiles to the same number of tensor-core products and tile copies and adds no int <-> float
    conversion; spread needs qkw, pvw an even divisor of the channel blocks, and a one-stage kernel has no next
    tile to spread."""
    arch = max(a for a in _nvrtc_archs() if a in (80, 86, 89, 120))
    base = {"bn": 128, "wpc": 8, "nq": 1, "stages": 2}

    def ptx(head_dim, **keys):
        return K.compile_ptx(
            K.KERNEL_SOURCE, f"sm_{arch}", I.kernel_defines(head_dim, dict(base, **keys))
        ).decode(errors = "replace")

    for head_dim in I.HEAD_DIMS:
        plain = ptx(head_dim)
        for keys in (
            {"qkw": 1},
            {"pvw": 2},
            {"pvw": 8},
            {"qkw": 1, "spread": 1},
            {"qkw": 1, "pvw": 8, "spread": 5},
            {"qkw": 1, "pvw": 8, "spread": 5, "nq": 2},
        ):
            got = ptx(head_dim, **keys)
            scale = keys.get("nq", 1)
            assert got.count("mma.sync.aligned.m16n8k32") == scale * plain.count(
                "mma.sync.aligned.m16n8k32"
            ), keys
            assert got.count("ldmatrix") <= scale * plain.count("ldmatrix"), keys
            assert got.count("cp.async.cg") == plain.count("cp.async.cg"), keys
            assert "cvt.rn.f32.s32" not in got and "cvt.rni" not in got
    assert ptx(128, qkw = 1, spread = 5, stages = 1) == ptx(128, qkw = 1, stages = 1)
    for keys in ({"spread": 1}, {"pvw": 3}, {"pvw": 6}):
        with pytest.raises(RuntimeError, match = "unsupported combination"):
            ptx(128, **keys)


def test_install_off_cuda_and_kill_switch(monkeypatch):
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cpu", dtype = torch.bfloat16) == 0
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16, force = True) == 0
    assert I.int8_attention_count(dit) == 0


def test_install_follows_the_race(monkeypatch):
    seen = _fake_cuda(
        monkeypatch,
        found = lambda head_dim: {"ok": True, "min_tokens": 2560 if head_dim == 128 else None},
    )
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16) == 2
    flags = [(m.processor._unsloth_int8, m.processor._unsloth_int8_min_tokens) for m in dit.blocks]
    assert flags == [(True, 2560), (True, 2560), (False, 0), (False, 0)]
    assert {s[1] for s in seen} == {128, 64} and all(s[2] == "_native_cudnn" and s[3] for s in seen)
    assert I.int8_attention_count(dit) == 2 and I.status(dit)["modules"] == 2
    assert I.remove_int8_attention(dit) == 2 and I.int8_attention_count(dit) == 0


def test_install_needs_the_accuracy_check_even_when_forced(monkeypatch):
    _fake_cuda(monkeypatch, found = lambda head_dim: {"ok": head_dim == 64, "min_tokens": None})
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16, force = True) == 1
    assert [m.processor._unsloth_int8 for m in dit.blocks] == [False, False, True, False]
    dit = _fake_dit()
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16) == 1


def test_install_scope_incumbent_and_refusals(monkeypatch):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit = _fake_dit()
    dit.blocks[1].processor._parallel_config = object()
    sage = ("sage_hub", lambda q, k, v: q)
    n = I.install_int8_attention(
        dit,
        device = "cuda",
        dtype = torch.bfloat16,
        modules = [dit.blocks[0], dit.blocks[1]],
        incumbent = sage,
    )
    assert n == 1 and [m.processor._unsloth_int8 for m in dit.blocks] == [True, False, False, False]
    assert seen == [(2, 128, "sage_hub", True)]
    _fake_cuda(monkeypatch, capability = (9, 0), found = {"ok": True, "min_tokens": 0})
    assert I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.bfloat16) == 0
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    assert I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.float64) == 0


def test_request_hook_needs_the_name_or_the_switch(monkeypatch):
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = _fake_dit())
    assert I.install_requested(pipe, None, target) == 0
    assert I.install_requested(pipe, "sage", target) == 0
    assert I.install_requested(pipe, "unsloth_int8", types.SimpleNamespace(device = "cpu")) == 0
    assert I.install_requested(pipe, "unsloth_int8", target) == 3
    pipe = types.SimpleNamespace(transformer = _fake_dit())
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert I.install_requested(pipe, None, target) == 3
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert (
        I.install_requested(types.SimpleNamespace(transformer = _fake_dit()), "unsloth_int8", target)
        == 0
    )


@pytest.mark.parametrize(
    "capability,dtype",
    [((7, 5), torch.float16), ((8, 9), torch.bfloat16), ((12, 0), torch.bfloat16)],
)
def test_request_hook_checks_fp16_on_sm75_when_the_target_names_no_dtype(
    monkeypatch, capability, dtype
):
    """sm75 has no bf16, so its pipelines run fp16: that is the dtype the card check has to pass in."""
    _fake_cuda(monkeypatch, capability = capability, found = {"ok": True, "min_tokens": 0})
    checked = []
    inner = I.verdict

    def verdict(device, dtype, *args, **kwargs):
        checked.append(dtype)
        return inner(device, dtype, *args, **kwargs)

    monkeypatch.setattr(I, "verdict", verdict)
    pipe = types.SimpleNamespace(transformer = _fake_dit())
    assert I.install_requested(pipe, "unsloth_int8", types.SimpleNamespace(device = "cuda")) == 3
    assert set(checked) == {dtype}


@pytest.mark.parametrize(
    "shape,bn", [((1, 300, 2, 64), 64), ((1, 200, 2, 128), 64), ((1, 260, 1, 128), 128)]
)
def test_the_method_stays_close_to_fp32_attention(shape, bn):
    q, k, v = I.probe_inputs("cpu", torch.float32, *shape[1:])
    p = reference_quantise(q, k, v, {"bn": bn, "wpc": 1, "nq": 1, "stages": 1})
    for name in ("qi", "ki", "vt"):
        assert p[name].abs().max() <= 127 and torch.equal(p[name], p[name].round())
    eye = torch.eye(shape[-1], dtype = torch.float64)
    rot = rotate_ref(eye)
    assert torch.equal(rot @ rot.T, float(min(I.ROT_GROUP, shape[-1])) * eye)
    acc = I.accuracy(emulate(p), I.reference_attention(q, k, v))
    assert acc["cos"] > 0.9995 and acc["rel_l1"] < 0.03, acc


@pytest.mark.parametrize("sinks", [(1,), (1030, 1031)])
def test_the_method_keeps_v_resolved_past_attention_sinks(sinks):
    """A V key 60 std out in most channels (an attention sink: the first text tokens of a joint sequence) draws almost no
    attention but set every channel's range, so every other key lost its resolution: the method's error against
    fp32 was several times that of the same call without it. Now the head clips the sink's stats chunk to the range
    of the others and adds the clipped-off part back exactly: the error stays at the sink-free level."""
    shape = (1, 1100, 2, 128)
    q, k, v = I.probe_inputs("cpu", torch.float32, *shape[1:])
    plain = reference_quantise(q, k, v, {"bn": 128, "wpc": 1, "nq": 1, "stages": 1})
    base = (emulate(plain) - I.reference_attention(q, k, v)).norm() / I.reference_attention(
        q, k, v
    ).norm()
    v = v.clone()
    for key in sinks:
        sign = torch.ones(shape[-1] - 16)
        sign[1::2] = -1
        v[:, key, :, 16:] += 60 * v[:, :, :, 16:].std() * sign
    p = reference_quantise(q, k, v, {"bn": 128, "wpc": 1, "nq": 1, "stages": 1})
    for bh in range(2):
        n = int(p["rcnt"][bh])
        keys = p["ridx"][bh, :n].tolist()
        assert set(sinks) <= set(keys) and keys == sorted(keys)
    ref = I.reference_attention(q, k, v)
    err = (emulate(p) - ref).norm() / ref.norm()
    assert err < 1.5 * base, (float(err), float(base))


def test_statistics_cover_every_key():
    for n in (1, 15, 16, 80, 1039, 4112, 10011, 75600, 300000):
        kstep, vstep, chunks, rows = I.stats_plan(n)
        assert vstep == 1 and rows % vstep == 0 and chunks <= I.V_RANGE_CHUNKS
        assert (chunks - 1) * rows < n <= chunks * rows
        assert 1 <= kstep <= I.K_MEAN_STEP and -(-n // kstep) >= min(n, I.K_MEAN_KEYS)


def test_the_quantisers_read_only_aligned_rows():
    """The quantisers load 16 bytes at a time: a tensor goes in as it is when its last-dim stride is 1 and every
    row starts on a 16-byte boundary (packed QKV views do), and as a contiguous copy otherwise."""
    H, D = 2, 64
    packed = torch.zeros(1, 9, 3 * H * D, dtype = torch.bfloat16)
    for view in packed.chunk(3, dim = -1):
        x = view.unflatten(-1, (H, D))
        assert I._strides_aligned(x) and I._aligned(x) is x
    odd = torch.zeros(1, 9, 3 * H * D + 1, dtype = torch.bfloat16)[..., : H * D].unflatten(-1, (H, D))
    assert not I._strides_aligned(odd)  # rows 2 bytes apart from a 16-byte boundary
    shifted = torch.zeros(1, 9, H, D + 8, dtype = torch.bfloat16)[..., 1 : D + 1]
    assert I._strides_aligned(shifted) and shifted.data_ptr() % 16 != 0
    sparse = torch.zeros(1, 9, H, 2 * D, dtype = torch.bfloat16)[..., ::2]
    assert not I._strides_aligned(sparse)
    for x in (odd, shifted, sparse):
        copy_ = I._aligned(x)
        assert copy_ is not x and copy_.is_contiguous() and copy_.data_ptr() % 16 == 0
        assert I._strides_aligned(copy_) and torch.equal(copy_, x)


def test_a_launch_plan_is_keyed_by_everything_it_depends_on(monkeypatch):
    """The launch arithmetic of a call is done once per signature: the same shape, dtypes, strides, config and
    scale find the plan again (the config by value: the per-card table hands out copies), any other call does not."""
    H, D = 2, 64
    config = {"bn": 128, "wpc": 4, "nq": 2, "stages": 2}
    packed = torch.zeros(1, 9, 3 * H * D, dtype = torch.bfloat16)
    q, k, v = (x.unflatten(-1, (H, D)) for x in packed.chunk(3, dim = -1))
    key = I._plan_key(q, k, v, config, None, 0)
    assert key == I._plan_key(q, k, v, dict(reversed(list(config.items()))), None, 0)
    assert hash(key) == hash(I._plan_key(q, k, v, dict(config), None, 0))
    others = [
        I._plan_key(q, k, v, dict(config, wpc = 8), None, 0),
        I._plan_key(q, k, v, config, 0.25, 0),
        I._plan_key(q, k, v, config, None, 1),
        I._plan_key(q.contiguous(), k, v, config, None, 0),  # other strides
        I._plan_key(q, k.float(), v.float(), config, None, 0),
        I._plan_key(q[:, :8], k, v, config, None, 0),
        I._plan_key(q, k[:, :8], v[:, :8], config, None, 0),
    ]
    assert len({key, *others}) == len(others) + 1
    monkeypatch.setattr(I, "V_RANGE_STEP", 8)  # the statistics plan is part of what a plan holds
    assert I._plan_key(q, k, v, config, None, 0) != key


def test_v_range_over_every_key_keeps_a_short_sequence_accurate(monkeypatch):
    """V range over every 8th key gives too small gains and 4x the error; over every key it is accurate."""
    q, k, v = I.probe_inputs("cpu", torch.float32, 16, 2, 128)
    v = v.clone()
    v[:, 3] *= 30.0
    config = {"bn": 64, "wpc": 1, "nq": 1, "stages": 1}
    ref = I.reference_attention(q, k, v)

    def run():
        got = emulate(reference_quantise(q, k, v, config))
        rows = F.cosine_similarity(got.flatten(0, 2), ref.flatten(0, 2), dim = -1)
        return I.accuracy(got, ref)["rel_l1"], float(rows.min())

    rel, worst = run()
    assert rel < 0.05 and worst > 0.99, (rel, worst)
    monkeypatch.setattr(I, "V_RANGE_STEP", 8)
    sampled, _ = run()
    assert sampled > 4 * rel, (sampled, rel)


def test_the_card_check_counts_for_its_dtype_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def operand(dtype):
        return types.SimpleNamespace(
            device = torch.device("cuda:0"), shape = (1, 8, 2, 64), dtype = dtype
        )

    bf16, fp16 = operand(torch.bfloat16), operand(torch.float16)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(I, "_VERIFIED", {(0, 64, "torch.bfloat16", "cuda")})
    assert I._verified(bf16, bf16, bf16)
    assert not I._verified(fp16, fp16, fp16) and not I._verified(bf16, bf16, fp16)
    assert not I._verified(operand(torch.bfloat16), bf16, operand(torch.float32))
    monkeypatch.setattr(
        I, "int8_attention", lambda q, k, v, scale = None, config = None: I.reference_attention(q, k, v)
    )
    monkeypatch.setattr(I, "_VERIFIED", set())
    assert I._check("cpu", torch.float32, 4, 64, None, timed = False)["ok"]
    assert I._VERIFIED == {(0, 64, "torch.float32", "cuda")}


def test_flux3_strided_processor_keeps_native_attention_for_calls_that_need_gradients(monkeypatch):
    tf3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    import contextlib

    from core.inference import diffusion_flux3_attn as fa

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda *a, **k: contextlib.nullcontext())
    fa.strided_processor_class.cache_clear()
    try:
        torch.manual_seed(1)
        attn = tf3.Flux3Attention(query_dim = 64, heads = 4, dim_head = 16).eval()
        proc = fa.strided_processor_class()()
        proc._attention_backend = "_native_cudnn"
        proc._unsloth_int8 = True
        calls = []
        monkeypatch.setattr(
            I,
            "int8_attention_impl",
            lambda q, k, v: (calls.append(tuple(q.shape)), I.native_attention(q, k, v))[1],
        )
        I.int8_attention_op()
        g = torch.Generator().manual_seed(2)
        ang = torch.rand(12, 8, generator = g) * 6.28
        rope = (
            torch.cos(ang).repeat_interleave(2, dim = -1),
            torch.sin(ang).repeat_interleave(2, dim = -1),
        )
        x = torch.randn(1, 12, 64, generator = g)
        with torch.no_grad():
            want = proc(attn, x, None, rope)
        assert len(calls) == 1
        x = x.clone().requires_grad_(True)
        got = proc(attn, x, None, rope)
        assert len(calls) == 1 and torch.allclose(got, want, atol = 1e-5, rtol = 1e-5)
        got.sum().backward()
        assert (
            x.grad is not None
            and bool(torch.isfinite(x.grad).all())
            and float(x.grad.abs().sum()) > 0
        )
        with torch.no_grad():
            proc(attn, x, None, rope)
        assert len(calls) == 2
    finally:
        fa.strided_processor_class.cache_clear()


def _h3():
    h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    from core.inference import video_minimax_h3_attn as A
    return h3, A


def _tiny_h3(h3, head_dim: int = 32):
    torch.manual_seed(0)
    return h3.MiniMaxH3Transformer3DModel(
        num_attention_heads = 2,
        attention_head_dim = head_dim,
        hidden_size = 2 * head_dim,
        num_layers = 2,
        num_refiner_layers = 1,
        ffn_dim = 128,
        in_channels = 4,
        audio_in_channels = 4,
        patch_size = (1, 2, 2),
        text_dim = 32,
        freq_dim = 32,
        time_embed_hidden_dim = 64,
        time_embed_dim = 32,
        rope_freq_dim = 4,
    ).eval()


def _h3_inputs(
    n_text: int = 5,
    n_video: int = 8,
    n_audio: int = 3,
) -> dict:
    g = torch.Generator().manual_seed(1)
    seq = n_text + n_video + n_audio
    tags = torch.zeros(seq, dtype = torch.long)
    tags[:n_text] = 1
    tags[n_text + n_video :] = 2
    return dict(
        hidden_states = torch.randn(1, n_video, 16, generator = g),
        audio_hidden_states = torch.randn(1, n_audio, 4, generator = g),
        encoder_hidden_states = torch.randn(1, n_text, 32, generator = g),
        timestep = torch.tensor([0.7, 0.2]),
        timestep_indices = torch.randint(0, 2, (seq,), generator = g),
        token_tags = tags,
        position_ids = torch.randint(0, 6, (seq, 3), generator = g),
        video_indices = torch.arange(n_text, n_text + n_video),
        audio_indices = torch.arange(n_text + n_video, seq),
        text_indices = torch.arange(0, n_text),
        return_dict = False,
    )


@pytest.fixture
def _active_backend():
    # Skip, not error, where diffusers is missing (the backend CPU job), as _h3() does for the test body.
    _AttentionBackendRegistry = pytest.importorskip(
        "diffusers.models.attention_dispatch"
    )._AttentionBackendRegistry

    active = _AttentionBackendRegistry._active_backend
    yield
    _AttentionBackendRegistry.set_active_backend(active)


def test_strided_processor_routes_flagged_calls_to_the_op(monkeypatch, _active_backend):
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    h3, A = _h3()
    model = _tiny_h3(h3)
    model.set_attention_backend("_native_math")
    # a processor with no backend dispatches to the registry's active one, which diffusers 0.43 no longer sets in
    # set_attention_backend
    _AttentionBackendRegistry.set_active_backend(AttentionBackendName("_native_math"))
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model) == 3
    calls = []
    monkeypatch.setattr(
        I,
        "int8_attention_impl",
        lambda q, k, v: (calls.append(tuple(q.shape)), I.native_attention(q, k, v))[1],
    )
    I.int8_attention_op()
    with torch.no_grad():
        want = stock(**_h3_inputs())
        assert all(torch.equal(a, b) for a, b in zip(want, model(**_h3_inputs()))) and not calls
        procs = [m.processor for m in I.int8_processors(model)]
        assert len(procs) == 3
        for proc in procs:
            proc._unsloth_int8 = True
        got = model(**_h3_inputs())
        assert calls and all(
            len(shape) == 4 and shape[2] == 2 and shape[3] == 32 for shape in calls
        )
        assert all(torch.allclose(a, b, atol = 1e-5, rtol = 1e-5) for a, b in zip(want, got))
        calls.clear()
        for proc in procs:
            proc._unsloth_int8_min_tokens = 10**6
        assert all(torch.equal(a, b) for a, b in zip(want, model(**_h3_inputs()))) and not calls
        for proc in procs:
            proc._unsloth_int8_min_tokens = 0
            proc._attention_backend = None
        got = model(**_h3_inputs())
        assert calls and all(torch.allclose(a, b, atol = 1e-5, rtol = 1e-5) for a, b in zip(want, got))
        calls.clear()
        for proc in procs:
            proc._unsloth_int8 = False
        assert all(torch.equal(a, b) for a, b in zip(want, model(**_h3_inputs()))) and not calls


def test_h3_strided_processor_keeps_native_attention_for_calls_that_need_gradients(
    monkeypatch, _active_backend
):
    h3, A = _h3()
    model = _tiny_h3(h3)
    model.set_attention_backend("_native_math")
    assert A.install_strided_attention(model) == 3
    calls = []
    monkeypatch.setattr(
        I,
        "int8_attention_impl",
        lambda q, k, v: (calls.append(tuple(q.shape)), I.native_attention(q, k, v))[1],
    )
    I.int8_attention_op()
    for m in I.int8_processors(model):
        m.processor._unsloth_int8 = True
        m.processor._unsloth_qk_rope = False
    with torch.no_grad():
        want = model(**_h3_inputs())
    assert calls
    calls.clear()
    inputs = _h3_inputs()
    inputs["hidden_states"].requires_grad_(True)
    got = model(**inputs)
    assert not calls
    assert all(torch.allclose(a, b, atol = 1e-5, rtol = 1e-5) for a, b in zip(want, got))
    sum(x.float().sum() for x in got).backward()
    grad = inputs["hidden_states"].grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_strided_processor_with_the_op_adds_no_graph_break(_active_backend):
    h3, A = _h3()

    def breaks(model) -> int:
        torch._dynamo.reset()
        with torch.no_grad():
            explained = torch._dynamo.explain(model)(**_h3_inputs())
        torch._dynamo.reset()
        return explained.graph_break_count

    model = _tiny_h3(h3)
    model.set_attention_backend("_native_math")
    assert A.install_strided_attention(model) == 3
    base = breaks(model)
    I.int8_attention_op()
    for m in I.int8_processors(model):
        m.processor._unsloth_int8 = True
    assert breaks(model) == base


def _h3_image():
    try:
        from core.inference import diffusion_minimax_h3_image as H
    except Exception as exc:  # noqa: BLE001 - the module needs diffusers' pipelines
        pytest.skip(f"the H3-Image module does not import here: {exc!r}"[:200])
    return H


def test_the_unset_switch_keeps_the_kernel_off_the_h3_image_blocks(monkeypatch):
    """MiniMax-H3-Image: the block race picks the kernel on L4 / G4 / A100, which then renders 6-13% slower end to
    end, so with the switch unset the blocks keep their backend with no check or race; forcing still routes them."""
    import logging

    H = _h3_image()
    installs: list = []
    monkeypatch.setattr(
        I, "install_int8_attention", lambda t, **kw: installs.append(kw) or len(kw["modules"])
    )
    monkeypatch.setattr(H, "race_checkpoint_attention", lambda *a, **k: pytest.fail("no race"))
    model, log = types.SimpleNamespace(), logging.getLogger(__name__)
    blocks = [types.SimpleNamespace(processor = types.SimpleNamespace()) for _ in range(2)]
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    for sage in (None, True, False):
        assert H._race_int8_kernel(model, blocks, {}, "cuda", torch.bfloat16, sage, False, log) == (
            0,
            None,
        )
    assert installs == []
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert H._race_int8_kernel(model, blocks, {}, "cuda", torch.bfloat16, True, True, log) == (
        2,
        None,
    )
    assert len(installs) == 1 and installs[0]["force"]


def test_checkpoint_race_times_the_kernel_on_a_block(monkeypatch, _active_backend):
    import logging
    import sys

    h3, A = _h3()
    H = _h3_image()
    monkeypatch.setattr(H, "KERNEL_AUTO", True)
    model = _tiny_h3(h3)
    model.set_attention_backend("_native_math")
    assert A.install_strided_attention(model) == 3
    mods = I.int8_processors(model)
    blocks = mods[:2]
    saved = {m: m.processor._attention_backend for m in mods}
    installs: list = []

    def fake_install(transformer, **kw):
        installs.append(kw)
        if kw.get("force"):
            for m in kw["modules"]:
                m.processor._unsloth_int8 = True
        return len(kw["modules"])

    race: dict = {}
    monkeypatch.setattr(I, "install_int8_attention", fake_install)
    monkeypatch.setattr(
        H,
        "race_checkpoint_attention",
        lambda *a, kernel = False: race.get("found") if kernel else 1 / 0,
    )
    log = logging.getLogger(__name__)

    def run(
        sage = False,
        forced = False,
        mods = blocks,
    ):
        installs.clear()
        for m in blocks:
            m.processor._unsloth_int8 = False
            m.processor._unsloth_int8_min_tokens = 0
        return H._race_int8_kernel(model, mods, saved, "cuda", torch.bfloat16, sage, forced, log)[0]

    def routed():
        return [(m.processor._unsloth_int8, m.processor._unsloth_int8_min_tokens) for m in blocks]

    race["found"] = {
        "native": "_native_math",
        "kernel": {4096: [10.0, 10.0, 9.0], 1024: [3.0, 3.0, 2.5]},
    }
    assert run() == 2 and routed() == [(True, 0)] * 2
    assert len(installs) == 1 and installs[0]["force"] and installs[0]["modules"] == blocks
    record = getattr(model, H.ATTENTION_RACE_ATTR)
    assert record["timed"] == "block" and record["blocks"] == 2 and record["min_tokens"] == 0
    assert record["ms"]["4096"] == [10.0, 10.0, 9.0] and record["seconds"] >= 0
    race["found"]["kernel"][1024] = [3.0, 3.0, 2.99]
    assert run() == 2 and routed() == [(True, H._KERNEL_LARGE_ROWS)] * 2
    race["found"]["kernel"][4096] = [10.0, 10.0, 9.9]
    assert run() == 0 and routed() == [(False, 0)] * 2
    race["found"]["kernel"] = {4096: [10.0, 8.0, 9.0], 1024: [3.0, 2.0, 2.5]}
    assert run(sage = True) == 0 and routed() == [(False, 0)] * 2
    assert run(sage = False) == 2
    race["found"]["kernel"] = {4096: [10.0, 8.0, 7.0], 1024: [3.0, 2.0, 2.5]}
    assert run(sage = True) == 2 and routed() == [(True, H._KERNEL_LARGE_ROWS)] * 2
    race["found"] = None
    assert run() == 2 and len(installs) == 2
    assert installs[1]["incumbent"][0] == "native" and "force" not in installs[1]
    assert getattr(model, H.ATTENTION_RACE_ATTR)["timed"] == "attention"
    calls = []
    base_module = sys.modules[A.h3_processor_base(mods[0]).__module__]
    monkeypatch.setattr(
        base_module,
        "dispatch_attention_fn",
        lambda q, k, v, **kw: (calls.append(kw), q)[1],
        raising = False,
    )
    mods[0].processor._attention_backend = "sage"
    assert run(sage = True) == 2
    name, fn = installs[1]["incumbent"]
    assert name == "sage" and fn(1, 2, 3) == 1
    assert calls[0]["backend"] == "sage" and calls[0]["attn_mask"] is None
    mods[0].processor._attention_backend = saved[mods[0]]
    race["found"] = {"native": "_native_math", "kernel": {4096: [1.0]}}
    assert run() == 0 and routed() == [(False, 0)] * 2
    assert "error" in getattr(model, H.ATTENTION_RACE_ATTR)
    assert run(forced = True) == 0 and not installs
    assert run(mods = []) == 0 and not installs
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert run() == 0 and not installs
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert run(forced = True) == 2 and len(installs) == 1 and routed() == [(True, 0)] * 2


def test_checkpoint_attention_races_each_backend_once(monkeypatch, _active_backend):
    """SageAttention's own race is not timed on DiT blocks the kernel takes without a race (its switch is 1), and the
    kernel's block race (which times native, SageAttention and the kernel) decides SageAttention as well."""
    import logging

    from core.inference import diffusion_attention as D

    h3, A = _h3()
    H = _h3_image()
    log = logging.getLogger(__name__)
    races: list = []
    plan: dict = {}

    def fake_race(
        transformer,
        saved,
        dtype,
        *,
        kernel = False,
        device = None,
    ):
        races.append(kernel)
        found = {"keep": plan["sage_keep"], "ms": [10.0, 9.0], "native": "_native_math"}
        if kernel:
            found["kernel"] = {4096: [10.0, 9.0, plan["kernel_ms"]], 1024: [3.0, 3.0, 3.0]}
        return found

    def fake_install(transformer, **kw):
        if not plan["ready"]:
            return 0
        if kw.get("force"):
            for m in kw["modules"]:
                m.processor._unsloth_int8 = True
        return len(kw["modules"])

    def fake_apply(
        pipe,
        want,
        logger = None,
        target = None,
    ):
        for m in H._attention_processors(pipe.transformer):
            m.processor._attention_backend = "sage_hub"
        return "sage_hub"

    monkeypatch.setattr(H, "race_checkpoint_attention", fake_race)
    monkeypatch.setattr(I, "install_int8_attention", fake_install)
    monkeypatch.setattr(D, "apply_attention_backend", fake_apply)

    def load(
        kernel_env = None,
        ready = True,
        sage_keep = True,
        kernel_ms = 8.0,
    ):
        if kernel_env is None:
            monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
        else:
            monkeypatch.setenv(I.INT8_ATTENTION_ENV, kernel_env)
        plan.update(ready = ready, sage_keep = sage_keep, kernel_ms = kernel_ms)
        races.clear()
        model = _tiny_h3(h3)
        model.set_attention_backend("_native_math")
        assert A.install_strided_attention(model) == 3
        pipe = types.SimpleNamespace(transformer = model)
        engaged = H.install_checkpoint_attention(pipe, "cuda", scheme = "int8", logger = log)
        blocks = [
            m.processor
            for b in model.transformer_blocks
            for m in b.modules()
            if hasattr(getattr(m, "processor", None), "_attention_backend")
        ]
        return engaged, list(races), blocks

    engaged, timed, blocks = load("1")
    assert engaged == I.BACKEND and timed == []
    assert all(p._unsloth_int8 for p in blocks)
    # switch unset: the kernel stays off the blocks (KERNEL_AUTO), SageAttention's own race decides them
    engaged, timed, blocks = load()
    assert engaged == "sage_hub" and timed == [False]
    assert not any(p._unsloth_int8 for p in blocks)
    monkeypatch.setattr(H, "KERNEL_AUTO", True)
    engaged, timed, blocks = load()
    assert engaged == I.BACKEND and timed == [True]
    engaged, timed, blocks = load(kernel_ms = 9.9)
    assert engaged == "sage_hub" and timed == [True]
    assert all(p._attention_backend == "sage_hub" and not p._unsloth_int8 for p in blocks)
    engaged, timed, blocks = load(kernel_ms = 9.9, sage_keep = False)
    assert engaged is None and timed == [True]
    assert all(p._attention_backend != "sage_hub" for p in blocks)
    assert load(ready = False)[:2] == ("sage_hub", [False])
    assert load("1", ready = False)[:2] == ("sage_hub", [False])
    assert load("0")[:2] == ("sage_hub", [False])
    assert load("0", sage_keep = False)[:2] == (None, [False])


class _ClockedAttention:
    """An attention module on a card whose clock steps once: every call takes its backend's cost times 1.0 before the
    ``step``-th call and ``after`` from it on. CUDA events read the same virtual clock."""

    COST = {"native": 10.0, "sage": 9.4, "kernel": 8.8}

    def __init__(self, step: int, after: float, kernel: bool):
        cls = type("Proc", (), {"_unsloth_int8": False} if kernel else {})
        self.processor = cls()
        self.processor._attention_backend = "sage"
        self.processor._unsloth_int8_min_tokens = 0
        self.step, self.after = step, after
        self.calls = 0
        self.now = 0.0

    def parameters(self):
        return iter([types.SimpleNamespace(is_cuda = True, device = torch.device("cpu"))])

    def __call__(self, x, rotary, mask):
        proc = self.processor
        name = "kernel" if getattr(proc, "_unsloth_int8", False) else proc._attention_backend
        factor = 1.0 if self.calls < self.step else self.after
        self.now += self.COST[name] * factor * x.shape[1] / 4396
        self.calls += 1


def _clocked_race(monkeypatch, step: int, after: float, kernel: bool):
    H = _h3_image()
    attn = _ClockedAttention(step, after, kernel)

    class Event:
        def __init__(self, enable_timing = False):
            self.t = 0.0

        def record(self):
            self.t = attn.now

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.t - self.t

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    block = types.SimpleNamespace(attn = attn, norm1 = types.SimpleNamespace(normalized_shape = (8,)))
    transformer = types.SimpleNamespace(transformer_blocks = [block], rope = lambda pos: None)
    found = H.race_checkpoint_attention(transformer, {attn: "native"}, torch.float32, kernel = kernel)
    return found, attn.calls


@pytest.mark.parametrize("after", [0.85, 1.15])
@pytest.mark.parametrize("kernel", [False, True])
def test_checkpoint_race_decision_does_not_follow_a_clock_step(monkeypatch, after, kernel):
    """SageAttention is 6% faster than native here and the kernel 6% faster than SageAttention: wherever in the race
    the card's clock steps by 15% either way, the race keeps SageAttention, and the kernel wins against it."""
    H = _h3_image()
    _, total = _clocked_race(monkeypatch, 10**6, 1.0, kernel)
    for step in range(total + 1):
        found, _ = _clocked_race(monkeypatch, step, after, kernel)
        assert found["keep"], (step, found)
        if kernel:
            large = found["paired"][H._KERNEL_RACE_IMAGES[0]]
            assert large[2] < I.RACE_MARGIN, (step, found)


class _StreamedAttention(_ClockedAttention):
    """Host-resident attention that only runs while ``_weights_on`` has staged it."""

    COST = {"native": 10.0, "sage": 10.0}

    def __init__(self):
        super().__init__(10**6, 1.0, False)
        self.on_card = False
        self.staged: list = []

    def parameters(self):
        return iter([types.SimpleNamespace(is_cuda = self.on_card, device = torch.device("cpu"))])

    def __call__(self, x, rotary, mask):
        assert (
            self.on_card
        ), "the race ran a streamed block's attention with its weights on the host"
        super().__call__(x, rotary, mask)


def _streamed_race(monkeypatch, device):
    H = _h3_image()
    attn = _StreamedAttention()

    class Event:
        def __init__(self, enable_timing = False):
            self.t = 0.0

        def record(self):
            self.t = attn.now

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.t - self.t

    @contextlib.contextmanager
    def weights_on(module, dev):
        assert module is attn and not attn.on_card
        attn.staged.append(str(dev))
        attn.on_card = True
        try:
            yield module
        finally:
            attn.on_card = False

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(H, "_weights_on", weights_on)
    norm = types.SimpleNamespace(normalized_shape = (8,))
    blocks = [types.SimpleNamespace(attn = attn, norm1 = norm)]
    blocks.append(types.SimpleNamespace(attn = _StreamedAttention(), norm1 = norm))
    transformer = types.SimpleNamespace(transformer_blocks = blocks, rope = lambda pos: None)
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        # no card here: the race's inputs stay on the host
        kwargs.pop("device", None)
        args = tuple(a for a in args if not isinstance(a, (str, torch.device)))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    found = H.race_checkpoint_attention(transformer, {attn: "native"}, torch.float32, device = device)
    return found, attn


def test_a_fully_streamed_dit_races_its_first_block_on_the_card(monkeypatch):
    """A fully streamed DiT still races (staged block) and decides as a resident layout: a tie keeps native."""
    found, attn = _streamed_race(monkeypatch, "cuda")
    assert found is not None and found["staged"] and not found["keep"]
    assert found["ms"] == [10.0 * 4396 / 4396] * 2
    assert (
        attn.staged == ["cuda"]
        and not attn.on_card
        and attn.calls == 2 * (2 + _h3_image()._RACE_ROUNDS)
    )
    assert attn.processor._attention_backend == "sage"
    # no device to stage on: untimed, the caller keeps its route
    found, attn = _streamed_race(monkeypatch, None)
    assert found is None and attn.calls == 0 and not attn.staged
    found, attn = _streamed_race(monkeypatch, "cpu")
    assert found is None and attn.calls == 0 and not attn.staged


def test_streamed_weights_come_back_to_the_host_after_the_race():
    """``_weights_on`` restores the same host tensors, also when the body raises."""
    H = _h3_image()
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    m[0].register_buffer("scale", torch.ones(4))
    before = {n: t for n, t in list(m.named_parameters()) + list(m.named_buffers())}
    with H._weights_on(m, torch.device("meta")):
        inside = list(m.parameters()) + list(m.buffers())
        assert inside and all(t.device.type == "meta" for t in inside)
        assert all(not p.requires_grad for p in m.parameters())
    after = {n: t for n, t in list(m.named_parameters()) + list(m.named_buffers())}
    assert after.keys() == before.keys() and all(after[n] is before[n] for n in before)
    with pytest.raises(RuntimeError):
        with H._weights_on(m, torch.device("meta")):
            raise RuntimeError("race failed")
    after = {n: t for n, t in list(m.named_parameters()) + list(m.named_buffers())}
    assert all(after[n] is before[n] for n in before)


def _fake_linear(in_features, out_features, device):
    """An nn.Linear whose weights are fake tensors on ``device``, built inside an active FakeTensorMode.

    ``nn.Linear(..., device="cuda:N")`` initialises its weights, and on a CPU-only torch build the fake
    ``uniform_`` reaches CUDA's generator and raises "PyTorch is not linked with support for cuda devices";
    moving a fake CUDA tensor that requires grad aborts the process there. Inference weights carry no grad,
    so these are plain uninitialised fakes and the test runs on any build."""
    lin = torch.nn.Linear(in_features, out_features, device = "meta")
    lin.weight = torch.nn.Parameter(
        torch.empty(out_features, in_features, device = device), requires_grad = False
    )
    lin.bias = torch.nn.Parameter(torch.empty(out_features, device = device), requires_grad = False)
    return lin


def test_streamed_weights_on_another_card_are_staged_too(monkeypatch):
    """A tensor on cuda:0 is staged when the race runs on cuda:1; one already on the stage card stays put."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    H = _h3_image()
    with FakeTensorMode(allow_non_fake_inputs = True):
        m = _fake_linear(4, 4, "cuda:0")
        m.register_buffer("scale", torch.ones(4, device = "cuda:1"))
        scale = m.scale
        with H._weights_on(m, torch.device("cuda", 1)):
            assert m.weight.device == torch.device("cuda", 1)
            assert m.bias.device == torch.device("cuda", 1)
            assert m.scale is scale
        assert m.weight.device == torch.device("cuda", 0)
        # ``cuda`` without an index is the current card (the race enters ``torch.cuda.device`` first).
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        with H._weights_on(m, torch.device("cuda")):
            assert m.weight.device == torch.device("cuda", 1)
            assert m.scale is scale


def test_install_races_a_fully_streamed_dit_on_its_device(monkeypatch):
    """``install_checkpoint_attention`` passes the load device so a fully streamed DiT is raced."""
    H = _h3_image()
    seen: list = []

    def race(transformer, saved, dtype, **kw):
        seen.append(kw)
        return {"keep": False, "ms": [1.0, 1.0], "native": "_native_flash", "staged": True}

    proc = types.SimpleNamespace(_attention_backend = "_native_flash")
    attn = torch.nn.Module()
    attn.processor = proc
    attn.register_parameter("w", torch.nn.Parameter(torch.zeros(1), requires_grad = False))
    block = torch.nn.Module()
    block.attn = attn
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([block])
    pipe = types.SimpleNamespace(transformer = transformer)
    from core.inference import diffusion_attention as da

    def fake_apply(
        p,
        backend,
        logger = None,
        target = None,
    ):
        proc._attention_backend = backend
        return backend

    monkeypatch.setattr(da, "apply_attention_backend", fake_apply)
    monkeypatch.setattr(H, "race_checkpoint_attention", race)
    monkeypatch.delenv(H.INT8_ATTENTION_ENV, raising = False)
    assert H.install_checkpoint_attention(pipe, "cuda", scheme = "int8") is None
    assert proc._attention_backend == "_native_flash"
    assert seen and str(seen[0].get("device")) == "cuda"


def _case(
    shape,
    dtype,
    seed = 0,
    strided = False,
):
    B, S, H, D = shape
    g = torch.Generator(device = "cuda").manual_seed(seed)
    q = torch.randn(B, S, H, D, generator = g, device = "cuda") * 1.5
    k = torch.randn(B, S, H, D, generator = g, device = "cuda") * 1.5 + 0.7
    v = (
        torch.randn(B, S, H, D, generator = g, device = "cuda")
        * (torch.rand(1, 1, H, D, generator = g, device = "cuda") * 2)
        + 0.5
    )
    q, k, v = (x.to(dtype) for x in (q, k, v))
    if strided:
        qkv = torch.cat([x.flatten(2) for x in (q, k, v)], dim = -1)
        q, k, v = (x.unflatten(-1, (H, D)) for x in qkv.chunk(3, dim = -1))
    return q, k, v


@needs_kernel
@pytest.mark.parametrize(
    "shape,dtype,config",
    [
        ((1, 64, 1, 128), torch.bfloat16, None),
        ((1, 150, 2, 64), torch.bfloat16, None),
        ((2, 333, 3, 128), torch.bfloat16, {"bn": 64, "wpc": 3, "nq": 1, "stages": 2}),
        ((1, 700, 4, 128), torch.float16, {"bn": 128, "wpc": 2, "nq": 1, "stages": 1}),
        ((1, 513, 2, 64), torch.float32, {"bn": 128, "wpc": 4, "nq": 1, "stages": 2}),
        ((1, 1025, 4, 128), torch.bfloat16, None),
        ((1, 777, 2, 64), torch.bfloat16, {"bn": 64, "wpc": 2, "nq": 2, "stages": 2}),
        ((1, 1300, 3, 64), torch.float16, {"bn": 64, "wpc": 5, "nq": 2, "stages": 1}),
        ((1, 2000, 2, 128), torch.bfloat16, None),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_kernel_matches_its_torch_model(shape, dtype, config, strided):
    q, k, v = _case(shape, dtype, seed = shape[1], strided = strided)
    config = config or _cuda_config(shape[-1])
    p = I.quantise(q, k, v, config)
    ref = reference_quantise(q, k, v, config, p["kc"], p["vc"], p["vg"])
    for name in ("qi", "ki", "vt"):
        got = p[name].double()[:, : shape[1]] if name == "qi" else p[name].double()
        diff = (got - ref[name]).abs()
        # the reference rounds in fp64, the quantiser in fp32: never more than one step, and rarely that
        assert diff.max() <= 1 and (diff > 0).double().mean() < 2e-3, name
    for name in ("qs", "ks", "vs"):
        got = p[name].double()[:, : shape[1]] if name == "qs" else p[name].double()
        assert (
            (got - ref[name].double()).abs() / ref[name].double().abs().clamp_min(1e-30)
        ).max() < 1e-5, name
    own = reference_quantise(q, k, v, config)
    for name in ("kc", "vc", "vg"):
        assert torch.allclose(p[name], own[name].to(p[name].device), rtol = 1e-4, atol = 1e-4), name
    out = I.attend(p)
    assert out.dtype == dtype and out.shape == q.shape and torch.isfinite(out.float()).all()
    want = emulate(p)
    rel = ((out.float() - want).abs().mean() / want.abs().mean()).item()
    assert rel < (2e-3 if dtype == torch.float32 else 4e-3), rel
    assert torch.equal(I.int8_attention(q, k, v, config = config), out)


def _sink_case(
    shape,
    dtype,
    sinks,
    seed = 0,
):
    """q / k / v of ``_case`` with attention sinks in V: the keys ``sinks`` 120 out in most channels (the first text
    tokens of a joint sequence carry such values) on every other head."""
    q, k, v = _case(shape, dtype, seed = seed)
    g = torch.Generator(device = "cuda").manual_seed(seed + 1)
    v = v.clone()
    H, D = shape[2], shape[3]
    for key in sinks:
        bump = torch.randn(shape[0], 1, H, D, generator = g, device = "cuda").sign() * 120
        bump[:, :, 1::2] = 0
        bump[..., : D // 8] = 0
        v[:, key : key + 1] += bump.to(dtype)
    return q, k, v


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
@pytest.mark.parametrize("sinks", [(4097, 4098), (1,), (2, 3)])
def test_attention_sinks_keep_the_rest_of_v_resolved(head_dim, sinks):
    """A few V keys 60x out set the range of every key: sink heads clip that chunk and add the
    clipped parts back after the kernel; error within 4x of no sinks (3.1x at most on T4, A100,
    L4; 5.5x or more with the range over every key); heads without sinks keep the full range."""
    shape = (1, 4128, 4, head_dim)
    dtype = _card_dtype()
    config = _cuda_config(head_dim)
    q, k, v = _sink_case(shape, dtype, sinks, seed = 11)
    out = I.int8_attention(q, k, v, config = config, mode = I.MODE_RACE)
    ref = I.reference_attention(q, k, v)
    plain = I.reference_attention(*_case(shape, dtype, seed = 11))
    plain_out = I.int8_attention(*_case(shape, dtype, seed = 11), config = config, mode = I.MODE_RACE)

    def err(x, r, heads):
        return float((x.float()[:, :, heads] - r[:, :, heads]).norm() / r[:, :, heads].norm())

    sink_heads, other = list(range(0, 4, 2)), list(range(1, 4, 2))
    assert err(out, ref, sink_heads) < 4.0 * err(plain_out, plain, sink_heads), (
        err(out, ref, sink_heads),
        err(plain_out, plain, sink_heads),
    )
    assert err(out, ref, other) < 1.5 * err(plain_out, plain, other)
    p = I.quantise(q, k, v, config, mode = I.MODE_RACE)
    counts = p["rcnt"].tolist()
    assert all(c > 0 for c in counts[::2]) and all(c == 0 for c in counts[1::2]), counts
    keys = {int(x) for x in p["ridx"][0, : counts[0]].tolist()}
    assert set(sinks) <= keys and keys == set(sorted(keys)), keys
    got = I.attend(p)
    assert torch.equal(got, out)
    want = emulate(p)
    rel = ((got.float() - want).abs().mean() / want.abs().mean()).item()
    assert rel < 4e-3, rel


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_more_clipped_keys_than_one_staging_pass(head_dim):
    """A head with more clipped keys than unsloth_i8resid stages at once (32) adds them back in several passes: the
    kernel still matches its torch model, and the keys are the bumped ones, in key order."""
    shape = (1, 2100, 2, head_dim)
    sinks = tuple(range(3, 3 + 40))
    q, k, v = _sink_case(shape, _card_dtype(), sinks, seed = 7)
    p = I.quantise(q, k, v, _cuda_config(head_dim), mode = I.MODE_RACE)
    counts = p["rcnt"].tolist()
    assert counts[0] >= len(sinks) > 32 and counts[1] == 0, counts
    keys = p["ridx"][0, : counts[0]].tolist()
    assert set(sinks) <= set(keys) and keys == sorted(keys)
    want = emulate(p)
    got = I.attend(p).float()
    rel = ((got - want).abs().mean() / want.abs().mean()).item()
    assert rel < 4e-3, rel


@needs_kernel
@pytest.mark.parametrize("with_sinks", [False, True])
def test_a_plan_keeps_the_sink_path_only_where_its_first_calls_clipped(with_sinks, monkeypatch):
    """A launch plan's first calls take the attention-sink path and report whether a head clipped; from its
    SINK_DECIDE_AT-th call a plan without sinks runs the kernels without that path, with the same bits (no head of it
    clips), and one with sinks keeps it where it wins its race (here: always)."""
    monkeypatch.setattr(I, "_sink_race", lambda *a: True)
    shape = (1, 700, 4, 128)
    config = _cuda_config(128)
    if with_sinks:
        q, k, v = _sink_case(shape, _card_dtype(), (1,), seed = 3)
    else:
        q, k, v = _case(shape, _card_dtype(), seed = 3)
    I.forget_plans()
    first = I.int8_attention(q, k, v, config = config, mode = I.MODE_RACE)
    (plan,) = I._plans().values()
    launched = []
    kernels = {
        id(f): name for name, f in (("sink", plan["kernels"][2]), ("base", plan["kernels0"][2]))
    }
    launch = K.Function.launch

    def spy(self, *args):
        launched.append(kernels.get(id(self)))
        return launch(self, *args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(K.Function, "launch", spy)
        for _ in range(I.SINK_DECIDE_AT):
            out = I.int8_attention(q, k, v, config = config, mode = I.MODE_RACE)
    assert (
        plan["sinks"] is with_sinks
        and plan["calls"] == I.SINK_DECIDE_AT + 1
        and plan["checks"] == []
    )
    # the attention launches: the last call only is past the decision
    assert launched.count("sink") == I.SINK_DECIDE_AT - (0 if with_sinks else 1)
    assert launched.count("base") == (0 if with_sinks else 1)
    assert torch.equal(out, first)


@needs_kernel
def test_a_new_prompt_length_learns_its_own_sinks(monkeypatch):
    """Plans sharing a DECIDE_BUCKET with a decided sink-free plan learn their own sink verdict from
    their first sink-path calls; a sink-free verdict never applies to a call with sinks."""
    monkeypatch.setattr(I, "_sink_race", lambda *a, **k: pytest.fail("no sinks: no race"))
    config = _cuda_config(128)
    I.forget_plans()
    q, k, v = _case((1, 700, 4, 128), _card_dtype(), seed = 3)
    for _ in range(I.SINK_DECIDE_AT + 1):
        I.int8_attention(q, k, v, config = config, mode = I.MODE_RACE)
    q, k, v = _case((1, 710, 4, 128), _card_dtype(), seed = 4)
    I.int8_attention(q, k, v, config = config, mode = I.MODE_RACE)
    first, second = I._plans().values()
    assert first["sinks"] is False and second["sinks"] is None and len(second["checks"]) == 1


@needs_kernel
def test_a_plan_whose_sink_path_loses_its_race_runs_torch_attention(monkeypatch):
    """A plan with attention sinks races the sink path against torch's attention on its deciding call's own inputs
    (the guarded op's path, _cuda_attention); where the path loses, that call and the next ones run torch's
    attention."""
    races = []
    monkeypatch.setattr(
        I, "_sink_race", lambda plan, q, k, v, scale, index, **_: races.append(q.shape) or "native"
    )
    shape = (1, 700, 4, 128)
    config = _cuda_config(128)
    q, k, v = _sink_case(shape, _card_dtype(), (1,), seed = 3)
    index = torch.cuda.current_device()
    I.forget_plans()
    for _ in range(I.SINK_DECIDE_AT):
        I._cuda_attention(q, k, v, config, None, index, I.MODE_RACE)
    (plan,) = I._plans().values()
    assert plan["sinks"] is None and races == []
    out = I._cuda_attention(q, k, v, config, None, index, I.MODE_RACE)
    assert plan["sinks"] == "native" and races == [shape]
    assert torch.equal(out, I.native_attention(q, k, v))
    assert torch.equal(
        I._cuda_attention(q, k, v, config, None, index, I.MODE_RACE), I.native_attention(q, k, v)
    )
    assert len(races) == 1


@needs_kernel
def test_the_sink_race_times_the_sink_path_on_the_call(monkeypatch):
    """_sink_race: the sink path (quantisers, kernel, residual launch) against torch's attention on the call's own
    inputs, counted toward no decision; its verdict is kept on the plan."""
    shape = (1, 2100, 4, 128)
    q, k, v = _sink_case(shape, _card_dtype(), (1,), seed = 5)
    I.forget_plans()
    I.int8_attention(q, k, v, config = _cuda_config(128), mode = I.MODE_RACE)
    (plan,) = I._plans().values()
    calls = plan["calls"]
    path = I._sink_race(plan, q, k, v, None, torch.cuda.current_device())
    assert (
        path in (True, False, "native")
        and set(plan["sink_race"]) == {"incumbent", "sinks", "plain"}
        and set(plan["sink_error"]) == {"sinks", "plain"}
        and plan["calls"] == calls
    )
    # the kernels without the sink path are picked only within PLAIN_ERROR_SLACK of its error
    assert (
        path is not False
        or plan["sink_error"]["plain"] <= I.PLAIN_ERROR_SLACK * plan["sink_error"]["sinks"]
    )


def test_the_sink_path_is_decided_by_the_first_calls_of_a_plan(monkeypatch):
    """_choose_path: every call takes the sink path until the plan's SINK_DECIDE_AT-th; from there only a plan one of
    whose first SINK_LEARN_CALLS calls clipped keeps it (a clip after them does not count), and a CUDA graph capture
    keeps what the plan has decided without counting a call."""

    class Event:
        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for clipped_at in (None, 0, I.SINK_LEARN_CALLS - 1, I.SINK_LEARN_CALLS):
        plan = {"calls": 0, "sinks": None, "checks": []}
        paths = []
        for _ in range(I.SINK_DECIDE_AT + 3):
            sinks, n = I._choose_path(plan)
            paths.append(sinks)
            if n < I.SINK_LEARN_CALLS and plan["sinks"] is None:
                plan["checks"].append(
                    (Event(), torch.tensor([0, int(n == clipped_at)], dtype = torch.int32))
                )
        keeps = clipped_at is not None and clipped_at < I.SINK_LEARN_CALLS
        assert all(paths[: I.SINK_DECIDE_AT + 1 if keeps else I.SINK_DECIDE_AT])
        assert all(path is keeps for path in paths[I.SINK_DECIDE_AT :])
        assert plan["sinks"] is keeps and plan["checks"] == []
    # a plan with sinks takes the path its race picks: the sink path, the kernels without it, or torch's attention
    for won in (True, False, "native"):
        races = []
        plan = {
            "calls": I.SINK_DECIDE_AT,
            "sinks": None,
            "checks": [(Event(), torch.tensor([1], dtype = torch.int32))],
        }
        path, _ = I._choose_path(plan, race = lambda: races.append(1) or won)
        assert path == won == plan["sinks"] and races == [1]
        assert I._choose_path(plan, race = lambda: races.append(1) or won)[0] == path and races == [1]
    plan = {
        "calls": I.SINK_DECIDE_AT,
        "sinks": None,
        "checks": [(Event(), torch.tensor([0], dtype = torch.int32))],
    }
    assert I._choose_path(plan, race = lambda: pytest.fail("no sinks: no race")) == (
        False,
        I.SINK_DECIDE_AT,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for decided, path in ((False, False), (True, True), (None, True), ("native", "native")):
        plan = {"calls": 7, "sinks": decided, "checks": []}
        assert I._choose_path(plan) == (path, None) and plan["calls"] == 7


def test_a_plan_reads_and_decides_no_sinks_while_another_thread_records(monkeypatch):
    """Event records and waits are prohibited while another thread records a CUDA graph (thread_local capture): there
    a call's clipped-key count goes unread and the decision waits for a later call, which makes it."""
    from core.inference import diffusion_cuda_graph as G

    class Event:
        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "Event", lambda: pytest.fail("no event record while recording"))
    monkeypatch.setattr(G, "_CAPTURE_DEPTH", 1)
    plan = {"calls": 0, "sinks": None, "checks": []}
    I._check_sinks(plan, None, 0)
    assert plan["checks"] == []
    plan = {
        "calls": I.SINK_DECIDE_AT,
        "sinks": None,
        "checks": [(Event(), torch.tensor([1], dtype = torch.int32))],
    }
    race = lambda: pytest.fail("no race while recording")  # noqa: E731
    assert I._choose_path(plan, race = race) == (True, I.SINK_DECIDE_AT)
    assert plan["sinks"] is None and len(plan["checks"]) == 1
    monkeypatch.setattr(G, "_CAPTURE_DEPTH", 0)
    assert I._choose_path(plan, race = lambda: "native") == ("native", I.SINK_DECIDE_AT + 1)
    assert plan["sinks"] == "native" and plan["checks"] == []


def test_the_sink_race_keeps_the_faster_path_as_accurate_as_the_sink_path():
    """_pick_path: the faster of the sink path and the kernels without it, those only within PLAIN_ERROR_SLACK of the
    sink path's error (Qwen-Image-2.1-Turbo keeps the old kernels where faster, Z-Image-Turbo's sinks keep the path),
    and torch's attention unless that beats it by RACE_MARGIN."""
    pick = I._pick_path
    near, far = {"sinks": 0.020, "plain": 0.0207}, {"sinks": 0.021, "plain": 0.034}
    assert pick({"incumbent": 1.53, "sinks": 1.40, "plain": 1.30}, near) is False
    assert pick({"incumbent": 1.53, "sinks": 1.40, "plain": 1.30}, far) is True
    assert pick({"incumbent": 1.53, "sinks": 1.30, "plain": 1.40}, near) is True
    assert pick({"incumbent": 1.25, "sinks": 1.29, "plain": 1.20}, near) is False
    assert pick({"incumbent": 1.25, "sinks": 1.29, "plain": 1.20}, far) == "native"
    assert pick({"incumbent": 1.25, "sinks": 1.29, "plain": 1.24}, near) == "native"
    # Qwen-Image-2.1-Turbo on G4: the sink path 1.7% faster than the old kernels, as accurate: a near tie keeps the old
    assert (
        pick(
            {"incumbent": 0.8704, "sinks": 0.6017, "plain": 0.6119},
            {"sinks": 0.01263, "plain": 0.01275},
        )
        is False
    )
    # the old kernels kept only while they beat torch's attention, else the sink path where it does
    assert pick({"incumbent": 1.0, "sinks": 0.97, "plain": 0.985}, near) is True


def test_the_sink_race_never_takes_a_path_with_a_non_finite_error():
    """A NaN / inf error (a broken output on the deciding call) disqualifies that path: the kernels without the sink
    path stand alone where theirs is finite, torch's attention where neither is; NaN timings never win."""
    pick, nan, inf = I._pick_path, float("nan"), float("inf")
    fast = {"incumbent": 10.0, "sinks": 4.0, "plain": 5.0}
    assert pick(fast, {"sinks": nan, "plain": 0.02}) is False
    assert pick(fast, {"sinks": inf, "plain": 0.02}) is False
    assert pick(fast, {"sinks": 0.02, "plain": nan}) is True
    assert pick(fast, {"sinks": nan, "plain": nan}) == "native"
    assert (
        pick({"incumbent": nan, "sinks": 4.0, "plain": 5.0}, {"sinks": 0.02, "plain": 0.02})
        == "native"
    )


def test_a_plan_that_decides_torch_attention_takes_its_bucket_out_of_the_raced_routes(monkeypatch):
    """Deciding torch attention lifts the token floor of raced routes past DECIDE_BUCKET for good
    and for the other plans in that bucket; a pipeline load forgets both."""
    from core.inference import diffusion_int8_attention_dispatch as D

    class Event:
        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    I.forget_plans()
    procs = [
        types.SimpleNamespace(_unsloth_int8 = True, _unsloth_int8_min_tokens = 2560) for _ in range(3)
    ]
    for proc, (index, head_dim) in zip(procs, ((0, 128), (1, 128), (0, 64))):
        I.register_route(index, head_dim, I._raise_processor_floor(proc))
    stock = object()
    strided = types.SimpleNamespace(
        _unsloth_int8 = True, _unsloth_int8_min_tokens = 2560, _unsloth_stock = stock
    )
    module = types.SimpleNamespace(processor = strided)
    I.register_route(0, 128, I._raise_processor_floor(strided, module))
    route = D.Int8Route("native", 2560, torch.bfloat16, 0, 128, cards = {0: 2560, 1: 3072})
    I.register_route(0, 128, D._raise_route_floor(route, 0))
    D._DECISIONS["stale"] = (route, None)
    plan = {
        "calls": I.SINK_DECIDE_AT,
        "sinks": None,
        "checks": [(Event(), torch.tensor([1], dtype = torch.int32))],
        "index": 0,
        "shape": (1, 4176, 4176, 24, 128),
        "decision": ("bucket",),
    }
    assert I._choose_path(plan, race = lambda: "native")[0] == "native"
    assert [p._unsloth_int8_min_tokens for p in procs] == [4352, 2560, 2560]
    # a processor that replaced a stock one gives it back, so the module's calls compile as before the install
    assert module.processor is stock and strided._unsloth_int8_min_tokens == 2560
    assert route.cards == {0: 4352, 1: 3072} and route.min_tokens == 3072 and not D._DECISIONS
    assert I._decisions() == {("bucket", True): "native"}
    I.forget_plans()
    assert I._decisions() == {} and I._ROUTES == []


def test_the_races_time_the_path_of_a_plan_without_sinks(monkeypatch):
    """Probe timing inside _median_ms takes the sink-free path and counts toward no decision;
    outside it the plan decides again."""

    class Event:
        def __init__(self, **_):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    plan = {"calls": 0, "sinks": None, "checks": []}
    seen = []
    I._median_ms({"int8": lambda: seen.append(I._choose_path(plan))})
    assert seen and set(seen) == {(False, None)} and plan["calls"] == 0
    assert I._choose_path(plan) == (True, 0) and plan["calls"] == 1


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
def test_without_sinks_the_kernels_have_no_sink_path():
    """SINKS compiles the attention-sink path in (two more attention-kernel parameters, the robust V range, the
    residual job and unsloth_i8resid); without it the kernels have none of it."""
    arch = max(_nvrtc_archs())
    for head_dim in I.HEAD_DIMS:
        config = {"bn": 128, "wpc": 8, "nq": 1, "stages": 2}
        defines = I.kernel_defines(head_dim, config)
        plain = K.compile_ptx(K.KERNEL_SOURCE, f"sm_{arch}", defines).decode(errors = "replace")
        sink = K.compile_ptx(K.KERNEL_SOURCE, f"sm_{arch}", dict(defines, SINKS = 1)).decode(
            errors = "replace"
        )
        params = [len(set(re.findall(r"unsloth_i8attn_param_\d+", x))) for x in (plain, sink)]
        assert params[1] == params[0] + 2, params
        quant = {
            "HD": head_dim,
            "BN": int(config["bn"]),
            "RGX": min(I.ROT_GROUP, head_dim).bit_length() - 5,
        }
        plain = K.compile_ptx(K.QUANT_SOURCE, f"sm_{arch}", quant).decode(errors = "replace")
        sink = K.compile_ptx(K.QUANT_SOURCE, f"sm_{arch}", dict(quant, SINKS = 1)).decode(
            errors = "replace"
        )
        assert ".entry unsloth_i8resid(" in sink and "unsloth_i8resid" not in plain
        assert all(f".entry {name}(" in plain for name in K.QUANT_KERNELS)


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_a_call_signature_is_planned_once(head_dim):
    """Two quantiser launches, the kernel and the residual launch, with their grids and arguments worked out on the first call of a
    signature only; packed views and separate tensors are two signatures with one answer, and the query rows are
    padded to the kernel's block, no further."""
    shape = (2, 333, 3, head_dim)
    config = _cuda_config(head_dim)
    q, k, v = _case(shape, _card_dtype(), seed = 5, strided = True)
    I.forget_plans()
    first = I.quantise(q, k, v, config)
    plan = first["plan"]
    assert len(I._plans()) == 1 and len(plan["stats_grids"]) == len(plan["kv_grids"]) == 1
    rows = 16 * config["nq"] * config["wpc"]
    assert first["pads"] == (-(-333 // rows) * rows, -(-333 // config["bn"]) * config["bn"])
    assert (
        first["qi"].shape == (6, first["pads"][0], head_dim) and first["f32"].dtype == torch.float32
    )
    out = I.attend(first)
    again = I.quantise(q, k, v, dict(config))
    assert again["plan"] is plan and len(I._plans()) == 1
    assert torch.equal(I.attend(again), out)
    apart = I.quantise(q.contiguous(), k.contiguous(), v.contiguous(), config)
    assert apart["plan"] is not plan and len(I._plans()) == 2
    assert torch.equal(I.attend(apart), out)
    assert torch.equal(I.int8_attention(q, k, v, scale = 0.05), I.int8_attention(q, k, v, scale = 0.05))
    assert len(I._plans()) == 3


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_long_overrides_do_not_change_the_output(head_dim):
    """The block shape moves no output bit (rows do not see each other, the tile is the same), so the kernel a call
    gets by its key count agrees with the other one; a call runs the long kernel exactly from long_keys keys on."""
    base = I.select_config(_cuda_config(head_dim), 0)
    entry = dict(base, long_keys = 300, long = {"wpc": 2 if base["wpc"] != 2 else 4})
    for keys in (299, 300, 700):
        q, k, v = _case((1, keys, 2, head_dim), _card_dtype(), seed = keys)
        q = q[:, :333].contiguous()
        p = I.quantise(q, k, v, entry)
        assert p["config"] == I.select_config(entry, keys)
        assert (p["config"]["wpc"] == base["wpc"]) is (keys < 300)
        got = I.attend(p)
        assert torch.equal(got, I.int8_attention(q, k, v, config = base))
        assert torch.equal(
            got, I.int8_attention(q, k, v, config = dict(base, wpc = entry["long"]["wpc"]))
        )


@needs_kernel
def test_inputs_off_the_16_byte_grid_are_copied():
    """A view whose rows do not start on a 16-byte boundary (here one element into a wider buffer) is read from
    an aligned copy: the same bits as the aligned call."""
    shape = (1, 200, 2, 128)
    q, k, v = _case(shape, _card_dtype(), seed = 9)
    want = I.int8_attention(q, k, v)
    wide = torch.zeros(1, 200, 2, 128 + 8, dtype = q.dtype, device = "cuda")
    wide[..., 1:129] = k
    shifted = wide[..., 1:129]
    assert shifted.data_ptr() % 16 != 0
    assert torch.equal(I.int8_attention(q, shifted, v), want)
    wide[..., 1:129] = q
    assert torch.equal(I.int8_attention(shifted, k, v), want)


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_a_long_call_splits_its_quantiser_grids(monkeypatch, head_dim):
    """A grid holds 65535 blocks along the axis the quantisers' jobs sit on; a call with more jobs than that runs
    them in several launches, each told its first job. With the limit turned down to three the bits stay the same."""
    q, k, v = _case((2, 700, 2, head_dim), _card_dtype(), seed = 11)
    want = I.int8_attention(q, k, v)
    monkeypatch.setattr(I, "GRID_LIMIT", 3)
    I.forget_plans()
    got = I.int8_attention(q, k, v)
    (plan,) = I._plans().values()
    assert len(plan["stats_grids"]) > 3 and len(plan["kv_grids"]) > 3
    assert all(grid[1] <= 3 for grid, _ in plan["stats_grids"] + plan["kv_grids"])
    assert torch.equal(got, want)


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_shipped_config_fits(head_dim):
    capability = torch.cuda.get_device_capability()
    for config, sinks in ((c, s) for c in _variants(_cuda_config(head_dim)) for s in (False, True)):
        fn = I.kernel_function("cuda", head_dim, config, sinks)
        assert fn.local_bytes <= 64, "the kernel spilled more than a few registers to the stack"
        cap = config.get("max_registers")
        assert cap is None or fn.registers <= cap
        # at least one block of this config is resident per SM (registers are granted in multiples of 8 per thread)
        assert 32 * config["wpc"] * ((fn.registers + 7) // 8 * 8) <= SM_REGISTERS
        assert I.shared_bytes(head_dim, config) <= SHARED_LIMIT.get(tuple(capability), 99 * 1024)


@needs_kernel
@pytest.mark.parametrize("shape", [(1, 2048, 8, 128), (1, 2048, 8, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_accuracy_against_fp32_attention(shape, dtype):
    q, k, v = I.probe_inputs("cuda", dtype, *shape[1:])
    ref = I.reference_attention(q, k, v)
    acc = I.accuracy(I.int8_attention(q, k, v), ref)
    assert acc["cos"] > 0.9995 and acc["rel_l1"] < 0.03, acc
    got = I.int8_attention(q, k, v, scale = 0.05)
    want = F.scaled_dot_product_attention(
        *(x.float().permute(0, 2, 1, 3) for x in (q, k, v)), scale = 0.05
    ).permute(0, 2, 1, 3)
    assert I.accuracy(got, want)["cos"] > 0.9995


@needs_kernel
def test_peaked_and_flat_rows_stay_accurate():
    # one key dominates half the rows by a wide margin (the online softmax rebases there); the other rows are flat
    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 1500, 4, 128)
    q = q.clone()
    q[:, ::2] = k[:, 700:701] * 6.0
    q[:, 1::2] *= 0.02
    acc = I.accuracy(I.int8_attention(q, k, v), I.reference_attention(q, k, v))
    assert acc["cos"] > 0.999, acc


@needs_kernel
def test_a_short_sequence_and_an_isolated_v_value_stay_accurate():
    """The V range comes from every key: 16 tokens with one key's V thirty times the rest, and 4096 tokens with one
    V value forty times its channel's range at a key the old every-8th-key statistics never saw."""

    def worst_row(got, ref):
        return float(
            F.cosine_similarity(got.float().flatten(0, 2), ref.flatten(0, 2), dim = -1).min()
        )

    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 16, 4, 128)
    v = v.clone()
    v[:, 3] *= 30.0
    out, ref = I.int8_attention(q, k, v), I.reference_attention(q, k, v)
    assert I.accuracy(out, ref)["rel_l1"] < 0.05 and worst_row(out, ref) > 0.99
    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 4096, 4, 128, seed = 5)
    v = v.clone()
    v[:, 13, :, 5] = v[:, :, :, 5].float().abs().amax(1) * 40.0
    out, ref = I.int8_attention(q, k, v), I.reference_attention(q, k, v)
    assert I.accuracy(out, ref)["rel_l1"] < 0.03 and worst_row(out, ref) > 0.99


@needs_kernel
@pytest.mark.parametrize("tokens", [65, 1000])
def test_padded_keys_get_no_weight_when_the_score_scale_underflows(tokens):
    """q = 0 and k = 0: both scales sit on their floor and their product underflows to 0. The padded keys of the last
    tile then scored 0 like every real key and took a share of the weight."""
    g = torch.Generator(device = "cuda").manual_seed(tokens)
    shape = (1, tokens, 2, 128)
    q = torch.zeros(shape, device = "cuda", dtype = torch.bfloat16)
    v = torch.exp(2.0 * torch.randn(shape, generator = g, device = "cuda")).to(torch.bfloat16)
    p = I.quantise(q, q.clone(), v, _cuda_config(128))
    out = I.attend(p).float()
    want = emulate(p)
    rel = float((out - want).abs().mean() / want.abs().mean())
    assert rel < 4e-3, rel
    acc = I.accuracy(out, v.float().mean(1, keepdim = True).expand_as(out))
    assert acc["cos"] > 0.999 and acc["rel_l1"] < 0.06, acc


@needs_kernel
@pytest.mark.parametrize("flush_tail", [0, 1])
def test_the_output_does_not_drift_with_the_key_count(flush_tail):
    """fp32 in and out, 65536 keys: the kernel against the plain-torch model of itself. With the rounding constant
    left in the output accumulators until the end, the gap grew with the key count (1.2e-3 here). Both places the
    constant can come off (the rebase branch, or a block of its own after P.V) are held to it."""
    q, k, v = I.probe_inputs("cuda", torch.float32, 65536, 2, 128, seed = 7)
    q = q[:, :256].contiguous()
    config = dict(_cuda_config(128), flush_tail = flush_tail)
    p = I.quantise(q, k, v, config)
    out = I.attend(p)
    want = emulate(p)
    rel = float((out - want).abs().mean() / want.abs().mean())
    assert rel < 3e-4, rel


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_sm75_instructions_give_the_same_bits(head_dim, dtype):
    """sm75 variants match other cards bit for bit; on a T4 they are compared to the shipped config."""
    q, k, v = I.probe_inputs("cuda", dtype, 2500, 3, head_dim, seed = 5)
    config = {k_: v_ for k_, v_ in _cuda_config(head_dim).items() if k_ != "max_registers"}
    ref = I.int8_attention(q, k, v)
    capability = torch.cuda.get_device_capability()
    variants = [
        {"sm75": 1, "stages": 1, "qreg": 2},
        {"sm75": 1, "stages": 1, "qreg": 2, "flush_tail": 1},
    ]
    if capability >= (8, 0):
        variants.append({"sm75": 0, "stages": 1, "qreg": 2})
    for variant in variants:
        candidate = {**config, "wpc": 4, "nq": 1, **variant}
        ptx = K.compile_ptx(
            K.KERNEL_SOURCE,
            f"sm_{capability[0]}{capability[1]}",
            I.kernel_defines(head_dim, candidate),
        ).decode(errors = "replace")
        assert ("m8n8k16" in ptx) == bool(variant["sm75"] or capability < (8, 0)), variant
        assert I.shared_bytes(head_dim, candidate) < I.shared_bytes(
            head_dim, dict(candidate, qreg = 1)
        )
        assert torch.equal(I.int8_attention(q, k, v, config = candidate), ref), variant


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_the_flush_placement_does_not_change_the_output(head_dim):
    """Both placements remove the constant at the same point of the accumulation (before tile 3, 7, ... here), so
    the cards that use one or the other give the same bits."""
    q, k, v = I.probe_inputs("cuda", torch.float32, 4500, 2, head_dim, seed = 9)
    config = _cuda_config(head_dim)
    rebase, tail = (
        I.int8_attention(q, k, v, config = dict(config, flush = 4, flush_tail = place))
        for place in (0, 1)
    )
    assert torch.equal(rebase, tail)
    assert torch.equal(
        I.int8_attention(q, k, v, config = dict(config, flush_tail = 1)), I.int8_attention(q, k, v)
    )


def _plain(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in ORDERINGS}


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
@pytest.mark.parametrize(
    "case",
    [
        (torch.bfloat16, 1039, 1.0, 0.0),
        (torch.float16, 2500, 1.0, 0.0),
        (torch.float32, 16 * 128 * 2 + 5, 1.0, 0.0),  # past two removals of the rounding constant
        (torch.bfloat16, 3000, 2.0, 6.0),  # late keys far above the early ones: the rebase branch
        (torch.bfloat16, 77, 3.0, 0.0),  # one padded tile
    ],
    ids = ["bf16", "fp16", "flush", "rebase", "short"],
)
def test_the_orderings_do_not_change_the_output(head_dim, case):
    """The order the products and the tile copies are issued in gives the bits of the plain order: integer sums
    are exact and every float accumulator sees the same operations. So a card tuned with them still agrees with
    the others bit for bit."""
    dtype, tokens, gain, spike = case
    generator = torch.Generator().manual_seed(tokens)
    q, k, v = (torch.randn(2, tokens, 3, head_dim, generator = generator) for _ in range(3))
    q, k = q * gain, k * gain
    if spike:
        k[:, tokens // 2 :] *= spike
        k[:, -3:] *= 4.0
    q = q[:, : min(tokens, 700)].contiguous()
    q, k, v = (x.to("cuda", dtype) for x in (q, k, v))
    plain = _plain(_cuda_config(head_dim))
    want = I.int8_attention(q, k, v, config = plain)
    assert torch.isfinite(want.float()).all()
    variants = [{"qkw": 1}, {"pvw": 2}, {"pvw": 8}, {"qkw": 1, "pvw": 16 if head_dim == 128 else 4}]
    if plain["stages"] == 2:
        variants += [{"qkw": 1, "spread": 1}, {"qkw": 1, "pvw": 8, "spread": 5}]
    for variant in variants:
        assert torch.equal(I.int8_attention(q, k, v, config = dict(plain, **variant)), want), variant
    assert torch.equal(I.int8_attention(q, k, v), want)


@needs_kernel
def test_batch_offsets_past_2_31_elements():
    """Three batches whose stride is 2^30 elements: the third starts 2^31 elements into the buffer, past int32."""
    stride, S, H, D = 1 << 30, 200, 2, 64
    need = (2 * stride + 3 * S * H * D) * 2
    free, _ = torch.cuda.mem_get_info()
    if free < need + (1 << 30):
        pytest.skip("needs 5.5 GB of free GPU memory")
    buf = torch.empty(2 * stride + 3 * S * H * D, device = "cuda", dtype = torch.bfloat16)
    q, k, v = (buf.as_strided((3, S, H, D), (stride, 3 * H * D, D, 1), i * H * D) for i in range(3))
    dense = [I.probe_inputs("cuda", torch.bfloat16, S, H, D, seed = 11 + b) for b in range(3)]
    for b in range(3):
        for view, value in zip((q, k, v), dense[b]):
            view[b] = value[0]
    out = I.int8_attention(q, k, v)
    for b in range(3):
        ref = I.reference_attention(*dense[b])
        assert I.accuracy(out[b : b + 1], ref)["cos"] > 0.999, b
    del buf, out
    torch.cuda.empty_cache()


@needs_kernel
def test_op_runs_the_kernel_only_where_it_was_verified(monkeypatch):
    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 512, 2, 128)
    I.int8_attention_op()
    monkeypatch.setattr(I, "_VERIFIED", set())
    monkeypatch.setattr(I, "_VERDICTS", {})
    before = dict(I.STATS)
    assert torch.equal(
        torch.ops.unsloth_studio.int8_attention_nhd(q, k, v), I.native_attention(q, k, v)
    )
    assert I.STATS["native"] == before["native"] + 1
    found = I.verdict("cuda", torch.bfloat16, 2, 128, timed = False)
    assert found["ok"] and found["cos"] >= I.MIN_COSINE
    got = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
    assert I.STATS["int8"] == before["int8"] + 1
    assert I.accuracy(got, I.reference_attention(q, k, v))["cos"] > 0.999
    half = tuple(x.half() for x in (q, k, v))
    native = I.STATS["native"]
    assert torch.equal(
        torch.ops.unsloth_studio.int8_attention_nhd(*half), I.native_attention(*half)
    )
    assert I.STATS["native"] == native + 1
    assert I.verdict("cuda", torch.float16, 2, 128, timed = False)["ok"]
    torch.ops.unsloth_studio.int8_attention_nhd(*half)
    assert I.STATS["int8"] == before["int8"] + 2
    monkeypatch.setattr(I, "_BROKEN", set())
    monkeypatch.setattr(
        I, "_launch_attention", lambda *a: (_ for _ in ()).throw(RuntimeError("launch failed"))
    )
    assert torch.equal(
        torch.ops.unsloth_studio.int8_attention_nhd(q, k, v), I.native_attention(q, k, v)
    )
    assert I.unsupported_reason(q, k, v) == "kernel failed on this card"


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
@pytest.mark.parametrize("gain", [8.0, 16.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_scores_stay_finite_and_accurate(head_dim, gain, dtype):
    """q and k as large as a late DiT block's: the score scale times 2^23 is past 2^16, where a tile's largest
    probability used to round to 256 and wrap to 0 (rows of garbage and non-finite values)."""
    g = torch.Generator().manual_seed(int(gain) + head_dim)
    shape = (1, 1100, 8, head_dim)
    q = (torch.randn(shape, generator = g) * gain).to("cuda", dtype)
    k = (torch.randn(shape, generator = g) * gain).to("cuda", dtype)
    v = torch.randn(shape, generator = g).to("cuda", dtype)
    p = I.quantise(q, k, v, _cuda_config(head_dim))
    assert float((12582912.0 * p["qs"][:, :1100] * p["ks"].amax((1, 2))[:, None]).max()) > 2**16
    out = I.int8_attention(q, k, v)
    assert bool(torch.isfinite(out).all())
    assert I.accuracy(out, I.reference_attention(q, k, v))["cos"] >= 0.98


@needs_kernel
def test_timed_verdict_reports_both_sides():
    found = I.verdict("cuda", torch.bfloat16, 8, 128)
    assert found["ok"] and set(found["ms"]) == set(I.RACE_TOKENS)
    for ms in found["ms"].values():
        assert ms["int8"] > 0 and ms["incumbent"] > 0
    assert found["min_tokens"] in (None, 0, sum(I.RACE_TOKENS) // 2)


@needs_kernel
def test_cuda_graph_replay_matches_eager():
    I.verdict("cuda", torch.bfloat16, 4, 128, timed = False)
    I.int8_attention_op()
    op = torch.ops.unsloth_studio.int8_attention_nhd
    q, k, v = (x.clone() for x in I.probe_inputs("cuda", torch.bfloat16, 1100, 4, 128))
    for _ in range(3):  # allocator and JIT warm, as before any capture
        op(q, k, v)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(q, k, v)
    for seed in (5, 6):
        fresh = I.probe_inputs("cuda", torch.bfloat16, 1100, 4, 128, seed = seed)
        for dst, src in zip((q, k, v), fresh):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op(*fresh))


OTHER = 1  # the second card of the two-card tests

needs_two_cards = pytest.mark.skipif(
    _cuda_config() is None
    or torch.cuda.device_count() <= OTHER
    or I.kernel_config(torch.cuda.get_device_capability(OTHER), 128) is None,
    reason = "needs two CUDA cards with an INT8 kernel config",
)


@needs_two_cards
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_each_card_loads_its_own_kernels(head_dim):
    """A loaded module belongs to the context of one card: each card compiles and loads its own kernels and keeps
    its own launch plans, and a call runs on the card of its operands whatever the current device is."""
    dtype = _card_dtype()
    there = torch.device("cuda", OTHER)
    calls = {
        index: I.probe_inputs(torch.device("cuda", index), dtype, 700, 3, head_dim)
        for index in (0, OTHER)
    }
    I.forget_plans()
    assert torch.cuda.current_device() == 0
    out = {index: I.int8_attention(*calls[index]) for index in (0, OTHER)}
    assert out[OTHER].device == there and torch.cuda.current_device() == 0
    with torch.cuda.device(OTHER):
        swapped = I.int8_attention(*calls[0])
        assert torch.cuda.current_device() == OTHER
    assert swapped.device.index == 0 and torch.equal(swapped, out[0])
    assert sorted(plan["index"] for plan in I._plans().values()) == [0, OTHER]
    loaded = {}
    for index in (0, OTHER):
        config = I.kernel_config(torch.cuda.get_device_capability(index), head_dim)
        attn, quant, _ = I._functions(index, head_dim, config)
        loaded[index] = [attn, *quant.values()]
        assert I.accuracy(out[index], I.reference_attention(*calls[index]))["cos"] >= I.MIN_COSINE
    assert len(loaded[0]) == 3 and all(a is not b for a, b in zip(loaded[0], loaded[OTHER]))
    if torch.cuda.get_device_capability(OTHER) == torch.cuda.get_device_capability(0):
        assert torch.equal(out[OTHER].cpu(), out[0].cpu())


@needs_two_cards
def test_the_card_check_counts_for_its_card_only(monkeypatch):
    """A check passed on one card says nothing about the other: the verdicts and the verified set are per device,
    and the op keeps native attention on a card until that card passed its own check."""
    monkeypatch.setattr(I, "_VERDICTS", {})
    monkeypatch.setattr(I, "_VERIFIED", set())
    dtype = _card_dtype()
    there = torch.device("cuda", OTHER)
    here_call = I.probe_inputs("cuda:0", dtype, 500, 2, 128)
    there_call = I.probe_inputs(there, dtype, 500, 2, 128)
    first = I.verdict("cuda:0", dtype, 2, 128, timed = False)
    assert first["ok"] and I._verified(*here_call) and not I._verified(*there_call)
    assert {key[0] for key in I._VERDICTS} == {0} and {key[0] for key in I._VERIFIED} == {0}
    native, ran = I.STATS["native"], I.STATS["int8"]
    assert I.int8_attention_impl(*there_call).device == there
    assert (I.STATS["native"], I.STATS["int8"]) == (native + 1, ran)
    second = I.verdict(there, dtype, 2, 128, timed = False)
    assert second["ok"] and second is not first and I._verified(*there_call)
    assert {key[0] for key in I._VERDICTS} == {0, OTHER}
    assert {key[0] for key in I._VERIFIED} == {0, OTHER}
    assert I.int8_attention_impl(*there_call).device == there
    assert (I.STATS["native"], I.STATS["int8"]) == (native + 1, ran + 1)


@needs_two_cards
def test_a_call_on_another_card_uses_that_cards_stream(monkeypatch):
    """Every launch of a call goes to the current stream of the card its operands are on, with that card's context
    current, also while another card is the current device."""
    q, k, v = I.probe_inputs(torch.device("cuda", OTHER), _card_dtype(), 700, 3, 128)
    seen = []
    launch = K.Function.launch

    def spy(self, grid, block, values, shared, stream):
        seen.append((self.name, stream, torch.cuda.current_device()))
        return launch(self, grid, block, values, shared, stream)

    monkeypatch.setattr(K.Function, "launch", spy)
    # a plan's first calls take the sink path, whose kernel adds the clipped keys back itself (RFOLD)
    names = {K.KERNEL_NAME, *K.QUANT_KERNELS} | (
        set() if I.resid_fold_enabled() else {"unsloth_i8resid"}
    )
    assert torch.cuda.current_device() == 0
    out = I.int8_attention(q, k, v)
    usual = torch.cuda.current_stream(OTHER).cuda_stream
    assert {name for name, _, _ in seen} == names
    assert all((stream, device) == (usual, OTHER) for _, stream, device in seen)
    side = torch.cuda.Stream(device = OTHER)
    assert side.cuda_stream != usual
    del seen[:]
    with (
        torch.cuda.stream(side),
        torch.cuda.device(0),
    ):  # card 0 is current, the other card's stream is `side`
        assert torch.cuda.current_device() == 0
        again = I.int8_attention(q, k, v)
    side.synchronize()
    assert {name for name, _, _ in seen} == names
    assert all((stream, device) == (side.cuda_stream, OTHER) for _, stream, device in seen)
    assert torch.equal(again, out)


@needs_two_cards
def test_a_cuda_graph_of_another_card_replays_the_call():
    """A graph captured on the other card holds the quantisers and the kernel: its replay follows new inputs. The
    capture names a stream of that card: torch's default capture stream is one per process, on the card of the first
    capture (after one on card 0 a capture here would be empty)."""
    there = torch.device("cuda", OTHER)
    dtype = _card_dtype()
    q, k, v = (x.clone() for x in I.probe_inputs(there, dtype, 1100, 4, 128))
    for _ in range(3):  # allocator and JIT warm, as before any capture
        I.int8_attention(q, k, v)
    with torch.cuda.device(OTHER):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream = torch.cuda.Stream(device = OTHER)):
            out = I.int8_attention(q, k, v)
        for seed in (5, 6):
            fresh = I.probe_inputs(there, dtype, 1100, 4, 128, seed = seed)
            for dst, src in zip((q, k, v), fresh):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, I.int8_attention(*fresh))


@needs_kernel
def test_torch_compile_keeps_the_op_opaque():
    dtype = _card_dtype()
    I.verdict("cuda", dtype, 4, 64, timed = False)
    I.int8_attention_op()

    def block(q, k, v):
        return torch.ops.unsloth_studio.int8_attention_nhd(q * 1.0, k, v).flatten(2, 3) + 1.0

    q, k, v = I.probe_inputs("cuda", dtype, 900, 4, 64)
    before = I.STATS["int8"]
    torch._dynamo.reset()
    assert torch._dynamo.explain(block)(q, k, v).graph_break_count == 0
    compiled = torch.compile(block, fullgraph = True)
    with torch.no_grad():
        assert torch.equal(compiled(q, k, v), block(q, k, v))
    assert I.STATS["int8"] >= before + 2
    torch._dynamo.reset()


@needs_kernel
def test_h3_strided_processor_runs_the_kernel_on_cuda(_active_backend):
    h3, A = _h3()
    dtype = _card_dtype()
    model = _tiny_h3(h3, head_dim = 64).to("cuda", dtype)
    model.set_attention_backend(_card_sdpa())
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model) == 3
    kw = {
        k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _h3_inputs(n_video = 256).items()
    }
    for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        kw[name] = kw[name].to(dtype)
    # the load hook with a target that names no dtype: it checks the card's own (fp16 on sm75)
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda", dtype = None)
    assert I.install_requested(types.SimpleNamespace(transformer = model), I.BACKEND, target) == 3
    before = I.STATS["int8"]
    with torch.no_grad():
        want, got = stock(**kw), model(**kw)
    assert I.STATS["int8"] >= before + 2
    for a, b in zip(want, got):
        assert F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim = 0) > 0.98
    assert I.remove_int8_attention(model) == 3
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(want, model(**kw)))


@needs_kernel
def test_checkpoint_race_runs_the_kernel_on_a_resident_block(_active_backend):
    h3, A = _h3()
    H = _h3_image()
    dtype = _card_dtype()
    model = _tiny_h3(h3, head_dim = 64).to("cuda", dtype)
    model.set_attention_backend(_card_sdpa())
    assert A.install_strided_attention(model) == 3
    mods = I.int8_processors(model)
    blocks = [m for m in mods if any(m is b.attn for b in model.transformer_blocks)]
    saved = {m: m.processor._attention_backend for m in mods}
    assert len(blocks) == 2
    assert (
        I.install_int8_attention(model, device = "cuda", dtype = dtype, modules = blocks, force = True) == 2
    )
    I.remove_int8_attention(model)
    before = I.STATS["int8"]
    found = H.race_checkpoint_attention(model, saved, dtype, kernel = True)
    calls = 2 * (2 + H._RACE_ROUNDS)
    assert I.STATS["int8"] == before + calls
    assert sorted(found["kernel"]) == [1024, 4096]
    assert all(len(v) == 3 and min(v) > 0 for v in found["kernel"].values())
    assert found["ms"] == found["kernel"][4096][:2]
    assert all(len(v) == 3 and min(v) > 0 for v in found["paired"].values())
    proc = blocks[0].processor
    assert not proc._unsloth_int8 and proc._attention_backend == saved[blocks[0]]
    plain = H.race_checkpoint_attention(model, saved, dtype)
    assert "kernel" not in plain and len(plain["ms"]) == 2 and I.STATS["int8"] == before + calls
    import logging

    H.KERNEL_AUTO = True
    try:
        routed, race = H._race_int8_kernel(
            model, blocks, saved, "cuda", dtype, None, False, logging.getLogger(__name__)
        )
    finally:
        H.KERNEL_AUTO = False
    assert race is not None and "keep" in race
    record = getattr(model, H.ATTENTION_RACE_ATTR)
    assert record["timed"] == "block" and record["blocks"] == routed and routed in (0, 2)
    assert I.int8_attention_count(model) == routed


def test_the_eager_call_skips_the_op_and_compile_traces_it(monkeypatch):
    """``int8_attention_call`` runs the op's body directly in eager, and the custom op under torch.compile
    (fullgraph: no break)."""
    I.int8_attention_op()
    seen = []
    monkeypatch.setattr(I, "int8_attention_impl", lambda q, k, v: seen.append("impl") or v * 2)
    q = torch.randn(1, 8, 2, 64)
    assert torch.equal(I.int8_attention_call(q, q, q), q * 2) and seen == ["impl"]
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    fn = torch.compile(
        lambda q: I.int8_attention_call(q, q, q) + 1, backend = backend, fullgraph = True
    )
    torch._dynamo.reset()
    assert torch.equal(fn(q), q * 2 + 1)
    targets = {str(node.target) for node in graphs[0].graph.nodes}
    assert any("int8_attention_nhd" in t for t in targets), targets
    # a tensor subclass always takes the op
    calls = []
    monkeypatch.setattr(
        I, "int8_attention_impl", lambda q, k, v: calls.append(type(q)) or v.clone()
    )
    p = torch.nn.Parameter(q.clone(), requires_grad = False)
    I.int8_attention_call(p, q, q)
    assert len(calls) == 1


def test_tracers_that_look_like_plain_tensors_still_see_the_op(monkeypatch):
    """A dispatch mode (make_fx, FlopCounterMode) records the op; raw launches would bypass it."""
    from torch.utils._python_dispatch import TorchDispatchMode

    I.int8_attention_op()
    monkeypatch.setattr(I, "int8_attention_impl", lambda q, k, v: v.clone())

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))

    q = torch.randn(1, 8, 2, 64)
    with Record() as mode:
        I.int8_attention_call(q, q, q)
    assert any("int8_attention_nhd" in op for op in mode.ops), mode.ops
    assert not I._tracing()


def test_torchscript_tracing_records_the_op(monkeypatch):
    """torch.jit.trace sees plain tensors; tracing the op body would record its allocations but not its launches."""
    I.int8_attention_op()
    monkeypatch.setattr(I, "int8_attention_impl", lambda q, k, v: v.clone())

    class Attend(torch.nn.Module):
        def forward(self, q, k, v):
            return I.int8_attention_call(q, k, v)

    q = torch.randn(1, 8, 2, 64)
    traced = torch.jit.trace(Attend(), (q, q, q), check_trace = False)
    assert "int8_attention_nhd" in str(traced.graph), traced.graph
    assert not I._tracing()
