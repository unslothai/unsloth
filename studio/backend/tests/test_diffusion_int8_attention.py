# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's INT8 attention kernel (``unsloth_int8``): switches, per-card configs, the processor route and its
fallbacks on CPU; the kernel against a plain-torch model of itself, spills, CUDA graphs and torch.compile on CUDA."""

from __future__ import annotations

import contextlib
import copy
import math
import types

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as K

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
    """What Studio's pipelines run in on this card: sm75 has no bf16 (Inductor and the fused SDPA kernels refuse
    it), so fp16 there."""
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
    """The quantisers' rotation on the last dim: fixed signs, then a Hadamard transform over every ROT_GROUP
    channels (the head dim where that is smaller)."""
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
    if kc is None:
        kstep, vstep, _, _ = I.stats_plan(Sk)
        kc = kh[:, ::kstep].mean(1)
        lo, hi = vh[:, ::vstep].amin(1), vh[:, ::vstep].amax(1)
        vc, vg = 0.5 * (hi + lo), (0.5 * (hi - lo)).clamp_min(1e-12)
    kc, vc, vg = kc.double(), vc.double(), vg.double()
    qr = rotate_ref(heads_first(q))
    qa = qr.abs().amax(-1).clamp_min(1e-20)
    n_kpad = (Sk + bn - 1) // bn * bn
    kr = F.pad(rotate_ref(kh - kc[:, None, :]), (0, 0, 0, n_kpad - Sk))
    # one K scale per run of 32 keys and accumulator lane group ((key % 8) // 2)
    ka = kr.abs().amax(-1).view(BH, n_kpad // 32, 4, 4, 2).amax((2, 4)).clamp_min(1e-20)
    per_key = ka[:, :, None, :, None].expand(-1, -1, 4, -1, 2).reshape(BH, n_kpad)
    vr = F.pad((vh - vc[:, None, :]) / vg[:, None, :], (0, 0, 0, n_kpad - Sk)).view(
        BH, n_kpad // bn, bn, D
    )
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
        "shape": (B, Sq, Sk, H, D),
        "dtype": v.dtype,
        "config": config,
    }


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


def test_every_config_fits_its_card():
    assert set(I.KERNEL_CONFIGS) == set(SHARED_LIMIT)
    for capability, table in I.KERNEL_CONFIGS.items():
        assert set(table) == set(I.HEAD_DIMS)
        for head_dim, config in table.items():
            assert config["bn"] in (64, 128) and config["nq"] >= 1 and config["stages"] in (1, 2)
            assert I.shared_bytes(head_dim, config) <= SHARED_LIMIT[capability]
            cap = config.get("max_registers")
            if cap is not None:
                assert 32 * config["wpc"] * cap <= SM_REGISTERS
            assert (
                32 * config["wpc"] <= 1024
            )  # sm75 holds 32 warps per SM, and a block is at most 1024 threads
            assert I.kernel_config(capability, head_dim) == config
    assert I.kernel_config((9, 0), 128) is None and I.kernel_config((12, 0), 96) is None
    assert I.kernel_config((7, 0), 128) is None and I.kernel_config((6, 1), 128) is None
    assert I.kernel_config(None, 128) is None


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
    for head_dim, config in I.KERNEL_CONFIGS[capability].items():
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
                "BN": config["bn"],
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


def test_statistics_cover_every_key():
    for n in (1, 15, 16, 80, 1039, 4112, 10011, 75600, 300000):
        kstep, vstep, chunks, rows = I.stats_plan(n)
        assert vstep == 1 and rows % vstep == 0 and chunks <= I.V_RANGE_CHUNKS
        assert (chunks - 1) * rows < n <= chunks * rows
        assert 1 <= kstep <= I.K_MEAN_STEP and -(-n // kstep) >= min(n, I.K_MEAN_KEYS)


def test_v_range_over_every_key_keeps_a_short_sequence_accurate(monkeypatch):
    """16 tokens, one of them with V thirty times the rest (a text refiner call). With the range of V taken over
    every 8th key (keys 0 and 8 here) the gain of most channels came out far too small and the error is more than
    four times larger; over every key it is as accurate as a long call."""
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
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

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


def test_checkpoint_race_times_the_kernel_on_a_block(monkeypatch, _active_backend):
    import logging
    import sys

    h3, A = _h3()
    H = _h3_image()
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


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_shipped_config_fits(head_dim):
    capability = torch.cuda.get_device_capability()
    config = _cuda_config(head_dim)
    fn = I.kernel_function("cuda", head_dim, config)
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
    """sm75 computes each 16x8x32 product as four 8x8x16 ones on the same operand registers, copies tiles with plain
    loads and reads the Q operands from global memory: the same integers, so the same output bits as every other
    card (the sm75 code runs on later cards too, which is how this is checked off a T4). A T4 cannot run the sm80
    instructions: there the sm75 variants are compared with the shipped config only."""
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
    monkeypatch.setattr(I, "attend", lambda p: (_ for _ in ()).throw(RuntimeError("launch failed")))
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

    routed, race = H._race_int8_kernel(
        model, blocks, saved, "cuda", dtype, None, False, logging.getLogger(__name__)
    )
    assert race is not None and "keep" in race
    record = getattr(model, H.ATTENTION_RACE_ATTR)
    assert record["timed"] == "block" and record["blocks"] == routed and routed in (0, 2)
    assert I.int8_attention_count(model) == routed
