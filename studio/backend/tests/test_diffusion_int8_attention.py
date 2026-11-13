# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's INT8 attention kernel (``unsloth_int8``): switches, per-card configs, the processor route and its
fallbacks on CPU; the kernel against a plain-torch model of itself, spills, CUDA graphs and torch.compile on CUDA."""

from __future__ import annotations

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
SHARED_LIMIT = {(8, 0): 163 * 1024, (8, 6): 99 * 1024, (8, 9): 99 * 1024, (12, 0): 99 * 1024}
SM_REGISTERS = 65536


def _cuda_config(head_dim: int = 128):
    if not torch.cuda.is_available():
        return None
    return I.kernel_config(torch.cuda.get_device_capability(), head_dim)


needs_kernel = pytest.mark.skipif(
    _cuda_config() is None, reason = "needs a CUDA card with an INT8 kernel config"
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    stats = dict(I.STATS)
    yield
    I.STATS.update(stats)


# ---------------------------------------------------------------- a plain-torch model of the quantisers and kernel


def rotate_ref(x):
    """The quantisers' rotation on the last dim: fixed signs, then a 16-point Hadamard transform per 16 channels."""
    D = x.shape[-1]
    cols = torch.arange(D, device = x.device)
    flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
    x = x * (1.0 - 2.0 * flip.to(x.dtype))
    for _ in range(4):
        y = x.reshape(*x.shape[:-1], D // 2, 2)
        z = torch.stack((y[..., 0] + y[..., 1], y[..., 0] - y[..., 1]), dim = -1)
        x = z.reshape(*x.shape[:-1], D // 16, 8, 2).transpose(-1, -2).reshape(*x.shape[:-1], D)
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
    """The quantisers' math in fp64. The per-channel K mean and V centre / gain are taken over every STATS_STEP-th
    key, or passed in (what the kernels computed) to compare the rounding alone."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bn = config["bn"]
    BH = B * H
    mul = 1.0 / math.sqrt(D) * I.LOG2E / 16.0

    def heads_first(t):
        return t.permute(0, 2, 1, 3).reshape(BH, t.shape[1], D).double()

    kh, vh = heads_first(k), heads_first(v)
    if kc is None:
        kc = kh[:, :: I.STATS_STEP].mean(1)
        lo, hi = vh[:, :: I.STATS_STEP].amin(1), vh[:, :: I.STATS_STEP].amax(1)
        vc, vg = 0.5 * (hi + lo), (0.5 * (hi - lo)).clamp_min(1e-12)
    kc, vc, vg = kc.double(), vc.double(), vg.double()
    qr = rotate_ref(heads_first(q))
    qa = qr.abs().amax(-1).clamp_min(1e-20)
    n_kpad = (Sk + bn - 1) // bn * bn
    kr = F.pad(rotate_ref(kh - kc[:, None, :]), (0, 0, 0, n_kpad - Sk))
    ka = kr.abs().amax(-1).view(BH, n_kpad // bn, bn // 8, 4, 2).amax((2, 4)).clamp_min(1e-20)
    per_key = ka[:, :, None, :, None].expand(-1, -1, bn // 8, -1, 2).reshape(BH, n_kpad)
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
    k_scale = p["ks"].reshape(BH, tiles * 4)[:, (keys // bn) * 4 + ((keys % 8) >> 1)]
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


# ---------------------------------------------------------------- switches, configs, aliases (CPU)


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
    # the same native pick an active speed profile makes, whatever the speed setting
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
                # a capped config is there to fit this many blocks: the cap must allow at least one
                assert 32 * config["wpc"] * cap <= SM_REGISTERS
            assert I.kernel_config(capability, head_dim) == config
    assert I.kernel_config((9, 0), 128) is None and I.kernel_config((12, 0), 96) is None
    assert I.kernel_config(None, 128) is None


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


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
def test_kernel_compiles_for_every_config():
    for capability, table in I.KERNEL_CONFIGS.items():
        for head_dim, config in table.items():
            ptx = K.compile_ptx(
                K.KERNEL_SOURCE,
                f"sm_{capability[0]}{capability[1]}",
                I.kernel_defines(head_dim, config),
            ).decode(errors = "replace")
            # Q.K^T: (head_dim / 32) k steps per 8-key block; P.V: (bn / 32) k steps per 8-channel block
            products = config["nq"] * (
                (head_dim // 32) * (config["bn"] // 8) + (config["bn"] // 32) * (head_dim // 8)
            )
            assert ptx.count("mma.sync.aligned.m16n8k32") == products
            # the accumulators are read as floats through their bit pattern: no signed int -> float conversion
            assert "cvt.rn.f32.s32" not in ptx


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


def test_install_off_cuda_and_kill_switch(monkeypatch):
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cpu", dtype = torch.bfloat16) == 0
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16, force = True) == 0
    assert I.int8_attention_count(dit) == 0


def test_install_follows_the_race(monkeypatch):
    # head dim 128 wins from 2560 tokens, head dim 64 loses the race, head dim 96 has no kernel
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
    # the env switch forces the same way
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
    # a card without a config, a dtype the kernel does not write
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
    "shape,bn", [((1, 300, 2, 64), 64), ((1, 200, 2, 128), 64), ((1, 260, 1, 128), 128)]
)
def test_the_method_stays_close_to_fp32_attention(shape, bn):
    # no GPU involved: the quantisers' and the kernel's math in plain torch against fp32 attention
    q, k, v = I.probe_inputs("cpu", torch.float32, *shape[1:])
    p = reference_quantise(q, k, v, {"bn": bn, "wpc": 1, "nq": 1, "stages": 1})
    for name in ("qi", "ki", "vt"):
        assert p[name].abs().max() <= 127 and torch.equal(p[name], p[name].round())
    # the rotation is orthogonal up to its factor 4, so Q.K^T is unchanged
    eye = torch.eye(shape[-1], dtype = torch.float64)
    rot = rotate_ref(eye)
    assert torch.equal(rot @ rot.T, 16.0 * eye)
    acc = I.accuracy(emulate(p), I.reference_attention(q, k, v))
    assert acc["cos"] > 0.9995 and acc["rel_l1"] < 0.03, acc


# ---------------------------------------------------------------- the H3 strided processor route (tiny real model)


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
    h3, A = _h3()
    model = _tiny_h3(h3)
    model.set_attention_backend("_native_math")
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
        # every unmasked call of a flagged processor went through the op, NHD in and out
        assert calls and all(
            len(shape) == 4 and shape[2] == 2 and shape[3] == 32 for shape in calls
        )
        assert all(torch.allclose(a, b, atol = 1e-5, rtol = 1e-5) for a, b in zip(want, got))
        # below the raced sequence length the native kernel stays
        calls.clear()
        for proc in procs:
            proc._unsloth_int8_min_tokens = 10**6
        assert all(torch.equal(a, b) for a, b in zip(want, model(**_h3_inputs()))) and not calls
        # a processor whose backend is not a native kernel never takes the route
        for proc in procs:
            proc._unsloth_int8_min_tokens = 0
            proc._attention_backend = None
        model(**_h3_inputs())
        assert not calls


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
        H, "race_checkpoint_attention", lambda *a, kernel = False: race.get("found") if kernel else 1 / 0
    )
    log = logging.getLogger(__name__)

    def run(sage = False, forced = False, mods = blocks):
        installs.clear()
        for m in blocks:
            m.processor._unsloth_int8 = False
            m.processor._unsloth_int8_min_tokens = 0
        return H._race_int8_kernel(model, mods, saved, "cuda", torch.bfloat16, sage, forced, log)

    def routed():
        return [(m.processor._unsloth_int8, m.processor._unsloth_int8_min_tokens) for m in blocks]

    # [native, engaged, kernel] ms per image row count. Faster than native on both canvases: every length
    race["found"] = {"native": "_native_math", "kernel": {4096: [10.0, 10.0, 9.0], 1024: [3.0, 3.0, 2.5]}}
    assert run() == 2 and routed() == [(True, 0)] * 2
    assert len(installs) == 1 and installs[0]["force"] and installs[0]["modules"] == blocks
    record = getattr(model, H.ATTENTION_RACE_ATTR)
    assert record["timed"] == "block" and record["blocks"] == 2 and record["min_tokens"] == 0
    assert record["ms"]["4096"] == [10.0, 10.0, 9.0] and record["seconds"] >= 0
    # only the large canvas: short sequences keep the native backend
    race["found"]["kernel"][1024] = [3.0, 3.0, 2.99]
    assert run() == 2 and routed() == [(True, H._KERNEL_LARGE_ROWS)] * 2
    # inside the 2% margin: the incumbent stays
    race["found"]["kernel"][4096] = [10.0, 10.0, 9.9]
    assert run() == 0 and routed() == [(False, 0)] * 2
    # SageAttention kept by its own race is the one to beat
    race["found"]["kernel"] = {4096: [10.0, 8.0, 9.0], 1024: [3.0, 2.0, 2.5]}
    assert run(sage = True) == 0 and routed() == [(False, 0)] * 2
    assert run(sage = False) == 2
    race["found"]["kernel"] = {4096: [10.0, 8.0, 7.0], 1024: [3.0, 2.0, 2.5]}
    assert run(sage = True) == 2 and routed() == [(True, H._KERNEL_LARGE_ROWS)] * 2
    # no block on the card: the attention-only race decides, against the call the blocks make
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
    # a broken race leaves the blocks as they were
    race["found"] = {"native": "_native_math", "kernel": {4096: [1.0]}}
    assert run() == 0 and routed() == [(False, 0)] * 2
    assert "error" in getattr(model, H.ATTENTION_RACE_ATTR)
    # SageAttention pinned by its switch, nothing to race, the kill switch
    assert run(forced = True) == 0 and not installs
    assert run(mods = []) == 0 and not installs
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert run() == 0 and not installs
    # the kernel's own switch forces it past the race, SageAttention pinned or not
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert run(forced = True) == 2 and len(installs) == 1 and routed() == [(True, 0)] * 2


# ---------------------------------------------------------------- the kernel itself (CUDA)


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
        # the layout the strided processors pass: a [B, S, H * D] projection unflattened, here sliced out of QKV
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
    # the sub-sampled statistics themselves
    own = reference_quantise(q, k, v, config)
    for name in ("kc", "vc", "vg"):
        assert torch.allclose(p[name], own[name].to(p[name].device), rtol = 1e-4, atol = 1e-4), name
    out = I.attend(p)
    assert out.dtype == dtype and out.shape == q.shape and torch.isfinite(out.float()).all()
    want = emulate(p)
    rel = ((out.float() - want).abs().mean() / want.abs().mean()).item()
    # the gap is the output dtype's rounding step plus fp32 accumulation order
    assert rel < (2e-3 if dtype == torch.float32 else 4e-3), rel
    assert torch.equal(I.int8_attention(q, k, v, config = config), out)


@needs_kernel
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_shipped_config_is_spill_free_and_fits(head_dim):
    capability = torch.cuda.get_device_capability()
    config = _cuda_config(head_dim)
    fn = I.kernel_function("cuda", head_dim, config)
    assert fn.local_bytes == 0, "the kernel spilled registers to the stack"
    cap = config.get("max_registers")
    assert cap is None or fn.registers <= cap
    # at least one block of this config is resident per SM (registers are granted in multiples of 8 per thread)
    assert 32 * config["wpc"] * ((fn.registers + 7) // 8 * 8) <= SM_REGISTERS
    assert I.shared_bytes(head_dim, config) <= SHARED_LIMIT[tuple(capability)]


@needs_kernel
@pytest.mark.parametrize("shape", [(1, 2048, 8, 128), (1, 2048, 8, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_accuracy_against_fp32_attention(shape, dtype):
    q, k, v = I.probe_inputs("cuda", dtype, *shape[1:])
    ref = I.reference_attention(q, k, v)
    acc = I.accuracy(I.int8_attention(q, k, v), ref)
    assert acc["cos"] > 0.9995 and acc["rel_l1"] < 0.03, acc
    # a different softmax scale than 1 / sqrt(head dim)
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
    # a failing launch falls back to native attention and stays there
    monkeypatch.setattr(I, "_BROKEN", set())
    monkeypatch.setattr(I, "attend", lambda p: (_ for _ in ()).throw(RuntimeError("launch failed")))
    assert torch.equal(
        torch.ops.unsloth_studio.int8_attention_nhd(q, k, v), I.native_attention(q, k, v)
    )
    assert I.unsupported_reason(q, k, v) == "kernel failed on this card"


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
    I.verdict("cuda", torch.bfloat16, 4, 64, timed = False)
    I.int8_attention_op()

    def block(q, k, v):
        return torch.ops.unsloth_studio.int8_attention_nhd(q * 1.0, k, v).flatten(2, 3) + 1.0

    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 900, 4, 64)
    torch._dynamo.reset()
    assert torch._dynamo.explain(block)(q, k, v).graph_break_count == 0
    compiled = torch.compile(block, fullgraph = True)
    with torch.no_grad():
        assert torch.equal(compiled(q, k, v), block(q, k, v))
    torch._dynamo.reset()


@needs_kernel
def test_h3_strided_processor_runs_the_kernel_on_cuda(_active_backend):
    h3, A = _h3()
    model = _tiny_h3(h3, head_dim = 64).to("cuda", torch.bfloat16)
    model.set_attention_backend("_native_cudnn")
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model) == 3
    kw = {
        k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _h3_inputs(n_video = 256).items()
    }
    for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        kw[name] = kw[name].to(torch.bfloat16)
    assert I.install_int8_attention(model, device = "cuda", dtype = torch.bfloat16, force = True) == 3
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
    model = _tiny_h3(h3, head_dim = 64).to("cuda", torch.bfloat16)
    model.set_attention_backend("_native_cudnn")
    assert A.install_strided_attention(model) == 3
    mods = I.int8_processors(model)
    blocks = [m for m in mods if any(m is b.attn for b in model.transformer_blocks)]
    saved = {m: m.processor._attention_backend for m in mods}
    assert len(blocks) == 2
    assert I.install_int8_attention(model, device = "cuda", dtype = torch.bfloat16, modules = blocks, force = True) == 2
    I.remove_int8_attention(model)
    before = I.STATS["int8"]
    found = H.race_checkpoint_attention(model, saved, torch.bfloat16, kernel = True)
    # 2 warm-ups + 5 timed calls of the kernel per canvas, and the processor is left as it was
    assert I.STATS["int8"] == before + 14
    assert sorted(found["kernel"]) == [1024, 4096]
    assert all(len(v) == 3 and min(v) > 0 for v in found["kernel"].values())
    assert found["ms"] == found["kernel"][4096][:2]
    proc = blocks[0].processor
    assert not proc._unsloth_int8 and proc._attention_backend == saved[blocks[0]]
    # without the kernel the race is the two-way one
    plain = H.race_checkpoint_attention(model, saved, torch.bfloat16)
    assert "kernel" not in plain and len(plain["ms"]) == 2 and I.STATS["int8"] == before + 14
    import logging

    routed = H._race_int8_kernel(
        model, blocks, saved, "cuda", torch.bfloat16, False, False, logging.getLogger(__name__)
    )
    record = getattr(model, H.ATTENTION_RACE_ATTR)
    assert record["timed"] == "block" and record["blocks"] == routed and routed in (0, 2)
    assert I.int8_attention_count(model) == routed
