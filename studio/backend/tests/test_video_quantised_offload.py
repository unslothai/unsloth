# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Conventional video families (Wan2.2, HunyuanVideo-1.5, LTX-2) when the card has to offload."""

import dataclasses
import sys
import types

import pytest

from .test_video_backend import (  # noqa: F401 -- fixtures
    _assume_the_restricted_load_is_available,
    _FakePipe,
    _load_gguf,
    fake_runtime,
)
from core.inference.video import VideoBackend

GIB = 1024
TIERS = (8, 12, 16, 24, 32, 48, 80)
CAPS = {"sm_120": (12, 0), "sm_89": (8, 9)}
SUPPORTED = {(12, 0): {"int8", "fp8", "mxfp8", "nvfp4"}, (8, 9): {"int8", "fp8"}}

FAMILIES = {
    "wan2.2-ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan2.2-t2v-a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "hv15-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "hv15-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "ltx-2": "Lightricks/LTX-2",
}

# Tiers: none, dit (TE streamed), stream (DiT streamed), both, model (whole-module). Precision: bf16, ao (torchao), w8 (torchao-free int8).
_BF, _AO, _W8 = "bf16", "ao", "w8"
EXPECTED = {
    "wan2.2-ti2v-5b": {
        8: ("model", _W8),
        12: ("model", _W8),
        16: ("model", _W8),
        24: ("dit", _AO),
        32: ("dit", _BF),
        48: ("none", _BF),
        80: ("none", _BF),
    },
    "wan2.2-t2v-a14b": {
        8: ("model", _W8),
        12: ("both", _W8),
        16: ("both", _W8),
        24: ("model", _W8),
        32: ("model", _W8),
        48: ("dit", _AO),
        80: ("dit", _BF),
    },
    "hv15-480p": {
        8: ("model", _W8),
        12: ("model", _W8),
        16: ("both", _W8),
        24: ("dit", _AO),
        32: ("dit", _BF),
        48: ("dit", _BF),
        80: ("none", _BF),
    },
    "ltx-2": {
        8: ("model", _W8),
        12: ("model", _W8),
        16: ("both", _W8),
        24: ("both", _W8),
        32: ("model", _W8),
        48: ("dit", _AO),
        80: ("dit", _BF),
    },
}
EXPECTED["hv15-720p"] = EXPECTED["hv15-480p"]


def _tier_code(plan) -> str:
    if plan.offload_policy == "group":
        if not plan.stream_transformer:
            return "dit"
        return "both" if plan.stream_text_encoders else "stream"
    return plan.offload_policy


def _spoof(
    monkeypatch,
    *,
    tier_gib: int,
    cap = (12, 0),
):
    import core.inference.video as V
    from core.inference import diffusion_transformer_quant as tq
    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_memory import DeviceMemory

    torch = sys.modules["torch"]
    target = DiffusionDeviceTarget(
        device = "cuda",
        dtype = torch.bfloat16,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )
    total = tier_gib * GIB
    mem = DeviceMemory(
        backend = "cuda",
        device = "cuda",
        memory_kind = "discrete_vram",
        free_mib = total - 600,
        total_mib = total,
    )
    monkeypatch.setattr(V, "settled_snapshot_device_memory", lambda *a, **k: mem)
    monkeypatch.setattr(VideoBackend, "_device_target", lambda self, ordinal = None: target)
    monkeypatch.setattr(V, "resolve_diffusion_device_target", lambda *a, **k: target)
    monkeypatch.setattr(V, "dense_transformer_supported", lambda t: True)
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda t: True)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: cap)
    monkeypatch.setattr(
        tq, "_scheme_supported", lambda s, d, unproven_ok = False: s in SUPPORTED[cap]
    )
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda d: tier_gib <= 32)
    monkeypatch.setattr(tq, "native_quant_host", lambda t: False)
    monkeypatch.setattr(tq, "native_offload_host", lambda t: True)
    monkeypatch.setattr(V, "stored_denoiser_precision", lambda *a, **k: None)
    spy = types.SimpleNamespace(plans = [], quant = [], speed = [], seeded = [])

    import core.inference.video_denoiser_prequant as vdp

    def _seed(fam, base, *, scheme, **kw):
        spy.seeded.append(scheme)
        return {"transformer": object()}

    monkeypatch.setattr(vdp, "denoiser_prequant_pipe_kwargs", _seed)

    def _quantize(
        view,
        tgt,
        *,
        mode,
        family = None,
        logger = None,
        **kw,
    ):
        spy.quant.append({"mode": mode, **kw})
        if kw.get("offload"):
            return "int8"
        return tq.select_transformer_quant_scheme(tgt, mode, family = family)

    monkeypatch.setattr(V, "quantize_transformer", _quantize)
    monkeypatch.setattr(V, "native_quant_reason", lambda m, s: f"native {s}")

    def _speed(view, tgt, *, is_gguf, family, speed_mode, cache_active, offload_active, **kw):
        compiled = (
            speed_mode in ("default", "max") and not is_gguf and family.supports_torch_compile
        )
        spy.speed.append({"compiled": compiled, "offload_active": offload_active})
        return {"compiled": compiled}

    monkeypatch.setattr(V, "apply_speed_optims", _speed)

    def _apply(
        pipe,
        plan,
        *,
        device = None,
        placement_device = None,
        logger = None,
    ):
        spy.plans.append(plan)
        return plan.offload_policy, plan.vae_tiling

    monkeypatch.setattr(V, "apply_memory_plan", _apply)
    return spy


def _precision_code(spy, status) -> str:
    engaged = status.get("transformer_quant")
    if not engaged or engaged == "off":
        return _BF
    assert engaged == "int8", engaged
    return _W8 if spy.quant and spy.quant[0].get("offload") else _AO


@pytest.mark.parametrize("cap_name", list(CAPS))
@pytest.mark.parametrize("tier_gib", TIERS)
@pytest.mark.parametrize("family", list(FAMILIES))
def test_auto_planner_table(fake_runtime, monkeypatch, family, tier_gib, cap_name):
    """Every conventional family at every card size: tier, precision, compile on, admitted."""
    spy = _spoof(monkeypatch, tier_gib = tier_gib, cap = CAPS[cap_name])
    status = VideoBackend().load_pipeline(FAMILIES[family])
    tier, precision = EXPECTED[family][tier_gib]
    assert _tier_code(spy.plans[-1]) == tier
    assert _precision_code(spy, status) == precision
    assert spy.speed and all(call["compiled"] for call in spy.speed)
    assert status["loaded"] is True
    if precision == _W8:
        # the torchao-free build is asked for by name: auto alone would walk the torchao ladder
        assert spy.quant[0]["mode"] == "int8"


@pytest.mark.parametrize("family", ["wan2.2-t2v-a14b", "ltx-2"])
def test_auto_keeps_bf16_where_the_dit_stays_resident_at_80(fake_runtime, monkeypatch, family):
    """80 GiB: bf16 DiT stays resident with the text encoder streamed, so auto keeps bf16."""
    spy = _spoof(monkeypatch, tier_gib = 80)
    status = VideoBackend().load_pipeline(FAMILIES[family])
    assert _tier_code(spy.plans[-1]) == "dit"
    assert status["transformer_quant"] in (None, "off")
    assert not spy.quant


def test_explicit_fp8_resident_dit_beside_streamed_text_encoder(fake_runtime, monkeypatch):
    """A torchao fp8 DiT on 24 GiB engages once the text encoder streams."""
    spy = _spoof(monkeypatch, tier_gib = 24)
    status = VideoBackend().load_pipeline(FAMILIES["wan2.2-ti2v-5b"], transformer_quant = "fp8")
    assert _tier_code(spy.plans[-1]) == "dit"
    assert status["transformer_quant"] == "fp8"
    assert spy.quant and not spy.quant[0].get("offload")


def test_explicit_fp8_refused_with_reason_when_the_dit_must_move(fake_runtime, monkeypatch):
    _spoof(monkeypatch, tier_gib = 16)
    with pytest.raises(RuntimeError, match = r"moves the DiT .*int8 runs torchao-free"):
        VideoBackend().load_pipeline(FAMILIES["wan2.2-ti2v-5b"], transformer_quant = "fp8")


def test_explicit_int8_under_streamed_dit_is_torchao_free(fake_runtime, monkeypatch):
    spy = _spoof(monkeypatch, tier_gib = 24)
    status = VideoBackend().load_pipeline(FAMILIES["ltx-2"], transformer_quant = "int8")
    assert _tier_code(spy.plans[-1]) == "both"
    assert status["transformer_quant"] == "int8"
    assert spy.quant[0] == {"mode": "int8", "offload": True, "act_int8": True}


def test_speed_off_keeps_bf16_under_offload(fake_runtime, monkeypatch):
    """speed_mode='off' is the bit-exact contract: no auto quant even where the DiT has to move."""
    spy = _spoof(monkeypatch, tier_gib = 16)
    status = VideoBackend().load_pipeline(FAMILIES["wan2.2-ti2v-5b"], speed_mode = "off")
    assert status["transformer_quant"] in (None, "off")
    assert not spy.quant


def test_hosted_nvfp4_seeds_beside_streamed_text_encoder(fake_runtime, monkeypatch):
    """Wan2.2-5B NVFP4 on 32 GiB: the split keeps the DiT resident and streams the 11.4 GB encoder."""
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    spy = _spoof(monkeypatch, tier_gib = 32)
    status = VideoBackend().load_pipeline(FAMILIES["wan2.2-ti2v-5b"], transformer_quant = "nvfp4")
    assert spy.seeded == ["nvfp4"]
    assert status["transformer_quant"] == "nvfp4"
    assert _tier_code(spy.plans[-1]) == "dit"


def test_hosted_nvfp4_refused_where_the_dit_must_stream(fake_runtime, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    spy = _spoof(monkeypatch, tier_gib = 16)
    with pytest.raises(RuntimeError, match = "moves the DiT"):
        VideoBackend().load_pipeline(FAMILIES["wan2.2-ti2v-5b"], transformer_quant = "nvfp4")
    assert not spy.seeded


def test_seed_stays_resident_counts_the_streamed_text_encoder(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.video as V
    from core.inference.diffusion_memory import DeviceMemory
    from core.inference.video_families import detect_video_family

    fam = detect_video_family(FAMILIES["wan2.2-ti2v-5b"])
    target = types.SimpleNamespace(
        device = "cuda", dtype = torch.bfloat16, supports_model_cpu_offload = True
    )

    def _at(gib):
        mem = DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "discrete_vram",
            free_mib = gib * GIB - 600,
            total_mib = gib * GIB,
        )
        monkeypatch.setattr(V, "settled_snapshot_device_memory", lambda *a, **k: mem)
        return V._video_seed_stays_resident(
            fam,
            target = target,
            scheme = "nvfp4",
            memory_mode = None,
            text_encoder_quant = None,
            base_repo = FAMILIES["wan2.2-ti2v-5b"],
        )

    assert _at(32) is True
    assert _at(12) is False


def test_offload_tiers_rank_fastest_first():
    from core.inference.video import _video_plan_label, _video_plan_rank

    def _plan(
        policy,
        te = False,
        dit = True,
    ):
        return types.SimpleNamespace(
            offload_policy = policy, stream_text_encoders = te, stream_transformer = dit
        )

    whole = _plan("model")
    whole.estimates = {"whole_module_fits_mib": 20000}
    order = [
        _plan("none"),
        _plan("group", te = True, dit = False),
        whole,
        _plan("group"),
        _plan("group", te = True),
        _plan("model"),
        _plan("streaming"),
    ]
    assert [_video_plan_rank(p) for p in order] == [0, 1, 2, 3, 4, 5, 6]
    assert _video_plan_label(order[1]) == "denoiser resident, text encoders streamed"


@pytest.mark.parametrize(
    "state, expected",
    [
        # a torchao DiT under whole-module offload: the hook swap fails under inference_mode
        (dict(transformer_quant = "int8", offload_policy = "model"), True),
        (dict(transformer_quant = "fp8", offload_policy = "group"), True),
        (dict(transformer_quant = "int8", offload_policy = "none"), False),
        (dict(transformer_quant = None, offload_policy = "model"), False),
        # MiniMax-H3: only its streamed denoiser switches; the pinned one never moves
        (dict(transformer_quant = "int8", offload_policy = "model", modular = True), False),
        (
            dict(
                transformer_quant = "int8",
                offload_policy = "model",
                modular = True,
                denoiser_streamed = True,
            ),
            True,
        ),
    ],
)
def test_render_grad_mode(state, expected):
    from core.inference.video import _render_under_no_grad

    modular = state.pop("modular", False)
    ns = types.SimpleNamespace(
        family = types.SimpleNamespace(modular_workflow = "t2v" if modular else None),
        denoiser_streamed = state.pop("denoiser_streamed", False),
        **state,
    )
    assert _render_under_no_grad(ns) is expected


def test_vram_floor_counts_what_each_tier_holds_at_once():
    torch = pytest.importorskip("torch")
    from core.inference.video import _video_offload_vram_floor_mib

    def _module(mib):
        module = torch.nn.Module()
        module.register_buffer("w", torch.zeros(mib * 1024 * 1024 // 2, dtype = torch.bfloat16))
        return module

    pipe = types.SimpleNamespace(
        components = {
            "transformer": _module(10),
            "transformer_2": _module(10),
            "text_encoder": _module(12),
            "vae": _module(3),
            "scheduler": object(),
        }
    )

    def _plan(
        policy,
        te = False,
        dit = True,
    ):
        return types.SimpleNamespace(
            offload_policy = policy, stream_text_encoders = te, stream_transformer = dit
        )

    assert _video_offload_vram_floor_mib(pipe, _plan("none")) is None
    assert _video_offload_vram_floor_mib(pipe, _plan("model")) == 12
    # a flat module is one group, so its whole payload is onloaded during its forward
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = True, dit = False)) == 35
    assert _video_offload_vram_floor_mib(pipe, _plan("group")) == 25
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = True)) == 15
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming")) == 15


def test_vram_floor_adds_the_largest_onloaded_block_or_leaf():
    """Streamed tiers hold the unmatched group plus the current and prefetched block (or leaf) on the device."""
    torch = pytest.importorskip("torch")
    from core.inference.video import _video_offload_vram_floor_mib

    def _linear(mib):
        return torch.nn.Linear(1024, mib * 512, bias = False).to(torch.bfloat16)

    dit = torch.nn.Module()
    dit.patch_embedding = _linear(1)
    dit.blocks = torch.nn.ModuleList([_linear(2) for _ in range(4)])
    encoder = torch.nn.Module()
    encoder.shared = torch.nn.Embedding(1024, 3 * 512).to(torch.bfloat16)
    encoder.layers = torch.nn.ModuleList([_linear(1) for _ in range(4)])
    vae = _linear(3)
    pipe = types.SimpleNamespace(
        components = {"transformer": dit, "text_encoder": encoder, "vae": vae}
    )

    def _plan(
        policy,
        backend,
        te = False,
        dit = True,
    ):
        return types.SimpleNamespace(
            offload_policy = policy,
            stream_text_encoders = te,
            stream_transformer = dit,
            device_memory = types.SimpleNamespace(backend = backend),
        )

    # resident TE 7 + VAE 3, plus DiT embed 1 + current and prefetched 2 MiB blocks
    assert _video_offload_vram_floor_mib(pipe, _plan("group", "cuda")) == 15
    assert _video_offload_vram_floor_mib(pipe, _plan("group", "mps")) == 13
    # resident DiT 9 + VAE 3, plus the 3 MiB embedding and the next 1 MiB leaf
    assert _video_offload_vram_floor_mib(pipe, _plan("group", "cuda", te = True, dit = False)) == 16
    # streamed modules run in turn: VAE 3 plus the larger of DiT 5 and TE 4
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming", "cuda")) == 8
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming", "cpu")) == 6


def test_shortfall_message_only_for_what_cannot_run():
    from core.inference.video import video_offload_shortfall_message

    assert (
        video_offload_shortfall_message(
            family = "wan2.2-t2v-a14b", floor_mib = 20000, available_mib = 24000, placement = "x"
        )
        is None
    )
    message = video_offload_shortfall_message(
        family = "wan2.2-t2v-a14b",
        floor_mib = 27300,
        available_mib = 24000,
        placement = "whole-module CPU offload",
        width = 1280,
        height = 720,
        frames = 81,
    )
    assert "wan2.2-t2v-a14b needs about" in message and "1280x720 at 81 frames" in message


def test_generate_requirement_grows_with_the_requested_clip():
    import re

    from core.inference.video import video_offload_shortfall_message

    def _required(width, height, frames):
        message = video_offload_shortfall_message(
            family = "wan2.2-ti2v-5b",
            floor_mib = 10000,
            available_mib = 0,
            placement = "x",
            width = width,
            height = height,
            frames = frames,
        )
        return float(re.search(r"needs about ([0-9.]+) GiB", message).group(1))

    base = _required(640, 480, 49)
    assert _required(1280, 480, 49) > base
    assert _required(640, 960, 49) > base
    assert _required(640, 480, 161) > base
    # a small clip still fits where a long 720p one does not
    kwargs = dict(family = "f", floor_mib = 10000, available_mib = 12500, placement = "x")
    assert video_offload_shortfall_message(**kwargs, width = 256, height = 256, frames = 9) is None
    assert video_offload_shortfall_message(**kwargs, width = 1280, height = 720, frames = 121)


def test_load_refuses_a_tier_whose_weights_cannot_fit(fake_runtime, monkeypatch):
    """A tier whose co-resident weights exceed the card is refused before placement, not at the first onload."""
    import core.inference.video as V

    spy = _spoof(monkeypatch, tier_gib = 24)
    monkeypatch.setattr(V, "_video_offload_vram_floor_mib", lambda pipe, plan: 30000)
    with pytest.raises(RuntimeError, match = "needs about"):
        VideoBackend().load_pipeline(FAMILIES["wan2.2-t2v-a14b"])
    assert not spy.plans


def test_generate_refuses_below_the_offload_floor(fake_runtime, monkeypatch, tmp_path):
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch,
        "device",
        lambda kind, index = None: types.SimpleNamespace(type = kind, index = index),
        raising = False,
    )
    free = {"bytes": 40 * 1024**3}
    monkeypatch.setattr(
        torch,
        "cuda",
        types.SimpleNamespace(
            is_available = lambda: False,
            mem_get_info = lambda device = None: (free["bytes"], 80 * 1024**3),
            memory_reserved = lambda device = None: 0,
        ),
    )
    backend._state = dataclasses.replace(
        backend._state, device = "cuda", offload_policy = "model", vram_floor_mib = 27300
    )
    backend.generate(prompt = "a sloth", width = 256, height = 256, num_frames = 9, fps = 8)
    free["bytes"] = 20 * 1024**3
    with pytest.raises(RuntimeError, match = "needs about"):
        backend.generate(prompt = "a sloth", width = 256, height = 256, num_frames = 9, fps = 8)
    # same free VRAM: a small clip runs, a long 720p clip is refused before the render
    free["bytes"] = 31 * 1024**3
    backend.generate(prompt = "a sloth", width = 256, height = 256, num_frames = 9, fps = 8)
    with pytest.raises(RuntimeError, match = "at 121 frames"):
        backend.generate(prompt = "a sloth", width = 1280, height = 720, num_frames = 121, fps = 8)


def test_quantise_stages_what_fits_on_the_card_and_unstages_it(monkeypatch):
    """The DiT is quantised on the card block by block and sent back to the host when the plan moves it."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    import core.inference.video as V

    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(1024, 1024, bias = False).to(torch.bfloat16) for _ in range(4)]
    )
    dit.proj_out = torch.nn.Linear(1024, 64, bias = False).to(torch.bfloat16)
    block_bytes = 1024 * 1024 * 2
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda")
    # room for the overhead, the linear workspace and exactly two blocks
    free = (
        V.DEFAULT_BASE_OVERHEAD_MIB * 1024 * 1024
        + V._STAGE_LINEAR_WORKSPACE * block_bytes
        + 2 * block_bytes
        + 1
    )
    monkeypatch.setattr(
        "utils.hardware.trusted_mem_get_info", lambda d, module = None: (free, 10 * free)
    )
    staged = V._stage_denoiser_for_quant(dit, target)
    on_card = [m for m in staged if next(m.parameters()).is_cuda]
    assert len(on_card) == len(staged) >= 2
    assert all(next(b.parameters()).is_cuda for b in dit.blocks[:2])
    assert not next(dit.blocks[3].parameters()).is_cuda
    V._unstage_modules(staged)
    assert all(not p.is_cuda for p in dit.parameters())
    # nothing staged off CUDA, or for a DiT already on the card
    assert V._stage_denoiser_for_quant(dit, types.SimpleNamespace(device = "mps")) == []


def test_applied_floor_counts_an_encoder_that_refused_leaf_offload():
    """_apply_group_offload keeps a refusing encoder resident under the same policy; the floor must follow the hooks."""
    torch = pytest.importorskip("torch")
    import inspect

    import core.inference.video as V

    def _module(mib):
        module = torch.nn.Module()
        module.register_buffer("w", torch.zeros(mib * 1024 * 1024 // 2, dtype = torch.bfloat16))
        return module

    dit, encoder = _module(10), _module(12)
    dit._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    pipe = types.SimpleNamespace(
        components = {"transformer": dit, "text_encoder": encoder, "vae": _module(3)}
    )
    plan = types.SimpleNamespace(
        offload_policy = "group",
        stream_text_encoders = True,
        stream_transformer = True,
        device_memory = types.SimpleNamespace(backend = "cuda"),
    )
    # as planned: VAE 3 plus the larger streamed unit (TE 12)
    assert V._video_offload_vram_floor_mib(pipe, plan) == 15
    # as applied: the unhooked encoder is resident, so VAE 3 + TE 12 plus the streamed DiT 10
    assert V._video_offload_vram_floor_mib(pipe, plan, applied = True) == 25
    assert "_video_offload_vram_floor_mib(pipe, plan, applied = True)" in inspect.getsource(V)


def test_failed_staging_leaves_nothing_on_the_card(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    import core.inference.video as V

    class _Refuses(torch.nn.Linear):
        def to(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(256, 256, bias = False).to(torch.bfloat16) for _ in range(2)]
        + [_Refuses(256, 256, bias = False)]
    )
    monkeypatch.setattr(
        "utils.hardware.trusted_mem_get_info", lambda d, module = None: (80 * 1024**3, 80 * 1024**3)
    )
    target = types.SimpleNamespace(device = "cuda", torch_device = "cuda")
    assert V._stage_denoiser_for_quant(dit, target) == []
    assert all(not p.is_cuda for p in dit.parameters())


def test_applied_floor_counts_a_dit_the_fallback_hooked():
    """A refusing encoder can make _apply_group_offload hook the DiT the plan kept resident; count it as streamed."""
    torch = pytest.importorskip("torch")
    import core.inference.video as V

    def _linear(mib):
        return torch.nn.Linear(1024, mib * 512, bias = False).to(torch.bfloat16)

    dit = torch.nn.Module()
    dit.patch_embedding = _linear(1)
    dit.blocks = torch.nn.ModuleList([_linear(2) for _ in range(4)])
    dit.blocks[0]._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    encoder = torch.nn.Module()
    encoder.register_buffer("w", torch.zeros(12 * 1024 * 1024 // 2, dtype = torch.bfloat16))
    vae = torch.nn.Module()
    vae.register_buffer("w", torch.zeros(3 * 1024 * 1024 // 2, dtype = torch.bfloat16))
    pipe = types.SimpleNamespace(components = {"transformer": dit, "text_encoder": encoder, "vae": vae})
    plan = types.SimpleNamespace(
        offload_policy = "group",
        stream_text_encoders = True,
        stream_transformer = False,
        device_memory = types.SimpleNamespace(backend = "cuda"),
    )
    # resident TE 12 + VAE 3, plus the streamed DiT's embed 1 and current and prefetched 2 MiB blocks
    assert V._video_offload_vram_floor_mib(pipe, plan, applied = True) == 20


def test_a_declined_quant_unstages_before_any_bf16_rollback():
    import inspect

    import core.inference.video as V

    assert "if staged and (video_offload or scheme is None):" in inspect.getsource(V)
