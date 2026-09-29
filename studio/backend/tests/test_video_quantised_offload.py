"""Conventional video families (Wan2.2, HunyuanVideo-1.5, LTX-2) when the card has to offload.

Precision is decided by the offload tier, never dropped by it: the planner gets the DiT / text-encoder / VAE split so
a DiT that fits beside the VAE stays resident while the text encoders stream; a bf16 DiT that fits there stays bf16;
an auto DiT that has to move runs the torchao-free int8 (plain buffers ride every hook) instead of bf16; a torchao DiT
under any offload renders under no_grad; and a tier whose co-resident weights cannot fit is refused up front.
"""

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

# Tier codes: "none" every component resident; "dit" DiT resident, text encoders streamed; "stream" DiT streamed,
# companions resident; "both" DiT and text encoders streamed; "model" whole-module offload (from 16 GiB up it is chosen
# over streaming only because each int8 component fits whole, so the DiT is paged once per call, not every step).
# Precision codes: "bf16"; "ao" torchao int8 on a resident DiT; "w8" the torchao-free int8 on a DiT that moves.
_BF, _AO, _W8 = "bf16", "ao", "w8"
EXPECTED = {
    "wan2.2-ti2v-5b": {
        8: ("model", _W8), 12: ("model", _W8), 16: ("model", _W8), 24: ("dit", _AO),
        32: ("dit", _BF), 48: ("none", _BF), 80: ("none", _BF),
    },
    "wan2.2-t2v-a14b": {
        8: ("model", _W8), 12: ("both", _W8), 16: ("both", _W8), 24: ("model", _W8),
        32: ("model", _W8), 48: ("dit", _AO), 80: ("dit", _BF),
    },
    "hv15-480p": {
        8: ("model", _W8), 12: ("model", _W8), 16: ("both", _W8), 24: ("dit", _AO),
        32: ("dit", _BF), 48: ("dit", _BF), 80: ("none", _BF),
    },
    "ltx-2": {
        8: ("model", _W8), 12: ("model", _W8), 16: ("both", _W8), 24: ("both", _W8),
        32: ("model", _W8), 48: ("dit", _AO), 80: ("dit", _BF),
    },
}
EXPECTED["hv15-720p"] = EXPECTED["hv15-480p"]


def _tier_code(plan) -> str:
    if plan.offload_policy == "group":
        if not plan.stream_transformer:
            return "dit"
        return "both" if plan.stream_text_encoders else "stream"
    return plan.offload_policy


def _spoof(monkeypatch, *, tier_gib: int, cap = (12, 0)):
    """The tests' faked runtime on an NVIDIA bf16 card of ``tier_gib`` with 600 MiB taken (a desktop), plus spies on
    the placement, the quantise call and the speed layer."""
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
    # An NVIDIA bf16 host: torchao is open, and the torchao-free int8 serves only a DiT that moves.
    monkeypatch.setattr(tq, "native_quant_host", lambda t: False)
    monkeypatch.setattr(tq, "native_offload_host", lambda t: True)
    monkeypatch.setattr(V, "stored_denoiser_precision", lambda *a, **k: None)
    spy = types.SimpleNamespace(plans = [], quant = [], speed = [], seeded = [])

    import core.inference.video_denoiser_prequant as vdp

    def _seed(fam, base, *, scheme, **kw):
        spy.seeded.append(scheme)
        return {"transformer": object()}

    monkeypatch.setattr(vdp, "denoiser_prequant_pipe_kwargs", _seed)

    def _quantize(view, tgt, *, mode, family = None, logger = None, **kw):
        spy.quant.append({"mode": mode, **kw})
        if kw.get("offload"):
            return "int8"
        return tq.select_transformer_quant_scheme(tgt, mode, family = family)

    monkeypatch.setattr(V, "quantize_transformer", _quantize)
    monkeypatch.setattr(V, "native_quant_reason", lambda m, s: f"native {s}")

    def _speed(view, tgt, *, is_gguf, family, speed_mode, cache_active, offload_active, **kw):
        compiled = speed_mode in ("default", "max") and not is_gguf and family.supports_torch_compile
        spy.speed.append({"compiled": compiled, "offload_active": offload_active})
        return {"compiled": compiled}

    monkeypatch.setattr(V, "apply_speed_optims", _speed)

    def _apply(pipe, plan, *, device = None, placement_device = None, logger = None):
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
    """Every conventional pipeline family at every card size: the tier, the precision it keeps, compile on, admitted.
    Before the split a bf16 plan had only 'none' or whole-module offload, and any offload meant a bf16 DiT: the 24 GiB
    A14B / LTX-2 and 16 GiB HunyuanVideo rows onloaded a bf16 component larger than the card."""
    spy = _spoof(monkeypatch, tier_gib = tier_gib, cap = CAPS[cap_name])
    status = VideoBackend().load_pipeline(FAMILIES[family])
    tier, precision = EXPECTED[family][tier_gib]
    assert _tier_code(spy.plans[-1]) == tier
    assert _precision_code(spy, status) == precision
    # the regional compile stays on in every cell, offloaded or not
    assert spy.speed and all(call["compiled"] for call in spy.speed)
    assert status["loaded"] is True
    if precision == _W8:
        # the torchao-free build is asked for by name: auto alone would walk the torchao ladder
        assert spy.quant[0]["mode"] == "int8"


@pytest.mark.parametrize("family", ["wan2.2-t2v-a14b", "ltx-2"])
def test_auto_keeps_bf16_where_the_dit_stays_resident_at_80(fake_runtime, monkeypatch, family):
    """80 GiB: the bf16 DiT fits beside the VAE with the text encoder streamed, so auto keeps bf16 (the resident
    #11831 rule) rather than quantising to squeeze the text encoder in too."""
    spy = _spoof(monkeypatch, tier_gib = 80)
    status = VideoBackend().load_pipeline(FAMILIES[family])
    assert _tier_code(spy.plans[-1]) == "dit"
    assert status["transformer_quant"] in (None, "off")
    assert not spy.quant


def test_explicit_fp8_resident_dit_beside_streamed_text_encoder(fake_runtime, monkeypatch):
    """A torchao fp8 DiT on a 24 GiB card: it fits beside the VAE once the text encoder streams, so it engages. Before,
    the summed plan offloaded and the explicit ask was refused."""
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
    """Wan2.2-5B NVFP4 on a 32 GiB card: 2.9 GB DiT, 11.4 GB text encoder. The summed plan dropped the seed and refused
    the explicit ask; the split keeps the DiT resident and streams the encoder."""
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
    """The pre-download twin of the load-time seed check uses the same split."""
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

    def _plan(policy, te = False, dit = True):
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
        # resident DiT beside streamed text encoders: still a hooked render
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

    def _plan(policy, te = False, dit = True):
        return types.SimpleNamespace(
            offload_policy = policy, stream_text_encoders = te, stream_transformer = dit
        )

    assert _video_offload_vram_floor_mib(pipe, _plan("none")) is None
    assert _video_offload_vram_floor_mib(pipe, _plan("model")) == 12
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = True, dit = False)) == 23
    assert _video_offload_vram_floor_mib(pipe, _plan("group")) == 15
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = True)) == 3
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming")) == 3


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
    # fits: generates
    backend.generate(prompt = "a sloth", width = 256, height = 256, num_frames = 9, fps = 8)
    free["bytes"] = 20 * 1024**3
    with pytest.raises(RuntimeError, match = "needs about"):
        backend.generate(prompt = "a sloth", width = 256, height = 256, num_frames = 9, fps = 8)
