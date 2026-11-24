# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 low-RAM route on fp16-only GPUs: bf16 storage, float32 compute, decided before any download."""

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_te_prequant as tp
import core.inference.diffusion_transformer_quant as tq
import core.inference.video as v
import core.inference.video_ltx25_lowram as lr
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

LTX25 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2.5")
LTX2 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2")
GIB = 1024
_GIB = 1e9 / 1024**3
# route: anonymous (float32 convolution companions, the encoder's float32 dense tensors, runtime) + the larger mapped
# working set (int8 DiT file, or the encoder's int8 codes + connectors)
_DENSE_PART = 2 * 13.1 - 23.9
ROUTE_INT8_TE_GIB = (
    2.82 * 2 + _DENSE_PART * 2 + 10.5 + max(25.11, 13.1 - _DENSE_PART + 6.34)
) * _GIB
ROUTE_DENSE_TE_GIB = (2.82 * 2 + 23.9 * 2 + 10.5 + 25.11) * _GIB
# the dense bf16 DiT shards (an explicit "none", a LoRA to merge, the hosted read off) instead of the 25.1 GB file
ROUTE_DENSE_DIT_GIB = (2.82 * 2 + _DENSE_PART * 2 + 10.5 + 42.0) * _GIB
# Speed "off" with both precisions unset: the dense encoder and the dense DiT
ROUTE_DENSE_ALL_GIB = (2.82 * 2 + 23.9 * 2 + 10.5 + 42.0) * _GIB
FP32_DENSE_TE_GIB = ((42.0 + 9.2) * 2 + 23.9 * 2) * _GIB


def _target(dtype):
    return DiffusionDeviceTarget(
        device = "cuda",
        dtype = dtype,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )


@pytest.fixture
def host(monkeypatch):
    for env in (
        "UNSLOTH_DIFFUSION_SMALL_HOST",
        "UNSLOTH_DIFFUSION_HOST_RAM_CHECK",
        "UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: True)
    for env in ("UNSLOTH_VIDEO_NATIVE_HOSTED", "UNSLOTH_VIDEO_OFFLOAD_HOSTED_READ"):
        monkeypatch.delenv(env, raising = False)
    # a CPU test host: the card is a Turing whose offloaded int8 build reads the hosted checkpoint
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (7, 5))
    monkeypatch.setattr(
        v,
        "_video_native_hosted_scheme",
        lambda fam, scheme, target: "int8"
        if scheme == "int8" and target.dtype is torch.bfloat16 and target.device == "cuda"
        else None,
    )

    def _no_hub(*a, **k):
        raise AssertionError("the route decision must not touch the Hub")

    import huggingface_hub

    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, _no_hub)

    def _set(dtype, host_gib):
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: _target(dtype))
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(host_gib * GIB))
        monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: int(host_gib * GIB * 0.9))

    return _set


def _decide(
    dtype = torch.float16,
    te = "int8",
    fam = LTX25,
    kind = "pipeline",
):
    return lr.decide_lowram(fam, _target(dtype), model_kind = kind, text_encoder_quant = te)


@pytest.mark.parametrize("te", ["int8", "auto", None])
def test_colab_t4_high_ram_host_takes_the_route(host, te):
    host(torch.float16, 51)
    route = _decide(te = te)
    assert route.engaged and route.refusal is None
    assert abs(route.need_mib / GIB - ROUTE_INT8_TE_GIB) < 0.05
    assert (
        v.ltx25_fp16_host_shortfall(
            LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = te
        )
        is None
    )
    v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = te)
    # the planner and the load see a bfloat16 card
    assert (
        v._ltx25_lowram_target(
            LTX25, _target(torch.float16), kind = "pipeline", text_encoder_quant = te
        ).dtype
        is torch.bfloat16
    )


@pytest.mark.parametrize("host_gib", [31.4, 12.7])
def test_small_host_is_still_refused_before_the_load(host, host_gib):
    host(torch.float16, host_gib)
    route = _decide()
    assert not route.engaged and route.refusal is not None
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = "int8")
    msg = str(err.value)
    assert msg.startswith("Not enough system RAM to load this model: LTX-2.5")
    assert f"about {ROUTE_INT8_TE_GIB:.1f} GB of host memory even on the low-memory route" in msg
    assert (
        "GGUF" in msg
        and "UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0" in msg
        and f"{host_gib} GB is available" in msg
    )
    assert (
        v._ltx25_lowram_target(
            LTX25, _target(torch.float16), kind = "pipeline", text_encoder_quant = "int8"
        ).dtype
        is torch.float16
    )


def test_dense_encoder_is_priced_at_float32(host):
    host(torch.float16, ROUTE_DENSE_TE_GIB + 0.01)
    assert _decide(te = "none").engaged
    host(torch.float16, ROUTE_DENSE_TE_GIB - 0.5)
    route = _decide(te = "none")
    assert not route.engaged and route.refusal is not None


@pytest.mark.parametrize(
    "dtype, fam, kind",
    [
        (torch.bfloat16, LTX25, "pipeline"),
        (torch.float16, LTX25, "gguf"),
        (torch.float16, LTX25, "single_file"),
        (torch.float16, LTX2, "pipeline"),
    ],
)
def test_route_never_engages_off_its_cell(host, dtype, fam, kind):
    host(dtype, 51)
    route = _decide(dtype = dtype, fam = fam, kind = kind)
    assert not route.engaged and route.refusal is None
    target = _target(dtype)
    assert v._ltx25_lowram_target(fam, target, kind = kind) is target


def test_host_that_holds_the_float32_load_is_unchanged(host):
    host(torch.float16, FP32_DENSE_TE_GIB + 1)
    route = _decide(te = "none")
    assert not route.engaged and route.refusal is None and "fits" in route.reason


def test_forced_route_on_a_large_host(host, monkeypatch):
    host(torch.float16, 512)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SMALL_HOST", "1")
    assert _decide().engaged


def test_kill_switch_keeps_the_float32_load(host, monkeypatch):
    host(torch.float16, 51)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SMALL_HOST", "0")
    assert not _decide().engaged
    # the float32 load as before #this change: its own kill switch semantics (no refusal)
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None
    target = _target(torch.float16)
    assert v._ltx25_lowram_target(LTX25, target, kind = "pipeline") is target


def test_host_check_off_tries_the_route_on_a_small_host(host, monkeypatch):
    host(torch.float16, 31)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", "0")
    route = _decide()
    assert route.engaged and route.refusal is None


def test_unreadable_host_keeps_the_old_load(host, monkeypatch):
    host(torch.float16, 51)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: None)
    route = _decide()
    assert not route.engaged and route.refusal is None


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, 32)
        self.lin = torch.nn.Linear(32, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.conv = torch.nn.Conv1d(64, 8, 3, padding = 1)
        self.scale = torch.nn.Parameter(torch.randn(64))

    def forward(self, ids):
        x = self.norm(self.lin(self.emb(ids)) * self.scale)
        return self.conv(x.transpose(1, 2))


def test_upcast_keeps_linear_and_conv_weights_bf16_and_computes_float32():
    pytest.importorskip("diffusers")  # the cast hook is a diffusers hook
    torch.manual_seed(0)
    ref = _Tiny().to(torch.bfloat16)
    dense = _Tiny()
    dense.load_state_dict({k: t.float() for k, t in ref.state_dict().items()})
    pipe = type("P", (), {})()
    pipe.transformer = ref
    kept = lr.upcast_compute_(pipe, torch.float32)
    assert set(kept) == {"transformer"}
    assert ref.lin.weight.dtype is torch.bfloat16 and ref.conv.weight.dtype is torch.bfloat16
    assert ref.norm.weight.dtype is torch.float32 and ref.emb.weight.dtype is torch.float32
    assert ref.scale.dtype is torch.float32
    assert kept["transformer"] == 0  # under 1 MiB
    ids = torch.randint(0, 16, (2, 5))
    out = ref(ids)
    assert out.dtype is torch.float32
    assert (
        ref.lin.weight.dtype is torch.bfloat16 and ref.conv.weight.dtype is torch.bfloat16
    )  # cast back
    torch.testing.assert_close(out, dense(ids), rtol = 0, atol = 0)


# -- what the load will read is what the route prices --------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["transformer_quant_none", "lora", "native_hosted_off", "offload_hosted_read_off"],
)
def test_dense_dit_is_priced_and_refused_on_51_gib(host, monkeypatch, case):
    host(torch.float16, 51)
    kw = {}
    if case == "transformer_quant_none":
        kw["transformer_quant"] = "none"
    elif case == "lora":
        kw["loras_active"] = True
    elif case == "native_hosted_off":
        monkeypatch.setenv("UNSLOTH_VIDEO_NATIVE_HOSTED", "0")
    else:
        monkeypatch.setenv("UNSLOTH_VIDEO_OFFLOAD_HOSTED_READ", "0")
    assert abs(ROUTE_DENSE_DIT_GIB - 58.4) < 0.05
    route = lr.decide_lowram(
        LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = "int8", **kw
    )
    assert not route.engaged and route.refusal is not None
    assert abs(route.need_mib / GIB - ROUTE_DENSE_DIT_GIB) < 0.05
    with pytest.raises(RuntimeError, match = f"about {ROUTE_DENSE_DIT_GIB:.1f} GB"):
        v.assert_video_precision_available(
            LTX25, model_kind = "pipeline", text_encoder_quant = "int8", **kw
        )
    # and on a host that holds it, the route engages at that price
    host(torch.float16, ROUTE_DENSE_DIT_GIB + 0.5)
    assert lr.decide_lowram(
        LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = "int8", **kw
    ).engaged


def test_speed_off_with_unset_encoder_prices_what_the_loader_prices(host):
    # speed "off" + both precisions unset: the loader resolves the encoder and the DiT to the dense ones (98.7 GiB on
    # the route); it passes the settled encoder and the raw DiT request with the speed mode
    host(torch.float16, 51)
    loader = lr.decide_lowram(
        LTX25,
        _target(torch.float16),
        model_kind = "pipeline",
        text_encoder_quant = v.video_te_quant_request(
            LTX25, None, base = LTX25.base_repo, speed_mode = "off"
        )[0],
        speed_mode = "off",
        resolve_te = False,
    )
    assert not loader.engaged and abs(loader.need_mib / GIB - ROUTE_DENSE_ALL_GIB) < 0.05
    preflight = lr.decide_lowram(
        LTX25, _target(torch.float16), model_kind = "pipeline", speed_mode = "off"
    )
    assert (preflight.engaged, preflight.need_mib) == (loader.engaged, loader.need_mib)
    with pytest.raises(RuntimeError, match = f"about {ROUTE_DENSE_ALL_GIB:.1f} GB"):
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", speed_mode = "off")
    # the default speed keeps the hosted int8 encoder and the route
    v.assert_video_precision_available(LTX25, model_kind = "pipeline")


def test_explicit_int8_is_judged_on_the_storage_target(host, monkeypatch):
    host(torch.float16, 51)
    seen = []
    monkeypatch.setattr(
        v, "_assert_video_precision_for_target", lambda fam, target, **k: seen.append(target.dtype)
    )
    v.assert_video_precision_available(LTX25, model_kind = "pipeline", transformer_quant = "int8")
    assert seen == [torch.bfloat16]
    # off the route (the host holds the float32 load) the real card is judged, as before
    host(torch.float16, FP32_DENSE_TE_GIB + 1)
    v.assert_video_precision_available(LTX25, model_kind = "pipeline", transformer_quant = "int8")
    assert seen == [torch.bfloat16, torch.float16]


def test_explicit_int8_preflight_passes_on_the_route(host, monkeypatch):
    # the real check: int8 is a native offload build on the bf16 storage target, refused on the float32 card
    host(torch.float16, 51)
    monkeypatch.setattr(tq, "_scheme_supported", lambda *a, **k: True)
    v.assert_video_precision_available(LTX25, model_kind = "pipeline", transformer_quant = "int8")


@pytest.mark.parametrize("cap", [(7, 0), (6, 1), None])
def test_pre_turing_is_not_eligible_and_refuses_cleanly(host, monkeypatch, cap):
    host(torch.float16, 51)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: cap)
    route = _decide()
    assert not route.engaged and "int8 tensor cores" in route.reason
    target = _target(torch.float16)
    assert (
        v._ltx25_lowram_target(LTX25, target, kind = "pipeline", text_encoder_quant = "int8") is target
    )
    # the float32 load's refusal, before any download
    with pytest.raises(RuntimeError, match = "Not enough system RAM"):
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = "int8")


def test_turing_and_newer_are_eligible(host, monkeypatch):
    host(torch.float16, 51)
    for cap in ((7, 5), (8, 0), (8, 9), (12, 0)):
        monkeypatch.setattr(tq, "_capability", lambda ordinal = None, cap = cap: cap)
        assert _decide().engaged, cap


# -- the upcast ---------------------------------------------------------------------------------------------------


class _Native(torch.nn.Module):
    """A native int8 Linear's shape: int8 codes, bf16 scale, and a ``weight`` property that dequantizes."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.randint(-127, 127, (64, 32), dtype = torch.int8))
        self.register_buffer("scale", torch.rand(64, dtype = torch.bfloat16))
        self.reads = 0

    @property
    def weight(self):
        self.reads += 1
        return self.qweight.to(torch.bfloat16) * self.scale[:, None]

    def forward(self, x):
        return x


def test_upcast_never_reads_a_native_linear_weight():
    pytest.importorskip("diffusers")  # the cast hook is a diffusers hook
    pipe = type("P", (), {})()
    pipe.transformer = torch.nn.Sequential(_Native(), torch.nn.Linear(32, 32).to(torch.bfloat16))
    lr.upcast_compute_(pipe, torch.float32)
    native = pipe.transformer[0]
    assert native.reads == 0
    assert native.qweight.dtype is torch.int8


class _Boom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8).to(torch.bfloat16)

    def forward(self, x):
        return self.lin(x)


def test_storage_tensors_are_restored_even_when_forward_raises(monkeypatch):
    pytest.importorskip("diffusers")  # the cast hook is a diffusers hook
    model = _Boom()
    pipe = type("P", (), {})()
    pipe.transformer = model
    lr.upcast_compute_(pipe, torch.float32)
    ptr = model.lin.weight.data_ptr()
    stored = model.lin.weight.data
    out = model(torch.randn(2, 8))
    assert out.dtype is torch.float32
    # the stored tensor itself comes back (an mmap or pinned alias is kept, not a fresh bf16 copy)
    assert model.lin.weight.dtype is torch.bfloat16 and model.lin.weight.data_ptr() == ptr
    assert model.lin.weight.data is stored or model.lin.weight.data.data_ptr() == stored.data_ptr()

    def _raise(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(torch.nn.functional, "linear", _raise)
    with pytest.raises(RuntimeError, match = "out of memory"):
        model(torch.randn(2, 8))
    assert model.lin.weight.dtype is torch.bfloat16 and model.lin.weight.data_ptr() == ptr


def test_route_runs_eager():
    from core.inference.diffusion_speed import compile_eligible

    card = _target(torch.float16)
    eager = lr.eager_target(card)
    assert eager.supports_default_torch_compile is False and eager.dtype is torch.float16
    assert not compile_eligible(eager, is_gguf = False, family = LTX25, probe = False)
    # even were the family flag dropped, the route's target stays eager
    relaxed = type("F", (), {"fp16_incompatible": False, "supports_torch_compile": True})()
    assert not compile_eligible(eager, is_gguf = False, family = relaxed, probe = False)


def test_native_cache_is_written_inside_the_load_on_the_route(monkeypatch):
    monkeypatch.setattr(v, "_native_cache_background", lambda: True)
    assert v._native_cache_write_deferred(lowram_engaged = False) is True
    assert v._native_cache_write_deferred(lowram_engaged = True) is False


@pytest.mark.parametrize("tq", [None, "auto"])
def test_speed_off_keeps_an_unset_dit_dense_in_the_price_and_the_plan(host, tq):
    # Speed "off" resolves an unset / auto DiT to dense bf16 in the load, so the route reads the 42 GB shards
    host(torch.float16, 51)
    assert v.video_transformer_quant_request(tq, "off") == "off"
    assert v.video_transformer_quant_request(tq, None) == "auto"
    storage = lr.storage_target(_target(torch.float16))
    assert not v._ltx25_lowram_hosted_dit(LTX25, storage, transformer_quant = tq, speed_mode = "off")
    assert v._ltx25_lowram_hosted_dit(LTX25, storage, transformer_quant = tq, speed_mode = "default")
    route = lr.decide_lowram(
        LTX25,
        _target(torch.float16),
        model_kind = "pipeline",
        text_encoder_quant = "int8",
        transformer_quant = tq,
        speed_mode = "off",
    )
    assert not route.engaged and abs(route.need_mib / GIB - ROUTE_DENSE_DIT_GIB) < 0.05
    with pytest.raises(RuntimeError, match = f"about {ROUTE_DENSE_DIT_GIB:.1f} GB"):
        v.assert_video_precision_available(
            LTX25,
            model_kind = "pipeline",
            transformer_quant = tq,
            text_encoder_quant = "int8",
            speed_mode = "off",
        )
    # an explicit int8 DiT under Speed "off" still reads the hosted file
    assert lr.decide_lowram(
        LTX25,
        _target(torch.float16),
        model_kind = "pipeline",
        text_encoder_quant = "int8",
        transformer_quant = "int8",
        speed_mode = "off",
    ).engaged
