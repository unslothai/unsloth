# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Group offload streams LTX-2's run-once text connectors beside a streamed DiT: off the card for the denoise and the
decode, out of the floor and out of the group tiers' planner floors."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_STREAM_ENV = "UNSLOTH_VIDEO_STREAM_RUN_ONCE_COMPANIONS"
_GROUP_ENV = "UNSLOTH_VIDEO_GROUP_RUN_ONCE_COMPANIONS"


@pytest.fixture(autouse = True)
def _envs(monkeypatch):
    monkeypatch.delenv(_STREAM_ENV, raising = False)
    monkeypatch.delenv(_GROUP_ENV, raising = False)


class LTX2TextConnectors(torch.nn.Module):
    def __init__(self, mib):
        super().__init__()
        self.layers = torch.nn.ModuleList([_linear(2) for _ in range(mib // 2)])
        self.moved_to = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args)
        return self


class _Recorded(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.inner = module
        self.moved_to = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args)
        return self


def _linear(mib):
    return torch.nn.Linear(1024, mib * 512, bias = False).to(torch.bfloat16)


class _DiT(torch.nn.Module):
    def to(self, *args, **kwargs):
        return self


def _pipe(connector_mib = 6):
    dit = _DiT()
    dit.blocks = torch.nn.ModuleList([_linear(2) for _ in range(4)])
    pipe = types.SimpleNamespace()
    pipe.transformer = dit
    pipe.connectors = LTX2TextConnectors(connector_mib)
    pipe.vae = _Recorded(_linear(3))
    pipe.components = {
        "transformer": dit,
        "connectors": pipe.connectors,
        "vae": pipe.vae,
        "scheduler": object(),
    }
    return pipe


def _plan(
    policy,
    te = False,
    stream_transformer = True,
):
    return types.SimpleNamespace(
        offload_policy = policy,
        stream_text_encoders = te,
        stream_transformer = stream_transformer,
        device_memory = types.SimpleNamespace(backend = "cuda"),
    )


# ── floor ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("te", [False, True])
def test_group_floor_leaves_the_connectors_off_the_card(te):
    from core.inference.video import _video_offload_vram_floor_mib

    pipe = _pipe()
    # vae 3 + DiT window (2x2 MiB); the 6 MiB connectors stream and are smaller than the window
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = te)) == 3 + 4
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = te), phase = "decode") == 3


@pytest.mark.parametrize("env", [_GROUP_ENV, _STREAM_ENV])
def test_group_floor_kill_switches_keep_the_connectors_resident(monkeypatch, env):
    from core.inference.video import _video_offload_vram_floor_mib

    monkeypatch.setenv(env, "0")
    pipe = _pipe()
    assert _video_offload_vram_floor_mib(pipe, _plan("group")) == 3 + 6 + 4
    assert _video_offload_vram_floor_mib(pipe, _plan("group"), phase = "decode") == 3 + 6


def test_resident_dit_group_plan_keeps_the_connectors_resident():
    from core.inference.video import _video_offload_vram_floor_mib
    pipe = _pipe()
    # the DiT is resident (8) and the connectors with it; nothing streamed
    assert (
        _video_offload_vram_floor_mib(pipe, _plan("group", te = True, stream_transformer = False))
        == 8 + 3 + 6
    )


def test_applied_floor_counts_connectors_that_refused_their_hooks_as_resident(monkeypatch):
    import core.inference.video as video

    pipe = _pipe()
    # only the DiT carries group-offload hooks: the connectors stayed resident
    monkeypatch.setattr(video, "_video_group_hooked", lambda module: module is pipe.transformer)
    assert video._video_offload_vram_floor_mib(pipe, _plan("group"), applied = True) == 3 + 6 + 4
    monkeypatch.setattr(video, "_video_group_hooked", lambda module: module is not pipe.vae)
    assert video._video_offload_vram_floor_mib(pipe, _plan("group"), applied = True) == 3 + 4


# ── _apply_group_offload ───────────────────────────────────────────────────────


def _stub_group(
    monkeypatch,
    budget_mib,
    *,
    defer = False,
):
    pytest.importorskip("diffusers.hooks")
    calls = []
    deferred = []

    def apply_group_offloading(
        module,
        onload_device = None,
        offload_device = None,
        offload_type = None,
        num_blocks_per_group = None,
        use_stream = False,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        calls.append((module, offload_type, low_cpu_mem_usage, use_stream))

    monkeypatch.setattr(
        sys.modules["diffusers.hooks"], "apply_group_offloading", apply_group_offloading
    )
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    for name, value in (
        ("host_pinning_allowed", True),
        ("_pinned_memory_capped", False),
        ("_pin_top_level_group", True),
        ("_any_layer_split", False),
        ("_background_pin_enabled", defer),
        ("_video_exact_pins", False),
        ("_encoder_background_pin_wanted", False),
        ("_late_denoiser_pin_enabled", False),
    ):
        monkeypatch.setattr(dm, name, lambda *a, _v = value, **k: _v)
    for name in (
        "install_group_offload_buffer_restore",
        "install_group_offload_hooks_eager",
        "install_group_pin_wait",
        "install_group_prefetch",
        "_skip_top_level_copy_back",
        "_pin_vision_embedding_device",
    ):
        monkeypatch.setattr(dm, name, lambda *a, **k: None)
    monkeypatch.setattr(
        dm,
        "_apply_group_offloading_packed",
        lambda apply, module, kwargs, logger: apply(module, **kwargs),
    )
    monkeypatch.setattr(
        dm, "_defer_pinning", lambda pipe, module, *a, **k: deferred.append(module) or True
    )
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: budget_mib)
    return calls, deferred


def _hooks(calls):
    return {id(m): (t, low, stream) for m, t, low, stream in calls}


def test_group_offload_streams_the_connectors_pinned_when_they_fit(monkeypatch):
    calls, _ = _stub_group(monkeypatch, 1 << 20)
    pipe = _pipe()
    assert dm._apply_group_offload(pipe, "cuda", None)
    hooked = _hooks(calls)
    assert hooked[id(pipe.transformer)][0] == "block_level"
    assert hooked[id(pipe.connectors)] == ("leaf_level", False, True)
    # never placed on the card whole; the VAE still is
    assert pipe.connectors.moved_to == []
    assert pipe.vae.moved_to


def test_group_offload_streams_the_connectors_unpinned_when_host_ram_is_short(monkeypatch):
    # the 8 MiB DiT pins, the 6 MiB connectors would not: they still leave the card, with no copy stream and no pin
    # at any onload (a refused pin there would fail the first prompt encode)
    calls, deferred = _stub_group(monkeypatch, 9)
    pipe = _pipe()
    assert dm._apply_group_offload(pipe, "cuda", None)
    assert _hooks(calls)[id(pipe.connectors)] == ("leaf_level", True, False)
    assert pipe.connectors.moved_to == []
    assert all(m is not pipe.connectors for m in deferred)


def test_unpinned_connectors_never_pin_even_with_background_pins(monkeypatch):
    calls, deferred = _stub_group(monkeypatch, 9, defer = True)
    pipe = _pipe()
    pipe._unsloth_background_pin_request = True
    monkeypatch.setattr(dm, "_BACKGROUND_PIN_REQUEST_ATTR", "_unsloth_background_pin_request")
    assert dm._apply_group_offload(pipe, "cuda", None)
    assert _hooks(calls)[id(pipe.connectors)] == ("leaf_level", True, False)
    assert all(m is not pipe.connectors for m in deferred)


def test_unpinned_connector_hook_failure_keeps_them_resident(monkeypatch):
    import sys

    calls, _ = _stub_group(monkeypatch, 9)
    pipe = _pipe()
    real = sys.modules["diffusers.hooks"].apply_group_offloading

    def apply(module, **kwargs):
        if module is pipe.connectors:
            raise RuntimeError("refused")
        return real(module, **kwargs)

    monkeypatch.setattr(sys.modules["diffusers.hooks"], "apply_group_offloading", apply)
    monkeypatch.setattr(dm, "_remove_group_offload_hooks", lambda module: None)
    assert dm._apply_group_offload(pipe, "cuda", None)
    assert pipe.connectors.moved_to


@pytest.mark.parametrize("backend, kind", [("xpu", "discrete_vram"), ("cuda", "unified_memory")])
def test_group_floor_keeps_the_connectors_where_no_copy_stream_streams_them(backend, kind):
    from core.inference.video import _video_offload_vram_floor_mib

    plan = _plan("group")
    plan.device_memory = types.SimpleNamespace(backend = backend, is_unified = kind == "unified_memory")
    assert _video_offload_vram_floor_mib(_pipe(), plan) == 3 + 6 + 4


def test_group_offload_defers_the_connector_pin_with_the_background_pins(monkeypatch):
    calls, deferred = _stub_group(monkeypatch, 1 << 20, defer = True)
    pipe = _pipe()
    pipe._unsloth_background_pin_request = True
    monkeypatch.setattr(dm, "_BACKGROUND_PIN_REQUEST_ATTR", "_unsloth_background_pin_request")
    assert dm._apply_group_offload(pipe, "cuda", None)
    assert _hooks(calls)[id(pipe.connectors)][:2] == ("leaf_level", True)
    assert any(m is pipe.connectors for m in deferred)


@pytest.mark.parametrize("why", ["group_switch", "stream_switch", "resident_dit", "no_stream"])
def test_connectors_stay_resident_off_the_streamed_dit_group_tier(monkeypatch, why):
    calls, _ = _stub_group(monkeypatch, 1 << 20)
    kwargs = {}
    if why == "group_switch":
        monkeypatch.setenv(_GROUP_ENV, "0")
    elif why == "stream_switch":
        monkeypatch.setenv(_STREAM_ENV, "0")
    elif why == "resident_dit":
        kwargs = dict(stream_text_encoders = True, stream_transformer = False)
    else:
        # integrated GPU: no copy stream, moving them frees nothing
        monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: False)
    pipe = _pipe()
    dm._apply_group_offload(pipe, "cuda", None, **kwargs)
    assert all(m is not pipe.connectors for m, *_ in calls)
    assert pipe.connectors.moved_to


def test_a_connector_hook_failure_keeps_them_resident(monkeypatch):
    calls, _ = _stub_group(monkeypatch, 1 << 20)
    pipe = _pipe()
    real = dm._apply_group_offloading_packed

    def packed(apply, module, kwargs, logger):
        if module is pipe.connectors:
            raise RuntimeError("refused")
        return real(apply, module, kwargs, logger)

    monkeypatch.setattr(dm, "_apply_group_offloading_packed", packed)
    monkeypatch.setattr(dm, "_remove_group_offload_hooks", lambda module: None)
    assert dm._apply_group_offload(pipe, "cuda", None)
    assert pipe.connectors.moved_to


# ── planner ────────────────────────────────────────────────────────────────────


def _device(free_mib):
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, free_mib)


def _target():
    return types.SimpleNamespace(supports_model_cpu_offload = True, device = "cuda")


def _video_plan(free_mib, streamed = 0):
    # LTX-2.5 shaped: 25.1 GB int8 DiT; companions 10.2 GB (connectors 6.0 of it), worker-held encoder priced 0
    return dm.plan_diffusion_memory(
        target = _target(),
        device_memory = _device(free_mib),
        model_dense_mib = 33_660,
        runtime_headroom_mib = 7_840,
        companion_dense_mib = 9_727,
        text_encoder_dense_mib = 0,
        streamed_companion_mib = streamed,
    )


def test_streamed_companions_leave_only_the_group_floors():
    base = _video_plan(20_000)
    head = _video_plan(20_000, streamed = 6_046)
    assert head.estimates["group_floor_mib"] == base.estimates["group_floor_mib"] - 6_046
    assert (
        head.estimates["group_floor_streamed_te_mib"]
        == base.estimates["group_floor_streamed_te_mib"] - 6_046
    )
    for key in ("model_dense_mib", "companion_dense_mib", "resident_required_mib"):
        assert head.estimates[key] == base.estimates[key]
    assert head.estimates["streamed_companion_mib"] == 6_046
    assert "streamed_companion_mib" not in base.estimates


def test_streamed_companions_admit_the_group_tier_on_a_smaller_card():
    # 16 GiB: the 19.6 GiB group floor with the connectors resident fails; 13.6 GiB without them fits
    assert _video_plan(16_384).offload_policy == dm.OFFLOAD_MODEL
    head = _video_plan(16_384, streamed = 6_046)
    assert head.offload_policy == dm.OFFLOAD_GROUP
    assert head.stream_transformer
    # a card that already took the group tier keeps it
    for streamed in (0, 6_046):
        plan = _video_plan(23_000, streamed = streamed)
        assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_transformer


def test_video_streamed_companion_mib_is_ltx25_on_a_discrete_cuda_card(monkeypatch):
    import core.inference.host_pinning as hp
    from core.inference.video import _video_streamed_companion_mib

    monkeypatch.setattr(hp, "host_pinning_allowed", lambda *a, **k: True)
    cuda = types.SimpleNamespace(device = "cuda", torch_device = "cuda:0")
    ltx25 = types.SimpleNamespace(name = "ltx-2.5")
    assert _video_streamed_companion_mib(ltx25, cuda) == int(6.34e9 / 2**20)
    assert _video_streamed_companion_mib(types.SimpleNamespace(name = "wan2.2-ti2v-5b"), cuda) == 0
    assert _video_streamed_companion_mib(ltx25, types.SimpleNamespace(device = "mps")) == 0
    monkeypatch.setenv(_GROUP_ENV, "0")
    assert _video_streamed_companion_mib(ltx25, cuda) == 0
    monkeypatch.delenv(_GROUP_ENV)
    monkeypatch.setattr(hp, "host_pinning_allowed", lambda *a, **k: False)
    assert _video_streamed_companion_mib(ltx25, cuda) == 0


# ── peak calibration key ───────────────────────────────────────────────────────


def test_calibration_key_separates_streamed_and_resident_connectors(monkeypatch):
    import core.inference.video as video

    pipe = _pipe()
    monkeypatch.setattr(video, "_video_group_hooked", lambda module: True)
    streamed = video._video_calibration_companions(pipe)
    monkeypatch.setattr(video, "_video_group_hooked", lambda module: False)
    resident = video._video_calibration_companions(pipe)
    assert streamed == {"run_once_companions": "streamed"}
    assert resident == {"run_once_companions": "resident"}
    from core.inference.video_stream_residency import calibration_key

    assert calibration_key("ltx-2.5", "r", "group", "cuda:0", **streamed) != calibration_key(
        "ltx-2.5", "r", "group", "cuda:0", **resident
    )


def test_calibration_key_is_unchanged_without_run_once_companions():
    import inspect

    import core.inference.video as video

    pipe = _pipe()
    del pipe.components["connectors"]
    assert video._video_calibration_companions(pipe) == {}
    # the request path keys its peaks with the entry
    assert "**_video_calibration_companions(state.pipe)" in inspect.getsource(video)
