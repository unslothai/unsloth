# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The small-host route pins the leading streamed denoiser groups that fit the host RAM free once the load returns,
instead of re-reading every streamed group from its memory-mapped storage on each denoise step."""

import inspect
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm


@pytest.fixture
def budget(monkeypatch):
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.PARTIAL_ENCODER_PIN_ENV,
        dm.LATE_DENOISER_PIN_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda device = None: True)
    state = {"mib": 0, "trims": 0}
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: state["mib"])

    def trim(logger = None):
        state["trims"] += 1
        return True

    monkeypatch.setattr(dm, "trim_host_heap_after_load", trim)
    return state


def test_the_switches(budget, monkeypatch):
    module = torch.nn.Linear(2, 2)
    assert dm._late_denoiser_pin_enabled(module, True)
    # no copy stream: diffusers copies synchronously from the host copy, nothing to pin for
    assert not dm._late_denoiser_pin_enabled(module, False)
    monkeypatch.setenv(dm.LATE_DENOISER_PIN_ENV, "0")
    assert not dm._late_denoiser_pin_enabled(module, True)
    monkeypatch.delenv(dm.LATE_DENOISER_PIN_ENV)
    monkeypatch.setenv(dm.PARTIAL_ENCODER_PIN_ENV, "0")
    assert not dm._late_denoiser_pin_enabled(module, True)
    monkeypatch.delenv(dm.PARTIAL_ENCODER_PIN_ENV)
    monkeypatch.setattr(dm, "_torchao_weight_classes", lambda m: {"Int8Tensor"})
    assert not dm._late_denoiser_pin_enabled(module, True)


def _fake_groups(sizes, resident = ()):
    groups = []
    for i, n in enumerate(sizes):
        weight = torch.nn.Parameter(torch.zeros(n, dtype = torch.uint8), requires_grad = False)
        g = types.SimpleNamespace(cpu_param_dict = {weight: weight.data})
        if i in resident:
            g._unsloth_resident = True
        groups.append(g)
    return groups


@pytest.fixture
def fake_pin(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))

    def pin_memory(self):
        out = self.clone()
        out._fake_pin = True
        return out

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    monkeypatch.setattr(dm, "_PARTIAL_PIN_MIN_MIB", 1)


def _pinned(groups):
    return [getattr(next(iter(g.cpu_param_dict.values())), "_fake_pin", False) for g in groups]


def test_resident_groups_spend_none_of_the_cap(budget, fake_pin):
    module, pipe = torch.nn.Module(), types.SimpleNamespace()
    # two resident groups first (the plan keeps the leading blocks on the card), then four streamed
    groups = _fake_groups([1 << 20] * 6, resident = (0, 1))
    before = [next(iter(g.cpu_param_dict.values())) for g in groups[:2]]
    pinner = dm._GroupPinner(
        module, groups, torch.device("cpu"), byte_cap = dm._late_image_encoder_pin_cap(pipe, module)
    )
    pinner.skip_resident = True
    setattr(pipe, dm._PENDING_PINS_ATTR, [pinner])
    budget["mib"] = 3
    pinner._run()
    assert _pinned(groups) == [False, False, True, True, True, False]
    # the resident groups' host copies are untouched (an eviction streams them again from there)
    assert [next(iter(g.cpu_param_dict.values())) for g in groups[:2]] == before
    assert all(pinner._done[id(g)].is_set() for g in groups)
    assert budget["trims"] == 1


def test_without_skip_resident_the_prefix_starts_at_the_first_group(budget, fake_pin):
    module = torch.nn.Module()
    groups = _fake_groups([1 << 20] * 6, resident = (0, 1))
    pinner = dm._GroupPinner(module, groups, torch.device("cpu"), byte_cap = 3 << 20)
    pinner._run()
    assert _pinned(groups) == [True, True, True, False, False, False]


@pytest.mark.parametrize("mib", [None, 0])
def test_no_room_at_pin_start_pins_nothing(budget, fake_pin, mib):
    module, pipe = torch.nn.Module(), types.SimpleNamespace()
    groups = _fake_groups([1 << 20] * 3)
    pinner = dm._GroupPinner(
        module, groups, torch.device("cpu"), byte_cap = dm._late_image_encoder_pin_cap(pipe, module)
    )
    pinner.skip_resident = True
    budget["mib"] = mib
    pinner._run()
    assert _pinned(groups) == [False, False, False]
    assert all(pinner._done[id(g)].is_set() for g in groups)


class _NoMove(torch.nn.Linear):
    def to(self, *args, **kwargs):  # the CPU test host has no card to place the companion on
        return self


def _route(
    monkeypatch,
    *,
    small_host: bool,
    torchao: bool = False,
    plan = (False, False),
    video: bool = False,
):
    pytest.importorskip("diffusers")
    import diffusers.hooks as hooks

    applied, deferred, waits = [], [], []

    def fake_apply(module, **kw):
        applied.append(kw)

    fake_apply.__signature__ = inspect.signature(hooks.apply_group_offloading)
    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(
        dm, "_apply_group_offloading_packed", lambda apply, module, kw, logger: applied.append(kw)
    )
    monkeypatch.setattr(dm, "_encoder_background_pin_wanted", lambda *a, **k: False)
    monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: plan)
    monkeypatch.setattr(dm, "install_group_pin_wait", lambda: waits.append(1) or True)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda module: None)
    monkeypatch.setattr(dm, "_pin_top_level_group", lambda *a, **k: True)
    monkeypatch.setattr(dm, "install_group_prefetch", lambda *a, **k: 0)
    monkeypatch.setattr(dm, "_keep_groups_resident", lambda *a, **k: 0)
    monkeypatch.setattr(dm, "_background_pin_enabled", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_video_exact_pins", lambda *a, **k: False)
    monkeypatch.setattr(
        dm, "_torchao_weight_classes", lambda m: {"Int8Tensor"} if torchao else set()
    )

    def defer(pipe, module, device, logger, **kw):
        deferred.append((module, kw))
        return True

    monkeypatch.setattr(dm, "_defer_pinning", defer)
    pipe = types.SimpleNamespace()
    pipe.transformer = torch.nn.Sequential(torch.nn.Linear(4, 4))
    pipe.text_encoder = torch.nn.Sequential(torch.nn.Linear(4, 4))
    pipe.vae = _NoMove(4, 4)
    pipe.components = {
        "transformer": pipe.transformer,
        "text_encoder": pipe.text_encoder,
        "vae": pipe.vae,
    }
    if small_host:
        pipe._unsloth_small_host = {"components": {}}
        monkeypatch.setattr(
            "core.inference.diffusion_small_host.install_encoder_prefetch", lambda *a, **k: 0
        )
    if video:
        dm.request_background_pins(pipe)
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, resident_transformer_mib = 100
    )
    return pipe, applied, deferred, waits


def test_routing_the_small_host_route_defers_the_denoiser_with_the_late_cap(budget, monkeypatch):
    pipe, applied, deferred, waits = _route(monkeypatch, small_host = True)
    assert [m for m, _kw in deferred] == [pipe.transformer]
    kw = deferred[0][1]
    assert kw["fast"] and kw["skip_resident"] and callable(kw["byte_cap"])
    budget["mib"] = 2500
    assert kw["byte_cap"]() == 2500 << 20
    assert waits
    assert all(a.get("low_cpu_mem_usage") for a in applied)


@pytest.mark.parametrize(
    "env, value",
    [("UNSLOTH_DIFFUSION_LATE_DENOISER_PIN", "0"), ("UNSLOTH_DIFFUSION_PARTIAL_ENCODER_PIN", "0")],
)
def test_routing_the_kill_switches_keep_the_old_route(budget, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    _pipe, applied, deferred, waits = _route(monkeypatch, small_host = True)
    assert deferred == [] and not waits
    assert all(a.get("low_cpu_mem_usage") for a in applied)


def test_routing_a_plan_time_miss_off_the_small_host_route_defers_too(budget, monkeypatch):
    # a dense FLUX.3 klein on a 12.7 GB T4: 2.9 GB of its transformer streams, the plan pinned nothing
    pipe, _applied, deferred, waits = _route(monkeypatch, small_host = False)
    assert [m for m, _kw in deferred][:1] == [pipe.transformer]
    assert deferred[0][1]["skip_resident"] and callable(deferred[0][1]["byte_cap"]) and waits


def test_routing_whole_pins_video_and_torchao_denoisers_are_unchanged(budget, monkeypatch):
    # the plan pins the denoiser whole (every high-RAM host): no late cap
    pipe, _applied, deferred, _waits = _route(monkeypatch, small_host = False, plan = (True, True))
    assert pipe.transformer not in [m for m, _kw in deferred]
    # video loads pin the denoiser in their own background pinner
    _pipe, _applied, deferred, _waits = _route(monkeypatch, small_host = False, video = True)
    assert all(not kw.get("skip_resident") for _m, kw in deferred)
    _pipe, _applied, deferred, _waits = _route(monkeypatch, small_host = True, torchao = True)
    assert deferred == []


def test_outstanding_bytes_skip_resident_groups(budget, fake_pin):
    # an encoder's late cap subtracts what the denoiser pinner may still place: never its resident groups
    groups = _fake_groups([1 << 20] * 4, resident = (0, 1, 2))
    pinner = dm._GroupPinner(torch.nn.Module(), groups, torch.device("cpu"), byte_cap = lambda: 0)
    assert pinner.outstanding_bytes() == 4 << 20
    pinner.skip_resident = True
    assert pinner.outstanding_bytes() == 1 << 20


def _stream_route(
    monkeypatch,
    *,
    plan = (False, False),
    leftover = 0,
    small_host = False,
    video = False,
):
    pytest.importorskip("diffusers")
    import diffusers.hooks as hooks

    applied, deferred, waits = [], [], []

    def fake_apply(module, **kw):
        applied.append(kw)

    fake_apply.__signature__ = inspect.signature(hooks.apply_group_offloading)
    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(
        dm,
        "_apply_group_offloading_packed",
        lambda apply, module, kw, logger: applied.append((module, kw)),
    )
    monkeypatch.setattr(dm, "_encoder_background_pin_wanted", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_background_pin_enabled", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_video_exact_pins", lambda *a, **k: False)
    monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: plan)
    monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", lambda *a, **k: leftover)
    monkeypatch.setattr(dm, "install_group_pin_wait", lambda: waits.append(1) or True)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda module: None)
    monkeypatch.setattr(dm, "_pin_top_level_group", lambda *a, **k: True)
    monkeypatch.setattr(dm, "install_group_prefetch", lambda *a, **k: 0)
    monkeypatch.setattr(dm, "_keep_groups_resident", lambda *a, **k: 0)

    def defer(pipe, module, device, logger, **kw):
        deferred.append((module, kw))
        return True

    monkeypatch.setattr(dm, "_defer_pinning", defer)
    pipe = types.SimpleNamespace()
    pipe.transformer = torch.nn.Sequential(torch.nn.Linear(4, 4))
    pipe.text_encoder = torch.nn.Sequential(torch.nn.Linear(4, 4))
    pipe.vae = _NoMove(4, 4)
    pipe.components = {
        "transformer": pipe.transformer,
        "text_encoder": pipe.text_encoder,
        "vae": pipe.vae,
    }
    if small_host:
        pipe._unsloth_small_host = {"components": {}}
    if video:
        dm.request_background_pins(pipe)
    dm._apply_streaming_offload(pipe, "cuda", None)
    # the plan-time reading only: a late cap reads the live budget again
    monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", _LEFTOVER)
    return pipe, applied, deferred, waits


_LEFTOVER = dm._streamed_pin_leftover_mib


def test_streaming_route_a_plan_time_miss_pins_late_denoiser_first(budget, monkeypatch):
    pipe, _applied, deferred, waits = _stream_route(monkeypatch)
    assert [m for m, _kw in deferred] == [pipe.transformer, pipe.text_encoder]
    dit, enc = deferred[0][1], deferred[1][1]
    assert dit["skip_resident"] and dit["fast"] and callable(dit["byte_cap"])
    assert callable(enc["byte_cap"]) and enc["fast"]
    budget["mib"] = 2000
    assert dit["byte_cap"]() == 2000 << 20
    assert waits


def test_streaming_route_kill_switches(budget, monkeypatch):
    monkeypatch.setenv(dm.LATE_DENOISER_PIN_ENV, "0")
    monkeypatch.setenv(dm.LATE_IMAGE_ENCODER_PIN_ENV, "0")
    # nothing fit at plan time: both stream from their host copies, as before
    _pipe, applied, deferred, _waits = _stream_route(monkeypatch)
    assert deferred == []
    # a plan-time encoder prefix: the fixed cap, as before
    pipe, _applied, deferred, _waits = _stream_route(monkeypatch, leftover = 5)
    assert [(m, kw["byte_cap"]) for m, kw in deferred] == [(pipe.text_encoder, 5 << 20)]


def test_streaming_route_whole_pins_small_host_and_video_are_unchanged(budget, monkeypatch):
    pipe, _applied, deferred, _waits = _stream_route(monkeypatch, plan = (True, True))
    assert [(m, kw["byte_cap"]) for m, kw in deferred] == [(pipe.text_encoder, None)]
    # small host: the encoder keeps its memory-mapped bytes out of host RAM; the denoiser prefix still pins late
    pipe, _applied, deferred, _waits = _stream_route(monkeypatch, small_host = True)
    assert [m for m, _kw in deferred] == [pipe.transformer]
    # video loads defer their own pins (request_background_pins): no late denoiser cap
    pipe, _applied, deferred, _waits = _stream_route(monkeypatch, video = True)
    assert all(not callable(kw.get("byte_cap")) for _m, kw in deferred)


def test_no_late_denoiser_pin_without_the_registered_pin(budget, monkeypatch):
    # ROCm or UNSLOTH_VIDEO_FAST_PIN=0: the pin_memory() fallback rounds past the page-rounded byte cap
    module = torch.nn.Linear(2, 2)
    seen = []
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda device = None: seen.append(device) or False)
    assert not dm._late_denoiser_pin_enabled(module, True, "cuda:0")
    assert seen == ["cuda:0"]


@pytest.mark.parametrize("committed", [None, 2048 << 20])
def test_the_late_denoiser_cap_reserves_a_later_encoder_pin_the_plan_committed(budget, committed):
    # plan (False, True): the encoder is queued after the denoiser for a whole (or fixed-cap) pin of 2 GiB
    pipe, dit, te = types.SimpleNamespace(), torch.nn.Module(), torch.nn.Module()
    encoder = types.SimpleNamespace(
        module = te, byte_cap = committed, outstanding_bytes = lambda: 2048 << 20
    )
    setattr(pipe, dm._PENDING_PINS_ATTR, [types.SimpleNamespace(module = dit), encoder])
    budget["mib"] = 4096
    assert dm._late_image_encoder_pin_cap(pipe, dit)() == 2048 << 20


def test_a_later_late_capped_pinner_owes_nothing_to_an_earlier_one(budget):
    pipe, dit, te = types.SimpleNamespace(), torch.nn.Module(), torch.nn.Module()
    late = types.SimpleNamespace(
        module = te, byte_cap = lambda: 0, outstanding_bytes = lambda: 2048 << 20
    )
    setattr(pipe, dm._PENDING_PINS_ATTR, [types.SimpleNamespace(module = dit), late])
    budget["mib"] = 4096
    assert dm._late_image_encoder_pin_cap(pipe, dit)() == 4096 << 20
