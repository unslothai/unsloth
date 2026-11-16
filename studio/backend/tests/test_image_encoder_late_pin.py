# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image load whose streamed text encoders miss the pin budget at plan time reads that budget again when the
pinner starts (after the load has placed the denoiser), instead of streaming the encoder from disk every prompt."""

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm


@pytest.fixture
def budget(monkeypatch):
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.PARTIAL_ENCODER_PIN_ENV,
        dm.ENCODER_BACKGROUND_PIN_ENV,
        dm.LATE_IMAGE_ENCODER_PIN_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    state = {"mib": 0, "trims": 0}
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: state["mib"])

    def trim(logger = None):
        state["trims"] += 1
        return True

    monkeypatch.setattr(dm, "trim_host_heap_after_load", trim)
    return state


def test_the_switches(monkeypatch):
    monkeypatch.delenv(dm.LATE_IMAGE_ENCODER_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.PARTIAL_ENCODER_PIN_ENV, raising = False)
    assert dm._late_image_encoder_pin_enabled(False)
    # the small-host route keeps its memory-mapped bytes out of host RAM
    assert not dm._late_image_encoder_pin_enabled(True)
    monkeypatch.setenv(dm.LATE_IMAGE_ENCODER_PIN_ENV, "0")
    assert not dm._late_image_encoder_pin_enabled(False)
    monkeypatch.delenv(dm.LATE_IMAGE_ENCODER_PIN_ENV)
    monkeypatch.setenv(dm.PARTIAL_ENCODER_PIN_ENV, "0")
    assert not dm._late_image_encoder_pin_enabled(False)


def test_the_cap_is_read_at_pin_start_after_a_heap_trim(budget):
    pipe, module = types.SimpleNamespace(), torch.nn.Module()
    cap = dm._late_image_encoder_pin_cap(pipe, module)
    # plan time: the load still holds the denoiser's host copy
    budget["mib"] = 0
    assert budget["trims"] == 0
    # pin start: the denoiser is on the card (standard Colab T4 klein: ~6 GiB pinnable)
    budget["mib"] = 6000
    assert cap() == 6000 << 20
    assert budget["trims"] == 1
    budget["mib"] = dm._PARTIAL_PIN_MIN_MIB - 1
    assert cap() == 0


@pytest.mark.parametrize("mib", [None, 0])
def test_no_room_at_pin_start_pins_nothing(budget, mib):
    budget["mib"] = mib
    assert dm._late_image_encoder_pin_cap(types.SimpleNamespace(), torch.nn.Module())() == 0


def test_earlier_pinners_outstanding_bytes_come_off_the_cap(budget):
    pipe, first, second = types.SimpleNamespace(), torch.nn.Module(), torch.nn.Module()
    owed = types.SimpleNamespace(module = first, outstanding_bytes = lambda: 1000 << 20)
    setattr(pipe, dm._PENDING_PINS_ATTR, [owed, types.SimpleNamespace(module = second)])
    budget["mib"] = 3000
    assert dm._late_image_encoder_pin_cap(pipe, second)() == 2000 << 20
    assert dm._late_image_encoder_pin_cap(pipe, first)() == 3000 << 20


@pytest.mark.parametrize(
    "env, value",
    [(dm.GROUP_OFFLOAD_PIN_ENV, "0"), (dm.GROUP_OFFLOAD_PIN_ENV, "1")],
)
def test_a_forced_plan_leaves_the_late_cap_empty(budget, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    budget["mib"] = 6000
    assert dm._late_image_encoder_pin_cap(types.SimpleNamespace(), torch.nn.Module())() == 0


def _fake_groups(sizes):
    groups = []
    for n in sizes:
        weight = torch.nn.Parameter(torch.zeros(n, dtype = torch.uint8), requires_grad = False)
        groups.append(types.SimpleNamespace(cpu_param_dict = {weight: weight.data}))
    return groups


def test_a_pinner_with_the_late_cap_pins_the_prefix_free_at_pin_start(budget, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))

    def pin_memory(self):
        out = self.clone()
        out._fake_pin = True
        return out

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    module = torch.nn.Module()
    pipe = types.SimpleNamespace()
    groups = _fake_groups([1 << 20, 1 << 20, 1 << 20, 1 << 20])
    budget["mib"] = 0
    pinner = dm._GroupPinner(
        module, groups, torch.device("cpu"), byte_cap = dm._late_image_encoder_pin_cap(pipe, module)
    )
    setattr(pipe, dm._PENDING_PINS_ATTR, [pinner])
    monkeypatch.setattr(dm, "_PARTIAL_PIN_MIN_MIB", 1)
    budget["mib"] = 3
    pinner._run()
    assert pinner.byte_cap == 3 << 20
    pinned = [getattr(next(iter(g.cpu_param_dict.values())), "_fake_pin", False) for g in groups]
    assert pinned == [True, True, True, False]
    assert all(pinner._done[id(g)].is_set() for g in groups)


_LEFTOVER = dm._streamed_pin_leftover_mib


class _NoMove(torch.nn.Linear):
    def to(self, *args, **kwargs):  # the CPU test host has no card to place the companion on
        return self


def _route(
    monkeypatch,
    *,
    late: bool,
    small_host: bool = False,
    plan = (False, False),
    leftover = 0,
):
    pytest.importorskip("diffusers")
    import diffusers.hooks as hooks

    if not late:
        monkeypatch.setenv(dm.LATE_IMAGE_ENCODER_PIN_ENV, "0")
    applied, deferred = [], []
    import inspect

    def fake_apply(module, **kw):
        applied.append(kw)

    # the route reads the real signature to decide which keywords it may pass
    fake_apply.__signature__ = inspect.signature(hooks.apply_group_offloading)
    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(dm, "_encoder_background_pin_wanted", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: plan)
    monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", lambda *a, **k: leftover)
    monkeypatch.setattr(dm, "install_group_pin_wait", lambda: True)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda module: None)

    def defer(pipe, module, device, logger, **kw):
        deferred.append(kw)
        return True

    monkeypatch.setattr(dm, "_defer_pinning", defer)
    pipe = types.SimpleNamespace()
    pipe.transformer = _NoMove(4, 4)
    pipe.text_encoder = torch.nn.Sequential(torch.nn.Linear(4, 4))
    pipe.components = {"transformer": pipe.transformer, "text_encoder": pipe.text_encoder}
    if small_host:
        pipe._unsloth_small_host = {"components": {}}
    dm.request_background_encoder_pins(pipe)
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    # the plan-time reading only: a late cap reads the live budget again
    monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", _LEFTOVER)
    return applied, deferred


def test_routing_a_plan_time_miss_defers_with_the_late_cap(budget, monkeypatch):
    applied, deferred = _route(monkeypatch, late = True)
    assert [kw.get("low_cpu_mem_usage") for kw in applied] == [True]
    assert len(deferred) == 1 and callable(deferred[0]["byte_cap"])
    assert deferred[0]["fast"] and deferred[0]["lazy_resident"]
    budget["mib"] = 4321
    assert deferred[0]["byte_cap"]() == 4321 << 20


def test_routing_the_kill_switch_keeps_the_plan_time_prefix(budget, monkeypatch):
    # nothing fit at plan time: the encoder streams from its memory-mapped file, as before
    applied, deferred = _route(monkeypatch, late = False)
    assert deferred == [] and [kw.get("low_cpu_mem_usage") for kw in applied] == [True]
    # a plan-time prefix: the fixed cap, as before
    _applied, deferred = _route(monkeypatch, late = False, leftover = 5)
    assert len(deferred) == 1 and deferred[0]["byte_cap"] == 5 << 20


def test_routing_a_forced_pin_off_queues_no_encoder_pinner(budget, monkeypatch):
    # GROUP_OFFLOAD_PIN=0 never pins: the encoder streams from its memory-mapped file with no pinner, as before
    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, "0")
    assert not dm._late_image_encoder_pin_enabled(False)
    applied, deferred = _route(monkeypatch, late = True)
    assert deferred == [] and [kw.get("low_cpu_mem_usage") for kw in applied] == [True]


def test_routing_a_whole_pin_and_the_small_host_route_are_unchanged(budget, monkeypatch):
    _applied, deferred = _route(monkeypatch, late = True, plan = (False, True))
    assert len(deferred) == 1 and deferred[0]["byte_cap"] is None
    applied, deferred = _route(monkeypatch, late = True, small_host = True)
    assert deferred == [] and [kw.get("low_cpu_mem_usage") for kw in applied] == [True]
