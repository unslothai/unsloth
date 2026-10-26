# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video loads price their deferred registered pins at page-rounded bytes, and a streamed encoder that does not fit
whole pins the prefix that fits the host RAM free once the load has returned."""

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_CUDA = torch.cuda.is_available()
_EXACT_ENV = getattr(dm, "VIDEO_EXACT_PIN_PLAN_ENV", "UNSLOTH_VIDEO_EXACT_PIN_PLAN")


@pytest.fixture
def host(monkeypatch):
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.PARTIAL_ENCODER_PIN_ENV,
        dm.ENCODER_BACKGROUND_PIN_ENV,
        dm.BACKGROUND_PIN_ENV,
        _EXACT_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "_PARTIAL_PIN_MIN_MIB", 1)
    state = {"mib": 0}
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: state["mib"])

    def set_mib(mib):
        state["mib"] = mib

    return set_mib


class _Pipe:
    pass


def _encoder():
    torch.manual_seed(0)
    # 4,500,000 bytes of bf16 weight per Linear: 8 MiB as a power of two, 1099 pages registered
    return torch.nn.Sequential(*[torch.nn.Linear(1500, 1500) for _ in range(4)]).to(torch.bfloat16)


def _video_pipe():
    pipe = _Pipe()
    pipe.transformer = torch.nn.Linear(8, 8).to(torch.bfloat16)
    pipe.text_encoder = _encoder()
    pipe.components = {"transformer": pipe.transformer, "text_encoder": pipe.text_encoder}
    dm.request_background_pins(pipe)
    dm.request_fast_pins(pipe)
    return pipe


def _host_copies(module):
    return [t for g in dm._offload_groups(module) for t in g.cpu_param_dict.values()]


def test_the_two_prices_differ_for_this_encoder():
    enc = _encoder()
    assert dm._module_host_mib(enc) == 32
    assert dm._module_registered_mib(enc) == 18


def test_the_kill_switch_and_an_unsupported_registered_pin_turn_the_exact_plan_off(monkeypatch):
    monkeypatch.delenv(_EXACT_ENV, raising = False)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: True)
    pipe = types.SimpleNamespace()
    assert not dm._video_exact_pins(pipe)
    dm.request_fast_pins(pipe)
    assert dm._video_exact_pins(pipe)
    monkeypatch.setenv(_EXACT_ENV, "0")
    assert not dm._video_exact_pins(pipe)
    monkeypatch.delenv(_EXACT_ENV)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: False)
    assert not dm._video_exact_pins(pipe)


def test_the_late_cap_reads_the_budget_at_pin_start_less_what_earlier_pinners_owe(host):
    mine = torch.nn.Linear(4, 4)
    pipe = types.SimpleNamespace()
    earlier = types.SimpleNamespace(
        module = torch.nn.Linear(4, 4), pinned = 1 << 20, planned_bytes = lambda: 3 << 20
    )
    me = types.SimpleNamespace(module = mine, pinned = 0, planned_bytes = lambda: 99 << 20)
    setattr(pipe, dm._PENDING_PINS_ATTR, [earlier, me])
    cap = dm._late_encoder_pin_cap(pipe, mine)
    host(0)
    assert cap() == 0
    host(10)  # read when called, not when made
    assert cap() == (10 << 20) - (2 << 20)


def test_a_callable_cap_is_read_when_the_pinner_runs(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))

    def pin_memory(self):
        out = self.clone()
        out._fake_pin = True
        return out

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    groups = []
    for n in (100, 5000, 300):
        w = torch.nn.Parameter(torch.zeros(n, dtype = torch.uint8), requires_grad = False)
        groups.append(types.SimpleNamespace(cpu_param_dict = {w: w.data}))
    budget = {"bytes": 0}
    pinner = dm._GroupPinner(
        torch.nn.Module(), groups, torch.device("cpu"), byte_cap = lambda: budget["bytes"]
    )
    budget["bytes"] = 3 * 4096 + 10  # 1 + 2 pages fit, the third group does not
    pinner.start()
    pinner.join(30)
    assert pinner.byte_cap == 3 * 4096 + 10
    assert [all(t.is_pinned() for t in g.cpu_param_dict.values()) for g in groups] == [
        True,
        True,
        False,
    ]


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_a_video_encoder_that_fits_its_registered_bytes_pins_whole(host, monkeypatch):
    pytest.importorskip("diffusers")
    x = torch.randn(4, 1500, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _encoder().to("cuda")(x)
    host(20)  # 18 MiB registered fits; the 32 MiB power-of-two price did not
    pipe = _video_pipe()
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    assert dm.start_background_pins(pipe) == 1
    dm.finish_background_pins(pipe)
    assert all(t.is_pinned() for t in _host_copies(pipe.text_encoder))
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(x), ref)
    dm.stop_background_pins(pipe)


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_a_video_encoder_too_big_at_the_plan_pins_the_prefix_free_at_pin_start(host, monkeypatch):
    pytest.importorskip("diffusers")
    x = torch.randn(4, 1500, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _encoder().to("cuda")(x)
    host(2)  # the plan: nothing fits (the load still holds its source shards)
    pipe = _video_pipe()
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    assert not any(t.is_pinned() for t in _host_copies(pipe.text_encoder))
    host(10)  # the load returned: 10 MiB free, two 4.3 MiB weight groups (and their bias pages) fit
    assert dm.start_background_pins(pipe) == 1
    dm.finish_background_pins(pipe)
    pinned = [t.is_pinned() for t in _host_copies(pipe.text_encoder)]
    assert pinned == [True] * 4 + [False] * 4  # weight + bias of the first two groups, in order
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(x), ref)
    dm.stop_background_pins(pipe)


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_the_kill_switch_keeps_the_power_of_two_price_and_no_prefix(host, monkeypatch):
    pytest.importorskip("diffusers")
    monkeypatch.setenv(_EXACT_ENV, "0")
    host(20)
    pipe = _video_pipe()
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    assert dm.start_background_pins(pipe) == 0
    assert not any(t.is_pinned() for t in _host_copies(pipe.text_encoder))
