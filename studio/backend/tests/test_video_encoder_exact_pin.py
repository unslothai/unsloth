# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Page-rounded video pin pricing and the late encoder prefix pin."""

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


def test_a_tied_embedding_is_priced_once_per_offload_group_that_copies_it():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(1024, 1024).to(torch.bfloat16)
            self.embed_tokens = self.shared
            self.encoder = torch.nn.Module()
            self.encoder.embed_tokens = self.shared  # UMT5: one tensor, two owning submodules
            self.proj = torch.nn.Linear(1024, 1024, bias = False).to(torch.bfloat16)

    m = Tied()
    assert dm._module_registered_mib(m) == 4
    assert dm._module_offload_registered_mib(m) == 4  # modules() visits the shared Embedding once
    m.encoder.embed_tokens = torch.nn.Embedding(1024, 1024).to(torch.bfloat16)
    m.encoder.embed_tokens.weight = (
        m.shared.weight
    )  # tied weight, separate module: two leaf groups copy it
    assert dm._module_registered_mib(m) == 4
    assert dm._module_offload_registered_mib(m) == 6


def test_the_kill_switch_and_an_unsupported_registered_pin_turn_the_exact_plan_off(monkeypatch):
    monkeypatch.delenv(_EXACT_ENV, raising = False)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *a, **k: True)
    pipe = types.SimpleNamespace()
    assert not dm._video_exact_pins(pipe)
    dm.request_fast_pins(pipe)
    assert dm._video_exact_pins(pipe)
    monkeypatch.setenv(_EXACT_ENV, "0")
    assert not dm._video_exact_pins(pipe)
    monkeypatch.delenv(_EXACT_ENV)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *a, **k: False)
    assert not dm._video_exact_pins(pipe)


def test_the_late_cap_reads_the_budget_at_pin_start_less_what_earlier_pinners_owe(host):
    mine = torch.nn.Linear(4, 4)
    pipe = types.SimpleNamespace()
    earlier = types.SimpleNamespace(module = torch.nn.Linear(4, 4), outstanding_bytes = lambda: 2 << 20)
    me = types.SimpleNamespace(module = mine, outstanding_bytes = lambda: 99 << 20)
    setattr(pipe, dm._PENDING_PINS_ATTR, [earlier, me])
    cap = dm._late_encoder_pin_cap(pipe, mine)
    host(0)
    assert cap() == 0
    host(10)
    assert cap() == (10 << 20) - (2 << 20)


def test_an_earlier_pinner_owes_its_unplaced_groups_once(monkeypatch):
    """Placed groups already leave the unpinned sum: the cap must not subtract them a second time."""
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))
    groups, srcs = [], []
    for _ in range(2):
        w = torch.nn.Parameter(torch.zeros(4 << 20, dtype = torch.uint8), requires_grad = False)
        src = w.data
        groups.append(types.SimpleNamespace(cpu_param_dict = {w: src}))
        srcs.append(src)
    pinner = dm._GroupPinner(torch.nn.Module(), groups, torch.device("cpu"))
    assert pinner.outstanding_bytes() == 8 << 20
    srcs[0]._fake_pin = True  # the first group is placed
    pinner.pinned = 4 << 20
    assert pinner.outstanding_bytes() == 4 << 20
    capped = dm._GroupPinner(torch.nn.Module(), groups, torch.device("cpu"), byte_cap = 6 << 20)
    capped.pinned = 4 << 20
    assert (
        capped.outstanding_bytes() == 2 << 20
    )  # what the cap has left, not the whole remaining group


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
    assert pinned == [True] * 4 + [False] * 4
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


class _Blocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(8, 8)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(2))

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = x + block(x)
        return x


def _streaming_video_pipe():
    pipe = _Pipe()
    pipe.transformer = _Blocks().to(torch.bfloat16)
    pipe.text_encoder = _encoder()
    pipe.components = {"transformer": pipe.transformer, "text_encoder": pipe.text_encoder}
    dm.request_background_pins(pipe)
    dm.request_fast_pins(pipe)
    return pipe


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_the_streaming_tier_pins_a_video_encoder_that_fits_its_registered_bytes(host, monkeypatch):
    pytest.importorskip("diffusers")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMING_PREFETCH", raising = False)
    x = torch.randn(4, 1500, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _encoder().to("cuda")(x)
    host(20)  # 18 MiB registered + a few KiB of blocks fit; the 32 MiB power-of-two price did not
    pipe = _streaming_video_pipe()
    dm._apply_streaming_offload(pipe, "cuda", None)
    assert dm.start_background_pins(pipe) == 2
    dm.finish_background_pins(pipe)
    assert all(t.is_pinned() for t in _host_copies(pipe.text_encoder))
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(x), ref)
    dm.stop_background_pins(pipe)


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_the_streaming_tier_pins_the_prefix_free_at_pin_start(host, monkeypatch):
    pytest.importorskip("diffusers")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMING_PREFETCH", raising = False)
    x = torch.randn(4, 1500, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _encoder().to("cuda")(x)
    host(2)  # the plan: the blocks fit, the encoder does not
    pipe = _streaming_video_pipe()
    dm._apply_streaming_offload(pipe, "cuda", None)
    assert not any(t.is_pinned() for t in _host_copies(pipe.text_encoder))
    host(10)  # once the load returned: two 4.3 MiB weight groups (and their bias pages) fit
    assert dm.start_background_pins(pipe) == 2
    dm.finish_background_pins(pipe)
    pinned = [t.is_pinned() for t in _host_copies(pipe.text_encoder)]
    assert pinned == [True] * 4 + [False] * 4
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(x), ref)
    dm.stop_background_pins(pipe)


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_the_streaming_tier_kill_switch_pins_no_encoder(host, monkeypatch):
    pytest.importorskip("diffusers")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMING_PREFETCH", raising = False)
    monkeypatch.setenv(_EXACT_ENV, "0")
    host(20)
    pipe = _streaming_video_pipe()
    dm._apply_streaming_offload(pipe, "cuda", None)
    dm.start_background_pins(pipe)
    dm.finish_background_pins(pipe)
    assert not any(t.is_pinned() for t in _host_copies(pipe.text_encoder))
    dm.stop_background_pins(pipe)


class _Writes:
    def __init__(self, finished = False):
        self.finished = finished

    def done(self):
        return self.finished


def _plan_only(monkeypatch, mib):
    """CPU: run the pin plan of a video load with every device-side step stubbed; returns the deferred pins."""
    pytest.importorskip("diffusers.hooks")
    import sys

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
        return None

    monkeypatch.setattr(
        sys.modules["diffusers.hooks"], "apply_group_offloading", apply_group_offloading
    )
    for name in (
        _EXACT_ENV,
        dm.PARTIAL_ENCODER_PIN_ENV,
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.BACKGROUND_PIN_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMING_PREFETCH", raising = False)
    for name, value in (
        ("host_pinning_allowed", True),
        ("_fast_pin_supported", True),
        ("_background_pin_enabled", True),
        ("_streaming_prefetch_enabled", True),
        ("_pinned_memory_capped", False),
        ("_pin_top_level_group", True),
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
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: mib)
    monkeypatch.setattr(
        dm,
        "_defer_pinning",
        lambda pipe, module, device, logger, *, fast = None, byte_cap = None: deferred.append(
            (module, byte_cap)
        ),
    )
    return deferred


def _te_deferrals(pipe, deferred):
    return [cap for module, cap in deferred if module is pipe.text_encoder]


@pytest.mark.parametrize("tier", ["group", "streaming"])
def test_a_first_load_writing_its_cast_cache_pins_no_encoder_prefix(monkeypatch, tier):
    deferred = _plan_only(
        monkeypatch, 10
    )  # 18 MiB registered, 32 MiB power-of-two: neither fits whole
    pipe = _streaming_video_pipe()
    dm.note_cast_cache_writes(pipe, _Writes(finished = False))
    if tier == "group":
        dm._apply_group_offload(
            pipe, "cuda", None, stream_text_encoders = True, stream_transformer = True
        )
    else:
        dm._apply_streaming_offload(pipe, "cuda", None)
    assert _te_deferrals(pipe, deferred) == []  # what main does: the encoder streams unpinned


@pytest.mark.parametrize("tier", ["group", "streaming"])
@pytest.mark.parametrize("writes", [None, "finished"])
def test_a_returning_load_pins_the_encoder_prefix(monkeypatch, tier, writes):
    deferred = _plan_only(monkeypatch, 10)
    pipe = _streaming_video_pipe()
    if writes:
        dm.note_cast_cache_writes(pipe, _Writes(finished = True))
    if tier == "group":
        dm._apply_group_offload(
            pipe, "cuda", None, stream_text_encoders = True, stream_transformer = True
        )
    else:
        dm._apply_streaming_offload(pipe, "cuda", None)
    caps = _te_deferrals(pipe, deferred)
    assert len(caps) == 1 and callable(caps[0])  # the late prefix cap


def test_the_pending_write_gate_reads_done_and_survives_a_broken_record():
    pipe = types.SimpleNamespace()
    assert not dm._cast_cache_write_pending(pipe)
    dm.note_cast_cache_writes(pipe, _Writes(finished = False))
    assert dm._cast_cache_write_pending(pipe)
    dm.note_cast_cache_writes(pipe, _Writes(finished = True))
    assert not dm._cast_cache_write_pending(pipe)
    dm.note_cast_cache_writes(pipe, object())  # no done(): treated as still writing
    assert dm._cast_cache_write_pending(pipe)


def test_the_video_load_records_its_cast_writes_before_the_pin_plan():
    from pathlib import Path

    src = (Path(dm.__file__).parent / "video.py").read_text(encoding = "utf-8")
    deferred = src.index("cast_writes = defer_writes(pipe, cast_missed")
    noted = src.index("note_cast_cache_writes(pipe, cast_writes)")
    planned = src.index("offload_policy, vae_tiling = apply_memory_plan(")
    assert deferred < noted < planned
