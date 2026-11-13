# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: MiniMax-H3 host pin arenas trimmed to payload; the host-looked-up vocab table stays mapped."""

from __future__ import annotations

import inspect
import logging
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_minimax_h3_te as te  # noqa: E402


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_H3_HOST_PIN_TRIM", raising = False)
    monkeypatch.delenv(te.H3_HOST_EMBED_ROWS_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN", raising = False)


def _module(kb):
    torch.manual_seed(0)
    m = torch.nn.Sequential(*[torch.nn.Linear(64, 64, bias = False) for _ in range(kb // 16)])
    return m  # 16 KiB per fp32 64x64 weight


def _pin(m, arena_bytes):
    sizes = []

    def factory(size):
        sizes.append(size)
        return torch.zeros(size, dtype = torch.uint8)

    payload = te.pin_module_in_place(m, arena_bytes = arena_bytes, _arena_factory = factory)
    return payload, sizes


def test_an_arena_holds_what_it_is_given_not_a_whole_arena():
    """Scaled: 304 KiB of weights against 1 MiB arenas (the 0.61 GB audio VAE against 2 GiB ones)."""
    m = _module(304)
    expect = {k: v.clone() for k, v in m.state_dict().items()}
    payload, sizes = _pin(m, 1 << 20)
    assert payload == 304 * 1024
    assert sizes == [304 * 1024]  # page-exact, not 1 MiB
    for k, v in m.state_dict().items():
        assert torch.equal(v, expect[k])


def test_full_arenas_stay_full_and_only_tails_are_trimmed():
    m = _module(2 * 1024 + 48)  # 2 MiB + 48 KiB against 1 MiB arenas
    payload, sizes = _pin(m, 1 << 20)
    assert sizes == [1 << 20, 1 << 20, 48 * 1024]
    assert sum(sizes) == payload


def test_trim_kill_switch_pins_whole_arenas(monkeypatch):
    assert _pin(_module(304), 1 << 20)[1] == [304 * 1024]
    monkeypatch.setenv("UNSLOTH_H3_HOST_PIN_TRIM", "0")
    _payload, sizes = _pin(_module(304), 1 << 20)
    assert sizes == [1 << 20]


def _host(monkeypatch, *, total_gb, available_gb, limit_gib, used_gib):
    import psutil
    from utils import host_memory

    gib = 1 << 30
    monkeypatch.setattr(
        host_memory,
        "cgroup_memory_budgets",
        lambda *a, **k: [((limit_gib - used_gib) * gib, limit_gib * gib)],
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = int(total_gb * 1e9), available = int(available_gb * 1e9)),
    )


def test_pins_stay_inside_the_cgroup_budget_they_were_admitted_against(monkeypatch):
    """Pinned bytes = admitted payload, not arena-rounded (which overran the granted cgroup headroom)."""
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    _host(monkeypatch, total_gb = 94, available_gb = 80, limit_gib = 70, used_gib = 50)
    # min(80 GB available, 20 GiB cgroup headroom) less max(4 GiB, 15% of the 70 GiB limit)
    assert dm._pin_budget_mib() == 20 * 1024 - int(70 * 1024 * 0.15)
    assert te.h3_te_pin_allowed(int(9.0 * 2**30))
    assert not te.h3_te_pin_allowed(int(9.6 * 2**30))
    # a payload one arena-tail over a whole arena: pinned bytes == payload rounded to a page
    m = _module(1024 + 304)
    payload, sizes = _pin(m, 1 << 20)
    assert sum(sizes) == -(-payload // 4096) * 4096


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(512, 32, padding_idx = 0).to(torch.bfloat16)
        self.proj = torch.nn.Linear(32, 32).to(torch.bfloat16)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        return self.proj(self.embed_tokens(input_ids))


class _TE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyLM()


def _stream_pinned(monkeypatch):
    pytest.importorskip("diffusers.hooks")
    import diffusers.hooks as hooks

    import core.inference.diffusion_memory as dm
    import core.inference.diffusion_prequant as prequant
    from core.inference import host_pinning

    real_apply = hooks.apply_group_offloading
    seen = {}

    def fake_apply(
        module,
        onload_device,
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        use_stream = False,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        seen.update(low_cpu_mem_usage = low_cpu_mem_usage, use_stream = use_stream)
        real_apply(
            module,
            onload_device = torch.device("cpu"),
            offload_device = torch.device("cpu"),
            offload_type = "leaf_level",
            use_stream = False,
        )

    assert "low_cpu_mem_usage" in inspect.signature(fake_apply).parameters
    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(host_pinning, "host_pinning_allowed", lambda device = None: True)
    monkeypatch.setattr(prequant, "_unhook_from_manager", lambda *a, **k: True)
    monkeypatch.setattr(prequant, "_evict_rotation_hook", lambda *a, **k: (lambda *x, **y: None))
    monkeypatch.setattr(dm, "install_group_offload_hooks_eager", lambda: False)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda target: None)
    arenas = []

    def arena(size, fill):
        buf = torch.zeros(size, dtype = torch.uint8)
        fill(buf)
        arenas.append(buf)
        return buf

    monkeypatch.setattr(te, "_pinned_arena", arena)
    reference = _TE()
    streamed = _TE()
    original_table = streamed.model.embed_tokens.weight.data_ptr()
    logger = logging.getLogger("h3_pin_trim_test")
    mode = te.stream_h3_text_encoder(object(), streamed, "cuda", pin = True, logger = logger)
    return reference, streamed, mode, seen, arenas, original_table


def _in_arena(t, arenas):
    return t.untyped_storage().data_ptr() in {a.untyped_storage().data_ptr() for a in arenas}


def test_the_vocabulary_table_stays_on_its_mapping_and_lookups_are_identical(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger = "h3_pin_trim_test"):
        reference, streamed, mode, seen, arenas, original = _stream_pinned(monkeypatch)
    assert mode == "stream"
    table = streamed.model.embed_tokens.weight
    assert table.data_ptr() == original and not _in_arena(table, arenas)
    assert _in_arena(streamed.model.proj.weight, arenas)
    assert (
        sum(a.numel() for a in arenas) < table.numel() * table.element_size()
    )  # the table is not pinned
    assert (
        seen["low_cpu_mem_usage"] is True
    )  # diffusers keeps the pinned rest as is, makes no pinned twin
    ids = torch.tensor([[0, 3, 7, 511, 3]])
    with torch.no_grad():
        assert torch.equal(reference.model(ids), streamed.model(ids))
    assert any(m.startswith("video.h3_host_pin_trim: conditioner pins") for m in caplog.messages)


def test_trim_kill_switch_pins_the_table_again(monkeypatch):
    _r, streamed, _m, _s, arenas, _o = _stream_pinned(monkeypatch)
    assert not _in_arena(streamed.model.embed_tokens.weight, arenas)
    monkeypatch.setenv("UNSLOTH_H3_HOST_PIN_TRIM", "0")
    reference, streamed, mode, seen, arenas, original = _stream_pinned(monkeypatch)
    assert mode == "stream"
    assert _in_arena(streamed.model.embed_tokens.weight, arenas)
    assert seen["low_cpu_mem_usage"] is False
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(reference.model(ids), streamed.model(ids))


def test_a_table_left_in_the_stream_is_pinned_after_all(monkeypatch):
    """Host rows off (or refused): the table streams, so it must not stream from pageable memory."""
    monkeypatch.setattr(te, "keep_h3_embedding_on_host", lambda target, logger = None: False)
    reference, streamed, mode, seen, arenas, original = _stream_pinned(monkeypatch)
    table = streamed.model.embed_tokens.weight
    assert seen["low_cpu_mem_usage"] is True  # left out of the up-front pin...
    assert _in_arena(table, arenas)  # ...and pinned once it stayed in the stream
    from core.inference.diffusion_memory import _offload_groups

    hosts = [
        g.cpu_param_dict.get(table)
        for g in _offload_groups(streamed.model)
        if table in (getattr(g, "cpu_param_dict", None) or {})
    ]
    assert all(h is None or h.data_ptr() == table.data_ptr() for h in hosts)
