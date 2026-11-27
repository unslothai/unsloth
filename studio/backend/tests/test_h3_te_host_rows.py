# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the streamed MiniMax-H3 conditioner's vocabulary table stays on the host (bit-identical lookups)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_minimax_h3_te as te  # noqa: E402


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(50, 8, padding_idx = 0).to(torch.bfloat16)
        self.proj = torch.nn.Linear(8, 8).to(torch.bfloat16)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        return self.proj(self.embed_tokens(input_ids))


def _streamed(install):
    from diffusers.hooks import apply_group_offloading

    m = _TinyLM().requires_grad_(False)
    apply_group_offloading(
        m,
        onload_device = torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "leaf_level",
        use_stream = False,
    )
    installed = te.keep_h3_embedding_on_host(m) if install else False
    return m, installed


def test_host_rows_bit_identical_and_out_of_the_stream(monkeypatch):
    pytest.importorskip("diffusers.hooks")
    monkeypatch.delenv(te.H3_HOST_EMBED_ROWS_ENV, raising = False)
    ref_mod, _ = _streamed(False)
    m, installed = _streamed(True)
    assert installed is True
    assert "group_offloading" not in (m.embed_tokens._diffusers_hook.hooks or {})
    assert "group_offloading" in m.proj._diffusers_hook.hooks  # every other leaf still streams
    ids = torch.tensor([[0, 3, 7, 49, 3]])
    with torch.no_grad():
        assert torch.equal(ref_mod(ids), m(ids))
        assert torch.equal(ref_mod.embed_tokens(ids), m.embed_tokens(ids))
    assert m.embed_tokens.weight.device.type == "cpu"
    assert (
        te.keep_h3_embedding_on_host(m) is False
    )  # already taken out: a second call changes nothing


def test_host_rows_kill_switch_and_refusals(monkeypatch):
    pytest.importorskip("diffusers.hooks")
    monkeypatch.setenv(te.H3_HOST_EMBED_ROWS_ENV, "0")
    m, installed = _streamed(True)
    assert installed is False and "group_offloading" in m.embed_tokens._diffusers_hook.hooks
    monkeypatch.delenv(te.H3_HOST_EMBED_ROWS_ENV)
    # not group offloaded: nothing to take out
    plain = _TinyLM()
    assert te.keep_h3_embedding_on_host(plain) is False
    assert "forward" not in plain.embed_tokens.__dict__


def test_host_rows_under_lazy_prefetch_tracking(monkeypatch):
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import group_offloading as go

    monkeypatch.delenv(te.H3_HOST_EMBED_ROWS_ENV, raising = False)
    from diffusers.hooks import apply_group_offloading

    def build():
        m = _TinyLM().requires_grad_(False)
        apply_group_offloading(
            m,
            onload_device = torch.device("cpu"),
            offload_device = torch.device("cpu"),
            offload_type = "leaf_level",
            use_stream = False,
        )
        # what use_stream=True adds (diffusers' own helper): a root group + a lazy prefetch hook tracking every leaf
        cpu = torch.device("cpu")
        config = go.GroupOffloadingConfig(
            onload_device = cpu,
            offload_device = cpu,
            offload_type = go.GroupOffloadingType.LEAF_LEVEL,
            non_blocking = False,
            record_stream = False,
            low_cpu_mem_usage = False,
        )
        root = go.ModuleGroup(
            modules = [],
            offload_device = cpu,
            onload_device = cpu,
            offload_leader = m,
            onload_leader = m,
            onload_self = True,
            group_id = "lazy_leafs",
        )
        go._apply_lazy_group_offloading_hook(m, root, config = config)
        return m

    ref_mod, m = build(), build()
    assert "layer_execution_tracker" in m.embed_tokens._diffusers_hook.hooks
    assert te.keep_h3_embedding_on_host(m) is True
    assert not m.embed_tokens._diffusers_hook.hooks
    lazy = m._diffusers_hook.hooks["lazy_prefetch_group_offloading"]
    assert "embed_tokens" not in lazy._layer_execution_tracker_module_names
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        for _ in range(2):  # the first forward builds the prefetch chain, the second uses it
            assert torch.equal(ref_mod(ids), m(ids))
