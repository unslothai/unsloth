# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 compact conditioning: the worker sends only valid rows, the server rebuilds the connector inputs, and the
two-stage render reuses the first call's connector outputs. Every arm is compared bit for bit with the full-width
path on real (tiny, random) diffusers connectors."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
connectors_mod = pytest.importorskip("diffusers.pipelines.ltx2.connectors")

import inspect  # noqa: E402

if (
    "per_modality_projections"
    not in inspect.signature(connectors_mod.LTX2TextConnectors.__init__).parameters
):
    pytest.skip("diffusers predates the LTX-2.3 connectors", allow_module_level = True)

from core.inference import diffusion_multistage as MS  # noqa: E402
from core.inference import video_ltx25 as v25  # noqa: E402
from core.inference import video_ltx25_te_worker as worker  # noqa: E402

CAPTION, LAYERS = 8, 3


def _connectors(per_modality: bool, registers: int = 16) -> object:
    torch.manual_seed(0)
    module = connectors_mod.LTX2TextConnectors(
        caption_channels = CAPTION,
        text_proj_in_factor = LAYERS,
        video_connector_num_attention_heads = 2,
        video_connector_attention_head_dim = 4,
        video_connector_num_layers = 1,
        video_connector_num_learnable_registers = registers,
        audio_connector_num_attention_heads = 2,
        audio_connector_attention_head_dim = 4,
        audio_connector_num_layers = 1,
        audio_connector_num_learnable_registers = registers,
        per_modality_projections = per_modality,
        # 2.3 / 2.5 project per modality to these widths; 2.0 projects to the caption width (also 2 x 4).
        video_hidden_dim = 8,
        audio_hidden_dim = 8,
    )
    return module.eval()


def _packed(valid: list[int], seq_len: int, dtype) -> tuple:
    """Left-padded packed hidden states whose padded rows are large garbage (what a real encoder leaves there)."""
    g = torch.Generator().manual_seed(1)
    embeds = torch.randn(len(valid), seq_len, CAPTION * LAYERS, generator = g) * 3
    mask = torch.zeros(len(valid), seq_len, dtype = torch.long)
    for b, n in enumerate(valid):
        if n:
            mask[b, seq_len - n :] = 1
        embeds[b, : seq_len - n] = torch.randn(seq_len - n, CAPTION * LAYERS, generator = g) * 1e4
    return embeds.to(dtype), mask


@pytest.mark.parametrize("per_modality", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("valid", [[2, 16], [1, 32], [32, 32], [7, 31]])
def test_compact_rows_give_bit_identical_connector_outputs(per_modality, dtype, valid):
    seq_len = 32
    conn = _connectors(per_modality).to(dtype)
    embeds, mask = _packed(valid, seq_len, dtype)
    rows = worker.compact_valid_rows(embeds, mask)
    assert rows.shape == (sum(valid), CAPTION * LAYERS)
    rebuilt = v25.expand_valid_rows(rows, mask, device = "cpu", dtype = dtype)
    assert rebuilt.shape == embeds.shape
    assert torch.equal(rebuilt[mask.bool()], embeds[mask.bool()])
    assert not rebuilt[~mask.bool()].any()
    with torch.inference_mode():
        old = conn(embeds, mask, padding_side = "left")
        new = conn(rebuilt, mask, padding_side = "left")
    for a, b in zip(old, new):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_shortening_the_sequence_would_not_be_equivalent():
    """Why the rebuild keeps the full length: the registers fill the tail by absolute position."""
    conn = _connectors(True)
    embeds, mask = _packed([2], 32, torch.float32)
    with torch.inference_mode():
        full = conn(embeds, mask)[0]
        short = conn(embeds[:, -16:], mask[:, -16:])[0]
    assert not torch.equal(full[:, :16], short)


class _Model:
    """Stands in for the Gemma encoder: ``LAYERS`` hidden states of width ``CAPTION``."""

    dtype = torch.float32

    def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
        b, l = input_ids.shape
        base = input_ids.float().unsqueeze(-1).expand(b, l, CAPTION)
        return SimpleNamespace(hidden_states = tuple(base * (i + 1) for i in range(LAYERS)))


def test_worker_encode_returns_only_valid_rows():
    inputs = SimpleNamespace(
        input_ids = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4]]),
        attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
    )
    full, mask_full = worker._run_encode(_Model(), {"resident": True}, inputs, "cpu", None)
    rows, mask = worker._run_encode(
        _Model(), {"resident": True, "_compact": True}, inputs, "cpu", None
    )
    assert full.shape == (2, 4, CAPTION * LAYERS) and rows.shape == (6, CAPTION * LAYERS)
    assert torch.equal(mask, mask_full)
    assert torch.equal(
        v25.expand_valid_rows(rows, mask, device = "cpu", dtype = torch.float32)[mask.bool()],
        full[mask.bool()],
    )


def _client(seq_len: int, valid) -> object:
    client = v25.Ltx25TextEncoderClient(
        repo = "r",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/x",
    )
    client._cmd, client._resp = queue.Queue(), queue.Queue()
    client._proc = SimpleNamespace(is_alive = lambda: True, exitcode = None)
    client._ready = True
    client.sent = []
    embeds, mask = _packed(valid if isinstance(valid, list) else [valid], seq_len, torch.float32)

    def fake_put(msg):
        # The worker's reply, built with its own compaction helper.
        client.sent.append(msg)
        out = worker.compact_valid_rows(embeds, mask) if msg.get("compact") else embeds.clone()
        client._resp.put({"type": "embeds", "id": msg["id"], "embeds": out, "mask": mask.clone()})

    client._cmd.put = fake_put
    return client


def _stats() -> dict:
    return dict(v25.CONDITIONING_STATS)


def _delta(before: dict) -> dict:
    return {k: v25.CONDITIONING_STATS[k] - before[k] for k in before}


class _Pipe:
    """The two calls diffusers' LTX-2 pipelines make per stage: ``encode_prompt`` then ``self.connectors``."""

    def __init__(self, conn):
        self._execution_device = torch.device("cpu")
        self.transformer = SimpleNamespace(dtype = torch.float32)
        self.connectors = conn

    def __call__(
        self,
        guidance: bool = False,
        prompt = "a fox",
        videos: int = 1,
    ):
        pe, pm, ne, nm = self.encode_prompt(
            prompt = prompt,
            negative_prompt = "blurry" if isinstance(prompt, str) else ["blurry"] * len(prompt),
            do_classifier_free_guidance = guidance,
            num_videos_per_prompt = videos,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            prompt_attention_mask = None,
            negative_prompt_attention_mask = None,
            max_sequence_length = 32,
            device = self._execution_device,
        )
        if guidance:
            pe, pm = torch.cat([ne, pe]), torch.cat([nm, pm])
        return self.connectors(pe, pm, padding_side = "left")


def _two_stage(
    monkeypatch,
    compact: str,
    guidance: bool = False,
) -> tuple:
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, compact)
    conn = _connectors(True)
    pipe = _Pipe(conn)
    client = _client(32, 5)
    v25.install_remote_encode_prompt(pipe, client)
    before = _stats()
    with torch.inference_mode(), MS.encode_once(pipe), v25.reuse_connector_outputs(pipe):
        first = pipe(guidance)
        second = pipe(guidance)
    # Everything it replaced is back.
    assert "forward" not in conn.__dict__ and pipe.encode_prompt.__name__ == "encode_prompt"
    return first, second, _delta(before), client


def test_two_stage_reuses_connectors_and_matches_the_old_path(monkeypatch):
    old_first, old_second, old_stats, old_client = _two_stage(monkeypatch, "0")
    new_first, new_second, new_stats, new_client = _two_stage(monkeypatch, "1")
    for ref, got in ((old_first, new_first), (old_second, new_second), (old_first, new_second)):
        for a, b in zip(ref, got):
            assert torch.equal(a, b)
    # Kill switch: the old path exactly (full-width transport, a connector run per stage, no counters).
    assert old_stats["full_encodes"] == 1 and old_stats["compact_encodes"] == 0
    assert old_stats["connector_runs"] == 0 and old_stats["connector_reuses"] == 0
    assert all(not m["compact"] for m in old_client.sent)
    # New path engaged: one compact encode, one connector run, one reuse, ~valid/seq_len of the bytes.
    assert new_stats["compact_encodes"] == 1 and new_stats["full_encodes"] == 0
    assert new_stats["connector_runs"] == 1 and new_stats["connector_reuses"] == 1
    assert new_stats["upload_bytes"] < old_stats["upload_bytes"] / 4
    assert new_stats["host_bytes"] < old_stats["host_bytes"] / 4


def test_cfg_concatenation_runs_the_connectors_each_time(monkeypatch):
    old_first, _, _, _ = _two_stage(monkeypatch, "0", guidance = True)
    first, second, stats, _ = _two_stage(monkeypatch, "1", guidance = True)
    assert stats["connector_runs"] == 2 and stats["connector_reuses"] == 0
    for a, b, c in zip(old_first, first, second):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_reuse_restores_an_offload_hook_forward(monkeypatch):
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "1")
    conn = _connectors(True)
    hooked = conn.forward
    conn.forward = lambda *a, **k: hooked(*a, **k)
    marker = conn.forward
    pipe = _Pipe(conn)
    pipe.encode_prompt = lambda **k: None
    with v25.reuse_connector_outputs(pipe):
        assert conn.forward is not marker
    assert conn.forward is marker


def test_kill_switch_makes_reuse_a_no_op(monkeypatch):
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "0")
    conn = _connectors(True)
    pipe = _Pipe(conn)
    encode = pipe.encode_prompt = lambda **k: None
    with v25.reuse_connector_outputs(pipe):
        assert pipe.encode_prompt is encode and "forward" not in conn.__dict__


def test_cache_key_separates_the_transport_forms(monkeypatch):
    client = _client(32, 5)
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "1")
    rows, _ = client.encode(["p"], 32, copy = False)
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "0")
    assert not client.cached(["p"], 32)
    full, _ = client.encode(["p"], 32, copy = False)
    assert rows.ndim == 2 and full.ndim == 3 and len(client.sent) == 2


def test_reuse_survives_an_offload_reset_between_the_calls(monkeypatch):
    """diffusers' end-of-call offload reset re-installs the hooks: a new ``forward`` on the module."""
    monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "1")
    conn = _connectors(True)
    pipe = _Pipe(conn)
    v25.install_remote_encode_prompt(pipe, _client(32, 5))
    before = _stats()
    with torch.inference_mode(), MS.encode_once(pipe), v25.reuse_connector_outputs(pipe):
        first = pipe()
        rearmed = conn.__dict__.pop("forward")
        assert rearmed is not None
        hook = lambda *a, **k: type(conn).forward(conn, *a, **k)  # noqa: E731
        conn.forward = hook
        second = pipe()
    delta = _delta(before)
    assert delta["connector_runs"] == 1 and delta["connector_reuses"] == 1
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert conn.forward is hook


@pytest.mark.parametrize("guidance", [False, True])
def test_batched_prompts_with_several_videos_keep_the_full_width(monkeypatch, guidance):
    """diffusers repeats the embeds per prompt but the mask per batch there, so padded rows can reach the connectors:
    the compact transport steps aside and the outputs stay those of the old path."""
    outs, stats = [], []
    for mode in ("0", "1"):
        monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, mode)
        pipe = _Pipe(_connectors(True))
        v25.install_remote_encode_prompt(pipe, _client(32, [3, 20]))
        before = _stats()
        with torch.inference_mode():
            outs.append(pipe(guidance, prompt = ["a", "b"], videos = 2))
        stats.append(_delta(before))
    assert all(torch.equal(a, b) for a, b in zip(*outs))
    assert stats[1]["full_encodes"] >= 1 and stats[1]["compact_encodes"] == 0
