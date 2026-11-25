# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 streamed text encode: one-layer upload lookahead and one denoiser lease per request."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from core.inference import video_ltx25 as v25
from core.inference import video_ltx25_te_worker as worker


def _three_layer_model(width = 8):
    """Gemma 4 layout in miniature: embeddings, three decoder layers (one with a buffer), a norm, a vision tower."""

    class Layer(torch.nn.Module):
        def __init__(self, scaled):
            super().__init__()
            self.proj = torch.nn.Linear(width, width)
            if scaled:
                self.register_buffer("gain", torch.full((width,), 0.75))

        def forward(self, h):
            h = torch.tanh(self.proj(h))
            return h * self.gain if hasattr(self, "gain") else h

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            language = torch.nn.Module()
            language.embed_tokens = torch.nn.Embedding(32, width)
            language.layers = torch.nn.ModuleList(Layer(i == 1) for i in range(3))
            language.norm = torch.nn.LayerNorm(width)
            inner = torch.nn.Module()
            inner.language_model = language
            inner.vision_tower = torch.nn.Linear(width, width)
            self.model = inner
            self.dtype = torch.float32

        def forward(
            self,
            input_ids,
            attention_mask,
            output_hidden_states,
            use_cache = None,
        ):
            language = self.model.language_model
            h = language.embed_tokens(input_ids)
            states = [h]
            for layer in language.layers:
                h = layer(h) * attention_mask[..., None]
                states.append(h)
            states[-1] = language.norm(h)
            return SimpleNamespace(hidden_states = tuple(states))

    return Model().eval()


class _SimUploads:
    """Simulated copy engine: uploads hold NaN until compute waits on their event. Logs every call."""

    def __init__(
        self,
        log,
        land = True,
    ):
        self.log, self.land = log, land
        self.uploads = 0
        self.dones = 0
        self.buffers = []
        self.max_live = 0

    def live_uploads(self):
        return sum(1 for refs in self.buffers if any(r() is not None for r in refs))

    def fence(self, events):
        for event in events:
            # The buffers being released are still held when their reuse fence is set.
            assert self.live_uploads() >= 1
            self.log.append(("fence", event["layer"]))

    def upload(self, sources):
        copies = {
            key: (data.data_ptr(), torch.full_like(data, float("nan"))) for key, data in sources
        }
        self.buffers.append([weakref.ref(dev) for _ptr, dev in copies.values()])
        self.max_live = max(self.max_live, self.live_uploads())
        event = {"layer": self.uploads, "copies": copies, "sources": dict(sources)}
        self.log.append(("upload", self.uploads))
        self.uploads += 1
        return copies, event

    def wait_ready(self, event):
        self.log.append(("wait", event["layer"]))
        if self.land:
            for key, (_ptr, dev) in event["copies"].items():
                dev.copy_(event["sources"][key])

    def done(self):
        self.log.append(("done", self.dones))
        self.dones += 1
        return {"layer": self.dones - 1}

    def drain(self):
        self.log.append(("drain",))


def _inputs():
    ids = torch.randint(0, 32, (1, 5), generator = torch.Generator().manual_seed(1))
    mask = torch.ones(1, 5)
    mask[0, :2] = 0
    return ids, mask


def _reference(model):
    ids, mask = _inputs()
    with torch.inference_mode():
        return worker._pack_hidden_states(model, ids, mask)


def _streamed_encode(
    model,
    monkeypatch,
    sim,
    log = None,
    pinned = True,
):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: pinned)
    monkeypatch.setattr(worker, "_lookahead_ops", lambda device: sim)
    parts = worker._stream_parts(model)
    layers = parts[2]
    ids, mask = _inputs()
    stats = {}
    with worker._streamed(parts, "cpu", stats):
        handles = []
        if log is not None:
            for i, layer in enumerate(layers):
                handles.append(
                    layer.register_forward_pre_hook(lambda m, a, i = i: log.append(("start", i)))
                )
                handles.append(
                    layer.register_forward_hook(lambda m, a, o, i = i: log.append(("end", i)))
                )
        with torch.inference_mode():
            out = worker._pack_hidden_states(model, ids, mask)
        for handle in handles:
            handle.remove()
    return out, stats


def test_lookahead_uploads_the_next_layer_before_this_one_finishes_and_matches(monkeypatch):
    model = _three_layer_model()
    reference = _reference(model)
    host = {id(t): t.data_ptr() for t in list(model.parameters()) + list(model.buffers())}
    log = []
    sim = _SimUploads(log)
    out, stats = _streamed_encode(model, monkeypatch, sim, log)

    assert torch.equal(out, reference)
    assert stats["prefetched_layers"] == 3 and sim.uploads == 3
    at = {entry: k for k, entry in enumerate(log)}
    for i in range(3):
        # compute i is ordered after its own upload landed
        assert at[("upload", i)] < at[("wait", i)] < at[("start", i)]
    for i in range(2):
        # copy i+1 is issued before compute i ends (it runs under compute i)
        assert at[("start", i)] > at[("upload", i + 1)] and at[("upload", i + 1)] < at[("end", i)]
    # layer i's buffers are released only behind a fence on layer i's compute, before the upload that reuses them
    assert at[("done", 0)] < at[("fence", 0)] < at[("upload", 2)]
    assert ("fence", 1) not in at and log[-1] == ("drain",)
    # bounded: never more than two layers' uploads alive
    assert sim.max_live == 2 and sim.live_uploads() == 0
    assert all(
        t.data_ptr() == host[id(t)] for t in list(model.parameters()) + list(model.buffers())
    )
    layers = worker._stream_parts(model)[2]
    assert all(not layer._forward_pre_hooks and not layer._forward_hooks for layer in layers)


def test_negative_control_without_the_readiness_wait_the_encode_reads_unlanded_weights(monkeypatch):
    """Drop the wait (the simulated copy never lands for compute): the parity check must catch it."""
    model = _three_layer_model()
    reference = _reference(model)
    sim = _SimUploads([], land = False)
    out, _stats = _streamed_encode(model, monkeypatch, sim)
    assert not torch.equal(out, reference) and torch.isnan(out).any()


def test_pageable_layers_keep_the_synchronous_upload(monkeypatch):
    model = _three_layer_model()
    reference = _reference(model)
    log = []
    sim = _SimUploads(log)
    out, stats = _streamed_encode(model, monkeypatch, sim, pinned = False)
    assert torch.equal(out, reference)
    assert log == [] and "prefetched_layers" not in stats


def test_kill_switch_restores_the_synchronous_upload_bit_for_bit(monkeypatch):
    model = _three_layer_model()
    reference = _reference(model)
    log = []
    sim = _SimUploads(log)
    gate = worker._lookahead_ops
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(worker, "_CudaUploads", lambda device: sim)
    assert gate("cuda:0") is sim
    monkeypatch.setenv(worker.PREFETCH_ENV, "0")
    assert gate("cuda:0") is None
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(worker, "_lookahead_ops", lambda device: gate("cuda:0"))
    onloads = []
    real_add = worker._Onload.add
    monkeypatch.setattr(
        worker._Onload,
        "add",
        lambda self, modules, recurse = True: onloads.append(self.ready)
        or real_add(self, modules, recurse),
    )
    parts = worker._stream_parts(model)
    ids, mask = _inputs()
    stats = {}
    with worker._streamed(parts, "cpu", stats), torch.inference_mode():
        out = worker._pack_hidden_states(model, ids, mask)
    assert torch.equal(out, reference)
    assert log == [] and "prefetched_layers" not in stats
    # resident + containers, then one plain onload per layer: the pre-lookahead sequence
    assert len(onloads) == 2 + 3 and all(not ready for ready in onloads)


@pytest.mark.parametrize(
    "device, hip, available",
    [
        ("cpu", None, True),
        ("xpu:0", None, True),
        ("mps", None, True),
        ("cuda:0", "6.2", True),
        ("cuda:0", None, False),
    ],
)
def test_lookahead_is_cuda_only(device, hip, available, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(worker, "_CudaUploads", lambda device: object())
    assert worker._lookahead_ops(device) is None


def test_an_out_of_order_layer_falls_back_and_stays_bounded(monkeypatch):
    """A layer that runs out of order uses the plain upload; the stale upload is dropped, never read."""
    model = _three_layer_model()
    language = model.model.language_model
    sim = _SimUploads([])
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    monkeypatch.setattr(worker, "_lookahead_ops", lambda device: sim)
    parts = worker._stream_parts(model)
    x = torch.randn(1, 4, 8, generator = torch.Generator().manual_seed(3))
    with torch.inference_mode():
        expect = language.layers[1](language.layers[2](language.layers[0](x)))
        with worker._streamed(parts, "cpu", {}):
            got = language.layers[1](language.layers[2](language.layers[0](x)))
    assert torch.equal(got, expect) and sim.max_live <= 2


def test_an_out_of_memory_lookahead_retries_on_the_card_before_the_cpu(monkeypatch):
    calls = []

    def run(model, config, inputs, device, stream):
        stats = config.setdefault("_stream_stats", {})
        calls.append((device, stream is not None, bool(stats.get("no_prefetch"))))
        if len(calls) == 1:
            stats["prefetched_layers"] = 2
            raise torch.OutOfMemoryError("CUDA out of memory")
        return torch.zeros(1, 2, 3), torch.ones(1, 2)

    class Tok:
        pad_token, eos_token, padding_side = "p", "e", "left"

        def __call__(self, prompts, **kw):
            return SimpleNamespace(input_ids = torch.ones(1, 2), attention_mask = torch.ones(1, 2))

    monkeypatch.setattr(
        worker, "_encode_plan", lambda config, model, reserved: ("cuda:0", ("parts",))
    )
    monkeypatch.setattr(worker, "_run_encode", run)
    monkeypatch.setattr(worker, "_empty_accelerator_cache", lambda device: None)
    config = {"device": "cuda:0"}
    _e, _m, device = worker._encode_on(Tok(), object(), config, ["a"], 2)
    assert calls == [("cuda:0", True, False), ("cuda:0", True, True)] and device == "cuda:0"
    assert config["_streamed"] is True


class _Client:
    def __init__(
        self,
        events,
        hits = (),
    ):
        self.events, self.hits = events, set(hits)

    def cached(self, texts, max_len):
        return tuple(texts) in self.hits

    def encode(
        self,
        texts,
        max_len,
        copy = True,
    ):
        self.events.append(("encode", tuple(texts)))
        seed = sum(map(ord, "".join(texts)))
        g = torch.Generator().manual_seed(seed)
        return torch.randn(len(texts), max_len, 6, generator = g), torch.ones(len(texts), max_len)


def _pipe():
    return SimpleNamespace(
        _execution_device = torch.device("cpu"), transformer = SimpleNamespace(dtype = torch.bfloat16)
    )


def _cfg_encode(
    monkeypatch,
    events,
    hits = (),
    **kw,
):
    def lend(pipe, client):
        events.append("lend")
        return lambda: events.append("restore")

    monkeypatch.setattr(v25, "lend_resident_dit_for_encode", lend)
    pipe = _pipe()
    client = _Client(events, hits)
    v25.install_remote_encode_prompt(pipe, client)
    return pipe.encode_prompt(prompt = "a fox", negative_prompt = "blurry", max_sequence_length = 4, **kw)


def test_a_cfg_request_lends_the_denoiser_room_once_for_both_misses(monkeypatch):
    events = []
    one = _cfg_encode(monkeypatch, events)
    assert events == ["lend", ("encode", ("a fox",)), ("encode", ("blurry",)), "restore"]
    assert events.count("lend") == 1

    monkeypatch.setenv(v25.ONE_LEASE_ENV, "0")
    events.clear()
    two = _cfg_encode(monkeypatch, events)
    assert events == [
        "lend",
        ("encode", ("a fox",)),
        "restore",
        "lend",
        ("encode", ("blurry",)),
        "restore",
    ]
    assert all(torch.equal(a, b) for a, b in zip(one, two))


def test_one_lease_skips_cache_hits_and_cfg_one(monkeypatch):
    events = []
    _cfg_encode(monkeypatch, events, hits = {("a fox",), ("blurry",)})
    assert events == [("encode", ("a fox",)), ("encode", ("blurry",))]
    events.clear()
    _cfg_encode(monkeypatch, events, hits = {("a fox",)})
    assert events == ["lend", ("encode", ("a fox",)), ("encode", ("blurry",)), "restore"]
    events.clear()
    pe, _pm, ne, _nm = _cfg_encode(monkeypatch, events, do_classifier_free_guidance = False)
    assert events == ["lend", ("encode", ("a fox",)), "restore"] and ne is None


def test_one_lease_stops_a_busy_worker_before_pinning_back(monkeypatch):
    events = []

    def lend(pipe, client):
        events.append("lend")
        return lambda: events.append("restore")

    monkeypatch.setattr(v25, "lend_resident_dit_for_encode", lend)
    pipe = _pipe()
    client = _Client(events)

    def encode(
        texts,
        max_len,
        copy = True,
    ):
        events.append(("encode", tuple(texts)))
        if texts == ["blurry"]:
            raise RuntimeError("cancelled")
        return torch.zeros(1, max_len, 6), torch.ones(1, max_len)

    client.encode = encode
    client.stop_if_busy = lambda: events.append("stop_if_busy")
    v25.install_remote_encode_prompt(pipe, client)
    with pytest.raises(RuntimeError):
        pipe.encode_prompt(prompt = "a fox", negative_prompt = "blurry", max_sequence_length = 4)
    assert events == [
        "lend",
        ("encode", ("a fox",)),
        ("encode", ("blurry",)),
        "stop_if_busy",
        "restore",
    ]


def test_one_lease_refuses_a_mismatched_negative_batch_before_encoding(monkeypatch):
    events = []
    monkeypatch.setattr(v25, "lend_resident_dit_for_encode", lambda pipe, client: None)
    pipe = _pipe()
    v25.install_remote_encode_prompt(pipe, _Client(events))
    with pytest.raises(ValueError):
        pipe.encode_prompt(prompt = ["a", "b"], negative_prompt = ["x"], max_sequence_length = 4)
    assert events == []


def test_an_encode_with_no_pinned_layer_never_creates_the_copy_stream(monkeypatch):
    """The lookahead's stream is made on the first upload, so a pageable host copy costs no stream-pool memory."""
    made = []
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: made.append(1) or object())
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: False)
    monkeypatch.setattr(worker, "_lookahead_ops", lambda device: worker._CudaUploads("cuda:0"))
    model = _three_layer_model()
    reference = _reference(model)
    parts = worker._stream_parts(model)
    ids, mask = _inputs()
    with worker._streamed(parts, "cpu", {}), torch.inference_mode():
        out = worker._pack_hidden_states(model, ids, mask)
    assert torch.equal(out, reference) and made == []
    ops = worker._CudaUploads("cuda:0")
    ops.drain()
    assert made == []
    assert ops.stream is ops.stream and made == [1]
