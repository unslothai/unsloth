# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /v1/images/generations pins the adapter set an omitted ``loras`` was resolved from.

A same-model reload keeps repo / base / family, so the identity pin alone admits it. A load thread here
commits such a reload between the route's status() read and the engine's generation slot, as a real
load can, and the request must be refused as replaced and retried with the new load's adapters.
Backend faked (its slot check mirrors DiffusionBackend.generate), so no GPU."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import httpx
import pytest

import core.inference.diffusion as diffusion_module
import core.inference.image_gallery as gallery_module
from core.inference import image_request_batching as batching
from core.inference.diffusion_families import (
    DiffusionModelReplacedError,
    load_identity,
    load_lora_signature,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_openai_images_generations_route import _make_client  # noqa: E402

TAROT = [{"id": "tarot-lora", "weight": 0.8}]
PIXEL = [{"id": "pixel-lora", "weight": 1.0}]


class _ReloadingEngine:
    """Its first status() read starts a load that rebakes after; generate() waits for that load to
    commit before its slot check, which compares pins with the live load."""

    def __init__(
        self,
        before,
        after,
        holder = False,
        auto = True,
    ):
        self.auto = auto
        self.baked = list(before)
        self.after = list(after)
        self.reads = 0
        self.committed = threading.Event()
        self.loader = None
        self.seen = []
        self.refused = 0
        self.refused_batched = 0
        self.rendering = False
        self._holder = object() if holder else None
        self.is_loaded = True

    def status(self):
        self.reads += 1
        out = {
            "loaded": True,
            "repo_id": "x/z-image",
            "base_repo": None,
            "family": "z-image",
            "transformer_quant": "int8",
            "baked_lora_specs": list(self.baked),
            "load_lora_specs": list(self.baked),
            "generation_defaults": {"steps": 8.0, "guidance": 1.0},
        }
        if self.reads == 1 and self.auto:
            self.loader = threading.Thread(target = self._reload, daemon = True)
            self.loader.start()
        return out

    def _reload(self):
        # The same model loaded again with other adapters: LoadIdentity is unchanged.
        self.baked = list(self.after)
        self.committed.set()

    def generation_holder(self):
        if self._holder is None:
            raise AttributeError("no holder")
        return self._holder if self.rendering else None

    def generate(
        self,
        *,
        expected_load = None,
        expected_load_loras = None,
        loras = None,
        **kwargs,
    ):
        assert self.committed.wait(5), "the load never committed"
        loaded = load_identity("x/z-image", None, "z-image")
        if expected_load is not None and expected_load != loaded:
            raise DiffusionModelReplacedError(expected_load, loaded)
        if expected_load_loras is not None and tuple(expected_load_loras) != load_lora_signature(
            self.baked
        ):
            self.refused += 1
            self.refused_batched += kwargs.get("prompts") is not None
            raise DiffusionModelReplacedError(expected_load or loaded, loaded)
        if loras:
            live = {spec["id"] for spec in self.baked}
            stale = [lora_id for lora_id, _ in loras if lora_id not in live]
            if stale:
                # What the engine does with a baked load asked for an adapter it never baked.
                from core.inference.diffusion_lora import LoraSelectionError
                raise LoraSelectionError(f"LoRA {stale[0]} is not baked into this load.")
        self.seen.append(loras)
        n = int(kwargs.get("batch_size") or 1)
        return {"images": [object() for _ in range(n)], "seed": 7, "repo_id": "x/z-image"}


def _client(monkeypatch, engine):
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: engine)
    cli, store, _save = _make_client(engine)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda i: "QUJD" if i in store else None)
    return cli, store


def _post(cli, **body):
    return cli.post(
        "/v1/images/generations",
        json = {"prompt": "trtcrd a sloth", "size": "256x256", "response_format": "b64_json", **body},
    )


def test_a_reload_that_bakes_adapters_retries_with_them(monkeypatch):
    """Resolved to none on a plain load; a rebake with tarot lands before the slot. Unpinned, the engine
    would read the missing selection as none and zero the new bake."""
    engine = _ReloadingEngine(before = [], after = TAROT)
    cli, store = _client(monkeypatch, engine)
    resp = _post(cli, seed = 7)
    assert resp.status_code == 200, resp.text
    assert engine.refused == 1
    assert engine.seen == [[("tarot-lora", 0.8)]]
    (record,) = store.values()
    assert record["loras"] == ["tarot-lora:0.8"]


def test_a_reload_with_other_adapters_retries_with_its_own_set(monkeypatch):
    """Resolved to tarot; the same model is rebaked with pixel. Unpinned, the stale tarot set is refused
    by the engine as a 400 although the request never named an adapter."""
    engine = _ReloadingEngine(before = TAROT, after = PIXEL)
    cli, store = _client(monkeypatch, engine)
    resp = _post(cli, seed = 7)
    assert resp.status_code == 200, resp.text
    assert engine.refused == 1
    assert engine.seen == [[("pixel-lora", 1.0)]]


def test_explicit_loras_are_not_pinned_to_the_load(monkeypatch):
    """A request that names its adapters does not depend on the load's set (same rule as /images/generate)."""
    engine = _ReloadingEngine(before = TAROT, after = TAROT + PIXEL)
    cli, _store = _client(monkeypatch, engine)
    resp = _post(cli, seed = 7, loras = [{"id": "tarot-lora", "weight": 0.5}])
    assert resp.status_code == 200, resp.text
    assert engine.refused == 0
    assert engine.seen == [[("tarot-lora", 0.5)]]


def test_an_unchanged_load_is_not_refused(monkeypatch):
    engine = _ReloadingEngine(before = TAROT, after = TAROT)
    cli, _store = _client(monkeypatch, engine)
    resp = _post(cli, seed = 7)
    assert resp.status_code == 200, resp.text
    assert engine.refused == 0
    assert engine.seen == [[("tarot-lora", 0.8)]]


def test_a_batch_waiting_through_a_reload_carries_the_pin(monkeypatch):
    """Requests queued behind a running generation wait longest, so a reload lands there most often:
    the batched call carries the adapter pin and every member retries with the new set."""
    monkeypatch.setattr(batching, "batch_max", lambda: 8)
    engine = _ReloadingEngine(before = [], after = TAROT, holder = True, auto = False)
    cli, store = _client(monkeypatch, engine)
    app = cli.app

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            engine.rendering = True
            body = {"size": "256x256", "response_format": "b64_json"}
            sent = [
                asyncio.ensure_future(
                    client.post("/v1/images/generations", json = {**body, "prompt": p})
                )
                for p in ("a fox", "a cat")
            ]
            deadline = asyncio.get_running_loop().time() + 5
            while sum(len(b.members) for b in batching._OPEN.values()) < 2:
                assert asyncio.get_running_loop().time() < deadline, "the batch never formed"
                await asyncio.sleep(0.01)
            # Both resolved to none on the plain load; the rebake commits while they wait.
            await asyncio.to_thread(engine._reload)
            engine.rendering = False
            return await asyncio.gather(*sent)

    responses = asyncio.run(run())
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert engine.refused_batched == 1
    assert engine.seen and all(seen == [("tarot-lora", 0.8)] for seen in engine.seen)
    assert sorted(r["loras"] for r in store.values()) == [["tarot-lora:0.8"]] * 2


class _RenderReached(Exception):
    pass


def test_the_real_diffusers_slot_refuses_a_rebake_that_landed_after_the_read(monkeypatch):
    """Same interleave on DiffusionBackend.generate: its slot check decides; the render is stubbed."""
    import types

    engine = diffusion_module.DiffusionBackend()
    pipe = types.SimpleNamespace()
    engine._state = types.SimpleNamespace(
        repo_id = "x/z-image",
        base_repo = None,
        family = types.SimpleNamespace(name = "z-image"),
        pipe = pipe,
        gguf_filename = None,
        gpu_ordinal = None,
        device = "cpu",
    )
    committed = threading.Event()
    reads = []

    def reload_with_tarot():
        # What a quantized load that baked tarot leaves on the pipe.
        pipe._unsloth_loras_baked = True
        diffusion_module._record_baked_lora_specs(pipe, [("tarot-lora", 0.8)])
        committed.set()

    def status():
        reads.append(1)
        specs = diffusion_module._load_lora_specs(pipe)
        out = {
            "loaded": True,
            "repo_id": "x/z-image",
            "base_repo": None,
            "family": "z-image",
            "baked_lora_specs": diffusion_module._baked_lora_specs(pipe),
            "load_lora_specs": specs,
            "generation_defaults": {"steps": 8.0, "guidance": 1.0},
        }
        if len(reads) == 1:
            threading.Thread(target = reload_with_tarot, daemon = True).start()
        return out

    def reached(state):
        raise _RenderReached()

    real_generate = engine.generate
    seen = []

    def generate(**kwargs):
        assert committed.wait(5)
        try:
            real_generate(**kwargs)
        except _RenderReached:
            seen.append(kwargs.get("loras"))
            n = int(kwargs.get("batch_size") or 1)
            return {"images": [object() for _ in range(n)], "seed": 7, "repo_id": "x/z-image"}
        raise AssertionError("the render stub was not reached")

    monkeypatch.setattr(engine, "status", status)
    monkeypatch.setattr(engine, "_state_device_target", reached)
    monkeypatch.setattr(engine, "generate", generate)
    cli, _store = _client(monkeypatch, engine)
    resp = _post(cli, seed = 7)
    assert resp.status_code == 200, resp.text
    # Unpinned, the engine renders the new bake with no selection, which sets tarot's weight to 0.
    assert seen == [[("tarot-lora", 0.8)]]
