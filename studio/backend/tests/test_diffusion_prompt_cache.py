# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the in-memory prompt-conditioning cache (``diffusion_prompt_cache.py``), on CPU with
stub pipes: keying, byte-bounded LRU, bypass rules, clone-on-hit, invalidation, and the MiniMax-H3 shim."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_cond_cache as cond_cache
from core.inference import diffusion_prompt_cache as prompt_cache

from .test_diffusion_backend import fake_runtime  # noqa: E402,F401


class _EncodePipe:
    def __init__(self, width = 4):
        self.calls = 0
        self.width = width
        self._execution_device = torch.device("cpu")
        self.text_encoder = torch.nn.Linear(2, 2)

    def encode_prompt(
        self,
        prompt,
        negative_prompt = None,
        device = None,
        num_images_per_prompt = 1,
        max_sequence_length = 256,
        prompt_embeds = None,
        image = None,
        dtype = None,
    ):
        if prompt_embeds is not None:
            return (prompt_embeds, None)
        self.calls += 1
        value = float(sum(map(ord, str(prompt)))) + max_sequence_length
        embeds = torch.full((num_images_per_prompt, self.width), value)
        neg = None if negative_prompt is None else torch.full((1, self.width), -value)
        return (
            embeds,
            torch.ones(num_images_per_prompt, self.width, dtype = torch.bool),
            [neg] if neg is not None else None,
        )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROMPT_CACHE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", raising = False)


def test_repeat_prompt_skips_the_encoder_and_matches_exactly():
    pipe = _EncodePipe()
    assert prompt_cache.install(pipe, identity = {"repo": "r"})
    first = pipe.encode_prompt("a cat", device = "cpu")
    second = pipe.encode_prompt("a cat", device = "cpu")
    assert pipe.calls == 1
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[2] is None and second[2] is None
    assert pipe._unsloth_prompt_cache.describe()["hits"] == 1


def test_every_argument_keys_the_entry():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe.encode_prompt("a dog")
    pipe.encode_prompt("a cat", max_sequence_length = 128)
    pipe.encode_prompt("a cat", num_images_per_prompt = 2)
    pipe.encode_prompt("a cat", negative_prompt = "blurry")
    pipe.encode_prompt("a cat", negative_prompt = "ugly")
    pipe.encode_prompt("a cat", dtype = torch.float16)
    assert pipe.calls == 7
    neg = pipe.encode_prompt("a cat", negative_prompt = "blurry")
    assert pipe.calls == 7 and neg[2][0] is not None


def test_device_is_placement_not_key():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", device = "cpu")
    pipe.encode_prompt("a cat", device = torch.device("cpu"))
    assert pipe.calls == 1


def test_hit_returns_fresh_tensors_so_in_place_edits_cannot_poison():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat")
    reference = first[0].clone()
    first[0].mul_(0)
    second = pipe.encode_prompt("a cat")
    assert torch.equal(second[0], reference)
    second[0].add_(5)
    third = pipe.encode_prompt("a cat")
    assert torch.equal(third[0], reference)
    assert pipe.calls == 1


def test_non_plain_arguments_bypass():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", image = object())
    pipe.encode_prompt("a cat", image = object())
    assert pipe.calls == 2
    embeds = torch.zeros(1, 4)
    out = pipe.encode_prompt("a cat", prompt_embeds = embeds)
    assert out[0] is embeds
    assert pipe._unsloth_prompt_cache.describe()["bypassed"] == 3


def test_lru_is_bounded_by_bytes(monkeypatch):
    # Entry = 1024 B float32 embeds + 256 B bool mask = 1280 B.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", str(3000 / 1024 / 1024))
    pipe = _EncodePipe(width = 256)
    prompt_cache.install(pipe)
    cache = pipe._unsloth_prompt_cache
    pipe.encode_prompt("a")
    pipe.encode_prompt("b")
    assert len(cache) == 2 and cache.bytes == 2560
    pipe.encode_prompt("a")  # refresh a, so b is the eviction victim
    pipe.encode_prompt("c")
    assert len(cache) == 2 and cache.bytes <= cache.budget
    assert cache.describe()["evictions"] == 1
    calls = pipe.calls
    pipe.encode_prompt("a")
    assert pipe.calls == calls
    pipe.encode_prompt("b")
    assert pipe.calls == calls + 1


def test_entry_larger_than_budget_is_not_stored(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", str(100 / 1024 / 1024))
    pipe = _EncodePipe(width = 256)
    prompt_cache.install(pipe)
    pipe.encode_prompt("a")
    pipe.encode_prompt("a")
    assert pipe.calls == 2 and len(pipe._unsloth_prompt_cache) == 0


def test_lora_change_misses_and_text_encoder_swap_misses():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe._unsloth_loras = (("style", "/x/style.safetensors", 0.8),)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 2
    pipe._unsloth_loras = (("style", "/x/style.safetensors", 0.5),)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 3
    pipe._unsloth_loras = ()
    pipe.encode_prompt("a cat")
    assert pipe.calls == 3
    pipe.text_encoder = torch.nn.Linear(2, 2)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 4


def test_lora_owner_is_read_for_workflow_pipes():
    owner = _EncodePipe()
    aux = _EncodePipe()
    prompt_cache.install(aux, lora_owner = owner)
    aux.encode_prompt("a cat")
    owner._unsloth_loras = (("style", "/x", 1.0),)
    aux.encode_prompt("a cat")
    assert aux.calls == 2


def test_release_and_disable(monkeypatch):
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    prompt_cache.release(pipe)
    assert len(pipe._unsloth_prompt_cache) == 0
    pipe.encode_prompt("a cat")
    assert pipe.calls == 2
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE", "0")
    other = _EncodePipe()
    assert not prompt_cache.install(other)
    other.encode_prompt("a")
    other.encode_prompt("a")
    assert other.calls == 2


def test_install_is_idempotent_and_signature_preserved():
    import inspect

    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    wrapped = pipe.encode_prompt
    prompt_cache.install(pipe)
    assert pipe.encode_prompt is wrapped
    assert "max_sequence_length" in inspect.signature(pipe.encode_prompt).parameters


def test_composes_with_the_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", str(tmp_path))
    pipe = _EncodePipe()
    assert cond_cache.install(pipe, family = "fam", repo_id = "r", dtype = torch.float32)
    assert prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe.encode_prompt("a cat")
    assert pipe.calls == 1
    assert pipe._unsloth_cond_cache_stats["misses"] == 1
    assert pipe._unsloth_cond_cache_stats["hits"] == 0  # the memory hit never reached the disk
    fresh = _EncodePipe()
    cond_cache.install(fresh, family = "fam", repo_id = "r", dtype = torch.float32)
    prompt_cache.install(fresh)
    fresh.encode_prompt("a cat")
    assert fresh.calls == 0 and fresh._unsloth_cond_cache_stats["hits"] == 1


def test_h3_shim_caches_text_only_calls(monkeypatch):
    calls = {"n": 0}

    def get_qwen3vl_prompt_embeds(
        text_encoder,
        processor,
        token_ids,
        vision_inputs = None,
        text_encoder_layer = 50,
        device = None,
        dtype = None,
    ):
        calls["n"] += 1
        return torch.full((1, len(token_ids), 3), float(sum(token_ids)))

    module = types.ModuleType("diffusers.modular_pipelines.minimax_h3.encoders")
    module.get_qwen3vl_prompt_embeds = get_qwen3vl_prompt_embeds
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3.encoders", module)

    class _Modular:
        blocks = ()

        def __init__(self):
            self.text_encoder = torch.nn.Linear(2, 2)

    pipe = _Modular()
    assert prompt_cache.install(pipe)
    shim = module.get_qwen3vl_prompt_embeds
    assert shim is not get_qwen3vl_prompt_embeds
    a = shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {},
        text_encoder_layer = 50,
        device = "cpu",
        dtype = torch.float32,
    )
    b = shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {},
        text_encoder_layer = 50,
        device = "cpu",
        dtype = torch.float32,
    )
    assert calls["n"] == 1 and torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    shim(pipe.text_encoder, None, [1, 2, 4], {}, 50, "cpu", torch.float32)
    assert calls["n"] == 2
    # Vision inputs bypass; an unregistered encoder runs the original.
    shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {"pixel_values": torch.zeros(1)},
        50,
        "cpu",
        torch.float32,
    )
    shim(torch.nn.Linear(2, 2), None, [1, 2, 3], {}, 50, "cpu", torch.float32)
    assert calls["n"] == 4
    # A second pipe reuses the shim, no double wrap.
    other = _Modular()
    prompt_cache.install(other)
    assert module.get_qwen3vl_prompt_embeds is shim


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_hit_lands_on_the_source_device():
    class _CudaPipe(_EncodePipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
            **kw,
        ):
            out = super().encode_prompt(prompt, **kw)
            return tuple(o.cuda() if torch.is_tensor(o) else o for o in out)

    pipe = _CudaPipe()
    prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat")
    second = pipe.encode_prompt("a cat")
    assert second[0].is_cuda and torch.equal(first[0], second[0]) and pipe.calls == 1


def test_load_installs_the_prompt_cache_and_unload_releases_it(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as diff_mod

    from .test_diffusion_backend import _loaded_backend

    calls = {"install": [], "release": []}
    monkeypatch.setattr(
        diff_mod.prompt_cache, "install", lambda pipe, **k: calls["install"].append(pipe) or True
    )
    monkeypatch.setattr(
        diff_mod.prompt_cache, "release", lambda pipe: calls["release"].append(pipe)
    )
    backend = _loaded_backend(tmp_path)
    pipe = backend._state.pipe
    assert calls["install"] == [pipe]
    backend.unload()
    assert pipe in calls["release"]


def test_budget_respects_a_cgroup_memory_limit(monkeypatch):
    # Pinned entries count against memory.max, so the budget is RAM/64 of the 2 GiB limit, not the host.
    from core.inference import diffusion_memory

    monkeypatch.delenv(prompt_cache._ENV_BUDGET_MB, raising = False)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 2048)
    assert prompt_cache.budget_bytes() == 32 * 1024 * 1024
    monkeypatch.setenv(prompt_cache._ENV_BUDGET_MB, "100")
    assert prompt_cache.budget_bytes() == 100 * 1024 * 1024
    monkeypatch.delenv(prompt_cache._ENV_BUDGET_MB)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    host = prompt_cache._host_ram_bytes()
    assert prompt_cache.budget_bytes() == min(256 * 1024 * 1024, max(16 * 1024 * 1024, host // 64))


def test_controlnet_pipe_gets_the_prompt_cache(monkeypatch):
    import threading

    from core.inference import diffusion_controlnet as dc
    from core.inference.diffusion import DiffusionBackend

    from . import test_diffusion_controlnet as tcn

    calls = []

    class _EncodingCNPipe(tcn._FakeCNPipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            calls.append(prompt)
            return torch.ones(1, 3, 4)

    fake = tcn._fake_diffusers()
    fake.FluxControlNetPipeline = _EncodingCNPipe
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    tcn._allow_cn_security(monkeypatch)
    backend = DiffusionBackend()
    state = tcn._state()
    backend._state = state
    pipe = backend._controlnet_pipe(
        state, dc.ResolvedControlNet("flux-union-pro", "repo/id", is_local = False), threading.Event()
    )
    assert prompt_cache.cache_for(pipe) is not None
    first = pipe.encode_prompt("a sloth", device = "cpu")
    again = pipe.encode_prompt("a sloth", device = "cpu")
    assert calls == ["a sloth"] and torch.equal(first, again)
