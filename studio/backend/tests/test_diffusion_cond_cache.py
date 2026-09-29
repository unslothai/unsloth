# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the inference-side conditioning cache (``diffusion_cond_cache.py``).

Runs against the real torch/safetensors on CPU with a stub ``encode_prompt`` pipe, so
the wrapper's hit/miss/bypass behaviour and the on-disk reuse (the reason warm repeats
never run the text encoder) are exercised without any model weights."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_cond_cache as cond_cache


class _EncodePipe:
    """A pipe exposing a deterministic ``encode_prompt`` that counts its calls."""

    def __init__(self):
        self.calls = 0
        self._execution_device = "cpu"

    def encode_prompt(
        self,
        prompt,
        device = None,
        num_images_per_prompt = 1,
        max_sequence_length = 256,
        prompt_embeds = None,
    ):
        if prompt_embeds is not None:
            return (prompt_embeds, None)
        self.calls += 1
        value = float(sum(map(ord, str(prompt))))
        return (
            torch.full((num_images_per_prompt, 4), value),
            None,  # mask-less returns must round-trip (None slots)
        )


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", str(tmp_path))
    return tmp_path


def _install(pipe, **overrides):
    kwargs = dict(family = "flux.1", repo_id = "unsloth/repo", dtype = "torch.bfloat16")
    kwargs.update(overrides)
    return cond_cache.install(pipe, **kwargs)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", raising = False)
    pipe = _EncodePipe()
    assert _install(pipe) is False
    assert pipe.encode_prompt.__func__ is _EncodePipe.encode_prompt  # untouched


def test_blank_dir_means_off(monkeypatch):
    # Same semantics as the trainers' cond_cache_dir: blank is "off", not cwd.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", "   ")
    assert cond_cache.cache_dir() is None
    assert _install(_EncodePipe()) is False


def test_repeated_prompt_skips_the_encode_forward(cache_env):
    pipe = _EncodePipe()
    assert _install(pipe) is True
    first = pipe.encode_prompt("a sloth", device = "cpu")
    second = pipe.encode_prompt("a sloth", device = "cpu")
    assert pipe.calls == 1  # warm repeat never ran the text encoder
    assert torch.equal(first[0], second[0])
    assert first[1] is None and second[1] is None  # None slot round-trips
    assert pipe._unsloth_cond_cache_stats == {"hits": 1, "misses": 1}


def test_distinct_prompts_and_arguments_key_separately(cache_env):
    pipe = _EncodePipe()
    _install(pipe)
    pipe.encode_prompt("a sloth")
    pipe.encode_prompt("a fox")
    pipe.encode_prompt("a sloth", num_images_per_prompt = 4)  # shape-changing arg
    assert pipe.calls == 3


def test_device_argument_excluded_from_the_key(cache_env):
    pipe = _EncodePipe()
    _install(pipe)
    pipe.encode_prompt("a sloth", device = "cpu")
    out = pipe.encode_prompt("a sloth", device = torch.device("cpu"))
    assert pipe.calls == 1  # placement detail: still a hit, moved to the target
    assert out[0].device.type == "cpu"


def test_warm_reuse_across_installs(cache_env):
    # A NEW pipe (fresh load) over the same directory hits the persisted entry without encoding: the property that lets warm loads keep the text encoder off GPU.
    first = _EncodePipe()
    _install(first)
    reference = first.encode_prompt("a sloth")
    second = _EncodePipe()
    _install(second)
    warm = second.encode_prompt("a sloth")
    assert second.calls == 0
    assert torch.equal(reference[0], warm[0])


def test_load_fingerprint_keys_apart(cache_env):
    # A different repo / TE quant produces different embeddings: never cross-hit.
    a = _EncodePipe()
    _install(a)
    a.encode_prompt("a sloth")
    b = _EncodePipe()
    _install(b, repo_id = "unsloth/other-repo")
    b.encode_prompt("a sloth")
    c = _EncodePipe()
    _install(c, te_quant = "fp8")
    c.encode_prompt("a sloth")
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


def test_companion_base_keys_apart(cache_env):
    # A GGUF / single-file checkpoint takes its TEXT ENCODERS from the companion base, so the same checkpoint reloaded
    # against a different base must re-encode rather than reuse the previous base's embeddings.
    first = _EncodePipe()
    _install(first, repo_id = "org/model-GGUF", base_repo = "base/one")
    first.encode_prompt("a sloth")
    second = _EncodePipe()
    _install(second, repo_id = "org/model-GGUF", base_repo = "base/two")
    second.encode_prompt("a sloth")
    assert (first.calls, second.calls) == (1, 1)
    # The same base is still a warm hit (the whole point of the cache).
    third = _EncodePipe()
    _install(third, repo_id = "org/model-GGUF", base_repo = "base/one")
    third.encode_prompt("a sloth")
    assert third.calls == 0


def test_a_local_base_updated_in_place_keys_apart(cache_env, tmp_path):
    # A directory path is not a version: editing the text encoder in place must MISS, or the run conditions on the old encoder's embeddings.
    base = tmp_path / "base"
    (base / "text_encoder").mkdir(parents = True)
    weights = base / "text_encoder" / "model.safetensors"
    weights.write_bytes(b"v1")
    first = _EncodePipe()
    _install(first, repo_id = "org/model-GGUF", base_repo = str(base))
    first.encode_prompt("a sloth")
    # Unchanged base -> warm hit (the cache still has to work).
    warm = _EncodePipe()
    _install(warm, repo_id = "org/model-GGUF", base_repo = str(base))
    warm.encode_prompt("a sloth")
    assert (first.calls, warm.calls) == (1, 0)
    # Same path, new contents -> re-encode.
    weights.write_bytes(b"v2-different-length")
    updated = _EncodePipe()
    _install(updated, repo_id = "org/model-GGUF", base_repo = str(base))
    updated.encode_prompt("a sloth")
    assert updated.calls == 1


def test_source_revision_never_raises():
    # Best-effort by contract: a missing path, a bare name and junk all resolve to a marker instead of blocking the load.
    for ref in (None, "", "no/such/repo-xyz", "/does/not/exist", 1234):
        assert isinstance(cond_cache._source_revision(ref), str)


def test_lora_attached_bypasses_the_cache(cache_env):
    pipe = _EncodePipe()
    _install(pipe)
    pipe._unsloth_loras = ("style",)  # adapters may target the text encoders
    pipe.encode_prompt("a sloth")
    pipe.encode_prompt("a sloth")
    assert pipe.calls == 2
    assert pipe._unsloth_cond_cache_stats == {"hits": 0, "misses": 0}


class _ListEncodePipe(_EncodePipe):
    """Returns per-prompt embedding LISTS like Z-Image's ``encode_prompt``."""

    def encode_prompt(
        self,
        prompt,
        device = None,
        do_classifier_free_guidance = True,
    ):
        self.calls += 1
        prompts = prompt if isinstance(prompt, list) else [prompt]
        embeds = [torch.full((1, 4), float(sum(map(ord, p)))) for p in prompts]
        return (embeds, None)


def test_tensor_list_slots_round_trip(cache_env):
    # Z-Image returns list-of-tensors slots, so the flatten/unflatten layout must reproduce them exactly on a warm hit.
    pipe = _ListEncodePipe()
    _install(pipe)
    cold = pipe.encode_prompt(["a", "bb"])
    warm = pipe.encode_prompt(["a", "bb"])
    assert pipe.calls == 1
    assert isinstance(warm[0], list) and len(warm[0]) == 2
    assert all(torch.equal(c, w) for c, w in zip(cold[0], warm[0]))
    assert warm[1] is None


def test_tensor_arguments_pass_through_uncached(cache_env):
    pipe = _EncodePipe()
    _install(pipe)
    supplied = torch.ones(1, 4)
    out = pipe.encode_prompt("a sloth", prompt_embeds = supplied)
    assert out[0] is supplied
    assert pipe.calls == 0
    assert pipe._unsloth_cond_cache_stats == {"hits": 0, "misses": 0}
