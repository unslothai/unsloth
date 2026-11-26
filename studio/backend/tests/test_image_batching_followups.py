# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family batch grouping through the route, with requests arriving interleaved: members that encode the same
way per call share one call even when they are not adjacent in the queue, members that do not never share, and
families a batch does not pay off for never wait to coalesce."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx
import pytest

from core.inference import image_request_batching as batching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_concurrency import (  # noqa: E402,F401  (fixture)
    _batched_calls,
    _openai,
    _post_while_rendering,
    studio,
)


def _words(text, **_kwargs):
    return {"input_ids": list(range(len(text.split())))}


class _ClipWords:
    """A CLIP-shaped tokenizer: one id per word, bos / eos."""

    bos_token_id, eos_token_id, pad_token_id = 0, 1, None

    def get_vocab(self):
        return {",</w>": 2}

    def __call__(self, text, **_kwargs):
        return {"input_ids": [10 + len(word) for word in text.split()]}


def _plan(
    backend,
    family,
    pipe,
    offload_policy = "sequential",
):
    """The real engine's coalesce_plan over a stand-in load state; streamed from host unless told otherwise."""
    from core.inference.diffusion import DiffusionBackend

    backend._state = types.SimpleNamespace(
        family = family, pipe = pipe, offload_policy = offload_policy, cuda_graphs = (), speed_optims = ()
    )
    backend.coalesce_plan = types.MethodType(DiffusionBackend.coalesce_plan, backend)
    backend.request_batching = types.MethodType(DiffusionBackend.request_batching, backend)


def _qwen21_pipe():
    transformer = type("QwenImage21Transformer2DModel", (), {})()
    return types.SimpleNamespace(
        transformer = transformer, prompt_template_t2i = "{}", tokenizer = _words
    )


def _groups(backend) -> list[list[str]]:
    return sorted(sorted(call["prompts"]) for call in _batched_calls(backend))


def _alone(backend) -> list[str]:
    return [call["prompt"] for call in backend.calls if call["prompts"] is None]


def _interleaved(*prompts):
    accounts = ("alice", "bob", "carol", "dave", "erin", "frank")
    return [
        ("/v1/images/generations", account, _openai(prompt))
        for account, prompt in zip(accounts, prompts)
    ]


@pytest.mark.parametrize("family_name", ["minimax-h3-image", "qwen-image-2.1"])
def test_excluded_families_render_each_request_at_once_without_opening_a_batch(studio, family_name):
    """MiniMax-H3-Image loops one image at a time, and a Qwen-Image-2.1 batch gains about 3% while drifting about
    35 dB from the solo render, even interleaved A B A B with equal lengths: no request waits to coalesce."""
    app, backend, _saved = studio
    pipe = _qwen21_pipe() if family_name == "qwen-image-2.1" else None
    _plan(backend, types.SimpleNamespace(name = family_name), pipe)
    opened: list[int] = []

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            out = []
            for path, account, body in _interleaved("a red fox", "a red fox in snow", "a blue car"):
                # Finishes while another render still holds the slot: it never waited to join a batch.
                response = await asyncio.wait_for(
                    client.post(path, json = body, headers = {"x-test-account": account}),
                    timeout = 5,
                )
                opened.append(len(batching._OPEN))
                out.append(response)
            backend.rendering = False
            return out

    responses = asyncio.run(run())
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert opened == [0, 0, 0]
    assert _batched_calls(backend) == []
    assert _alone(backend) == ["a red fox", "a red fox in snow", "a blue car"]


def test_flux1_members_share_a_call_only_within_one_t5_bucket_when_interleaved(studio):
    """A solo short prompt pads T5 to 256 and a long one to 512, so a short member next to a long one would
    not match its solo render. Interleaved short / long requests group by bucket."""
    from core.inference.diffusion_text_length import flux_t5_sequence_length

    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer_2 = _words)
    _plan(backend, types.SimpleNamespace(name = "flux.1"), pipe)
    short1, short2 = "a red fox", "a blue car on a road"
    long1, long2 = " ".join(["fox"] * 300), " ".join(["car"] * 400)
    assert (
        flux_t5_sequence_length(_words, [short1])
        == flux_t5_sequence_length(_words, [short2])
        == 256
    )
    assert (
        flux_t5_sequence_length(_words, [long1]) == flux_t5_sequence_length(_words, [long2]) == 512
    )
    responses = _post_while_rendering(
        app, backend, _interleaved(short1, long1, short2, long2), waiting = 4
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([short1, short2]), sorted([long1, long2])]
    assert _alone(backend) == []


def test_sdxl_a_long_prompt_never_shares_a_call_with_short_ones(studio, monkeypatch):
    """Over 75 tokens SDXL encodes two CLIP chunks and the batch would pad every member to two: a long member
    renders with other long ones (or alone), never beside a short one."""
    import core.inference.diffusion_embeddings as embeddings

    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer = _ClipWords(), text_encoder = object())
    _plan(backend, types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"), pipe)
    long_prompt = " ".join(["word"] * 80)
    short1, short2, short3 = "a red fox", "a blue car", "an owl at night"
    responses = _post_while_rendering(
        app, backend, _interleaved(short1, long_prompt, short2, short3), waiting = 4
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([short1, short2, short3])]
    assert _alone(backend) == [long_prompt]


@pytest.mark.parametrize("words", [76, 150])
def test_sdxl_two_long_prompts_with_the_same_chunk_count_still_batch(studio, monkeypatch, words):
    import core.inference.diffusion_embeddings as embeddings

    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer = _ClipWords(), text_encoder = object())
    _plan(backend, types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"), pipe)
    long1, long2 = " ".join(["fox"] * words), " ".join(["car"] * words)
    responses = _post_while_rendering(
        app, backend, _interleaved(long1, "a red fox", long2), waiting = 3
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([long1, long2])]
    assert _alone(backend) == ["a red fox"]


class _FastWords:
    """A Rust-backed tokenizer stand-in: while a render's encode holds it, any other call raises like the real one."""

    is_fast = True

    def __init__(self) -> None:
        self.held = False
        self.calls = 0

    def __call__(self, text, **_kwargs):
        self.calls += 1
        if self.held:
            raise RuntimeError("Already borrowed")
        return {"input_ids": list(range(len(text.split())))}


@pytest.mark.parametrize("family_name", ["flux.1", "flux.1-kontext"])
def test_the_batch_key_never_touches_the_tokenizer_a_render_is_using(family_name):
    """The key is read while another request renders: it reads a private copy made from the generation slot, so
    neither the arriving request nor the running render sees "Already borrowed"."""
    from core.inference.diffusion import coalesce_prompt_key, prime_coalesce_tokenizers

    family = types.SimpleNamespace(name = family_name)
    tok = _FastWords()
    pipe = types.SimpleNamespace(tokenizer_2 = tok)
    prime_coalesce_tokenizers(family, pipe)
    tok.held = True  # the running render is inside its prompt encode
    short, same, longer = "a red fox", "a blue car", " ".join(["fox"] * 300)
    key = coalesce_prompt_key(family, pipe, short)
    assert key is not None
    assert key == coalesce_prompt_key(family, pipe, same)
    assert key != coalesce_prompt_key(family, pipe, longer)
    assert tok.calls == 0


def test_a_fast_tokenizer_without_a_private_copy_renders_alone_without_touching_it():
    from core.inference.diffusion import coalesce_prompt_key

    tok = _FastWords()
    pipe = types.SimpleNamespace(tokenizer_2 = tok)
    assert coalesce_prompt_key(types.SimpleNamespace(name = "flux.1"), pipe, "a fox") is None
    assert tok.calls == 0


def test_loads_whose_keys_never_tokenize_make_no_copy():
    from core.inference import diffusion

    tok = _FastWords()
    pipe = types.SimpleNamespace(tokenizer = tok)
    diffusion.prime_coalesce_tokenizers(types.SimpleNamespace(name = "z-image"), pipe)
    assert tok not in diffusion._KEY_TOKENIZERS


def _serves_at_once(app, backend, prompts):
    """Posts while another render holds the slot; each must finish without waiting to coalesce."""
    opened: list[int] = []

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            out = []
            for path, account, body in _interleaved(*prompts):
                response = await asyncio.wait_for(
                    client.post(path, json = body, headers = {"x-test-account": account}),
                    timeout = 5,
                )
                opened.append(len(batching._OPEN))
                out.append(response)
            backend.rendering = False
            return out

    return asyncio.run(run()), opened


def test_a_resident_load_never_batches_and_never_waits(studio):
    """Resident weights: a shared call measured 0.81x to 1.19x of serial while members drifted from their solos."""
    app, backend, _saved = studio
    _plan(backend, types.SimpleNamespace(name = "z-image"), None, offload_policy = "none")
    responses, opened = _serves_at_once(app, backend, ("a fox", "a cat", "an owl"))
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert opened == [0, 0, 0]
    assert _batched_calls(backend) == []


def test_a_streamed_load_still_batches_across_accounts(studio):
    app, backend, _saved = studio
    _plan(backend, types.SimpleNamespace(name = "z-image"), None, offload_policy = "group")
    responses = _post_while_rendering(
        app, backend, _interleaved("a fox", "a cat", "an owl"), waiting = 3
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted(["a fox", "a cat", "an owl"])]


def test_group_offload_with_every_group_pinned_counts_as_resident(monkeypatch):
    import core.inference.diffusion as diffusion_module
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_memory import (
        DENOISER_PARTIAL,
        DENOISER_RESIDENT,
        DENOISER_STREAMED,
    )

    engine = DiffusionBackend()
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "z-image"),
        pipe = object(),
        offload_policy = "group",
        cuda_graphs = (),
        speed_optims = (),
    )
    placement = [DENOISER_RESIDENT]
    monkeypatch.setattr(diffusion_module, "denoiser_offload", lambda pipe, policy: placement[0])
    policy = engine.request_batching()
    assert policy == {
        "mode": "serial",
        "reason": "resident weights: a shared call is no faster than one request at a time, so concurrent "
        "requests render one at a time",
    }
    assert engine.coalesce_plan("a fox", False) is None
    for streamed in (DENOISER_PARTIAL, DENOISER_STREAMED):
        placement[0] = streamed
        assert engine.request_batching()["mode"] == "any"
        assert engine.coalesce_plan("a fox", False) == ((), False)
    for name in ("model", "sequential", "streaming"):
        engine._state.offload_policy = name
        assert engine.request_batching()["mode"] == "any"
    engine._state.offload_policy = "none"
    assert engine.request_batching()["mode"] == "serial"


def _scaled_engine(transformer):
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "z-image"),
        pipe = types.SimpleNamespace(transformer = transformer),
        offload_policy = "sequential",
        cuda_graphs = (),
        speed_optims = (),
    )
    return engine


def test_a_load_with_per_call_activation_scales_serves_serially():
    """A per-tensor activation scale comes from the whole call's amax: in a shared call one request's prompt would
    set every other member's activation quantization."""
    import torch

    class ComfyNVFP4DynamicLinear(torch.nn.Linear):
        pass

    class NVFP4Tensor(torch.nn.Parameter):
        pass

    dynamic = torch.nn.Sequential(torch.nn.Linear(4, 4), ComfyNVFP4DynamicLinear(4, 4))
    torchao_like = torch.nn.Sequential(torch.nn.Linear(4, 4))
    torchao_like[0].weight = NVFP4Tensor(torch.zeros(4, 4))
    for transformer in (dynamic, torchao_like):
        engine = _scaled_engine(transformer)
        policy = engine.request_batching()
        assert policy["mode"] == "serial"
        assert policy["reason"].startswith("per-tensor dynamic activation scales (transformer.")
        assert engine.coalesce_plan("a fox", False) is None


def test_frozen_or_per_row_activation_scales_still_batch():
    """Studio's NVFP4 linear freezes its activation scale, and FP8 / INT8 scale per row: a batch leaves each
    member's activations as they are alone."""
    import torch

    class NVFP4FlashInferLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("a_gsf", torch.ones(1))

    for transformer in (torch.nn.Sequential(torch.nn.Linear(4, 4)), NVFP4FlashInferLinear()):
        engine = _scaled_engine(transformer)
        assert engine.request_batching()["mode"] == "any"
        assert engine.coalesce_plan("a fox", False) == ((), False)
