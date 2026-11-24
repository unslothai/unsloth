# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A decode-graph capture that dies between capture_begin and capture_end falls back to eager decode,
and must not leave the CUDA RNG marked as capturing: every later random op (sampling, dropout, vLLM's
sampler in GRPO) would raise "Offset increment outside graph capture encountered unexpectedly"."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import FastLanguageModel
import torch

import unsloth.models._decode_graphs as G


def cuda(test):
    return pytest.mark.gpu(
        pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")(test)
    )


def _load():
    model, tok = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 2048, load_in_4bit = False
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _generate_then_check_rng(
    model,
    tok,
    monkeypatch,
    expect_failure = True,
):
    monkeypatch.setattr(G, "ENABLED", True)
    enc = tok(["Once upon a time there was"], return_tensors = "pt").to("cuda")
    out = model.generate(
        **enc, max_new_tokens = 40, min_new_tokens = 40, do_sample = False,
        pad_token_id = tok.pad_token_id,
    )  # fmt: skip
    assert out.shape[1] == enc["input_ids"].shape[1] + 40
    causal = getattr(getattr(model, "base_model", model), "model", model)
    disabled = G.stats(causal).get("disabled") or G.stats(model).get("disabled")
    if expect_failure:
        assert disabled and disabled.startswith("capture failed"), disabled

    torch.rand(4, device = "cuda")
    torch.cuda.manual_seed(0)
    a = torch.rand(4, device = "cuda")
    torch.cuda.manual_seed(0)
    b = torch.rand(4, device = "cuda")
    assert torch.equal(a, b)
    sampled = model.generate(**enc, max_new_tokens = 8, do_sample = True, pad_token_id = tok.pad_token_id)
    assert sampled.shape[1] <= enc["input_ids"].shape[1] + 8
    return disabled


@cuda
def test_rng_usable_after_inference_mode_rng_state(monkeypatch):
    """vLLM captures its graphs under inference mode first, so the RNG's graph state tensors are
    inference tensors; our capture_begin then fails refilling them (the Qwen3 GRPO notebook case)."""
    model, tok = _load()
    keep = torch.cuda.CUDAGraph()
    x = torch.zeros(1, device = "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        keep.capture_begin()
        x.add_(1)
        keep.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        # Only some torch releases fail this capture (2.11+ makes the RNG's graph state outside
        # inference mode, 2.12+ keeps it per capture); either way the RNG must stay usable.
        disabled = _generate_then_check_rng(model, tok, monkeypatch, expect_failure = False)
        if disabled is not None:
            assert "inference tensor" in disabled, disabled
    finally:
        del keep


@cuda
def test_rng_usable_after_failed_decode_capture(monkeypatch):
    model, tok = _load()

    real_run_step = G._run_step

    def run_step(*args, **kwargs):
        # A host sync is illegal while capturing: it invalidates the capture, so capture_end raises
        # too, which is how a real capture failure ends.
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return real_run_step(*args, **kwargs)

    monkeypatch.setattr(G, "_run_step", run_step)
    _generate_then_check_rng(model, tok, monkeypatch)
