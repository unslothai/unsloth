# SPDX-License-Identifier: AGPL-3.0-only
"""Presence-penalty parity between the GGUF path and the safetensors/MLX paths.

The safetensors path historically dropped ``presence_penalty``, so the SAME model
looked worse served as safetensors. These tests pin the processor semantics
(subtract once per distinct completion token, prompt excluded, presence not
frequency, zero a no-op, negatives raise) plus a param-propagation regression
over route -> orchestrator cmd -> worker gen_kwargs.
"""

import threading

import pytest
import torch

from core.inference.presence_penalty import (
    apply_presence_penalty,
    _make_presence_penalty_processor,
)


def test_seen_token_gets_exactly_minus_penalty_unseen_unchanged():
    input_ids = torch.tensor([[0, 1, 3]])  # prompt [0, 1], completion [3]
    scores = torch.zeros(1, 5)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.5, prompt_len = 2)
    assert out[0, 3].item() == pytest.approx(-1.5)
    for tok in (0, 1, 2, 4):
        assert out[0, tok].item() == pytest.approx(0.0)


def test_multiplicity_ignored_presence_not_frequency():
    # Token 3 emitted three times -> still a single -penalty (presence, not freq).
    input_ids = torch.tensor([[0, 3, 3, 3]])
    scores = torch.zeros(1, 5)
    out = apply_presence_penalty(input_ids, scores, penalty = 2.0, prompt_len = 1)
    assert out[0, 3].item() == pytest.approx(-2.0)


def test_negative_penalty_raises_seen_logits():
    input_ids = torch.tensor([[0, 2]])
    scores = torch.zeros(1, 4)
    out = apply_presence_penalty(input_ids, scores, penalty = -0.5, prompt_len = 1)
    assert out[0, 2].item() == pytest.approx(0.5)


def test_prompt_tokens_excluded():
    # Token 7 is prompt-only (untouched); token 4 in the completion is penalized.
    input_ids = torch.tensor([[7, 4, 4]])
    scores = torch.zeros(1, 8)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    assert out[0, 7].item() == pytest.approx(0.0)
    assert out[0, 4].item() == pytest.approx(-1.0)


def test_batch_rows_isolated():
    input_ids = torch.tensor([[0, 1], [0, 2]])  # row completions [1] and [2]
    scores = torch.zeros(2, 4)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    assert out[0, 1].item() == pytest.approx(-1.0)
    assert out[0, 2].item() == pytest.approx(0.0)
    assert out[1, 2].item() == pytest.approx(-1.0)
    assert out[1, 1].item() == pytest.approx(0.0)


def test_zero_penalty_is_noop():
    input_ids = torch.tensor([[0, 1, 2]])
    scores = torch.randn(1, 5)
    original = scores.clone()
    out = apply_presence_penalty(input_ids, scores, penalty = 0.0, prompt_len = 1)
    assert torch.equal(out, original)


def test_empty_completion_is_noop():
    # prompt_len covers the whole sequence -> nothing generated yet.
    input_ids = torch.tensor([[0, 1, 2]])
    scores = torch.randn(1, 5)
    original = scores.clone()
    out = apply_presence_penalty(input_ids, scores, penalty = 1.5, prompt_len = 3)
    assert torch.equal(out, original)


def test_out_of_vocab_id_ignored():
    # A generated id >= vocab_size (defensive) must not index out of bounds.
    input_ids = torch.tensor([[0, 9]])
    scores = torch.zeros(1, 5)  # vocab 5, token 9 is out of range
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    assert torch.equal(out, torch.zeros(1, 5))


def test_negative_generated_id_ignored():
    # A negative generated id (defensive) must be dropped, not wrap to scores[-1].
    input_ids = torch.tensor([[0, -1]])
    scores = torch.zeros(1, 5)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    # Nothing penalized; in particular the last row (the numpy/torch wrap target
    # for id -1) is untouched.
    assert torch.equal(out, torch.zeros(1, 5))


def test_mixed_oob_negative_and_valid_ids_only_in_range_penalized():
    # Completion mixes a valid id (1), an out-of-vocab id (9 >= vocab 5) and a
    # negative id (-1). Only the in-range distinct id is penalized; OOB/negative
    # ids are ignored with no crash and no wrong-index wrap. This fails under the
    # old ``seen[seen < vocab_size]`` filter (id -1 wraps to the last row) and
    # passes only with the both-ends bound.
    input_ids = torch.tensor([[0, 1, 9, -1, 1]])  # prompt [0], completion [1, 9, -1, 1]
    scores = torch.zeros(1, 5)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    expected = torch.zeros(1, 5)
    expected[0, 1] = -1.0  # once per distinct in-range id (multiplicity ignored)
    assert torch.equal(out, expected)
    assert out[0, 4].item() == pytest.approx(0.0)  # id -1 did not wrap to the last row


def test_dtype_and_device_preserved():
    input_ids = torch.tensor([[0, 1]])
    scores = torch.zeros(1, 4, dtype = torch.float16)
    out = apply_presence_penalty(input_ids, scores, penalty = 1.0, prompt_len = 1)
    assert out.dtype == torch.float16
    assert out.device == scores.device


def test_processor_none_when_zero():
    assert _make_presence_penalty_processor(0.0, prompt_len = 0) is None


def test_processor_applies_penalty():
    proc = _make_presence_penalty_processor(1.5, prompt_len = 2)
    assert proc is not None
    input_ids = torch.tensor([[0, 1, 3]])
    scores = torch.zeros(1, 5)
    out = proc(input_ids, scores)
    assert out[0, 3].item() == pytest.approx(-1.5)


def test_processor_composes_with_other_processors():
    # LogitsProcessorList must run our processor alongside a pre-existing one.
    from transformers import LogitsProcessor, LogitsProcessorList

    class _AddToTokenZero(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, 0] = scores[:, 0] + 100.0
            return scores

    presence = _make_presence_penalty_processor(1.0, prompt_len = 1)
    combined = LogitsProcessorList([_AddToTokenZero(), *presence])
    input_ids = torch.tensor([[5, 2]])  # completion = [2]
    scores = torch.zeros(1, 6)
    out = combined(input_ids, scores)
    assert out[0, 0].item() == pytest.approx(100.0)  # other processor ran
    assert out[0, 2].item() == pytest.approx(-1.0)  # presence ran


def test_mlx_presence_penalty_callable():
    mx = pytest.importorskip("mlx.core", reason = "MLX only ships on arm64 macOS")
    from core.inference.mlx_inference import _make_mlx_presence_penalty_processor

    proc = _make_mlx_presence_penalty_processor(1.5)
    # First call = prompt only (latches prompt_len, penalizes nothing).
    prompt = mx.array([10, 11])
    logits0 = mx.zeros((1, 20))
    out0 = proc(prompt, logits0)
    assert float(out0[0, 10]) == pytest.approx(0.0)
    # Second call: one completion token (5) appended -> penalized once.
    seq = mx.array([10, 11, 5])
    logits1 = mx.zeros((1, 20))
    out1 = proc(seq, logits1)
    assert float(out1[0, 5]) == pytest.approx(-1.5)
    assert float(out1[0, 10]) == pytest.approx(0.0)  # prompt token untouched


def test_mlx_presence_penalty_bounds_out_of_range_ids():
    # Documents (and, on Apple Silicon CI, enforces) the intended MLX bound:
    # out-of-vocab and negative completion ids must be ignored. MLX does no
    # bounds checking and OOB indexing is undefined behavior (crash / memory
    # corruption), so the processor routes stray ids to a discarded scratch slot
    # and penalizes only in-range distinct ids -- matching the torch filter
    # seen[(seen >= 0) & (seen < vocab)]. Skips off arm64 macOS where MLX is absent.
    mx = pytest.importorskip("mlx.core", reason = "MLX only ships on arm64 macOS")
    from core.inference.mlx_inference import _make_mlx_presence_penalty_processor

    proc = _make_mlx_presence_penalty_processor(1.0)
    proc(mx.array([10, 11]), mx.zeros((1, 8)))  # first call latches prompt_len = 2
    # Completion appends a valid id (3), an out-of-vocab id (99 >= vocab 8) and a
    # negative id (-1); only the in-range id is penalized and nothing crashes.
    seq = mx.array([10, 11, 3, 99, -1])
    out = proc(seq, mx.zeros((1, 8)))
    assert float(out[0, 3]) == pytest.approx(-1.0)
    for tok in range(8):
        if tok != 3:
            assert float(out[0, tok]) == pytest.approx(0.0)


# Param propagation: route payload -> orchestrator cmd -> worker gen_kwargs
_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "presence_penalty": 1.5,
}


def test_orchestrator_cmd_carries_all_sampling_params():
    from core.inference.orchestrator import InferenceOrchestrator

    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    cmd = o._build_generate_cmd(
        "req1",
        None,
        messages = [{"role": "user", "content": "hi"}],
        max_new_tokens = 128,
        **_SAMPLING,
    )
    for key, val in _SAMPLING.items():
        assert cmd[key] == val, f"{key} dropped/altered in orchestrator cmd"


def test_worker_forwards_all_sampling_params_to_backend():
    from core.inference.worker import _handle_generate

    class _RecordingBackend:
        last_generation_stats = None

        def __init__(self):
            self.received = None

        def generate_chat_response(self, **kwargs):
            self.received = kwargs
            return iter(())  # empty stream -> loop exits, gen_done is sent

    class _FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    cmd = {
        "type": "generate",
        "request_id": "r",
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": 128,
        **_SAMPLING,
    }
    backend = _RecordingBackend()
    _handle_generate(backend, cmd, _FakeQueue(), threading.Event())

    assert backend.received is not None
    for key, val in _SAMPLING.items():
        assert backend.received[key] == val, f"{key} dropped/altered in worker gen_kwargs"
