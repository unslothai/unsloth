# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO padded loop must survive a forward that returns real logits.

``_get_per_token_logps_and_entropies`` sets ``UNSLOTH_RETURN_HIDDEN_STATES=1``,
but ``outputs.logits`` only carries hidden states when the forward that ran is
Unsloth's generated one. Any other forward (a plain transformers model, a
wrapper, a model Unsloth did not patch) honours the name and returns a real
``[batch, seq, vocab]`` tensor. Feeding that into the hidden-states helper hits
the ``lm_head`` matmul with a vocab-wide operand and blows up:

    a and b must have same reduction dim, but got [((s47*s87 + 255)//256), s33] X [1536, 151936]

So both arms of the padded loop dispatch on width: hidden states (last dim ==
``lm_head.shape[1]``) go through the fused hidden-states helper, real logits go
straight to the plain log-softmax helper, which skips the ``lm_head`` matmul and
the scale/softcap (the model forward already applied those).

The loop body is lifted out of the live source with ``ast`` and executed here
against a stub model, so the test tracks the shipped code instead of a copy of
it. Everything is tiny and CPU-only; ``unsloth_zoo`` supplies the two helpers
when it is importable, and eager mirrors stand in when it is not.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402

_DISPATCH_HELPERS = load_dispatch_helpers()


# The block under test, lifted structurally out of the live source.

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
_TARGET_FUNCTION = "_get_per_token_logps_and_entropies"
_LOOP_ITERABLE = "zipped_inputs"


def _find_target_function(tree: ast.AST) -> ast.FunctionDef:
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == _TARGET_FUNCTION
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one def {_TARGET_FUNCTION} in {_SOURCE_PATH}, found {len(found)}"
        )
    return found[0]


def _find_padded_loop_with(function: ast.AST) -> ast.With:
    """The ``with`` statement whose direct body holds the ``for ... in zipped_inputs`` loop."""
    found = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Name)
            and stmt.iter.id == _LOOP_ITERABLE
            for stmt in node.body
        )
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one padded-loop `with` inside {_TARGET_FUNCTION}, found {len(found)}"
        )
    return found[0]


def _extract_padded_loop_source() -> str:
    source = _SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(_SOURCE_PATH))
    node = _find_padded_loop_with(_find_target_function(tree))
    segment = ast.get_source_segment(source, node, padded = True)
    if segment is None:
        raise AssertionError("could not recover the padded-loop source segment")
    return textwrap.dedent(segment)


def _free_variables(block_source: str) -> set[str]:
    """Names the block reads without ever binding them itself."""
    tree = ast.parse(block_source)
    loaded, stored = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            else:
                stored.add(node.id)
    return {name for name in loaded - stored if not hasattr(builtins, name)}


_BLOCK_SOURCE = _extract_padded_loop_source()
_BLOCK_CODE = compile(_BLOCK_SOURCE, "<rl_replacements padded loop>", "exec")


# The real unsloth_zoo helpers when available, eager mirrors when not.


def _eager_chunked_hidden_states_selective_log_softmax(
    hidden_states,
    lm_head,
    index,
    chunks = 4,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
    temperature = 1.0,
):
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)
    all_per_token_logps = []
    for chunk_hidden_states, chunk_index in zip(
        torch.chunk(flat_hidden_states, chunks = chunks, dim = 0),
        torch.chunk(flat_index, chunks = chunks, dim = 0),
    ):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()
        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        all_per_token_logps.append(selected - torch.logsumexp(chunk_logits, dim = -1))
    out = torch.concat(all_per_token_logps)
    return out.reshape((hidden_states.shape[0], hidden_states.shape[1]))


def _eager_chunked_selective_log_softmax(
    logits,
    index,
    temperature = 1.0,
    chunks = 4,
):
    all_per_token_logps = []
    for chunk_logits, chunk_index in zip(
        torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = chunks, dim = 0),
        torch.chunk(index.reshape(-1), chunks = chunks, dim = 0),
    ):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        all_per_token_logps.append(selected - torch.logsumexp(chunk_logits, dim = -1))
    out = torch.concat(all_per_token_logps)
    return out.reshape((logits.shape[0], logits.shape[1]))


def _load_helpers():
    """Prefer the shipped helpers; fall back to the eager mirrors above.

    The real ones are ``torch.compile``d, so they are smoke-called once on the
    shapes this file uses before being accepted.
    """
    try:
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax as real_hidden,
            chunked_selective_log_softmax as real_raw,
        )

        probe_hidden = torch.zeros(1, 2, _HIDDEN)
        probe_head = torch.zeros(_VOCAB, _HIDDEN)
        probe_index = torch.zeros(1, 2, dtype = torch.long)
        real_hidden(probe_hidden, probe_head, probe_index, 1, 0.0, 0.0, 0.0, 1.0)
        real_raw(torch.zeros(1, 2, _VOCAB), probe_index, 1.0, 1)
    except Exception:
        return (
            _eager_chunked_hidden_states_selective_log_softmax,
            _eager_chunked_selective_log_softmax,
            "eager mirrors",
        )
    return real_hidden, real_raw, "unsloth_zoo"



_VOCAB = 17
_HIDDEN = 8
_BATCH = 2
_SEQ = 9
_LOGITS_TO_KEEP = 3
_MAX_LEFT_PAD = 2
_MULTIPLIER = 2

_HELPER_HIDDEN, _HELPER_RAW, _HELPER_SOURCE = _load_helpers()


class _StubModel:
    """A forward whose ``.logits`` is either hidden states or real logits.

    ``logits_to_keep`` is honoured because the VLM arm of the loop passes it and
    then slices the returned tensor assuming the forward already trimmed it.
    """

    def __init__(self, embedding, lm_head, returns_hidden_states):
        self.embedding = embedding
        self.lm_head = lm_head
        self.returns_hidden_states = returns_hidden_states
        self.calls = []

    def __call__(
        self,
        input_ids = None,
        logits_to_keep = None,
        **kwargs,
    ):
        hidden = self.embedding[input_ids]
        out = (
            hidden
            if self.returns_hidden_states
            else hidden.to(self.lm_head.dtype) @ self.lm_head.t()
        )
        if logits_to_keep is not None:
            out = out[:, -logits_to_keep:, :]
        self.calls.append({"logits_to_keep": logits_to_keep, "width": out.shape[-1]})
        return SimpleNamespace(logits = out)


def _make_data():
    generator = torch.Generator().manual_seed(1234)
    return SimpleNamespace(
        embedding = torch.randn(_VOCAB, _HIDDEN, generator = generator),
        lm_head = torch.randn(_VOCAB, _HIDDEN, generator = generator),
        input_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator = generator),
        attention_mask = torch.ones(_BATCH, _SEQ, dtype = torch.long),
    )


def _reference_logprobs(
    data,
    *,
    is_vlm,
    temperature = 1.0,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
):
    """Per-row expected result, computed independently with ``torch.log_softmax``."""
    logits = data.embedding[data.input_ids].to(data.lm_head.dtype) @ data.lm_head.t()
    if logit_scale_multiply != 0.0:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0.0:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    logits = logits.to(torch.float32)
    if temperature != 1.0:
        logits = logits / temperature
    width = _LOGITS_TO_KEEP if is_vlm else _LOGITS_TO_KEEP + _MAX_LEFT_PAD
    predictions = torch.log_softmax(logits, dim = -1)[:, -(width + 1) : -1, :]
    targets = data.input_ids[:, -width:]
    return torch.gather(predictions, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)


def _build_namespace(
    data, stub, *, is_vlm, temperature, logit_softcapping, logit_scale_multiply, logit_scale_divide
):
    rows = [
        (
            data.input_ids[i : i + 1],
            data.attention_mask[i : i + 1],
            torch.zeros(1, 3) if is_vlm else None,  # pixel_values_chunk (the stub ignores it) image_grid_thw_chunk
            None,
            None,
            None,
            None,
            None,
        )
        for i in range(_BATCH)
    ]
    return {
        "torch": torch,
        "os": os,
        **_DISPATCH_HELPERS,
        "chunked_hidden_states_selective_log_softmax": _HELPER_HIDDEN,
        "chunked_selective_log_softmax": _HELPER_RAW,
        "device_synchronize": lambda *a, **k: None,
        "_get_inference_mode_context_manager": lambda _model: contextlib.nullcontext(),
        "model": stub,
        "unwrapped_model": stub,
        "self": SimpleNamespace(_autocast_dtype = torch.float32),
        "pixel_values": torch.zeros(1, 3) if is_vlm else None,
        "lm_head": data.lm_head,
        "zipped_inputs": rows,
        "logits_to_keep": _LOGITS_TO_KEEP,
        "max_left_pad": _MAX_LEFT_PAD,
        "multiplier": _MULTIPLIER,
        "logit_scale_multiply": logit_scale_multiply,
        "logit_scale_divide": logit_scale_divide,
        "logit_softcapping": logit_softcapping,
        "temperature": temperature,
        "all_logprobs_list": [],
        "logprobs": None,
    }


def _run_padded_loop(
    *,
    returns_hidden_states,
    is_vlm,
    temperature = 1.0,
    logit_softcapping = 0.0,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
):
    data = _make_data()
    stub = _StubModel(data.embedding, data.lm_head, returns_hidden_states)
    namespace = _build_namespace(
        data,
        stub,
        is_vlm = is_vlm,
        temperature = temperature,
        logit_softcapping = logit_softcapping,
        logit_scale_multiply = logit_scale_multiply,
        logit_scale_divide = logit_scale_divide,
    )
    exec(_BLOCK_CODE, namespace)
    return SimpleNamespace(
        data = data,
        stub = stub,
        logprobs = namespace["logprobs"],
        entropies = namespace["entropies"],
    )




def test_extracted_block_is_the_padded_loop():
    """The lifted segment is the padded loop, located by shape and not by text search."""
    tree = ast.parse(_BLOCK_SOURCE)
    assert len(tree.body) == 1
    with_node = tree.body[0]
    assert isinstance(with_node, ast.With)
    loops = [stmt for stmt in with_node.body if isinstance(stmt, ast.For)]
    assert len(loops) == 1
    assert isinstance(loops[0].iter, ast.Name) and loops[0].iter.id == _LOOP_ITERABLE

    # Both arms of the branch inside the loop must dispatch through the shared helper, which is what compares the width
    # and consults the explicit UNSLOTH_RETURN_HIDDEN_STATES signal.
    dispatch_tests = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "_unsloth_grpo_returns_hidden_states"
    ]
    assert (
        len(dispatch_tests) >= 2
    ), f"expected a hidden-states dispatch in both the text and the VLM arm, found {len(dispatch_tests)}"


def test_block_free_variables_are_all_stubbed():
    """Every name the live block reads is supplied, so the exec cannot silently drift."""
    data = _make_data()
    namespace = _build_namespace(
        data,
        _StubModel(data.embedding, data.lm_head, True),
        is_vlm = False,
        temperature = 1.0,
        logit_softcapping = 0.0,
        logit_scale_multiply = 0.0,
        logit_scale_divide = 0.0,
    )
    missing = sorted(_free_variables(_BLOCK_SOURCE) - set(namespace))
    assert missing == [], f"padded loop reads names this test does not stub: {missing}"


def test_text_branch_with_raw_logits_matches_reference():
    """Regression: a forward returning real logits must not reach the lm_head matmul."""
    result = _run_padded_loop(returns_hidden_states = False, is_vlm = False)
    expected = _reference_logprobs(result.data, is_vlm = False)
    assert result.logprobs.shape == (_BATCH, _LOGITS_TO_KEEP + _MAX_LEFT_PAD)
    assert result.entropies is None
    assert [call["width"] for call in result.stub.calls] == [_VOCAB] * _BATCH
    torch.testing.assert_close(result.logprobs, expected, rtol = 1e-5, atol = 1e-5)


def test_text_branch_with_hidden_states_matches_reference():
    """The unchanged path: hidden states still go through the fused helper."""
    result = _run_padded_loop(returns_hidden_states = True, is_vlm = False)
    expected = _reference_logprobs(result.data, is_vlm = False)
    assert [call["width"] for call in result.stub.calls] == [_HIDDEN] * _BATCH
    torch.testing.assert_close(result.logprobs, expected, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize(
    "returns_hidden_states", [True, False], ids = ["hidden_states", "raw_logits"]
)
def test_vlm_branch_matches_reference(returns_hidden_states):
    """The VLM arm keeps its own slicing and stays correct at both widths."""
    result = _run_padded_loop(returns_hidden_states = returns_hidden_states, is_vlm = True)
    expected = _reference_logprobs(result.data, is_vlm = True)
    assert result.logprobs.shape == (_BATCH, _LOGITS_TO_KEEP)
    assert [call["logits_to_keep"] for call in result.stub.calls] == [_LOGITS_TO_KEEP + 1] * _BATCH
    torch.testing.assert_close(result.logprobs, expected, rtol = 1e-5, atol = 1e-5)


def test_temperature_is_applied_on_both_widths():
    """Temperature is the one transform both helpers apply."""
    expected = _reference_logprobs(_make_data(), is_vlm = False, temperature = 0.7)
    for returns_hidden_states in (True, False):
        result = _run_padded_loop(
            returns_hidden_states = returns_hidden_states,
            is_vlm = False,
            temperature = 0.7,
        )
        torch.testing.assert_close(result.logprobs, expected, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("is_vlm", [False, True], ids = ["text", "vlm"])
def test_raw_logits_skip_scale_and_softcap(is_vlm):
    """Real logits are final: the model forward already scaled and softcapped them.

    Only the hidden-states helper owns those transforms, because only it does the
    lm_head matmul that produces unfinished logits.
    """
    softcapping = 3.0
    data = _make_data()
    with_softcap = _reference_logprobs(data, is_vlm = is_vlm, logit_softcapping = softcapping)
    without_softcap = _reference_logprobs(data, is_vlm = is_vlm, logit_softcapping = 0.0)
    assert not torch.allclose(
        with_softcap, without_softcap, rtol = 1e-3, atol = 1e-3
    ), "the two references are indistinguishable, so this test would prove nothing"

    raw = _run_padded_loop(
        returns_hidden_states = False,
        is_vlm = is_vlm,
        logit_softcapping = softcapping,
    )
    hidden = _run_padded_loop(
        returns_hidden_states = True,
        is_vlm = is_vlm,
        logit_softcapping = softcapping,
    )
    torch.testing.assert_close(raw.logprobs, without_softcap, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(hidden.logprobs, with_softcap, rtol = 1e-5, atol = 1e-5)
