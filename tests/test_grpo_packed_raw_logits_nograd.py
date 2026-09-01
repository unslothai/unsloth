# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO packed (no-grad) path needs the width guard on both of its call sites.

`_get_per_token_logps_and_entropies` sets UNSLOTH_RETURN_HIDDEN_STATES=1, but
`.logits` only carries hidden states when the forward is Unsloth's generated
one. Any other forward hands back a real [.., vocab] tensor, and sending that
into the lm_head matmul helper raises on the reduction dim.

Both packed call sites (the flattened forward and the first-use verifier below
it) sit inside one `except Exception`, so the raise is swallowed: the batch is
silently dropped back to the padded loop, `_unsloth_seq_packing_nograd_ok` is
pinned False, and a whole packed forward is wasted every step for the rest of
the run. The symptom is therefore not a crash but packing going away, so the
real block is exec'd here against a stub forward that returns vocab logits and
the assertions read the locals the block itself produced.

The verifier runs on the first packed batch of every run, so without the guard
on that second site the packed raw-logits branch would never be reachable at
all. Runs on CPU with tiny shapes and never skips.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402

_DISPATCH_HELPERS = load_dispatch_helpers()

_REPO = Path(__file__).resolve().parents[1]
_SOURCE = _REPO / "unsloth" / "models" / "rl_replacements.py"

VOCAB, HIDDEN = 17, 8
PAD_ID, SEQ_LEN, KEEP = 0, 8, 4


# The real unsloth_zoo helpers when importable, otherwise eager mirrors with identical semantics so a runner without


# Helpers: the real unsloth_zoo ones when importable, otherwise eager mirrors with identical semantics so a runner
def _fallback_chunked_selective_log_softmax(
    logits,
    index,
    temperature = 1.0,
    chunks = 4,
):
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = chunks, dim = 0)
    chunked_index = torch.chunk(index.reshape(-1), chunks = chunks, dim = 0)
    all_per_token_logps = []
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected_logits = torch.gather(
            chunk_logits,
            dim = -1,
            index = chunk_index.unsqueeze(-1),
        ).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        all_per_token_logps.append(selected_logits - logsumexp_values)
    all_per_token_logps = torch.concat(all_per_token_logps)
    return all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))


def _fallback_chunked_hidden_states_selective_log_softmax(
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
    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks = chunks, dim = 0)
    chunked_index = torch.chunk(index.reshape(-1), chunks = chunks, dim = 0)
    all_per_token_logps = []
    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
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
        selected_logits = torch.gather(
            chunk_logits,
            dim = -1,
            index = chunk_index.unsqueeze(-1),
        ).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        all_per_token_logps.append(selected_logits - logsumexp_values)
    all_per_token_logps = torch.concat(all_per_token_logps)
    return all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))


def _fallback_calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, pad_token_id):
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")
    return (input_ids[:, :-logits_to_keep] == pad_token_id).sum(dim = 1)


def _fallback_create_completion_attention_mask(
    completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, pad_token_id
):
    completion_len = completion_input_ids.shape[1]
    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt
    indices = torch.arange(completion_len, device = completion_input_ids.device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)
    return shift_mask & (completion_input_ids != pad_token_id)


def _resolve_helpers():
    names = (
        "chunked_hidden_states_selective_log_softmax",
        "chunked_selective_log_softmax",
        "create_completion_attention_mask",
        "calculate_pad_tokens_in_prompt",
    )
    fallbacks = {
        "chunked_hidden_states_selective_log_softmax": _fallback_chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": _fallback_chunked_selective_log_softmax,
        "create_completion_attention_mask": _fallback_create_completion_attention_mask,
        "calculate_pad_tokens_in_prompt": _fallback_calculate_pad_tokens_in_prompt,
    }
    try:
        import unsloth_zoo.rl_replacements as zoo
    except Exception:
        zoo = None
    return {name: getattr(zoo, name, None) or fallbacks[name] for name in names}


HELPERS = _resolve_helpers()
_completion_mask_of = HELPERS["create_completion_attention_mask"]
_left_pad_of = HELPERS["calculate_pad_tokens_in_prompt"]


class _Model(torch.nn.Module):
    """`hidden_states = False` ignores UNSLOTH_RETURN_HIDDEN_STATES and returns
    real [.., vocab] logits; True is Unsloth's generated forward.

    Position-local, so the packed block-diagonal forward and the per-row
    forward agree exactly and the verifier's own tolerance is not what is under
    test. Every call is recorded so a test can pin which sites actually ran.
    """

    def __init__(
        self,
        hidden_states = False,
        vocab = VOCAB,
        hidden = HIDDEN,
        degraded = False,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias = False)
        self.hidden_states = hidden_states
        self.calls = []
        if degraded:
            # What _install_grpo_hidden_states_forward_wrapper leaves behind when it could not get hidden states out of
            # what _install_grpo_hidden_states_forward_wrapper in unsloth/models/rl.py leaves behind when it could not
            self._unsloth_grpo_hidden_states_forward_wrapped = True
            self._unsloth_grpo_hidden_states_warning_issued = True

    def forward(
        self,
        input_ids = None,
        position_ids = None,
        attention_mask = None,
        packed_seq_lengths = None,
        use_cache = None,
        **kwargs,
    ):
        self.calls.append(
            SimpleNamespace(
                shape = tuple(input_ids.shape),
                packed = packed_seq_lengths is not None,
            )
        )
        h = torch.tanh(self.emb(input_ids))
        return SimpleNamespace(logits = h if self.hidden_states else self.head(h))


def _statement_lists(node):
    for child in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(child, field, None)
            if isinstance(seq, list) and seq and all(isinstance(s, ast.stmt) for s in seq):
                yield seq


def _named_function(node, name):
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return child
    raise AssertionError(f"function {name} not found in {_SOURCE}")


def _assigns_none(stmt, name):
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == name
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is None
    )


def _guards_with_handler(stmt, handler_name):
    """True if `stmt` is an `if` whose body holds a `try/except ... as <name>`."""
    if not isinstance(stmt, ast.If):
        return False
    return any(
        handler.name == handler_name
        for node in ast.walk(stmt)
        if isinstance(node, ast.Try)
        for handler in node.handlers
    )


def _packed_block_source():
    """Return the dedented source of the packed no-grad block.

    Located structurally: the statement run inside
    `_get_per_token_logps_and_entropies` that begins with `_pk_result = None`
    and ends with the `if` holding the `except ... as _pk_err` handler. No text
    search, so a comment that happens to quote the same code cannot match.
    """
    text = _SOURCE.read_text(encoding = "utf-8")
    tree = ast.parse(text)
    factory = _named_function(tree, "grpo_trainer__get_per_token_logps_and_entropies")
    inner = _named_function(factory, "_get_per_token_logps_and_entropies")

    found = []
    for seq in _statement_lists(inner):
        start = next(
            (i for i, s in enumerate(seq) if _assigns_none(s, "_pk_result")),
            None,
        )
        if start is None:
            continue
        end = next(
            (j for j in range(start + 1, len(seq)) if _guards_with_handler(seq[j], "_pk_err")),
            None,
        )
        if end is not None:
            found.append((seq[start], seq[end]))
    assert len(found) == 1, f"expected one packed no-grad block, found {len(found)}"

    first, last = found[0]
    lines = text.splitlines(keepends = True)[first.lineno - 1 : last.end_lineno]
    return textwrap.dedent("".join(lines))


def _batch():
    # already left-padded by the caller, uneven pad counts across rows
    return torch.tensor(
        [
            [PAD_ID, PAD_ID, 3, 5, 7, 9, 11, 13],
            [PAD_ID, 2, 4, 6, 8, 10, 12, 14],
        ]
    )


def _run_packed_block(hidden_states = False, model = None):
    """Exec the real packed + verify block and hand back its locals."""
    if model is None:
        model = _Model(hidden_states = hidden_states)
    lm_head = model.head.weight  # [vocab, hidden]
    input_ids = _batch()
    left_pad = _left_pad_of(input_ids, KEEP, PAD_ID)
    max_left_pad = int(left_pad.max())

    namespace = {
        "os": __import__("os"),
        "torch": torch,
        **_DISPATCH_HELPERS,
        "chunked_hidden_states_selective_log_softmax": HELPERS[
            "chunked_hidden_states_selective_log_softmax"
        ],
        "chunked_selective_log_softmax": HELPERS["chunked_selective_log_softmax"],
        "create_completion_attention_mask": HELPERS["create_completion_attention_mask"],
        "calculate_pad_tokens_in_prompt": HELPERS["calculate_pad_tokens_in_prompt"],
        "_get_inference_mode_context_manager": lambda _model: torch.no_grad(),
        "device_synchronize": lambda *args, **kwargs: None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "UNSLOTH_GRPO_SEQ_PACKING_ON": True,
        "UNSLOTH_ZOO_HAS_MASKED_COL_GUARD": True,
        "self": SimpleNamespace(
            processing_class = SimpleNamespace(pad_token_id = PAD_ID),
            _autocast_dtype = torch.bfloat16,
        ),
        "model": model,
        "unwrapped_model": model,
        "lm_head": lm_head,
        "input_ids": input_ids,
        "left_pad_tokens_per_prompt": left_pad,
        "max_left_pad": max_left_pad,
        "logits_to_keep": KEEP,
        "total_rows": input_ids.shape[0],
        "batch_size": input_ids.shape[0],
        "seq_len": input_ids.shape[1],
        "multiplier": 1,
        "pixel_values": None,
        "token_type_ids": None,
        "mm_token_type_ids": None,
        "_pg_skip_pk": False,
        "logit_scale_multiply": 0.0,
        "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0,
        "temperature": 1.0,
    }
    exec(_packed_block_source(), namespace)
    return namespace, model, input_ids, max_left_pad


def _reference_logprobs(model, input_ids, max_left_pad):
    """Per-row logprobs straight from the model, no packing involved.

    Pads are dropped first, so a row's leading token has no predecessor and
    stays 0, exactly as both the packed scatter and the padded loop leave it.
    """
    width = KEEP + max_left_pad
    out = torch.zeros(input_ids.shape[0], input_ids.shape[1])
    for row in range(input_ids.shape[0]):
        cols = (input_ids[row] != PAD_ID).nonzero(as_tuple = False).squeeze(1)
        real = input_ids[row][cols].unsqueeze(0)
        with torch.no_grad():
            raw = model(input_ids = real).logits.float()
            if model.hidden_states:
                raw = raw @ model.head.weight.t().float()
            logps = torch.log_softmax(raw, dim = -1)[0]
        for j in range(1, real.shape[1]):
            out[row, cols[j]] = logps[j - 1, real[0, j]]
    return out[:, -width:]


def test_packed_path_survives_a_forward_that_returns_real_logits():
    namespace, model, _input_ids, _max_left_pad = _run_packed_block(hidden_states = False)
    assert getattr(model, "_unsloth_seq_packing_nograd_ok", None) is not False, (
        "a packed call site sent raw vocab logits into the lm_head matmul, the "
        "outer handler swallowed the raise and pinned packing off for the run"
    )
    assert namespace["_pk_use"] is True
    assert namespace["_pk_result"] is not None
    assert namespace["_pk_result"].shape == (2, KEEP + 2)


@pytest.mark.parametrize("hidden_states", [False, True])
def test_packed_result_matches_the_per_row_logprobs(hidden_states):
    namespace, model, input_ids, max_left_pad = _run_packed_block(hidden_states = hidden_states)
    if not namespace["_pk_use"]:
        pytest.fail("packed path declined the batch, so there is nothing to compare")
    mask = _completion_mask_of(
        input_ids[:, -(KEEP + max_left_pad) :],
        _left_pad_of(input_ids, KEEP, PAD_ID),
        max_left_pad,
        PAD_ID,
    ).float()
    reference = _reference_logprobs(model, input_ids, max_left_pad)
    got = namespace["_pk_result"].detach().float()
    assert torch.allclose(got * mask, reference * mask, atol = 1e-5), (
        got * mask,
        reference * mask,
    )


def test_square_lm_head_raw_logits_are_routed_by_the_explicit_signal():
    """vocab_size == hidden_size: the width comparison cannot tell them apart.

    Real logits are then hidden-width, so a width-only guard sends them through
    the lm_head a second time. The packed matmul is square, so nothing raises,
    and the per-row verifier misreads the width in exactly the same way, agrees
    with the corrupted packed result and marks the shape trusted. Both call
    sites therefore have to defer to the explicit
    UNSLOTH_RETURN_HIDDEN_STATES signal instead.
    """
    model = _Model(hidden_states = False, vocab = VOCAB, hidden = VOCAB, degraded = True)
    namespace, model, input_ids, max_left_pad = _run_packed_block(model = model)

    assert namespace["_pk_use"] is True
    assert namespace["_pk_ref"] is not None, "the per-row verifier never ran"
    mask = _completion_mask_of(
        input_ids[:, -(KEEP + max_left_pad) :],
        _left_pad_of(input_ids, KEEP, PAD_ID),
        max_left_pad,
        PAD_ID,
    ).float()
    reference = _reference_logprobs(model, input_ids, max_left_pad)
    got = namespace["_pk_result"].detach().float()
    assert torch.allclose(got * mask, reference * mask, atol = 1e-5), (
        got * mask,
        reference * mask,
    )


def test_square_lm_head_double_application_is_actually_detectable():
    """Guard against the assertion above passing vacuously."""
    model = _Model(hidden_states = False, vocab = VOCAB, hidden = VOCAB)
    input_ids = _batch()
    with torch.no_grad():
        logits = model(input_ids = input_ids).logits.float()
        doubled = logits @ model.head.weight.t().float()
    assert not torch.allclose(
        torch.log_softmax(logits, dim = -1),
        torch.log_softmax(doubled, dim = -1),
        atol = 1e-2,
    )


def test_first_use_verify_branch_runs_the_per_row_forwards():
    """Without this, the raw-logits test above would never reach the second
    call site: a model already inside the trusted envelope takes the shortcut
    and the verifier never runs."""
    namespace, model, input_ids, _max_left_pad = _run_packed_block(hidden_states = False)

    assert (
        "_pk_ref" in namespace and namespace["_pk_ref"] is not None
    ), "the first-use verify branch never ran, so the verifier call site is untested"
    assert (
        "_pk_diff" in namespace
    ), "the verifier raised before it could compare packed against per-row"
    packed_calls = [call for call in model.calls if call.packed]
    per_row_calls = [call for call in model.calls if not call.packed]
    assert len(packed_calls) == 1
    assert (
        len(per_row_calls) == input_ids.shape[0]
    ), f"expected one per-row verify forward per row, got {per_row_calls}"
    assert namespace["_pk_diff"] < 7e-1
    assert namespace["_pk_use"] is True
    assert getattr(model, "_unsloth_seq_packing_nograd_ok", None) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
