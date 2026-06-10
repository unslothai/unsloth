"""Regression guard for batched left-padded generation (issues #1066, #3699).

`_fast_prepare_inputs_for_generation` in unsloth/models/llama.py is shared by
every decoder family wired through `fix_prepare_inputs_for_generation` (llama,
qwen2/3, mistral, gemma/2, cohere, granite, qwen3_moe, glm4_moe). Two historical
bugs lived in it:

  (a) the 2D attention mask was truncated to its last column during cached
      decode, losing padding information (introduced cc4c5d77, fixed by #2216);
  (b) position_ids were taken directly from cache_position, a global counter
      that includes left-pad tokens, so padded rows generated garbage
      (introduced cc4c5d77, reported in #1066/#3699, fixed by #4100).

These tests pin the fixed behavior with synthetic inputs on CPU. They must fail
on any code that reintroduces either bug.

Companion structural guard: test_prepare_inputs_ast_guard.py (import-free).
"""

import pytest
import torch


PAST_LEN = 4

# Three rows with different amounts of left padding (0 = pad).
MASK = torch.tensor(
    [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
    ],
    dtype = torch.long,
)
BS, SEQ = MASK.shape

# Per-row positions: cumsum(-1) - 1 with pad slots filled with 1.
EXPECTED_PREFILL_POSITIONS = torch.tensor(
    [
        [1, 1, 0, 1, 2],
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
    ],
    dtype = torch.long,
)


class FakeDynamicCache:
    """Minimal stand-in for transformers DynamicCache with a non-empty cache."""

    def __init__(self, seq_length):
        self._seq_length = seq_length

    def __len__(self):
        return 1

    def get_seq_length(self):
        return self._seq_length


class FakeModel:
    """Bare-minimum `self` for _fast_prepare_inputs_for_generation."""

    dtype = torch.float32
    config = None


class FakeModelWith4DMask(FakeModel):
    """Variant exposing the HF 4D mask builder; records what it receives."""

    def __init__(self):
        self.mask_calls = []

    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask,
        sequence_length,
        target_length,
        dtype,
        device,
        cache_position,
        batch_size,
        config = None,
        past_key_values = None,
    ):
        self.mask_calls.append(
            {
                "mask_shape": tuple(attention_mask.shape) if attention_mask is not None else None,
                "sequence_length": sequence_length,
                "target_length": target_length,
                "batch_size": batch_size,
            }
        )
        return torch.zeros((batch_size, 1, sequence_length, target_length), dtype = dtype)


def _prepare(model, input_ids, attention_mask, **kwargs):
    from unsloth.models import llama as llama_mod
    return llama_mod._fast_prepare_inputs_for_generation(
        model, input_ids, attention_mask = attention_mask, **kwargs
    )


def test_prefill_position_ids_derived_from_left_padded_mask():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(FakeModel(), input_ids, MASK)

    position_ids = result.get("position_ids", None)
    assert (
        position_ids is not None
    ), "prefill with a left-padded 2D attention mask must populate position_ids"
    assert torch.equal(position_ids.long().cpu(), EXPECTED_PREFILL_POSITIONS), (
        "prefill position_ids must be derived per row from the attention mask "
        "(cumsum - 1, pads masked), so each row starts counting at its first "
        f"real token; got {position_ids.tolist()}"
    )
    assert result["input_ids"].shape == (BS, SEQ)


@pytest.mark.parametrize("pass_cache_position", [True, False])
def test_cached_decode_position_ids_ignore_left_padding(pass_cache_position):
    # Decode step: PAST_LEN tokens cached, current token is the mask's last
    # column. Row 0 has 2 pads, so its current token sits at logical position 2,
    # NOT at cache_position == PAST_LEN. This is exactly issue #3699.
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    kwargs = {"past_key_values": FakeDynamicCache(PAST_LEN)}
    if pass_cache_position:
        kwargs["cache_position"] = torch.arange(PAST_LEN, PAST_LEN + 1)

    result = _prepare(FakeModel(), input_ids, MASK, **kwargs)

    assert result["input_ids"].shape == (
        BS,
        1,
    ), "cached decode must slice input_ids to the last token only"
    position_ids = result.get("position_ids", None)
    assert position_ids is not None
    expected = torch.tensor([[2], [4], [3]], dtype = torch.long)
    assert torch.equal(position_ids.long().cpu().reshape(BS, 1), expected), (
        "left-padded cached decode must derive per-row position_ids from the "
        "attention mask, not from cache_position which counts pad tokens; got "
        f"{position_ids.tolist()}, expected {expected.tolist()} "
        "(row 0 has 2 pads: its position must be 2, not 4)"
    )


def test_cached_decode_does_not_truncate_2d_attention_mask():
    # Without a 4D mask builder the original 2D mask must survive untouched.
    # The historical bug replaced it with attention_mask[:, [-1]].
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(FakeModel(), input_ids, MASK, past_key_values = FakeDynamicCache(PAST_LEN))
    mask_out = result["attention_mask"]
    assert mask_out is not None
    assert mask_out.dim() != 2 or mask_out.shape[-1] == SEQ, (
        "the 2D attention mask must not be truncated to its last column during "
        f"cached decode (got shape {tuple(mask_out.shape)}); padding rows lose "
        "their pad information otherwise"
    )


def test_cached_decode_4d_mask_builder_receives_full_target_length():
    model = FakeModelWith4DMask()
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(model, input_ids, MASK, past_key_values = FakeDynamicCache(PAST_LEN))
    assert len(model.mask_calls) == 1
    call = model.mask_calls[0]
    assert call["mask_shape"] == (BS, SEQ), (
        "the 4D mask builder must receive the full 2D padding mask, not a "
        f"truncated one (got {call['mask_shape']})"
    )
    assert call["sequence_length"] == 1
    assert call["target_length"] == SEQ, (
        "target_length must cover the whole mask so padded positions stay "
        f"masked (got {call['target_length']})"
    )
    assert result["attention_mask"].dim() == 4


def test_caller_supplied_position_ids_are_passed_through():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    custom = torch.full((BS, SEQ), 7, dtype = torch.long)
    result = _prepare(FakeModel(), input_ids, MASK, position_ids = custom)
    assert torch.equal(
        result["position_ids"], custom
    ), "caller-supplied position_ids must not be overwritten"


def test_legacy_tuple_cache_still_takes_cached_decode_path():
    # Legacy cache format: tuple of (K, V) per layer; past length from K.shape[-2].
    k = torch.zeros((BS, 1, PAST_LEN, 8))
    legacy_cache = ((k, k.clone()),)
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(FakeModel(), input_ids, MASK, past_key_values = legacy_cache)
    assert result["input_ids"].shape == (BS, 1)
    expected = torch.tensor([[2], [4], [3]], dtype = torch.long)
    assert torch.equal(result["position_ids"].long().cpu().reshape(BS, 1), expected)
