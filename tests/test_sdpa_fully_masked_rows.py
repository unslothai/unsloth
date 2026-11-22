# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Left-padded rows that attend to nothing give NaN under SDPA; transformers' sdpa_mask needs a fix.

The fix lets such a row attend to its own key only: opening it to every key let the last pad query of a
left-padded training row (whose logits predict the first real token) read that token and later ones."""

import inspect

import pytest

torch = pytest.importorskip("torch")
masking_utils = pytest.importorskip("transformers.masking_utils")

from unsloth.import_fixes import (  # noqa: E402
    _left_padded_probe_mask,
    _sdpa_mask_is_patched,
    _unmask_rows_attending_to_nothing,
    _sdpa_mask_leaves_rows_fully_masked,
    fix_transformers_fully_masked_rows,
)


def _call_sdpa_mask(fn, attention_mask):
    """Calls sdpa_mask with whichever signature ships: q_length on 5.x, cache_position on 4.57."""
    length = attention_mask.shape[-1]
    kwargs = {
        "batch_size": attention_mask.shape[0],
        "kv_length": length,
        "attention_mask": attention_mask,
        "allow_is_causal_skip": False,
    }
    params = inspect.signature(fn).parameters
    if "q_length" in params:
        kwargs["q_length"] = length
    elif "cache_position" in params:
        kwargs["cache_position"] = torch.arange(length)
    else:
        pytest.skip("sdpa_mask signature is neither shape this fix supports")
    return fn(**kwargs)


def _registered():
    """The sdpa builder models actually dispatch to (create_causal_mask reads the registry)."""
    return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]


@pytest.fixture
def unpatched():
    """Yields the unpatched sdpa_mask via __wrapped__, since earlier tests may already have patched it."""
    original_global = masking_utils.sdpa_mask
    original = getattr(original_global, "__wrapped__", original_global)
    interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    original_registered = interface["sdpa"] if interface is not None else None
    flag = getattr(masking_utils, "_unsloth_patched_sdpa_mask", False)
    masking_utils.sdpa_mask = original
    if interface is not None:
        interface.register("sdpa", original)
    masking_utils._unsloth_patched_sdpa_mask = False
    try:
        yield original
    finally:
        masking_utils.sdpa_mask = original_global
        if interface is not None and original_registered is not None:
            interface.register("sdpa", original_registered)
        masking_utils._unsloth_patched_sdpa_mask = flag


def test_the_bug_this_fix_exists_for_is_really_here(unpatched):
    """Fails if upstream restores the guard. Do not delete it to go green; re-measure first."""
    mask = _call_sdpa_mask(unpatched, _left_padded_probe_mask(torch))
    assert mask is not None and not mask.is_floating_point()
    fully_masked = int((~mask.bool().any(dim = -1)).sum())
    assert fully_masked == 1, (
        "the unpatched sdpa_mask no longer leaves a left-padded query row "
        "attending to nothing, so unsloth #9708 may be fixed upstream -- "
        "re-measure on a real model before simplifying the fix away"
    )


def test_the_probe_answers_true_when_the_bug_is_present(unpatched):
    assert _sdpa_mask_leaves_rows_fully_masked() is True


def test_the_patch_leaves_no_row_attending_to_nothing(unpatched):
    fix_transformers_fully_masked_rows()
    mask = _call_sdpa_mask(_registered(), _left_padded_probe_mask(torch))
    assert int((~mask.bool().any(dim = -1)).sum()) == 0, (
        "a query row still attends to nothing, so SDPA can still return NaN "
        "for it and a left-padded batch can still decode to the empty string"
    )


def test_the_patch_changes_nothing_a_real_row_could_read(unpatched):
    """Only rows attending to nothing may change, or a real token could attend across padding."""
    attention_mask = _left_padded_probe_mask(torch)
    before = _call_sdpa_mask(unpatched, attention_mask).bool()
    fix_transformers_fully_masked_rows()
    after = _call_sdpa_mask(_registered(), attention_mask).bool()

    attends_to_something = before.any(dim = -1)
    assert torch.equal(
        before[attends_to_something], after[attends_to_something]
    ), "the patch altered a row that already attended to something"


def test_left_pad_queries_attend_only_their_own_key(unpatched):
    """The leak: a pad query opened to every key sees the first real token it predicts, and later ones."""
    attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype = torch.bool)
    before = _call_sdpa_mask(unpatched, attention_mask).bool()
    fix_transformers_fully_masked_rows()
    after = _call_sdpa_mask(_registered(), attention_mask).bool()
    eye = torch.eye(6, dtype = torch.bool)
    for q in range(3):
        assert not before[0, 0, q].any()
        assert torch.equal(after[0, 0, q], eye[q]), (
            f"pad query {q} attends to {after[0, 0, q].nonzero().flatten().tolist()}, not only itself: "
            "a left-padded row's last pad query would read the token it is trained to predict"
        )
    assert torch.equal(before[0, :, 3:], after[0, :, 3:])
    assert torch.equal(before[1], after[1])


def test_with_a_cache_the_own_key_is_the_last_q_len_keys(unpatched):
    """Prefill on top of a cache: queries are the last q_len keys."""
    mask = torch.zeros((1, 1, 2, 5), dtype = torch.bool)
    mask[0, 0, 1, :] = True
    fixed = _unmask_rows_attending_to_nothing(mask)
    assert fixed[0, 0, 0].tolist() == [False, False, False, True, False]
    assert torch.equal(fixed[0, 0, 1], mask[0, 0, 1])


def test_only_the_registry_is_patched_and_eager_keeps_upstream_rows(unpatched):
    """eager_mask builds from the module global: an all-min row (0 output with sinks) must stay all-min."""
    if getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None) is None:
        pytest.skip("this transformers has no ALL_MASK_ATTENTION_FUNCTIONS")
    attention_mask = torch.tensor([[0, 0, 1, 1]], dtype = torch.bool)
    before = _call_sdpa_mask(masking_utils.eager_mask, attention_mask)
    fix_transformers_fully_masked_rows()
    assert _registered() is not unpatched
    assert masking_utils.sdpa_mask is unpatched
    after = _call_sdpa_mask(masking_utils.eager_mask, attention_mask)
    assert after.is_floating_point() and torch.equal(before, after)
    assert bool((after[0, 0, :2] == torch.finfo(after.dtype).min).all())


def test_left_padded_model_loss_cannot_read_future_tokens(unpatched):
    """End to end on a random tiny Llama (transformers' own forward, sdpa): the last pad position's
    logits must not change when a later token does, and real rows must be bitwise unchanged."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(3, 64, (2, 10))
    am = torch.ones_like(ids)
    am[0, :4] = 0
    future = ids.clone()
    future[0, 7] = (future[0, 7] + 5) % 64

    def logits(x):
        with torch.no_grad():
            return model(input_ids = x, attention_mask = am, use_cache = False).logits

    base = logits(ids)
    fix_transformers_fully_masked_rows()
    fixed, fixed_future = logits(ids), logits(future)
    assert torch.isfinite(fixed).all()
    assert torch.equal(fixed[0, 3], fixed_future[0, 3]), "the last pad position reads a later token"
    assert torch.equal(fixed[0, 4:], base[0, 4:]) and torch.equal(fixed[1], base[1])


def test_patching_twice_does_not_stack_wrappers(unpatched):
    fix_transformers_fully_masked_rows()
    once = _registered()
    fix_transformers_fully_masked_rows()
    assert _registered() is once
    assert _registered().__wrapped__ is unpatched


def test_the_original_stays_reachable(unpatched):
    """Undoable and probeable. Without this the probe would read the patched
    function on a second call and report the bug as fixed."""
    fix_transformers_fully_masked_rows()
    assert _registered().__wrapped__ is unpatched
    assert _sdpa_mask_leaves_rows_fully_masked() is True


def test_the_probe_says_no_when_the_build_already_corrects_itself(unpatched):
    """A build that already corrects itself: the probe must answer False and the fix changes nothing."""

    def already_correct(*args, **kwargs):
        mask = unpatched(*args, **kwargs)
        if mask is not None and not mask.is_floating_point():
            mask = mask | ~mask.any(dim = -1, keepdim = True)
        return mask

    already_correct.__signature__ = inspect.signature(unpatched)
    masking_utils.sdpa_mask = already_correct
    assert _sdpa_mask_leaves_rows_fully_masked() is False
    fix_transformers_fully_masked_rows()
    assert (
        masking_utils.sdpa_mask is already_correct and _registered() is unpatched
    ), "the fix patched a transformers that does not need it"
    assert not getattr(masking_utils, "_unsloth_patched_sdpa_mask", False)


def test_the_probe_is_dtype_honest(unpatched):
    """Probe with a bool mask like real callers; an int64 probe once gave a false not-affected answer."""
    assert _left_padded_probe_mask(torch).dtype == torch.bool
    mask = _call_sdpa_mask(unpatched, _left_padded_probe_mask(torch))
    assert mask.dtype == torch.bool


def test_a_batch_with_no_padding_is_untouched(unpatched):
    """No pad, no fully-masked row, nothing for the patch to do."""
    attention_mask = torch.ones((2, 2), dtype = torch.bool)
    before = _call_sdpa_mask(unpatched, attention_mask).bool()
    fix_transformers_fully_masked_rows()
    after = _call_sdpa_mask(_registered(), attention_mask).bool()
    assert torch.equal(before, after)


def test_the_probe_is_pinned_to_cpu_whatever_the_default_device_is(unpatched):
    """The probe mask is pinned to CPU, so a meta default device cannot abort the import."""
    torch.set_default_device("meta")
    try:
        assert _left_padded_probe_mask(torch).device.type == "cpu"
        assert _sdpa_mask_leaves_rows_fully_masked() is True
    finally:
        torch.set_default_device(None)


def test_a_reloaded_masking_utils_is_patched_again(unpatched):
    """Gate on the live binding, not a module marker: reload reverts the registry but keeps the marker."""
    fix_transformers_fully_masked_rows()
    assert _sdpa_mask_is_patched(masking_utils)

    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", unpatched)
    assert getattr(masking_utils, "_unsloth_patched_sdpa_mask", False) is True
    assert not _sdpa_mask_is_patched(masking_utils)

    fix_transformers_fully_masked_rows()
    assert _sdpa_mask_is_patched(masking_utils)
    mask = _call_sdpa_mask(_registered(), _left_padded_probe_mask(torch))
    assert int((~mask.bool().any(dim = -1)).sum()) == 0


def test_the_correction_itself_on_every_dtype_it_can_meet():
    """Checks the unmasking helper directly on every dtype it can meet; no transformers needed."""
    bool_mask = torch.tensor([[[[False, False], [False, True]]]])
    fixed = _unmask_rows_attending_to_nothing(bool_mask)
    assert fixed.tolist() == [[[[True, False], [False, True]]]], (
        "the row attending to nothing was not opened to its own key only, or a "
        "row that already attended to something was changed"
    )
    assert fixed.dtype == torch.bool

    int_mask = bool_mask.to(torch.int64)
    assert _unmask_rows_attending_to_nothing(int_mask).tolist() == [[[[1, 0], [0, 1]]]]

    float_mask = torch.zeros((1, 1, 2, 2), dtype = torch.float32)
    assert _unmask_rows_attending_to_nothing(float_mask) is float_mask

    assert _unmask_rows_attending_to_nothing(None) is None


def test_the_correction_is_idempotent_and_self_neutralising():
    """The row unmasking is idempotent and an identity on already-correct masks, so it is safe to
    leave in place if a future transformers restores its own guard."""
    mask = torch.tensor([[[[False, False], [False, True]]]])
    once = _unmask_rows_attending_to_nothing(mask)
    twice = _unmask_rows_attending_to_nothing(once)
    assert torch.equal(once, twice)

    already_fine = torch.tensor([[[[True, False], [True, True]]]])
    assert torch.equal(_unmask_rows_attending_to_nothing(already_fine), already_fine)
