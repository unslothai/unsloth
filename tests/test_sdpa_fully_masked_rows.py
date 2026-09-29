# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A left-padded row that attends to nothing must not reach SDPA.

`transformers.masking_utils.sdpa_mask` builds a boolean mask and returns it with
no correction for query rows that attend to no key at all, and the parameter that
used to make that correction is now documented `"Deprecated and has no effect.
Will be removed in version 5.18.0."`. In 4.57.6 the same function still carried
it, guarded on `not _is_torch_greater_or_equal_than_2_5` -- upstream retired it
believing torch 2.5 had made it unnecessary.

Measured on a B200, torch 2.13.0+cu130, transformers 5.15.1, unquantized fp16
`google/gemma-4-E2B-it`, no unsloth in the process: a SINGLE forward pass returns
NaN logits on exactly the rows that received a left pad token and finite logits
on every row that did not, 16 rows of 16 across batch sizes 2, 4 and 8, and under
`generate` those rows decode to the empty string. That is unsloth #9708.

Every test here DRIVES the real functions. The rules in this repo have been
caught before passing against a hand-written dict while the code that produces
it was broken, so nothing below asserts on a literal that the code did not
compute.
"""

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
    """Call whichever signature this transformers ships.

    5.x takes `q_length`; 4.57.6 binds `sdpa_mask` to `sdpa_mask_recent_torch`,
    which takes `cache_position`. The fix supports both, so the tests must too.
    """
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


@pytest.fixture
def unpatched():
    """The original function, and the module put back afterwards.

    Other tests in a session may already have installed the patch, so reach for
    `__wrapped__` rather than assuming the module global is pristine.
    """
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
    """The negative control, and it is the load-bearing test in this file.

    If a future transformers restores the guard, this FAILS and says so, rather
    than leaving a wrapper nobody can justify. Do not delete it to make the
    suite green: re-measure first, then remove the fix and this file together.
    """
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
    mask = _call_sdpa_mask(masking_utils.sdpa_mask, _left_padded_probe_mask(torch))
    assert int((~mask.bool().any(dim = -1)).sum()) == 0, (
        "a query row still attends to nothing, so SDPA can still return NaN "
        "for it and a left-padded batch can still decode to the empty string"
    )


def test_the_patch_changes_nothing_a_real_row_could_read(unpatched):
    """The correction must be confined to rows that attend to nothing.

    Those are pad positions whose outputs are discarded, which is why upstream's
    own docstring said this "does not change the final result". A patch that
    also loosened a real row would silently let a token attend across padding.
    """
    attention_mask = _left_padded_probe_mask(torch)
    before = _call_sdpa_mask(unpatched, attention_mask).bool()
    fix_transformers_fully_masked_rows()
    after = _call_sdpa_mask(masking_utils.sdpa_mask, attention_mask).bool()

    attends_to_something = before.any(dim = -1)
    assert torch.equal(
        before[attends_to_something], after[attends_to_something]
    ), "the patch altered a row that already attended to something"


def test_both_bindings_are_patched(unpatched):
    """`eager_mask` reads the module global; the interface captured the original.

    They are different references to the same function and both have to move, or
    half the models in transformers keep the old one. Confirmed by file search:
    `sdpa_mask` is defined once in transformers and no other module imports it.
    """
    fix_transformers_fully_masked_rows()
    interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    if interface is None:
        pytest.skip("this transformers has no ALL_MASK_ATTENTION_FUNCTIONS")
    assert interface["sdpa"] is masking_utils.sdpa_mask
    assert masking_utils.sdpa_mask is not unpatched


def test_patching_twice_does_not_stack_wrappers(unpatched):
    fix_transformers_fully_masked_rows()
    once = masking_utils.sdpa_mask
    fix_transformers_fully_masked_rows()
    assert masking_utils.sdpa_mask is once
    assert masking_utils.sdpa_mask.__wrapped__ is unpatched


def test_the_original_stays_reachable(unpatched):
    """Undoable and probeable. Without this the probe would read the patched
    function on a second call and report the bug as fixed."""
    fix_transformers_fully_masked_rows()
    assert masking_utils.sdpa_mask.__wrapped__ is unpatched
    assert _sdpa_mask_leaves_rows_fully_masked() is True


def test_the_probe_says_no_when_the_build_already_corrects_itself(unpatched):
    """The gate, exercised in the direction that matters for an unaffected user.

    A stub standing in for a future fixed transformers: the probe must answer
    False, and `fix_...` must then leave the module alone byte for byte.
    """

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
        masking_utils.sdpa_mask is already_correct
    ), "the fix patched a transformers that does not need it"
    assert not getattr(masking_utils, "_unsloth_patched_sdpa_mask", False)


def test_the_probe_is_dtype_honest(unpatched):
    """The probe feeds a BOOL mask because the real callers do.

    Written after an int64 probe mask came back int64 and an earlier
    `dtype == torch.bool` check answered "not affected" for a reason that had
    nothing to do with the bug.
    """
    assert _left_padded_probe_mask(torch).dtype == torch.bool
    mask = _call_sdpa_mask(unpatched, _left_padded_probe_mask(torch))
    assert mask.dtype == torch.bool


def test_a_batch_with_no_padding_is_untouched(unpatched):
    """No pad, no fully-masked row, nothing for the patch to do."""
    attention_mask = torch.ones((2, 2), dtype = torch.bool)
    before = _call_sdpa_mask(unpatched, attention_mask).bool()
    fix_transformers_fully_masked_rows()
    after = _call_sdpa_mask(masking_utils.sdpa_mask, attention_mask).bool()
    assert torch.equal(before, after)


def test_the_probe_is_pinned_to_cpu_whatever_the_default_device_is(unpatched):
    """A meta default device must answer the question, not abort the import.

    `torch.set_default_device("meta")` around `import unsloth` used to put the
    probe mask on meta, where `sdpa_mask` builds its own index tensors on CPU
    and raises, or hands back a meta mask whose truth value cannot be read --
    and that read sat outside the guard, so it propagated out of the import.
    """
    torch.set_default_device("meta")
    try:
        assert _left_padded_probe_mask(torch).device.type == "cpu"
        assert _sdpa_mask_leaves_rows_fully_masked() is True
    finally:
        torch.set_default_device(None)


def test_a_reloaded_masking_utils_is_patched_again(unpatched):
    """The guard reads the live bindings, not a mark on the module.

    `importlib.reload` re-executes the module body in the SAME namespace, so
    `sdpa_mask` and the registry entry revert to upstream while any attribute
    we set on the module survives. Gating on that attribute refuses to re-patch
    a build that is vulnerable again -- protection lost, silently.
    """
    fix_transformers_fully_masked_rows()
    assert _sdpa_mask_is_patched(masking_utils)

    # Exactly what a reload leaves behind: upstream functions, our marker.
    interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    masking_utils.sdpa_mask = unpatched
    if interface is not None:
        interface.register("sdpa", unpatched)
    assert getattr(masking_utils, "_unsloth_patched_sdpa_mask", False) is True
    assert not _sdpa_mask_is_patched(masking_utils)

    fix_transformers_fully_masked_rows()
    assert _sdpa_mask_is_patched(masking_utils)
    mask = _call_sdpa_mask(masking_utils.sdpa_mask, _left_padded_probe_mask(torch))
    assert int((~mask.bool().any(dim = -1)).sum()) == 0


def test_a_half_installed_patch_is_completed_rather_than_skipped(unpatched):
    """Module global ours, registry upstream: not patched, so run again."""
    interface = getattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS", None)
    if interface is None:
        pytest.skip("this transformers has no ALL_MASK_ATTENTION_FUNCTIONS")
    fix_transformers_fully_masked_rows()
    interface.register("sdpa", unpatched)
    assert not _sdpa_mask_is_patched(masking_utils)
    fix_transformers_fully_masked_rows()
    assert interface["sdpa"] is masking_utils.sdpa_mask
    assert _sdpa_mask_is_patched(masking_utils)


def test_the_correction_itself_on_every_dtype_it_can_meet():
    """The helper, driven directly. No transformers needed, no fixture state.

    Replaces an earlier version of this test that asserted a tautology and would
    have passed against any implementation at all.
    """
    bool_mask = torch.tensor([[[[False, False], [False, True]]]])
    fixed = _unmask_rows_attending_to_nothing(bool_mask)
    assert fixed.tolist() == [[[[True, True], [False, True]]]], (
        "the row attending to nothing was not opened up, or a row that already "
        "attended to something was changed"
    )
    assert fixed.dtype == torch.bool

    int_mask = bool_mask.to(torch.int64)
    assert _unmask_rows_attending_to_nothing(int_mask).tolist() == [[[[1, 1], [0, 1]]]]

    # The eager path: returned untouched, and by identity, not by value.
    float_mask = torch.zeros((1, 1, 2, 2), dtype = torch.float32)
    assert _unmask_rows_attending_to_nothing(float_mask) is float_mask

    # `is_causal` was used instead of a mask.
    assert _unmask_rows_attending_to_nothing(None) is None


def test_the_correction_is_idempotent_and_self_neutralising():
    """Applied twice is applied once, and applied to an already-correct mask it
    is the identity in value. That is what makes it safe to leave in place if a
    future transformers restores its own guard."""
    mask = torch.tensor([[[[False, False], [False, True]]]])
    once = _unmask_rows_attending_to_nothing(mask)
    twice = _unmask_rows_attending_to_nothing(once)
    assert torch.equal(once, twice)

    already_fine = torch.tensor([[[[True, False], [True, True]]]])
    assert torch.equal(_unmask_rows_attending_to_nothing(already_fine), already_fine)
