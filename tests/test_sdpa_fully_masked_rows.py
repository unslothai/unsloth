# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Left-padded rows that attend to nothing give NaN under SDPA; transformers' sdpa_mask needs a fix.

The fix lets such a row attend to its own key only: opening it to every key let the last pad query of a
left-padded training row (whose logits predict the first real token) read that token and later ones.
The own key is the query's absolute position (cache_position / q_offset - kv_offset), so preallocated
caches whose key columns outnumber the filled slots pick the right column."""

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


def test_without_coordinates_the_own_key_is_the_last_q_len_keys(unpatched):
    """A direct call with no coordinates (prefill on a dynamic cache): queries are the last q_len keys."""
    mask = torch.zeros((1, 1, 2, 5), dtype = torch.bool)
    mask[0, 0, 1, :] = True
    fixed = _unmask_rows_attending_to_nothing(mask)
    assert fixed[0, 0, 0].tolist() == [False, False, False, True, False]
    assert torch.equal(fixed[0, 0, 1], mask[0, 0, 1])


def test_both_bindings_are_patched_so_eager_and_aliases_keep_the_fix(unpatched):
    """eager_mask and `register("name", sdpa_mask)` aliases read the module global, so it is patched too."""
    from transformers import AttentionMaskInterface

    fix_transformers_fully_masked_rows()
    assert masking_utils.sdpa_mask is _registered()
    assert _registered().__wrapped__ is unpatched
    AttentionMaskInterface.register("unsloth_test_alias", masking_utils.sdpa_mask)
    try:
        alias = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["unsloth_test_alias"]
        mask = _call_sdpa_mask(alias, torch.tensor([[0, 0, 1, 1]], dtype = torch.bool))
        assert int((~mask.bool().any(dim = -1)).sum()) == 0, "an sdpa_mask alias lost the NaN fix"
    finally:
        AttentionMaskInterface._global_mapping.pop("unsloth_test_alias", None)


def _eager_mask(
    attention_mask,
    dtype,
    config = None,
):
    length = attention_mask.shape[-1]
    kwargs = {
        "batch_size": attention_mask.shape[0],
        "kv_length": length,
        "attention_mask": attention_mask,
        "dtype": dtype,
    }
    if config is not None:
        kwargs["config"] = config
    params = inspect.signature(masking_utils.eager_mask).parameters
    if "q_length" in params:
        kwargs["q_length"] = length
    else:
        kwargs["cache_position"] = torch.arange(length)
    return masking_utils.eager_mask(**kwargs)


def test_eager_pad_rows_attend_their_own_key_and_cannot_overflow_in_fp16(unpatched):
    """A sink-free eager all-min row is uniform over every key (future tokens included), and in fp16
    `score + finfo.min` is -inf once every score is below about -16, so the softmax returns NaN."""
    from transformers import LlamaConfig

    attention_mask = torch.tensor([[0, 0, 1, 1]], dtype = torch.bool)
    before = _eager_mask(attention_mask, torch.float16, LlamaConfig())
    probs = torch.softmax(
        before[0, 0, 1] + torch.full((4,), -1024.0, dtype = torch.float16), -1, dtype = torch.float32
    )
    assert not torch.isfinite(
        probs
    ).all(), "upstream's fp16 eager row no longer overflows; re-check"
    fix_transformers_fully_masked_rows()
    after = _eager_mask(attention_mask, torch.float16, LlamaConfig())
    probs = torch.softmax(
        after[0, 0, 1] + torch.full((4,), -1024.0, dtype = torch.float16), -1, dtype = torch.float32
    )
    torch.testing.assert_close(probs, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(before[0, :, 2:], after[0, :, 2:])


def test_eager_with_attention_sinks_keeps_upstream_rows(unpatched):
    """With sinks (gpt-oss) an all-min row puts all of its weight on the sink and outputs 0, as flex does."""
    from transformers import GptOssConfig

    attention_mask = torch.tensor([[0, 0, 1, 1]], dtype = torch.bool)
    before = _eager_mask(attention_mask, torch.float32, GptOssConfig())
    fix_transformers_fully_masked_rows()
    after = _eager_mask(attention_mask, torch.float32, GptOssConfig())
    assert after.is_floating_point() and torch.equal(before, after)
    assert bool((after[0, 0, :2] == torch.finfo(after.dtype).min).all())


def _causal_mask(config, inputs_embeds, attention_mask, cache, positions):
    create = masking_utils.create_causal_mask
    params = inspect.signature(create).parameters
    kwargs = {"past_key_values": cache}
    if "input_embeds" in params:
        kwargs["input_embeds"] = inputs_embeds
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    if "cache_position" in params:
        kwargs["cache_position"] = positions
    if "position_ids" in params:
        kwargs["position_ids"] = positions[None].expand(inputs_embeds.shape[0], -1)
    return create(config, attention_mask = attention_mask, **kwargs)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("prefix", [0, 3])
def test_a_preallocated_cache_pad_query_attends_its_own_slot(unpatched, static, prefix):
    """A StaticCache has more key columns than filled slots: the own key is the query's position, not
    one of the last q_len (unfilled) columns. Through the real registry with a live cache."""
    from transformers import DynamicCache, LlamaConfig, StaticCache

    fix_transformers_fully_masked_rows()
    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    config._attn_implementation = "sdpa"
    if static:
        cache = StaticCache(config = config, max_cache_len = 16)
    else:
        cache = DynamicCache()
    if prefix:
        kv = torch.zeros((2, 2, prefix, 8))
        cache.update(kv, kv, 0, {"cache_position": torch.arange(prefix)})
    positions = torch.arange(prefix, prefix + 4)
    keep = torch.ones((2, prefix + 4), dtype = torch.bool)
    keep[0, : prefix + 2] = False
    mask = _causal_mask(config, torch.zeros((2, 4, 32)), keep, cache, positions)
    assert mask is not None and mask.dtype == torch.bool
    assert mask.shape[-1] == (16 if static else prefix + 4)
    assert mask[0, 0, 0].nonzero().flatten().tolist() == [prefix]
    assert mask[0, 0, 1].nonzero().flatten().tolist() == [prefix + 1]
    assert int((~mask.any(dim = -1)).sum()) == 0


def test_the_coordinates_are_read_from_either_signature(unpatched):
    """4.57 passes cache_position, 5.x q_offset (int or tensor) and kv_offset; positional or keyword."""
    from unsloth.import_fixes import _mask_coordinates

    def v4(
        batch_size,
        cache_position,
        kv_length,
        kv_offset = 0,
        **kwargs,
    ):
        pass

    def v5(
        batch_size,
        q_length,
        kv_length,
        q_offset = 0,
        kv_offset = 0,
        **kwargs,
    ):
        pass

    positions = torch.arange(3, 5)
    assert _mask_coordinates(v4)((1, positions, 8), {"kv_offset": 2})["cache_position"] is positions
    assert _mask_coordinates(v4)((1, positions, 8), {"kv_offset": 2})["kv_offset"] == 2
    assert _mask_coordinates(v5)(
        (), {"batch_size": 1, "q_length": 2, "kv_length": 8, "q_offset": 3}
    ) == {
        "cache_position": None,
        "q_offset": 3,
        "kv_offset": 0,
    }
    assert _mask_coordinates(v5)((1, 2, 8), {})["q_offset"] == 0
    # A transitional 5.x took the positions tensor in the q_length slot.
    assert _mask_coordinates(v5)((1, positions, 8), {})["cache_position"] is positions

    mask = torch.zeros((1, 1, 2, 8), dtype = torch.bool)
    fixed = _unmask_rows_attending_to_nothing(mask, q_offset = torch.tensor(5), kv_offset = 2)
    assert fixed[0, 0].nonzero().tolist() == [[0, 3], [1, 4]]
    fixed = _unmask_rows_attending_to_nothing(
        mask, cache_position = torch.tensor([5, 6]), kv_offset = 2
    )
    assert fixed[0, 0].nonzero().tolist() == [[0, 3], [1, 4]]


def test_the_correction_traces_once_under_dynamic_shapes():
    """No shape branch on the coordinate path: one graph across lengths, no graph breaks."""
    import torch._dynamo

    torch._dynamo.reset()
    counts = {"graphs": 0}

    def backend(gm, example_inputs):
        counts["graphs"] += 1
        return gm.forward

    @torch.compile(backend = backend, dynamic = True, fullgraph = True)
    def fix(mask, q_offset):
        return _unmask_rows_attending_to_nothing(mask, q_offset = q_offset, kv_offset = 0)

    for q_len, kv_len in [(4, 8), (6, 12), (12, 12), (5, 16)]:
        mask = torch.zeros((2, 1, q_len, kv_len), dtype = torch.bool)
        out = fix(mask, torch.tensor(kv_len - q_len))
        assert out[0, 0].nonzero()[:, 1].tolist() == list(range(kv_len - q_len, kv_len))
    assert counts["graphs"] <= 2
    torch._dynamo.reset()


def test_encoder_registrations_use_the_repaired_builder(unpatched, monkeypatch):
    """Decision and SentenceTransformer register their own mask key; it must carry the fix."""
    from transformers import AttentionMaskInterface

    fix_transformers_fully_masked_rows()
    calls = {}
    register = AttentionMaskInterface.register.__func__

    def record(cls, name, builder):
        calls[name] = builder
        return register(cls, name, builder)

    monkeypatch.setattr(AttentionMaskInterface, "register", classmethod(record))
    from types import SimpleNamespace

    from transformers import BertConfig, BertModel

    _compilable_attention = pytest.importorskip(
        "unsloth.models._decision_fast"
    )._compilable_attention
    config = BertConfig(
        hidden_size = 16, num_attention_heads = 2, num_hidden_layers = 1, intermediate_size = 16
    )
    config._attn_implementation = "sdpa"
    wrapper = SimpleNamespace(encoder = BertModel(config))
    try:
        with _compilable_attention(wrapper, enabled = True):
            pass
    except Exception as e:  # the context needs more of a Decision model on some builds
        pytest.skip(f"_compilable_attention needs a fuller model here ({e})")
    finally:
        AttentionMaskInterface._global_mapping.pop("unsloth_decision_sdpa", None)
    assert calls.get("unsloth_decision_sdpa") is _registered()


def test_native_sdpa_dispatch_opens_only_the_own_key_for_a_causal_pad_row():
    """Unsloth's own SDPA dispatch: a causal left pad row attends to its own key, not every key."""
    ad = pytest.importorskip("unsloth.utils.attention_dispatch")
    torch.manual_seed(0)
    B, H, L, D = 2, 2, 6, 8
    Q, K, V = (torch.randn(B, H, L, D) for _ in range(3))
    keep = torch.ones((B, L), dtype = torch.long)
    keep[0, :3] = 0

    def attend(V):
        config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = H, n_groups = 1)
        context = ad.AttentionContext(
            bsz = B,
            q_len = L,
            kv_seq_len = L,
            n_heads = H,
            head_dim = D,
            requires_grad = False,
            seq_info = None,
            attention_mask = keep,
            causal_mask = None,
            training = True,
        )
        out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
        return (
            out.reshape(B, L, H, D) if out.dim() == 4 and out.shape[1] == L else out.transpose(1, 2)
        )

    out = attend(V)
    later = V.clone()
    later[0, :, 4] += 1.0
    assert torch.equal(attend(later)[0, 2], out[0, 2]), "the last pad query reads a later token"
    torch.testing.assert_close(out[0, 0], V[0, :, 0])


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
    """Keep the original builder reachable and the upstream probe repeatable."""
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
