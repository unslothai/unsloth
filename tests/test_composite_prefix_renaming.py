# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A submodule's prefix renaming must not be applied to the composite model.

transformers 5.4.0 (PR #44300) made `get_model_conversion_mapping` recurse into
`PreTrainedModel` submodules and merge their conversion mappings into the parent's.
The standalone Qwen3.5 / Qwen3.5-MoE / Gemma 3n text models register
`^model.language_model.` -> `^model.`, which is correct for themselves and wrong for
the composite model whose weights really are named `model.language_model....`. Renamings
run before the bitsandbytes converter, so a pre-quantized checkpoint loses every
`absmax` / `quant_map` / `nested_absmax` / `nested_quant_map` /
`quant_state.bitsandbytes__nf4` sidecar and every `Linear4bit` comes back with
`quant_state is None`: measured 352 of 352 on `unsloth/qwen3.8-27b-unsloth-bnb-4bit` and
439 of 439 on `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit`, at 5.4.0 and at 5.5.4.
Upstream fixed it in 5.6.0 with PR #45567.

Every test here drives the real functions against a real transformers model built on the
meta device, so nothing asserts on a hand-written mapping that the code did not produce.
The pathology tests are conditioned on what the installed transformers actually does, not
on its version number: on a build that leaks they prove the fix removes the leak, and on a
build that does not leak they prove the fix leaves it alone.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from unsloth.import_fixes import (  # noqa: E402
    _COMPOSITE_PREFIX_RENAMING_FLAG,
    _leaked_submodule_prefix_renamings,
    _prefixed_pattern,
    _renaming_destroys_keys,
    _renaming_signature,
    _rescope_conversions,
    _rescoped_renaming,
    _transformers_rescopes_submodule_prefix_renamings,
    fix_transformers_composite_prefix_renaming,
)


def _weight_renaming():
    """transformers 4.x has no `core_model_loading`, and so no pathology to test."""
    try:
        from transformers.core_model_loading import WeightRenaming
    except Exception:
        pytest.skip("this transformers has no core_model_loading.WeightRenaming")
    return WeightRenaming


def _conversion_mapping():
    try:
        from transformers import conversion_mapping
    except Exception:
        pytest.skip("this transformers has no conversion_mapping module")
    if not hasattr(conversion_mapping, "get_model_conversion_mapping"):
        pytest.skip("this transformers has no get_model_conversion_mapping")
    return conversion_mapping


def _unpatched_mapping_fn():
    """The upstream function, underneath every wrapper installed over it.

    `__wrapped__` alone is not enough: unsloth_zoo patches the same function
    (`temporary_patches/moe_utils_bnb4bit.patch_bnb4bit_model_conversion_mapping`, which
    prepends per-expert converters for quantized MoE) and keeps its original in a closure
    cell rather than in `__wrapped__`. A test that stopped at the first wrapper would be
    measuring this fix through this fix, and would report no pathology on a transformers
    that really has one.
    """
    fn = _conversion_mapping().get_model_conversion_mapping
    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        nxt = getattr(fn, "__wrapped__", None)
        if nxt is None:
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    candidate = cell.cell_contents
                except ValueError:
                    continue
                if callable(candidate) and getattr(candidate, "__name__", "") == (
                    "get_model_conversion_mapping"
                ):
                    nxt = candidate
                    break
        if nxt is None:
            return fn
        fn = nxt
    return fn


def _meta_model(model_type, shrink, auto_class):
    """A real transformers model with no memory behind it.

    Built on `meta` so a composite multimodal model costs milliseconds and no VRAM. Its
    parameter NAMES are the entire subject of this fix, and those are identical to a
    materialised model's.
    """
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING:
        pytest.skip(f"this transformers has no {model_type} model type")
    factory = getattr(transformers, auto_class, None)
    if factory is None:
        pytest.skip(f"this transformers has no {auto_class}")
    config = CONFIG_MAPPING[model_type]()
    shrink(config)
    try:
        with torch.device("meta"):
            return factory.from_config(config)
    except Exception as exc:
        pytest.skip(f"cannot build a meta {model_type}: {exc!r}")


def _shrink_qwen3_5(config):
    text = config.text_config
    text.num_hidden_layers = 2
    text.layer_types = ["linear_attention", "full_attention"]
    if hasattr(text, "mtp_num_hidden_layers"):
        text.mtp_num_hidden_layers = 0
    config.vision_config.depth = 1


@pytest.fixture
def composite_model():
    """A composite Qwen3.5: a text `PreTrainedModel` nested under `model.language_model`."""
    model = _meta_model("qwen3_5", _shrink_qwen3_5, "AutoModelForImageTextToText")
    names = {name for name, _ in model.named_parameters(remove_duplicate = False)}
    assert any(name.startswith("model.language_model.") for name in names), (
        "this model is not the nested shape the fix is about"
    )
    return model


@pytest.fixture
def standalone_text_model():
    """The model the offending renaming was actually written for."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "qwen3_5_text" not in CONFIG_MAPPING:
        pytest.skip("this transformers has no qwen3_5_text model type")
    config = CONFIG_MAPPING["qwen3_5_text"]()
    config.num_hidden_layers = 2
    config.layer_types = ["linear_attention", "full_attention"]
    if hasattr(config, "mtp_num_hidden_layers"):
        config.mtp_num_hidden_layers = 0
    try:
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(config)
    except Exception as exc:
        pytest.skip(f"cannot build a meta qwen3_5_text: {exc!r}")


def _destructive_renamings(model, conversions):
    """Which of `conversions` rewrite this model's own parameter names off the map?

    This is the pathology itself, measured rather than assumed: it is what makes the
    bitsandbytes sidecars land on keys the model does not have.
    """
    WeightRenaming = _weight_renaming()

    keys = {name for name, _ in model.named_parameters(remove_duplicate = False)}
    keys |= {name for name, _ in model.named_buffers(remove_duplicate = False)}
    found = []
    for conversion in conversions:
        if not isinstance(conversion, WeightRenaming):
            continue
        for key in sorted(keys):
            renamed, matched = conversion.rename_source_key(key)
            if matched is not None and renamed != key and renamed not in keys:
                found.append((conversion, key, renamed))
                break
    return found


# --- the pattern helpers -----------------------------------------------------------


def test_prefixed_pattern_keeps_a_start_anchor_anchored():
    assert _prefixed_pattern("^model.language_model.", "model.language_model") == (
        "^model.language_model.model.language_model."
    )


def test_prefixed_pattern_does_not_invent_an_anchor():
    assert _prefixed_pattern("model.", "model.language_model") == "model.language_model.model."


def test_signature_reads_the_unprocessed_patterns_when_they_are_kept():
    """`__post_init__` rewrites the live patterns, so two copies must still compare equal."""
    WeightRenaming = _weight_renaming()

    one = WeightRenaming(source_patterns = r"^a.b.", target_patterns = r"^a.(?!b.)")
    two = WeightRenaming(source_patterns = r"^a.b.", target_patterns = r"^a.(?!b.)")
    assert _renaming_signature(one) == _renaming_signature(two)
    assert _renaming_signature(one) != _renaming_signature(
        WeightRenaming(source_patterns = r"^a.c.", target_patterns = r"^a.(?!c.)")
    )


# --- the discriminator -------------------------------------------------------------


def test_a_renaming_that_lands_on_a_real_key_is_not_destructive():
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(source_patterns = r"^old.", target_patterns = "new.")
    assert not _renaming_destroys_keys(renaming, ["old.w"], {"new.w"})


def test_a_renaming_that_lands_off_the_map_is_destructive():
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(source_patterns = r"^old.", target_patterns = "new.")
    assert _renaming_destroys_keys(renaming, ["old.w"], {"old.w"})


def test_a_renaming_that_matches_nothing_is_not_destructive():
    """The standalone case in miniature: the entry exists for checkpoint keys, not model keys."""
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(source_patterns = r"^old.", target_patterns = "new.")
    assert not _renaming_destroys_keys(renaming, ["other.w"], {"other.w"})


def test_one_landing_on_a_real_key_outvotes_the_rest():
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(source_patterns = r"^old.", target_patterns = "new.")
    assert not _renaming_destroys_keys(renaming, ["old.a", "old.b"], {"new.a"})


# --- the re-scoping ----------------------------------------------------------------


def test_rescoped_renaming_matches_upstreams_doubled_prefix_semantics():
    """What `PrefixChange.with_submodel_prefix` produces, on a transformers without one."""
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(
        source_patterns = r"^model.language_model.", target_patterns = r"^model.(?!language_model.)"
    )
    real = "model.language_model.layers.0.mlp.up_proj.weight"
    doubled = "model.language_model.model.language_model.layers.0.mlp.up_proj.weight"
    rescoped = _rescoped_renaming(renaming, "model.language_model", [real], {real})
    assert rescoped is not None
    assert rescoped.rename_source_key(real) == (real, None)
    assert rescoped.rename_source_key(doubled)[0] == (
        "model.language_model.model.layers.0.mlp.up_proj.weight"
    )


def test_rescoping_refuses_a_replacement_that_is_still_destructive():
    """An unanchored renaming cannot be scoped away, and must not be claimed to be."""
    WeightRenaming = _weight_renaming()

    renaming = WeightRenaming(source_patterns = r"\.gate\.", target_patterns = ".router.")
    key = "model.language_model.layers.0.gate.weight"
    assert _rescoped_renaming(renaming, "model.language_model", [key], {key}) is None


# --- the composite model, end to end -----------------------------------------------


def test_the_probe_agrees_with_what_this_transformers_really_does(composite_model):
    """The install gate must answer for the behaviour, on whatever version is installed."""
    conversions = _unpatched_mapping_fn()(composite_model)
    leaks = _destructive_renamings(composite_model, conversions)
    rescopes = _transformers_rescopes_submodule_prefix_renamings()
    assert bool(leaks) != bool(rescopes), (
        f"probe says rescopes={rescopes} but the real mapping "
        f"{'does' if leaks else 'does not'} rewrite this model's own weight names "
        f"({[(c.source_patterns, k, r) for c, k, r in leaks]})"
    )


def test_rescoping_removes_every_destructive_renaming_from_a_composite(composite_model):
    """The fix itself: after `_rescope_conversions`, nothing rewrites a real key off the map.

    Fails on an unfixed transformers without the patch -- that is the whole point -- and
    passes on a fixed one, where the input already had no leak and the output is the input.
    """
    conversions = _unpatched_mapping_fn()(composite_model)
    rescoped = _rescope_conversions(composite_model, conversions)
    assert _destructive_renamings(composite_model, rescoped) == []


def test_rescoping_keeps_every_conversion_that_was_not_destructive(composite_model):
    """Only the leaked entries may change: everything else must come back identical."""
    conversions = _unpatched_mapping_fn()(composite_model)
    leaked, _ = _leaked_submodule_prefix_renamings(composite_model)
    rescoped = _rescope_conversions(composite_model, conversions)

    kept = [c for c in conversions if _renaming_signature(c) not in leaked]
    survivors = [_renaming_signature(c) for c in rescoped]
    for conversion in kept:
        assert _renaming_signature(conversion) in survivors
    assert len(rescoped) == len(conversions)


def test_the_standalone_text_model_keeps_its_own_renaming(standalone_text_model):
    """The entry belongs to this model, and the fix must never take it away.

    `^model.language_model.` -> `^model.` is how the standalone text model reads a
    checkpoint saved from the composite one. Fired against its own weight names it matches
    nothing, which is exactly why the discriminator leaves it alone.
    """
    conversions = _unpatched_mapping_fn()(standalone_text_model)
    leaked, _ = _leaked_submodule_prefix_renamings(standalone_text_model)
    assert leaked == {}
    rescoped = _rescope_conversions(standalone_text_model, conversions)
    assert [_renaming_signature(c) for c in rescoped] == [
        _renaming_signature(c) for c in conversions
    ]


def test_a_non_composite_model_is_untouched():
    """A flat text-only model has no nested `PreTrainedModel`, so there is nothing to leak."""
    model = _meta_model(
        "llama", lambda config: setattr(config, "num_hidden_layers", 2), "AutoModelForCausalLM"
    )
    conversions = _unpatched_mapping_fn()(model)
    leaked, _ = _leaked_submodule_prefix_renamings(model)
    assert leaked == {}
    assert _rescope_conversions(model, conversions) is conversions


# --- installation ------------------------------------------------------------------


def test_installation_is_gated_on_the_probe():
    conversion_mapping = _conversion_mapping()
    fix_transformers_composite_prefix_renaming()
    installed = getattr(
        conversion_mapping.get_model_conversion_mapping, _COMPOSITE_PREFIX_RENAMING_FLAG, False
    )
    assert installed == (not _transformers_rescopes_submodule_prefix_renamings())


def test_the_patch_is_idempotent_and_undoable():
    conversion_mapping = _conversion_mapping()
    if _transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers already re-scopes, so nothing is installed to undo")
    fix_transformers_composite_prefix_renaming()
    first = conversion_mapping.get_model_conversion_mapping
    fix_transformers_composite_prefix_renaming()
    assert conversion_mapping.get_model_conversion_mapping is first, (
        "a second call wrapped the wrapper"
    )
    original = first.__wrapped__
    assert not getattr(original, _COMPOSITE_PREFIX_RENAMING_FLAG, False)
    conversion_mapping.get_model_conversion_mapping = original
    try:
        assert conversion_mapping.get_model_conversion_mapping is original
    finally:
        conversion_mapping.get_model_conversion_mapping = first


def test_the_patch_rebinds_the_copies_other_modules_imported():
    """`from .conversion_mapping import get_model_conversion_mapping` holds the object."""
    conversion_mapping = _conversion_mapping()
    if _transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers already re-scopes, so nothing is installed")
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "get_model_conversion_mapping"):
        pytest.skip("this transformers' modeling_utils does not hold its own binding")
    fix_transformers_composite_prefix_renaming()
    assert getattr(
        modeling_utils.get_model_conversion_mapping, _COMPOSITE_PREFIX_RENAMING_FLAG, False
    )


def test_the_wrapper_returns_the_upstream_mapping_when_it_cannot_reason(composite_model):
    """A model it cannot walk must cost the caller nothing but the upstream answer."""
    conversion_mapping = _conversion_mapping()
    if _transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers already re-scopes, so nothing is installed")
    fix_transformers_composite_prefix_renaming()

    class Unwalkable:
        """Walks for upstream, refuses to walk for us."""

        config = composite_model.config

        def modules(self):
            return iter(())

        def named_modules(self, *args, **kwargs):
            raise RuntimeError("no")

        def named_parameters(self, *args, **kwargs):
            raise RuntimeError("no")

        def named_buffers(self, *args, **kwargs):
            raise RuntimeError("no")

    through_patch = conversion_mapping.get_model_conversion_mapping(Unwalkable())
    upstream = _unpatched_mapping_fn()(Unwalkable())
    assert [_renaming_signature(c) for c in through_patch] == [
        _renaming_signature(c) for c in upstream
    ]
