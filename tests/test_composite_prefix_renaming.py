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

import sys

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
    assert any(
        name.startswith("model.language_model.") for name in names
    ), "this model is not the nested shape the fix is about"
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


@pytest.fixture
def forced_install(monkeypatch):
    """Open the gate so the tests of what installation DOES are never vacuous.

    Outside the defect window the probe declines, which is the right answer and is what
    `test_installation_is_gated_on_the_probe` measures. It also meant every test of the
    installation itself either skipped or, worse, passed on a release where the code it
    names never ran: the sweep tests below reported green on 5.17.0 without sweeping
    anything. Forcing the gate open keeps the two questions apart -- whether to install,
    and whether installing is done correctly -- and lets the second one be answered on any
    release. Everything the installer rebinds is restored on the way out.
    """
    import unsloth.import_fixes as import_fixes

    conversion_mapping = _conversion_mapping()
    live = conversion_mapping.get_model_conversion_mapping
    holders = [
        (module, module.__dict__["get_model_conversion_mapping"])
        for module in list(sys.modules.values())
        if isinstance(getattr(module, "__dict__", None), dict)
        and "get_model_conversion_mapping" in module.__dict__
    ]
    monkeypatch.setattr(
        import_fixes, "_transformers_rescopes_submodule_prefix_renamings", lambda: False
    )
    try:
        yield import_fixes
    finally:
        conversion_mapping.get_model_conversion_mapping = live
        for module, binding in holders:
            module.get_model_conversion_mapping = binding


def test_installation_is_gated_on_the_probe():
    conversion_mapping = _conversion_mapping()
    fix_transformers_composite_prefix_renaming()
    installed = getattr(
        conversion_mapping.get_model_conversion_mapping, _COMPOSITE_PREFIX_RENAMING_FLAG, False
    )
    assert installed == (not _transformers_rescopes_submodule_prefix_renamings())


def test_the_patch_is_idempotent_and_undoable(forced_install):
    conversion_mapping = _conversion_mapping()
    forced_install.fix_transformers_composite_prefix_renaming()
    first = conversion_mapping.get_model_conversion_mapping
    forced_install.fix_transformers_composite_prefix_renaming()
    assert (
        conversion_mapping.get_model_conversion_mapping is first
    ), "a second call wrapped the wrapper"
    original = first.__wrapped__
    assert not getattr(original, _COMPOSITE_PREFIX_RENAMING_FLAG, False)
    conversion_mapping.get_model_conversion_mapping = original
    try:
        assert conversion_mapping.get_model_conversion_mapping is original
    finally:
        conversion_mapping.get_model_conversion_mapping = first


def test_the_patch_rebinds_the_copies_other_modules_imported(forced_install):
    """`from .conversion_mapping import get_model_conversion_mapping` holds the object."""
    conversion_mapping = _conversion_mapping()
    import transformers.modeling_utils as modeling_utils

    if "get_model_conversion_mapping" not in modeling_utils.__dict__:
        pytest.skip("this transformers' modeling_utils does not hold its own binding")
    forced_install.fix_transformers_composite_prefix_renaming()
    assert getattr(
        modeling_utils.get_model_conversion_mapping, _COMPOSITE_PREFIX_RENAMING_FLAG, False
    )


def test_the_wrapper_returns_the_upstream_mapping_when_it_cannot_reason(
    composite_model, forced_install
):
    """A model it cannot walk must cost the caller nothing but the upstream answer."""
    conversion_mapping = _conversion_mapping()
    forced_install.fix_transformers_composite_prefix_renaming()

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

    # Upstream has to survive this object for the question to mean anything: from 5.6 on it
    # walks `named_modules` itself to set `scope_prefix`, so it raises here too and there is
    # no upstream answer to preserve. That is a fact about the release, not about the fix.
    try:
        upstream = _unpatched_mapping_fn()(Unwalkable())
    except Exception as e:
        pytest.skip(f"upstream cannot walk this object either on this release ({e!r})")
    through_patch = conversion_mapping.get_model_conversion_mapping(Unwalkable())
    assert [_renaming_signature(c) for c in through_patch] == [
        _renaming_signature(c) for c in upstream
    ]


def test_a_module_holding_the_pre_zoo_function_is_still_rebound(monkeypatch, forced_install):
    """unsloth_zoo patches the same function first, WITHOUT `__wrapped__`.

    `temporary_patches/moe_utils_bnb4bit.py` sets only `_unsloth_moe_patched`, no
    functools.wraps, so the `getattr(original, "__wrapped__", original)` unwrap cannot see
    past it and `original` stays zoo's wrapper. A module that imported the name before zoo
    ran still holds the underlying upstream function and matches neither object, so an
    identity test leaves it bound to the unscoped mapping and
    `PeftAdapterMixin.load_adapter()` renames Qwen3.5 and Gemma 3n adapter keys away from
    their real `model.language_model.*` modules.
    """
    import types

    from transformers import conversion_mapping

    import_fixes = forced_install

    # Pin the starting state: importing unsloth may already have installed this patch.
    pristine = conversion_mapping.get_model_conversion_mapping
    while getattr(pristine, import_fixes._COMPOSITE_PREFIX_RENAMING_FLAG, False):
        unwrapped = getattr(pristine, "__wrapped__", None)
        if unwrapped is None:
            break
        pristine = unwrapped

    # Zoo's wrapper, spelled the way zoo really spells it: no functools.wraps, no
    # __wrapped__, just its own marker attribute.
    def zoo_wrapper(model, key_mapping = None, hf_quantizer = None, add_legacy = True):
        return pristine(model, key_mapping, hf_quantizer, add_legacy)

    zoo_wrapper._unsloth_moe_patched = True
    assert not hasattr(zoo_wrapper, "__wrapped__"), "this test models zoo's real wrapper"
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", zoo_wrapper)

    # A module that imported the function BEFORE zoo wrapped, so it holds `pristine`.
    early = types.ModuleType("transformers._unsloth_test_early_importer")
    early.get_model_conversion_mapping = pristine
    monkeypatch.setitem(sys.modules, early.__name__, early)

    import_fixes.fix_transformers_composite_prefix_renaming()

    patched = conversion_mapping.get_model_conversion_mapping
    assert patched is not zoo_wrapper, "the repair declined even with the gate forced open"
    assert early.get_model_conversion_mapping is patched, (
        "a module holding the pre-zoo function was left bound to the unscoped mapping"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "some_user_notebook_helper",
        "transformers._unsloth_test_unrelated",
        "peft._unsloth_test_unrelated",
        "unsloth_zoo._unsloth_test_unrelated",
        "unsloth._unsloth_test_unrelated",
    ],
)
def test_an_unrelated_module_keeps_its_own_same_named_function(
    monkeypatch, forced_install, module_name
):
    """The sweep must not touch a helper somebody else happens to call the same thing.

    The binding test cannot be an identity test, because unsloth_zoo wraps without
    `__wrapped__` and the pre-zoo upstream function matches neither object. So the sweep
    asks whether the binding IS an alias of the function it is replacing: `original`
    itself, something whose `__module__` is transformers' own `conversion_mapping`, or a
    wrapper carrying unsloth_zoo's marker. The package restriction alone is not enough --
    these four packages are large, and something inside their namespace may legitimately
    define its own helper under this name, which is why the cases below include modules
    inside them as well as outside.
    """
    import types

    def mine(*args, **kwargs):
        return "mine"

    outsider = types.ModuleType(module_name)
    outsider.get_model_conversion_mapping = mine
    monkeypatch.setitem(sys.modules, module_name, outsider)

    forced_install.fix_transformers_composite_prefix_renaming()

    assert outsider.get_model_conversion_mapping is mine
    assert outsider.get_model_conversion_mapping() == "mine"


def test_it_defers_to_the_unsloth_zoo_copy_of_the_same_repair(monkeypatch):
    """Two packages carry this repair; exactly one of them must install it.

    unsloth_zoo owns the bitsandbytes Linear4bit patch that reports the failure and is
    importable without unsloth, so it carries the same re-scope in
    `temporary_patches/conversion_mapping_rescope.py`. A second wrapper on top of the first
    is measurably inert -- the first pass leaves no leaked signature for the second to match
    -- but it is still a wrapper nobody needs, and one of the two has to yield. This one does.
    """
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    import unsloth.import_fixes as import_fixes

    if import_fixes._transformers_rescopes_submodule_prefix_renamings():
        pytest.skip("this transformers carries the upstream fix; neither copy installs")

    before = conversion_mapping.get_model_conversion_mapping

    def zoo_wrapper(*args, **kwargs):
        return before(*args, **kwargs)

    zoo_wrapper.__wrapped__ = before
    setattr(zoo_wrapper, "_unsloth_zoo_patched_composite_prefix_renaming", True)
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", zoo_wrapper)

    import_fixes.fix_transformers_composite_prefix_renaming()

    assert conversion_mapping.get_model_conversion_mapping is zoo_wrapper
    assert not getattr(
        conversion_mapping.get_model_conversion_mapping,
        "_unsloth_patched_composite_prefix_renaming", False,
    )


def test_it_finds_the_zoo_mark_under_an_unmarked_wrapper(monkeypatch):
    """unsloth_zoo's moe_utils_bnb4bit wraps the same function without `__wrapped__`.

    The detector therefore walks the whole chain rather than reading only the top object.
    """
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    import unsloth.import_fixes as import_fixes

    def zoo_repair():
        pass
    setattr(zoo_repair, "_unsloth_zoo_patched_composite_prefix_renaming", True)

    def moe_wrapper():
        pass
    moe_wrapper.__wrapped__ = zoo_repair

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", moe_wrapper)
    assert import_fixes._zoo_composite_prefix_renaming_installed() is True


def test_the_zoo_detector_cannot_spin_on_a_cycle(monkeypatch):
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    import unsloth.import_fixes as import_fixes

    def a():
        pass
    def b():
        pass
    a.__wrapped__ = b
    b.__wrapped__ = a

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", a)
    assert import_fixes._zoo_composite_prefix_renaming_installed() is False
