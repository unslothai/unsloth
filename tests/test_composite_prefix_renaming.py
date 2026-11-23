# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers 5.4 merges submodule renamings into composite models, dropping bitsandbytes sidecars."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from unsloth.import_fixes import (  # noqa: E402
    _COMPOSITE_PREFIX_RENAMING_FLAG,
    _composite_prefix_renaming_repaired,
    _leaked_submodule_prefix_renamings,
    _next_in_wrapper_chain,
    _prefixed_pattern,
    _renaming_destroys_keys,
    _renaming_signature,
    _rescope_conversions,
    _rescoped_renaming,
    _transformers_rescopes_submodule_prefix_renamings,
    fix_transformers_composite_prefix_renaming,
)


def _skip_a_stand_in(module):
    """Skips an unsloth stand-in: a real module is a file on disk, while a stub's __file__ is not."""
    path = getattr(module, "__file__", None)
    if not os.path.isfile(path or ""):
        pytest.skip(
            f"{module.__name__} is a stand-in ({path!r}), not a module this transformers ships"
        )


def _weight_renaming():
    """transformers 4.x has no `core_model_loading`, and so no pathology to test."""
    try:
        from transformers import core_model_loading
        from transformers.core_model_loading import WeightRenaming
    except Exception:
        pytest.skip("this transformers has no core_model_loading.WeightRenaming")
    _skip_a_stand_in(core_model_loading)
    return WeightRenaming


def _conversion_mapping():
    try:
        from transformers import conversion_mapping
    except Exception:
        pytest.skip("this transformers has no conversion_mapping module")
    _skip_a_stand_in(conversion_mapping)
    if not hasattr(conversion_mapping, "get_model_conversion_mapping"):
        pytest.skip("this transformers has no get_model_conversion_mapping")
    return conversion_mapping


def _unpatched_mapping_fn():
    """Unwraps to the upstream function: zoo keeps its original in a closure, not __wrapped__."""
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
    """Real transformers model on meta: its parameter names match a materialised one at no VRAM cost."""
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
    """Conversions that rename a model's own parameter names; these strand bitsandbytes sidecars."""
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
    """After _rescope_conversions, no conversion may rename a real key; fails on unpatched transformers."""
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
    """The standalone text model's renaming reads composite-saved checkpoints; the fix must keep it."""
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


@pytest.fixture
def forced_install(monkeypatch):
    """Forces the install gate open so install tests are not vacuous on releases the probe declines."""
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

    conversion_mapping = _conversion_mapping()
    live = conversion_mapping.get_model_conversion_mapping
    holders = [
        (module, module.__dict__["get_model_conversion_mapping"])
        for module in list(sys.modules.values())
        if isinstance(getattr(module, "__dict__", None), dict)
        and "get_model_conversion_mapping" in module.__dict__
    ]
    # Start from upstream's own function: within 5.4.0-5.5.4 importing unsloth already installed
    # this repair under zoo's MoE wrapper, which publishes no `__wrapped__`.
    chain = []
    function = live
    while function is not None and len(chain) < 8:
        chain.append(function)
        function = _next_in_wrapper_chain(function)
    upstream = chain[-1]
    assert not getattr(upstream, _COMPOSITE_PREFIX_RENAMING_FLAG, False)
    # Through monkeypatch so this undo runs last and leaves `live` in place.
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", upstream)
    for module, binding in holders:
        if any(binding is link for link in chain):
            monkeypatch.setattr(module, "get_model_conversion_mapping", upstream)
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
    _conversion_mapping()
    fix_transformers_composite_prefix_renaming()
    # Walk the whole chain: zoo's MoE wrapper can sit on top of the repair.
    assert _composite_prefix_renaming_repaired() == (
        not _transformers_rescopes_submodule_prefix_renamings()
    )


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

    # From 5.6 upstream walks `named_modules` itself and raises here too, so there is nothing to keep.
    try:
        upstream = _unpatched_mapping_fn()(Unwalkable())
    except Exception as e:
        pytest.skip(f"upstream cannot walk this object either on this release ({e!r})")
    through_patch = conversion_mapping.get_model_conversion_mapping(Unwalkable())
    assert [_renaming_signature(c) for c in through_patch] == [
        _renaming_signature(c) for c in upstream
    ]


def test_a_module_holding_the_pre_zoo_function_is_still_rebound(monkeypatch, forced_install):
    """Zoo's patch has no __wrapped__, so a module holding the pre-zoo function still needs rebinding."""
    import types

    from transformers import conversion_mapping

    import_fixes = forced_install

    # Importing unsloth may already have installed this patch.
    pristine = conversion_mapping.get_model_conversion_mapping
    while getattr(pristine, import_fixes._COMPOSITE_PREFIX_RENAMING_FLAG, False):
        unwrapped = getattr(pristine, "__wrapped__", None)
        if unwrapped is None:
            break
        pristine = unwrapped

    # Zoo's wrapper as zoo spells it: no functools.wraps, no __wrapped__, just its marker.
    def zoo_wrapper(
        model,
        key_mapping = None,
        hf_quantizer = None,
        add_legacy = True,
    ):
        return pristine(model, key_mapping, hf_quantizer, add_legacy)

    zoo_wrapper._unsloth_moe_patched = True
    assert not hasattr(zoo_wrapper, "__wrapped__"), "this test models zoo's real wrapper"
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", zoo_wrapper)

    early = types.ModuleType("transformers._unsloth_test_early_importer")
    early.get_model_conversion_mapping = pristine
    monkeypatch.setitem(sys.modules, early.__name__, early)

    import_fixes.fix_transformers_composite_prefix_renaming()

    patched = conversion_mapping.get_model_conversion_mapping
    assert patched is not zoo_wrapper, "the repair declined even with the gate forced open"
    assert (
        early.get_model_conversion_mapping is patched
    ), "a module holding the pre-zoo function was left bound to the unscoped mapping"


def test_a_third_party_wrapper_is_kept_in_the_chain(forced_install):
    """Install on top of third-party wrappers; unwrapping via __wrapped__ would drop one from the chain."""
    import functools

    conversion_mapping = _conversion_mapping()
    # Unwrapped, and without copying __dict__: inheriting this repair's mark would make it decline.
    live = _unpatched_mapping_fn()
    calls = []

    @functools.wraps(live)
    def third_party(*args, **kwargs):
        calls.append(1)
        return live(*args, **kwargs)

    third_party.__dict__.pop(_COMPOSITE_PREFIX_RENAMING_FLAG, None)
    conversion_mapping.get_model_conversion_mapping = third_party
    forced_install.fix_transformers_composite_prefix_renaming()

    patched = conversion_mapping.get_model_conversion_mapping
    assert getattr(patched, _COMPOSITE_PREFIX_RENAMING_FLAG, False), "the repair declined"
    assert patched.__wrapped__ is third_party, "the third-party wrapper was discarded"

    @functools.wraps(patched)
    def someone_else(*args, **kwargs):
        return patched(*args, **kwargs)

    conversion_mapping.get_model_conversion_mapping = someone_else
    forced_install.fix_transformers_composite_prefix_renaming()
    assert conversion_mapping.get_model_conversion_mapping is someone_else


def test_a_vllm_module_holding_its_own_copy_is_rebound(monkeypatch, forced_install):
    """vLLM imports this function by value at import time, so any copy it holds must be rebound too."""
    import types

    conversion_mapping = _conversion_mapping()
    # An earlier test can leave a stand-in in place, so wrap the live object.
    backend = types.ModuleType("vllm.model_executor.models.transformers.base")
    backend.get_model_conversion_mapping = conversion_mapping.get_model_conversion_mapping
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    forced_install.fix_transformers_composite_prefix_renaming()

    patched = conversion_mapping.get_model_conversion_mapping
    assert getattr(patched, _COMPOSITE_PREFIX_RENAMING_FLAG, False), "the repair declined"
    assert (
        backend.get_model_conversion_mapping is patched
    ), "vllm's own copy was left bound to the unscoped mapping"


@pytest.mark.parametrize(
    "module_name",
    [
        "some_user_notebook_helper",
        "transformers._unsloth_test_unrelated",
        "peft._unsloth_test_unrelated",
        "unsloth_zoo._unsloth_test_unrelated",
        "unsloth._unsloth_test_unrelated",
        "vllm._unsloth_test_unrelated",
    ],
)
def test_an_unrelated_module_keeps_its_own_same_named_function(
    monkeypatch, forced_install, module_name
):
    """The sweep rebinds only true aliases of the upstream function, not unrelated same-named helpers."""
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
    """Unsloth defers to unsloth_zoo's copy of the repair so only one wrapper is installed."""
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

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
        "_unsloth_patched_composite_prefix_renaming",
        False,
    )


def test_it_finds_the_zoo_mark_under_an_unmarked_wrapper(monkeypatch):
    """The detector walks the whole wrapper chain, since zoo's MoE wrapper publishes no __wrapped__."""
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

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
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

    def a():
        pass

    def b():
        pass

    a.__wrapped__ = b
    b.__wrapped__ = a

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", a)
    assert import_fixes._zoo_composite_prefix_renaming_installed() is False


def test_both_probes_see_the_repair_under_the_real_moe_wrapper(monkeypatch):
    """A __wrapped__ on zoo's MoE wrapper would make zoo replace it and drop its converters."""
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    moe = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    if not hasattr(moe, "patch_bnb4bit_model_conversion_mapping"):
        pytest.skip("this unsloth_zoo has no MoE conversion-mapping patch")
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

    def zoo_repair(*args, **kwargs):
        pass

    setattr(zoo_repair, "_unsloth_zoo_patched_composite_prefix_renaming", True)

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", zoo_repair)
    moe.patch_bnb4bit_model_conversion_mapping()
    live = conversion_mapping.get_model_conversion_mapping
    if live is zoo_repair:
        pytest.skip("the MoE patch declined to install on this transformers")

    assert (
        getattr(live, "__wrapped__", None) is None
    ), "the MoE wrapper must not publish __wrapped__; see the docstring"
    if getattr(live, "_unsloth_wrapper_inner", None) is None:
        # An older unsloth_zoo publishes no link, so the repair is invisible; skip, not fail.
        pytest.skip("this unsloth_zoo's MoE wrapper publishes no link to what it wrapped")

    assert import_fixes._zoo_composite_prefix_renaming_installed() is True
    assert import_fixes._composite_prefix_renaming_repaired() is True


def test_the_chain_walk_is_bounded(monkeypatch):
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    # Reuse the loaded module: re-importing runs `unsloth/__init__`, which needs unsloth_zoo.
    import_fixes = sys.modules[_transformers_rescopes_submodule_prefix_renamings.__module__]

    def a():
        pass

    def b():
        pass

    a.__wrapped__ = b
    b.__wrapped__ = a

    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", a)
    assert import_fixes._zoo_composite_prefix_renaming_installed() is False
    assert import_fixes._composite_prefix_renaming_repaired() is False
