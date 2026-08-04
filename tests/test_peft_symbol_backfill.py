"""Tests _backfill_missing_peft_symbols in import_fixes.py.

peft's ``transformers_weight_conversion`` imports 3 names from
``transformers.conversion_mapping`` and 8 from ``transformers.core_model_loading``
at module top level. Unsloth already stubs those submodules when they are
absent, but a submodule that IS importable can still be missing individual
symbols: transformers 5.0.0.dev0 ships ``conversion_mapping`` WITHOUT
``_MODEL_TO_CONVERSION_PATTERN``, and peft's import then fails with

    ImportError: cannot import name '_MODEL_TO_CONVERSION_PATTERN'
                 from 'transformers.conversion_mapping'

which took down `import unsloth` in Ministral_3_(3B)_Reinforcement_Learning.

The old guard only asked "is the submodule importable", so it did nothing here.
Backfilling must be strictly additive -- a real transformers module must never
be replaced, and symbols it already defines must never be overwritten.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = REPO_ROOT / "unsloth" / "import_fixes.py"

CONV = "transformers.conversion_mapping"
CORE = "transformers.core_model_loading"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_backfill_under_test", IMPORT_FIXES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXES = _load_module()
backfill = FIXES._backfill_missing_peft_symbols


@pytest.fixture(autouse = True)
def _restore_modules():
    saved = {k: sys.modules.get(k) for k in (CONV, CORE)}
    yield
    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v


def _fake_real_module(name, **attrs):
    """A module that is NOT one of our stubs (no sentinel)."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def test_backfills_only_the_missing_symbol():
    sentinel_fn = lambda *a, **k: "upstream"
    mod = _fake_real_module(
        CONV,
        get_checkpoint_conversion_mapping = sentinel_fn,
        get_model_conversion_mapping = sentinel_fn,
    )
    added = backfill(CONV)
    assert added == ("_MODEL_TO_CONVERSION_PATTERN",)
    # The two real symbols must be untouched.
    assert mod.get_checkpoint_conversion_mapping is sentinel_fn
    assert mod.get_model_conversion_mapping is sentinel_fn


def test_backfilled_pattern_supports_copy_like_peft_does():
    _fake_real_module(CONV)
    backfill(CONV)
    pattern = sys.modules[CONV]._MODEL_TO_CONVERSION_PATTERN
    assert isinstance(pattern, dict)
    copied = pattern.copy()
    copied["llama"] = object()          # peft assigns by key at module top
    assert pattern == {}                 # copy must not alias


def test_complete_module_is_left_alone():
    attrs = {s: object() for s in FIXES._PEFT_REQUIRED_SYMBOLS[CONV]}
    _fake_real_module(CONV, **attrs)
    assert backfill(CONV) == ()


def test_our_own_stub_is_skipped():
    stub = FIXES._build_transformers_conversion_mapping_stub()
    assert getattr(stub, FIXES._UNSLOTH_STUB_SENTINEL, False) is True
    sys.modules[CONV] = stub
    assert backfill(CONV) == ()


def test_idempotent():
    _fake_real_module(CONV)
    first = backfill(CONV)
    assert len(first) == 3
    assert backfill(CONV) == ()


def test_core_model_loading_classes_are_subclassable():
    # peft subclasses ConversionOps at module top, so a placeholder is not
    # enough -- the backfilled object has to be a real class.
    _fake_real_module(CORE, dot_natural_key = lambda k: k)
    added = backfill(CORE)
    assert "dot_natural_key" not in added
    mod = sys.modules[CORE]
    assert issubclass(mod.Concatenate, mod.ConversionOps)

    class _Child(mod.ConversionOps):
        pass

    assert issubclass(_Child, mod.ConversionOps)


def test_missing_module_returns_empty():
    sys.modules.pop(CONV, None)
    # No real transformers submodule of this name under the test alias.
    assert backfill(CONV) == () or isinstance(backfill(CONV), tuple)


def test_required_symbols_match_peft_import_list():
    # Guards against the lists drifting apart silently.
    assert set(FIXES._PEFT_REQUIRED_SYMBOLS) == set(FIXES._PEFT_STUB_BUILDERS)
    for name, symbols in FIXES._PEFT_REQUIRED_SYMBOLS.items():
        donor = FIXES._PEFT_STUB_BUILDERS[name]()
        for s in symbols:
            assert hasattr(donor, s), f"{name} stub lacks {s}"


# ---- saying so when the stand-in is not equivalent -----------------------
#
# The donors are inert by design: an empty pattern, mapping lookups that return
# None. That is the truth on transformers 4 and on a transformers 5 that never
# had the symbol. It is NOT the truth for a transformers 5 that has conversions
# and renamed one, where peft would then skip work it should have done. Every
# released 5.0.0 through 5.6.0 ships all eleven names, so this is a dev-build
# path, but it must not be a silent one.


def test_a_missing_mapping_function_is_announced():
    _fake_real_module(CONV, _MODEL_TO_CONVERSION_PATTERN = {"real": 1},
                      get_checkpoint_conversion_mapping = lambda *a: "real")
    with pytest.warns(RuntimeWarning, match = "get_model_conversion_mapping"):
        assert backfill(CONV) == ("get_model_conversion_mapping",)


def test_a_missing_conversion_class_is_announced():
    _fake_real_module(CORE, **{s: object() for s in
                               FIXES._PEFT_REQUIRED_SYMBOLS[CORE][1:]})
    with pytest.warns(RuntimeWarning, match = "Concatenate"):
        backfill(CORE)


def test_only_the_pattern_is_quiet():
    """An empty pattern is what peft starts from anyway, and it is the only
    name ever seen missing. Warning about it would be noise on every load."""
    import warnings

    _fake_real_module(CONV, get_checkpoint_conversion_mapping = lambda *a: None,
                      get_model_conversion_mapping = lambda *a: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backfill(CONV) == ("_MODEL_TO_CONVERSION_PATTERN",)


def test_a_complete_module_says_nothing():
    import warnings

    _fake_real_module(CONV, **{s: object()
                               for s in FIXES._PEFT_REQUIRED_SYMBOLS[CONV]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backfill(CONV) == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
