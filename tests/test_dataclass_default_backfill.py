"""_backfill_dataclass_defaults must not shadow an inherited default.

The fix gives class-local bare annotations a `None` default so a transformers
5 config subclass survives the dataclass ordering rule. It decided "no default
yet" with `name not in cls.__dict__`, which is class-local: a subclass that
re-annotates an inherited field without assigning to it has a default already,
through the MRO, and `None` overwrote it.

Reproduced directly, so these run on transformers 4 as well -- the helper is
plain Python and needs no config machinery, unlike the live tests next door.
For the same reason import_fixes.py is loaded by file spec, as
test_peft_symbol_backfill.py does: `import unsloth.import_fixes` would run
unsloth/__init__.py first, which pulls _gpu_init and with it torch, numpy and
unsloth_zoo, so a dependency-light run would fail before reaching the helper.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = REPO_ROOT / "unsloth" / "import_fixes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_dataclass_under_test", IMPORT_FIXES
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module()
_backfill_dataclass_defaults = _MODULE._backfill_dataclass_defaults
_transformers_configs_are_kw_only = _MODULE._transformers_configs_are_kw_only


def test_an_inherited_default_is_not_shadowed():
    """A subclass may re-annotate an inherited field without assigning to it.
    The name is then absent from cls.__dict__ but present through the MRO, and
    writing None over it changes the config's value."""

    class Base:
        window: int = 7

    class Child(Base):
        window: int  # re-annotated, no assignment

    assert _backfill_dataclass_defaults(Child) == []
    assert Child.window == 7


def test_a_genuinely_new_field_is_still_backfilled():
    """The narrowing must not disarm the fix."""

    class Base:
        window: int = 7

    class Child(Base):
        vision_config: object  # new, and bare: the case that raises

    assert _backfill_dataclass_defaults(Child) == ["vision_config"]
    assert Child.vision_config is None


def test_an_inherited_method_of_the_same_name_counts_too():
    """dataclass reads defaults with getattr, so anything the MRO resolves is
    already a default, whatever it is."""

    class Base:
        def helper(self):
            return 1

    class Child(Base):
        helper: object

    assert _backfill_dataclass_defaults(Child) == []
    assert Child().helper() == 1


class _KwOnlyConfig:
    """Stands in for transformers 5.5.1+, whose hook passes kw_only=True."""

    def __init_subclass__(cls, **kwargs):
        _fake_dataclass(cls, kw_only = True)


class _OrderedConfig:
    """Stands in for 5.4.0 to 5.5.0, whose hook does not."""

    def __init_subclass__(cls, **kwargs):
        _fake_dataclass(cls)


def _fake_dataclass(cls, **kwargs):
    return cls


def _config_without_readable_source():
    """A hook whose source cannot be read, as on a stripped or frozen install.

    Compiling under a filename that is not on disk is the honest reproduction:
    the class object is fine, `inspect.getsource` is what fails.
    """
    namespace = {}
    exec(
        compile(
            "class Config:\n    def __init_subclass__(cls, **kwargs):\n        pass\n",
            "<unsloth-test-no-source-on-disk>",
            "exec",
        ),
        namespace,
    )
    return namespace["Config"]


def _pretend_transformers_is(monkeypatch, version):
    module = types.ModuleType("transformers")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_the_source_beats_the_version_fallback(monkeypatch):
    """The version is only a fallback. Where the source can be read it decides,
    because the change was backported and the version alone cannot see that."""
    _pretend_transformers_is(monkeypatch, "5.4.0")
    assert _transformers_configs_are_kw_only(_KwOnlyConfig) is True
    _pretend_transformers_is(monkeypatch, "5.5.1")
    assert _transformers_configs_are_kw_only(_OrderedConfig) is False


def test_unreadable_source_on_5_5_1_stands_down(monkeypatch):
    """The case that matters. Returning False here would patch a transformers
    that already passes kw_only=True, and the backfill would then give a None
    default to fields upstream requires as keyword arguments."""
    _pretend_transformers_is(monkeypatch, "5.5.1")
    assert _transformers_configs_are_kw_only(_config_without_readable_source())


def test_unreadable_source_inside_the_window_still_patches(monkeypatch):
    """5.4.0 to 5.5.0 is where the fix is needed, source readable or not."""
    for version in ("5.4.0", "5.5.0", "5.5.0.post1"):
        _pretend_transformers_is(monkeypatch, version)
        assert not _transformers_configs_are_kw_only(_config_without_readable_source()), version


def test_an_unreadable_version_does_not_raise(monkeypatch):
    """Dev builds, git suffixes and vendored forks can carry a version string
    packaging will not parse, and a missing transformers must not raise out of
    a probe either. No evidence of the window means stand down."""
    for version in ("not-a-version", "", None):
        _pretend_transformers_is(monkeypatch, version)
        assert _transformers_configs_are_kw_only(_config_without_readable_source()), repr(version)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert _transformers_configs_are_kw_only(_config_without_readable_source())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
