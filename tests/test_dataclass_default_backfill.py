"""_backfill_dataclass_defaults must not shadow an inherited default.

The fix gives class-local bare annotations a `None` default so a transformers
5 config subclass survives the dataclass ordering rule. Deciding "no default
yet" with `name not in cls.__dict__` was wrong: a subclass re-annotating an
inherited field without assigning has a default already, through the MRO.

The helper is plain Python, so these reproduce it directly and run on
transformers 4 too. import_fixes.py is loaded by file spec because
`import unsloth.import_fixes` would run unsloth/__init__.py first, pulling in
torch, numpy and unsloth_zoo.
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
ifx = _MODULE  # for monkeypatching its internals


def test_an_inherited_default_is_not_shadowed():
    """Absent from cls.__dict__ but present through the MRO: writing None over
    it would change the config's value."""

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
    already a default."""

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
    Compiled under a filename not on disk, so `inspect.getsource` is what
    fails, not the class object."""
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
    """The change was backported, so where the source can be read it decides
    and the version is only a fallback."""
    _pretend_transformers_is(monkeypatch, "5.4.0")
    assert _transformers_configs_are_kw_only(_KwOnlyConfig) is True
    _pretend_transformers_is(monkeypatch, "5.5.1")
    assert _transformers_configs_are_kw_only(_OrderedConfig) is False


def test_unreadable_source_on_5_5_1_stands_down(monkeypatch):
    """Returning False here would patch a transformers that already passes
    kw_only=True, giving a None default to required keyword fields."""
    _pretend_transformers_is(monkeypatch, "5.5.1")
    assert _transformers_configs_are_kw_only(_config_without_readable_source())


def test_unreadable_source_falls_back_to_probing_the_behaviour(monkeypatch):
    """With no source to read, ask the installed transformers by trying it.
    Not a version check: the ordering rule was backported, so a number labels
    the wrong builds."""
    monkeypatch.setattr(
        ifx,
        "_transformers_needs_bare_annotation_fix",
        lambda: True,
    )
    assert not _transformers_configs_are_kw_only(_config_without_readable_source())
    monkeypatch.setattr(
        ifx,
        "_transformers_needs_bare_annotation_fix",
        lambda: False,
    )
    assert _transformers_configs_are_kw_only(_config_without_readable_source())


def test_the_probe_answers_from_the_class_not_the_version(monkeypatch):
    """Faking `transformers.__version__` used to change the answer, which
    mislabels a backported install."""
    for version in ("5.4.0", "5.5.0", "5.5.0.post1", "4.57.6", "9.9.9"):
        _pretend_transformers_is(monkeypatch, version)
        assert ifx._transformers_needs_bare_annotation_fix() is False, version


def test_the_probe_detects_a_config_that_raises(monkeypatch):
    """An `__init_subclass__` that rejects the bare-annotation shape is what
    the fix exists for, whatever version it calls itself."""
    class Raising:
        def __init_subclass__(cls, **kwargs):
            raise TypeError("non-default argument follows default argument")

    module = types.ModuleType("transformers.configuration_utils")
    module.PretrainedConfig = Raising
    monkeypatch.setitem(sys.modules, "transformers.configuration_utils", module)
    assert ifx._transformers_needs_bare_annotation_fix() is True


def test_an_unreadable_version_does_not_raise(monkeypatch):
    """Dev builds and vendored forks can carry a version packaging will not
    parse, and a missing transformers must not raise out of a probe either."""
    for version in ("not-a-version", "", None):
        _pretend_transformers_is(monkeypatch, version)
        assert _transformers_configs_are_kw_only(_config_without_readable_source()), repr(version)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert _transformers_configs_are_kw_only(_config_without_readable_source())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
