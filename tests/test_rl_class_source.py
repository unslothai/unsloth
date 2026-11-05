# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_class_source (unsloth/models/rl_replacements.py) must equal inspect.getsource byte for byte.

The helper is lifted via ast so the test needs neither a GPU nor `import unsloth`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import pkgutil
import re
import sys
from pathlib import Path

import pytest

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _load_class_source():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_class_source"
    )
    namespace = {"ast": ast, "inspect": inspect, "re": re, "sys": sys}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_class_source"]


_class_source = _load_class_source()


def _module_classes(module):
    for value in vars(module).values():
        if isinstance(value, type) and value.__module__ == module.__name__:
            yield value


def _pristine(name):
    """Re-execute a TRL module from its file: conftest's `import unsloth` may already have
    rebound the patched trainers to generated Unsloth* copies, so the module dict is not TRL's."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    package, _, leaf = name.rpartition(".")
    pristine_name = f"{package}._unsloth_pristine_{leaf}"
    pristine_spec = importlib.util.spec_from_file_location(pristine_name, spec.origin)
    module = importlib.util.module_from_spec(pristine_spec)
    module.__package__ = package
    sys.modules[pristine_name] = module
    try:
        pristine_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(pristine_name, None)
        return None
    return module


@pytest.fixture(scope = "module")
def trl_modules():
    pytest.importorskip("trl")
    import trl.trainer

    names = set()
    for package in ("trl.trainer", "trl.experimental"):
        try:
            pkg = importlib.import_module(package)
        except Exception:
            continue
        for info in pkgutil.walk_packages(pkg.__path__, package + "."):
            leaf = info.name.rsplit(".", 1)[-1]
            if leaf.endswith(("_trainer", "_config")):
                names.add(info.name)
    modules = {}
    for name in sorted(names):
        module = _pristine(name)
        if module is not None:
            modules[name] = module
    yield modules
    for module in modules.values():
        sys.modules.pop(module.__name__, None)


def test_trl_trainer_and_config_classes_match_getsource(trl_modules):
    checked = 0
    for module in trl_modules.values():
        for cls in _module_classes(module):
            try:
                expected = inspect.getsource(cls)
            except (OSError, TypeError):
                with pytest.raises((OSError, TypeError)):
                    _class_source(cls)
                continue
            assert _class_source(cls) == expected, f"{module.__name__}.{cls.__qualname__}"
            checked += 1
    # The patched trainers (SFT, GRPO, DPO, KTO, ...) and their configs are all in here.
    assert checked >= 30, checked


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("trl.trainer.sft_trainer", "SFTTrainer"),
        ("trl.trainer.sft_config", "SFTConfig"),
        ("trl.trainer.grpo_trainer", "GRPOTrainer"),
        ("trl.trainer.grpo_config", "GRPOConfig"),
        ("trl.trainer.dpo_trainer", "DPOTrainer"),
        ("trl.trainer.dpo_config", "DPOConfig"),
    ],
)
def test_patched_trl_classes_take_fast_path(trl_modules, monkeypatch, module_name, class_name):
    module = trl_modules.get(module_name)
    if module is None or not hasattr(module, class_name):
        pytest.skip(f"{module_name}.{class_name} not in this TRL")
    cls = getattr(module, class_name)
    expected = inspect.getsource(cls)
    fallbacks = []
    real = inspect.getsource

    def spy(obj):
        fallbacks.append(obj)
        return real(obj)

    monkeypatch.setattr(inspect, "getsource", spy)
    assert _class_source(cls) == expected
    # The fast path must actually run on Python < 3.13 (no silent fallback).
    if sys.version_info < (3, 13):
        assert fallbacks == []


def test_unsloth_efficient_grpo_matches_getsource(monkeypatch):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    zoo_rl = pytest.importorskip("unsloth_zoo.rl_replacements")
    cls = zoo_rl.RL_REPLACEMENTS["UnslothEfficientGRPO"]
    expected = inspect.getsource(cls)
    fallbacks = []
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: fallbacks.append(obj) or real(obj))
    assert _class_source(cls) == expected
    if sys.version_info < (3, 13):
        assert fallbacks == []
    monkeypatch.undo()
    for cls in _module_classes(zoo_rl):
        assert _class_source(cls) == inspect.getsource(cls), cls.__qualname__


SYNTHETIC = '''
import functools


def deco(cls):
    return cls


def deco_factory(*args, **kwargs):
    return lambda cls: cls


class Plain:
    def method(self):
        return 1


@deco
@functools.total_ordering
class Decorated:
    def __lt__(self, other):
        return True


@deco_factory(1, key = 2)
class DecoratedCall:
    x = 1

    @staticmethod
    def method():
        return 1


@deco_factory(
    1,
    key = 2,
)
class MultiLineDecorator:
    def method(self):
        return 1


@deco

class BlankAfterDecorator:
    def method(self):
        return 1


@deco
# a comment between decorator and class
class CommentAfterDecorator:
    def method(self):
        return 1


class Outer:
    class Nested:
        def method(self):
            return 1


class Duplicate:
    def method(self):
        return 1


Duplicate_first = Duplicate


class Duplicate:
    def method(self):
        return 2


DOC = """
class InString:
    pass
"""


if True:
    class InString:
        def method(self):
            return 3


class NoMethods:
    pass


TABLE = dict(
    a = 1,
)


class AfterCall:
    def method(self):
        return 4


STUB_ONLY = False
if STUB_ONLY:
    class Stubbed:
        def method(self):
            return 0


class Stubbed:
    def method(self):
        return 5
'''


@pytest.fixture(scope = "module")
def synthetic(tmp_path_factory):
    directory = tmp_path_factory.mktemp("class_source")
    (directory / "unsloth_class_source_fixture.py").write_text(SYNTHETIC, encoding = "utf-8")
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module("unsloth_class_source_fixture")
        yield module
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("unsloth_class_source_fixture", None)


@pytest.mark.parametrize(
    "name",
    [
        "Plain",
        "Decorated",
        "DecoratedCall",
        "MultiLineDecorator",
        "BlankAfterDecorator",
        "CommentAfterDecorator",
        "Duplicate",
        "InString",
        "NoMethods",
        "AfterCall",
        "Stubbed",
    ],
)
def test_synthetic_matches_getsource(synthetic, name):
    cls = getattr(synthetic, name)
    assert _class_source(cls) == inspect.getsource(cls)


def test_synthetic_nested_and_duplicate(synthetic):
    assert _class_source(synthetic.Outer.Nested) == inspect.getsource(synthetic.Outer.Nested)
    first = synthetic.Duplicate_first
    assert _class_source(first) == inspect.getsource(first)


@pytest.mark.skipif(sys.version_info >= (3, 13), reason = "3.13+ delegates to inspect.getsource")
def test_ambiguous_cases_fall_back(synthetic, monkeypatch):
    calls = []
    real = inspect.getsource

    def spy(obj):
        calls.append(obj.__qualname__)
        return real(obj)

    monkeypatch.setattr(inspect, "getsource", spy)
    for cls in (
        synthetic.Outer.Nested,  # nested qualname
        synthetic.Duplicate,  # two column-0 class lines
        synthetic.MultiLineDecorator,  # multi-line decorator
        synthetic.BlankAfterDecorator,  # decorator separated by a blank line
        synthetic.CommentAfterDecorator,  # decorator separated by a comment
        synthetic.InString,  # unique column-0 hit is inside a string, methods disagree
        synthetic.NoMethods,  # no method to confirm the match
        synthetic.Stubbed,  # an indented stub earlier in the file is what inspect returns
    ):
        calls.clear()
        _class_source(cls)
        assert calls == [cls.__qualname__], cls.__qualname__
    for cls in (
        synthetic.Plain,
        synthetic.Decorated,
        synthetic.DecoratedCall,
        synthetic.AfterCall,  # column-0 `)` above: steps up to the statement start
    ):
        calls.clear()
        _class_source(cls)
        assert calls == [], cls.__qualname__
