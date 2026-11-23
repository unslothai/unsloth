# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An SFTConfig subclass such as GKDConfig must survive SFTTrainer.__init__ without being downcast."""

from __future__ import annotations

import ast
import copyreg
import inspect
import logging
import sys
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"

NAMES = (
    "_is_unsloth_patched_config",
    "_reduce_pristine_rl_config",
    "_config_reduction_is_safe",
    "_register_config_pickle_fallback",
    "_widen_sft_config_instance_check",
)
CONSTANTS = (
    "_UNSLOTH_PATCHED_CONFIG_FLAG",
    "_UNSLOTH_CONFIG_PICKLE_TARGET",
    "_UNSLOTH_SFT_CONFIG_SHIM_FLAG",
)


def _load():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets
        ):
            wanted.append(node)
    found = {n.name for n in wanted if isinstance(n, ast.FunctionDef)}
    missing = set(NAMES) - found
    if missing:
        raise AssertionError(f"missing module-level defs in {SOURCE_PATH}: {sorted(missing)}")
    namespace: dict = {
        "inspect": inspect,
        "logger": logging.getLogger("unsloth-repro"),
        "sys": sys,
        "copyreg": copyreg,
    }
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


NS = _load()


@pytest.fixture
def trl_like(monkeypatch):
    """The helper resolves ``import trl.trainer.sft_trainer`` through the parent package, so the whole chain has to be stood in, not just the leaf ``sys.modules`` entry."""
    import sys
    import types

    class TrainingArguments:
        pass

    class SFTConfig(TrainingArguments):
        pass

    class GKDConfig(SFTConfig):  # a TRL config that builds on SFT
        def __init__(
            self,
            lmbda = 0.5,
            beta = 0.5,
        ):
            self.lmbda = lmbda
            self.beta = beta

    # `_patch_config_pickle_identity` gives the generated class the pristine module and name.
    patched = type(
        "SFTConfig",
        (SFTConfig,),
        {NS["_UNSLOTH_PATCHED_CONFIG_FLAG"]: True},
    )
    # Home module (`sft_config`) differs from the guard's (`sft_trainer`); model both.
    patched.__module__ = "trl.trainer.sft_config"
    patched.__qualname__ = "SFTConfig"

    module = types.ModuleType("trl.trainer.sft_trainer")
    module.SFTConfig = patched
    config_module = types.ModuleType("trl.trainer.sft_config")
    config_module.SFTConfig = patched
    trainer_pkg = types.ModuleType("trl.trainer")
    trainer_pkg.__path__ = []
    trainer_pkg.sft_trainer = module
    trainer_pkg.sft_config = config_module
    trl_pkg = types.ModuleType("trl")
    trl_pkg.__path__ = []
    trl_pkg.trainer = trainer_pkg
    trl_pkg.SFTConfig = patched

    monkeypatch.setitem(sys.modules, "trl", trl_pkg)
    monkeypatch.setitem(sys.modules, "trl.trainer", trainer_pkg)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", module)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_config", config_module)
    try:
        yield TrainingArguments, SFTConfig, GKDConfig, patched, module, trl_pkg
    finally:
        for cls in (SFTConfig, GKDConfig, patched):
            copyreg.dispatch_table.pop(cls, None)


def _guard_fires(args, module):
    """TRL's own predicate, verbatim from ``SFTTrainer.__init__``."""
    TrainingArguments = args.__class__.__mro__[-2]
    return isinstance(args, TrainingArguments) and not isinstance(args, module.SFTConfig)


def test_subclass_config_is_downcast_without_the_fix(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    gkd = GKDConfig(lmbda = 0.25, beta = 0.75)
    assert isinstance(gkd, TrainingArguments)
    assert not isinstance(gkd, module.SFTConfig)


def test_widening_stops_the_downcast(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    assert NS["_widen_sft_config_instance_check"](patched) is True

    gkd = GKDConfig(lmbda = 0.25, beta = 0.75)
    assert isinstance(gkd, module.SFTConfig), "the guard still downcasts a GKDConfig"
    assert gkd.lmbda == 0.25 and gkd.beta == 0.75


def test_plain_training_arguments_are_still_converted(trl_like):
    """The guard exists to convert a bare TrainingArguments; keep that working."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert not isinstance(TrainingArguments(), module.SFTConfig)


def test_instances_of_the_installed_config_still_match(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert isinstance(patched(), module.SFTConfig)
    assert isinstance(pristine(), module.SFTConfig)


def test_calling_the_name_still_builds_the_unsloth_config(trl_like):
    """Widening must not cost the generated config: SFTConfig(...) is still ours."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    built = module.SFTConfig()
    assert isinstance(built, patched)
    assert getattr(built, NS["_UNSLOTH_PATCHED_CONFIG_FLAG"], False)


def test_is_idempotent(trl_like):
    """``patch_trl_rl_trainers`` can run more than once; do not stack shims."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    assert NS["_widen_sft_config_instance_check"](patched) is True
    first = module.SFTConfig
    assert NS["_widen_sft_config_instance_check"](patched) is False
    assert module.SFTConfig is first


def test_noop_when_the_module_still_holds_the_pristine_class(trl_like):
    """Nothing was replaced, so the guard already behaves and we leave it alone."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    module.SFTConfig = pristine
    assert NS["_widen_sft_config_instance_check"](patched) is False
    assert module.SFTConfig is pristine


def test_shim_keeps_the_name_and_answers_to_where_it_lives(trl_like):
    """Same name, but its OWN module: see the comment at the rebinding."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    shim = module.SFTConfig
    assert shim.__name__ == patched.__name__
    assert shim.__qualname__ == patched.__qualname__
    assert shim.__module__ == "trl.trainer.sft_trainer"
    assert patched.__module__ == "trl.trainer.sft_config", "the displaced class must not move"


def test_the_shim_is_reachable_by_pickle_under_its_own_module_and_name(trl_like):
    """Pickle refuses a class unless the object at its ``__module__`` plus ``__qualname__`` IS it, so a shim advertising the displaced class's home raises PicklingError from ``Trainer._save_checkpoint``."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    shim = module.SFTConfig
    home = sys.modules[shim.__module__]
    # Exactly the identity check pickle performs before it will save the class.
    assert getattr(home, shim.__qualname__) is shim


def test_only_the_guards_own_module_is_rebound(trl_like):
    """Rebinding every holder of the name is WRONG: the generated class must stay what ``trl.SFTConfig`` resolves to (tests/python/test_rl_config_pickling.py, and the pristine-class walk in tests/version_compat/)."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert module.SFTConfig is not patched, "the guard's module was not widened"
    assert trl_pkg.SFTConfig is patched, "the top level binding must not move"
    assert (
        sys.modules["trl.trainer.sft_config"].SFTConfig is patched
    ), "the class's home module must keep holding it, or it stops pickling"


def test_the_displaced_class_keeps_its_own_home_and_pickles(trl_like):
    """Both classes stay reachable by pickle under their own module and name."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    home = sys.modules[patched.__module__]
    assert getattr(home, patched.__qualname__) is patched


def test_the_shim_carries_the_patched_config_marker(trl_like):
    """Callers walk back with ``while "_unsloth_patched_rl_config" in cls.__dict__``, so an inherited marker is not enough: the walk would stop on the shim and read Unsloth's field set as TRL's."""
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    cls = module.SFTConfig
    assert NS["_UNSLOTH_PATCHED_CONFIG_FLAG"] in cls.__dict__
    while NS["_UNSLOTH_PATCHED_CONFIG_FLAG"] in cls.__dict__ or cls.__name__.startswith("Unsloth"):
        cls = cls.__bases__[0]
    assert cls is pristine


def test_subclasscheck_is_widened_too(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert issubclass(GKDConfig, module.SFTConfig)
    assert not issubclass(TrainingArguments, module.SFTConfig)
