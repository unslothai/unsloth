# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A config that subclasses ``SFTConfig`` must survive ``SFTTrainer.__init__``.

Replacing ``trl.trainer.sft_trainer.SFTConfig`` leaves two live classes of that
name. TRL guards with ``isinstance(args, TrainingArguments) and not
isinstance(args, SFTConfig)`` and rebuilds the config when it fires; against our
class that test is true for every config still derived from the pristine one, so
a ``GKDConfig`` was rebuilt as a plain SFT config and lost ``lmbda``, ``beta``,
``temperature``, ``teacher_model_name_or_path`` and the rest. See
unslothai/unsloth#1941.

Lifted out of ``unsloth/models/rl.py`` with ``ast`` the same way
``_rl_source`` does, so this stays CPU-only and needs neither torch nor trl.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"

NAMES = ("_is_unsloth_patched_config", "_widen_sft_config_instance_check")
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
    namespace: dict = {"inspect": inspect, "logger": logging.getLogger("unsloth-repro")}
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


NS = _load()


@pytest.fixture
def trl_like(monkeypatch):
    """Rebuild the two-classes-of-one-name situation without importing trl.

    The helper resolves the module with ``import trl.trainer.sft_trainer``, which
    reads the attribute off the parent package, so stand in a whole fake chain
    rather than only the leaf entry in ``sys.modules``.
    """
    import sys
    import types

    class TrainingArguments:
        pass

    class SFTConfig(TrainingArguments):  # the pristine class
        pass

    class GKDConfig(SFTConfig):  # a TRL config that builds on SFT
        def __init__(
            self,
            lmbda = 0.5,
            beta = 0.5,
        ):
            self.lmbda = lmbda
            self.beta = beta

    # What the compiler installs over the pristine class.
    patched = type(
        "SFTConfig",
        (SFTConfig,),
        {NS["_UNSLOTH_PATCHED_CONFIG_FLAG"]: True},
    )

    module = types.ModuleType("trl.trainer.sft_trainer")
    module.SFTConfig = patched
    trainer_pkg = types.ModuleType("trl.trainer")
    trainer_pkg.__path__ = []
    trainer_pkg.sft_trainer = module
    trl_pkg = types.ModuleType("trl")
    trl_pkg.__path__ = []
    trl_pkg.trainer = trainer_pkg

    monkeypatch.setitem(sys.modules, "trl", trl_pkg)
    monkeypatch.setitem(sys.modules, "trl.trainer", trainer_pkg)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", module)
    return TrainingArguments, SFTConfig, GKDConfig, patched, module


def _guard_fires(args, module):
    """TRL's own predicate, verbatim from ``SFTTrainer.__init__``."""
    TrainingArguments = args.__class__.__mro__[-2]
    return isinstance(args, TrainingArguments) and not isinstance(args, module.SFTConfig)


def test_subclass_config_is_downcast_without_the_fix(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    gkd = GKDConfig(lmbda = 0.25, beta = 0.75)
    assert isinstance(gkd, TrainingArguments)
    # This is the bug: a GKDConfig is not an instance of the installed SFTConfig.
    assert not isinstance(gkd, module.SFTConfig)


def test_widening_stops_the_downcast(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    assert NS["_widen_sft_config_instance_check"](patched) is True

    gkd = GKDConfig(lmbda = 0.25, beta = 0.75)
    assert isinstance(gkd, module.SFTConfig), "the guard still downcasts a GKDConfig"
    # The subclass keeps its own fields, which is the whole point.
    assert gkd.lmbda == 0.25 and gkd.beta == 0.75


def test_plain_training_arguments_are_still_converted(trl_like):
    """The guard exists to convert a bare TrainingArguments; keep that working."""
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert not isinstance(TrainingArguments(), module.SFTConfig)


def test_instances_of_the_installed_config_still_match(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert isinstance(patched(), module.SFTConfig)
    assert isinstance(pristine(), module.SFTConfig)


def test_calling_the_name_still_builds_the_unsloth_config(trl_like):
    """Widening must not cost the generated config: SFTConfig(...) is still ours."""
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    built = module.SFTConfig()
    assert isinstance(built, patched)
    assert getattr(built, NS["_UNSLOTH_PATCHED_CONFIG_FLAG"], False)


def test_is_idempotent(trl_like):
    """``patch_trl_rl_trainers`` can run more than once; do not stack shims."""
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    assert NS["_widen_sft_config_instance_check"](patched) is True
    first = module.SFTConfig
    assert NS["_widen_sft_config_instance_check"](patched) is False
    assert module.SFTConfig is first


def test_noop_when_the_module_still_holds_the_pristine_class(trl_like):
    """Nothing was replaced, so the guard already behaves and we leave it alone."""
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    module.SFTConfig = pristine
    assert NS["_widen_sft_config_instance_check"](patched) is False
    assert module.SFTConfig is pristine


def test_shim_keeps_the_module_and_name_it_stands_in_for(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert module.SFTConfig.__name__ == patched.__name__
    assert module.SFTConfig.__qualname__ == patched.__qualname__
    assert module.SFTConfig.__module__ == patched.__module__


def test_subclasscheck_is_widened_too(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    assert issubclass(GKDConfig, module.SFTConfig)
    assert not issubclass(TrainingArguments, module.SFTConfig)
