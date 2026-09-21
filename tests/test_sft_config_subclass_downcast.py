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

    # What the compiler installs over the pristine class. `_patch_config_pickle_identity`
    # gives it the pristine class's module and name so `torch.save(trainer.args, ...)`
    # keeps working, and binds it wherever the pristine class used to live.
    patched = type(
        "SFTConfig",
        (SFTConfig,),
        {NS["_UNSLOTH_PATCHED_CONFIG_FLAG"]: True},
    )
    # The pristine class's home is `trl.trainer.sft_config`, a DIFFERENT module
    # from `trl.trainer.sft_trainer` where the guard reads the name. Model both,
    # or a shim installed only at the guard's module looks reachable by pickle
    # when it is not.
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
    # The top level name is the same object, exactly as after patching.
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
    # This is the bug: a GKDConfig is not an instance of the installed SFTConfig.
    assert not isinstance(gkd, module.SFTConfig)


def test_widening_stops_the_downcast(trl_like):
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    assert NS["_widen_sft_config_instance_check"](patched) is True

    gkd = GKDConfig(lmbda = 0.25, beta = 0.75)
    assert isinstance(gkd, module.SFTConfig), "the guard still downcasts a GKDConfig"
    # The subclass keeps its own fields, which is the whole point.
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
    """``torch.save(trainer.args, ...)`` must keep working.

    Pickle stores a class as ``__module__`` + ``__qualname__`` and refuses unless
    the object living there IS the class, which is why
    ``_patch_config_pickle_identity`` exists. A shim that advertises the displaced
    class's home while the displaced class still sits there makes
    ``Trainer._save_checkpoint`` raise ``PicklingError``, so the shim has to take
    that attribute over.
    """
    TrainingArguments, pristine, GKDConfig, patched, module, trl_pkg = trl_like
    NS["_widen_sft_config_instance_check"](patched)
    shim = module.SFTConfig
    home = sys.modules[shim.__module__]
    # Exactly the identity check pickle performs before it will save the class.
    assert getattr(home, shim.__qualname__) is shim


def test_only_the_guards_own_module_is_rebound(trl_like):
    """Widening takes over one attribute, not every binding of the name.

    Rebinding `trl.SFTConfig` and the class's home module as well was tried and
    is worse: the generated class stops being what `trl.SFTConfig` resolves to,
    which is an invariant other patching passes and their tests rely on
    (tests/python/test_rl_config_pickling.py, and the pristine-class walk in
    tests/version_compat/). Only the module holding TRL's guard needs widening.
    """
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
    """A walk back to TRL's pristine class must not stop on the shim.

    Callers find the pristine class with
    ``while "_unsloth_patched_rl_config" in cls.__dict__``, which is the right
    test because the generated subclass is renamed onto TRL's own name. The
    marker has to be in the shim's own __dict__, not inherited, or that walk
    stops on the shim and reads back Unsloth's field set as if it were TRL's.
    """
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
