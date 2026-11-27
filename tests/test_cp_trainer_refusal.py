# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_CP_SIZE on a trainer without a context parallel wrapper: the refusal names the real cause (CPU backend,
non-SFT / DPO / GRPO trainer, failed patched-trainer build), and the build skips unsloth_zoo's CPU trainer quietly."""

import sys
import types

import pytest
import torch

import unsloth  # noqa: F401


@pytest.fixture
def cp_env(monkeypatch):
    for name in ("UNSLOTH_PARALLEL", "UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_ALLOW_CPU"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_CP_SIZE", "2")
    return monkeypatch


def _refusal(trainer_cls):
    from unsloth.models import _utils

    trainer = object.__new__(trainer_cls)
    with pytest.raises(NotImplementedError) as info:
        _utils._parallel_capabilities_pre(trainer, trainer_cls.__init__, (), {}, None)
    return str(info.value)


def _sft_like():
    import trl

    class SFTTrainer(trl.SFTTrainer):  # what TRL's own class looks like: no CP wrapper on it
        pass

    SFTTrainer.__unsloth_context_parallel__ = False
    return SFTTrainer


def test_cpu_backend_names_the_backend_not_sfttrainer(cp_env):
    cp_env.setitem(sys.modules, "unsloth._cpu_init", types.ModuleType("unsloth._cpu_init"))
    message = _refusal(_sft_like())
    assert "CPU backend" in message and "no GPU was found" in message, message
    assert "Instead: launch on GPUs" in message and "use SFTTrainer" not in message, message


def test_sft_trainer_whose_patched_build_failed_names_the_build_error(cp_env):
    from unsloth.models import rl

    cp_env.delitem(sys.modules, "unsloth._cpu_init", raising = False)
    cp_env.setitem(
        rl.PATCHED_TRAINER_BUILD_ERRORS, "sft_trainer", "OSError: source code not available"
    )
    message = _refusal(_sft_like())
    assert (
        "Unsloth's patched SFTTrainer could not be built (OSError: source code not available)"
        in message
    )
    assert "use SFTTrainer" not in message, message


def test_allow_cpu_names_the_variable(cp_env):
    cp_env.delitem(sys.modules, "unsloth._cpu_init", raising = False)
    cp_env.setenv("UNSLOTH_ALLOW_CPU", "1")
    message = _refusal(_sft_like())
    assert "UNSLOTH_ALLOW_CPU=1 skips Unsloth's trainer patches" in message, message


def test_patched_trainer_build_skips_the_zoo_cpu_trainer(monkeypatch):
    """unsloth_zoo.cpu replaces trl.trainer.sft_trainer.SFTTrainer with a subclass inspect cannot read; the build
    used to fail on it with "OSError: source code not available" and warn. It now keeps it, quietly."""
    from trl.trainer import sft_trainer

    from unsloth.models import rl

    base = sft_trainer.SFTTrainer
    while getattr(base, "_unsloth_cpu", False) or base.__name__.startswith("Unsloth"):
        base = base.__mro__[1]
    namespace = {"_unsloth_cpu": True, "__module__": "unsloth_zoo.cpu.trainer"}
    cpu_cls = type("SFTTrainer", (base,), namespace)  # no __firstlineno__, as in the rehearsal
    with pytest.raises((OSError, TypeError)):
        __import__("inspect").getsource(cpu_cls)
    monkeypatch.setattr(sft_trainer, "SFTTrainer", cpu_cls)
    warnings = []
    monkeypatch.setattr(rl.logger, "warning_once", lambda msg, *a, **k: warnings.append(msg))
    monkeypatch.setattr(rl, "PATCHED_TRAINER_BUILD_ERRORS", {}, raising = False)
    rl._patch_trl_rl_trainers("sft_trainer")
    assert not [w for w in warnings if "Could not build" in w], warnings
    assert sft_trainer.SFTTrainer is cpu_cls
    assert "CPU backend" in rl.PATCHED_TRAINER_BUILD_ERRORS.get("sft_trainer", "")
