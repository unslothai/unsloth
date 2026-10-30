# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism beside UNSLOTH_TP_SIZE (or a model split by hand) is refused before the trainer __init__,
without a process group (tests/test_parallel_cp_compose_gloo.py trains the layouts on gloo ranks)."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

_ENV = ("UNSLOTH_TP_SIZE", "UNSLOTH_FSDP2", "UNSLOTH_CP_SIZE", "WORLD_SIZE")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    return monkeypatch


def _fake(monkeypatch, inits, kind):
    class Trainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            inits.append(model)
            self.model = model
            self.args = args

    if kind == "sft":
        import trl
        monkeypatch.setattr(trl, "SFTTrainer", Trainer)
        cp.patch_sft_trainer()
    else:
        cp.patch_rl_trainer(Trainer, kind)
    return Trainer


def _config(size, **kwargs):
    return types.SimpleNamespace(
        context_parallel_size = size, context_parallel_backend = "ulysses", **kwargs
    )


@pytest.mark.parametrize("kind", ["sft", "dpo_trainer", "grpo_trainer"])
@pytest.mark.parametrize("via", ["config", "kwarg", "positional"])
def test_cp_with_tp_env_is_refused_before_the_trainer_init(clean_env, kind, via):
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    if kind != "sft" and via == "kwarg":
        pytest.skip("the RL trainers read context_parallel_size from their config only")
    inits = []
    Trainer = _fake(clean_env, inits, kind)
    model = torch.nn.Linear(2, 2)
    if via == "config":
        call = lambda: Trainer(model = model, args = _config(2))
    elif via == "positional":
        call = lambda: Trainer(model, _config(2))
    else:
        call = lambda: Trainer(model, _config(1), context_parallel_size = 2)
    with pytest.raises(NotImplementedError, match = "does not compose with") as info:
        call()
    assert "tensor parallel (UNSLOTH_TP_SIZE=2)" in str(info.value), str(info.value)
    assert inits == [], "refused after the trainer __init__ ran (the model may already be split)"


@pytest.mark.parametrize("split", ["is_tensor_parallel", "is_pipeline_parallel"])
def test_cp_on_a_model_split_by_hand_is_refused(clean_env, split):
    from unsloth.models import pp_native, tp_native

    module = tp_native if split == "is_tensor_parallel" else pp_native
    clean_env.setattr(module, split, lambda model: True)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    with pytest.raises(NotImplementedError, match = "apply_tensor_parallel|apply_pipeline_parallel"):
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    assert inits == []


@pytest.mark.parametrize(
    "env, size",
    [
        ({}, 2),
        ({"UNSLOTH_FSDP2": "1"}, 2),
        ({"UNSLOTH_TP_SIZE": "1"}, 2),
        ({"UNSLOTH_TP_SIZE": "2"}, 1),
    ],
    ids = ["cp_alone", "cp_fsdp2", "tp1_cp2", "tp2_cp1"],
)
def test_cp_alone_with_fsdp2_and_tp_without_cp_are_not_refused(clean_env, monkeypatch, env, size):
    for key, value in env.items():
        clean_env.setenv(key, value)
    # Past the refusal, the CP __init__ needs a process group; stop at the original __init__.
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    try:
        Trainer(model = torch.nn.Linear(2, 2), args = _config(size))
    except NotImplementedError as e:
        assert "does not compose with" not in str(e), str(e)
    except Exception:
        pass
    assert len(inits) == 1


@pytest.mark.parametrize("raw", ["+2", " 2 ", "02", "4"])
def test_cp_with_any_tp_env_spelling_tp_env_accepts_is_refused(clean_env, raw):
    """tp_env reads UNSLOTH_TP_SIZE with int(): every value it turns into tp > 1 is refused beside CP."""
    from unsloth.models import tp_env

    clean_env.setenv("UNSLOTH_TP_SIZE", raw)
    assert tp_env._int_env("UNSLOTH_TP_SIZE") > 1
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    with pytest.raises(NotImplementedError, match = "does not compose with"):
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    assert inits == []


@pytest.mark.parametrize("raw", ["\u00b2", "2.0", "abc"])
def test_cp_with_a_tp_env_value_tp_env_rejects_is_left_to_tp_env(clean_env, monkeypatch, raw):
    """A value tp_env refuses is not parsed into a crash here: tp_env names it itself."""
    clean_env.setenv("UNSLOTH_TP_SIZE", raw)
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    try:
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    except Exception:
        pass
    assert len(inits) == 1
