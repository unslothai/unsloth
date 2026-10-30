# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL / unsloth.parallelize and the context parallel x tensor / pipeline parallel refusals, without a
process group (tests/test_parallel_cp_compose_gloo.py trains the composed layouts on gloo ranks)."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_FSDP2",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_PP_MICROBATCHES",
    "WORLD_SIZE",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    return monkeypatch


def _fake_sft(monkeypatch, inits):
    import trl

    class Trainer:
        def __init__(self, model = None, args = None, **kwargs):
            inits.append(model)
            self.model = model
            self.args = args

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    return Trainer


@pytest.mark.parametrize(
    "env, axis",
    [
        ({"UNSLOTH_TP_SIZE": "2"}, "tensor parallel (UNSLOTH_TP_SIZE=2)"),
        ({"UNSLOTH_PP_SIZE": "2"}, "pipeline parallel (UNSLOTH_PP_SIZE=2)"),
        ({"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2"}, "pipeline parallel (UNSLOTH_PP_SIZE=2) and tensor"),
    ],
)
@pytest.mark.parametrize("via", ["config", "kwarg"])
def test_cp_with_tp_or_pp_env_is_refused_before_the_trainer_init(clean_env, env, axis, via):
    import types

    for key, value in env.items():
        clean_env.setenv(key, value)
    inits = []
    Trainer = _fake_sft(clean_env, inits)
    model = torch.nn.Linear(2, 2)
    if via == "config":
        call = lambda: Trainer(model = model, args = types.SimpleNamespace(context_parallel_size = 2))
    else:
        call = lambda: Trainer(model, types.SimpleNamespace(context_parallel_size = 1), context_parallel_size = 2)
    with pytest.raises(NotImplementedError, match = "does not compose with") as info:
        call()
    assert axis in str(info.value), str(info.value)
    assert inits == [], "refused after the trainer __init__ ran (the model may already be split)"


def test_cp_alone_and_tp_without_cp_are_not_refused(clean_env):
    import types

    inits = []
    Trainer = _fake_sft(clean_env, inits)
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    Trainer(model = torch.nn.Linear(2, 2), args = types.SimpleNamespace(context_parallel_size = 1))
    assert len(inits) == 1
