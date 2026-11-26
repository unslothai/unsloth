# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL="ep=N" with inherited legacy variables (UNSLOTH_FSDP2 / TP / PP / CP), and native FSDP2
refusing an expert parallel model, without a process group."""

import pytest
import torch.nn as nn

import unsloth  # noqa: F401
from unsloth import parallel_env as pe
from unsloth.models import fsdp2_native as fn


@pytest.mark.parametrize(
    "name, value",
    [
        ("UNSLOTH_FSDP2", "1"),
        ("UNSLOTH_FSDP2", "native"),
        ("UNSLOTH_TP_SIZE", "2"),
        ("UNSLOTH_PP_SIZE", "2"),
    ],
)
def test_ep_refuses_inherited_parallel_variables(name, value):
    env = {"WORLD_SIZE": "2", name: value}
    with pytest.raises(ValueError, match = "expert parallel"):
        pe.translate(pe.parse_parallel_spec("ep=2"), env)


def test_ep_refuses_an_inherited_cp_the_flag_does_not_give():
    # ep x cp composes, but only as the flag gives it ("ep=2,cp=2"): a disagreeing UNSLOTH_CP_SIZE is refused.
    with pytest.raises(ValueError, match = "UNSLOTH_CP_SIZE=2 asks for something else"):
        pe.translate(pe.parse_parallel_spec("ep=2"), {"WORLD_SIZE": "2", "UNSLOTH_CP_SIZE": "2"})


@pytest.mark.parametrize(
    "env",
    [
        {"WORLD_SIZE": "2"},
        {"WORLD_SIZE": "2", "UNSLOTH_FSDP2": "0"},
        {"WORLD_SIZE": "2", "UNSLOTH_TP_SIZE": "1", "UNSLOTH_PP_SIZE": "1", "UNSLOTH_CP_SIZE": "1"},
    ],
)
def test_ep_accepts_inactive_inherited_variables(env):
    out = pe.translate(pe.parse_parallel_spec("ep=2"), env)
    assert out["UNSLOTH_EP_SIZE"] == "2"


def test_ep_refuses_spec_fsdp():
    with pytest.raises(ValueError, match = "expert parallel"):
        pe.translate(pe.parse_parallel_spec("ep=2,fsdp=1"), {"WORLD_SIZE": "2"})


def test_native_fsdp2_refuses_expert_parallel_model():
    inner = nn.Linear(2, 2)
    model = nn.Module()
    model.model = inner
    inner.__dict__["_unsloth_ep"] = object()
    with pytest.raises(RuntimeError, match = "expert parallel"):
        fn._refuse_tensor_parallel_model(model)
    with pytest.raises(RuntimeError, match = "expert parallel"):
        fn._refuse_tensor_parallel_model(model, mesh = object(), compose = True)
    assert not fn._refuse_tensor_parallel_model(nn.Linear(2, 2))
