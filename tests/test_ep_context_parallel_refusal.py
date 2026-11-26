# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CP beside EP goes through the capability registry with the expert parallel size (its ep x cp cells decide),
before trainer __init__; tensor / pipeline parallel beside CP keep their own refusal."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth import parallel_capabilities as caps

_ENV = ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_EP_SIZE", "UNSLOTH_FSDP2", "UNSLOTH_CP_SIZE")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    return monkeypatch


@pytest.fixture
def layouts(clean_env):
    seen = []
    clean_env.setattr(caps, "check_layout", lambda layout, *a, **k: seen.append(layout))
    return seen


def test_cp_with_ep_env_reaches_the_registry_with_ep(clean_env, layouts):
    clean_env.setenv("UNSLOTH_EP_SIZE", "2")
    cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))
    assert [(x.cp, x.ep, x.dp) for x in layouts] == [(2, 2, 1)]


def test_cp_on_an_expert_parallel_model_reaches_the_registry_with_ep(clean_env, layouts):
    model = torch.nn.Linear(2, 2)
    model.__dict__["_unsloth_ep"] = object()  # apply_expert_parallel's marker (its ep mesh)
    cp._refuse_model_parallel_cp(2, model)
    assert [(x.cp, x.ep) for x in layouts] == [(2, 2)]


def test_cp_with_ep_on_a_model_without_moe_blocks_is_refused(clean_env):
    clean_env.setenv("UNSLOTH_EP_SIZE", "2")
    with pytest.raises(NotImplementedError, match = "ep x model"):
        cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))


def test_cp_alone_still_runs_the_registry(clean_env, layouts):
    cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))
    assert [(x.cp, x.ep) for x in layouts] == [(2, 1)]


def test_cp_with_tensor_parallel_env_is_still_refused(clean_env):
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    with pytest.raises(NotImplementedError, match = "does not compose with tensor parallel"):
        cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))
