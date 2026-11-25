# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CP beside EP is refused before trainer __init__: CP would build its own mesh over ranks with different experts."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

_ENV = ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_EP_SIZE", "UNSLOTH_FSDP2", "UNSLOTH_CP_SIZE")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    return monkeypatch


def test_cp_with_ep_env_is_refused(clean_env):
    clean_env.setenv("UNSLOTH_EP_SIZE", "2")
    with pytest.raises(NotImplementedError, match = "expert parallel \\(UNSLOTH_EP_SIZE=2\\)"):
        cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))


def test_cp_on_an_expert_parallel_model_is_refused(clean_env):
    model = torch.nn.Linear(2, 2)
    model.__dict__["_unsloth_ep"] = object()  # apply_expert_parallel's marker (its ep mesh)
    with pytest.raises(NotImplementedError, match = "expert parallel \\(apply_expert_parallel\\)"):
        cp._refuse_model_parallel_cp(2, model)


def test_cp_alone_still_runs_the_registry(clean_env):
    cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2))
