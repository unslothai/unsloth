# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed NATIVE (sd.cpp) replacement load must hand GPU residency back to the displaced account.

sd.cpp can be selected on a GPU host (UNSLOTH_DIFFUSION_ENGINE=sd_cpp), where /images/load takes
the arbiter with ``acquire_for_request(DIFFUSION, _begin_load)`` before the background load runs.
That claim moves ``gpu_arbiter._owner_account`` to the requester, so when the load then fails with
the previous account's pipeline still resident, the failure path has to undo it -- the diffusers
and video backends both call ``restore_owner_account`` there.
"""

from __future__ import annotations

import types

import pytest

from auth import policy
from core.inference import gpu_arbiter
from core.inference.sd_cpp_backend import SdCppDiffusionBackend, _SdLoading
from hub.services.models import account_access as access
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_resident_components", {}, raising = False)
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(access, "_uncommitted_resident", {}, raising = False)
    monkeypatch.setattr(access, "_uncommitted_components", {}, raising = False)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)
    yield


def _failing_backend(monkeypatch):
    backend = SdCppDiffusionBackend(engine = None)
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_resolve_backend",
        lambda self: (_ for _ in ()).throw(
            RuntimeError("sd-cli binary is present but not runnable.")
        ),
    )
    # Alice's native pipeline is what stays resident through the failure.
    backend._state = types.SimpleNamespace(repo_id = "org/alice-model")
    backend._loading = _SdLoading(repo_id = "org/bob-model", base_repo = "")
    return backend


def test_failed_native_load_returns_gpu_residency_to_the_displaced_account(monkeypatch):
    backend = _failing_backend(monkeypatch)
    # /images/load on a GPU host: the arbiter claim runs under Bob, displacing Alice.
    run_as(BOB, gpu_arbiter.acquire_for, "diffusion", lambda: None)
    run_as(BOB, access.note_resident_account, "diffusion", "org/bob-model")
    assert gpu_arbiter.owner_account() == BOB.account_id

    run_as(
        BOB,
        lambda: backend._run_load(
            repo_id = "org/bob-model",
            gguf_filename = "flux1-dev-Q4_K_M.gguf",
            base = "",
            fam = None,
            hf_token = None,
            _load_token = backend._load_token,
        ),
    )

    assert backend._loading.error  # the load really failed
    assert backend._state is not None  # Alice's pipeline is still resident
    assert gpu_arbiter.owner_account() == ALICE.account_id
    assert run_as(ALICE, access.resident_hidden, "diffusion", "org/alice-model") is False
    assert run_as(BOB, access.resident_hidden, "diffusion", "org/alice-model") is True


def test_failed_native_cpu_load_leaves_a_released_arbiter_alone(monkeypatch):
    """Single-user / CPU parity: the CPU path releases DIFFUSION, so the restore must no-op."""
    backend = _failing_backend(monkeypatch)
    gpu_arbiter.release("diffusion")
    run_as(
        BOB,
        lambda: backend._run_load(
            repo_id = "org/bob-model",
            gguf_filename = "flux1-dev-Q4_K_M.gguf",
            base = "",
            fam = None,
            hf_token = None,
            _load_token = backend._load_token,
        ),
    )
    assert gpu_arbiter.current_owner() is None
    assert gpu_arbiter.owner_account() is None
