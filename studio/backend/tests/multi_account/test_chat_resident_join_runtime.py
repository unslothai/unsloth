# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Another account's ordinary load of the resident GGUF joins it even when its runtime settings
differ (#12365). That account cannot see the resident's settings before joining, so a strict
runtime match turned every such load into a replacement: a 409 while the loader generated, and an
eviction and reload once it went idle."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from core.inference import gpu_arbiter
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from hub.services.models import account_access as access
from models.inference import LoadRequest
from routes import inference
from state import active_generations
from studio.backend.tests.multi_account.test_chat_shared_resident import (  # noqa: F401
    RESIDENT,
    VARIANT,
    FakeLlama,
    client_for,
    shared,
    sees_resident,
)
from utils.account_context import run_as


class ReachedReplacement(Exception):
    """The load left the reuse path for config resolution, the first step of a fresh load."""


class StrictLlama(FakeLlama):
    """Bob's resident runs 2 slots with tensor parallelism; reuse needs the same runtime.

    The cross-account join runs the real component comparison against these fields."""

    tensor_parallel = True
    extra_args = None
    hf_repo = RESIDENT
    gguf_path = None
    layer_preserves_tensor_intent = False
    _is_diffusion = False
    _chat_template_override = None
    _requested_spec_mode = None
    components_match_intent = LlamaCppBackend.components_match_intent

    def __init__(self):
        super().__init__()
        self.last_load_intent = GgufLoadIntent(
            model_identifier = RESIDENT,
            hf_repo = RESIDENT,
            hf_variant = VARIANT,
            n_parallel = 2,
            tensor_parallel = True,
        )
        self.adopted = []

    def matches_load_source(self, intent):
        return (
            intent.model_identifier.lower() == RESIDENT.lower()
            and (intent.hf_variant or "").lower() == VARIANT.lower()
        )

    def adopt_load_intent_if_matched(self, intent):
        resident = self.last_load_intent
        matched = self.matches_load_source(intent) and (
            intent.n_parallel,
            intent.tensor_parallel,
        ) == (resident.n_parallel, resident.tensor_parallel)
        if matched:
            self.adopted.append(intent)
        return matched


@pytest.fixture
def strict(monkeypatch, shared):
    llama = StrictLlama()
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(
        inference,
        "_resolve_model_identifier_for_request",
        lambda *a, **k: (RESIDENT, RESIDENT, False),
    )
    monkeypatch.setattr(inference, "resolve_effective_chat_template_override", lambda *a, **k: None)
    monkeypatch.setattr(access, "require_model_access", lambda *a, **k: None)

    def replacement(*a, **k):
        raise ReachedReplacement()

    monkeypatch.setattr(inference.ModelConfig, "from_identifier", replacement)
    return llama


def load(
    account,
    *,
    force_reload = False,
    gguf_variant = VARIANT,
    **runtime,
):
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    request = LoadRequest(
        model_path = RESIDENT, gguf_variant = gguf_variant, force_reload = force_reload, **runtime
    )
    try:
        return run_as(
            account,
            lambda: asyncio.run(
                inference._load_model_impl(
                    request, fastapi_request, current_subject = account.username
                )
            ),
        )
    except Exception as exc:
        # The route wraps load failures in a 500; surface the sentinel it carries.
        cause = exc
        while cause is not None and not isinstance(cause, ReachedReplacement):
            cause = cause.__cause__ or cause.__context__
        if cause is not None:
            raise cause from None
        raise


def test_matching_runtime_joins(strict, accounts):
    response = load(accounts["alice"], n_parallel = 2, tensor_parallel = True)
    assert response.status == "already_loaded"
    assert accounts["alice"].account_id in access._resident_sharers["chat"]


@pytest.mark.parametrize("busy", [False, True], ids = ["loader_idle", "loader_generating"])
def test_other_account_default_runtime_joins_instead_of_replacing(strict, accounts, busy):
    assert not sees_resident(accounts["alice"])
    if busy:
        with run_as(accounts["bob"], active_generations.ActiveGeneration, threading.Event()):
            response = load(accounts["alice"])
    else:
        response = load(accounts["alice"])
    assert response.status == "already_loaded"
    assert not strict.unloaded
    # The resident keeps Bob's runtime; Alice's defaults are not adopted.
    assert strict.adopted == []
    assert strict.last_load_intent.n_parallel == 2 and strict.last_load_intent.tensor_parallel
    assert access._resident_sharers["chat"] == {
        accounts["alice"].account_id,
        accounts["bob"].account_id,
    }
    assert sees_resident(accounts["alice"]) and sees_resident(accounts["bob"])


def test_explicit_reload_from_another_account_is_still_a_replacement(strict, accounts):
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"], force_reload = True)
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]


def test_different_variant_from_another_account_is_still_a_replacement(strict, accounts):
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"], gguf_variant = "Q8_0")
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]


def test_the_loader_changing_its_own_runtime_is_still_a_replacement(strict, accounts):
    with pytest.raises(ReachedReplacement):
        load(accounts["bob"], n_parallel = 4)


def test_a_custom_template_on_the_resident_is_not_joined_by_default(strict, accounts):
    strict._chat_template_override = "{{ bob's template }}"
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"])
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]


def test_resident_extras_are_not_joined_by_an_account_that_did_not_send_them(strict, accounts):
    strict.requested_extra_args = ["--lora", "/bob/workspace/adapter.gguf"]
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"], llama_extra_args = [])
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]


def test_joining_a_resident_with_no_sharer_record_keeps_its_loader(monkeypatch, strict, accounts):
    # Loaded before the first managed account existed, so publish_resident recorded nothing.
    monkeypatch.setattr(access, "_resident_sharers", {})
    assert load(accounts["alice"]).status == "already_loaded"
    assert access._resident_sharers["chat"] == {
        accounts["alice"].account_id,
        gpu_arbiter.owner_account(),
    }
    with client_for(accounts["alice"]) as client:
        left = client.post("/api/inference/unload", json = {"model_path": RESIDENT})
    assert left.status_code == 200, left.text
    assert not strict.unloaded


def test_the_installation_owner_sees_the_resident_so_its_mismatch_still_replaces(strict, accounts):
    with pytest.raises(ReachedReplacement):
        load(accounts["unsloth"])


def test_repeating_the_load_that_joined_joins_again(strict, accounts):
    # Now a sharer, so status shows the resident, but its saved settings still differ.
    assert load(accounts["alice"]).status == "already_loaded"
    assert load(accounts["alice"]).status == "already_loaded"
    assert not strict.unloaded and strict.adopted == []


def test_inherited_extras_do_not_join_a_resident_with_extras(strict, accounts):
    strict.requested_extra_args = ["--lora", "/bob/workspace/adapter.gguf"]
    strict.extra_args = list(strict.requested_extra_args)
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"])
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]
