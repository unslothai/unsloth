# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#12365: another account's ordinary load joins the resident GGUF despite other settings."""

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
    """Bob's resident: 2 slots, tensor parallel; the join uses the real component comparison."""

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

    def load_intent_satisfied(self, intent):
        resident = self.last_load_intent
        return self.matches_load_source(intent) and (
            intent.n_parallel,
            intent.tensor_parallel,
        ) == (resident.n_parallel, resident.tensor_parallel)

    def adopt_load_intent_if_matched(self, intent):
        matched = self.load_intent_satisfied(intent)
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
    assert load(accounts["alice"]).status == "already_loaded"
    assert load(accounts["alice"]).status == "already_loaded"
    assert not strict.unloaded and strict.adopted == []


def test_inherited_extras_do_not_join_a_resident_with_extras(strict, accounts):
    strict.requested_extra_args = ["--lora", "/bob/workspace/adapter.gguf"]
    strict.extra_args = list(strict.requested_extra_args)
    with pytest.raises(ReachedReplacement):
        load(accounts["alice"])
    assert accounts["alice"].account_id not in access._resident_sharers["chat"]


def test_a_zero_vram_loader_changing_its_own_runtime_still_replaces(monkeypatch, strict, accounts):
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    with pytest.raises(ReachedReplacement):
        load(accounts["bob"], n_parallel = 4)


def test_a_retired_loader_is_not_seeded_back(strict, accounts):
    access.retire_resident_shares(accounts["bob"].account_id)
    assert load(accounts["alice"]).status == "already_loaded"
    assert access._resident_sharers["chat"] == {accounts["alice"].account_id}


@pytest.mark.parametrize("prior_preview", [False, True])
@pytest.mark.parametrize(
    "path, claim_resident",
    [
        ("/v1/chat/completions", True),
        ("/v1/messages/count_tokens", False),
        ("/v1/completions", False),
    ],
)
def test_configured_alias_refuses_capacity_relaxed_join(
    monkeypatch, strict, accounts, prior_preview, path, claim_resident
):
    # The real loader's ordinary shared-runtime join cannot count as applying an explicit alias config.
    from dataclasses import replace
    from fastapi import HTTPException
    from core.inference import local_model_resolver as resolver
    from utils import openai_auto_switch_settings as settings
    from utils.model_aliases import MODEL_ALIAS_CONFIG_SCOPE_KEY

    strict.last_load_intent = replace(
        strict.last_load_intent, n_ctx = 2048, cache_type_kv = "q4_0", speculative_type = "off"
    )
    strict._requested_spec_mode = "off"
    strict._loaded_by_user_action = True
    prior_marker = RESIDENT if prior_preview else None
    monkeypatch.setattr(inference, "_preview_resident_ident", prior_marker)
    prior_sharers = set(access._resident_sharers.get("chat", ()))
    target = f"{RESIDENT}:{VARIANT}"
    config = {"max_seq_length": 98304, "kv_cache_dtype": "q5_1", "speculative_type": "off"}
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(
        resolver, "resolve_local_gguf", lambda *a, **k: (RESIDENT, VARIANT, RESIDENT)
    )
    request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1)),
        scope = {
            MODEL_ALIAS_CONFIG_SCOPE_KEY: {
                "model": target,
                "load_config": config,
                "unsupported": False,
            }
        },
        state = SimpleNamespace(),
        url = SimpleNamespace(path = path),
    )
    with pytest.raises(HTTPException) as exc:
        run_as(
            accounts["alice"],
            lambda: asyncio.run(
                inference._maybe_auto_switch_model(
                    target, request, accounts["alice"].username, claim_resident = claim_resident
                )
            ),
        )
    assert exc.value.status_code == 409
    assert strict.last_load_intent.n_ctx == 2048
    assert strict.last_load_intent.cache_type_kv == "q4_0"
    assert not strict.unloaded
    assert strict._loaded_by_user_action is True
    assert inference._get_preview_resident() == prior_marker
    assert access._resident_sharers.get("chat", set()) == prior_sharers
    if not prior_preview:
        with pytest.raises(HTTPException) as preview:
            run_as(
                accounts["alice"],
                lambda: asyncio.run(
                    inference.load_model_for_preview(
                        LoadRequest(model_path = "/outputs/another-checkpoint"),
                        request,
                        accounts["alice"].username,
                    )
                ),
            )
        assert preview.value.status_code == 503
        assert "Unload it before using this preview" in preview.value.detail
        assert not strict.unloaded


@pytest.mark.parametrize("resident_diffusion", [False, True])
def test_configured_alias_refuses_a_diffusion_target_before_replacing(
    monkeypatch, strict, accounts, resident_diffusion
):
    # The diffusion runner ignores the alias settings and records no load intent: refuse before teardown.
    from fastapi import HTTPException

    class StopBeforeLaunch(Exception):
        pass

    def stop(*a, **k):
        raise StopBeforeLaunch()

    async def diffusion_placement(*a, **k):
        return inference._LoadPlacement(None, None, False, True)

    monkeypatch.setattr(
        inference.ModelConfig, "from_identifier", lambda *a, **k: SimpleNamespace(is_gguf = True)
    )
    monkeypatch.setattr(inference, "_require_resolved_base_access", lambda *a, **k: None)
    monkeypatch.setattr(inference, "_resolve_inherited_extra_args", lambda *a, **k: None)
    monkeypatch.setattr(inference, "_prepare_load_placement", diffusion_placement)
    monkeypatch.setattr(inference, "_validate_reasoning_budget_preflight", stop)
    strict.is_diffusion = resident_diffusion
    request = LoadRequest(
        model_path = RESIDENT,
        gguf_variant = VARIANT if resident_diffusion else "Q8_0",
        n_parallel = 2,
        tensor_parallel = True,
    )
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )

    def run():
        return run_as(
            accounts["alice"],
            lambda: asyncio.run(
                inference._load_model_impl(
                    request, fastapi_request, current_subject = accounts["alice"].username
                )
            ),
        )

    # Without an alias configuration the same request continues toward the launch.
    if resident_diffusion:
        assert run().status == "already_loaded"
    else:
        with pytest.raises(HTTPException) as reached:
            run()
        assert isinstance(reached.value.__cause__ or reached.value.__context__, StopBeforeLaunch)
    request._alias_load = True
    with pytest.raises(HTTPException) as exc:
        run()
    assert exc.value.status_code == 400
    assert "DiffusionGemma" in exc.value.detail
    assert not strict.unloaded
