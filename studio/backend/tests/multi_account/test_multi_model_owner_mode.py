# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's load reads the owner's Keep multiple models loaded mode, not its own DB."""

import asyncio

import pytest

import core.inference.model_slots as model_slots
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from models.inference import LoadRequest
from utils import multi_model_settings
from utils.account_context import AccountContext, bind_account, reset_account, run_as


class FakeLlama:
    holds_no_vram = False

    def __init__(self, identifier = None, variant = None):
        self.model_identifier, self.hf_variant = identifier, variant
        self.is_loaded = self.is_active = identifier is not None
        self.is_vision, self.context_length = False, 4096

    def unload_model(self):
        self.is_loaded = self.is_active = False

    def _cleanup(self):
        pass


class FakeOrch:
    def __init__(self):
        self.active_model_name, self.models = None, {}

    def unload_model(self, name):
        self.active_model_name = None

    def _cleanup(self):
        pass


@pytest.fixture
def fresh_mode_cache():
    multi_model_settings.invalidate_multi_model_cache()
    yield
    multi_model_settings.invalidate_multi_model_cache()


@pytest.mark.parametrize("mode, keeps", [("fit", True), ("always", True), ("off", False)])
def test_a_managed_load_reads_the_owners_mode_and_keeps_its_idle_models(accounts, monkeypatch, fresh_mode_cache, mode, keeps):
    alice = accounts["alice"]
    ctx = AccountContext(alice.account_id, alice.username, alice.role)
    multi_model_settings.set_multi_model_mode(mode)  # isolated_auth binds OWNER; alice's DB never stores it
    extra = model_slots.ExtraSlot(FakeLlama("org/B-GGUF", "Q8_0"), FakeOrch(), "owner")
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama("org/A-GGUF", "Q4_K_M"))
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrch())
    monkeypatch.setattr(model_slots, "slots", [extra])
    monkeypatch.setattr(model_slots, "stuck", [])
    unloads: list[int] = []
    monkeypatch.setattr(model_slots, "unload_idle", lambda: unloads.append(1) or 0)
    assert run_as(ctx, multi_model_settings.get_multi_model_mode) == mode
    multi_model_settings.invalidate_multi_model_cache()  # make the load itself read the DB as alice
    token = bind_account(ctx)
    try:
        asyncio.run(inf._select_load_slot(LoadRequest(model_path = "org/C-GGUF")))
    finally:
        reset_account(token)
    assert unloads == ([] if keeps else [1])
    assert model_slots.slots == [extra]
