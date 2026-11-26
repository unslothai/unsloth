# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API quant defaults use persistent account settings and existing model identity rules."""

import pytest
from fastapi import HTTPException

from routes import settings as routes
from utils import openai_auto_switch_settings as switching
from utils.account_context import AccountContext, run_as
from utils.model_gguf_preferences import (
    get_model_gguf_preference,
    set_model_gguf_preference,
    with_model_gguf_preference,
)

MODEL = "unsloth/B-GGUF"


@pytest.fixture(autouse = True)
def fresh_settings(monkeypatch):
    """Use each test's temporary database, without retaining another home's cached settings."""
    switching._cache.clear()
    monkeypatch.setattr(switching, "get_openai_auto_switch_enabled", lambda: True)
    yield
    switching._cache.clear()


def test_routes_save_read_and_clear_a_persistent_preference():
    """The settings routes round-trip a quant and still read it after dropping the memory cache."""
    payload = routes.ModelGgufPreferencePayload(model_id = MODEL, gguf_variant = "Q8_0")
    assert routes.update_model_gguf_preference_route(payload, "tester").gguf_variant == "Q8_0"
    switching._cache.clear()
    assert routes.get_model_gguf_preference_route(MODEL, "tester").gguf_variant == "Q8_0"
    assert with_model_gguf_preference(MODEL) == f"{MODEL}:Q8_0"
    cleared = routes.ModelGgufPreferencePayload(model_id = MODEL, gguf_variant = None)
    assert routes.update_model_gguf_preference_route(cleared, "tester").gguf_variant is None
    assert with_model_gguf_preference(MODEL) == MODEL


def test_saving_or_clearing_one_model_preserves_another():
    """Independent model entries share the setting map without overwriting each other."""
    set_model_gguf_preference(MODEL, "Q8_0")
    set_model_gguf_preference("org/Other-GGUF", "Q6_K")
    set_model_gguf_preference(MODEL, None)
    assert get_model_gguf_preference("org/Other-GGUF") == "Q6_K"


@pytest.mark.parametrize(
    "saved, requested",
    [
        (MODEL, MODEL.lower()),
        (MODEL, "/cache/models--unsloth--B-GGUF/snapshots/revision"),
        (r"C:\Models\B", "c:/models/b"),
        ("/mnt/c/Models/B", "/mnt/c/models/b"),
    ],
)
def test_model_identity_is_shared_with_existing_saved_defaults(saved, requested):
    """Repo casing, HF snapshots and Windows/WSL path spellings retain one preference."""
    set_model_gguf_preference(saved, "Q8_0")
    assert get_model_gguf_preference(requested) == "Q8_0"


def test_posix_paths_keep_distinct_case():
    """Two case-sensitive model folders must not inherit each other's defaults."""
    set_model_gguf_preference("/models/B", "Q8_0")
    assert get_model_gguf_preference("/models/b") is None


@pytest.mark.parametrize("suffix", ["Q4_K_M", "q8_0", "folder/custom-Q4_K_M", "latest"])
def test_an_explicit_api_suffix_is_preserved(suffix):
    """The saved default never replaces a client's explicit quant or model tag."""
    set_model_gguf_preference(MODEL, "Q8_0")
    requested = f"{MODEL}:{suffix}"
    assert with_model_gguf_preference(requested) == requested


def test_unsaved_models_and_disabled_auto_switch_keep_existing_behavior(monkeypatch):
    """A preference takes effect only for its named model while API switching is enabled."""
    assert with_model_gguf_preference(MODEL) == MODEL
    set_model_gguf_preference(MODEL, "Q8_0")
    assert with_model_gguf_preference("org/Other-GGUF") == "org/Other-GGUF"
    monkeypatch.setattr(switching, "get_openai_auto_switch_enabled", lambda: False)
    assert with_model_gguf_preference(MODEL) == MODEL


def test_preferences_belong_to_the_acting_account():
    """Two users can choose different quants without reading or clearing the other's setting."""
    alice = AccountContext("a" * 32, "alice")
    bob = AccountContext("b" * 32, "bob")
    run_as(alice, set_model_gguf_preference, MODEL, "Q8_0")
    assert run_as(bob, get_model_gguf_preference, MODEL) is None
    run_as(bob, set_model_gguf_preference, MODEL, "Q4_K_M")
    assert run_as(alice, get_model_gguf_preference, MODEL) == "Q8_0"
    assert run_as(bob, get_model_gguf_preference, MODEL) == "Q4_K_M"
    run_as(alice, set_model_gguf_preference, MODEL, None)
    assert run_as(bob, get_model_gguf_preference, MODEL) == "Q4_K_M"


def test_a_local_folders_public_api_alias_uses_its_saved_preference(monkeypatch):
    """A path-free API id resolves through the real inventory to the folder configured in the UI."""
    from core.inference import local_model_resolver as resolver

    folder = "/models/Local-GGUF"
    set_model_gguf_preference(folder, "Q8_0")
    entry = resolver._LocalGgufEntry(folder, folder, ("Q4_K_M", "Q8_0"))
    monkeypatch.setattr(resolver, "_build_index", lambda: {"local-gguf": entry})
    resolver.invalidate_index()
    assert with_model_gguf_preference("Local-GGUF") == "Local-GGUF:Q8_0"
    assert with_model_gguf_preference("Local-GGUF:Q4_K_M") == "Local-GGUF:Q4_K_M"
    assert with_model_gguf_preference("unknown-model") == "unknown-model"


@pytest.mark.parametrize("variant", ["", "../Q8_0", "Q8_0\n", "/Q8_0"])
def test_invalid_variant_is_rejected_without_erasing_the_saved_value(variant):
    """Settings validation uses the existing GGUF key validator before writing storage."""
    set_model_gguf_preference(MODEL, "Q8_0")
    with pytest.raises(HTTPException) as error:
        routes.update_model_gguf_preference_route(
            routes.ModelGgufPreferencePayload(model_id = MODEL, gguf_variant = variant), "tester"
        )
    assert error.value.status_code == 400
    assert get_model_gguf_preference(MODEL) == "Q8_0"
