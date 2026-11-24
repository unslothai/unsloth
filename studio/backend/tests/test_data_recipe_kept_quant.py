# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference import model_slots
from core.inference.model_slots import routed_slot
from routes.data_recipe import jobs

REPO = "unsloth/SmolLM2-135M-Instruct-GGUF"
KEPT = object()


def _recipe(variant):
    return {
        "model_configs": [
            {"provider": "local", "alias": "m", "model": REPO, "gguf_variant": variant},
        ],
        "columns": [{"column_type": "llm-text", "model_alias": "m"}],
    }


@pytest.fixture
def twins(monkeypatch):
    monkeypatch.setattr(model_slots, "visible", lambda: [KEPT])
    resident = {None: (True, REPO, "Q4_K_M"), KEPT: (True, REPO, "Q8_0")}
    monkeypatch.setattr(jobs, "_loaded_local_model_identity", lambda: resident[routed_slot.get()])


def test_recipe_runs_on_a_quant_kept_beside_its_repo(twins):
    assert jobs._ensure_selected_local_model_loaded(_recipe("Q8_0"), {"local"}) == (
        f"{REPO}:Q8_0",
        KEPT,
    )
    assert jobs._ensure_selected_local_model_loaded(_recipe("Q4_K_M"), {"local"}) == (
        f"{REPO}:Q4_K_M",
        None,
    )


def test_recipe_still_refuses_a_quant_that_is_not_loaded(twins):
    with pytest.raises(ValueError, match = "Selected local model is not loaded"):
        jobs._ensure_selected_local_model_loaded(_recipe("Q2_K"), {"local"})


def test_a_lone_quant_is_pinned_before_a_sibling_loads(monkeypatch):
    monkeypatch.setattr(model_slots, "visible", lambda: [])
    monkeypatch.setattr(jobs, "_loaded_local_model_identity", lambda: (True, REPO, "Q4_K_M"))
    assert jobs._ensure_selected_local_model_loaded(_recipe("Q4_K_M"), {"local"}) == (
        f"{REPO}:Q4_K_M",
        None,
    )


def test_a_lone_model_without_a_quant_keeps_its_plain_name(monkeypatch):
    monkeypatch.setattr(model_slots, "visible", lambda: [])
    monkeypatch.setattr(jobs, "_loaded_local_model_identity", lambda: (True, REPO, "Q4_K_M"))
    assert jobs._ensure_selected_local_model_loaded(_recipe(None), {"local"}) == (None, None)
