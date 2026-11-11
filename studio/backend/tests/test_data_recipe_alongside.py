# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from routes.data_recipe import jobs

RECIPE = {
    "model_providers": [{"name": "local", "is_local": True}],
    "model_configs": [{"alias": "gen", "provider": "local", "model": "unsloth/gemma-3-4b-it-GGUF"}],
    "columns": [{"name": "text", "column_type": "llm-text", "model_alias": "gen"}],
}


def test_a_generator_kept_loaded_beside_the_primary_passes_the_check(monkeypatch):
    from core.inference import model_slots

    monkeypatch.setattr(
        jobs, "_loaded_local_model_identity", lambda: (True, "unsloth/Qwen3-4B-GGUF", "")
    )
    kept = {"slot": object()}
    monkeypatch.setattr(
        model_slots, "serving_slot", lambda requested, slots, satisfies: kept["slot"]
    )
    monkeypatch.setattr(model_slots, "visible", lambda: [])
    jobs._ensure_selected_local_model_loaded(RECIPE, {"local"})
    kept["slot"] = None
    with pytest.raises(ValueError, match = "not loaded"):
        jobs._ensure_selected_local_model_loaded(RECIPE, {"local"})
