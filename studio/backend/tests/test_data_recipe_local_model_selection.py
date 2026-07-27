# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The recipe runner's check that the loaded model is the one the recipe selected.

The recipe stores its GGUF variant; the loaded model reports the label its file
carries. The two spellings diverge for a recipe saved before #7460, and the
mismatch is unrecoverable because the stored key never changes.
"""

import pytest

import routes.data_recipe.jobs as jobs


def _recipe(gguf_variant):
    return {
        "columns": [{"type": "llm", "alias": "gen"}],
        "model_configs": [
            {
                "alias": "gen",
                "provider": "local",
                "model": "/models/Gemma",
                "gguf_variant": gguf_variant,
            }
        ],
    }


@pytest.fixture
def loaded(monkeypatch):
    def _set(variant):
        monkeypatch.setattr(
            jobs,
            "_loaded_local_model_identity",
            lambda: (True, "/models/Gemma", variant),
        )

    return _set


def test_selection_accepts_pre_7460_base_quant(loaded, monkeypatch):
    # The recipe stores Q6_K; the loaded file labels itself Q6_K-MTP. Comparing
    # exactly blocks the run for good: reloading never changes the stored key.
    loaded("Q6_K-MTP")
    monkeypatch.setattr(jobs, "_used_local_model_selections", _selections("Q6_K"))

    jobs._ensure_selected_local_model_loaded(_recipe("Q6_K"), {"local"})


def test_selection_rejects_ambiguous_base_quant(loaded, monkeypatch):
    # Q6_K cannot name a specific flavor, so a loaded Q6_K-PT-MTP is not proof
    # the recipe's model is up. Ask for a reload rather than run the wrong quant.
    loaded("Q6_K-PT-MTP")
    monkeypatch.setattr(jobs, "_used_local_model_selections", _selections("Q6_K-MTP"))

    with pytest.raises(ValueError, match = "not loaded"):
        jobs._ensure_selected_local_model_loaded(_recipe("Q6_K-MTP"), {"local"})


def test_selection_rejects_unrelated_variant(loaded, monkeypatch):
    loaded("Q8_0")
    monkeypatch.setattr(jobs, "_used_local_model_selections", _selections("Q6_K"))

    with pytest.raises(ValueError, match = "not loaded"):
        jobs._ensure_selected_local_model_loaded(_recipe("Q6_K"), {"local"})


def test_selection_accepts_exact_variant(loaded, monkeypatch):
    loaded("Q6_K-MTP")
    monkeypatch.setattr(jobs, "_used_local_model_selections", _selections("Q6_K-MTP"))

    jobs._ensure_selected_local_model_loaded(_recipe("Q6_K-MTP"), {"local"})


def _selections(gguf_variant):
    return lambda _recipe, _providers: {("/models/Gemma", gguf_variant): ["gen"]}
