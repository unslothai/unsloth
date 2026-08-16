# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Muse Glimmer resolves to its published sampling defaults.

Muse Glimmer recommends temperature 1.0, top_p 0.95, top_k 64. Without a family
entry every id fell through to ``default.yaml`` at 0.7 / 0.95 / -1 / 0.01, so
top_k was disabled outright and min_p was applied where the model asks for none.

The defaults live in ``inference_defaults.json`` rather than a ``model_defaults``
YAML, for the reason #7619 moved Kimi-K3's there: a YAML is reached only by exact
alias or a one/two-component path suffix, so it would match the bare repo id and
miss ``repo:variant``, the cache snapshot path and a plain ``.gguf`` path. Family
patterns are substring-matched against the id with the org stripped, so one entry
covers the GGUF, 4-bit and bf16 repos and every path shape they arrive as.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


EXPECTED = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0}

# Every shape an id reaches load_inference_config as. The snapshot path is not
# hypothetical: _repo_gguf_load_id publishes a snapshot filesystem path as the
# load_id for a GGUF repo in a non-active cache root.
MUSE_GLIMMER_IDS = [
    "unsloth/Muse-Glimmer-30B-GGUF",
    "unsloth/Muse-Glimmer-30B",
    "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
    "meta-models/Muse-Glimmer-30B",
    "unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL",
    "/home/u/.cache/huggingface/hub/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/deadbeef",
    "/data/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
]


def _resolve(model_id):
    from utils.inference.inference_config import load_inference_config
    return load_inference_config(model_id)


@pytest.mark.parametrize("model_id", MUSE_GLIMMER_IDS)
def test_every_id_shape_resolves_to_published_sampling(model_id):
    config = _resolve(model_id)
    for key, want in EXPECTED.items():
        assert config[key] == want, f"{model_id}: {key} was {config[key]}, expected {want}"


def test_family_entry_is_registered_in_patterns():
    """A family dict with no matching pattern never resolves: get_family_inference_params
    iterates ``patterns``, not ``families``."""
    import json

    path = _backend_root / "assets" / "configs" / "inference_defaults.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    assert "muse-glimmer" in data["families"]
    assert "muse-glimmer" in data["patterns"]


def test_family_lookup_is_case_insensitive_and_org_stripped():
    from utils.inference.inference_config import get_family_inference_params

    params = get_family_inference_params("unsloth/MUSE-GLIMMER-30B-GGUF")
    assert params.get("top_k") == 64
    assert params.get("temperature") == 1.0


def test_top_k_is_within_the_api_bound():
    """InferenceRequest bounds top_k at 100, so a family default above it would be
    rejected by validation before it ever reached llama-server."""
    from models.inference import ChatCompletionRequest

    field = ChatCompletionRequest.model_fields["top_k"]
    bounds = [m for m in field.metadata if hasattr(m, "le")]
    assert bounds, "top_k lost its upper bound"
    assert EXPECTED["top_k"] <= bounds[0].le


def test_unrelated_families_are_untouched():
    """The pattern is distinctive, but substring matching means a new entry can
    shadow an existing one. Spot-check the neighbours it sits between."""
    assert _resolve("unsloth/gemma-2-9b-it")["top_k"] == 64
    assert _resolve("unsloth/Llama-4-Scout-17B-16E-Instruct")["top_k"] == -1
    assert _resolve("unsloth/Qwen3-8B")["temperature"] == 0.6
    assert _resolve("unsloth/Kimi-K3-GGUF")["temperature"] == 1.0
