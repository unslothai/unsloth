# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 ships generation_config.json with temperature 1.0 and no truncation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


EXPECTED = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}

INKLING2_IDS = [
    "unsloth/Inkling-2-Small-GGUF",
    "unsloth/Inkling-2-Small-GGUF:UD-Q4_K_XL",
    "thinkingmachines/Inkling-2-Small",
    "/home/u/.cache/huggingface/hub/models--unsloth--Inkling-2-Small-GGUF/snapshots/deadbeef",
    "/data/models/Inkling-2-Small-GGUF/Inkling-2-Small-UD-Q4_K_XL.gguf",
]


def _resolve(model_id):
    from utils.inference.inference_config import load_inference_config
    return load_inference_config(model_id)


@pytest.mark.parametrize("model_id", INKLING2_IDS)
def test_every_id_shape_resolves_to_published_sampling(model_id):
    config = _resolve(model_id)
    for key, want in EXPECTED.items():
        assert config[key] == want, f"{model_id}: {key} was {config[key]}, expected {want}"


def test_family_entry_is_registered_in_patterns():
    import json

    path = _backend_root / "assets" / "configs" / "inference_defaults.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    assert "inkling-2" in data["families"]
    assert "inkling-2" in data["patterns"]


def test_unrelated_families_are_untouched():
    assert _resolve("unsloth/gemma-2-9b-it")["top_k"] == 64
    assert _resolve("unsloth/Qwen3-8B")["temperature"] == 0.6
    assert _resolve("unsloth/Muse-Spark-GGUF")["top_k"] == -1
