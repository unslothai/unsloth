# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kimi-K3 loads with thinking on, and with Moonshot's sampling.

Kimi-K3's template branches ``reasoning_effort`` on ``'none'`` as its disable
sentinel, so the literal scan used to surface ``none`` as the weakest level.
The chat store ships ``medium``, which the ladder does not offer, so the clamp
fell back to ``levels[0] == 'none'`` -- which ``_request_reasoning_kwargs``
turns into ``enable_thinking=false``. A reasoning model therefore loaded with
reasoning off, via a level the Think menu hides and no one can pick. Dropping
the sentinel leaves ``low`` as the floor, so the same fallback now lands on a
real level; the Think menu still offers high and max, and the pick persists.

The sampling defaults live in ``inference_defaults.json`` rather than a
``model_defaults`` YAML: those are matched by family substring, so every id
shape resolves (bare repo, ``repo:variant``, cache snapshot path, ``.gguf``
path), and the training form still resets from ``default.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Faithful slice of the Kimi-K3 template: the reasoning_effort pre-pass that
# maps 'none' to off and 'low'/'high'/'max' to a level, plus the enable_thinking
# gate layered over it.
KIMI_K3_TEMPLATE = """
{%- set rens = namespace(off = false, effort = none) -%}
{%- if reasoning_effort is defined and reasoning_effort is not none -%}
{%- if reasoning_effort == 'none' -%}{%- set rens.off = true -%}
{%- elif reasoning_effort in ['low', 'high', 'max'] -%}
{%- set rens.effort = reasoning_effort -%}{%- endif -%}{%- endif -%}
{%- if thinking is not defined -%}
{%- if enable_thinking is defined -%}{%- set thinking = enable_thinking -%}
{%- elif rens.off -%}{%- set thinking = false -%}
{%- else -%}{%- set thinking = true -%}{%- endif -%}{%- endif -%}
"""

KIMI_K3_IDS = [
    "unsloth/Kimi-K3-GGUF",
    "unsloth/Kimi-K3",
    "moonshotai/Kimi-K3",
    "unsloth/Kimi-K3-GGUF:UD-IQ1_S",
    "/home/u/.cache/huggingface/hub/models--unsloth--Kimi-K3-GGUF/snapshots/deadbeef",
    "/data/models/Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf",
]


def _detect(template, model_id = "unsloth/Kimi-K3-GGUF"):
    from core.inference.llama_cpp import detect_reasoning_flags
    return detect_reasoning_flags(template, model_id)


def test_none_is_not_offered_as_an_effort_level():
    flags = _detect(KIMI_K3_TEMPLATE)
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["low", "high", "max"]


def test_a_template_offering_only_none_is_not_an_effort_ladder():
    # Dropping the sentinel leaves nothing, so this is a plain on/off model.
    flags = _detect("{% if reasoning_effort == 'none' %}{{ enable_thinking }}{% endif %}")
    assert flags["reasoning_style"] == "enable_thinking"
    assert flags["reasoning_effort_levels"] == []


def test_disabling_still_reaches_the_template():
    # The off switch is enable_thinking=false, so removing the level costs
    # nothing: a raw caller sending reasoning_effort="none" still disables.
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking_effort"
    backend._reasoning_effort_levels = ["low", "high", "max"]

    assert backend._request_reasoning_kwargs(False, None) == {"enable_thinking": False}
    assert backend._request_reasoning_kwargs(None, "none") == {"enable_thinking": False}
    assert backend._request_reasoning_kwargs(True, "high") == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize("model_id", KIMI_K3_IDS)
def test_sampling_defaults_resolve_for_every_id_shape(model_id):
    from utils.inference.inference_config import load_inference_config

    config = load_inference_config(model_id)
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["min_p"] == 0.0


def test_kimi_k2_keeps_its_own_defaults():
    from utils.inference.inference_config import load_inference_config

    config = load_inference_config("unsloth/Kimi-K2-Instruct")
    assert config["temperature"] == 0.6
    assert config["min_p"] == 0.01


@pytest.mark.parametrize("model_id", ["unsloth/Kimi-K3", "moonshotai/Kimi-K3"])
def test_training_defaults_still_come_from_default_yaml(model_id):
    # load_model_defaults replaces default.yaml rather than merging with it, so
    # an inference-only YAML would leave the previous model's hyperparameters
    # in the training form.
    from utils.models.model_config import load_model_defaults
    assert "training" in load_model_defaults(model_id)


def test_every_mapping_entry_points_at_a_real_file():
    from utils.models.model_config import MODEL_NAME_MAPPING

    defaults_dir = _backend_root / "assets" / "configs" / "model_defaults"
    missing = [name for name in MODEL_NAME_MAPPING if not any(defaults_dir.rglob(name))]
    assert missing == []
