# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth train --config` rejected `lora.target_modules` written as a YAML or JSON list,
the form Studio's own model_defaults use, with "Input should be a valid string"."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.config import load_config  # noqa: E402


@pytest.mark.parametrize(
    "body",
    [
        "lora:\n  target_modules:\n    - q_proj\n    - v_proj\n",
        "lora:\n  target_modules: [q_proj, v_proj]\n",
    ],
)
def test_a_yaml_list_of_target_modules_loads(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding = "utf-8")

    cfg = load_config(path)

    assert cfg.model_kwargs(use_lora = True, is_vision = False)["target_modules"] == [
        "q_proj",
        "v_proj",
    ]


def test_a_json_list_of_target_modules_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"lora": {"target_modules": ["q_proj", "v_proj"]}}), encoding = "utf-8"
    )

    cfg = load_config(path)

    assert cfg.model_kwargs(use_lora = True, is_vision = False)["target_modules"] == [
        "q_proj",
        "v_proj",
    ]


def test_a_comma_separated_string_still_loads(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lora:\n  target_modules: q_proj, v_proj\n", encoding = "utf-8")

    cfg = load_config(path)

    assert cfg.model_kwargs(use_lora = True, is_vision = False)["target_modules"] == [
        "q_proj",
        "v_proj",
    ]


def test_an_empty_list_falls_back_to_the_trainer_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lora:\n  target_modules: []\n", encoding = "utf-8")

    cfg = load_config(path)

    assert cfg.model_kwargs(use_lora = True, is_vision = False)["target_modules"] is None
