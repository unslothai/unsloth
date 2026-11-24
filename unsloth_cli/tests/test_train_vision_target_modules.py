# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.config import load_config  # noqa: E402

_VISION_LORA_YAML = _REPO_ROOT / "studio" / "backend" / "assets" / "configs" / "vision_lora.yaml"


def _vision_targets(tmp_path, yaml_text, **overrides):
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding = "utf-8")
    cfg = load_config(path)
    cfg.apply_overrides(**overrides)
    return cfg.model_kwargs(use_lora = True, is_vision = True)["target_modules"]


def test_vision_lora_keeps_target_modules_from_the_config(tmp_path):
    targets = _vision_targets(tmp_path, "lora:\n  target_modules: q_proj,v_proj\n")

    assert targets == ["q_proj", "v_proj"]


def test_vision_lora_keeps_the_target_modules_flag(tmp_path):
    targets = _vision_targets(tmp_path, "model: x\n", target_modules = "q_proj,v_proj")

    assert targets == ["q_proj", "v_proj"]


def test_vision_lora_without_target_modules_uses_the_trainer_default(tmp_path):
    assert _vision_targets(tmp_path, "model: x\n") is None
    assert _vision_targets(tmp_path, "lora:\n  target_modules: ''\n") is None


def test_vision_all_linear_still_targets_all_linear(tmp_path):
    yaml_text = "lora:\n  target_modules: q_proj\n  vision_all_linear: true\n"

    assert _vision_targets(tmp_path, yaml_text) == "all-linear"
    assert load_config(_VISION_LORA_YAML).model_kwargs(True, True)["target_modules"] == "all-linear"
