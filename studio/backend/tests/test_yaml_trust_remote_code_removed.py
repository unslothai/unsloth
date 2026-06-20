# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression: model-default YAMLs must not pre-set trust_remote_code.

trust_remote_code is a per-load decision made through the remote-code consent dialog
(which scans and pins the exact auto_map code), never a config default. A YAML that
ships trust_remote_code: true would re-introduce a path to enabling remote code without
the user reviewing it -- the bypass class fixed alongside this change. Models that
genuinely run custom code ship auto_map, which the consent gate detects on its own, so
the dialog still fires for them without a YAML flag.
"""

from pathlib import Path

import yaml

_CONFIGS = Path(__file__).resolve().parent.parent / "assets" / "configs"
_MODEL_DEFAULTS = _CONFIGS / "model_defaults"


def test_no_model_default_yaml_sets_trust_remote_code():
    offenders = []
    for f in _MODEL_DEFAULTS.rglob("*.yaml"):
        doc = yaml.safe_load(f.read_text()) or {}
        if not isinstance(doc, dict):
            continue
        for section, body in doc.items():
            if isinstance(body, dict) and "trust_remote_code" in body:
                offenders.append(
                    f"{f.relative_to(_CONFIGS)} [{section}={body['trust_remote_code']}]"
                )
    assert not offenders, (
        "trust_remote_code must not be pre-set in model defaults; it is enabled only via "
        f"the consent dialog. Remove it from: {offenders}"
    )


def test_base_templates_have_no_trust_remote_code():
    for name in ("full_finetune.yaml", "lora_text.yaml", "vision_lora.yaml"):
        doc = yaml.safe_load((_CONFIGS / name).read_text()) or {}
        flat = yaml.safe_dump(doc)
        assert "trust_remote_code" not in flat, f"{name} should not set trust_remote_code"


def test_loader_defaults_trust_remote_code_off_for_formerly_flagged_models():
    # The 4 models that used to ship trust_remote_code: true. The loader must now report
    # no default (auto_map models get the dialog; GLM-4.7-Flash loads natively TRC=False).
    from utils.models.model_config import load_model_defaults
    for model in (
        "unsloth/GLM-4.7-Flash",
        "unsloth/Nemotron-3-Nano-30B-A3B",
        "unsloth/PaddleOCR-VL",
        "unsloth/ERNIE-4.5-VL-28B-A3B-PT",
    ):
        d = load_model_defaults(model)
        for section in ("training", "inference"):
            assert not (d.get(section) or {}).get(
                "trust_remote_code", False
            ), f"{model} [{section}] still carries a trust_remote_code default"
