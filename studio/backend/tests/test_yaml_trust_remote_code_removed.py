# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression: model-default YAMLs must not pre-set trust_remote_code.

It is a per-load decision made through the consent dialog (which scans and pins the
auto_map code), never a config default -- a YAML flag would re-open the no-review
bypass. Models that run custom code ship auto_map, so the dialog still fires without it.
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


def test_no_model_default_yaml_has_empty_or_none_section():
    # A bare `inference:` header (no keys) parses to None and crashes the .get() loaders.
    offenders = []
    for f in _MODEL_DEFAULTS.rglob("*.yaml"):
        doc = yaml.safe_load(f.read_text())
        if not isinstance(doc, dict):
            offenders.append(f"{f.relative_to(_CONFIGS)} (not a mapping)")
            continue
        for section, body in doc.items():
            if body is None or (isinstance(body, dict) and not body):
                offenders.append(f"{f.relative_to(_CONFIGS)} [{section}]")
    assert not offenders, (
        "empty/None YAML section would crash the config loaders; drop the bare section "
        f"header instead. Offending: {offenders}"
    )


def test_formerly_flagged_models_load_inference_config_without_crash():
    # Models whose inference section was emptied by the TRC removal must still load.
    from utils.inference import load_inference_config
    for model in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Qwen2.5-7B",
    ):
        cfg = load_inference_config(model)
        assert isinstance(cfg, dict)
        assert cfg.get("trust_remote_code", False) is False


def test_all_model_yamls_load_for_training_and_inference():
    # Every YAML must load through both config paths (training + inference) as the routes do.
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults

    infer_keys = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "trust_remote_code",
    }
    failures = []
    for f in sorted(_MODEL_DEFAULTS.rglob("*.yaml")):
        stem = f.stem
        try:
            md = load_model_defaults(stem)
            assert isinstance(md, dict), f"load_model_defaults -> {type(md).__name__}"
            assert not [k for k, v in md.items() if v is None], "has a None section"
            # the dict sections the loaders read via .get('sect', {}).get(...)
            for sect in ("training", "inference", "lora", "logging"):
                assert isinstance(md.get(sect, {}), dict), f"{sect!r} is not a mapping"
            md.get("training", {}).get("trust_remote_code", False)  # routes/training.py:263
            cfg = load_inference_config(stem)
            assert infer_keys <= set(cfg), f"inference config missing {infer_keys - set(cfg)}"
        except Exception as e:  # noqa: BLE001 - aggregate so one failure does not hide others
            failures.append(f"{f.relative_to(_CONFIGS)}: {type(e).__name__}: {e}")
    assert not failures, "YAML config loaders crashed on: " + "; ".join(failures)


def test_base_templates_have_no_trust_remote_code():
    for name in ("full_finetune.yaml", "lora_text.yaml", "vision_lora.yaml"):
        doc = yaml.safe_load((_CONFIGS / name).read_text()) or {}
        flat = yaml.safe_dump(doc)
        assert "trust_remote_code" not in flat, f"{name} should not set trust_remote_code"


def test_loader_defaults_trust_remote_code_off_for_formerly_flagged_models():
    # The 4 models that used to ship trust_remote_code: true must now report no default.
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


def test_formerly_flagged_auto_map_models_still_require_consent_dialog():
    # Crux: an auto_map model must STILL surface the dialog (driven by auto_map, not the
    # YAML flag). Real backend path, mocking only the Hub json + .py fetch.
    from unittest.mock import patch
    from utils.security import consent, preflight_remote_code_consent_for_targets

    auto_map_cfg = [
        {
            "auto_map": {
                "AutoConfig": "configuration_x.XConfig",
                "AutoModelForCausalLM": "modeling_x.XForCausalLM",
            }
        }
    ]
    benign_py = {"modeling_x.py": "class XForCausalLM:\n    pass\n"}
    for model in (
        "unsloth/Nemotron-3-Nano-30B-A3B",
        "unsloth/PaddleOCR-VL",
        "unsloth/ERNIE-4.5-VL-28B-A3B-PT",
    ):
        with (
            patch.object(consent, "_load_remote_code_configs", return_value = auto_map_cfg),
            patch.object(consent, "repo_remote_code_files", return_value = benign_py),
        ):
            decision = preflight_remote_code_consent_for_targets([model], hf_token = None)
        # routes/models.py opens the dialog from decision.has_remote_code.
        assert decision.has_remote_code is True, (
            f"{model} ships auto_map but the consent scan did not flag it -> dialog would "
            "not fire"
        )


def test_no_auto_map_model_takes_no_dialog():
    # Flip side: GLM-4.7-Flash ships no auto_map -> no dialog; its old YAML flag was a no-op.
    from unittest.mock import patch
    from utils.security import consent, preflight_remote_code_consent_for_targets

    with patch.object(
        consent, "_load_remote_code_configs", return_value = [{"model_type": "glm4_moe_lite"}]
    ):
        decision = preflight_remote_code_consent_for_targets(
            ["unsloth/GLM-4.7-Flash"], hf_token = None
        )
    assert decision.has_remote_code is False
