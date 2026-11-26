# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402
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


def test_target_modules_flag_overrides_the_vision_templates_all_linear(tmp_path):
    yaml_text = _VISION_LORA_YAML.read_text(encoding = "utf-8")

    assert _vision_targets(tmp_path, yaml_text, target_modules = "q_proj,v_proj") == [
        "q_proj",
        "v_proj",
    ]
    assert (
        _vision_targets(tmp_path, yaml_text, target_modules = "q_proj,v_proj", vision_all_linear = True)
        == "all-linear"
    )


def test_vision_all_linear_flag_overrides_the_text_templates_target_modules(tmp_path):
    yaml_text = (_VISION_LORA_YAML.parent / "lora_text.yaml").read_text(encoding = "utf-8")

    assert _vision_targets(tmp_path, yaml_text, vision_all_linear = True) == "all-linear"


def test_a_blank_target_modules_flag_keeps_the_vision_templates_all_linear(tmp_path):
    yaml_text = _VISION_LORA_YAML.read_text(encoding = "utf-8")

    assert _vision_targets(tmp_path, yaml_text, target_modules = " , ") == "all-linear"


def _run_train(
    tmp_path,
    monkeypatch,
    yaml_text,
    *args,
    is_vlm = True,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model: x\ndata:\n  dataset: y\ntraining:\n  output_dir: '{tmp_path / 'out'}'\n"
        + yaml_text,
        encoding = "utf-8",
    )
    sent = {}
    trainer = SimpleNamespace(
        is_vlm = is_vlm,
        training_thread = None,
        load_model = lambda **kwargs: True,
        prepare_model_for_training = lambda **kwargs: sent.update(kwargs) or True,
        load_and_format_dataset = lambda **kwargs: ([], None),
        start_training = lambda **kwargs: True,
        get_training_progress = lambda: SimpleNamespace(error = None, is_completed = True),
    )
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *a: trainer)
    monkeypatch.setattr(train_module, "_dataset_modalities", lambda *a: (True, False))
    app = typer.Typer()
    app.command()(train_module.train)

    result = CliRunner().invoke(app, ["--config", str(tmp_path / "config.yaml"), *args])

    assert result.exit_code == 0, result.output
    return result.output, sent["target_modules"]


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--target-modules", "q_proj,v_proj"], ["q_proj", "v_proj"]),
        (["--target-modules", "q_proj,v_proj", "--vision-all-linear"], "all-linear"),
    ],
)
def test_train_flags_choose_the_vision_templates_targets(tmp_path, monkeypatch, flags, expected):
    lora = {"lora": yaml.safe_load(_VISION_LORA_YAML.read_text(encoding = "utf-8"))["lora"]}

    _, sent = _run_train(tmp_path, monkeypatch, yaml.safe_dump(lora), *flags)

    assert sent == expected


@pytest.mark.parametrize(
    "target_modules, all_linear, is_vlm, warned",
    [
        ("q_proj,v_proj", "true", True, True),
        ("", "true", True, False),
        (None, "true", True, False),
        ("q_proj,v_proj", "false", True, False),
        ("q_proj,v_proj", "true", False, False),
        ("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", "true", True, False),
    ],
)
def test_train_warns_when_vision_all_linear_drops_listed_target_modules(
    tmp_path, monkeypatch, target_modules, all_linear, is_vlm, warned
):
    listed = "" if target_modules is None else f"  target_modules: '{target_modules}'\n"
    yaml_text = f"lora:\n{listed}  vision_all_linear: {all_linear}\n"

    output, _ = _run_train(tmp_path, monkeypatch, yaml_text, is_vlm = is_vlm)

    assert ("is ignored" in output) is warned
    if warned:
        assert "target_modules (q_proj, v_proj) is ignored" in output
