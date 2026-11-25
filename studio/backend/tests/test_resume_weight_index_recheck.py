# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resume re-checks the checkpoint's weight index right before trainer.train: the folder can be
edited while the model loads. The diffusion LoRA export can no longer write a weight index."""

import ast
import json
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _checkpoint(tmp_path, target):
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"w": target}}), encoding = "utf-8"
    )
    return ckpt


def test_an_escaping_index_is_refused_at_train_time(tmp_path):
    from core.training.resume import checked_resume_checkpoint
    ckpt = _checkpoint(tmp_path, str(tmp_path / "owner" / "model.safetensors"))
    with pytest.raises(ValueError):
        checked_resume_checkpoint(str(ckpt))


def test_a_normal_checkpoint_and_no_resume_pass_through(tmp_path):
    from core.training.resume import checked_resume_checkpoint

    ckpt = _checkpoint(tmp_path, "model-00001-of-00001.safetensors")
    assert checked_resume_checkpoint(str(ckpt)) == str(ckpt)
    assert checked_resume_checkpoint(None) is None
    assert checked_resume_checkpoint(True) is True


@pytest.mark.parametrize("rel", ["core/training/trainer.py", "core/training/worker.py"])
def test_every_resuming_train_call_is_rechecked(rel):
    tree = ast.parse((BACKEND / rel).read_text(encoding = "utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
        and any(kw.arg == "resume_from_checkpoint" for kw in node.keywords)
    ]
    assert calls
    for call in calls:
        value = next(kw.value for kw in call.keywords if kw.arg == "resume_from_checkpoint")
        assert (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "checked_resume_checkpoint"
        ), f"{rel}:{call.lineno}"


def test_diffusion_lora_export_never_writes_a_weight_index(tmp_path, monkeypatch):
    from core.inference import diffusion_lora as dl

    loras = tmp_path / "loras"
    loras.mkdir()
    (loras / "model.safetensors.index.safetensors").write_bytes(b"w")
    (loras / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "/elsewhere/model.safetensors"}}), encoding = "utf-8"
    )
    monkeypatch.setattr(dl, "loras_dir", lambda: loras)
    out_dir = tmp_path / "out"
    out = dl.export_local_lora("model.safetensors.index", out_dir)
    assert out.name != "model.safetensors.index.safetensors"
    assert not (out_dir / "model.safetensors.index.json").exists()
