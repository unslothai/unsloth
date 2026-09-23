# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import worker


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    return tmp_path / "outputs"


def _model_dir(
    root,
    name,
    config=None,
    adapter=None,
):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or {"model_type": "llama"}))
    (d / "model.safetensors").write_bytes(b"")
    if adapter is not None:
        (d / "adapter_config.json").write_text(json.dumps(adapter))
    return str(d)


def _mc(
    path,
    is_lora=False,
    base_model=None,
):
    return SimpleNamespace(is_lora=is_lora, path=path, base_model=base_model)


def test_full_finetune_output_loads_16bit(outputs):
    path = _model_dir(outputs, "unsloth_Qwen3-0.6B_1771227800")
    assert worker._resolve_lora_4bit(_mc(path), True) is False
    assert worker._resolve_lora_4bit(_mc(path), False) is False


def test_quantized_or_foreign_models_keep_4bit(outputs, tmp_path):
    quantized = _model_dir(outputs, "q", {"quantization_config": {"quant_method": "bitsandbytes"}})
    adapter = _model_dir(outputs, "a", adapter={})
    outside = _model_dir(tmp_path / "exports", "merged")
    for path in (quantized, adapter, outside, "unsloth/Qwen3-0.6B"):
        assert worker._resolve_lora_4bit(_mc(path), True) is True, path
    qlora = _model_dir(outputs, "qlora", adapter={"unsloth_training_method": "qlora"})
    assert worker._resolve_lora_4bit(_mc(qlora, is_lora=True, base_model="x"), True) is True


def test_symlink_loop_under_outputs_keeps_4bit(outputs, monkeypatch):
    """A looped link under outputs/ must read as "not a full fine-tune", not fault the load.

    Before 3.13, resolve() reports a symlink loop as RuntimeError whatever `strict` is,
    and RuntimeError is neither OSError nor ValueError. The call sites in this module and
    in routes/inference.py sit outside any handler, so an escape would surface as a 500.
    """
    outputs.mkdir(parents=True, exist_ok=True)
    looped, partner = outputs / "loop_a", outputs / "loop_b"
    looped.symlink_to(partner)
    partner.symlink_to(looped)
    assert worker._resolve_lora_4bit(_mc(str(looped)), True) is True

    # Pin that spelling on every interpreter; on 3.13+ resolve() no longer raises, so
    # without this the test is vacuous there.
    real_resolve = Path.resolve

    def loop_raises(self, *args, **kwargs):
        if self.name.startswith("loop_"):
            raise RuntimeError(f"Symlink loop from {self!s}")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", loop_raises)
    assert worker._resolve_lora_4bit(_mc(str(looped)), True) is True
