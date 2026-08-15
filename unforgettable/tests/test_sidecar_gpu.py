# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unsloth sidecar tests that need a real CUDA GPU. Skip otherwise."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _cuda_ready() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _gpu_base() -> str:
    return os.environ.get("UNFORGETTABLE_GPU_BASE", "unsloth/Qwen3.5-4B")


def _hf_weights_cached(model_id: str) -> bool:
    slug = "models--" + model_id.replace("/", "--")
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub" / slug,
        Path(os.environ["HF_HUB_CACHE"]) / slug
        if os.environ.get("HF_HUB_CACHE")
        else None,
    ]
    for root in roots:
        if root is None or not root.is_dir():
            continue
        if any(root.rglob("*.safetensors")):
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="CUDA torch is not available",
)


@pytest.mark.skipif(
    not _hf_weights_cached(_gpu_base()),
    reason=f"{_gpu_base()} safetensors are not in the Hugging Face cache",
)
def test_unsloth_train_writes_peft_adapter(tmp_path):
    from unforgettable.sidecar.train import UnslothTrainBackend

    examples = [
        {
            "messages": [
                {"role": "user", "content": f"Playbook {i}"},
                {"role": "assistant", "content": f"steps {i}"},
            ]
        }
        for i in range(4)
    ]
    out = tmp_path / "adapter"
    backend = UnslothTrainBackend(base_model=_gpu_base())
    backend.train(examples, output_dir=out, base_model=_gpu_base(), recipe="sft")
    cfg_path = out / "adapter_config.json"
    assert cfg_path.is_file()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("fake") is not True
    assert cfg.get("peft_type") or cfg.get("base_model_name_or_path")
    assert list(out.glob("*.safetensors")) or list(out.glob("adapter_model.bin"))


@pytest.mark.skipif(
    not _hf_weights_cached(_gpu_base()),
    reason=f"{_gpu_base()} safetensors are not in the Hugging Face cache",
)
def test_unsloth_preference_writes_peft_adapter(tmp_path):
    from unforgettable.sidecar.train import UnslothTrainBackend

    examples = [
        {
            "prompt": [{"role": "user", "content": f"broke {i}"}],
            "chosen": f"Tried: broke {i}\nThen: fixed {i}",
            "rejected": f"broke {i}",
        }
        for i in range(4)
    ]
    out = tmp_path / "adapter"
    backend = UnslothTrainBackend(base_model=_gpu_base())
    backend.train(
        examples, output_dir=out, base_model=_gpu_base(), recipe="preference"
    )
    assert (out / "pairs.jsonl").is_file()
    cfg_path = out / "adapter_config.json"
    assert cfg_path.is_file()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("fake") is not True
    assert cfg.get("peft_type") or cfg.get("base_model_name_or_path")
    assert list(out.glob("*.safetensors")) or list(out.glob("adapter_model.bin"))
