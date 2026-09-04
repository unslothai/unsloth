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

import os
from pathlib import Path

import pytest

from unforgettable.sidecar.peft import is_peft_adapter_dir


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
    roots = [Path.home() / ".cache" / "huggingface" / "hub" / slug]
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache) / slug)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub" / slug)
    for root in roots:
        if not root.is_dir():
            continue
        if any(root.rglob("*.safetensors")):
            return True
    return False


# `gpu` keeps this file out of a default `pytest unforgettable/tests` on a box
# that HAS CUDA. Root addopts is `-m 'not gpu and not slow'`; run with
# `pytest -o addopts= unforgettable/tests/test_sidecar_gpu.py`.
# CUDA skip is a fixture, not skipif: skipif would import torch at collection
# and fail test_import_hygiene when this file is collected.
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse = True)
def _require_cuda():
    if not _cuda_ready():
        pytest.skip("CUDA torch is not available")


@pytest.mark.skipif(
    not _hf_weights_cached(_gpu_base()),
    reason = f"{_gpu_base()} safetensors are not in the Hugging Face cache",
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
    backend = UnslothTrainBackend(base_model = _gpu_base())
    backend.train(examples, output_dir = out, base_model = _gpu_base(), recipe = "sft")
    assert is_peft_adapter_dir(out)
    assert list(out.glob("*.safetensors")) or list(out.glob("adapter_model.bin"))
    messages = [{"role": "user", "content": "Playbook 0"}]
    adapter_text = backend.complete(messages, adapter_path = str(out), max_tokens = 16)
    assert isinstance(adapter_text, str)
    base_text = backend.complete(messages, adapter_path = None, max_tokens = 16)
    assert isinstance(base_text, str)


@pytest.mark.skipif(
    not _hf_weights_cached(_gpu_base()),
    reason = f"{_gpu_base()} safetensors are not in the Hugging Face cache",
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
    backend = UnslothTrainBackend(base_model = _gpu_base())
    backend.train(examples, output_dir = out, base_model = _gpu_base(), recipe = "preference")
    assert (out / "pairs.jsonl").is_file()
    assert is_peft_adapter_dir(out)
    assert list(out.glob("*.safetensors")) or list(out.glob("adapter_model.bin"))
