# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real _run_mlx_training runs (CPT, continuing a saved adapter, streaming), on Apple Silicon only."""

from __future__ import annotations

import json
import math
import platform
import queue
import sys

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason = "MLX runs on Apple Silicon only",
    ),
    pytest.mark.allow_network,
]

_TINY = "mlx-community/SmolLM-135M-Instruct-4bit"


class _Events:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)

    def of(self, kind):
        return [e for e in self.events if e.get("type") == kind]

    def statuses(self):
        return [e.get("status_message") or "" for e in self.of("status")]


def _rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    with path.open("w", encoding = "utf-8") as f:
        for i in range(16):
            f.write(
                json.dumps(
                    {
                        "instruction": f"Add {i} and {i}.",
                        "input": "",
                        "output": f"{i} plus {i} is {2 * i}.",
                        "text": f"Row {i}: the number {i} doubled is {2 * i}.",
                    }
                )
                + "\n"
            )
    return path


def _train(tmp_path, name, **overrides):
    pytest.importorskip("mlx_lm")
    pytest.importorskip("unsloth_zoo.mlx.loader")
    import core.training.worker as worker

    config = {
        "model_name": _TINY,
        "training_type": "LoRA/QLoRA",
        "load_in_4bit": True,
        "max_seq_length": 128,
        "max_steps": 2,
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": "2e-4",
        "warmup_steps": 0,
        "eval_steps": 0,
        "save_steps": 0,
        "lora_r": 8,
        "lora_alpha": 16,
        "random_seed": 3407,
        "output_dir": str(tmp_path / name),
        "allow_external_output_dir": True,
        "local_datasets": [str(_rows(tmp_path))],
        **overrides,
    }
    events = _Events()
    worker._run_mlx_training(events, queue.Queue(), config)
    assert not events.of("error"), events.of("error")
    complete = events.of("complete")
    assert complete and complete[-1].get("output_dir"), events.events[-5:]
    losses = [e.get("loss") for e in events.of("progress") if e.get("loss") is not None]
    assert losses and all(math.isfinite(loss) for loss in losses), losses
    return events, complete[-1]["output_dir"]


def test_mlx_continued_pretraining_on_a_4bit_checkpoint(tmp_path):
    events, _ = _train(
        tmp_path,
        "cpt",
        training_type = "Continued Pretraining",
        target_modules = ["q_proj", "v_proj", "embed_tokens", "lm_head"],
    )
    assert any("continued pretraining" in s for s in events.statuses())
    # The quantized embedding cannot train in full; the run says so instead of failing in zoo.
    assert any("embed_tokens" in e.get("message", "") for e in events.of("warning"))


def test_mlx_lora_continues_from_its_saved_adapter(tmp_path):
    _, adapter_dir = _train(tmp_path, "first")
    events, _ = _train(
        tmp_path,
        "second",
        model_name = adapter_dir,
        model_format = "adapter",
    )
    statuses = events.statuses()
    assert any("Continuing from existing LoRA adapters" in s for s in statuses), statuses
    assert not any("Configuring LoRA adapters" in s for s in statuses), statuses


def test_mlx_streams_a_hub_dataset(tmp_path):
    events, _ = _train(
        tmp_path,
        "stream",
        local_datasets = None,
        hf_dataset = "yahma/alpaca-cleaned",
        dataset_streaming = True,
    )
    assert any("Streaming dataset" in s for s in events.statuses())
