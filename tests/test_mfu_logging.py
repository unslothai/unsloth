# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FLOP accounting of unsloth/utils/mfu.py against hand counts on a tiny Llama."""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")

MFU = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "mfu.py"


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_mfu", MFU)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mfu = _load()

# hidden 16, 4 heads of 4, 2 kv heads, intermediate 32, vocab 64, 2 layers, untied head.
PER_LAYER = 16 * 16 + 16 * 8 + 16 * 8 + 16 * 16 + 3 * 16 * 32  # 2304
LAYERS = 2 * PER_LAYER
HEAD = 16 * 64
N = LAYERS + HEAD
# LoRA r=2 on q_proj (16->16) and v_proj (16->8): A 16*2 + B 2*16 and A 16*2 + B 2*8 per layer.
LORA = 2 * (32 + 32 + 32 + 16)


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 64,
        tie_word_embeddings = False,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def test_full_finetune_is_6n():
    total, parts = mfu.count_linear_flops_per_token(_model())
    assert parts == {"forward": 2 * N, "input_grad": 2 * N, "weight_grad": 2 * N, "recompute": 0}
    assert total == 6 * N


def test_full_finetune_checkpointing_recomputes_layers_not_head():
    total, parts = mfu.count_linear_flops_per_token(_model(), gradient_checkpointing = True)
    assert parts["recompute"] == 2 * LAYERS
    assert total == 6 * N + 2 * LAYERS


def test_frozen_is_4n():
    model = _model().requires_grad_(False)
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts["weight_grad"] == 0
    assert total == 4 * N


def test_lora_counts_weight_grads_only_for_adapters():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _model(),
        peft.LoraConfig(r = 2, target_modules = ["q_proj", "v_proj"], lora_dropout = 0.0),
    )
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts == {
        "forward": 2 * (N + LORA),
        "input_grad": 2 * (N + LORA),
        "weight_grad": 2 * LORA,
        "recompute": 0,
    }
    total_gc, parts_gc = mfu.count_linear_flops_per_token(model, gradient_checkpointing = True)
    assert parts_gc["recompute"] == 2 * (LAYERS + LORA)
    assert total_gc == total + 2 * (LAYERS + LORA)


def test_attention_flops_per_pair():
    config = _model().config
    # QK^T and AV: 2 FLOPs per multiply add, head_dim 4, 4 heads, 2 layers.
    forward = 2 * 2 * 4 * 4 * 2
    assert mfu.count_attention_flops_per_pair(config) == 3 * forward
    assert mfu.count_attention_flops_per_pair(config, gradient_checkpointing = True) == 4 * forward


def test_sequence_lengths_exclude_padding_and_split_packing():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert mfu._sequence_lengths({"attention_mask": mask}, ()).tolist() == [3, 2]
    packed = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    lengths = mfu._sequence_lengths({"input_ids": packed, "position_ids": packed}, ())
    assert lengths.tolist() == [3, 2, 4]
    ids = torch.zeros(2, 5, dtype = torch.long)
    assert mfu._sequence_lengths({"input_ids": ids}, ()).tolist() == [5, 5]


def test_peak_tflops_override_and_table(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "123.5")
    assert mfu.get_peak_tflops() == 123.5
    monkeypatch.delenv("UNSLOTH_PEAK_TFLOPS")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "Tesla T4")
    assert mfu.get_peak_tflops(dtype = torch.float16) == 65.0
    assert mfu.get_peak_tflops(dtype = torch.bfloat16) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "NVIDIA H100 PCIe")
    assert mfu.get_peak_tflops(dtype = torch.bfloat16) == 756.0
    assert mfu.get_peak_tflops(dtype = torch.float32) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "Mystery GPU")
    assert mfu.get_peak_tflops() is None


def test_callback_logs_tflops_and_mfu(tmp_path, monkeypatch):
    from transformers import Trainer, TrainingArguments

    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "1")
    model = _model()
    rows = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 0], "labels": [1, 2, 3, -100]},
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 0, 0], "labels": [5, 6, -100, -100]},
    ]
    dataset = [{k: torch.tensor(v) for k, v in row.items()} for row in rows]
    trainer = Trainer(
        model = model,
        args = TrainingArguments(
            output_dir = str(tmp_path),
            per_device_train_batch_size = 2,
            max_steps = 2,
            logging_steps = 1,
            report_to = [],
            use_cpu = True,
            save_strategy = "no",
        ),
        train_dataset = dataset,
    )
    callback = mfu.add_mfu_callback(trainer)
    assert mfu.add_mfu_callback(trainer) is callback
    assert trainer.callback_handler.callbacks[0] is callback

    seen = []
    original = callback._count_batch

    def spy(module, args, kwargs):
        before = int(callback._tokens)
        original(module, args, kwargs)
        seen.append(int(callback._tokens) - before)

    callback._count_batch = spy
    trainer.train()
    assert seen == [5, 5]
    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logs) == 2
    for entry in logs:
        assert entry["train_tflops_per_gpu"] > 0
        assert entry["train_mfu"] == pytest.approx(entry["train_tflops_per_gpu"])
    assert callback._handle is None
