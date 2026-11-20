# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_MEMORY_REPORT / UNSLOTH_BATCH_REPORT (unsloth/utils/run_report.py): counts, wiring, off by default."""

import ast
import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "unsloth" / "utils" / "run_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_run_report", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load()


def test_padded_rows_count_attention_mask_and_labels():
    batch = {
        "input_ids": torch.ones(2, 4, dtype = torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, 5, 5, 5], [-100, 5, -100, -100]]),
    }
    assert rr.batch_token_counts(batch).tolist() == [8, 6, 4, 2, 4]


def test_packed_row_counts_packed_seq_lengths():
    batch = {
        "input_ids": torch.ones(1, 16, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([5, 7], dtype = torch.int32),
        "labels": torch.full((1, 16), 3),
    }
    # 12 real tokens in 16 slots (4 pad-to-multiple slots); labels say 16 supervised before masking.
    assert rr.batch_token_counts(batch).tolist()[:2] == [16, 12]


def test_padding_free_row_without_mask_is_all_real():
    batch = {
        "input_ids": torch.ones(1, 10, dtype = torch.long),
        "position_ids": torch.arange(10)[None],
        "completion_mask": torch.tensor([[0] * 4 + [1] * 6]),
    }
    assert rr.batch_token_counts(batch).tolist() == [10, 10, 6, 1, 10]


def test_non_token_batches_are_skipped():
    assert rr.batch_token_counts({"pixel_values": torch.zeros(1, 3)}) is None
    assert rr.batch_token_counts(None) is None


class _FakeTrainer:
    def __init__(self):
        self.calls = 0
        self.callbacks = []
        self.args = type("Args", (), {"packing": False, "padding_free": False})()

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        self.calls += 1
        return torch.tensor(0.0)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def is_world_process_zero(self):
        return True


def _padded_batch():
    return {
        "input_ids": torch.ones(2, 4, dtype = torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, 5, 5, 5], [-100, 5, -100, -100]]),
    }


def test_batch_report_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BATCH_REPORT", raising = False)
    trainer = _FakeTrainer()
    original = trainer.training_step
    assert rr.install_batch_report(trainer) is False
    assert trainer.training_step == original
    assert "training_step" not in vars(trainer)


def test_batch_report_prints_once_after_n(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_BATCH_REPORT", "3")
    trainer = _FakeTrainer()
    assert rr.install_batch_report(trainer) is True
    for _ in range(5):
        trainer.training_step(None, _padded_batch(), 4)
    assert trainer.calls == 5  # every step still runs the real training_step
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Unsloth: first 3 micro-batches (padded rows): 18 of 24 token slots hold real tokens (75.0%), "
        "12 carry a loss (50.0%); 6 rows, widest 4 tokens."
    ]
    state = trainer._unsloth_batch_report_state
    assert state["seen"] == 3 and state["done"]


def test_batch_report_flushes_short_runs(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_BATCH_REPORT", "true")
    trainer = _FakeTrainer()
    trainer.args.packing = True
    rr.install_batch_report(trainer)
    trainer.training_step(None, _padded_batch())
    assert capsys.readouterr().out == ""
    trainer.callbacks[0].on_train_end(None, None, None)
    assert "first 1 micro-batches (packing)" in capsys.readouterr().out


def _fake_cuda(monkeypatch, used):
    """Device 0 in use, device 1 untouched; mem_get_info on device 1 is an error (would open a context)."""
    gb = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i = 0: 10 * gb if i == 0 else 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i = 0: 8 * gb if i == 0 else 0)

    def mem_get_info(i = 0):
        assert i == 0, "touched a device this process does not use"
        used.append(i)
        return 70 * gb, 80 * gb

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)


def test_memory_report_outside_allocator(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_MEMORY_REPORT", "1")
    used = []
    _fake_cuda(monkeypatch, used)
    owner = type("Owner", (), {})()
    lines = rr.report_memory("load", owner)
    assert lines == [
        "Unsloth: memory at load on cuda:0: 80.00 GB total, 70.00 GB free, torch reserved 10.00 GB "
        "(allocated 8.00 GB), outside torch's allocator 0.00 GB (CUDA context, NCCL, vLLM, other processes)."
    ]
    # Once per (stage, owner).
    assert rr.report_memory("load", owner) is None
    assert used == [0]


def test_memory_report_off_never_touches_cuda(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MEMORY_REPORT", raising = False)

    def boom(*args, **kwargs):
        raise AssertionError("queried CUDA while off")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert rr.report_memory("load") is None
    assert rr.install_memory_report(object()) is False


def _tiny_trainer(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            vocab_size = 32,
        )
    )
    rows = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [-100, 2, 3, 4]},
        {
            "input_ids": [1, 2, 0, 0],
            "attention_mask": [1, 1, 0, 0],
            "labels": [-100, 2, -100, -100],
        },
    ] * 4

    def collate(examples):
        return {k: torch.tensor([e[k] for e in examples]) for k in examples[0]}

    args = TrainingArguments(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 3,
        report_to = [],
        use_cpu = True,
        logging_steps = 100,
        save_strategy = "no",
    )
    return Trainer(model = model, args = args, train_dataset = rows, data_collator = collate)


def test_trainer_end_to_end_both_reports(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("UNSLOTH_BATCH_REPORT", "2")
    monkeypatch.setenv("UNSLOTH_MEMORY_REPORT", "1")
    stages = []
    monkeypatch.setattr(rr, "report_memory", lambda stage, owner = None: stages.append(stage))
    trainer = _tiny_trainer(tmp_path)
    assert rr.install_batch_report(trainer) and rr.install_memory_report(trainer)
    trainer.train()
    out = capsys.readouterr().out
    assert (
        "Unsloth: first 2 micro-batches (padded rows): 12 of 16 token slots hold real tokens (75.0%), "
        "8 carry a loss (50.0%); 4 rows, widest 4 tokens."
    ) in out
    # One-shot forward hook plus the per-step callback (report_memory dedupes the later steps).
    assert stages[0] == "first forward" and stages.count("first forward") == 1
    assert stages.count("first optimizer step") == 3


def _calls_in(path, function):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return [
                n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            ]
    raise AssertionError(f"{function} not found in {path}")


def test_wiring():
    trainer = ROOT / "unsloth" / "trainer.py"
    sft = _calls_in(trainer, "_patch_sft_trainer_auto_packing")
    assert "install_batch_report" in sft
    assert sft.index("install_batch_report") > sft.index("install_token_budget")
    assert "install_memory_report" in _calls_in(trainer, "_backwards_compatible_trainer")
    assert "report_memory" in _calls_in(
        ROOT / "unsloth" / "models" / "_utils.py", "_mark_requested_float32"
    )
    for name in ("llama.py", "vision.py"):
        source = (ROOT / "unsloth" / "models" / name).read_text()
        assert 'report_memory("vLLM init", llm)' in source
