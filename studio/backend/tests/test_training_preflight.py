# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_preflight_first_batch rejects an empty/non-integer first batch (the base-model
empty-chat-template crash) before train(). The real methods are bound onto a light
fake self so the production logic runs against controlled batches."""

import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the CPU backend CI job does not install.

    The pytest job has studio.txt + torch + transformers but not unsloth/trl,
    which core.training.trainer imports at module scope. Stub the absent ones
    (real installs are left alone) so importing it for the two pure helper
    methods never breaks test collection. __spec__ = None keeps the trainer's
    own _ensure_real_packages namespace-shadow guard a no-op on the stub.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing(
    "unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported")
)
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training.trainer import UnslothTrainer  # noqa: E402

_preflight = UnslothTrainer._preflight_first_batch
_renders_empty = UnslothTrainer._chat_template_renders_empty


class _FakeInnerTrainer:
    def __init__(
        self,
        *,
        batch = None,
        dataloader_error = None,
        train_dataset = None,
    ):
        self._batch = batch
        self._dataloader_error = dataloader_error
        self.train_dataset = train_dataset

    def get_train_dataloader(self):
        if self._dataloader_error is not None:
            raise self._dataloader_error
        return [self._batch]


def _fake_self(
    *,
    inner,
    model_name = "org/Some-Model",
    tokenizer = None,
):
    s = SimpleNamespace(trainer = inner, model_name = model_name, tokenizer = tokenizer)
    # Bind real methods so self._chat_template_renders_empty() resolves.
    s._preflight_first_batch = _preflight.__get__(s)
    s._chat_template_renders_empty = _renders_empty.__get__(s)
    return s


class _EmptyTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return ""


class _RealTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return "<|im_start|>user\nhi<|im_end|>"


class TestPreflightFirstBatch(unittest.TestCase):
    def test_float_input_ids_with_empty_template_suggests_instruct(self):
        ds = [
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
        ]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner,
            model_name = "Qwen/Qwen2-VL-7B",
            tokenizer = _EmptyTemplateTokenizer(),
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("chat template", msg)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", msg)
        self.assertIn("base (pretrained) model", msg)

    def test_no_instruct_hint_when_model_already_instruct(self):
        ds = [
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
        ]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner,
            model_name = "org/Foo-Instruct",
            tokenizer = _EmptyTemplateTokenizer(),
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertNotIn("such as", msg)  # no Instruct suggestion for an Instruct model
        self.assertIn("instruction-tuned variant", msg)

    def test_empty_int_input_ids_generic_message(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.long)},
            train_dataset = [{"text": "already tokenized path"}],
        )
        s = _fake_self(inner = inner, tokenizer = _RealTemplateTokenizer())
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("invalid token IDs", msg)
        self.assertNotIn("chat template", msg)

    def test_valid_batch_returns_none(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.randint(0, 1000, (2, 34), dtype = torch.long)},
        )
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())

    def test_dataloader_error_is_surfaced(self):
        inner = _FakeInnerTrainer(dataloader_error = RuntimeError("boom"))
        s = _fake_self(inner = inner, model_name = "org/M")
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("failed to build the first training batch", msg)
        self.assertIn("org/M", msg)

    def test_missing_input_ids_does_not_false_positive(self):
        inner = _FakeInnerTrainer(batch = {"pixel_values": torch.zeros((1, 3))})
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())


class TestChatTemplateRendersEmpty(unittest.TestCase):
    def _self(self, *, train_dataset, tokenizer):
        inner = _FakeInnerTrainer(train_dataset = train_dataset)
        return _fake_self(inner = inner, tokenizer = tokenizer)

    def test_empty_render_detected(self):
        ds = [
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
        ]
        s = self._self(train_dataset = ds, tokenizer = _EmptyTemplateTokenizer())
        self.assertTrue(s._chat_template_renders_empty())

    def test_nonempty_render_not_flagged(self):
        ds = [
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
        ]
        s = self._self(train_dataset = ds, tokenizer = _RealTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())

    def test_no_messages_key_not_flagged(self):
        s = self._self(
            train_dataset = [{"text": "raw"}], tokenizer = _EmptyTemplateTokenizer()
        )
        self.assertFalse(s._chat_template_renders_empty())


def _clear_trainer_module(package: str):
    sys.modules.pop(f"{package}.trainer", None)
    pkg = sys.modules.get(package)
    if pkg is not None and hasattr(pkg, "trainer"):
        delattr(pkg, "trainer")


def _set_training_platform(monkeypatch, package: str, backend: str):
    training_mod = importlib.import_module(f"{package}.training")
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(
        training_mod.platform,
        "system",
        lambda: "Darwin" if backend == "mlx" else "Linux",
    )
    monkeypatch.setattr(
        training_mod.platform,
        "machine",
        lambda: "arm64" if backend == "mlx" else "x86_64",
    )


def _load_trainer_module(
    monkeypatch,
    backend: str,
    package: str = "core.training",
):
    _set_training_platform(monkeypatch, package, backend)
    _clear_trainer_module(package)
    if package in sys.modules:
        importlib.reload(sys.modules[package])
    trainer_mod = importlib.import_module(f"{package}.trainer")
    training_mod = importlib.import_module(f"{package}.training")
    monkeypatch.setattr(
        training_mod._MLXTrainerAdapter,
        "_activate_transformers_for_model",
        lambda self, model_name, hf_token: None,
    )
    return trainer_mod


class _ExitedProc:
    def join(self, timeout = None):
        return None

    def is_alive(self):
        return False


class _TerminableProc:
    def __init__(self):
        self.terminated = False
        self._done = threading.Event()

    def join(self, timeout = None):
        self._done.wait(timeout = timeout or 5)

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True
        self._done.set()


def test_unsloth_trainer_dispatches_for_mlx_and_torch(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    mlx_trainer = trainer_mod.UnslothTrainer()

    assert type(mlx_trainer).__module__ == "core.training.training"
    assert mlx_trainer.get_training_progress().status_message == "Ready to train"

    trainer_mod = _load_trainer_module(monkeypatch, "torch")

    assert trainer_mod.UnslothTrainer().__class__ is trainer_mod.UnslothTrainer


def test_cli_mlx_trainer_activates_before_importing_trainer():
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import json
import sys
import unsloth_cli.commands.train as train_cmd
from studio.backend.core.training import training as training_mod
from utils.hardware import hardware as hw

training_mod.platform.system = lambda: "Darwin"
training_mod.platform.machine = lambda: "arm64"
hw.DEVICE = None
events = []

def fake_activate(model_name, hf_token):
    events.append({
        "model_name": model_name,
        "trainer_loaded": "studio.backend.core.training.trainer" in sys.modules,
    })

train_cmd._activate_mlx_transformers = fake_activate
trainer = train_cmd._create_cli_trainer("mlx-community/Qwen3-0.6B-4bit", None)
print(json.dumps({
    "trainer_module": type(trainer).__module__,
    "events": events,
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo_root),
            str(repo_root / "studio" / "backend"),
            env.get("PYTHONPATH", ""),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = repo_root,
        env = env,
        text = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = True,
    )
    payload = json.loads(result.stdout)

    assert payload["trainer_module"] == "studio.backend.core.training.training"
    assert payload["events"] == [
        {"model_name": "mlx-community/Qwen3-0.6B-4bit", "trainer_loaded": False}
    ]


def test_mlx_adapter_builds_config_and_reports_completion(tmp_path, monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")
    captured = {}

    def fake_run_worker(config, event_queue, stop_queue):
        captured["config"] = config
        event_queue.put({"type": "progress", "step": 1, "total_steps": 1, "loss": 0.25})
        event_queue.put(
            {
                "type": "complete",
                "status_message": "done",
                "output_dir": config["output_dir"],
            }
        )

    trainer = trainer_mod.UnslothTrainer()
    monkeypatch.setattr(trainer, "_run_mlx_worker", fake_run_worker)

    assert trainer.load_model("mlx-community/Qwen3-0.6B-4bit", max_seq_length = 1024)
    assert trainer.prepare_model_for_training(use_lora = False)
    dataset, eval_dataset = trainer.load_and_format_dataset("org/dataset")
    output_dir = tmp_path / "mlx-out"

    assert trainer.start_training(
        dataset = dataset,
        eval_dataset = eval_dataset,
        output_dir = output_dir,
        project_name = "Sales Assistant",
        max_steps = 1,
        learning_rate = 3e-4,
    )
    trainer.training_thread.join(timeout = 5)

    progress = trainer.get_training_progress()
    config = captured["config"]
    assert progress.is_completed
    assert progress.output_dir == str(output_dir.resolve())
    progress.status_message = "mutated"
    assert trainer.get_training_progress().status_message == "done"
    assert config["model_name"] == "mlx-community/Qwen3-0.6B-4bit"
    assert config["project_name"] == "Sales Assistant"
    assert config["hf_dataset"] == "org/dataset"
    assert config["training_type"] == "Full Finetuning"
    assert config["load_in_4bit"] is False
    assert config["max_seq_length"] == 1024
    assert config["learning_rate"] == 3e-4
    assert config["output_dir"] == str(output_dir.resolve())
    assert config["allow_external_output_dir"] is True


def test_mlx_worker_helpers_cover_cli_paths(tmp_path, monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training.worker import (
        _resolve_mlx_local_dataset_files,
        _resolve_mlx_output_dir,
    )

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_mlx_local_dataset_files(["train.jsonl"]) == [str(dataset)]
    assert _resolve_mlx_output_dir(
        {"output_dir": "cli-out", "allow_external_output_dir": True},
        "mlx-community/Qwen3-0.6B-4bit",
    ) == str((tmp_path / "cli-out").resolve())


def test_run_mlx_training_process_applies_side_effects_before_hardware_detection(
    monkeypatch,
):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    order = []

    def fake_activate(model_name, hf_token):
        order.append(("activate", model_name, hf_token))

    def fake_detect_hardware():
        order.append("detect")
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising = False)
    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", fake_activate)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)

    event_queue = queue.Queue()
    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {"model_name": "mlx-community/Gemma-4-12B", "disable_xet": True},
    )

    event = event_queue.get_nowait()
    assert order == [("activate", "mlx-community/Gemma-4-12B", None), "detect"]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert "MLX training requires Apple Silicon" in event["error"]


if __name__ == "__main__":
    unittest.main()
