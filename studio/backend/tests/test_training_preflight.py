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
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import typer
import yaml
from typer.testing import CliRunner


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


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
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
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "Qwen/Qwen2-VL-7B", tokenizer = _EmptyTemplateTokenizer()
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("chat template", msg)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", msg)
        self.assertIn("base (pretrained) model", msg)

    def test_no_instruct_hint_when_model_already_instruct(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "org/Foo-Instruct", tokenizer = _EmptyTemplateTokenizer()
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
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _EmptyTemplateTokenizer())
        self.assertTrue(s._chat_template_renders_empty())

    def test_nonempty_render_not_flagged(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _RealTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())

    def test_no_messages_key_not_flagged(self):
        s = self._self(train_dataset = [{"text": "raw"}], tokenizer = _EmptyTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())


def _clear_trainer_modules(package: str):
    pkg = sys.modules.get(package)
    for suffix in ("trainer",):
        sys.modules.pop(f"{package}.{suffix}", None)
        if pkg is not None and hasattr(pkg, suffix):
            delattr(pkg, suffix)


def _set_training_platform(monkeypatch, package: str, backend: str):
    training_mod = importlib.import_module(f"{package}.training")
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", None)
    if backend == "mlx":
        monkeypatch.setattr(training_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(training_mod.platform, "machine", lambda: "arm64")
    else:
        monkeypatch.setattr(training_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(training_mod.platform, "machine", lambda: "x86_64")


def _load_trainer_module(
    monkeypatch,
    backend: str,
    package: str = "core.training",
):
    _set_training_platform(monkeypatch, package, backend)
    _clear_trainer_modules(package)
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


def _load_training_package(
    monkeypatch,
    backend: str,
    package: str = "core.training",
):
    _set_training_platform(monkeypatch, package, backend)
    _clear_trainer_modules(package)
    if package in sys.modules:
        return importlib.reload(sys.modules[package])

    return importlib.import_module(package)


def test_unsloth_trainer_dispatches_to_mlx_backend(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()

    assert trainer_mod.__name__ == "core.training.trainer"
    assert trainer_mod.UnslothTrainer.__module__ == "core.training.trainer"
    assert trainer.__class__ is not trainer_mod.UnslothTrainer
    assert callable(getattr(trainer, "start_training", None))
    assert trainer.get_training_progress().status_message == "Ready to train"
    assert "core.training.trainer" in sys.modules
    assert sys.modules["core.training.trainer"] is trainer_mod


def test_unsloth_trainer_dispatches_to_mlx_from_cli_namespace(monkeypatch):
    trainer_mod = _load_trainer_module(
        monkeypatch,
        "mlx",
        package = "studio.backend.core.training",
    )

    trainer = trainer_mod.UnslothTrainer()

    assert trainer_mod.__name__ == "studio.backend.core.training.trainer"
    assert trainer_mod.UnslothTrainer.__module__ == "studio.backend.core.training.trainer"
    assert trainer.__class__ is not trainer_mod.UnslothTrainer
    assert callable(getattr(trainer, "start_training", None))
    assert trainer.get_training_progress().status_message == "Ready to train"
    assert sys.modules["studio.backend.core.training.trainer"] is trainer_mod


def test_torch_backend_package_import_keeps_trainer_module_lazy(monkeypatch):
    _load_training_package(monkeypatch, "torch")

    assert "core.training.trainer" not in sys.modules


def test_torch_backend_instantiates_existing_unsloth_trainer(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "torch")

    trainer = trainer_mod.UnslothTrainer()

    assert trainer.__class__ is trainer_mod.UnslothTrainer


def test_unsloth_trainer_uses_torch_when_detected_device_is_cpu(monkeypatch):
    package = "core.training"
    _set_training_platform(monkeypatch, package, "mlx")
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    _clear_trainer_modules(package)
    trainer_mod = importlib.import_module(f"{package}.trainer")

    trainer = trainer_mod.UnslothTrainer()

    assert trainer.__class__ is trainer_mod.UnslothTrainer


def test_mlx_trainer_builds_worker_config_and_reports_completion(tmp_path, monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    captured = {}

    def fake_run_mlx_worker(config, event_queue, stop_queue):
        captured["config"] = config
        event_queue.put(
            {
                "type": "progress",
                "step": 1,
                "total_steps": 1,
                "loss": 0.25,
                "learning_rate": 2e-4,
            }
        )
        event_queue.put(
            {
                "type": "complete",
                "status_message": "done",
                "output_dir": str(output_dir.resolve()),
            }
        )

    trainer = trainer_mod.UnslothTrainer()
    monkeypatch.setattr(trainer, "_run_mlx_worker", fake_run_mlx_worker)

    assert trainer_mod.TrainingProgress().status_message == "Ready to train"
    assert trainer.load_model(
        "mlx-community/Qwen3-0.6B-4bit",
        max_seq_length = 1024,
        load_in_4bit = True,
        hf_token = "hf_test",
        trust_remote_code = True,
        gpu_ids = [0],
    )
    assert trainer.prepare_model_for_training(
        use_lora = True,
        target_modules = ["q_proj", "v_proj"],
        lora_r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        use_rslora = True,
    )

    dataset, eval_dataset = trainer.load_and_format_dataset(
        "org/dataset",
        format_type = "chatml",
        local_datasets = ["train.jsonl"],
        local_eval_datasets = ["eval.jsonl"],
        custom_format_mapping = {"messages": "conversations"},
        subset = "default",
        train_split = "train",
        eval_split = "validation",
        dataset_streaming = True,
        eval_steps = 0.2,
        dataset_slice_start = 5,
        dataset_slice_end = 25,
        is_cpt = False,
    )

    assert dataset["final_format"] == "deferred_mlx_cli"
    assert eval_dataset is None

    output_dir = tmp_path / "mlx-out"
    assert trainer.start_training(
        dataset = dataset,
        eval_dataset = eval_dataset,
        output_dir = output_dir,
        num_epochs = 2,
        learning_rate = 3e-4,
        batch_size = 4,
        gradient_accumulation_steps = 2,
        max_steps = 1,
        save_steps = 10,
        eval_steps = 10,
        random_seed = 123,
        train_on_completions = True,
    )

    trainer.training_thread.join(timeout = 5)
    progress = trainer.get_training_progress()
    assert not trainer._pump_thread.is_alive()
    assert progress.is_completed
    assert not progress.is_training
    assert progress.step == 1
    assert progress.total_steps == 1
    assert progress.loss == 0.25
    assert progress.status_message == "done"
    assert progress.output_dir == str(output_dir.resolve())
    assert trainer.output_dir == str(output_dir.resolve())

    config = captured["config"]
    assert config["model_name"] == "mlx-community/Qwen3-0.6B-4bit"
    assert config["max_seq_length"] == 1024
    assert config["hf_token"] == "hf_test"
    assert config["trust_remote_code"] is True
    assert config["gpu_ids"] == [0]
    assert config["hf_dataset"] == "org/dataset"
    assert config["local_datasets"] == ["train.jsonl"]
    assert config["local_eval_datasets"] == ["eval.jsonl"]
    assert config["format_type"] == "chatml"
    assert config["custom_format_mapping"] == {"messages": "conversations"}
    assert config["subset"] == "default"
    assert config["train_split"] == "train"
    assert config["eval_split"] == "validation"
    assert config["dataset_streaming"] is True
    assert config["dataset_slice_start"] == 5
    assert config["dataset_slice_end"] == 25
    assert config["training_type"] == "LoRA/QLoRA"
    assert config["use_lora"] is True
    assert config["lora_r"] == 8
    assert config["target_modules"] == ["q_proj", "v_proj"]
    assert config["use_rslora"] is True
    assert config["num_epochs"] == 2
    assert config["learning_rate"] == 3e-4
    assert config["batch_size"] == 4
    assert config["gradient_accumulation_steps"] == 2
    assert config["max_steps"] == 1
    assert config["eval_steps"] == 10
    assert config["train_on_completions"] is True
    assert config["random_seed"] == 123
    assert config["output_dir"] == str(output_dir.resolve())
    assert config["allow_external_output_dir"] is True


def test_mlx_trainer_rejects_cpt_before_worker_start(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()
    assert trainer.load_model("mlx-community/Qwen3-0.6B-4bit")
    assert trainer.load_and_format_dataset("org/dataset", is_cpt = True) is not None

    assert not trainer.start_training(max_steps = 1)
    assert (
        trainer.get_training_progress().error
        == "Continued Pretraining is not supported for MLX training yet."
    )


def test_mlx_trainer_full_finetune_forces_16bit(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()
    assert trainer.load_model("mlx-community/Qwen3-0.6B-4bit", load_in_4bit = True)
    assert trainer.prepare_model_for_training(use_lora = False)
    assert trainer.load_and_format_dataset("org/dataset") is not None

    config = trainer._build_worker_config({"max_steps": 1})

    assert config["training_type"] == "Full Finetuning"
    assert config["load_in_4bit"] is False


def test_mlx_trainer_default_output_dir_uses_worker_run_dir(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()
    assert trainer.load_model("mlx-community/Qwen3-0.6B-4bit")
    assert trainer.load_and_format_dataset("org/dataset") is not None

    config = trainer._build_worker_config({"output_dir": None})

    assert config["output_dir"] is None
    assert config["allow_external_output_dir"] is False


def test_mlx_trainer_cancel_complete_event_is_not_completed(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()
    trainer.should_stop = True

    trainer._handle_event({"type": "complete", "status_message": "Training cancelled"})

    progress = trainer.get_training_progress()
    assert not progress.is_training
    assert not progress.is_completed
    assert progress.error is None
    assert progress.status_message == "Training cancelled"


def test_mlx_trainer_stop_complete_event_is_not_completed(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    trainer = trainer_mod.UnslothTrainer()
    trainer.should_stop = True

    trainer._handle_event(
        {"type": "complete", "status_message": "Training stopped", "output_dir": "/tmp/out"}
    )

    progress = trainer.get_training_progress()
    assert not progress.is_training
    assert not progress.is_completed
    assert progress.error is None
    assert progress.status_message == "Training stopped"
    assert progress.output_dir == "/tmp/out"
    assert trainer.output_dir == "/tmp/out"


def test_mlx_trainer_rejects_new_run_while_old_pump_is_alive(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    class StuckPump:
        def __init__(self):
            self.joined_with = None

        def is_alive(self):
            return True

        def join(self, timeout = None):
            self.joined_with = timeout

    trainer = trainer_mod.UnslothTrainer()
    stuck = StuckPump()
    trainer._pump_thread = stuck

    assert not trainer.start_training()
    assert stuck.joined_with == 2.0
    assert (
        trainer.get_training_progress().error == "Previous training event pump is still finalizing"
    )


def test_mlx_trainer_uses_shared_mlx_worker_entrypoint(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker

    captured = {}

    def fake_run_mlx_training_process(*, event_queue, stop_queue, config):
        captured["event_queue"] = event_queue
        captured["stop_queue"] = stop_queue
        captured["config"] = config

    monkeypatch.setattr(worker, "run_mlx_training_process", fake_run_mlx_training_process)

    event_queue = queue.Queue()
    stop_queue = queue.Queue()
    config = {"model_name": "mlx-community/Qwen3-0.6B-4bit"}
    trainer_mod.UnslothTrainer()._run_mlx_worker(config, event_queue, stop_queue)

    assert captured == {"event_queue": event_queue, "stop_queue": stop_queue, "config": config}


def test_cli_mlx_trainer_uses_studio_namespace_worker(monkeypatch):
    import unsloth_cli.commands.train as train_cmd
    from studio.backend.core.training import training as training_mod
    from utils.hardware import hardware as hw

    monkeypatch.setattr(training_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(training_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(train_cmd, "_activate_mlx_transformers", lambda model_name, hf_token: None)
    trainer = train_cmd._create_cli_trainer("mlx-community/Qwen3-0.6B-4bit", None)

    from studio.backend.core.training import worker

    captured = {}

    def fake_run_mlx_training_process(*, event_queue, stop_queue, config):
        captured["event_queue"] = event_queue
        captured["stop_queue"] = stop_queue
        captured["config"] = config

    monkeypatch.setattr(worker, "run_mlx_training_process", fake_run_mlx_training_process)

    event_queue = queue.Queue()
    stop_queue = queue.Queue()
    config = {"model_name": "mlx-community/Qwen3-0.6B-4bit"}
    trainer._run_mlx_worker(config, event_queue, stop_queue)

    assert captured == {"event_queue": event_queue, "stop_queue": stop_queue, "config": config}


def test_mlx_local_dataset_relative_path_resolves_from_cwd(tmp_path, monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training.worker import _resolve_mlx_local_dataset_files

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_mlx_local_dataset_files(["train.jsonl"]) == [str(dataset)]


def test_mlx_worker_eval_steps_fraction_becomes_positive_integer():
    from core.training.worker import _coerce_mlx_eval_steps

    assert _coerce_mlx_eval_steps(0.1, 37) == 3
    assert _coerce_mlx_eval_steps(0.01, 10) == 1
    assert _coerce_mlx_eval_steps(10, 37) == 10


def test_mlx_worker_external_output_dir_resolves_from_cwd(tmp_path, monkeypatch):
    from core.training.worker import _resolve_mlx_output_dir

    monkeypatch.chdir(tmp_path)

    output_dir = _resolve_mlx_output_dir(
        {"output_dir": "cli-out", "allow_external_output_dir": True},
        "mlx-community/Qwen3-0.6B-4bit",
    )

    assert output_dir == str((tmp_path / "cli-out").resolve())


def test_mlx_worker_studio_output_dir_uses_outputs_root(monkeypatch):
    from core.training.worker import _resolve_mlx_output_dir
    from utils import paths

    monkeypatch.setattr(
        paths, "resolve_output_dir", lambda output_dir: Path("/studio") / output_dir
    )

    output_dir = _resolve_mlx_output_dir(
        {"output_dir": "studio-run", "allow_external_output_dir": False},
        "mlx-community/Qwen3-0.6B-4bit",
    )

    assert output_dir == "/studio/studio-run"


def test_mlx_stop_poller_exits_on_worker_complete():
    from core.training.worker import _MLX_WORKER_COMPLETE, _start_mlx_stop_poller

    stop_queue = queue.Queue()
    _stop_save, stop_requested, _trainer_ref, _is_stop_requested, stop_thread = (
        _start_mlx_stop_poller(stop_queue)
    )

    stop_queue.put({"type": _MLX_WORKER_COMPLETE})
    stop_thread.join(timeout = 2)

    assert not stop_thread.is_alive()
    assert stop_requested[0] is False
    assert _is_stop_requested() is False


def test_cli_mlx_selector_uses_backend_aware_unsloth_trainer():
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import json
import os
import sys

import unsloth_cli.commands.train as train_cmd
from studio.backend.core.training import training as training_mod
from utils.hardware import hardware as hw
training_mod.platform.system = lambda: "Darwin"
training_mod.platform.machine = lambda: "arm64"
hw.DEVICE = None
train_cmd._activate_mlx_transformers = lambda model_name, hf_token: None
trainer = train_cmd._create_cli_trainer("mlx-community/Qwen3-0.6B-4bit", None)
print(json.dumps({
    "has_start_training": callable(getattr(trainer, "start_training", None)),
    "has_progress": callable(getattr(trainer, "get_training_progress", None)),
    "legacy_trainer_loaded": (
        "core.training.trainer" in sys.modules
        or "studio.backend.core.training.trainer" in sys.modules
    ),
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

    assert payload["has_start_training"] is True
    assert payload["has_progress"] is True
    assert payload["legacy_trainer_loaded"] is True


def test_cli_train_dry_run_output_dir_matches_runtime_default(monkeypatch):
    import unsloth_cli.commands.train as train_cmd

    app = typer.Typer()
    app.command()(train_cmd.train)
    runner = CliRunner()

    args = ["--model", "org/model", "--dataset", "org/dataset"]
    dry_run = runner.invoke(app, [*args, "--dry-run"])

    assert dry_run.exit_code == 0, dry_run.output
    output_dir = yaml.safe_load(dry_run.stdout)["training"]["output_dir"]

    captured = {}

    class FakeTrainer:
        is_vlm = False
        training_thread = None

        def load_model(self, **kwargs):
            return True

        def prepare_model_for_training(self, **kwargs):
            return True

        def load_and_format_dataset(self, **kwargs):
            return ({"dataset": []}, None)

        def start_training(self, **kwargs):
            captured.update(kwargs)
            return True

        def get_training_progress(self):
            return SimpleNamespace(error = None)

    monkeypatch.setattr(
        train_cmd, "_create_cli_trainer", lambda model_name, hf_token: FakeTrainer()
    )

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert captured["output_dir"] == output_dir


def test_cli_mlx_auto_selector_is_platform_only(monkeypatch):
    import unsloth_cli.commands.train as train_cmd
    from studio.backend.core.training import training as training_mod
    from utils.hardware import hardware as hw

    monkeypatch.setattr(training_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(training_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hw, "DEVICE", None)

    assert train_cmd._should_use_mlx_backend_for_cli()
    assert "mlx_lm" not in sys.modules
    assert "mlx_vlm" not in sys.modules


def test_training_backend_cancel_complete_event_is_not_completed():
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    backend._should_stop = True

    backend._handle_event({"type": "complete", "status_message": "Training cancelled"})

    progress = backend._progress
    assert not progress.is_training
    assert not progress.is_completed
    assert progress.status_message == "Training cancelled"


def test_shared_mlx_worker_activates_transformers_before_hardware_detection(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    order = []

    def fake_activate(model_name, hf_token):
        order.append(("activate", model_name, hf_token))

    def fake_detect_hardware():
        order.append("detect")
        hw.DEVICE = hw.DeviceType.MLX
        return hw.DEVICE

    def fake_run_mlx_training(event_queue, stop_queue, config):
        order.append("run")

    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", fake_activate)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)
    monkeypatch.setattr(worker, "_run_mlx_training", fake_run_mlx_training)

    config = {"model_name": "mlx-community/Gemma-4-12B", "hf_token": "hf_test"}
    worker.run_mlx_training_process(
        event_queue = queue.Queue(),
        stop_queue = queue.Queue(),
        config = config,
    )

    assert order == [("activate", "mlx-community/Gemma-4-12B", "hf_test"), "detect", "run"]


def test_shared_mlx_worker_applies_disable_xet_before_detection(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    def fake_detect_hardware():
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising = False)
    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", lambda *args: None)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)

    event_queue = queue.Queue()
    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {"model_name": "mlx-community/Gemma-4-12B", "disable_xet": True},
    )

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert "MLX training requires Apple Silicon" in event_queue.get_nowait()["error"]


def test_shared_mlx_worker_sends_stop_poller_sentinel(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from core.training.worker import _MLX_WORKER_COMPLETE
    from utils.hardware import hardware as hw

    def fake_detect_hardware():
        hw.DEVICE = hw.DeviceType.MLX
        return hw.DEVICE

    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", lambda *args: None)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)
    monkeypatch.setattr(worker, "_run_mlx_training", lambda event_queue, stop_queue, config: None)

    stop_queue = queue.Queue()
    worker.run_mlx_training_process(
        event_queue = queue.Queue(),
        stop_queue = stop_queue,
        config = {"model_name": "mlx-community/Gemma-4-12B"},
    )

    assert stop_queue.get_nowait() == {"type": _MLX_WORKER_COMPLETE}


def test_studio_training_process_preactivates_mlx_before_hardware_detection(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    order = []

    def fake_activate(model_name, hf_token):
        order.append(("activate", model_name, hf_token))

    def fake_detect_hardware():
        order.append("detect")
        hw.DEVICE = hw.DeviceType.MLX
        return hw.DEVICE

    def fake_run_mlx_training_process(*, event_queue, stop_queue, config, transformers_activated):
        order.append(("run", transformers_activated))

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(worker, "apply_gpu_ids", lambda gpu_ids: None)
    monkeypatch.setattr(worker, "_is_current_process_apple_silicon", lambda: True)
    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", fake_activate)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)
    monkeypatch.setattr(worker, "run_mlx_training_process", fake_run_mlx_training_process)
    monkeypatch.setattr("loggers.config.LogConfig.setup_logging", lambda **kwargs: None)

    worker.run_training_process(
        event_queue = queue.Queue(),
        stop_queue = queue.Queue(),
        config = {"model_name": "mlx-community/Gemma-4-12B", "resolved_gpu_ids": None},
    )

    assert order == [("activate", "mlx-community/Gemma-4-12B", None), "detect", ("run", True)]


def test_studio_training_process_broken_mlx_stack_uses_mlx_error_path(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    order = []

    def fake_detect_hardware():
        order.append("detect")
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    def fake_run_mlx_training_process(*, event_queue, stop_queue, config, transformers_activated):
        order.append(("run_mlx", transformers_activated))

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(worker, "apply_gpu_ids", lambda gpu_ids: None)
    monkeypatch.setattr(worker, "_is_current_process_apple_silicon", lambda: True)
    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", lambda *args: None)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)
    monkeypatch.setattr(worker, "run_mlx_training_process", fake_run_mlx_training_process)
    monkeypatch.setattr("loggers.config.LogConfig.setup_logging", lambda **kwargs: None)

    worker.run_training_process(
        event_queue = queue.Queue(),
        stop_queue = queue.Queue(),
        config = {"model_name": "mlx-community/Gemma-4-12B", "resolved_gpu_ids": None},
    )

    assert order == ["detect", ("run_mlx", True)]


if __name__ == "__main__":
    unittest.main()
