# SPDX-License-Identifier: AGPL-3.0-only

import ast
import importlib.util
import inspect
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_legacy_cli():
    spec = importlib.util.spec_from_file_location(
        "unsloth_legacy_cli", _REPO_ROOT / "unsloth-cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_collective_probe():
    sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))
    import mlx.core as mx
    import mlx.nn as nn
    from core.training.worker import _configure_mlx_training_schedule, _finalize_mlx_training
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    probe_dir = Path(os.environ["UNSLOTH_MLX_DDP_PROBE_DIR"])
    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = False
    rank = trainer.distributed_rank

    class TinyTokenizer:
        pad_token_id = eos_token_id = 2

        def encode(self, value):
            return [int(part) for part in str(value).split()]

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 8)
            self.proj = nn.Linear(8, 64, bias = False)
            self._config = {"model_type": "tiny"}

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

    mx.random.seed(0)
    epoch_args = MLXTrainingConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        max_steps = 0,
        logging_steps = 1,
        learning_rate = 1e-3,
        max_seq_length = 8,
        output_dir = str(probe_dir / "epoch-train"),
        use_cce = False,
        compile = False,
        gradient_checkpointing = False,
        dataset_order = "sequential",
    )
    epoch_data = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(16)]
    epoch_trainer = MLXTrainer(TinyLM(), TinyTokenizer(), epoch_data, args = epoch_args)
    epoch_steps, _ = _configure_mlx_training_schedule(epoch_trainer, 0, 16, 2, 2, 3)
    assert epoch_steps == 6
    epoch_trainer.save_model = lambda *_args, **_kwargs: None
    epoch_events = []
    epoch_trainer.add_step_callback(
        lambda step, total, *_args: epoch_events.append((int(step), int(total), int(_args[5])))
    )
    epoch_result = epoch_trainer.train()
    assert epoch_result["train_steps"] == 6
    assert epoch_result["trained_tokens"] == 141
    if rank == 0:
        assert epoch_events[-1] == (6, 6, 141)
    else:
        assert not epoch_events

    modes = (
        ("completed", "Training completed"),
        ("stopped", "Training stopped"),
        ("cancelled", "Training cancelled"),
        ("late_stopped", "Training stopped"),
        ("late_cancelled", "Training cancelled"),
        ("sync_failure", None),
        ("failure", None),
    )
    for mode, completion_message in modes:
        output_dir = probe_dir / mode
        trainer.stop_requested = mode in {"stopped", "cancelled"} and rank == 0
        stop_save = not (mode in {"cancelled", "late_cancelled"} and rank == 0)
        events = []

        def save_model(_path):
            if mode == "failure":
                raise KeyboardInterrupt("probe save interrupted")
            (output_dir / f"save-{rank}.txt").touch(exist_ok = False)
            if mode in {"late_stopped", "late_cancelled"} and rank == 0:
                trainer.stop_requested = True

        def synchronize():
            if mode == "sync_failure" and rank == 1:
                raise KeyboardInterrupt("probe synchronization interrupted")
            mx.synchronize()

        def write_stop_checkpoint():
            (output_dir / f"checkpoint-{rank}.txt").touch(exist_ok = False)
            return True

        trainer.save_model = save_model
        try:
            _finalize_mlx_training(
                trainer,
                lambda: (trainer.stop_requested, stop_save),
                str(output_dir),
                synchronize,
                lambda event_type, **payload: events.append({"type": event_type, **payload}),
                write_stop_checkpoint,
            )
        except BaseException as exc:
            if mode not in {"sync_failure", "failure"} or not isinstance(
                exc, (KeyboardInterrupt, RuntimeError)
            ):
                raise
            continue
        if mode in {"sync_failure", "failure"}:
            raise AssertionError(f"distributed {mode} was not propagated")
        complete = next(event for event in events if event["type"] == "complete")
        assert complete["status_message"] == completion_message
        assert complete["output_dir"] == (
            str(output_dir) if rank == 0 and mode not in {"cancelled", "late_cancelled"} else None
        )
        save_markers = sorted(path.name for path in output_dir.glob("save-*.txt"))
        assert save_markers == ([] if mode == "cancelled" else ["save-0.txt"])
        checkpoint_markers = sorted(path.name for path in output_dir.glob("checkpoint-*.txt"))
        assert checkpoint_markers == (
            ["checkpoint-0.txt"] if mode in {"stopped", "late_stopped"} else []
        )

    legacy_cli = _load_legacy_cli()
    distributed_save = legacy_cli._save_or_push_model_with_mlx_ddp
    legacy_args = (object(), object(), object(), True)

    def save_legacy_model(*args):
        assert args == legacy_args
        (probe_dir / f"legacy-save-{rank}.txt").touch(exist_ok = False)

    legacy_cli._save_or_push_model = save_legacy_model
    distributed_save(*legacy_args, trainer)
    assert sorted(path.name for path in probe_dir.glob("legacy-save-*.txt")) == [
        "legacy-save-0.txt"
    ]

    def fail_legacy_save(*_args):
        raise SystemExit("probe legacy save interrupted")

    legacy_cli._save_or_push_model = fail_legacy_save
    try:
        distributed_save(*legacy_args, trainer)
    except BaseException as exc:
        assert isinstance(exc, (RuntimeError, SystemExit))
        if isinstance(exc, RuntimeError):
            expected = "rank 0 failed" if rank == 0 else "peer rank failed"
            assert expected in str(exc)
    else:
        raise AssertionError("legacy save failure was not propagated")

    launcher_env = (os.environ["MLX_RANK"], os.environ["MLX_HOSTFILE"])
    import unsloth_cli.commands.train as train_cmd
    import utils.transformers_version as transformers_version
    from typer.testing import CliRunner
    from unsloth_cli import app
    from unsloth_cli.config import Config

    sidecar_active = probe_dir / "sidecar-active"
    sidecar_contender = probe_dir / "sidecar-contender"

    def activate_sidecar(*_args):
        try:
            fd = os.open(sidecar_active, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise AssertionError("Transformers sidecar activation overlapped") from exc
        try:
            os.close(fd)
            if rank == 0:
                deadline = time.monotonic() + 5
                while not sidecar_contender.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert sidecar_contender.exists(), "peer rank did not attempt sidecar activation"
                time.sleep(0.1)
        finally:
            sidecar_active.unlink(missing_ok = True)

    transformers_version.activate_transformers_for_subprocess = activate_sidecar
    if rank != 0:
        deadline = time.monotonic() + 5
        while not sidecar_active.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sidecar_active.exists(), "rank 0 did not start sidecar activation"
        sidecar_contender.touch()
    train_cmd._activate_mlx_transformers("test/model", None)

    cli_trainer = Mock(is_vlm = False, training_thread = None)
    cli_trainer.load_and_format_dataset.return_value = ({}, None)
    cli_trainer.get_training_progress.return_value = Mock(error = None)
    train_cmd.load_config = lambda _path: Config(
        model = "test/model", data = {"dataset": "test/dataset"}
    )
    train_cmd._create_cli_trainer = Mock(return_value = cli_trainer)
    result = CliRunner().invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    cli_trainer.load_model.assert_called_once()
    cli_trainer.start_training.assert_called_once()
    assert (os.environ["MLX_RANK"], os.environ["MLX_HOSTFILE"]) == launcher_env
    (probe_dir / f"passed-{rank}.txt").touch(exist_ok = False)


def test_mlx_sidecar_activation_failure_is_fatal_for_mpi(tmp_path, monkeypatch):
    import unsloth_cli.commands.train as train_cmd

    train_cmd.ensure_studio_backend_path()
    import utils.transformers_version as transformers_version

    monkeypatch.delenv("MLX_RANK", raising = False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))

    def fail_activation(*_args):
        raise RuntimeError("sidecar activation failed")

    monkeypatch.setattr(
        transformers_version, "activate_transformers_for_subprocess", fail_activation
    )
    with pytest.raises(RuntimeError, match = "sidecar activation failed"):
        train_cmd._activate_mlx_transformers("test/model", None)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason = "real MLX collectives require Apple Silicon",
)
def test_mlx_launch_cli_ddp_contracts(tmp_path):
    launcher = Path(sys.executable).with_name("mlx.launch")
    assert launcher.exists()
    for mode in (
        "completed",
        "stopped",
        "cancelled",
        "late_stopped",
        "late_cancelled",
        "sync_failure",
        "failure",
    ):
        (tmp_path / mode).mkdir()
    env = os.environ.copy()
    env["UNSLOTH_MLX_DDP_PROBE_DIR"] = str(tmp_path)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), str(_REPO_ROOT / "studio" / "backend"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [launcher, "-n", "2", "--", sys.executable, str(Path(__file__).resolve())],
        cwd = _REPO_ROOT,
        env = env,
        text = True,
        capture_output = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(tmp_path.glob("passed-*.txt"))) == 2, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("is_mlx", "trainer"),
    ((True, Mock(distributed_world_size = 1)), (False, object())),
)
def test_legacy_non_distributed_save_is_unchanged(is_mlx, trainer):
    legacy_cli = _load_legacy_cli()
    save_model = Mock()
    legacy_cli._save_or_push_model = save_model
    run_node = ast.parse(inspect.getsource(legacy_cli.run)).body[0]
    run_body = run_node.body
    run_calls = [
        statement.value
        for statement in run_body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    (train_call,) = (
        call for call in run_calls if call.func.id == "_train_with_legacy_save_control"
    )
    (save_call,) = (
        call for call in run_calls if call.func.id == "_save_or_push_model_with_mlx_ddp"
    )
    all_call_names = [
        node.func.id
        for node in ast.walk(run_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert all_call_names.count("_save_or_push_model_with_mlx_ddp") == 1
    assert "_save_or_push_model" not in all_call_names
    assert train_call.lineno < save_call.lineno
    assert [arg.id for arg in save_call.args] == [
        "model",
        "tokenizer",
        "args",
        "is_mlx",
        "trainer",
    ]

    model, tokenizer, args = object(), object(), object()
    legacy_cli._save_or_push_model_with_mlx_ddp(
        model,
        tokenizer,
        args,
        is_mlx,
        trainer,
    )

    save_model.assert_called_once_with(model, tokenizer, args, is_mlx)


if __name__ == "__main__":
    _run_collective_probe()
