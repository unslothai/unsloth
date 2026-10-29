# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    from core.training.worker import _configure_mlx_training_schedule, _finalize_mlx_training
    from unsloth_zoo.mlx.trainer import MLXTrainer

    root = Path(os.environ["UNSLOTH_MLX_DDP_PROBE_DIR"])
    trainer = object.__new__(MLXTrainer)
    rank = trainer.distributed_rank
    args = SimpleNamespace(max_steps = 0, warmup_steps = 0, eval_steps = 0)
    schedule_trainer = SimpleNamespace(args = args, distributed_world_size = 2)
    assert _configure_mlx_training_schedule(schedule_trainer, 0, 16, 2, 2, 3) == (6, 0)

    import unsloth_cli.commands.train as train_cmd
    import utils.transformers_version as transformers_version

    active, contender = root / "sidecar-active", root / "sidecar-contender"

    def activate_sidecar(*_args):
        try:
            active.touch(exist_ok = False)
            if rank == 0:
                deadline = time.monotonic() + 5
                while not contender.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert contender.exists()
                time.sleep(0.1)
        finally:
            active.unlink(missing_ok = True)

    transformers_version.activate_transformers_for_subprocess = activate_sidecar
    if rank:
        deadline = time.monotonic() + 5
        while not active.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert active.exists()
        contender.touch()
    train_cmd._activate_mlx_transformers("test/model", None)

    trainer.stop_requested = False
    trainer.save_model = lambda _path: (root / "worker-save").touch(exist_ok = False)
    complete = {}
    _finalize_mlx_training(
        trainer,
        lambda: (False, True),
        str(root),
        mx.synchronize,
        lambda event_type, **payload: complete.update(type = event_type, **payload),
        lambda: True,
    )
    assert complete["type"] == "complete"
    assert complete["output_dir"] == (str(root) if rank == 0 else None)

    legacy_cli = _load_legacy_cli()
    legacy_cli._save_or_push_model = lambda *_args: (root / "legacy-save").touch(exist_ok = False)
    legacy_cli._save_or_push_model_with_mlx_ddp(object(), object(), object(), True, trainer)

    trainer.save_model = lambda _path: (_ for _ in ()).throw(KeyboardInterrupt("probe"))
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        _finalize_mlx_training(
            trainer,
            lambda: (False, True),
            str(root),
            mx.synchronize,
            lambda *_args, **_kwargs: None,
            lambda: True,
        )
    (root / f"passed-{rank}").touch()


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason = "real MLX collectives require Apple Silicon",
)
def test_mlx_launch_cli_ddp_contracts(tmp_path):
    launcher = Path(sys.executable).with_name("mlx.launch")
    env = os.environ.copy()
    env["UNSLOTH_MLX_DDP_PROBE_DIR"] = str(tmp_path)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio")
    env["PYTHONPATH"] = os.pathsep.join((str(_REPO_ROOT), str(_REPO_ROOT / "studio/backend")))
    result = subprocess.run(
        [launcher, "-n", "2", "--", sys.executable, str(Path(__file__).resolve())],
        cwd = _REPO_ROOT,
        env = env,
        capture_output = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(tmp_path.glob("passed-*"))) == 2, result.stdout + result.stderr


if __name__ == "__main__":
    _run_collective_probe()
