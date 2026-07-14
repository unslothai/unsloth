# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def _load_worker_module():
    stub_names = (
        "structlog",
        "loggers",
        "utils",
        "utils.hardware",
        "utils.training_runs",
        "utils.wheel_utils",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    try:
        sys.modules["structlog"] = types.ModuleType("structlog")

        loggers = types.ModuleType("loggers")
        loggers.get_logger = lambda *_args, **_kwargs: None
        sys.modules["loggers"] = loggers

        utils = types.ModuleType("utils")
        utils.__path__ = []
        sys.modules["utils"] = utils

        hardware = types.ModuleType("utils.hardware")
        hardware.apply_gpu_ids = lambda *_args, **_kwargs: None
        sys.modules["utils.hardware"] = hardware

        training_runs = sys.modules["utils.training_runs"] = types.ModuleType("utils.training_runs")
        training_runs.build_default_output_dir_name = lambda *_args, **_kwargs: "output"

        wheel_utils = types.ModuleType("utils.wheel_utils")
        for name in (
            "direct_wheel_url",
            "flash_attn_wheel_url",
            "has_blackwell_gpu",
            "install_wheel",
            "probe_torch_wheel_env",
            "url_exists",
        ):
            setattr(wheel_utils, name, lambda *_args, **_kwargs: None)
        sys.modules["utils.wheel_utils"] = wheel_utils

        worker_path = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"
        spec = importlib.util.spec_from_file_location("mlx_training_worker_under_test", worker_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_worker = _load_worker_module()
_normalize_mlx_studio_optimizer = _worker._normalize_mlx_studio_optimizer
_normalize_mlx_studio_scheduler = _worker._normalize_mlx_studio_scheduler
_mlx_vlm_max_resized_size = _worker._mlx_vlm_max_resized_size
_mlx_vlm_resized_image_layout = _worker._mlx_vlm_resized_image_layout
_copy_mlx_vlm_image_processor = _worker._copy_mlx_vlm_image_processor
_resize_mlx_vlm_image = _worker._resize_mlx_vlm_image
_adapt_for_mlx_vlm = _worker._adapt_for_mlx_vlm
_configure_mlx_training_schedule = _worker._configure_mlx_training_schedule


def test_mlx_studio_optimizer_aliases_are_explicit():
    assert _normalize_mlx_studio_optimizer("adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("paged_adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("adafactor") == "adafactor"


def test_mlx_studio_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match = "Supported"):
        _normalize_mlx_studio_optimizer("adamw_typo")


def test_mlx_studio_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match = "Unsupported LR scheduler for MLX training"):
        _normalize_mlx_studio_scheduler("linear_typo")


def test_mlx_studio_keeps_hf_style_tokenizer_dual_purpose():
    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text()

    assert "tokenizer = tokenizer" in source
    assert "processor = tokenizer if is_vlm else None" not in source


def test_mlx_wandb_run_config_excludes_subject_and_secrets():
    # The MLX W&B run config uploads the whole config minus a sensitive set. The owner's
    # subject (authenticated username / API-key id) must be filtered alongside the secrets,
    # otherwise it lands in W&B run config even though DB history already strips it.
    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text()

    assert (
        '_wandb_sensitive = {"hf_token", "wandb_token", "s3_config", "subject"}' in source
    ), "MLX W&B run config must exclude subject and the token/s3 secrets"
    run_source = inspect.getsource(_worker._run_mlx_training)
    assert run_source.index("trainer = MLXTrainer(") < run_source.index(
        "_prepare_mlx_output_dir(trainer, output_dir, ensure_dir)"
    )
    assert run_source.index("trainer = MLXTrainer(") < run_source.index(
        "max_steps, eval_steps_val = _configure_mlx_training_schedule("
    )
    assert "_setup_mlx_tracking(trainer, config, output_dir, _send)" in run_source
    assert "_finalize_mlx_training(trainer, _snapshot_stop" in run_source


def test_mlx_rank_owned_worker_setup(monkeypatch):
    namespace = types.SimpleNamespace
    wandb_run, tb_writer = object(), object()
    wandb_init, summary_writer = Mock(return_value = wandb_run), Mock(return_value = tb_writer)
    monkeypatch.setitem(sys.modules, "wandb", namespace(init = wandb_init))
    monkeypatch.setitem(sys.modules, "tensorboardX", namespace(SummaryWriter = summary_writer))
    config = {"enable_wandb": True, "enable_tensorboard": True, "wandb_project": "project"}
    contexts = ["output directory setup", "Weights & Biases setup", "TensorBoard setup"]

    for world_size, is_main in ((1, True), (2, True), (2, False)):
        output_setup, sync = Mock(), Mock()
        trainer = namespace(
            distributed_world_size = world_size,
            is_main_process = is_main,
            _raise_distributed_failure = sync,
        )
        _worker._prepare_mlx_output_dir(trainer, "output", output_setup)
        resources = _worker._setup_mlx_tracking(trainer, config, "output", lambda *_a, **_k: None)
        count = int(is_main)
        assert [output_setup.call_count, wandb_init.call_count, summary_writer.call_count] == [
            count
        ] * 3
        assert resources == ((wandb_run, tb_writer) if is_main else (None, None))
        assert [call.args[1] for call in sync.call_args_list] == (
            [] if world_size == 1 else contexts
        )
        if is_main:
            output_setup.assert_called_once_with(Path("output"))
            wandb_init.assert_called_once_with(project = "project", config = config, reinit = True)
            summary_writer.assert_called_once_with(log_dir = "output/runs")
        wandb_init.reset_mock()
        summary_writer.reset_mock()

    failure = OSError("read-only output")
    sync = Mock(side_effect = RuntimeError("distributed failure during output"))
    trainer = namespace(
        distributed_world_size = 2,
        is_main_process = True,
        _raise_distributed_failure = sync,
    )
    with pytest.raises(RuntimeError, match = "distributed failure during output"):
        _worker._prepare_mlx_output_dir(
            trainer, "output", lambda _path: (_ for _ in ()).throw(failure)
        )
    sync.assert_called_once_with(True, "output directory setup", failure)

    trainer = namespace(distributed_world_size = 2, is_main_process = True)
    with pytest.raises(RuntimeError, match = "Upgrade unsloth-zoo"):
        _worker._prepare_mlx_output_dir(trainer, "output", lambda _path: None)


def test_mlx_epoch_steps_use_global_ddp_batch():
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(max_steps = 0, warmup_steps = 0, eval_steps = 0),
        distributed_world_size = 2,
    )

    result = _configure_mlx_training_schedule(
        trainer, 0, 16, 2, 2, 3, warmup_ratio = 0.5, eval_steps_ratio = 0.5
    )

    assert result == (6, 3)
    assert (trainer.args.max_steps, trainer.args.warmup_steps, trainer.args.eval_steps) == (6, 3, 3)
    assert _configure_mlx_training_schedule(trainer, 7, 16, 2, 2, 3)[0] == 7
    assert _configure_mlx_training_schedule(trainer, 0, 16, 2, 2, 0)[0] == 1
    assert _configure_mlx_training_schedule(trainer, 0.5, 16, 2, 2, 3)[0] == 0.5

    trainer.distributed_world_size = 1
    assert _configure_mlx_training_schedule(
        trainer, 0, 17, 2, 2, 1, warmup_ratio = 0.3, eval_steps_ratio = 0.3
    ) == (5, 1)
    assert (trainer.args.warmup_steps, trainer.args.eval_steps) == (2, 1)


def test_mlx_finalization_reads_trainer_before_atomic_stop_snapshot():
    reads = []

    class Trainer:
        distributed_world_size = 1

        @property
        def stop_requested(self):
            reads.append("trainer")
            return False

        @stop_requested.setter
        def stop_requested(self, _value):
            reads.append("setter")

    state = _worker._mlx_worker_finalization_state(
        Trainer(), lambda: (reads.append("snapshot") or True, False)
    )

    assert state == (True, True)
    assert reads[:2] == ["trainer", "snapshot"]


def test_mlx_vlm_resize_uses_max_dimension_like_torch_trainer():
    assert _mlx_vlm_max_resized_size(1000, 500, 512) == (512, 256)
    assert _mlx_vlm_max_resized_size(500, 1000, 512) == (256, 512)
    assert _mlx_vlm_max_resized_size(1000, 1000, 512) == (512, 512)
    assert _mlx_vlm_max_resized_size(256, 128, 1536) == (256, 128)
    assert _mlx_vlm_max_resized_size(512, 256, 512) == (512, 256)
    # Half-pixel cases must match the Torch collator (not banker's round).
    assert _mlx_vlm_max_resized_size(333, 1000, 500) == (167, 500)
    assert _mlx_vlm_max_resized_size(1000, 333, 500) == (500, 167)


def test_mlx_vlm_resize_keeps_default_numpy_layout_hwc():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))

    resized = _resize_mlx_vlm_image(image, 128)

    assert resized.shape == (80, 128, 3)
    assert resized.flags.c_contiguous


def test_mlx_vlm_resize_uses_requested_chw_numpy_layout():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))

    resized = _resize_mlx_vlm_image(image, 128, image_layout = "chw")

    assert resized.shape == (3, 80, 128)
    assert resized.flags.c_contiguous


def test_mlx_vlm_resized_image_layout_probes_processor_contract():
    class ChwOnlyImageProcessor:
        def __call__(self, images = None):
            image = images[0]
            if image.shape[0] == 3:
                return {"pixel_values": image}
            raise ValueError("expected CHW")

    class HwcImageProcessor:
        def __call__(self, images = None):
            image = images[0]
            if image.shape[-1] == 3:
                return {"pixel_values": image}
            raise ValueError("expected HWC")

    assert (
        _mlx_vlm_resized_image_layout(
            types.SimpleNamespace(image_processor = ChwOnlyImageProcessor())
        )
        == "chw"
    )
    assert (
        _mlx_vlm_resized_image_layout(types.SimpleNamespace(image_processor = HwcImageProcessor()))
        is None
    )


def test_mlx_vlm_layout_probe_copies_image_processor():
    class StatefulImageProcessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, images = None):
            self.calls += 1
            image = images[0]
            if image.shape[0] == 3:
                return {"pixel_values": image}
            raise ValueError("expected CHW")

    image_processor = StatefulImageProcessor()

    layout = _mlx_vlm_resized_image_layout(types.SimpleNamespace(image_processor = image_processor))

    assert layout == "chw"
    assert image_processor.calls == 0


def test_mlx_vlm_image_processor_copy_refuses_uncopyable_processors():
    class UncopyableImageProcessor:
        def __copy__(self):
            raise RuntimeError("no copy")

        def __deepcopy__(self, _memo):
            raise RuntimeError("no deepcopy")

    image_processor = UncopyableImageProcessor()

    assert _copy_mlx_vlm_image_processor(image_processor) is None


def test_mlx_vlm_layout_probe_skips_uncopyable_processors():
    class UncopyableImageProcessor:
        def __copy__(self):
            raise RuntimeError("no copy")

        def __deepcopy__(self, _memo):
            raise RuntimeError("no deepcopy")

        def __call__(self, images = None):
            raise AssertionError("live processor should not be probed")

    assert (
        _mlx_vlm_resized_image_layout(
            types.SimpleNamespace(image_processor = UncopyableImageProcessor())
        )
        is None
    )


def test_mlx_vlm_adapter_applies_chw_layout_to_message_images():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))
    item = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe it."},
                ],
            }
        ]
    }

    adapted = _adapt_for_mlx_vlm([item], resize = 128, image_layout = "chw")

    assert adapted[0]["image"].shape == (3, 80, 128)
    assert adapted[0]["messages"][0]["content"][0] == {"type": "image"}


# ---- issue #6103: MLX transformers-version activation must not fail silently ----


def test_activate_transformers_version_or_warn_logs_on_failure(monkeypatch):
    """A failed activation in the MLX fast-path must be logged, not swallowed.

    The non-MLX path already surfaces this failure; the MLX path used a bare
    ``except Exception: pass`` so a missing/broken transformers venv produced
    no trace and a confusing downstream crash.
    """
    warnings_logged = []
    fake_logger = types.SimpleNamespace(
        warning = lambda *a, **k: warnings_logged.append((a, k)),
    )
    monkeypatch.setattr(_worker, "logger", fake_logger)

    def _boom(_name, _hf_token = None):
        raise RuntimeError("venv .venv_t5_550 missing")

    monkeypatch.setattr(_worker, "_activate_transformers_version", _boom)

    # Non-fatal: the MLX path falls through, so this must not raise.
    _worker._activate_transformers_version_or_warn("google/gemma-4-12b")

    assert len(warnings_logged) == 1, "activation failure was not logged"
    assert "gemma-4-12b" in str(warnings_logged[0]), "log does not name the model"


def test_activate_transformers_version_or_warn_silent_on_success(monkeypatch):
    warnings_logged = []
    fake_logger = types.SimpleNamespace(
        warning = lambda *a, **k: warnings_logged.append((a, k)),
    )
    monkeypatch.setattr(_worker, "logger", fake_logger)
    monkeypatch.setattr(
        _worker, "_activate_transformers_version", lambda _name, _hf_token = None: None
    )

    _worker._activate_transformers_version_or_warn("meta-llama/Llama-3-8B")

    assert warnings_logged == [], "should not warn when activation succeeds"
