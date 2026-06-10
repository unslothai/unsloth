# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_worker_module():
    stub_names = (
        "structlog",
        "loggers",
        "utils",
        "utils.hardware",
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


def test_mlx_studio_optimizer_aliases_are_explicit():
    assert _normalize_mlx_studio_optimizer("adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("paged_adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("adafactor") == "adafactor"


def test_mlx_studio_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match = "Unsupported optimizer for MLX training"):
        _normalize_mlx_studio_optimizer("adamw_typo")


def test_mlx_studio_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match = "Unsupported LR scheduler for MLX training"):
        _normalize_mlx_studio_scheduler("linear_typo")


def test_mlx_vlm_resize_uses_max_dimension_like_torch_trainer():
    assert _mlx_vlm_max_resized_size(1000, 500, 512) == (512, 256)
    assert _mlx_vlm_max_resized_size(500, 1000, 512) == (256, 512)
    assert _mlx_vlm_max_resized_size(1000, 1000, 512) == (512, 512)
    assert _mlx_vlm_max_resized_size(256, 128, 1536) == (256, 128)
    assert _mlx_vlm_max_resized_size(512, 256, 512) == (512, 256)
    # Half-pixel cases must match the Torch collator (not banker's round).
    assert _mlx_vlm_max_resized_size(333, 1000, 500) == (167, 500)
    assert _mlx_vlm_max_resized_size(1000, 333, 500) == (500, 167)


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

    def _boom(_name):
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
    monkeypatch.setattr(_worker, "_activate_transformers_version", lambda _name: None)

    _worker._activate_transformers_version_or_warn("meta-llama/Llama-3-8B")

    assert warnings_logged == [], "should not warn when activation succeeds"
