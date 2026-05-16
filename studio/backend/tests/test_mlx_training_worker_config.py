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

        worker_path = (
            Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mlx_training_worker_under_test", worker_path
        )
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
