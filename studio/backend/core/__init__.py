# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unified core module for Unsloth backend

Imports are LAZY (via __getattr__) so that training subprocesses can
import core.training.worker without pulling in heavy ML dependencies
like unsloth, transformers, or torch before the version activation
code has a chance to run.
"""

import sys
from pathlib import Path

# Ensure the backend directory is on sys.path so that bare "from utils.*"
# imports used throughout the backend work when core is imported as a package
# (e.g. from the CLI: "from studio.backend.core import ModelConfig").
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

__all__ = [
    # Inference
    "InferenceBackend",
    "get_inference_backend",
    # Training
    "get_training_backend",
    "TrainingBackend",
    "TrainingProgress",
    # Config
    "ModelConfig",
    "is_vision_model",
    "scan_trained_models",
    "scan_trained_loras",
    "load_model_defaults",
    "get_base_model_from_lora",
    # Utils
    "format_and_template_dataset",
    "normalize_path",
    "is_local_path",
    "is_model_cached",
    "without_hf_auth",
    "format_error_message",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_device",
    "is_apple_silicon",
    "clear_gpu_cache",
    "DeviceType",
]


def __getattr__(name):
    # Inference
    if name in ("InferenceBackend", "get_inference_backend"):
        from .inference import InferenceBackend, get_inference_backend

        globals()["InferenceBackend"] = InferenceBackend
        globals()["get_inference_backend"] = get_inference_backend
        return globals()[name]

    # Training
    if name in ("TrainingBackend", "get_training_backend", "TrainingProgress"):
        from .training import TrainingBackend, get_training_backend, TrainingProgress

        globals()["TrainingBackend"] = TrainingBackend
        globals()["get_training_backend"] = get_training_backend
        globals()["TrainingProgress"] = TrainingProgress
        return globals()[name]

    # Config (from utils.models)
    if name in (
        "is_vision_model",
        "ModelConfig",
        "scan_trained_models",
        "scan_trained_loras",
        "load_model_defaults",
        "get_base_model_from_lora",
    ):
        from utils.models import (
            is_vision_model,
            ModelConfig,
            scan_trained_models,
            load_model_defaults,
            get_base_model_from_lora,
        )

        globals()["is_vision_model"] = is_vision_model
        globals()["ModelConfig"] = ModelConfig
        globals()["scan_trained_models"] = scan_trained_models
        globals()["scan_trained_loras"] = scan_trained_models
        globals()["load_model_defaults"] = load_model_defaults
        globals()["get_base_model_from_lora"] = get_base_model_from_lora
        return globals()[name]

    # Paths
    if name in ("normalize_path", "is_local_path", "is_model_cached"):
        from utils.paths import normalize_path, is_local_path, is_model_cached

        globals()["normalize_path"] = normalize_path
        globals()["is_local_path"] = is_local_path
        globals()["is_model_cached"] = is_model_cached
        return globals()[name]

    # Utils
    if name in ("without_hf_auth", "format_error_message"):
        from utils.utils import without_hf_auth, format_error_message

        globals()["without_hf_auth"] = without_hf_auth
        globals()["format_error_message"] = format_error_message
        return globals()[name]

    # Hardware
    if name in (
        "get_device",
        "is_apple_silicon",
        "clear_gpu_cache",
        "get_gpu_memory_info",
        "log_gpu_memory",
        "DeviceType",
    ):
        from utils.hardware import (
            get_device,
            is_apple_silicon,
            clear_gpu_cache,
            get_gpu_memory_info,
            log_gpu_memory,
            DeviceType,
        )

        globals()["get_device"] = get_device
        globals()["is_apple_silicon"] = is_apple_silicon
        globals()["clear_gpu_cache"] = clear_gpu_cache
        globals()["get_gpu_memory_info"] = get_gpu_memory_info
        globals()["log_gpu_memory"] = log_gpu_memory
        globals()["DeviceType"] = DeviceType
        return globals()[name]

    # Datasets
    if name == "format_and_template_dataset":
        from utils.datasets import format_and_template_dataset

        globals()["format_and_template_dataset"] = format_and_template_dataset
        return format_and_template_dataset

    raise AttributeError(f"module 'core' has no attribute {name!r}")
