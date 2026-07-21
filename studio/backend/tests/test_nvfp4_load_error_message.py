# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 load failures should not expose verbose MLX quantization metadata."""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from models.inference import LoadRequest, ValidateModelRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module():
    spec = importlib.util.spec_from_file_location(
        "inference_route_nvfp4_error",
        _BACKEND_ROOT / "routes/inference.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_failure(
    message: str,
    exception_type: type[Exception] = RuntimeError,
    native: bool = False,
) -> HTTPException:
    inference_route = _load_route_module()
    model_path = "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    model_label = "Qwen3.6-35B-A3B-NVFP4-Fast" if native else model_path
    request = LoadRequest(model_path = model_path)
    backend = MagicMock(active_model_name = None)
    with (
        patch.object(
            inference_route,
            "_resolve_model_identifier_for_request",
            return_value = (model_path, model_label, native),
        ),
        patch.object(
            inference_route,
            "resolve_effective_chat_template_override",
            return_value = None,
        ),
        patch.object(inference_route, "get_inference_backend", return_value = backend),
        patch.object(
            inference_route, "get_llama_cpp_backend", return_value = MagicMock()
        ),
        patch.object(
            inference_route.ModelConfig,
            "from_identifier",
            side_effect = exception_type(message),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(
            inference_route.load_model(
                request, MagicMock(), current_subject = "test-user"
            )
        )
    return exc.value


def _validation_failure(
    message: str,
    exception_type: type[Exception] = RuntimeError,
    native: bool = False,
) -> HTTPException:
    inference_route = _load_route_module()
    model_path = "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    model_label = "Qwen3.6-35B-A3B-NVFP4-Fast" if native else model_path
    request = ValidateModelRequest(model_path = model_path)
    with (
        patch.object(
            inference_route,
            "_resolve_model_identifier_for_request",
            return_value = (model_path, model_label, native),
        ),
        patch.object(
            inference_route.ModelConfig,
            "from_identifier",
            side_effect = exception_type(message),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(
            inference_route.validate_model(request, current_subject = "test-user")
        )
    return exc.value


@pytest.mark.parametrize("exception_type", [Exception, RuntimeError, ValueError])
@pytest.mark.parametrize("native", [False, True])
def test_nvfp4_mlx_metadata_error_is_replaced_with_short_message(
    exception_type, native
):
    error = _load_failure(
        "Unsloth: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast' has per-module MLX "
        "quantization metadata {'config_groups': {'group_0': {'format': "
        "'float-quantized'}, 'group_1': {'format': 'nvfp4-pack-quantized'}}}",
        exception_type = exception_type,
        native = native,
    )

    assert error.status_code == 500
    assert error.detail == (
        "We are working on supporting NVFP4 inference. For now it is not supported"
    )
    assert "quantization metadata" not in error.detail


def test_unrelated_load_error_keeps_existing_message():
    error = _load_failure("Network connection timed out")

    assert error.status_code == 500
    assert error.detail == "Failed to load model: Network connection timed out"


@pytest.mark.parametrize("native", [False, True])
def test_unrelated_value_error_keeps_existing_message(native):
    error = _load_failure(
        "Invalid gpu_ids [99]", exception_type = ValueError, native = native
    )

    assert error.status_code == 400
    assert error.detail == "Invalid gpu_ids [99]"


@pytest.mark.parametrize("exception_type", [Exception, RuntimeError, ValueError])
@pytest.mark.parametrize("native", [False, True])
def test_nvfp4_validation_error_is_replaced_with_short_message(exception_type, native):
    error = _validation_failure(
        "Unsloth: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast' has per-module MLX "
        "quantization metadata {'config_groups': {'group_0': {'format': "
        "'float-quantized'}, 'group_1': {'format': 'nvfp4-pack-quantized'}}}",
        exception_type = exception_type,
        native = native,
    )

    assert error.status_code == 400
    assert error.detail == (
        "We are working on supporting NVFP4 inference. For now it is not supported"
    )
    assert "quantization metadata" not in error.detail


@pytest.mark.parametrize(
    ("native", "expected_detail"),
    [
        (False, "Network connection timed out"),
        (
            True,
            "Invalid native model Qwen3.6-35B-A3B-NVFP4-Fast: Network connection timed out",
        ),
    ],
)
def test_unrelated_validation_error_keeps_existing_message(native, expected_detail):
    error = _validation_failure("Network connection timed out", native = native)

    assert error.status_code == 400
    assert error.detail == expected_detail
