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


NO_ENGINE = (
    "NVFP4 models run on the vLLM or SGLang engine.\nThose need Linux or Windows with an "
    "NVIDIA GPU. On this computer, try a GGUF or MLX build of this model instead."
)


@pytest.fixture(autouse = True)
def _host_without_engines():
    with patch(
        "core.inference.engine_install.support_reason",
        return_value = "Managed engines currently require Linux x86_64 or Windows x64.",
    ):
        yield


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
        patch.object(inference_route, "get_llama_cpp_backend", return_value = MagicMock()),
        patch.object(
            inference_route.ModelConfig,
            "from_identifier",
            side_effect = exception_type(message),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(inference_route.load_model(request, MagicMock(), current_subject = "test-user"))
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
        asyncio.run(inference_route.validate_model(request, current_subject = "test-user"))
    return exc.value


@pytest.mark.parametrize("exception_type", [Exception, RuntimeError, ValueError])
@pytest.mark.parametrize("native", [False, True])
def test_nvfp4_mlx_metadata_error_is_replaced_with_short_message(exception_type, native):
    error = _load_failure(
        "Unsloth: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast' has per-module MLX "
        "quantization metadata {'config_groups': {'group_0': {'format': "
        "'float-quantized'}, 'group_1': {'format': 'nvfp4-pack-quantized'}}}",
        exception_type = exception_type,
        native = native,
    )

    assert error.status_code == 500
    assert error.detail == NO_ENGINE
    assert "quantization metadata" not in error.detail


def test_unrelated_load_error_keeps_existing_message():
    error = _load_failure("Network connection timed out")

    assert error.status_code == 500
    assert error.detail == "Failed to load model: Network connection timed out"


@pytest.mark.parametrize("native", [False, True])
def test_unrelated_value_error_keeps_existing_message(native):
    error = _load_failure("Invalid gpu_ids [99]", exception_type = ValueError, native = native)

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
    assert error.detail == NO_ENGINE
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


_MLX_NVFP4_ERROR = (
    "Unsloth: 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast' has per-module MLX quantization metadata "
    "{'config_groups': {'group_1': {'format': 'nvfp4-pack-quantized'}}}"
)


@pytest.mark.parametrize(
    "platform, reason, engine_message",
    [
        ("cuda", None, True),
        ("rocm", None, False),
        ("cuda", "Checking for a supported NVIDIA GPU.", False),
    ],
)
def test_nvfp4_points_at_the_engines_only_where_they_run(platform, reason, engine_message):
    inference_route = _load_route_module()
    with (
        patch("core.inference.engine_install.support_reason", return_value = reason),
        patch("core.inference.engine_install.gpu_platform", return_value = platform),
    ):
        detail = inference_route._unsupported_quantization_detail(_MLX_NVFP4_ERROR)
    if engine_message:
        assert detail == inference_route._NVFP4_ENGINE_MESSAGE
        assert "Settings > System > More engines" in detail
    else:
        assert detail == NO_ENGINE


@pytest.mark.parametrize(
    "metadata",
    [
        "{'config_groups': {'group_0': {'format': 'float-quantized'}}}",
        "{'quant_method': 'fp8', 'weight_block_size': [128, 128]}",
        "{'quant_method': 'modelopt', 'quant_algo': 'FP8'}",
    ],
)
def test_fp8_gets_the_same_message_naming_fp8(metadata):
    inference_route = _load_route_module()
    msg = f"Unsloth: 'x' has per-module MLX quantization metadata {metadata}, but requested {{}}"
    assert inference_route._unsupported_quantization_detail(msg) == NO_ENGINE.replace(
        "NVFP4", "FP8"
    )


def test_a_mixed_nvfp4_checkpoint_is_named_nvfp4():
    inference_route = _load_route_module()
    msg = (
        "Unsloth: 'x' has per-module MLX quantization metadata {'config_groups': {'group_0': "
        "{'format': 'float-quantized'}, 'group_1': {'format': 'nvfp4-pack-quantized'}}}"
    )
    assert inference_route._unsupported_quantization_detail(msg) == NO_ENGINE


def test_other_per_module_metadata_keeps_its_own_error():
    inference_route = _load_route_module()
    msg = "Unsloth: 'x' has per-module MLX quantization metadata {'format': 'pack-quantized', 'type': 'int'}"
    assert inference_route._unsupported_quantization_detail(msg) is None
