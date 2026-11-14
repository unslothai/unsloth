# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""compressed-tensors refusals must match transformers' wording on every host, not only MLX."""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from models.inference import LoadRequest, ValidateModelRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

EXPECTED = (
    "This model is quantized with compressed-tensors, which Unsloth cannot run yet. "
    "Installing compressed-tensors will not help. Try a GGUF or a bitsandbytes 4-bit "
    "build of this model instead."
)

CONFIG_IMPORT_ERROR = (
    "compressed_tensors is not installed and is required for compressed-tensors "
    "quantization. Please install it with `pip install compressed-tensors`."
)
CONFIG_IMPORT_ERROR_5_10 = (
    "compressed-tensors>=0.15.0 is required for compressed-tensors quantization. "
    "Please install it with `pip install compressed-tensors>=0.15.0`."
)
QUANTIZER_IMPORT_ERROR = (
    "Using `compressed_tensors` quantized models requires the compressed-tensors "
    "library: `pip install compressed-tensors`"
)
QUANTIZER_IMPORT_ERROR_5_10 = (
    "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
    "`pip install compressed-tensors>=0.15.0`"
)

REFUSALS = {
    "config-pre-5.10": CONFIG_IMPORT_ERROR,
    "config-5.10-plus": CONFIG_IMPORT_ERROR_5_10,
    "quantizer-pre-5.10": QUANTIZER_IMPORT_ERROR,
    "quantizer-5.10-plus": QUANTIZER_IMPORT_ERROR_5_10,
}


VLLM_HINT = " Or run it with vLLM: open this model's run settings and set Inference engine to vLLM."


@pytest.fixture(autouse = True)
def _host_without_vllm():
    """The hint depends on the GPU; pin "no vLLM" so results do not depend on the machine."""
    with patch(
        "core.inference.engine_install.support_reason",
        return_value = "Requires an NVIDIA GPU with compute capability 8.0 or newer.",
    ):
        yield


def _load_route_module():
    spec = importlib.util.spec_from_file_location(
        "inference_route_compressed_tensors_error",
        _BACKEND_ROOT / "routes/inference.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_failure(
    message,
    exception_type = ImportError,
    native = False,
) -> HTTPException:
    inference_route = _load_route_module()
    model_path = "unsloth/gemma-4-E2B-it-NVFP4"
    model_label = "gemma-4-E2B-it-NVFP4" if native else model_path
    request = LoadRequest(model_path = model_path)
    backend = MagicMock(active_model_name = None)
    with (
        patch.object(
            inference_route,
            "_resolve_model_identifier_for_request",
            return_value = (model_path, model_label, native),
        ),
        patch.object(
            inference_route, "resolve_effective_chat_template_override", return_value = None
        ),
        patch.object(inference_route, "get_inference_backend", return_value = backend),
        patch.object(inference_route, "get_llama_cpp_backend", return_value = MagicMock()),
        patch.object(
            inference_route.ModelConfig, "from_identifier", side_effect = exception_type(message)
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(inference_route.load_model(request, MagicMock(), current_subject = "test-user"))
    return exc.value


def _validation_failure(
    message,
    exception_type = ImportError,
    native = False,
) -> HTTPException:
    inference_route = _load_route_module()
    model_path = "unsloth/gemma-4-E2B-it-NVFP4"
    model_label = "gemma-4-E2B-it-NVFP4" if native else model_path
    request = ValidateModelRequest(model_path = model_path)
    with (
        patch.object(
            inference_route,
            "_resolve_model_identifier_for_request",
            return_value = (model_path, model_label, native),
        ),
        patch.object(
            inference_route.ModelConfig, "from_identifier", side_effect = exception_type(message)
        ),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(inference_route.validate_model(request, current_subject = "test-user"))
    return exc.value


@pytest.mark.parametrize("message", REFUSALS.values(), ids = list(REFUSALS))
@pytest.mark.parametrize("native", [False, True])
def test_the_load_route_refuses_with_something_the_user_can_act_on(message, native):
    error = _load_failure(message, native = native)

    assert error.status_code == 500
    assert error.detail == EXPECTED
    assert "pip install" not in error.detail


@pytest.mark.parametrize("message", REFUSALS.values(), ids = list(REFUSALS))
@pytest.mark.parametrize("native", [False, True])
def test_validate_refuses_the_same_way(message, native):
    error = _validation_failure(message, native = native)

    assert error.status_code == 400
    assert error.detail == EXPECTED


def test_no_signature_carries_a_version_number():
    """No signature may carry a version number, since the minimum version moved in transformers 5.10."""
    inference_route = _load_route_module()

    for signature in inference_route._MISSING_COMPRESSED_TENSORS_SIGNATURES:
        assert ">=" not in signature
        assert not any(character.isdigit() for character in signature)
        assert len(signature) > 30


@pytest.mark.parametrize("exception_type", [ImportError, RuntimeError, Exception])
def test_the_exception_class_does_not_matter(exception_type):
    """transformers raises ImportError today; the route must not depend on that."""
    assert _load_failure(CONFIG_IMPORT_ERROR, exception_type = exception_type).detail == EXPECTED


def _raise_with_compressed_tensors_absent(call):
    """Runs a call with compressed-tensors reported absent, so the real transformers message is reached."""
    with pytest.raises(ImportError) as exc:
        call()
    return str(exc.value)


def test_the_message_the_installed_transformers_really_raises_is_recognised():
    """The installed transformers' refusal wording must be recognised by the route matcher."""
    transformers = pytest.importorskip("transformers")
    inference_route = _load_route_module()

    from transformers.quantizers import quantizer_compressed_tensors as quantizer_module
    from transformers.utils import quantization_config as config_module

    messages = {}
    with patch.object(config_module, "is_compressed_tensors_available", return_value = False):
        messages["config"] = _raise_with_compressed_tensors_absent(
            lambda: config_module.CompressedTensorsConfig()
        )
    with patch.object(quantizer_module, "is_compressed_tensors_available", return_value = False):
        quantizer = quantizer_module.CompressedTensorsHfQuantizer.__new__(
            quantizer_module.CompressedTensorsHfQuantizer
        )
        messages["quantizer"] = _raise_with_compressed_tensors_absent(
            lambda: quantizer.validate_environment()
        )

    for site, message in messages.items():
        assert "pip install" in message, (site, message)
        assert inference_route._unsupported_quantization_detail(message) == EXPECTED, (
            f"transformers {transformers.__version__} reworded the {site} refusal: {message!r}. "
            "Add the part that survived to _MISSING_COMPRESSED_TENSORS_SIGNATURES."
        )


@pytest.mark.parametrize("message", REFUSALS.values(), ids = list(REFUSALS))
def test_the_message_survives_the_trip_out_of_the_inference_worker(message):
    """format_error_message must pass the compressed-tensors refusal through unchanged."""
    from utils.utils import format_error_message
    assert format_error_message(ImportError(message), "unsloth/gemma-4-E2B-it-NVFP4") == message


def _quantizer_refusal_messages():
    """Collects all quantizers' missing-dependency messages from the installed transformers, unexecuted."""
    import ast
    import re
    from pathlib import Path as _Path

    import transformers.quantizers as quantizers_package

    package_dir = _Path(quantizers_package.__file__).parent
    out = []
    for source_file in sorted(package_dir.glob("quantizer_*.py")):
        tree = ast.parse(source_file.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            for literal in ast.walk(node.exc):
                if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
                    continue
                text = literal.value
                if not re.search(r"requires|install", text, re.IGNORECASE):
                    continue
                out.append((source_file.stem, text))
    return out


def test_no_other_quantization_family_is_re_routed_into_this_refusal():
    """No other quantization family may be re-routed into the compressed-tensors refusal."""
    pytest.importorskip("transformers")
    inference_route = _load_route_module()

    messages = _quantizer_refusal_messages()
    assert len({module for module, _ in messages}) >= 8, messages

    for module, message in messages:
        detail = inference_route._unsupported_quantization_detail(message)
        if module == "quantizer_compressed_tensors":
            continue
        assert (
            detail is None
        ), f"{module} message now matches the compressed-tensors refusal: {message!r}"


def test_the_mlx_refusal_is_unchanged():
    """The signature this route already recognised keeps its own wording."""
    error = _load_failure(
        "Unsloth: 'unsloth/gemma-4-E2B-it-NVFP4' has per-module MLX quantization "
        "metadata {'config_groups': {'group_1': {'format': 'nvfp4-pack-quantized'}}}",
        exception_type = RuntimeError,
    )

    assert error.detail == (
        "We are working on supporting NVFP4 inference. For now it is not supported"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Network connection timed out",
        "Failed to parse the config of this compressed-tensors export",
    ],
)
def test_an_unrelated_failure_keeps_its_own_message(message):
    error = _load_failure(message, exception_type = RuntimeError)

    assert error.detail == f"Failed to load model: {message}"


def test_the_child_process_diagnostics_block_is_not_matched():
    """A llama-server tail is quoted from an untrusted process, so it is not ours to read."""
    inference_route = _load_route_module()
    msg = (
        "llama-server exited before becoming healthy\n\nllama-server output:\n"
        f"{CONFIG_IMPORT_ERROR}\n"
    )

    assert inference_route._unsupported_quantization_detail(msg) is None


def test_the_refusal_names_no_quantization_scheme():
    """The refusal names no quantization scheme, since the same wording fires for every scheme."""
    inference_route = _load_route_module()
    refusal = inference_route._COMPRESSED_TENSORS_INFERENCE_UNSUPPORTED_MESSAGE

    for scheme in ("NVFP4", "FP8", "W4A16", "W8A8", "INT8", "MXFP4"):
        assert scheme.lower() not in refusal.lower(), (
            f"the compressed-tensors refusal names {scheme}, but the errors it matches "
            f"do not identify the scheme"
        )
    assert "compressed-tensors" in refusal
    assert "GGUF" in refusal
    assert "bitsandbytes 4-bit" in refusal

    assert "NVFP4" in inference_route._NVFP4_INFERENCE_UNSUPPORTED_MESSAGE


@pytest.mark.parametrize("message", REFUSALS.values(), ids = list(REFUSALS))
@pytest.mark.parametrize("native", [False, True])
def test_a_host_that_can_run_vllm_is_pointed_at_it(message, native):
    with patch("core.inference.engine_install.support_reason", return_value = None):
        load_error = _load_failure(message, native = native)
        validate_error = _validation_failure(message, native = native)

    assert load_error.status_code == 500
    assert load_error.detail == EXPECTED + VLLM_HINT
    assert validate_error.status_code == 400
    assert validate_error.detail == EXPECTED + VLLM_HINT
    assert "pip install" not in load_error.detail


@pytest.mark.parametrize(
    "reason",
    [
        "Checking for a supported NVIDIA GPU.",
        "Managed engines currently require Linux x86_64 or Windows x64.",
        RuntimeError("probe failed"),
    ],
    ids = ["probe-pending", "unsupported-host", "probe-raises"],
)
def test_no_hint_unless_this_host_can_run_vllm(reason):
    inference_route = _load_route_module()
    kwargs = {"side_effect": reason} if isinstance(reason, Exception) else {"return_value": reason}
    with patch("core.inference.engine_install.support_reason", **kwargs):
        assert inference_route._unsupported_quantization_detail(CONFIG_IMPORT_ERROR) == EXPECTED


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_no_hint_when_the_request_already_chose_an_optional_engine(engine):
    inference_route = _load_route_module()
    with patch("core.inference.engine_install.support_reason", return_value = None):
        assert (
            inference_route._unsupported_quantization_detail(CONFIG_IMPORT_ERROR, engine)
            == EXPECTED
        )
        assert (
            inference_route._unsupported_quantization_detail(CONFIG_IMPORT_ERROR, "auto")
            == EXPECTED + VLLM_HINT
        )


def test_the_vllm_hint_never_touches_other_refusals():
    inference_route = _load_route_module()
    with patch("core.inference.engine_install.support_reason", return_value = None):
        assert (
            inference_route._unsupported_quantization_detail(
                "Unsloth: 'x' has per-module MLX quantization metadata {'format': 'nvfp4-pack-quantized'}"
            )
            == inference_route._NVFP4_INFERENCE_UNSUPPORTED_MESSAGE
        )
        assert (
            inference_route._unsupported_quantization_detail("Network connection timed out") is None
        )
