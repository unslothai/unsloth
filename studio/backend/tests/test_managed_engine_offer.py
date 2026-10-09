# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/validate offers vLLM for checkpoints the Default engine cannot run (#11728).

Only where the host can run vLLM, only for the Default engine, and only when the
checkpoint's own config.json says compressed-tensors, or AWQ / GPTQ without the
packages the Default engine would need for them.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.inference import ValidateModelResponse

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _route():
    spec = importlib.util.spec_from_file_location(
        "inference_route_managed_engine_offer", _BACKEND_ROOT / "routes/inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


route = _route()
NVFP4 = {
    "quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}
}


def offer(
    metadata,
    supported = lambda name: True,
    **overrides,
):
    kwargs = {"engine": "auto", "is_gguf": False, "is_lora": False, "supported": supported}
    kwargs.update(overrides)
    return route._managed_engine_offer(metadata, **kwargs)


@pytest.fixture(autouse = True)
def _no_awq_or_gptq_packages():
    real = importlib.util.find_spec
    missing = {"gptqmodel", "awq", "auto_gptq"}
    with patch.object(
        importlib.util,
        "find_spec",
        side_effect = lambda name, *a: None if name in missing else real(name, *a),
    ):
        yield


def test_a_compressed_tensors_checkpoint_is_offered_vllm():
    assert offer(NVFP4) == {"quantization": "compressed-tensors", "engines": ["vllm", "sglang"]}
    # Vision checkpoints keep it under text_config.
    assert offer({"text_config": NVFP4})["engines"] == ["vllm", "sglang"]
    assert (
        offer({"quantization_config": {"quant_method": " Compressed-Tensors "}})["quantization"]
        == "compressed-tensors"
    )


@pytest.mark.parametrize("method", ["awq", "gptq", "AWQ"])
def test_awq_and_gptq_are_offered_while_the_default_engine_has_no_kernels_for_them(method):
    assert offer({"quantization_config": {"quant_method": method}})["engines"] == ["vllm", "sglang"]


def test_awq_and_gptq_stay_on_the_default_engine_once_it_can_run_them():
    with patch.object(importlib.util, "find_spec", return_value = object()):
        assert offer({"quantization_config": {"quant_method": "gptq"}}) is None
        assert offer({"quantization_config": {"quant_method": "awq"}}) is None
        assert offer(NVFP4) is not None


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"quantization_config": {"quant_method": "bitsandbytes"}},
        {"quantization_config": {"quant_method": "fp8"}},
        {"quantization_config": {}},
        {"quantization_config": "compressed-tensors"},
        {"quantization_config": {"quant_method": None}},
        {"text_config": "x"},
        None,
        [],
    ],
)
def test_nothing_is_offered_for_anything_else(metadata):
    assert offer(metadata) is None


def test_only_the_engines_this_host_can_run_are_offered():
    asked = []
    assert offer(NVFP4, supported = lambda name: asked.append(name) or False) is None
    assert asked == ["vllm", "sglang"]
    assert offer(NVFP4, supported = lambda name: name == "sglang")["engines"] == ["sglang"]


def test_the_gpu_probe_is_only_asked_about_these_checkpoints():
    asked = []
    offer(
        {"quantization_config": {"quant_method": "bitsandbytes"}},
        supported = lambda name: asked.append(name),
    )
    offer(NVFP4, engine = "vllm", supported = lambda name: asked.append(name))
    assert asked == []


@pytest.mark.parametrize(
    "overrides", [{"engine": "vllm"}, {"engine": "sglang"}, {"is_gguf": True}, {"is_lora": True}]
)
def test_nothing_is_offered_for_an_explicit_engine_a_gguf_or_an_adapter(overrides):
    assert offer(NVFP4, **overrides) is None


def _config(tmp_path, metadata, **extra):
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding = "utf-8")
    return SimpleNamespace(is_local = True, path = str(tmp_path), identifier = "local/model", **extra)


def test_the_check_reads_the_checkpoints_own_config(tmp_path):
    with patch("core.inference.engine_install.support_reason", return_value = None):
        assert route._managed_engine_offer_for(_config(tmp_path, NVFP4), None) == {
            "quantization": "compressed-tensors",
            "engines": ["vllm", "sglang"],
        }
        assert route._managed_engine_offer_for(_config(tmp_path, NVFP4, is_lora = True), None) is None


@pytest.mark.parametrize(
    "reason",
    [
        "Checking for a supported NVIDIA GPU.",
        "Managed engines currently require Linux x86_64 or Windows x64.",
        "Requires an NVIDIA GPU with compute capability 8.0 or newer and driver 580 or newer.",
    ],
)
def test_the_check_follows_the_engines_own_support_verdict(tmp_path, reason):
    with patch("core.inference.engine_install.support_reason", return_value = reason):
        assert route._managed_engine_offer_for(_config(tmp_path, NVFP4), None) is None


def test_the_check_never_fails_validation(tmp_path):
    missing = SimpleNamespace(is_local = True, path = str(tmp_path / "nope"), identifier = "x")
    assert route._managed_engine_offer_for(missing, None) is None
    (tmp_path / "config.json").write_text("{not json", encoding = "utf-8")
    assert (
        route._managed_engine_offer_for(
            SimpleNamespace(is_local = True, path = str(tmp_path), identifier = "x"), None
        )
        is None
    )
    with patch("core.inference.engine_install.support_reason", side_effect = RuntimeError("probe")):
        assert route._managed_engine_offer_for(_config(tmp_path, NVFP4), None) is None


def test_the_response_carries_the_offer_and_defaults_to_none():
    base = {"valid": True, "message": "ok"}
    assert ValidateModelResponse(**base).managed_engine_offer is None
    response = ValidateModelResponse(
        **base, managed_engine_offer = {"quantization": "compressed-tensors", "engines": ["vllm"]}
    )
    assert response.model_dump()["managed_engine_offer"] == {
        "quantization": "compressed-tensors",
        "engines": ["vllm"],
    }
