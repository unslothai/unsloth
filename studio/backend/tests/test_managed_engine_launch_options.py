# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""KV cache dtype, seed and GPU memory ceiling for the managed vLLM / SGLang engines."""

import json
from types import SimpleNamespace

import pytest

from core.inference.engine_adapters import ADAPTERS

# What validate_model returns for an unquantized text checkpoint with a tool parser.
_DEFAULT_OPTIONS = {
    "tool_parser": "hermes",
    "precision": "auto",
    "parallelism": "tensor",
    "is_vision": False,
    "quantization": None,
    "disable_cuda_graph": False,
    "load_format": "auto",
    "engine_version": "0.31.0",
}

# The argv main built for _DEFAULT_OPTIONS (launcher path at index 2 left out).
_MAIN_ARGV = {
    "vllm": [
        "/env/bin/python",
        "-I",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "org/model",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.8",
        "--tensor-parallel-size",
        "1",
        "--tool-call-parser",
        "hermes",
        "--enable-auto-tool-choice",
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        "45678",
        "--served-model-name",
        "org/model",
        "--load-format",
        "auto",
    ],
    "sglang": [
        "/env/bin/python",
        "-I",
        "--model-path",
        "org/model",
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.8",
        "--tensor-parallel-size",
        "1",
        "--tool-call-parser",
        "hermes",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--host",
        "127.0.0.1",
        "--port",
        "45678",
        "--served-model-name",
        "org/model",
        "--load-format",
        "auto",
    ],
}


def _argv(engine, **extra):
    argv = ADAPTERS[engine].command(
        "/env/bin/python",
        "org/model",
        45678,
        "key",
        4096,
        0.8,
        1,
        options = {**_DEFAULT_OPTIONS, **extra},
    )
    return argv[:2] + argv[3:]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_default_options_launch_the_same_argv_as_before(engine):
    assert _argv(engine) == _MAIN_ARGV[engine]


@pytest.mark.parametrize(
    "engine, kv, seed", [("vllm", "fp8", "--seed"), ("sglang", "fp8_e4m3", "--random-seed")]
)
def test_kv_cache_dtype_and_seed_reach_each_engine(engine, kv, seed):
    argv = _argv(engine, kv_cache_dtype = "fp8", seed = 7)
    assert argv[argv.index("--kv-cache-dtype") + 1] == kv
    assert argv[argv.index(seed) + 1] == "7"
    zero = _argv(engine, seed = 0)
    assert zero[zero.index(seed) + 1] == "0"


@pytest.mark.parametrize(
    "engine, flag", [("vllm", "--gpu-memory-utilization"), ("sglang", "--mem-fraction-static")]
)
def test_memory_ceiling_lowers_but_never_raises_the_measured_budget(engine, flag):
    lowered = _argv(engine, gpu_memory_utilization = 0.5)
    assert lowered[lowered.index(flag) + 1] == "0.5"
    kept = _argv(engine, gpu_memory_utilization = 0.95)
    assert kept[kept.index(flag) + 1] == "0.8"


def _config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 8}))
    return SimpleNamespace(is_local = True, path = str(tmp_path))


def _compute_cap(monkeypatch, cap):
    from core.inference import managed_engine
    monkeypatch.setattr(managed_engine, "profile", lambda engine: {"platform": "cuda"})
    monkeypatch.setattr(
        managed_engine.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(stdout = f"{cap}\n")
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_unset_options_stay_out_of_the_launch_options(tmp_path, engine, monkeypatch):
    from core.inference.managed_engine import validate_model

    _compute_cap(monkeypatch, "9.0")
    options = validate_model(_config(tmp_path), engine = engine)
    assert not {"kv_cache_dtype", "seed", "gpu_memory_utilization"} & set(options)
    options = validate_model(
        _config(tmp_path),
        engine = engine,
        kv_cache_dtype = "fp8",
        seed = 0,
        gpu_memory_utilization = 0.6,
    )
    assert (options["kv_cache_dtype"], options["seed"], options["gpu_memory_utilization"]) == (
        "fp8",
        0,
        0.6,
    )


def test_seed_and_memory_ceiling_share_the_compile_cache():
    from core.inference.managed_engine import ManagedEngine

    engine = ManagedEngine("vllm")
    info = {"profile_digest": "d"}
    base = engine._cache_key(info, "m", [0], dict(_DEFAULT_OPTIONS))
    assert base == engine._cache_key(
        info, "m", [0], {**_DEFAULT_OPTIONS, "seed": 3, "gpu_memory_utilization": 0.5}
    )
    assert base != engine._cache_key(info, "m", [0], {**_DEFAULT_OPTIONS, "kv_cache_dtype": "fp8"})
    assert engine._cache_key(info, "m", [0], None) == engine._cache_key(info, "m", [0], None)


@pytest.mark.parametrize(
    "field, value",
    [
        ("engine_kv_cache_dtype", "fp8"),
        ("engine_seed", 1),
        ("engine_gpu_memory_utilization", 0.5),
    ],
)
def test_a_changed_launch_option_reloads_the_resident_engine(field, value):
    from models.inference import LoadRequest
    from routes.inference import _non_gguf_runtime_settings_match

    backend = SimpleNamespace(
        active_model_name = "model",
        models = {"model": {"engine": "vllm", "gpu_ids_requested": [0]}},
    )
    request = LoadRequest(model_path = "model", engine = "vllm", gpu_ids = [0])
    assert _non_gguf_runtime_settings_match(backend, request)
    assert not _non_gguf_runtime_settings_match(backend, request.model_copy(update = {field: value}))
    backend.models["model"][field] = value
    assert _non_gguf_runtime_settings_match(backend, request.model_copy(update = {field: value}))


def test_overrides_store_only_set_options_and_load_them_back():
    from utils.openai_auto_switch_settings import (
        model_override_load_kwargs,
        normalize_model_override,
    )

    assert normalize_model_override(
        {
            "engine": "vllm",
            "engine_kv_cache_dtype": "auto",
            "engine_seed": None,
            "engine_gpu_memory_utilization": None,
        }
    ) == {"engine": "vllm"}
    entry = normalize_model_override(
        {
            "engine": "vllm",
            "engine_kv_cache_dtype": "fp8",
            "engine_seed": 0,
            "engine_gpu_memory_utilization": 0.75,
        }
    )
    assert entry == {
        "engine": "vllm",
        "engine_kv_cache_dtype": "fp8",
        "engine_seed": 0,
        "engine_gpu_memory_utilization": 0.75,
    }
    assert normalize_model_override({"engine_gpu_memory_utilization": 1.5}) == {}
    kwargs = model_override_load_kwargs(entry, is_gguf = False)
    assert kwargs["engine_kv_cache_dtype"] == "fp8"
    assert kwargs["engine_seed"] == 0
    assert kwargs["engine_gpu_memory_utilization"] == 0.75
    assert "engine_seed" not in model_override_load_kwargs(entry, is_gguf = True)


@pytest.mark.parametrize("cap,ok", [("7.5", False), ("8.0", True), ("8.9", True)])
def test_an_fp8_kv_cache_is_refused_before_ampere(tmp_path, monkeypatch, cap, ok):
    # Colab T4, vLLM 0.30: "FP8 KV cache is not supported ... on Tesla T4 (compute capability 7.5)".
    from core.inference.managed_engine import validate_model
    _compute_cap(monkeypatch, cap)
    if ok:
        assert validate_model(_config(tmp_path), kv_cache_dtype = "fp8")["kv_cache_dtype"] == "fp8"
    else:
        with pytest.raises(ValueError, match = "Ampere"):
            validate_model(_config(tmp_path), kv_cache_dtype = "fp8")


def test_a_default_kv_cache_never_probes_the_gpu(tmp_path, monkeypatch):
    from core.inference import managed_engine

    def refuse(argv, **kwargs):
        raise AssertionError("nvidia-smi probed for a default KV cache")

    monkeypatch.setattr(managed_engine, "profile", lambda engine: {"platform": "cuda"})
    monkeypatch.setattr(managed_engine.subprocess, "run", refuse)
    managed_engine.validate_model(_config(tmp_path), kv_cache_dtype = "auto")
