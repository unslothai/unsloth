# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import fnmatch
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class DispatchReached(Exception):
    pass


@pytest.fixture
def loader():
    path = Path(__file__).resolve().parents[1] / "unsloth/models/loader.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastLanguageModel"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )
    method.decorator_list = []
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_precision_flags_conflict"
    )
    captured = {"config_calls": 0}

    def dispatch(**kwargs):
        captured["dispatch"] = kwargs
        raise DispatchReached()

    def config(*args, **kwargs):
        captured["config_calls"] += 1
        return SimpleNamespace(model_type = "llama", rope_scaling = None)

    def no_adapter(*args, **kwargs):
        raise ValueError("No adapter")

    env = dict(
        os = os,
        DEFAULT_DEVICE_MAP = "sequential",
        OFFLOAD_EMBEDDING_AUTO = "auto",
        torch = SimpleNamespace(
            float16 = "float16", bfloat16 = "bfloat16", float32 = "float32", dtype = type(None)
        ),
        _requested_float32 = lambda dtype: False,
        hf_login = lambda token: token,
        requested_device_map = lambda device: device,
        prepare_device_map = lambda: ("sequential", False),
        ALLOW_BITSANDBYTES = True,
        ALLOW_PREQUANTIZED_MODELS = True,
        USE_MODELSCOPE = False,
        SUPPORTS_LLAMA32 = True,
        get_model_name = lambda name, **kwargs: name,
        _revision_for_resolved_repo = lambda revision, *args: revision,
        AutoConfig = SimpleNamespace(from_pretrained = config),
        PeftConfig = SimpleNamespace(from_pretrained = no_adapter),
        get_transformers_model_type = lambda *args, **kwargs: ["llama"],
        FastLlamaModel = SimpleNamespace(from_pretrained = dispatch),
        apply_unsloth_gradient_checkpointing = lambda value, *args: value,
        _resolve_checkpoint_tokenizer_name = lambda *args: None,
        patch_compiling_bitsandbytes = lambda: None,
        _get_dtype = lambda dtype: dtype,
        _revision_for_tokenizer_repo = lambda *args: None,
    )
    exec(compile(ast.Module(body = [helper, method], type_ignores = []), str(path), "exec"), env)
    return env, captured


@pytest.mark.parametrize(
    "kwargs",
    [
        {"load_in_16bit": True},
        {"load_in_4bit": True, "load_in_16bit": True},
        {"load_in_16bit": True, "quantization_config": {"load_in_4bit": True}},
        {"load_in_fp8": True},
    ],
)
def test_conflicts_fail_before_model_loading(loader, kwargs):
    env, captured = loader
    with pytest.raises(RuntimeError, match = "Can only load in"):
        env["from_pretrained"](**kwargs)
    assert "dispatch" not in captured


@pytest.mark.parametrize(
    "kwargs, allow_bnb, expected_4bit",
    [
        ({}, True, True),
        ({"load_in_4bit": False, "load_in_16bit": True}, True, False),
        ({"model_name": "example/model-bf16", "load_in_16bit": True}, True, False),
        ({"load_in_16bit": True}, False, False),
        ({"load_in_16bit": True, "quantization_config": {"quant_method": "awq"}}, True, False),
    ],
)
def test_valid_precision_and_existing_overrides(loader, kwargs, allow_bnb, expected_4bit):
    env, captured = loader
    env["ALLOW_BITSANDBYTES"] = allow_bnb
    with pytest.raises(DispatchReached):
        env["from_pretrained"](**kwargs)
    assert captured["dispatch"]["load_in_4bit"] is expected_4bit
    if "quantization_config" in kwargs:
        assert captured["dispatch"]["quantization_config"] == kwargs["quantization_config"]


@pytest.mark.parametrize("base_name, expected", [("owner/base-bf16", False), ("owner/base", None)])
def test_adapter_base_precision_is_resolved_before_validation(loader, base_name, expected):
    env, captured = loader

    def config(name, **kwargs):
        if name == "owner/adapter":
            raise ValueError("Adapter has no model config")
        return SimpleNamespace(model_type = "llama", rope_scaling = None)

    env["AutoConfig"] = SimpleNamespace(from_pretrained = config)
    env["PeftConfig"] = SimpleNamespace(
        from_pretrained = lambda *a, **k: SimpleNamespace(base_model_name_or_path = base_name)
    )
    if expected is None:
        with pytest.raises(RuntimeError, match = "Can only load in"):
            env["from_pretrained"](model_name = "owner/adapter", load_in_16bit = True)
        assert "dispatch" not in captured
    else:
        with pytest.raises(DispatchReached):
            env["from_pretrained"](model_name = "owner/adapter", load_in_16bit = True)
        assert captured["dispatch"]["model_name"] == base_name
        assert captured["dispatch"]["load_in_4bit"] is expected


@pytest.fixture
def modelscope_snapshot(monkeypatch, tmp_path):
    cache = tmp_path / "modelscope-cache"
    cache.mkdir()
    downloaded = []
    calls = []

    def snapshot_download(name, allow_file_pattern = None):
        calls.append((name, allow_file_pattern))
        for filename in (
            "config.json",
            "adapter_config.json",
            "configuration_custom.py",
            "model.safetensors",
        ):
            if allow_file_pattern is None or any(
                fnmatch.fnmatch(filename, pattern) for pattern in allow_file_pattern
            ):
                (cache / filename).write_text("test fixture")
                downloaded.append(filename)
        return str(cache)

    monkeypatch.setitem(
        sys.modules, "modelscope", SimpleNamespace(snapshot_download = snapshot_download)
    )
    return cache, downloaded, calls


def use_modelscope_adapter(env, cache, base_name):
    def config(name, **kwargs):
        if name == str(cache):
            raise ValueError("Adapter has no model config")
        return SimpleNamespace(model_type = "llama", rope_scaling = None)

    env["AutoConfig"] = SimpleNamespace(from_pretrained = config)
    env["PeftConfig"] = SimpleNamespace(
        from_pretrained = lambda *a, **k: SimpleNamespace(base_model_name_or_path = base_name)
    )


@pytest.mark.parametrize(
    "kwargs, adapter_base",
    [
        ({"load_in_16bit": True}, None),
        ({"load_in_4bit": True, "load_in_16bit": True}, None),
        ({"load_in_16bit": True, "quantization_config": {"load_in_4bit": True}}, None),
        ({"load_in_16bit": True}, "owner/base"),
    ],
)
def test_modelscope_conflicts_do_not_download_weights(
    loader, modelscope_snapshot, kwargs, adapter_base
):
    env, captured = loader
    cache, downloaded, calls = modelscope_snapshot
    env["USE_MODELSCOPE"] = True
    if adapter_base:
        use_modelscope_adapter(env, cache, adapter_base)

    with pytest.raises(RuntimeError, match = "Can only load in"):
        env["from_pretrained"](model_name = "owner/model", **kwargs)

    assert "dispatch" not in captured
    assert "model.safetensors" not in downloaded
    assert "config.json" in downloaded
    assert "adapter_config.json" in downloaded
    assert "configuration_custom.py" in downloaded
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, adapter_base, expected_4bit",
    [
        ({}, None, True),
        ({"load_in_4bit": False, "load_in_16bit": True}, None, False),
        ({"load_in_16bit": True}, "owner/base-bf16", False),
    ],
)
def test_modelscope_valid_loads_download_weights(
    loader, modelscope_snapshot, kwargs, adapter_base, expected_4bit
):
    env, captured = loader
    cache, downloaded, calls = modelscope_snapshot
    env["USE_MODELSCOPE"] = True
    if adapter_base:
        use_modelscope_adapter(env, cache, adapter_base)

    with pytest.raises(DispatchReached):
        env["from_pretrained"](model_name = "owner/model", **kwargs)

    assert "model.safetensors" in downloaded
    assert captured["dispatch"]["model_name"] == (adapter_base or str(cache))
    assert captured["dispatch"]["load_in_4bit"] is expected_4bit
    assert calls[-1] == ("owner/model", None)
