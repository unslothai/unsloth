# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""An explicit quantization_config must not be swapped for a pre-quantized repo's saved one by the name remap."""

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from unsloth.models import loader, loader_utils  # noqa: E402

BitsAndBytesConfig = transformers.BitsAndBytesConfig

FP4 = dict(
    load_in_4bit = True,
    bnb_4bit_quant_type = "fp4",
    bnb_4bit_use_double_quant = False,
)


class _Resolved(Exception):
    pass


def _no_remote_mapper():
    return {}, {}, {}, {}, {}


@pytest.fixture
def bnb_host(monkeypatch):
    # CPU hosts turn bitsandbytes off, which skips the 4bit remap entirely; pretend a CUDA host where it runs.
    monkeypatch.setattr(loader, "ALLOW_BITSANDBYTES", True)
    monkeypatch.setattr(loader, "ALLOW_PREQUANTIZED_MODELS", True)
    monkeypatch.setattr(loader_utils, "_get_new_mapper", _no_remote_mapper)
    monkeypatch.setattr(loader_utils, "_complete_in_cache", lambda *a, **k: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _resolve(cls, name, **kwargs):
    seen = []

    def stop(resolved):
        seen.append(resolved)
        raise _Resolved

    with pytest.raises(_Resolved):
        cls.from_pretrained(name, on_model_resolved = stop, **kwargs)
    return seen[0]


LOADERS = [loader.FastLanguageModel, loader.FastModel]


@pytest.mark.parametrize("cls", LOADERS, ids = lambda c: c.__name__)
@pytest.mark.parametrize(
    "config",
    [BitsAndBytesConfig(**FP4), dict(FP4)],
    ids = ["BitsAndBytesConfig", "dict"],
)
def test_explicit_config_keeps_the_unquantized_repo(bnb_host, cls, config):
    assert _resolve(cls, "unsloth/Qwen3-0.6B", quantization_config = config) == "unsloth/Qwen3-0.6B"


@pytest.mark.parametrize("cls", LOADERS, ids = lambda c: c.__name__)
def test_explicit_config_maps_to_the_16bit_mirror(bnb_host, cls):
    resolved = _resolve(cls, "Qwen/Qwen3-8B", quantization_config = BitsAndBytesConfig(**FP4))
    assert resolved == "unsloth/Qwen3-8B"


@pytest.mark.parametrize("cls", LOADERS, ids = lambda c: c.__name__)
@pytest.mark.parametrize("kwargs", [{}, {"load_in_4bit": True}], ids = ["default", "load_in_4bit"])
def test_flags_only_still_map_to_the_prequantized_repo(bnb_host, cls, kwargs):
    assert _resolve(cls, "unsloth/Qwen3-0.6B", **kwargs) == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"


@pytest.mark.parametrize("cls", LOADERS, ids = lambda c: c.__name__)
def test_explicitly_named_prequantized_repo_is_kept(bnb_host, cls):
    name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    assert _resolve(cls, name, quantization_config = BitsAndBytesConfig(**FP4)) == name


@pytest.mark.parametrize("cls", LOADERS, ids = lambda c: c.__name__)
def test_8bit_config_still_maps_a_4bit_name_to_16bit(bnb_host, cls):
    name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    config = BitsAndBytesConfig(load_in_8bit = True)
    assert _resolve(cls, name, quantization_config = config) == "unsloth/Qwen3-0.6B"


@pytest.mark.parametrize(
    "name,config,expected",
    [
        ("unsloth/Qwen3-0.6B", None, "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-0.6B", dict(FP4), "unsloth/Qwen3-0.6B"),
        ("Qwen/Qwen3-8B", dict(FP4), "unsloth/Qwen3-8B"),
        # An FP8 checkpoint is no better a home for a bnb config: keep the old remap.
        ("Qwen/Qwen3-8B-FP8", dict(FP4), "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
        ("unsloth/Qwen3-0.6B-unsloth-bnb-4bit", dict(FP4), "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"),
    ],
)
def test_get_model_name_with_quantization_config(monkeypatch, name, config, expected):
    monkeypatch.setattr(loader_utils, "_get_new_mapper", _no_remote_mapper)
    monkeypatch.setattr(loader_utils, "_complete_in_cache", lambda *a, **k: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    mapped = loader_utils.get_model_name(name, load_in_4bit = True, quantization_config = config)
    assert mapped.lower() == expected.lower()


@pytest.mark.parametrize("load_in_4bit,load_in_fp8", [(False, False), (True, "block")])
@pytest.mark.parametrize(
    "name", ["unsloth/Qwen3-0.6B", "Qwen/Qwen3-8B", "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"]
)
def test_get_model_name_16bit_and_fp8_paths_ignore_the_config(
    monkeypatch, name, load_in_4bit, load_in_fp8
):
    monkeypatch.setattr(loader_utils, "_get_new_mapper", _no_remote_mapper)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    kwargs = dict(load_in_4bit = load_in_4bit, load_in_fp8 = load_in_fp8)
    plain = loader_utils.get_model_name(name, **kwargs)
    assert loader_utils.get_model_name(name, quantization_config = dict(FP4), **kwargs) == plain


@pytest.mark.parametrize("offline_via", ["env", "local_files_only"])
def test_offline_without_the_16bit_repo_keeps_the_prequantized_remap(monkeypatch, offline_via):
    monkeypatch.setattr(loader_utils, "_get_new_mapper", _no_remote_mapper)
    monkeypatch.setattr(
        loader_utils, "_complete_in_cache", lambda repo, *a, **k: "bnb-4bit" in repo
    )
    kwargs = dict(load_in_4bit = True, quantization_config = dict(FP4))
    if offline_via == "env":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        kwargs["local_files_only"] = True
    mapped = loader_utils.get_model_name("unsloth/Qwen3-0.6B", **kwargs)
    assert mapped == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    monkeypatch.setattr(loader_utils, "_complete_in_cache", lambda *a, **k: True)
    assert loader_utils.get_model_name("unsloth/Qwen3-0.6B", **kwargs) == "unsloth/Qwen3-0.6B"
