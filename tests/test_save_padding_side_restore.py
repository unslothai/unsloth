# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A save/push that raises must not leave the tokenizer's `padding_side` stuck at "left"."""

from __future__ import annotations

import ast
import contextlib
import gc
import json
import os
import pickle
import re
import shutil
import sys
import types
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE)


def _function_source(name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(_SOURCE, node)
            indent = len(segment) - len(segment.lstrip())
            if indent:
                segment = "\n".join(line[indent:] for line in segment.split("\n"))
            return segment
    raise AssertionError(f"{name} not found in unsloth/save.py")


def _load(*names, **env):
    """Exec the named top-level functions against `env` and return the namespace."""
    namespace = {
        # save.py imports this from models.mistral_format (#12144); no fixture is a Mistral view.
        "raise_if_merging_mistral_format_view": lambda model, save_method: None,
        "lora_relative_to_original_base": lambda model: contextlib.nullcontext(),
        # Unmerges FSDP2 streamed layers when a save raises; these fixtures never stream.
        "_unmerge_flags_on_error": lambda streamed: contextlib.nullcontext(),
        "nullcontext": contextlib.nullcontext,
        "PeftModel": type("PeftModel", (), {}),
        "_release_nf4_dequant_cache": lambda: None,
        "_save_bare_params4bit_quant_state": lambda model: contextlib.nullcontext(),
        "_with_bare_params4bit_state_dict": lambda model, original, args, kwargs: (args, kwargs),
        **env,
    }
    for name in names:
        exec(compile(_function_source(name), str(_SAVE_PY), "exec"), namespace)
    # _load strips decorators, so extracted context managers arrive as bare generators.
    for name in ("_left_padding_side", "_without_runtime_truncation"):
        if name in names:
            namespace[name] = contextlib.contextmanager(namespace[name])
    return namespace


class _Tokenizer:
    """Records the padding side each write observed; `error` makes the write raise."""

    def __init__(self, padding_side = "right", error = None):
        self.padding_side = padding_side
        self.error = error
        self.observed = []
        self.saved = []
        self.pushed = []

    def _record(self, sink, args, kwargs):
        self.observed.append(self.padding_side)
        if self.error is not None:
            raise self.error
        sink.append((args, kwargs))

    def save_pretrained(self, *args, **kwargs):
        self._record(self.saved, args, kwargs)

    def push_to_hub(self, *args, **kwargs):
        self._record(self.pushed, args, kwargs)


class _Processor:
    """Wraps a tokenizer like a ProcessorMixin: the save flips `.tokenizer`'s side, not ours."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.observed = []
        self.saved = []
        self.pushed = []

    def _record(self, sink, args, kwargs):
        self.observed.append(self.tokenizer.padding_side)
        if self.tokenizer.error is not None:
            raise self.tokenizer.error
        sink.append((args, kwargs))

    def save_pretrained(self, *args, **kwargs):
        self._record(self.saved, args, kwargs)

    def push_to_hub(self, *args, **kwargs):
        self._record(self.pushed, args, kwargs)


def _tokenizer(padding_side, processor, error = None):
    source = _Tokenizer(padding_side, error = error)
    return _Processor(source) if processor else source


def _source(tokenizer_or_processor):
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


class _AdapterModel:
    """Stands in for a PEFT-wrapped model on the adapter save and push paths."""

    def __init__(self):
        self.config = types.SimpleNamespace(_name_or_path = "base/model", model_type = "llama")
        self.saved = []
        self.pushed = []

    def add_model_tags(self, tags):
        pass

    def save_pretrained(self, *args, **kwargs):
        self.saved.append((args, kwargs))

    def push_to_hub(self, **kwargs):
        self.pushed.append(kwargs)


class _Config:
    _name_or_path = "base/model"
    model_type = "llama"

    def to_dict(self):
        return {"_name_or_path": self._name_or_path, "model_type": self.model_type}

    @classmethod
    def from_dict(cls, fields):
        config = cls()
        config._name_or_path = fields.get("_name_or_path")
        config.model_type = fields.get("model_type")
        return config


class _MergeModel:
    """Enough of a base model for the merged_16bit branch: `.model.layers` iterates empty."""

    def __init__(self):
        import torch

        self.config = _Config()
        self.model = types.SimpleNamespace(
            config = self.config,
            layers = [],
            embed_tokens = types.SimpleNamespace(weight = torch.zeros(2, 2)),
            norm = types.SimpleNamespace(weight = torch.zeros(2)),
        )
        self.lm_head = types.SimpleNamespace(weight = torch.zeros(2, 2))
        self.saved = []

    def add_model_tags(self, tags):
        pass

    def save_pretrained(self, *args, **kwargs):
        self.saved.append((args, kwargs))


def _save_model_environment(monkeypatch, tmp_path, preserve_error = None):
    """Exec unsloth_save_model with every helper stubbed except the two under test."""
    import huggingface_hub
    import psutil
    import torch

    # unsloth_save_model checks credentials via whoami before pushing; these tests stay offline.
    monkeypatch.setattr(huggingface_hub, "whoami", lambda token = None: {"name": "owner"})

    # Stubbed, not importorskip'd: the cross-platform runners ship no peft.
    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")
        peft.PeftModelForCausalLM = type("PeftModelForCausalLM", (), {})
        monkeypatch.setitem(sys.modules, "peft", peft)

    calls = {"upload": [], "eos": []}

    def preserve_eos(*args, **kwargs):
        calls["eos"].append((args, kwargs))
        if preserve_error is not None:
            raise preserve_error

    namespace = _load(
        "_normalize_safe_serialization",
        "_filter_push_to_hub_kwargs",
        "_collect_garbage",
        "_without_runtime_truncation",
        "_left_padding_side",
        "unsloth_save_model",
        os = os,
        re = re,
        shutil = shutil,
        pickle = pickle,
        json = json,
        gc = gc,
        sys = sys,
        psutil = psutil,
        torch = torch,
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        patch_saving_functions = lambda tokenizer: tokenizer,
        get_token = lambda: "fixture-token",
        upload_to_huggingface = lambda *args, **kwargs: calls["upload"].append(kwargs),
        logger = types.SimpleNamespace(warning_once = lambda *a, **k: None),
        clean_gpu_cache = lambda: None,
        fast_save_pickle = lambda *args, **kwargs: None,
        _refuse_mtp_variant = lambda model, variant: None,
        _attached_mtp_head = lambda model: None,
        _internal_mtp_save = contextlib.nullcontext,
        _save_mtp_head_beside_adapter = lambda model, directory: None,
        _stitch_mtp_head = lambda model, directory: None,
        _mtp_head_tensor_names = lambda model: [],
        _push_with_mtp_head = lambda kwargs: None,
        _source_tokenizer_class = lambda *args, **kwargs: None,
        _preserve_tokenizer_eos_token = preserve_eos,
        _restore_source_tokenizer_class = lambda *args, **kwargs: None,
        _preserve_repaired_tokenizer_class = lambda *args, **kwargs: None,
        _strip_absent_mtp_declaration = lambda config, names: None,
        _determine_username = lambda repo, old, token: (repo, "owner"),
        _free_cached_model = lambda model: None,
        _merge_lora = lambda *args: (None, None),
        dtype_from_config = lambda config: torch.float32,
        IS_KAGGLE_ENVIRONMENT = False,
        IS_COLAB_ENVIRONMENT = False,
        KAGGLE_TMP = str(tmp_path),
        LLAMA_WEIGHTS = (),
        LLAMA_LAYERNORMS = (),
        HfApi = types.SimpleNamespace(),
    )
    return namespace, calls


def _generic_save_environment(monkeypatch):
    """Exec unsloth_generic_save the way tests/saving/test_save_method_lora_routing.py does."""
    import torch

    zoo = types.ModuleType("unsloth_zoo.saving_utils")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", zoo)

    return _load(
        "_normalize_safe_serialization",
        "_is_adapter_save_method",
        "_honours_safe_serialization",
        "_refuse_unsaveable_text_core",
        "_cast_state_dict_16bit",
        "_requests_float32",
        "_stamp_saved_config_float32",
        "_without_runtime_truncation",
        "_left_padding_side",
        "_collect_garbage",
        "unsloth_generic_save",
        os = os,
        json = json,
        sys = sys,
        PeftModel = type("PeftModel", (), {}),
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        patch_saving_functions = lambda tokenizer: tokenizer,
        get_token = lambda: "fixture-token",
        get_model_name = lambda name: name,
        _push_merged_to_hub_revision = lambda kwargs: None,
        _prewarm_base_model_hub_cache = lambda *args, **kwargs: None,
        _is_qwen3_5_vlm = lambda model: False,
        _qwen3_5_vlm_state_dict_for_save = lambda state_dict: state_dict,
        _determine_username = lambda repo, old, token: (repo, "owner"),
        _modelscope_base_model_name = lambda *args, **kwargs: None,
        unsloth_save_model = lambda *args, **kwargs: None,
        _refuse_mtp_variant = lambda model, variant: None,
        _attached_mtp_head = lambda model: None,
        _internal_mtp_save = contextlib.nullcontext,
        _stitch_mtp_head = lambda model, directory: None,
        _mtp_head_tensor_names = lambda model: [],
        _release_layout_for_merged_save = lambda model, directory, method: False,
        _push_lfm3_audio_release_config = lambda model, save_directory, token = None: False,
        logger = types.SimpleNamespace(warning_once = lambda *a, **k: None),
        gc = types.SimpleNamespace(collect = lambda: None, get_stats = lambda: [{}, {}, {}]),
        torch = types.SimpleNamespace(
            bfloat16 = "bfloat16",
            float16 = "float16",
            float32 = "float32",
            save = lambda *args, **kwargs: None,
            cuda = types.SimpleNamespace(is_bf16_supported = lambda: False),
        ),
    )


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError("save failed")])
def test_left_padding_side_restores(padding_side, processor, error):
    """The guard itself: "left" during the write, the original side after, exception or not."""
    namespace = _load("_left_padding_side")
    guard = namespace["_left_padding_side"]
    tokenizer = _tokenizer(padding_side, processor, error = error)
    with (
        pytest.raises(RuntimeError, match = "save failed")
        if error is not None
        else contextlib.nullcontext()
    ):
        with guard(tokenizer):
            tokenizer.save_pretrained("out")
    assert tokenizer.observed == ["left"]
    assert _source(tokenizer).padding_side == padding_side


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError("push failed")])
def test_lora_push_restores_padding_side(
    monkeypatch, tmp_path, padding_side, processor, error
):
    """save.py's push_to_hub branch flipped padding_side without a finally."""
    namespace, _ = _save_model_environment(monkeypatch, tmp_path)
    tokenizer = _tokenizer(padding_side, processor, error = error)
    model = _AdapterModel()
    run = lambda: namespace["unsloth_save_model"](
        model,
        tokenizer,
        "owner/model",
        save_method = "lora",
        push_to_hub = True,
        token = "fixture-token",
    )
    if error is not None:
        with pytest.raises(RuntimeError, match = "push failed"):
            run()
    else:
        run()
        assert len(model.pushed) == 1
        assert tokenizer.pushed, "the tokenizer push must run"
    assert _source(tokenizer).padding_side == padding_side


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError("save failed")])
def test_lora_save_restores_padding_side(
    monkeypatch, tmp_path, padding_side, processor, error
):
    """The lora/direct save_pretrained branch had the same unguarded flip."""
    namespace, _ = _save_model_environment(monkeypatch, tmp_path)
    tokenizer = _tokenizer(padding_side, processor, error = error)
    model = _AdapterModel()
    run = lambda: namespace["unsloth_save_model"](
        model,
        tokenizer,
        str(tmp_path / "lora"),
        save_method = "lora",
    )
    if error is not None:
        with pytest.raises(RuntimeError, match = "save failed"):
            run()
    else:
        run()
        assert len(model.saved) == 1
        assert tokenizer.saved
    assert _source(tokenizer).padding_side == padding_side


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize(
    "where",
    [
        "save_pretrained",
        "preserve_eos",
        "none",
    ],
)
def test_merged_save_restores_padding_side(
    monkeypatch, tmp_path, padding_side, processor, where
):
    """The merged branch also ran post-save helpers before restoring padding_side."""
    error = RuntimeError("save failed") if where == "save_pretrained" else None
    preserve_error = RuntimeError("preserve failed") if where == "preserve_eos" else None
    namespace, calls = _save_model_environment(
        monkeypatch, tmp_path, preserve_error = preserve_error
    )
    tokenizer = _tokenizer(padding_side, processor, error = error)
    model = _MergeModel()
    run = lambda: namespace["unsloth_save_model"](
        model,
        tokenizer,
        str(tmp_path / "merged"),
        save_method = "merged_16bit",
        temporary_location = str(tmp_path / "scratch"),
    )
    if where == "none":
        run()
        assert len(model.saved) == 1
        assert len(calls["eos"]) == 1
    else:
        with pytest.raises(RuntimeError, match = "failed"):
            run()
    assert _source(tokenizer).padding_side == padding_side


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError("save failed")])
def test_generic_full_save_restores_padding_side(
    monkeypatch, tmp_path, padding_side, processor, error
):
    """unsloth_generic_save's non-PEFT branch flipped padding_side with no restore on raise."""
    namespace = _generic_save_environment(monkeypatch)
    tokenizer = _tokenizer(padding_side, processor, error = error)
    model = _AdapterModel()
    run = lambda: namespace["unsloth_generic_save"](
        model,
        tokenizer,
        save_directory = str(tmp_path / "full"),
        save_method = "lora",
    )
    if error is not None:
        with pytest.raises(RuntimeError, match = "save failed"):
            run()
    else:
        run()
        assert len(model.saved) == 1
        assert tokenizer.saved
    assert _source(tokenizer).padding_side == padding_side
