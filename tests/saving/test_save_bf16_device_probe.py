# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""The bf16-or-f16 decisions in unsloth/save.py ask the device-aware
`is_bfloat16_supported()` probe, not `torch.cuda.is_bf16_supported()`: CUDA keeps its
per-device answer while XPU and NPU hosts stop being silently rewritten to
f16/float16, and hosts with no accelerator still get f16. Device mocks only —
no real XPU or NPU hardware was exercised.
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
import types
from pathlib import Path

import pytest

_SAVE_PY = Path(__file__).resolve().parent.parent.parent / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE)

_BF16_CHECKED_FUNCTIONS = (
    "save_to_gguf",
    "unsloth_generic_save",
    "_gguf_source_dtype",
    "_preflight_gguf_disk",
)

# (label for the mocked host, the device-aware probe's answer there). The torch.cuda
# probe is held at False in every fixture, so pre-fix code — which asked torch.cuda
# only — always lands on f16 and the bf16 cases fail until the probe is swapped.
DEVICE_PROBES = (
    ("cuda-ampere", True),
    ("cuda-pre-ampere", False),
    ("xpu-native-bf16", True),
    ("xpu-emulated-bf16", False),
    ("npu-ascend", True),
    ("cpu-only", False),
)


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
    namespace = {**env}
    for name in names:
        exec(compile(_function_source(name), str(_SAVE_PY), "exec"), namespace)
    return namespace


def _torch():
    """torch.cuda reports "no bf16" everywhere; only the device-aware probe varies."""
    return types.SimpleNamespace(
        bfloat16 = "bfloat16",
        float16 = "float16",
        float32 = "float32",
        save = lambda *args, **kwargs: None,
        cuda = types.SimpleNamespace(is_bf16_supported = lambda: False),
    )


def test_save_py_asks_the_device_aware_probe_for_bf16():
    for name in _BF16_CHECKED_FUNCTIONS:
        source = _function_source(name)
        assert "torch.cuda.is_bf16_supported" not in source, (
            f"{name} still asks torch.cuda whether bf16 is supported"
        )
        assert "is_bfloat16_supported(" in source, (
            f"{name} must consult the device-aware is_bfloat16_supported()"
        )
    assert "is_bfloat16_supported" in _SOURCE


@pytest.mark.parametrize("device, bf16", DEVICE_PROBES)
def test_gguf_source_dtype_follows_the_device_probe(device, bf16):
    """The disk-estimate mirror of save_to_gguf names the same dtype the export writes."""
    namespace = _load(
        "_gguf_source_dtype",
        dtype_from_config = lambda config: "bfloat16",
        torch = _torch(),
        is_bfloat16_supported = lambda: bf16,
    )
    model = types.SimpleNamespace(config = types.SimpleNamespace())
    assert namespace["_gguf_source_dtype"](model) == ("bf16" if bf16 else "f16")


class _StopConversion(Exception):
    pass


def _save_to_gguf_namespace(bf16, observed):
    printed = []

    def _choose_first_conversion(quantization_methods, model_dtype, has_imatrix = False):
        observed["model_dtype"] = model_dtype
        return "bf16"

    namespace = _load(
        "save_to_gguf",
        os = os,
        print = lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
        logger = types.SimpleNamespace(warning = lambda *args, **kwargs: None),
        torch = _torch(),
        is_bfloat16_supported = lambda: bf16,
        _choose_first_conversion = _choose_first_conversion,
        IMATRIX_QUANTS = {},
        ALLOWED_QUANTS = {"q4_k_m": "80% quality", "bf16": "", "f16": "", "q8_0": ""},
        check_llama_cpp = lambda: ("llama-quantize", "convert.py"),
        # check_llama_cpp's bare `except:` swallows a sentinel raise; the staging call is past
        # everything under test and stops the function before real llama.cpp work starts.
        _gguf_conversion_staging = lambda directory: (_ for _ in ()).throw(
            _StopConversion()
        ),
    )
    return namespace, printed


@pytest.mark.parametrize("device, bf16", DEVICE_PROBES)
def test_save_to_gguf_drops_bf16_only_when_the_device_probe_says_so(device, bf16):
    """model_dtype reaches _choose_first_conversion after the hardware drop, and the
    announced initial conversion is the one actually written."""
    observed = {}
    namespace, printed = _save_to_gguf_namespace(bf16, observed)
    with pytest.raises(_StopConversion):
        namespace["save_to_gguf"](
            model_name = "MyModel",
            model_type = "llama",
            model_dtype = "bfloat16",
            model_directory = "model",
            quantization_method = ["q4_k_m"],
        )
    assert observed["model_dtype"] == ("bf16" if bf16 else "f16")
    assert any(f"Converting HF to GGUF {'bf16' if bf16 else 'f16'}" in line for line in printed)


class _FullModel:
    """A model with no adapter: unsloth_generic_save writes its own weights."""

    def __init__(self):
        self.config = types.SimpleNamespace(_name_or_path = "base/model", model_type = "llama")
        self.saved = []

    def state_dict(self):
        return {"weight": object()}

    def save_pretrained(self, directory, **kwargs):
        self.saved.append((directory, kwargs))


def _generic_save_namespace(bf16, cast_dtypes):
    namespace = _load(
        "_normalize_safe_serialization",
        "_is_adapter_save_method",
        "_honours_safe_serialization",
        "_refuse_unsaveable_text_core",
        "_requests_float32",
        "_stamp_saved_config_float32",
        "unsloth_generic_save",
        os = os,
        torch = _torch(),
        is_bfloat16_supported = lambda: bf16,
        _cast_state_dict_16bit = lambda model, state_dict, target_dtype: (
            cast_dtypes.append(target_dtype),
            state_dict,
        )[1],
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
        unsloth_save_model = lambda *args, **kwargs: None,
        _refuse_mtp_variant = lambda model, variant: None,
        _attached_mtp_head = lambda model: None,
        _internal_mtp_save = contextlib.nullcontext,
        _stitch_mtp_head = lambda model, directory: None,
        _mtp_head_tensor_names = lambda model: [],
        _release_layout_for_merged_save = lambda model, directory, method: False,
        _push_lfm3_audio_release_config = lambda model, save_directory, token = None: False,
        raise_if_merging_mistral_format_view = lambda model, save_method: None,
        lora_relative_to_original_base = lambda model: contextlib.nullcontext(),
        _unmerge_flags_on_error = lambda streamed: contextlib.nullcontext(),
        nullcontext = contextlib.nullcontext,
        logger = types.SimpleNamespace(warning_once = lambda *args, **kwargs: None),
        gc = types.SimpleNamespace(collect = lambda: None, get_stats = lambda: [{}, {}, {}]),
        _release_nf4_dequant_cache = lambda: None,
        _save_bare_params4bit_quant_state = lambda model: contextlib.nullcontext(),
        _with_bare_params4bit_state_dict = lambda model, original, args, kwargs: (args, kwargs),
    )
    return namespace


@pytest.mark.parametrize("device, bf16", DEVICE_PROBES)
def test_a_merged_16bit_save_casts_by_the_device_probe(device, bf16, tmp_path, monkeypatch):
    """A full finetune on a bf16-capable XPU or NPU was written float16; the device-aware
    answer is what the cast must follow."""
    # An empty module makes the reconcile_mtp_config import raise ImportError, which the
    # save already treats as "older zoo, nothing to reconcile".
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.saving_utils",
        types.ModuleType("unsloth_zoo.saving_utils"),
    )
    cast_dtypes = []
    namespace = _generic_save_namespace(bf16, cast_dtypes)
    model = _FullModel()
    namespace["unsloth_generic_save"](
        model, None, save_directory = str(tmp_path), save_method = "merged_16bit"
    )
    assert cast_dtypes == ["bfloat16" if bf16 else "float16"]
    assert len(model.saved) == 1
