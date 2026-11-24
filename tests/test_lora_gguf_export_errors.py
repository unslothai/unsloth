# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""LoRA GGUF export of an MoE expert (PEFT target_parameters) adapter: llama.cpp's converter decides, and its failure
is named. Runs the real _unsloth_save_lora_gguf on a stand-in converter, without `import unsloth` (which wants a GPU)."""

import ast
import collections
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

_SAVE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
_TREE = ast.parse(_SAVE.read_text(encoding = "utf-8"))
_EXPERTS = ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
_UNMAPPED = "ValueError: Can not map tensor 'model.layers.0.mlp.experts.base_layer.weight'"

# Today's converter on an expert adapter, and one that has learned the layout.
_CONVERTER_TODAY = f"""
import sys
print("INFO:lora-to-gguf:Exporting model...")
print("Traceback (most recent call last):")
print({_UNMAPPED!r})
sys.exit(1)
"""
_CONVERTER_FUTURE = """
import sys
out = sys.argv[sys.argv.index("--outfile") + 1]
open(out, "wb").write(b"GGUF")
print("INFO:lora-to-gguf:Model successfully exported to " + out)
"""


class _Peft:
    def __init__(self, target_parameters = None):
        self.peft_config = {"default": SimpleNamespace(target_parameters = target_parameters)}


def _load(names):
    nodes = []
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_LORA_GGUF_OUTTYPES" for t in node.targets
        ):
            nodes.append(node)
    return nodes


def _writer(tmp_path, converter_source):
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_lora_to_gguf.py").write_text(converter_source, encoding = "utf-8")

    def save_lora_to_custom_dir(model, tokenizer, directory):
        (pathlib.Path(directory) / "adapter_model.safetensors").write_bytes(b"x" * 1024)
        (pathlib.Path(directory) / "adapter_config.json").write_text("{}", encoding = "utf-8")

    namespace = {
        "PeftModelForCausalLM": _Peft,
        "PeftModel": _Peft,
        "_clean_save_token": lambda token: None,
        "get_token": lambda: None,
        "_lora_base_model_id": lambda model: "unsloth/Inkling-2-Mini",
        "get_model_name": lambda name, load_in_4bit = False: name,
        "_model_basename": lambda name: name.split("/")[-1],
        "save_lora_to_custom_dir": save_lora_to_custom_dir,
        "install_llama_cpp": lambda **kwargs: None,
        "LLAMA_CPP_DEFAULT_DIR": str(llama_dir),
        "_loaded_via_remote_code": lambda model: False,
        "_base_config_dir": lambda base, token: None,
        "_apply_token_to_child_env": lambda env, token, explicit = False: None,
        "collections": collections,
        "os": os,
        "re": re,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "tempfile": tempfile,
    }
    names = {
        "_unsloth_save_lora_gguf",
        "_lora_target_parameters",
        "_converter_error_line",
        "_lora_gguf_conversion_error",
    }
    exec(compile(ast.Module(_load(names), []), str(_SAVE), "exec"), namespace)
    return namespace["_unsloth_save_lora_gguf"]


def test_expert_adapter_stops_with_the_converters_reason(tmp_path):
    out_dir = tmp_path / "export"
    save = _writer(tmp_path, _CONVERTER_TODAY)
    with pytest.raises(NotImplementedError) as raised:
        save(_Peft(_EXPERTS), None, str(out_dir), outtype = "q8_0")
    message = str(raised.value)
    assert "MoE expert weights" in message and "mlp.experts.gate_up_proj" in message
    assert _UNMAPPED in message
    assert "merged model as GGUF" in message


def test_expert_adapter_exports_once_the_converter_can_map_it(tmp_path):
    out_dir = tmp_path / "export"
    save = _writer(tmp_path, _CONVERTER_FUTURE)
    gguf = save(_Peft(_EXPERTS), None, str(out_dir), outtype = "q8_0")
    assert pathlib.Path(gguf).read_bytes() == b"GGUF"
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "Inkling-2-Mini-lora-q8_0.gguf",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]


def test_other_failures_name_the_converters_error():
    namespace = {}
    nodes = _load({"_lora_gguf_conversion_error"})
    exec(compile(ast.Module(nodes, []), str(_SAVE), "exec"), namespace)
    error = namespace["_lora_gguf_conversion_error"]
    plain = error(1, "KeyError: 'lm_head'", [])
    assert type(plain) is RuntimeError and "KeyError: 'lm_head'" in str(plain)
    # An expert adapter that fails for another reason gets the plain message, not the expert one.
    other = error(1, "KeyError: 'lm_head'", _EXPERTS)
    assert type(other) is RuntimeError
    silent = error(2, "", [])
    assert "See the output above" in str(silent)


def test_converter_error_line_is_the_last_exception_line():
    namespace = {"re": re}
    exec(compile(ast.Module(_load({"_converter_error_line"}), []), str(_SAVE), "exec"), namespace)
    line = namespace["_converter_error_line"]
    assert line(["INFO: start", "Traceback (most recent call last):", _UNMAPPED]) == _UNMAPPED
    assert line(["INFO: nothing went wrong"]) == ""
