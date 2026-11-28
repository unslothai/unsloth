# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""LoRA GGUF export errors, run on stand-in converters without `import unsloth` (which wants a GPU)."""

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


def _writer(
    tmp_path,
    converter_source,
    base = "unsloth/Inkling-2-Mini",
):
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
        "_lora_base_model_id": lambda model: base,
        "get_model_name": lambda name, load_in_4bit = False: name,
        "_model_basename": lambda name: name.split("/")[-1],
        "save_lora_to_custom_dir": save_lora_to_custom_dir,
        "install_llama_cpp": lambda **kwargs: None,
        "LLAMA_CPP_DEFAULT_DIR": str(llama_dir),
        "_loaded_via_remote_code": lambda model: False,
        "_base_config_dir": lambda base, token: None,
        "_apply_token_to_child_env": lambda env, token, explicit = False: None,
        "is_inkling2_config": lambda config: False,
        "inkling2_lora_gguf_adapter": None,
        "_lora_gguf_converter_input": lambda model,
        tokenizer,
        lora_dir,
        scratch_dir,
        **kwargs: lora_dir,
        "collections": collections,
        "os": os,
        "re": re,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "tempfile": tempfile,
        "Path": pathlib.Path,
        "IS_WINDOWS": False,
        "_WINDOWS_MAX_PATH_CHARS": 259,
    }
    names = {
        "_unsloth_save_lora_gguf",
        "_lora_target_parameters",
        "_converter_error_line",
        "_lora_gguf_conversion_error",
        "_local_base_config_dir",
        "_place_gguf_output",
        "_partial_gguf_path",
    }
    exec(compile(ast.Module(_load(names), []), str(_SAVE), "exec"), namespace)
    loader = _SAVE.parent / "models" / "loader_utils.py"
    helpers = [
        n
        for n in ast.parse(loader.read_text(encoding = "utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_folder_quant_method"
    ]
    exec(compile(ast.Module(helpers, []), str(loader), "exec"), namespace)
    return namespace["_unsloth_save_lora_gguf"]


# Stands in for llama.cpp's converter, which indexes every tensor in --base: Inkling2 rejects bnb's packed 4-bit
# weights there (uint8 / quant_state companions), and a LoRA needs only the config.
_CONVERTER_READS_BASE = """
import json, os, sys
base = sys.argv[sys.argv.index("--base") + 1]
names = sorted(os.listdir(base))
config = json.load(open(os.path.join(base, "config.json"), encoding = "utf-8"))
if any(n.endswith(".safetensors") for n in names) or "quantization_config" in config:
    print("ValueError: Inkling2: unexpected packed tensor or orphan companion model.layers.0.attn.attn_sink_proj.weight")
    sys.exit(1)
out = sys.argv[sys.argv.index("--outfile") + 1]
open(out, "w", encoding = "utf-8").write(json.dumps({"base": base, "names": names, "config": config}))
"""


def _local_checkpoint(tmp_path, name, quant):
    import json

    folder = tmp_path / name
    folder.mkdir()
    config = {"model_type": "inkling2_loop", "text_config": {"model_type": "inkling2_loop_text"}}
    if quant:
        config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    (folder / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (folder / "model.safetensors").write_bytes(b"packed")
    return str(folder)


def test_prequantized_local_base_gives_the_converter_only_its_config(tmp_path):
    import json

    base = _local_checkpoint(tmp_path, "inkling2_bnb", quant = True)
    save = _writer(tmp_path, _CONVERTER_READS_BASE, base = base)
    gguf = save(_Peft(), None, str(tmp_path / "export"), outtype = "q8_0")
    seen = json.loads(pathlib.Path(gguf).read_text(encoding = "utf-8"))
    assert seen["base"] != base and seen["names"] == ["config.json"]
    assert seen["config"] == {
        "model_type": "inkling2_loop",
        "text_config": {"model_type": "inkling2_loop_text"},
    }
    # The config-only folder is temporary; the checkpoint is untouched.
    assert not os.path.exists(seen["base"])
    assert sorted(os.listdir(base)) == ["config.json", "model.safetensors"]


def test_16bit_local_base_is_still_passed_as_is(tmp_path):
    import json

    base = _local_checkpoint(tmp_path, "inkling2_bf16", quant = False)
    os.remove(os.path.join(base, "model.safetensors"))
    save = _writer(tmp_path, _CONVERTER_READS_BASE, base = base)
    gguf = save(_Peft(), None, str(tmp_path / "export"), outtype = "q8_0")
    assert json.loads(pathlib.Path(gguf).read_text(encoding = "utf-8"))["base"] == base


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


_CONVERTER_KILLED_MIDWAY = """
import sys
out = sys.argv[sys.argv.index("--outfile") + 1]
open(out, "wb").write(b"GG")
sys.exit(-9)
"""


def test_a_failed_reexport_keeps_the_prior_lora_gguf(tmp_path):
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    prior = out_dir / "Inkling-2-Mini-lora-q8_0.gguf"
    prior.write_bytes(b"GGUF prior export")
    save = _writer(tmp_path, _CONVERTER_KILLED_MIDWAY)
    with pytest.raises(RuntimeError):
        save(_Peft(), None, str(out_dir), outtype = "q8_0")
    assert prior.read_bytes() == b"GGUF prior export"
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
    assert (
        line(["Traceback (most recent call last):", "NotImplementedError"]) == "NotImplementedError"
    )
    # Unsupported architectures and adapter layouts exit 1 through logger.error, with no traceback.
    logged = [
        "INFO:lora-to-gguf:Loading base model: base",
        "ERROR:lora-to-gguf:Model Foo is not supported",
    ]
    assert line(logged) == "Model Foo is not supported"
    assert line(logged + ["Traceback (most recent call last):", _UNMAPPED]) == _UNMAPPED
