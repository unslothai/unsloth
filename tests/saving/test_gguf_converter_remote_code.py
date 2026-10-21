# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The llama.cpp GGUF converter follows the load's trust_remote_code decision.

The converter loads several architectures' tokenizers with trust_remote_code=True. A model that
loaded with built-in classes must not have a tokenizer ``auto_map`` module run during export; a
model whose load already ran remote code (and so passed consent) converts as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

import unsloth.save as save_mod
from unsloth.models import decision_gguf

_KEY = "HF_MODULES_CACHE"

# What several conversion/*.py set_vocab implementations do.
_CONVERTER = (
    "import sys; from transformers import AutoTokenizer; "
    "AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code = True)"
)


def _checkpoint_with_tokenizer_code(
    folder,
    marker,
    list_form = False,
):
    """Loads with built-in classes, but declares a tokenizer auto_map that writes `marker`."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token = "<unk>")),
        unk_token = "<unk>",
    ).save_pretrained(str(folder))
    (folder / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    path = folder / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding = "utf-8"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    entry = [None, "tokenization_custom.CustomTokenizer"]
    config["auto_map"] = entry if list_form else {"AutoTokenizer": entry}
    path.write_text(json.dumps(config), encoding = "utf-8")
    (folder / "tokenization_custom.py").write_text(
        "from transformers import PreTrainedTokenizerFast\n"
        f"open({str(marker)!r}, 'w').close()\n"
        "class CustomTokenizer(PreTrainedTokenizerFast):\n"
        "    pass\n",
        encoding = "utf-8",
    )


def _child(folder, tmp_path):
    return subprocess.run(
        [sys.executable, "-c", _CONVERTER, str(folder)],
        env = dict(os.environ),
        capture_output = True,
        text = True,
        timeout = 300,
    )


@pytest.mark.parametrize("approved", [False, True])
def test_converter_child_runs_auto_map_code_only_after_a_remote_code_load(
    monkeypatch, tmp_path, approved
):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    marker = tmp_path / "code_ran"
    _checkpoint_with_tokenizer_code(folder, marker)
    monkeypatch.setenv(_KEY, str(tmp_path / "modules"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with save_mod._gguf_converter_remote_code(approved):
        child = _child(folder, tmp_path)

    assert marker.exists() is approved, child.stderr
    assert (child.returncode == 0) is approved, child.stderr
    if not approved:
        assert save_mod._gguf_converter_wanted_remote_code(RuntimeError(child.stderr))
    assert os.environ[_KEY] == str(tmp_path / "modules")


def test_a_folder_without_auto_map_converts_identically(monkeypatch, tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    _checkpoint_with_tokenizer_code(folder, tmp_path / "unused")
    path = folder / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding = "utf-8"))
    config.pop("auto_map")
    path.write_text(json.dumps(config), encoding = "utf-8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with save_mod._gguf_converter_remote_code(False):
        child = _child(folder, tmp_path)
    assert child.returncode == 0, child.stderr


def _save_to_gguf(monkeypatch, tmp_path, convert, **kwargs):
    import contextlib

    monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
    monkeypatch.setattr(
        save_mod,
        "_download_convert_hf_to_gguf",
        lambda: (str(tmp_path / "convert.py"), {"LlamaForCausalLM"}, set()),
    )
    monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
    monkeypatch.setattr(save_mod, "convert_to_gguf", convert)
    (tmp_path / "model_dir").mkdir()
    return save_mod.save_to_gguf(
        model_name = "testmodel",
        model_type = "llama",
        model_dtype = "float16",
        model_directory = str(tmp_path / "model_dir"),
        quantization_method = ["f16"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs, refused",
    [({}, True), ({"trust_remote_code": False}, True), ({"trust_remote_code": True}, False)],
)
def test_save_to_gguf_scopes_the_cache_to_the_conversion(monkeypatch, tmp_path, kwargs, refused):
    monkeypatch.setenv(_KEY, "/mine")
    seen = []

    def _convert(**kw):
        seen.append(os.environ.get(_KEY))
        out = tmp_path / "testmodel.F16.gguf"
        out.write_bytes(b"GGUF")
        return [str(out)], False

    _save_to_gguf(monkeypatch, tmp_path, _convert, **kwargs)
    assert len(seen) == 1
    if refused:
        assert seen[0] != "/mine"
        with pytest.raises(OSError):
            os.makedirs(seen[0], exist_ok = True)
    else:
        assert seen == ["/mine"]
    assert os.environ[_KEY] == "/mine"


def test_a_failed_conversion_restores_the_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(_KEY, raising = False)

    def _convert(**kw):
        raise RuntimeError("converter failed")

    with pytest.raises(RuntimeError, match = "converter failed"):
        _save_to_gguf(monkeypatch, tmp_path, _convert)
    assert _KEY not in os.environ


def _instance(module_name, **attrs):
    cls = type(
        "Fake",
        (),
        {
            "chat_template": None,
            "save_pretrained": lambda self, path, *a, **k: os.makedirs(path, exist_ok = True),
        },
    )
    cls.__module__ = module_name
    instance = cls()
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


def _export(monkeypatch, tmp_path, model_module, tokenizer_module, save_to_gguf):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = types.SimpleNamespace(
        _name_or_path = str(checkpoint),
        architectures = ["LlamaForCausalLM"],
        model_type = "llama",
    )
    monkeypatch.setattr(save_mod, "_is_vlm", lambda _m: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _m: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _t: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_k: None)
    monkeypatch.setattr(save_mod, "dtype_from_config", lambda _c: save_mod.torch.float16)
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_k: None)
    monkeypatch.setattr(save_mod, "save_to_gguf", save_to_gguf)
    return save_mod.unsloth_save_pretrained_gguf(
        _instance(model_module, config = config),
        str(tmp_path / "export" / "model"),
        tokenizer = _instance(tokenizer_module),
        quantization_method = "q8_0",
    )


@pytest.mark.parametrize(
    "model_module, tokenizer_module, expected",
    [
        ("transformers.models.llama.modeling_llama", "transformers.tokenization_utils_fast", False),
        ("transformers.models.llama.modeling_llama", "transformers_modules.x.tokenization_x", True),
        ("transformers_modules.x.modeling_x", "transformers.tokenization_utils_fast", True),
    ],
)
def test_the_export_passes_on_the_load_decision(
    monkeypatch, tmp_path, model_module, tokenizer_module, expected
):
    seen = {}

    def _save_to_gguf(**kwargs):
        seen.update(kwargs)
        out = tmp_path / "export" / "model_gguf" / "model.Q8_0.gguf"
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_bytes(b"GGUF")
        return [str(out)], True, False

    _export(monkeypatch, tmp_path, model_module, tokenizer_module, _save_to_gguf)
    assert seen["trust_remote_code"] is expected


def test_the_refusal_names_trust_remote_code(monkeypatch, tmp_path):
    def _save_to_gguf(**kwargs):
        raise RuntimeError(
            'File ".../transformers/dynamic_module_utils.py", line 94, in init_hf_modules\n'
            "NotADirectoryError: [Errno 20] Not a directory"
        )

    with pytest.raises(RuntimeError, match = "trust_remote_code = True"):
        _export(
            monkeypatch,
            tmp_path,
            "transformers.models.llama.modeling_llama",
            "transformers.tokenization_utils_fast",
            _save_to_gguf,
        )


def test_the_decision_converter_gets_no_usable_modules_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(_KEY, str(tmp_path / "modules"))
    outfile = tmp_path / "model.gguf"
    seen = {}

    def _run(command, what, print_output, env):
        seen.update(env)
        outfile.write_bytes(b"GGUF")

    monkeypatch.setattr(decision_gguf, "_run", _run)
    decision_gguf._convert(tmp_path, tmp_path / "checkpoint", "f16", outfile, False, False)
    assert seen[_KEY] != str(tmp_path / "modules")
    with pytest.raises(OSError):
        os.makedirs(seen[_KEY], exist_ok = True)
    assert os.environ[_KEY] == str(tmp_path / "modules")


@pytest.mark.parametrize("list_form", [False, True])
def test_a_built_in_load_never_runs_tokenizer_code_in_the_converter(
    monkeypatch, tmp_path, list_form
):
    """End to end through save_to_gguf: the stub converter child inherits os.environ, as
    unsloth_zoo's convert_to_gguf does, and loads the tokenizer with trust_remote_code=True."""
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    marker = tmp_path / "code_ran"
    _checkpoint_with_tokenizer_code(folder, marker, list_form = list_form)
    monkeypatch.setenv(_KEY, str(tmp_path / "modules"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    results = []

    def _convert(**kw):
        results.append(_child(folder, tmp_path))
        out = tmp_path / "testmodel.F16.gguf"
        out.write_bytes(b"GGUF")
        return [str(out)], False

    _save_to_gguf(monkeypatch, tmp_path, _convert)
    assert results and not marker.exists(), results[0].stderr
