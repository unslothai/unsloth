# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""push_to_hub_gguf(save_method="lora") with a bare repo name must upload to <user>/<name>."""

from __future__ import annotations

import ast
import builtins
import collections
import os
import shutil
import symtable
import sys
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"


class _FakePeft:
    pass


class _FakeApi:
    calls = []

    def __init__(self, token = None):
        self.token = token

    def whoami(self):
        _FakeApi.calls.append(("whoami",))
        return {"name": "someone"}

    def create_repo(self, repo_id, **kwargs):
        _FakeApi.calls.append(("create_repo", repo_id))

    def create_branch(self, repo_id, **kwargs):
        _FakeApi.calls.append(("create_branch", repo_id, kwargs.get("branch")))

    def upload_folder(self, repo_id, **kwargs):
        _FakeApi.calls.append(("upload_folder", repo_id))
        return None


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        # The converter writes the --outfile it is given.
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF")
        self.stdout = []
        self.returncode = 0
        self.args = cmd

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self):
        return 0


class _FakeSubprocess:
    PIPE = -1
    STDOUT = -2
    Popen = _FakePopen
    CalledProcessError = RuntimeError


def _load(tmp_path, refused):
    # Importing save.py needs the full ML stack, so lift the function out with ast.
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_save_lora_gguf"
    ]
    assert nodes, "unsloth/save.py does not define _unsloth_save_lora_gguf"
    nodes[0].decorator_list = []
    converter_dir = tmp_path / "llama.cpp"
    converter_dir.mkdir()
    (converter_dir / "convert_lora_to_gguf.py").write_text("", encoding = "utf-8")
    namespace = {
        "os": os,
        "sys": sys,
        "shutil": shutil,
        "subprocess": _FakeSubprocess,
        "PeftModelForCausalLM": _FakePeft,
        "PeftModel": _FakePeft,
        "_LORA_GGUF_OUTTYPES": ("f32", "f16", "bf16", "q8_0", "auto"),
        "_clean_save_token": lambda token: token,
        "get_token": lambda: "hf_token",
        "collections": collections,
        "_lora_target_parameters": lambda model: [],
        "is_inkling2_config": lambda config: False,
        "inkling2_lora_gguf_adapter": lambda model, lora_dir: lora_dir,
        "checkpoint_folder_quant_method": lambda path: None,
        "_local_base_config_dir": lambda path: path,
        "_converter_error_line": lambda tail: "",
        "_lora_gguf_conversion_error": lambda code, line, targets: RuntimeError(line),
        "_lora_base_model_id": lambda model: "unsloth/base",
        "get_model_name": lambda name, load_in_4bit = False: name,
        "_model_basename": lambda name: name.split("/")[-1],
        "save_lora_to_custom_dir": lambda model, tokenizer, directory: None,
        "install_llama_cpp": lambda **kwargs: None,
        "LLAMA_CPP_DEFAULT_DIR": str(converter_dir),
        "_loaded_via_remote_code": lambda model: False,
        "_base_config_dir": lambda base, token: None,
        "_apply_token_to_child_env": lambda env, token, explicit = False: None,
        "_refuse_private_push_to_public_repo": lambda repo_id, private, api = None: refused.append(
            repo_id
        ),
    }
    # Name any global the function reads that the namespace lacks, so a new helper in
    # save.py shows up here instead of as a NameError at its call site.
    module = ast.Module(nodes, [])
    table = symtable.symtable(ast.unparse(module), str(_SAVE_PY), "exec")
    (function,) = table.get_children()
    missing = sorted(
        symbol.get_name()
        for symbol in function.get_symbols()
        if symbol.is_global()
        and symbol.get_name() not in namespace
        and not hasattr(builtins, symbol.get_name())
    )
    assert not missing, f"_unsloth_save_lora_gguf reads {missing}; add them to the test namespace"
    exec(compile(module, str(_SAVE_PY), "exec"), namespace)
    return namespace["_unsloth_save_lora_gguf"]


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub

    _FakeApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    return _FakeApi.calls


def test_bare_repo_name_is_pushed_under_the_token_user(tmp_path, fake_hub):
    refused = []
    save = _load(tmp_path, refused)

    result = save(_FakePeft(), object(), "my-lora", push_to_hub = True, revision = "dev")

    assert result == "someone/my-lora"
    assert ("create_repo", "someone/my-lora") in fake_hub
    assert ("create_branch", "someone/my-lora", "dev") in fake_hub
    assert ("upload_folder", "someone/my-lora") in fake_hub
    # The private check must look at the repo that was created, not the bare name.
    assert refused == ["someone/my-lora"]


def test_full_repo_id_is_kept_and_needs_no_whoami(tmp_path, fake_hub):
    refused = []
    save = _load(tmp_path, refused)

    result = save(_FakePeft(), object(), "org/my-lora", push_to_hub = True)

    assert result == "org/my-lora"
    assert ("whoami",) not in fake_hub
    assert ("upload_folder", "org/my-lora") in fake_hub
    assert refused == ["org/my-lora"]
