# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""push_to_ollama has to call create_ollama_modelfile with the arguments it takes.

create_ollama_modelfile was `(tokenizer, gguf_location)` when push_to_ollama was written in
#1648. It is now `(tokenizer, base_model_name, model_location)`, and this caller was left
behind, so it raised `TypeError: unexpected keyword argument 'gguf_location'` before it
reached Ollama. save.py needs a GPU to import, so the function is ast-extracted the way
tests/test_ollama_eos_token_order.py extracts its own.
"""

import ast
import os

import pytest

_SAVE = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "save.py")


def _extract(name):
    with open(_SAVE, encoding = "utf-8") as f:
        source = f.read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in save.py")


def _load(modelfile):
    """Exec push_to_ollama beside stubs that carry the REAL signatures of what it calls."""
    seen = {}

    # signature copied verbatim from save.py's create_ollama_modelfile
    def create_ollama_modelfile(tokenizer, base_model_name, model_location):
        seen["modelfile"] = (tokenizer, base_model_name, model_location)
        return modelfile

    def create_ollama_model(username, model_name, tag, modelfile_path):
        seen["model"] = (username, model_name, tag, modelfile_path)

    def push_to_ollama_hub(username, model_name, tag):
        seen["hub"] = (username, model_name, tag)

    namespace = {
        "create_ollama_modelfile": create_ollama_modelfile,
        "create_ollama_model": create_ollama_model,
        "push_to_ollama_hub": push_to_ollama_hub,
    }
    exec(compile(_extract("push_to_ollama"), "push_to_ollama", "exec"), namespace)
    return namespace["push_to_ollama"], seen


def test_push_to_ollama_reaches_create_ollama_modelfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    push_to_ollama, seen = _load("FROM ./model.gguf\n")

    push_to_ollama(
        tokenizer = "TOKENIZER",
        base_model_name = "unsloth/llama-3-8b",
        gguf_location = "./model.gguf",
        username = "user",
        model_name = "my-model",
        tag = "latest",
    )

    assert seen["modelfile"] == ("TOKENIZER", "unsloth/llama-3-8b", "./model.gguf")
    assert seen["model"] == ("user", "my-model", "latest", "Modelfile_my-model")
    assert seen["hub"] == ("user", "my-model", "latest")
    assert (tmp_path / "Modelfile_my-model").read_text(encoding = "utf-8") == "FROM ./model.gguf\n"


def test_push_to_ollama_reports_a_missing_template(tmp_path, monkeypatch):
    # create_ollama_modelfile returns None when the model has no Ollama template mapping;
    # writing that to the Modelfile used to fail with an opaque TypeError from f.write.
    monkeypatch.chdir(tmp_path)
    push_to_ollama, _ = _load(None)

    with pytest.raises(RuntimeError, match = "No Ollama template mapping"):
        push_to_ollama(
            tokenizer = "TOKENIZER",
            base_model_name = "some/unmapped-model",
            gguf_location = "./model.gguf",
            username = "user",
            model_name = "my-model",
            tag = "latest",
        )

    assert not (tmp_path / "Modelfile_my-model").exists()
