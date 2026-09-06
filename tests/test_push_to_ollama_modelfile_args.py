# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""push_to_ollama has to call create_ollama_modelfile with the arguments it takes.

create_ollama_modelfile was `(tokenizer, gguf_location)` when push_to_ollama was written in
#1648. It is now `(tokenizer, base_model_name, model_location)`, and this caller was left
behind, so it raised `TypeError: unexpected keyword argument 'gguf_location'` before it
reached Ollama. save.py needs a GPU to import, so the function is ast-extracted the way
tests/test_ollama_eos_token_order.py extracts its own.

The stub signatures below are checked against save.py's own AST rather than trusted, so a
later rename of a callee parameter fails here instead of drifting past a hand-copied stub.
The last two tests run the real create_ollama_modelfile against the real template mapper
(both are stdlib-only), so a Modelfile is actually produced rather than only asked for.
"""

import ast
import importlib.util
import os

import pytest

_TESTS = os.path.dirname(__file__)
_SAVE = os.path.join(_TESTS, os.pardir, "unsloth", "save.py")
_MAPPERS = os.path.join(_TESTS, os.pardir, "unsloth", "ollama_template_mappers.py")

MAPPED = "unsloth/llama-3-8b-Instruct"
UNMAPPED = "some/unmapped-model"


def _parse():
    with open(_SAVE, encoding = "utf-8") as f:
        source = f.read()
    return source, ast.parse(source)


def _extract(name):
    source, tree = _parse()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in save.py")


def _params(name):
    """save.py's real parameter names for `name`, so a stub cannot drift away from it."""
    _, tree = _parse()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    raise AssertionError(f"{name} not found in save.py")


def _mappers():
    """ollama_template_mappers is stdlib-only, so it loads without the GPU import chain."""
    spec = importlib.util.spec_from_file_location("_ollama_mappers", _MAPPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(modelfile):
    """Exec push_to_ollama beside stubs that carry the REAL signatures of what it calls."""
    seen = {}

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
    for name, stub in namespace.items():
        assert _params(name) == list(stub.__code__.co_varnames[:stub.__code__.co_argcount]), (
            f"the {name} stub no longer matches save.py; update it and this test's assertions"
        )
    exec(compile(_extract("push_to_ollama"), "push_to_ollama", "exec"), namespace)
    return namespace["push_to_ollama"], seen


def _load_real():
    """Same, but with save.py's own create_ollama_modelfile and the real templates."""
    mappers = _mappers()
    calls = []
    namespace = {
        "OLLAMA_TEMPLATES": mappers.OLLAMA_TEMPLATES,
        "MODEL_TO_OLLAMA_TEMPLATE_MAPPER": mappers.MODEL_TO_OLLAMA_TEMPLATE_MAPPER,
        "create_ollama_model": lambda **kw: calls.append(("create", kw)),
        "push_to_ollama_hub": lambda **kw: calls.append(("push", kw)),
    }
    for name in ("create_ollama_modelfile", "push_to_ollama"):
        exec(compile(_extract(name), name, "exec"), namespace)
    return namespace["push_to_ollama"], calls, mappers


class _Tokenizer:
    eos_token = "<|eot_id|>"


def test_push_to_ollama_reaches_create_ollama_modelfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    push_to_ollama, seen = _load("FROM ./model.gguf\n")

    push_to_ollama(
        tokenizer = "TOKENIZER",
        base_model_name = MAPPED,
        gguf_location = "./model.gguf",
        username = "user",
        model_name = "my-model",
        tag = "latest",
    )

    assert seen["modelfile"] == ("TOKENIZER", MAPPED, "./model.gguf")
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
            base_model_name = UNMAPPED,
            gguf_location = "./model.gguf",
            username = "user",
            model_name = "my-model",
            tag = "latest",
        )

    assert not (tmp_path / "Modelfile_my-model").exists()


def test_push_to_ollama_writes_a_real_modelfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    push_to_ollama, calls, mappers = _load_real()
    assert mappers.MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(MAPPED), f"{MAPPED} left the mapper"

    push_to_ollama(
        tokenizer = _Tokenizer(),
        base_model_name = MAPPED,
        gguf_location = "./model.gguf",
        username = "user",
        model_name = "my-model",
        tag = "latest",
    )

    text = (tmp_path / "Modelfile_my-model").read_text(encoding = "utf-8")
    assert "FROM ./model.gguf" in text
    assert "__FILE_LOCATION__" not in text and "__EOS_TOKEN__" not in text
    assert [c[0] for c in calls] == ["create", "push"]
    assert calls[0][1]["modelfile_path"] == "Modelfile_my-model"


def test_push_to_ollama_rejects_an_unmapped_model_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    push_to_ollama, calls, mappers = _load_real()
    assert mappers.MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(UNMAPPED) is None

    with pytest.raises(RuntimeError, match = "No Ollama template mapping"):
        push_to_ollama(
            tokenizer = _Tokenizer(),
            base_model_name = UNMAPPED,
            gguf_location = "./model.gguf",
            username = "user",
            model_name = "my-model",
            tag = "latest",
        )

    assert not (tmp_path / "Modelfile_my-model").exists()
    assert calls == []
