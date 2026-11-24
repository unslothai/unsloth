# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastSentenceTransformer: a model's own sentence_bert_config.json must not choose from_pretrained
arguments that run its code (`weights_only: false` unpickled pytorch_model.bin with remote code off)."""

import ast
import importlib
import json
import logging
import os
import pathlib
import re

import pytest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "sentence_transformer.py"
)
WANTED = {
    "_ST_CONFIG_REFUSED_ARGS",
    "_ST_PLAIN_ATTN",
    "_st_config_path_escapes",
    "_scrub_st_config_load_args",
    "_install_st_config_load_args_guard",
}


def _helpers():
    """The guard, from source (the module needs the GPU package)."""
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in n.targets))
    ]
    assert {getattr(n, "name", None) or n.targets[0].id for n in body} == WANTED
    ns = {
        "re": re,
        "importlib": importlib,
        "logging": logging,
        "__name__": "unsloth.models.sentence_transformer",
    }
    exec(compile(ast.Module(body = body, type_ignores = []), str(SOURCE), "exec"), ns)
    return ns


def test_from_pretrained_installs_the_guard():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "FastSentenceTransformer"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    calls = {
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_install_st_config_load_args_guard" in calls


def test_scrub_drops_code_paths_and_keeps_plain_args():
    scrub = _helpers()["_scrub_st_config_load_args"]
    out = scrub(
        {
            "model_args": {
                "weights_only": False,
                "use_safetensors": False,
                "attn_implementation": "kernels-community/flash-attn",
                "allow_all_kernels": True,
                "dtype": "float32",
            },
            "config_args": {"attn_implementation": "sdpa"},
            "tokenizer_name_or_path": "../../etc",
        }
    )
    assert out == {
        "model_args": {"dtype": "float32"},
        "config_args": {"attn_implementation": "sdpa"},
    }


class _Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_repo_set_weights_only_false_does_not_unpickle(tmp_path):
    st = pytest.importorskip("sentence_transformers")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder = tmp_path / "attack"
    cfg = transformers.BertConfig(
        vocab_size = 32,
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 16,
        max_position_embeddings = 16,
    )
    transformers.BertModel(cfg).save_pretrained(str(folder), safe_serialization = False)
    marker = tmp_path / "pickle-ran"
    config = json.loads((folder / "config.json").read_text())
    config.pop("dtype", None)
    config.pop("torch_dtype", None)
    (folder / "config.json").write_text(json.dumps(config))
    torch.save({"payload": _Marker(str(marker))}, str(folder / "pytorch_model.bin"))
    (folder / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    (folder / "sentence_bert_config.json").write_text(
        json.dumps(
            {"model_args": {"weights_only": False, "use_safetensors": False, "dtype": "auto"}}
        )
    )
    (folder / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        )
    )
    _helpers()["_install_st_config_load_args_guard"]()
    try:
        st.SentenceTransformer(str(folder), device = "cpu", trust_remote_code = False)
    except Exception:
        pass
    assert not os.path.exists(marker), "the model's own config turned torch.load's weights_only off"
