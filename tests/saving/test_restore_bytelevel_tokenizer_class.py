# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import json
import os
import types
from pathlib import Path

import pytest
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _restore():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restore_source_tokenizer_class"
    )
    namespace = {
        "json": json,
        "os": os,
        "logger": types.SimpleNamespace(warning_once = lambda *args: None),
        "_source_tokenizer_class": lambda *args, **kwargs: None,
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(source), "exec"), namespace)
    return namespace[node.name]


def _backend(sequence = False):
    backend = Tokenizer(models.BPE(unk_token = "<unk>"))
    bytelevel = pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.pre_tokenizer = (
        pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), bytelevel])
        if sequence
        else bytelevel
    )
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Hello world, this is a test."] * 4,
        trainers.BpeTrainer(
            vocab_size = 300,
            special_tokens = ["<unk>"],
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return backend


@pytest.mark.parametrize("source_class", ["LlamaTokenizer", "LlamaTokenizerFast"])
@pytest.mark.parametrize("sequence", [False, True])
def test_bytelevel_llama_uses_portable_class_and_preserves_ids(tmp_path, source_class, sequence):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = _backend(sequence), unk_token = "<unk>")
    probe = "Hello world, this is a test."
    expected = tokenizer.encode(probe, add_special_tokens = False)
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config), encoding = "utf-8")

    _restore()(tokenizer, tmp_path, source_class = source_class)

    config = json.loads(config_path.read_text(encoding = "utf-8"))
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded.encode(probe, add_special_tokens = False) == expected
    assert reloaded.decode(expected) == tokenizer.decode(expected)


@pytest.mark.parametrize(
    "source_class,bytelevel", [("LlamaTokenizer", False), ("Qwen2Tokenizer", True)]
)
def test_valid_source_classes_remain_specific(tmp_path, source_class, bytelevel):
    backend = _backend()
    if not bytelevel:
        backend.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "<unk>")
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}), encoding = "utf-8")

    _restore()(tokenizer, tmp_path, source_class = source_class)

    assert json.loads(config_path.read_text(encoding = "utf-8"))["tokenizer_class"] == source_class


def test_unknown_source_uses_portable_class_and_preserves_ids(tmp_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = _backend(), unk_token = "<unk>")
    probe = "Hello world, this is a test."
    expected = tokenizer.encode(probe, add_special_tokens = False)
    tokenizer.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config), encoding = "utf-8")

    _restore()(tokenizer, tmp_path)

    assert (
        json.loads(config_path.read_text(encoding = "utf-8"))["tokenizer_class"]
        == "PreTrainedTokenizerFast"
    )
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded.encode(probe, add_special_tokens = False) == expected
    assert reloaded.decode(expected) == probe


@pytest.mark.skipif(
    transformers.__version__.split(".")[0] != "5", reason = "V5 generic backend reload"
)
def test_v5_source_llama_bytelevel_round_trip_preserves_spaces(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _backend().save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "LlamaTokenizerFast", "unk_token": "<unk>", "add_bos_token": False}
        ),
        encoding = "utf-8",
    )
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}), encoding = "utf-8")
    tokenizer = AutoTokenizer.from_pretrained(source)
    assert type(tokenizer).__name__ == "TokenizersBackend"
    assert not getattr(tokenizer, "_unsloth_tokenizer_json_repaired", False)
    probe = "Hello world, this is a test."
    expected = tokenizer.encode(probe, add_special_tokens = False)
    output = tmp_path / "output"
    tokenizer.save_pretrained(output)

    _restore()(tokenizer, output, source_class = "LlamaTokenizerFast")

    reloaded = AutoTokenizer.from_pretrained(output)
    assert reloaded.encode(probe, add_special_tokens = False) == expected
    assert reloaded.decode(expected) == probe
