# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers
from packaging.version import Version
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, Qwen2TokenizerFast


SOURCE = Path(__file__).parents[2] / "unsloth/save.py"
EXPORTS = (
    "_unsloth_save_compressed_tensors",
    "_unsloth_save_torchao",
    "_unsloth_save_openvino",
)


def _tree():
    return ast.parse(SOURCE.read_text(encoding = "utf-8"))


def _calls(node, name):
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == name
        for child in ast.walk(node)
    )


def _execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = []))
    exec(compile(module, str(SOURCE), "exec"), namespace)


def _namespace(tokenizer, output):
    namespace = {
        "os": os,
        "json": json,
        "tokenizer": tokenizer,
        "token": None,
        "out_dir": str(output),
        "final_dir": str(output),
        "source_tokenizer_class": None,
        "logger": SimpleNamespace(warning_once = lambda *a, **k: None),
    }
    helpers = {
        "_capture_source_tokenizer_class",
        "_source_tokenizer_class",
        "_restore_source_tokenizer_class",
        "_preserve_repaired_tokenizer_class",
    }
    _execute(
        [
            node
            for node in _tree().body
            if isinstance(node, ast.FunctionDef) and node.name in helpers
        ],
        namespace,
    )
    capture = namespace.get("_capture_source_tokenizer_class")
    if capture is not None and tokenizer is not None:
        capture(tokenizer, source = tokenizer.name_or_path)
    return namespace


def _capture_and_finalize(name, namespace, action):
    fn = next(
        node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    capture = [
        node
        for node in fn.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "source_tokenizer_class"
            for target in node.targets
        )
    ]
    _execute(capture, namespace)
    action()
    finalizers = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Expr) and _calls(child, "_restore_source_tokenizer_class")
            for child in node.body
        )
        and any(
            isinstance(child, ast.Expr) and _calls(child, "_preserve_repaired_tokenizer_class")
            for child in node.body
        )
    ]
    _execute(finalizers, namespace)


def _tokenizer(source):
    source.mkdir(parents = True, exist_ok = True)
    Tokenizer(models.BPE({"a": 0, "b": 1}, [])).model.save(str(source))
    tokenizer = Qwen2TokenizerFast(
        vocab_file = str(source / "vocab.json"), merges_file = str(source / "merges.txt")
    )
    tokenizer.save_pretrained(source)
    return Qwen2TokenizerFast.from_pretrained(source)


@pytest.mark.parametrize("export", EXPORTS)
@pytest.mark.parametrize("repaired", [False, True])
def test_final_export_repairs_secondary_tokenizer_save(tmp_path, export, repaired):
    tokenizer = _tokenizer(tmp_path / "source")
    tokenizer._unsloth_tokenizer_json_repaired = repaired
    output = tmp_path / "final"
    namespace = _namespace(tokenizer, output)

    def secondary_save():
        tokenizer.save_pretrained(output)
        config_path = output / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding = "utf-8"))
        config["tokenizer_class"] = "TokenizersBackend"
        config["exporter_metadata"] = "retained"
        config_path.write_text(json.dumps(config), encoding = "utf-8")

    _capture_and_finalize(export, namespace, secondary_save)

    config = json.loads((output / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["tokenizer_class"] == (
        "PreTrainedTokenizerFast" if repaired else "Qwen2Tokenizer"
    )
    assert config["exporter_metadata"] == "retained"


@pytest.mark.parametrize("export", EXPORTS)
def test_final_export_without_tokenizer_leaves_exporter_metadata(tmp_path, export):
    output = tmp_path / "final"
    namespace = _namespace(None, output)

    def secondary_save():
        output.mkdir()
        (output / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"TokenizersBackend"}', encoding = "utf-8"
        )

    _capture_and_finalize(export, namespace, secondary_save)
    assert (
        json.loads((output / "tokenizer_config.json").read_text(encoding = "utf-8"))[
            "tokenizer_class"
        ]
        == "TokenizersBackend"
    )


def test_merged_save_captures_source_before_cache_purge(tmp_path):
    tokenizer = _tokenizer(tmp_path / "source")
    output = tmp_path / "final"
    namespace = _namespace(tokenizer, output)
    fn = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "unsloth_save_model"
    )
    purge_index = next(i for i, node in enumerate(fn.body) if _calls(node, "_free_cached_model"))
    capture = [
        node
        for node in fn.body[:purge_index]
        if isinstance(node, ast.Assign) and _calls(node, "_source_tokenizer_class")
    ]
    _execute(capture, namespace)
    assert namespace["source_tokenizer_class"] == "Qwen2Tokenizer"
    (tmp_path / "source/tokenizer_config.json").unlink()
    tokenizer.save_pretrained(output)
    config_path = output / "tokenizer_config.json"
    config_path.write_text('{"tokenizer_class":"TokenizersBackend"}', encoding = "utf-8")
    namespace["tokenizer_save_settings"] = {"save_directory": str(output)}
    finalizers = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Expr) and _calls(node, "_restore_source_tokenizer_class")
    ]
    _execute(finalizers, namespace)
    assert (
        json.loads(config_path.read_text(encoding = "utf-8"))["tokenizer_class"] == "Qwen2Tokenizer"
    )


@pytest.mark.skipif(
    not Version("5.2") <= Version(transformers.__version__) <= Version("5.5.0"),
    reason = "generic Qwen3.5 serialization in Transformers 5.2-5.5",
)
@pytest.mark.parametrize("export", EXPORTS)
def test_real_secondary_qwen3_5_save_keeps_compatible_class(tmp_path, export):
    original = _tokenizer(tmp_path / "staging")
    (tmp_path / "staging/config.json").write_text('{"model_type":"qwen3_5"}', encoding = "utf-8")
    reloaded = AutoTokenizer.from_pretrained(tmp_path / "staging")
    output = tmp_path / "final"
    namespace = _namespace(original, output)
    assert type(reloaded).__name__ == "TokenizersBackend"

    _capture_and_finalize(export, namespace, lambda: reloaded.save_pretrained(output))

    config = json.loads((output / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["tokenizer_class"] == "Qwen2Tokenizer"
