import ast
import json
import os
import types
from pathlib import Path


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _load_preserve_helper():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_preserve_tokenizer_eos_token"
    )
    module = ast.Module(body = [helper], type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {"json": json, "os": os, "logger": _Logger()}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_preserve_tokenizer_eos_token"]


def test_preserve_tokenizer_eos_token_restores_gemma4_turn_token(tmp_path):
    preserve = _load_preserve_helper()
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"eos_token": "<eos>", "other": True}),
        encoding = "utf-8",
    )
    tokenizer = types.SimpleNamespace(eos_token = "<turn|>")

    preserve(tokenizer, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["eos_token"] == "<turn|>"
    assert saved_config["other"] is True


def test_preserve_tokenizer_eos_token_supports_processor_tokenizer(tmp_path):
    preserve = _load_preserve_helper()
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"eos_token": "<eos>"}), encoding = "utf-8")
    processor = types.SimpleNamespace(
        tokenizer = types.SimpleNamespace(eos_token = "<turn|>")
    )

    preserve(processor, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["eos_token"] == "<turn|>"


class _StringableToken:
    def __str__(self):
        return "<turn|>"


def test_preserve_tokenizer_eos_token_serializes_stringable_tokens(tmp_path):
    preserve = _load_preserve_helper()
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"eos_token": "<eos>"}), encoding = "utf-8")
    tokenizer = types.SimpleNamespace(eos_token = _StringableToken())

    preserve(tokenizer, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["eos_token"] == "<turn|>"


def test_preserve_tokenizer_eos_token_supports_filename_prefix(tmp_path):
    preserve = _load_preserve_helper()
    prefixed_config = tmp_path / "adapter-tokenizer_config.json"
    prefixed_config.write_text(
        json.dumps({"eos_token": "<eos>", "other": True}),
        encoding = "utf-8",
    )
    tokenizer = types.SimpleNamespace(eos_token = "<turn|>")

    preserve(tokenizer, tmp_path, filename_prefix = "adapter")

    saved_config = json.loads(prefixed_config.read_text(encoding = "utf-8"))
    assert saved_config["eos_token"] == "<turn|>"
    assert saved_config["other"] is True
    # Unprefixed file must not be created as a side effect.
    assert not (tmp_path / "tokenizer_config.json").exists()


def test_preserve_tokenizer_eos_token_filename_prefix_none_uses_default(tmp_path):
    preserve = _load_preserve_helper()
    default_config = tmp_path / "tokenizer_config.json"
    default_config.write_text(
        json.dumps({"eos_token": "<eos>"}),
        encoding = "utf-8",
    )
    tokenizer = types.SimpleNamespace(eos_token = "<turn|>")

    preserve(tokenizer, tmp_path, filename_prefix = None)

    saved_config = json.loads(default_config.read_text(encoding = "utf-8"))
    assert saved_config["eos_token"] == "<turn|>"
