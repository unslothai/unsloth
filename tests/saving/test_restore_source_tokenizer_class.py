import ast
import json
import os
import types
from pathlib import Path


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _load_restore_helper(hub_path = None):
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restore_source_tokenizer_class"
    )
    module = ast.Module(body = [helper], type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "os": os,
        "logger": _Logger(),
        "_resolve_hub_repo_cached_file": lambda *args, **kwargs: hub_path,
        "_tokenizer_cache_dir": lambda tokenizer: None,
        "_tokenizer_revision": lambda tokenizer: None,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_restore_source_tokenizer_class"]


def _write(path, config):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(config), encoding = "utf-8")


def _read(path):
    return json.loads(path.read_text(encoding = "utf-8"))


def test_restores_class_from_local_source(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(
        tmp_path / "out" / "tokenizer_config.json",
        {"tokenizer_class": "TokenizersBackend", "other": 1},
    )
    tokenizer = types.SimpleNamespace(name_or_path = str(tmp_path / "base"))

    restore(tokenizer, tmp_path / "out")

    saved = _read(tmp_path / "out" / "tokenizer_config.json")
    assert saved["tokenizer_class"] == "Qwen2Tokenizer"
    assert saved["other"] == 1


def test_restores_class_from_cached_hub_source_through_processor(tmp_path):
    hub_config = tmp_path / "hub" / "tokenizer_config.json"
    _write(hub_config, {"tokenizer_class": "Qwen2Tokenizer"})
    restore = _load_restore_helper(hub_path = str(hub_config))
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    processor = types.SimpleNamespace(
        tokenizer = types.SimpleNamespace(name_or_path = "unsloth/Qwen3.5-0.8B")
    )

    restore(processor, tmp_path / "out")

    assert _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "Qwen2Tokenizer"


def test_leaves_specific_classes_alone(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "Qwen3_5Tokenizer"})
    tokenizer = types.SimpleNamespace(name_or_path = str(tmp_path / "base"))

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "Qwen3_5Tokenizer"
    )


def test_keeps_tokenizers_backend_when_source_has_no_specific_class(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    tokenizer = types.SimpleNamespace(name_or_path = str(tmp_path / "base"))

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "TokenizersBackend"
    )


def test_uncached_hub_source_is_a_no_op(tmp_path):
    restore = _load_restore_helper(hub_path = None)
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    tokenizer = types.SimpleNamespace(name_or_path = "someone/not-cached")

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "TokenizersBackend"
    )


def test_filename_prefix(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(
        tmp_path / "out" / "adapter-tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"}
    )
    tokenizer = types.SimpleNamespace(name_or_path = str(tmp_path / "base"))

    restore(tokenizer, tmp_path / "out", filename_prefix = "adapter")

    assert (
        _read(tmp_path / "out" / "adapter-tokenizer_config.json")["tokenizer_class"]
        == "Qwen2Tokenizer"
    )
