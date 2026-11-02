# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import json
import os
import types
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _load_restore_helper(hub_path = None):
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {
        "_source_tokenizer_class",
        "_restore_source_tokenizer_class",
        "_unsloth_tokenizer_save_pretrained",
    }
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body = nodes, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "os": os,
        "logger": _Logger(),
        "_resolve_hub_repo_cached_file": lambda *args, **kwargs: hub_path,
        "_tokenizer_cache_dir": lambda tokenizer: None,
        "_tokenizer_revision": lambda tokenizer: None,
        "_preserve_sentencepiece_tokenizer_assets": lambda *args, **kwargs: None,
        "_preserve_tokenizer_eos_token": lambda *args, **kwargs: None,
        "_preserve_repaired_tokenizer_class": lambda *args, **kwargs: None,
    }
    exec(compile(module, str(source), "exec"), namespace)
    restore = namespace["_restore_source_tokenizer_class"]
    restore.save = namespace["_unsloth_tokenizer_save_pretrained"]
    return restore


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


def test_in_place_save_keeps_the_source_class(tmp_path):
    restore = _load_restore_helper()
    checkpoint = tmp_path / "checkpoint"
    _write(checkpoint / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})

    def overwrite_with_generic_class(path, **kwargs):
        _write(Path(path) / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})

    tokenizer = types.SimpleNamespace(
        name_or_path = str(checkpoint), original_save_pretrained = overwrite_with_generic_class
    )

    restore.save(tokenizer, str(checkpoint))

    assert _read(checkpoint / "tokenizer_config.json")["tokenizer_class"] == "Qwen2Tokenizer"


def _load_cache_helpers():
    root = Path(__file__).parents[2] / "unsloth"
    namespace = {
        "os": os,
        "json": json,
        "functools": __import__("functools"),
        "_LOADED_CACHE_DIR_ATTR": "_unsloth_loaded_cache_dir",
        "_LOADED_REVISION_ATTR": "_unsloth_loaded_revision",
        "_get_effective_local_files_only": lambda kwargs: False,
        "_progress_bars_were_disabled": lambda: False,
    }
    wanted = {
        "models/loader_utils.py": {
            "_mark_loaded_revision",
            "_tokenizer_cache_dir",
            "_tokenizer_revision",
            "_resolve_hub_repo_cached_file",
            "_resolve_hub_repo_local_dir",
            "_offline_aware_load",
        },
        "tokenizer_utils.py": {"_apply_post_load_tokenizer_fixes", "_load_correct_tokenizer"},
        "save.py": {"_source_tokenizer_class", "_restore_source_tokenizer_class"},
    }
    for filename, names in wanted.items():
        source = root / filename
        tree = ast.parse(source.read_text(encoding = "utf-8"))
        nodes = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        for node in nodes:
            node.body = [
                child
                for child in node.body
                if not (isinstance(child, ast.ImportFrom) and child.level)
            ]
        module = ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = []))
        exec(compile(module, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("notebook_cache", [False, True])
def test_online_custom_cache_restores_source_class(tmp_path, monkeypatch, notebook_cache):
    monkeypatch.chdir(tmp_path)
    helpers = _load_cache_helpers()
    cache = Path("huggingface_tokenizers_cache") if notebook_cache else tmp_path / "custom"
    repo = "synthetic-review/tokenizer-class-cache"
    snapshot = cache / "models--synthetic-review--tokenizer-class-cache" / "snapshots" / ("1" * 40)
    _write(snapshot / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    refs = snapshot.parents[1] / "refs"
    refs.mkdir()
    (refs / "main").write_text("1" * 40, encoding = "utf-8")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token = "[UNK]")),
        unk_token = "[UNK]",
        name_or_path = repo,
    )
    assert helpers["_source_tokenizer_class"](tokenizer) is None
    if notebook_cache:
        helpers.update(
            IS_COLAB_ENVIRONMENT = True,
            IS_KAGGLE_ENVIRONMENT = False,
            AutoTokenizer = types.SimpleNamespace(from_pretrained = lambda *args, **kwargs: tokenizer),
            _repair_tokenizer_backend_from_json = lambda value, **kwargs: value,
            _fix_llava_num_additional_image_tokens = lambda value, **kwargs: value,
        )
        loaded = helpers["_load_correct_tokenizer"](repo, fix_tokenizer = False)
    else:
        load = helpers["_offline_aware_load"](lambda **kwargs: tokenizer)
        loaded = load(cache_dir = cache)
    assert loaded is tokenizer
    assert not getattr(loaded, "_unsloth_local_files_only", False)
    assert helpers["_tokenizer_cache_dir"](loaded) == str(cache)
    assert helpers["_source_tokenizer_class"](loaded) == "Qwen2Tokenizer"
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    helpers["_restore_source_tokenizer_class"](loaded, tmp_path / "out")
    assert _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "Qwen2Tokenizer"
