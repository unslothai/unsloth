# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, Qwen2TokenizerFast


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _load_restore_helper(hub_path = None):
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {
        "_capture_source_tokenizer_class",
        "_source_tokenizer_class",
        "_restore_source_tokenizer_class",
        "_unsloth_tokenizer_save_pretrained",
        "_without_runtime_truncation",
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
        "contextmanager": contextmanager,
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
    restore.capture = namespace["_capture_source_tokenizer_class"]
    return restore


def _write(path, config):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(config), encoding = "utf-8")


def _read(path):
    return json.loads(path.read_text(encoding = "utf-8"))


def _real_source(path, source_class = "Qwen2Tokenizer"):
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object = Tokenizer(models.BPE({"[UNK]": 0, "a": 1}, [], unk_token = "[UNK]")),
        unk_token = "[UNK]",
    )
    tokenizer.save_pretrained(path)
    tokenizer._tokenizer.model.save(str(path))
    config = _read(path / "tokenizer_config.json")
    config["tokenizer_class"] = source_class
    _write(path / "tokenizer_config.json", config)
    loaded = Qwen2TokenizerFast.from_pretrained(path, local_files_only = True)
    _load_restore_helper().capture(loaded, source = str(path))
    return loaded


def test_restores_class_from_local_source(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(
        tmp_path / "out" / "tokenizer_config.json",
        {"tokenizer_class": "TokenizersBackend", "other": 1},
    )
    tokenizer = _real_source(tmp_path / "base")

    restore(tokenizer, tmp_path / "out")

    saved = _read(tmp_path / "out" / "tokenizer_config.json")
    assert saved["tokenizer_class"] == "Qwen2Tokenizer"
    assert saved["other"] == 1


def test_restores_class_from_cached_hub_source_through_processor(tmp_path):
    hub_config = tmp_path / "hub" / "tokenizer_config.json"
    _write(hub_config, {"tokenizer_class": "Qwen2Tokenizer"})
    restore = _load_restore_helper(hub_path = str(hub_config))
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    processor = types.SimpleNamespace(tokenizer = _real_source(hub_config.parent))

    restore(processor, tmp_path / "out")

    assert _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "Qwen2Tokenizer"


def test_leaves_specific_classes_alone(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "Qwen3_5Tokenizer"})
    tokenizer = _real_source(tmp_path / "base")

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"] == "Qwen3_5Tokenizer"
    )


def test_generic_source_uses_portable_backend_alias(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    tokenizer = _real_source(tmp_path / "base", source_class = "TokenizersBackend")

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"]
        == "PreTrainedTokenizerFast"
    )


def test_unknown_source_uses_portable_backend_alias(tmp_path):
    restore = _load_restore_helper(hub_path = None)
    _write(tmp_path / "out" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    tokenizer = types.SimpleNamespace(name_or_path = "someone/not-cached")

    restore(tokenizer, tmp_path / "out")

    assert (
        _read(tmp_path / "out" / "tokenizer_config.json")["tokenizer_class"]
        == "PreTrainedTokenizerFast"
    )


def test_filename_prefix(tmp_path):
    restore = _load_restore_helper()
    _write(tmp_path / "base" / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(
        tmp_path / "out" / "adapter-tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"}
    )
    tokenizer = _real_source(tmp_path / "base")

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

    tokenizer = _real_source(checkpoint)
    tokenizer.original_save_pretrained = overwrite_with_generic_class

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
        "save.py": {
            "_capture_source_tokenizer_class",
            "_source_tokenizer_class",
            "_restore_source_tokenizer_class",
        },
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


@pytest.mark.parametrize("cache_kind", ["explicit", "colab", "kaggle"])
def test_online_custom_cache_restores_source_class(tmp_path, monkeypatch, cache_kind):
    monkeypatch.chdir(tmp_path)
    helpers = _load_cache_helpers()
    cache = {
        "explicit": tmp_path / "custom",
        "colab": Path("huggingface_tokenizers_cache"),
        "kaggle": tmp_path / "kaggle" / "custom",
    }[cache_kind]
    repo = "synthetic-review/tokenizer-class-cache"
    snapshot = cache / "models--synthetic-review--tokenizer-class-cache" / "snapshots" / ("1" * 40)
    _real_source(snapshot)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir()
    (refs / "main").write_text("1" * 40, encoding = "utf-8")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        repo,
        cache_dir = cache,
        local_files_only = True,
    )
    assert not hasattr(tokenizer, "_unsloth_source_tokenizer_class")
    if cache_kind != "explicit":
        helpers.update(
            IS_COLAB_ENVIRONMENT = cache_kind == "colab",
            IS_KAGGLE_ENVIRONMENT = cache_kind == "kaggle",
            KAGGLE_TMP = str(tmp_path / "kaggle"),
            AutoTokenizer = types.SimpleNamespace(from_pretrained = lambda *args, **kwargs: tokenizer),
            _repair_tokenizer_backend_from_json = lambda value, **kwargs: value,
            _fix_llava_num_additional_image_tokens = lambda value, **kwargs: value,
        )
        load = helpers["_offline_aware_load"](
            lambda **kwargs: helpers["_load_correct_tokenizer"](repo, fix_tokenizer = False, **kwargs)
        )
        loaded = load(cache_dir = "custom") if cache_kind == "kaggle" else load()
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


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("subfolder", ["", "nested"])
def test_managed_tokenizer_json_source_survives_source_changes(tmp_path, cached, subfolder):
    helpers = _load_cache_helpers()
    cache = tmp_path / "cache"
    repo = "synthetic-review/source-provenance"
    snapshot = cache / "models--synthetic-review--source-provenance" / "snapshots" / ("1" * 40)
    root = snapshot if cached else tmp_path / "source"
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token = "[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "[UNK]")
    tokenizer.save_pretrained(root / subfolder)
    config_path = root / subfolder / "tokenizer_config.json"
    config = _read(config_path)
    config["tokenizer_class"] = "Qwen2Tokenizer"
    _write(config_path, config)
    if subfolder:
        _write(root / "tokenizer_config.json", {"tokenizer_class": "LlamaTokenizer"})
    if cached:
        refs = snapshot.parents[1] / "refs"
        refs.mkdir()
        (refs / "main").write_text("1" * 40, encoding = "utf-8")
    source = repo if cached else str(root)
    load_kwargs = dict(cache_dir = cache, local_files_only = True, subfolder = subfolder)
    load = lambda **kwargs: PreTrainedTokenizerFast.from_pretrained(source, **kwargs)
    external = load(**load_kwargs)
    assert helpers["_source_tokenizer_class"](external) is None
    loaded = helpers["_offline_aware_load"](load)(**load_kwargs)
    assert helpers["_source_tokenizer_class"](loaded) == "Qwen2Tokenizer"
    if cached:
        replacement = snapshot.parent / ("2" * 40)
        _write(
            replacement / subfolder / "tokenizer_config.json", {"tokenizer_class": "BertTokenizer"}
        )
        (refs / "main").write_text("2" * 40, encoding = "utf-8")
    config_path.unlink()
    assert helpers["_source_tokenizer_class"](loaded) == "Qwen2Tokenizer"


@pytest.mark.parametrize("known_source", [False, True])
def test_inner_tokenizer_context_wins_over_model_only_subfolder(tmp_path, known_source):
    helpers = _load_cache_helpers()
    helpers.update(
        _repair_tokenizer_backend_from_json = lambda value, **kwargs: value,
        _fix_llava_num_additional_image_tokens = lambda value, **kwargs: value,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token = "[UNK]")),
        unk_token = "[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    config = _read(tmp_path / "tokenizer_config.json")
    config["tokenizer_class"] = "Qwen2Tokenizer" if known_source else None
    _write(tmp_path / "tokenizer_config.json", config)
    _write(tmp_path / "nested" / "tokenizer_config.json", {"tokenizer_class": "BertTokenizer"})

    def load(**kwargs):
        loaded = PreTrainedTokenizerFast.from_pretrained(tmp_path, local_files_only = True)
        return helpers["_apply_post_load_tokenizer_fixes"](loaded, fix_tokenizer = False)

    loaded = helpers["_offline_aware_load"](load)(subfolder = "nested")
    assert helpers["_source_tokenizer_class"](loaded) == (
        "Qwen2Tokenizer" if known_source else None
    )


def test_resolved_vocab_directory_wins_over_outer_source_context(tmp_path):
    helpers = _load_cache_helpers()
    _real_source(tmp_path / "nested")
    _write(tmp_path / "tokenizer_config.json", {"tokenizer_class": "BertTokenizer"})
    loaded = Qwen2TokenizerFast.from_pretrained(tmp_path, subfolder = "nested", local_files_only = True)
    assert Path(loaded.init_kwargs["vocab_file"]).parent == tmp_path / "nested"
    helpers["_mark_loaded_revision"](loaded, None, load_kwargs = {})
    assert helpers["_source_tokenizer_class"](loaded) == "Qwen2Tokenizer"
