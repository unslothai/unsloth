# SPDX-License-Identifier: AGPL-3.0-or-later
import ast
import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
import pytest


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _noop(*args, **kwargs):
    pass


def _load_save_helpers():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {
        "_without_runtime_truncation",
        "_unsloth_tokenizer_save_pretrained",
        "_source_tokenizer_class",
        "_restore_source_tokenizer_class",
    }
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body = nodes, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {
        "contextmanager": contextmanager,
        "json": json,
        "os": os,
        "logger": _Logger(),
        "_resolve_hub_repo_cached_file": lambda *args, **kwargs: None,
        "_tokenizer_cache_dir": lambda tokenizer: None,
        "_tokenizer_revision": lambda tokenizer: None,
        "_preserve_sentencepiece_tokenizer_assets": _noop,
        "_preserve_tokenizer_eos_token": _noop,
        "_preserve_repaired_tokenizer_class": _noop,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_unsloth_tokenizer_save_pretrained"]


def _trained_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate("abcdefgh")}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, pad_token = "[PAD]", unk_token = "[UNK]"
    )
    tokenizer(["a b c d e f g h"], truncation = True, max_length = 3, padding = "max_length")
    assert tokenizer._tokenizer.truncation["max_length"] == 3
    tokenizer.original_save_pretrained = tokenizer.save_pretrained
    return tokenizer


def _load_repair_helper():
    source = Path(__file__).parents[2] / "unsloth" / "tokenizer_utils.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {
        "_tokenizer_objects",
        "_repair_tokenizer_backend_from_json",
        "_inferred_tokenizer_max_length",
        "_resolve_tokenizer_json",
    }
    module = ast.Module(
        body = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
        ],
        type_ignores = [],
    )
    namespace = {"os": os, "json": json}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_repair_tokenizer_backend_from_json"]


def _capture_loaded_limit(tokenizer, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    return _load_repair_helper()(tokenizer)


def test_saved_tokenizer_json_has_no_training_truncation(tmp_path):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()

    save(tokenizer, str(tmp_path))

    saved = json.loads((tmp_path / "tokenizer.json").read_text(encoding = "utf-8"))
    assert saved["truncation"] is None
    assert saved["padding"] is None
    reloaded = PreTrainedTokenizerFast.from_pretrained(str(tmp_path))
    assert len(reloaded("a b c d e f g h")["input_ids"]) == 8


def test_training_truncation_is_restored_after_save(tmp_path):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()

    save(tokenizer, str(tmp_path))

    assert tokenizer._tokenizer.truncation["max_length"] == 3
    assert tokenizer._tokenizer.padding["length"] == 3


def test_processor_wrapper_clears_inner_tokenizer_truncation(tmp_path):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    processor = types.SimpleNamespace(
        tokenizer = tokenizer,
        original_save_pretrained = lambda path, **kwargs: tokenizer.save_pretrained(path),
    )

    save(processor, str(tmp_path))

    saved = json.loads((tmp_path / "tokenizer.json").read_text(encoding = "utf-8"))
    assert saved["truncation"] is None
    assert tokenizer._tokenizer.truncation["max_length"] == 3


def test_resaving_a_truncated_checkpoint_drops_leaked_max_length(tmp_path, monkeypatch):
    save = _load_save_helpers()
    broken = tmp_path / "broken"
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(str(broken))
    reloaded = PreTrainedTokenizerFast.from_pretrained(str(broken))
    _capture_loaded_limit(reloaded, monkeypatch)
    reloaded.original_save_pretrained = reloaded.save_pretrained

    fixed = tmp_path / "fixed"
    save(reloaded, str(fixed))

    config = json.loads((fixed / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config.get("max_length") is None
    assert (
        len(PreTrainedTokenizerFast.from_pretrained(str(fixed))("a b c d e f g h")["input_ids"])
        == 8
    )


class _PaddingClearFails:
    def __init__(self, tokenizer):
        self._inner = tokenizer._tokenizer

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def no_padding(self):
        raise RuntimeError("no_padding failed")


def test_partial_clear_failure_still_restores_truncation(tmp_path):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    inner = tokenizer._tokenizer
    processor = types.SimpleNamespace(
        tokenizer = types.SimpleNamespace(_tokenizer = _PaddingClearFails(tokenizer)),
        original_save_pretrained = lambda path, **kwargs: None,
    )

    save(processor, str(tmp_path))

    assert inner.truncation["max_length"] == 3
    assert inner.padding["length"] == 3


@pytest.mark.parametrize("padding_only", [False, True])
def test_nested_save_cleans_limit_before_upload(tmp_path, monkeypatch, padding_only):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    if padding_only:
        tokenizer._tokenizer.no_truncation()
    tokenizer.save_pretrained(tmp_path / "broken")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "broken")
    _capture_loaded_limit(tokenizer, monkeypatch)
    tokenizer.original_save_pretrained = tokenizer.save_pretrained
    tokenizer.save_pretrained = types.MethodType(save, tokenizer)
    before = dict(tokenizer.init_kwargs)

    def inspect_upload(repo_id, **kwargs):
        config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
        assert "max_length" not in config

    tokenizer.push_to_hub = inspect_upload
    processor = types.SimpleNamespace(
        tokenizer = tokenizer,
        original_save_pretrained = lambda path, **kwargs: tokenizer.save_pretrained(
            path, push_to_hub = True
        ),
    )
    save(processor, tmp_path / "fixed")
    assert tokenizer.init_kwargs == before
    assert tokenizer._tokenizer.padding["length"] == 3


def test_failed_save_restores_serialization_limit(tmp_path, monkeypatch):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "broken")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "broken")
    _capture_loaded_limit(tokenizer, monkeypatch)

    def fail(*args, **kwargs):
        assert "max_length" not in tokenizer.init_kwargs
        raise ValueError("save failed")

    tokenizer.original_save_pretrained = fail
    with pytest.raises(ValueError, match = "save failed"):
        save(tokenizer, tmp_path)
    assert tokenizer.init_kwargs["max_length"] == 3
    assert tokenizer._tokenizer.truncation["max_length"] == 3


@pytest.mark.parametrize("retrain", [False, True])
def test_loaded_limit_survives_tokenizer_calls(tmp_path, monkeypatch, retrain):
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    repair = _load_repair_helper()
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "broken")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "broken")
    processor = types.SimpleNamespace(tokenizer = tokenizer)
    assert repair(processor) is processor

    if retrain:
        tokenizer("a b c d e f g h", truncation = True, max_length = 7, padding = "max_length")
    else:
        tokenizer("a")
    repair(processor)
    before = dict(tokenizer.init_kwargs)
    truncation, padding = tokenizer._tokenizer.truncation, tokenizer._tokenizer.padding
    tokenizer.original_save_pretrained = tokenizer.save_pretrained

    _load_save_helpers()(tokenizer, tmp_path / "fixed")

    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert "max_length" not in config
    assert tokenizer._unsloth_loaded_runtime_max_length == 3
    assert tokenizer.init_kwargs == before
    assert tokenizer._tokenizer.truncation == truncation
    assert tokenizer._tokenizer.padding == padding


def test_loaded_limit_preserves_explicit_override_and_restores_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "broken")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "broken")
    _load_repair_helper()(tokenizer)
    tokenizer("a")
    before = dict(tokenizer.init_kwargs)

    def fail(*args, **kwargs):
        assert "max_length" not in tokenizer.init_kwargs
        raise ValueError("save failed")

    tokenizer.original_save_pretrained = fail
    with pytest.raises(ValueError, match = "save failed"):
        _load_save_helpers()(tokenizer, tmp_path / "fixed")
    assert tokenizer.init_kwargs == before
    assert tokenizer._tokenizer.truncation is None
    assert tokenizer._tokenizer.padding is None

    tokenizer.init_kwargs["max_length"] = 11
    tokenizer.original_save_pretrained = tokenizer.save_pretrained
    _load_save_helpers()(tokenizer, tmp_path / "fixed")
    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["max_length"] == 11
    assert tokenizer.init_kwargs["max_length"] == 11


def test_limit_without_loaded_provenance_is_not_recaptured(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    repair = _load_repair_helper()
    tokenizer = _trained_tokenizer()
    tokenizer.init_kwargs["max_length"] = 11
    repair(tokenizer)
    tokenizer("a b", truncation = True, max_length = 11)
    repair(tokenizer)
    tokenizer("a")
    _load_save_helpers()(tokenizer, tmp_path)
    config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["max_length"] == 11


@pytest.mark.parametrize("reload_source", [False, True])
def test_matching_explicit_limit_keeps_processor_default(tmp_path, monkeypatch, reload_source):
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object = _trained_tokenizer()._tokenizer,
        pad_token = "[PAD]",
        unk_token = "[UNK]",
        max_length = 3,
    )
    if reload_source:
        tokenizer.save_pretrained(tmp_path / "source")
        tokenizer = Qwen2TokenizerFast.from_pretrained(tmp_path / "source")
    _capture_loaded_limit(tokenizer, monkeypatch)
    tokenizer.original_save_pretrained = tokenizer.save_pretrained
    before = dict(tokenizer.init_kwargs)

    _load_save_helpers()(tokenizer, tmp_path / "fixed")

    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["max_length"] == 3
    assert tokenizer.init_kwargs == before
    processor = Qwen3VLProcessor(
        image_processor = Qwen2VLImageProcessor(),
        tokenizer = Qwen2TokenizerFast.from_pretrained(tmp_path / "fixed"),
        video_processor = Qwen3VLVideoProcessor(),
    )
    assert len(processor(text = "a b c d e f g h")["input_ids"][0]) == 3


@pytest.mark.parametrize("config_contents", [None, "not json", "[]"])
def test_unknown_source_preserves_init_limit(tmp_path, monkeypatch, config_contents):
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "source")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "source")
    config_path = tmp_path / "source/tokenizer_config.json"
    if config_contents is None:
        config_path.unlink()
    else:
        config_path.write_text(config_contents, encoding = "utf-8")
    _capture_loaded_limit(tokenizer, monkeypatch)
    tokenizer("a")
    tokenizer.original_save_pretrained = tokenizer.save_pretrained

    _load_save_helpers()(tokenizer, tmp_path / "fixed")

    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["max_length"] == 3


def test_inferred_limit_uses_supplied_hub_cache_and_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "source")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "source")
    tokenizer.name_or_path = "test/runtime-limit"
    revision = "a" * 40
    cached = tmp_path / "hub/models--test--runtime-limit/snapshots" / revision
    cached.mkdir(parents = True)
    (cached / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    repair = _load_repair_helper()
    repair(tokenizer, cache_dir = str(tmp_path / "hub"), revision = revision)
    tokenizer("a")
    tokenizer.original_save_pretrained = tokenizer.save_pretrained

    _load_save_helpers()(tokenizer, tmp_path / "fixed")

    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert "max_length" not in config


def test_missing_custom_cache_config_does_not_use_default_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants, try_to_load_from_cache

    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "default_hub"))
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(tmp_path / "source")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "source")
    tokenizer.name_or_path = "test/runtime-limit"
    revision = "b" * 40
    cached = tmp_path / "default_hub/models--test--runtime-limit/snapshots" / revision
    cached.mkdir(parents = True)
    cached_config = cached / "tokenizer_config.json"
    cached_config.write_text("{}", encoding = "utf-8")
    custom_cache = tmp_path / "custom_hub"
    custom_cache.mkdir()
    repair = _load_repair_helper()
    assert try_to_load_from_cache(
        tokenizer.name_or_path, "tokenizer_config.json", revision = revision
    ) == str(cached_config)

    repair(tokenizer, cache_dir = str(custom_cache), revision = revision)
    tokenizer("a")
    tokenizer.original_save_pretrained = tokenizer.save_pretrained
    _load_save_helpers()(tokenizer, tmp_path / "fixed")

    config = json.loads((tmp_path / "fixed/tokenizer_config.json").read_text(encoding = "utf-8"))
    assert config["max_length"] == 3
    assert tokenizer._unsloth_loaded_runtime_max_length is None
