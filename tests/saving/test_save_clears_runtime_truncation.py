# SPDX-License-Identifier: AGPL-3.0-or-later
import ast
import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import pytest


class _Logger:
    def warning_once(self, *args, **kwargs):
        pass


def _noop(*args, **kwargs):
    pass


def _load_save_helpers():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    wanted = {"_without_runtime_truncation", "_unsloth_tokenizer_save_pretrained"}
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


def test_resaving_a_truncated_checkpoint_drops_leaked_max_length(tmp_path):
    save = _load_save_helpers()
    broken = tmp_path / "broken"
    tokenizer = _trained_tokenizer()
    tokenizer.save_pretrained(str(broken))
    reloaded = PreTrainedTokenizerFast.from_pretrained(str(broken))
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
def test_nested_save_cleans_limit_before_upload(tmp_path, padding_only):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    if padding_only:
        tokenizer._tokenizer.no_truncation()
    tokenizer.save_pretrained(tmp_path / "broken")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path / "broken")
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


def test_failed_save_restores_serialization_limit(tmp_path):
    save = _load_save_helpers()
    tokenizer = _trained_tokenizer()
    tokenizer.init_kwargs["max_length"] = 3

    def fail(*args, **kwargs):
        assert "max_length" not in tokenizer.init_kwargs
        raise ValueError("save failed")

    tokenizer.original_save_pretrained = fail
    with pytest.raises(ValueError, match = "save failed"):
        save(tokenizer, tmp_path)
    assert tokenizer.init_kwargs["max_length"] == 3
    assert tokenizer._tokenizer.truncation["max_length"] == 3
