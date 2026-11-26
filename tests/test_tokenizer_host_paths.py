# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tokenizer folder's own JSON must not make the tokenizer open a host file outside the folder."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import AutoTokenizer  # noqa: E402
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_FLAGS = (
    "_unsloth_patched_tokenizer_config_vocab_paths",
    "_unsloth_refuses_tokenizer_host_paths",
)
_OWN_VOCAB = "[PAD]\n[UNK]\nown1\nown2\n"
_SECRET = "SECRET_LINE_A\nSECRET_LINE_B\nSECRET_LINE_C\n[UNK]\n"


def _load_import_fixes():
    # UNSLOTH_IMPORT_FIXES points the test at another revision for a base-vs-head run.
    path = os.environ.get("UNSLOTH_IMPORT_FIXES", str(REPO / "unsloth" / "import_fixes.py"))
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_host_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guarded():
    # Start from the plain transformers method (conftest may have installed an older wrapper).
    saved = PreTrainedTokenizerBase.__dict__["_from_pretrained"]
    function = saved.__func__
    while any(getattr(function, flag, False) for flag in _FLAGS):
        function = function.__wrapped__
    PreTrainedTokenizerBase._from_pretrained = classmethod(function)
    fixes = _load_import_fixes()
    fixes.fix_transformers_tokenizer_config_vocab_file_override()
    fixes.fix_transformers_tokenizer_config_vocab_file_override()  # idempotent
    yield fixes
    PreTrainedTokenizerBase._from_pretrained = saved


def _folder(tmp_path, name, config, files):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "tokenizer_config.json").write_text(json.dumps(config))
    for file_name, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content)
        (folder / file_name).write_text(text)
    return folder


def _tokenizer_json_with_vocab(vocab):
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "vocab": vocab,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
        },
    }


def _crafted(tmp_path, secret):
    bert = {"tokenizer_class": "BertTokenizer"}
    return {
        "vocab": _folder(tmp_path, "vocab", {**bert, "vocab": secret}, {"vocab.txt": _OWN_VOCAB}),
        "init_inputs": _folder(tmp_path, "init_inputs", {**bert, "init_inputs": [secret]}, {}),
        "special_tokens_map": _folder(
            tmp_path,
            "special_map",
            bert,
            {"vocab.txt": _OWN_VOCAB, "special_tokens_map.json": {"vocab": secret}},
        ),
        "tokenizer_json": _folder(
            tmp_path, "tokenizer_json", bert, {"tokenizer.json": _tokenizer_json_with_vocab(secret)}
        ),
    }


@pytest.mark.parametrize(
    "spelling", ["vocab", "init_inputs", "special_tokens_map", "tokenizer_json"]
)
def test_tokenizer_files_cannot_read_a_host_file(guarded, tmp_path, spelling):
    secret = tmp_path / "owner-notes.txt"
    secret.write_text(_SECRET)
    folder = _crafted(tmp_path, str(secret))[spelling]
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
    except Exception:
        return  # refused or failed to load: nothing was read back
    tokens = tokenizer.convert_ids_to_tokens([0, 1, 2])
    assert not any("SECRET" in str(token) for token in tokens), tokens


def test_normal_folder_and_round_trip_load_unchanged(guarded, tmp_path):
    folder = _folder(
        tmp_path,
        "normal",
        # A legacy config records where it was saved from; that path exists but is never opened.
        {"tokenizer_class": "BertTokenizer", "name_or_path": str(tmp_path / "owner-notes.txt")},
        {"vocab.txt": _OWN_VOCAB},
    )
    (tmp_path / "owner-notes.txt").write_text(_SECRET)
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    assert tokenizer.convert_ids_to_tokens([0, 1, 2]) == ["[PAD]", "[UNK]", "own1"]
    saved = tmp_path / "saved"
    tokenizer.save_pretrained(str(saved))
    again = AutoTokenizer.from_pretrained(str(saved))
    assert again.convert_ids_to_tokens([0, 1, 2]) == ["[PAD]", "[UNK]", "own1"]
    assert again("own1 own2")["input_ids"] == tokenizer("own1 own2")["input_ids"]


def test_a_vocab_inside_the_folder_is_still_allowed(guarded, tmp_path):
    folder = tmp_path / "inside"
    folder.mkdir()
    (folder / "vocab.txt").write_text(_OWN_VOCAB)
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "vocab": str(folder / "vocab.txt")})
    )
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    assert tokenizer.convert_ids_to_tokens([2]) == ["own1"]


def test_caller_kwargs_are_trusted(guarded, tmp_path):
    own = tmp_path / "caller_vocab.txt"
    own.write_text(_OWN_VOCAB)
    folder = _folder(
        tmp_path, "caller", {"tokenizer_class": "BertTokenizer"}, {"vocab.txt": _OWN_VOCAB}
    )
    tokenizer = AutoTokenizer.from_pretrained(str(folder), vocab = str(own))
    assert tokenizer.convert_ids_to_tokens([2]) == ["own1"]
