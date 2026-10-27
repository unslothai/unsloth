# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pickle
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

datasets_dill = pytest.importorskip("datasets.utils._dill")
fingerprint = pytest.importorskip("datasets.fingerprint")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from utils.datasets import tokenizer_fingerprint


def _tokenizer(extra_vocab = 0, keep_init_state = True):
    vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3}
    vocab.update({f"t{i}": 4 + i for i in range(extra_vocab)})
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab = dict(vocab), merges = [("a", "b")], unk_token = "<unk>")
    )
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "<unk>")
    if keep_init_state:
        # As transformers 5 model tokenizers keep the vocab and merges they were built from.
        tok._vocab = dict(vocab)
        tok._merges = [("a", "b")]
    return tok


@pytest.fixture
def restore_reducer():
    original = datasets_dill._save_transformersPreTrainedTokenizerBase
    dispatch = dict(datasets_dill.Pickler.dispatch)
    yield original
    datasets_dill._save_transformersPreTrainedTokenizerBase = original
    datasets_dill.Pickler.dispatch.clear()
    datasets_dill.Pickler.dispatch.update(dispatch)


def _hash(tok):
    return fingerprint.Hasher.hash(lambda batch: tok(batch["text"]))


def test_install_is_idempotent_and_replaces_registered_reducers(restore_reducer):
    tok = _tokenizer()
    _hash(tok)  # datasets registers its reducer for this class on first sight
    assert datasets_dill.Pickler.dispatch[type(tok)] is restore_reducer

    assert tokenizer_fingerprint.install_tokenizer_fingerprint() is True
    installed = datasets_dill._save_transformersPreTrainedTokenizerBase
    assert installed is not restore_reducer
    assert datasets_dill.Pickler.dispatch[type(tok)] is installed
    assert tokenizer_fingerprint.install_tokenizer_fingerprint() is True
    assert datasets_dill._save_transformersPreTrainedTokenizerBase is installed


def test_init_only_state_is_left_out_of_the_hash(restore_reducer):
    tokenizer_fingerprint.install_tokenizer_fingerprint()
    tok = _tokenizer()
    before = _hash(tok)
    tok._vocab["never_read_after_init"] = 99
    tok._merges.append(("b", "a"))
    assert _hash(tok) == before
    assert "_vocab" in tok.__dict__ and "_merges" in tok.__dict__


def test_hash_still_follows_vocab_added_tokens_and_template(restore_reducer):
    tokenizer_fingerprint.install_tokenizer_fingerprint()
    base = _hash(_tokenizer())
    assert _hash(_tokenizer()) == base
    assert _hash(_tokenizer(extra_vocab = 1)) != base

    added = _tokenizer()
    added.add_tokens(["<new_token>"])
    assert _hash(added) != base

    templated = _tokenizer()
    templated.chat_template = "{{ messages }}"
    assert _hash(templated) != base


def test_tokenizers_without_init_state_hash_as_before(restore_reducer):
    tok = _tokenizer(keep_init_state = False)
    before = _hash(tok)
    tokenizer_fingerprint.install_tokenizer_fingerprint()
    assert _hash(tok) == before


def test_real_pickling_keeps_the_whole_tokenizer(restore_reducer):
    tokenizer_fingerprint.install_tokenizer_fingerprint()
    tok = _tokenizer()
    clone = pickle.loads(pickle.dumps(tok))
    assert clone._vocab == tok._vocab and clone._merges == tok._merges
    assert clone("ab")["input_ids"] == tok("ab")["input_ids"]


def test_missing_reducer_is_a_no_op(monkeypatch, restore_reducer):
    monkeypatch.delattr(datasets_dill, "_save_transformersPreTrainedTokenizerBase")
    assert tokenizer_fingerprint.install_tokenizer_fingerprint() is False


def test_init_state_kept_without_a_rust_tokenizer_to_stand_for_it(monkeypatch, restore_reducer):
    tok = _tokenizer()
    before = _hash(tok)
    # Seen from the reducer, this tokenizer's backend is no longer a Rust tokenizer.
    monkeypatch.setattr(tokenizer_fingerprint, "_is_rust_tokenizer", lambda value: False)
    tokenizer_fingerprint.install_tokenizer_fingerprint()
    assert _hash(tok) == before
