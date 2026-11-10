# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""_load_correct_tokenizer must not load the slow tokenizer on branches that return the fast one unchanged."""

import types
from unittest.mock import patch

import pytest
import transformers

import unsloth.tokenizer_utils as tu

_needs_v5 = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason = "transformers 4 loads a real slow tokenizer before the fast one",
)


class _Tok:
    def __init__(self, kind):
        self.kind = kind


def _run(
    name,
    fix_tokenizer,
    same_tokenization = True,
):
    calls = []

    def from_pretrained(tokenizer_name, **kwargs):
        kind = "slow" if kwargs.get("use_fast") is False else "fast"
        calls.append(kind)
        return _Tok(kind)

    repaired = []
    with (
        patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)),
        patch.object(
            tu,
            "_repair_tokenizer_backend_from_json",
            lambda tok, **k: repaired.append(tok.kind) or tok,
        ),
        patch.object(tu, "assert_same_tokenization", lambda a, b: same_tokenization),
        patch.object(tu, "convert_to_fast_tokenizer", lambda slow, **k: _Tok("converted")),
    ):
        result = tu._load_correct_tokenizer(name, fix_tokenizer = fix_tokenizer)
    return result, calls, repaired


@pytest.mark.parametrize(
    "name, fix_tokenizer",
    [
        ("unsloth/Llama-3.2-1B-Instruct", False),
        ("unsloth/Qwen2.5-Coder-7B-Instruct", True),
        ("unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit", True),
        ("unsloth/mistral-7b-instruct-v0.3", True),
        ("mistralai/Mistral-Nemo-Instruct-2407", True),
        ("unsloth/phi-4", True),
        ("microsoft/Phi-4-mini-instruct", True),
    ],
)
def test_early_return_skips_slow_tokenizer(name, fix_tokenizer):
    result, calls, repaired = _run(name, fix_tokenizer)
    assert result.kind == "fast"
    assert calls == ["fast"], f"{name}: slow tokenizer loaded but never used ({calls})"
    assert "slow" not in repaired


@pytest.mark.parametrize("same", [True, False])
def test_compared_path_still_loads_slow_tokenizer(same):
    result, calls, repaired = _run("unsloth/Llama-3.2-1B-Instruct", True, same)
    assert sorted(calls) == ["fast", "slow"]
    assert repaired.count("slow") == 1
    assert result.kind == ("fast" if same else "converted")


class _V5Tok:
    bos_token = "<s>"
    all_special_tokens = ["<s>"]

    def __init__(
        self,
        kind,
        legacy = False,
        sync_changes_ids = False,
    ):
        self.kind = kind
        self.legacy = legacy
        self.sync_changes_ids = sync_changes_ids
        self.rebuilds = 0
        self._add_bos_token = False

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.rebuilds += 1

    @property
    def add_eos_token(self):
        return False

    @add_eos_token.setter
    def add_eos_token(self, value):
        self.rebuilds += 1

    def __call__(self, text):
        drop_bos = self.sync_changes_ids and self.rebuilds > 0
        return types.SimpleNamespace(input_ids = [len(text)] + ([] if drop_bos else [1]))


def _v5_class(name, module = "transformers.models.qwen2.tokenization_qwen2"):
    return type(name, (_V5Tok,), {"__module__": module})


def _run_v5(
    cls,
    monkeypatch,
    v5 = True,
    **tok_kwargs,
):
    monkeypatch.setattr(tu, "_TRANSFORMERS_V5", v5, raising = False)
    calls, made = [], []

    def from_pretrained(tokenizer_name, **kwargs):
        kind = "slow" if kwargs.get("use_fast") is False else "fast"
        calls.append(kind)
        made.append(cls(kind, **tok_kwargs))
        return made[-1]

    with (
        patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)),
        patch.object(tu, "_repair_tokenizer_backend_from_json", lambda tok, **k: tok),
        patch.object(tu, "convert_to_fast_tokenizer", lambda slow, **k: slow),
    ):
        result = tu._load_correct_tokenizer("org/model", fix_tokenizer = True)
    return result, calls


@pytest.mark.parametrize(
    "name",
    [
        "Qwen2Tokenizer",
        "Qwen3_5Tokenizer",
        "GemmaTokenizer",
        "GPT2Tokenizer",
        "CohereTokenizer",
        "TokenizersBackend",
    ],
)
def test_v5_skips_slow_load_that_copies_the_fast_tokenizer(name, monkeypatch):
    result, calls = _run_v5(_v5_class(name), monkeypatch)
    assert calls == [
        "fast"
    ], f"{name}: transformers 5 slow load only rebuilt the fast tokenizer ({calls})"
    assert result.kind == "fast"
    assert result.rebuilds == 2


def test_v5_llama_skips_only_at_legacy_false(monkeypatch):
    result, calls = _run_v5(_v5_class("LlamaTokenizer"), monkeypatch, legacy = False)
    assert calls == ["fast"] and result.kind == "fast"
    result, calls = _run_v5(_v5_class("LlamaTokenizer"), monkeypatch, legacy = True)
    assert calls == ["fast", "slow"]


def test_v5_sync_changing_ids_returns_the_slow_copy(monkeypatch):
    result, calls = _run_v5(_v5_class("GemmaTokenizer"), monkeypatch, sync_changes_ids = True)
    assert calls == ["fast", "slow"]
    assert result.kind == "slow"


@pytest.mark.parametrize(
    "cls, v5",
    [
        (_v5_class("Qwen2Tokenizer"), False),
        (_v5_class("Qwen2Tokenizer", module = "transformers_modules.org.tokenization"), True),
        (_v5_class("NllbTokenizer"), True),
    ],
)
def test_slow_load_kept_when_it_may_differ(cls, v5, monkeypatch):
    result, calls = _run_v5(cls, monkeypatch, v5 = v5)
    assert sorted(calls) == ["fast", "slow"]
    assert result.kind == "fast"


@_needs_v5
def test_v5_real_qwen2_tokenizer_constructed_once(tmp_path):
    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz\u0120!")}
    transformers.Qwen2Tokenizer(
        vocab = vocab, merges = [], eos_token = "<|endoftext|>", unk_token = None, bos_token = None
    ).save_pretrained(tmp_path)
    calls = []
    original = tu.AutoTokenizer.from_pretrained

    def from_pretrained(*args, **kwargs):
        calls.append("slow" if kwargs.get("use_fast") is False else "fast")
        return original(*args, **kwargs)

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        tokenizer = tu.load_correct_tokenizer(str(tmp_path))
    assert calls == ["fast"]
    assert type(tokenizer).__name__ == "Qwen2Tokenizer"
    assert tokenizer("hello world!", add_special_tokens = False).input_ids == [
        7,
        4,
        11,
        11,
        14,
        26,
        22,
        14,
        17,
        11,
        3,
        27,
    ]


@_needs_v5
def test_v5_failed_slow_load_keeps_the_tokenizer_as_loaded(tmp_path):
    from tokenizers import processors

    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz\u0120!")}
    vocab["<s>"] = len(vocab)
    tokenizer = transformers.Qwen2Tokenizer(
        vocab = vocab, merges = [], eos_token = "<|endoftext|>", unk_token = None, bos_token = "<s>"
    )
    tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
        single = "<s>:0 $A:0", pair = "<s>:0 $A:0 $B:1", special_tokens = [("<s>", vocab["<s>"])]
    )
    tokenizer.save_pretrained(tmp_path)
    original = tu.AutoTokenizer.from_pretrained
    expected = original(str(tmp_path))("ab").input_ids
    assert expected[0] == vocab["<s>"]

    def from_pretrained(*args, **kwargs):
        if kwargs.get("use_fast") is False:
            raise OSError("slow load failed")
        return original(*args, **kwargs)

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        loaded = tu.load_correct_tokenizer(str(tmp_path))
    assert loaded("ab").input_ids == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
