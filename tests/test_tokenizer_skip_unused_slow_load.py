# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""_load_correct_tokenizer must not load the slow tokenizer on branches that return the fast one unchanged."""

import types
from unittest.mock import patch

import pytest

import unsloth.tokenizer_utils as tu


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
    assert calls == ["slow", "fast"]
    assert repaired.count("slow") == 1
    assert result.kind == ("fast" if same else "converted")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
