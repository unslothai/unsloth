# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from eval.dataset import load_eval_examples, DatasetRef


def test_load_local_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"q": "1+1?", "a": "2"}, {"q": "2+2?", "a": "4"}, {"q": "x", "a": "y"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=2)
    assert examples == [("1+1?", "2"), ("2+2?", "4")]  # first 2 only


def test_load_local_limit_none_returns_all(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"q": "a", "a": "1"}, {"q": "b", "a": "2"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=None)
    assert len(examples) == 2


def test_missing_column_raises(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"q": "a", "a": "1"}))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    with pytest.raises(ValueError, match="column"):
        load_eval_examples(ref, input_col="missing", reference_col="a", limit=10)


def test_reference_dict_preserved(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"q": "extract", "a": {"total": 5}}))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=10)
    assert examples[0][1] == {"total": 5}
