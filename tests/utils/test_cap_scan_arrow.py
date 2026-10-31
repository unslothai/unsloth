# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The pre-tokenized cap scan answers from Arrow lengths when it can, and its verdict
never differs from the row-by-row pass it short-circuits."""

from __future__ import annotations

import ast
import re
import textwrap
import types
from pathlib import Path

import pytest

datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")
from datasets import Dataset, Features, Image, LargeList, Sequence, Value  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_PATH = REPO_ROOT / "unsloth" / "models" / "rl.py"
CAP = 3


def _load_helpers():
    source = RL_PATH.read_text(encoding = "utf-8")
    wanted = {
        "_attested_within_cap",
        "_arrow_within_cap",
        "pretokenized_within_cap",
        "_SCAN_ROWS",
        "_TRUNCATION_ATTESTATION_ATTR",
    }
    kept = [
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        )
    ]
    module = types.ModuleType("rl_cap_helpers")
    exec(compile(ast.Module(body = kept, type_ignores = []), str(RL_PATH), "exec"), module.__dict__)
    return module, source


rl, _SOURCE = _load_helpers()


def _inlined():
    start = _SOURCE.index('"    def _unsloth_within_cap(_ds):\\n"')
    end = _SOURCE.index('"    def _unsloth_splits_within_cap(_ev):\\n"')
    body = "".join(re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*$', _SOURCE[start:end], re.MULTILINE))
    namespace = {"_unsloth_cap": CAP, "_UNSLOTH_SCAN_ROWS": 1024}
    exec(textwrap.dedent(body.encode().decode("unicode_escape")), namespace)
    return namespace["_unsloth_within_cap"]


def _row_pass(ds, cap = CAP):
    """The row loop as it was before the Arrow short-circuit."""
    if ds is None:
        return True
    try:
        try:
            n = len(ds)
        except Exception:
            n = None
        rows = iter(ds)
        if rows is iter(ds):
            return False
        seen = 0
        for row in rows:
            if "input_ids" not in row:
                return True
            if len(row["input_ids"]) > cap:
                return False
            seen += 1
            if n is None and seen >= 1024:
                return False
    except Exception:
        return False
    return True


def _shapes():
    fits = Dataset.from_dict(
        {"input_ids": [[1, 2], [3, 4, 5], []], "labels": [[1, 2], [3, 4, 5], []]}
    )
    over = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5, 6]]})
    none_first = Dataset.from_dict({"input_ids": [None, [1, 2]]})
    none_late = Dataset.from_dict({"input_ids": [[1], [1, 2], None]})
    big = Dataset.from_dict({"input_ids": [[1] * (i % 4) for i in range(5000)]})
    big_over = Dataset.from_dict(
        {"input_ids": [[1] * (3 if i != 4321 else 4) for i in range(5000)]}
    )
    large = Dataset.from_dict(
        {"input_ids": [[1, 2], [3]]}, features = Features({"input_ids": LargeList(Value("int32"))})
    )
    fixed = Dataset.from_dict(
        {"input_ids": [[1, 2], [3, 4]]},
        features = Features({"input_ids": Sequence(Value("int32"), length = 2)}),
    )
    fixed_over = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]]},
        features = Features({"input_ids": Sequence(Value("int32"), length = 4)}),
    )
    nested = Dataset.from_dict({"input_ids": [[[1, 2, 3, 4]], [[1], [2], [3]]]})
    as_text = Dataset.from_dict({"input_ids": ["abcdef", "ab"]})
    as_int = Dataset.from_dict({"input_ids": [1, 2]})
    raw = Dataset.from_dict({"text": ["a", "b"]})
    empty = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]}).select([])
    # An indices mapping: the long row is stored but not selected, and the other way round.
    hidden_long = over.select([0])
    shown_long = over.select([1, 0])
    filtered = big_over.filter(lambda e: len(e["input_ids"]) < 4)
    shuffled = big_over.shuffle(seed = 0)
    torch_fmt = over.with_format("numpy")
    cols_hidden = over.with_format(
        None, columns = ["labels"] if "labels" in over.column_names else []
    )
    with_image = Dataset.from_dict(
        {"input_ids": [[1, 2]], "image": [None]},
        features = Features({"input_ids": Sequence(Value("int32")), "image": Image()}),
    )
    transformed = fits.with_transform(lambda b: {"input_ids": [[1] * 9 for _ in b["input_ids"]]})
    attested = Dataset.from_dict({"input_ids": [[1] * 9]})
    attested._unsloth_truncated_to = CAP
    return {
        "fits": fits,
        "over": over,
        "none_first": none_first,
        "none_late": none_late,
        "big": big,
        "big_over": big_over,
        "large": large,
        "fixed": fixed,
        "fixed_over": fixed_over,
        "nested": nested,
        "as_text": as_text,
        "as_int": as_int,
        "raw": raw,
        "empty": empty,
        "hidden_long": hidden_long,
        "shown_long": shown_long,
        "filtered": filtered,
        "shuffled": shuffled,
        "numpy_fmt": torch_fmt,
        "cols_hidden": cols_hidden,
        "with_image": with_image,
        "transformed": transformed,
        "attested": attested,
        "none": None,
    }


SHAPES = _shapes()


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_both_copies_match_the_row_pass(name):
    ds = SHAPES[name]
    expected = True if name == "attested" else _row_pass(ds)
    assert rl.pretokenized_within_cap(ds, CAP) is expected
    assert _inlined()(ds) is expected


def test_arrow_never_claims_what_the_rows_deny():
    for name, ds in SHAPES.items():
        if rl._arrow_within_cap(ds, CAP) is True:
            assert _row_pass(ds) is True, name


def test_a_plain_split_is_answered_without_reading_a_row(monkeypatch):
    ds = Dataset.from_dict({"input_ids": [[1] * (i % 4) for i in range(2000)]}).filter(
        lambda e: len(e["input_ids"]) > 0
    )

    def refuse(self, *a, **k):
        raise AssertionError("rows were read")

    monkeypatch.setattr(Dataset, "__iter__", refuse)
    monkeypatch.setattr(Dataset, "__getitem__", refuse)
    assert rl.pretokenized_within_cap(ds, CAP) is True
    assert _inlined()(ds) is True
