# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized datasets (input_ids / attention_mask / labels) route to training as-is (#7139), and nothing else moves."""

import inspect
import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils.dataset_format import check_dataset_format as hub_check_dataset_format  # noqa: E402
from utils.datasets.dataset_utils import (  # noqa: E402
    _format_dataset,
    check_dataset_format,
    format_and_template_dataset,
    format_dataset,
)
from utils.datasets.pretokenized import (  # noqa: E402
    has_custom_labels,
    is_pretokenized_dataset,
    prepare_pretokenized_dataset,
)


def _pretokenized(with_labels = True, n = 4):
    rows = {
        "input_ids": [[1, 2, 3, 4, 5][: 3 + i % 3] for i in range(n)],
        "attention_mask": [[1] * (3 + i % 3) for i in range(n)],
    }
    if with_labels:
        rows["labels"] = [[-100] + ids[1:] for ids in rows["input_ids"]]
    return Dataset.from_dict(rows)


NORMAL_DATASETS = {
    "alpaca": (
        Dataset.from_dict({"instruction": ["Say hi"], "input": [""], "output": ["hi"]}),
        "alpaca",
    ),
    "chatml": (
        Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "hi"},
                    ]
                ]
            }
        ),
        "chatml",
    ),
    "sharegpt": (
        Dataset.from_dict(
            {
                "conversations": [
                    [{"from": "human", "value": "Say hi"}, {"from": "gpt", "value": "hi"}]
                ]
            }
        ),
        "sharegpt",
    ),
    # A text dataset that also ships token ids keeps its text route.
    "question_answer_with_ids": (
        Dataset.from_dict({"question": ["Say hi"], "answer": ["hi"], "input_ids": [[1, 2, 3]]}),
        "custom_heuristic",
    ),
}


@pytest.mark.parametrize("with_labels", [True, False])
def test_pretokenized_is_detected_and_needs_no_mapping(with_labels):
    ds = _pretokenized(with_labels)
    assert is_pretokenized_dataset(ds)
    for check in (check_dataset_format, hub_check_dataset_format):
        result = check(ds)
        assert result["requires_manual_mapping"] is False
        assert result["detected_format"] == "pretokenized"


def test_format_passes_pretokenized_rows_through_untouched():
    ds = _pretokenized()
    info = format_and_template_dataset(ds, model_name = "unsloth/Qwen3-0.6B", tokenizer = None)
    assert info["success"] is True
    assert info["final_format"] == "pretokenized"
    assert info["dataset"].column_names == ds.column_names
    assert info["dataset"].to_dict() == ds.to_dict()


@pytest.mark.parametrize("name", sorted(NORMAL_DATASETS))
def test_normal_datasets_keep_their_route(name):
    ds, expected = NORMAL_DATASETS[name]
    result = hub_check_dataset_format(ds)
    assert result["detected_format"] == expected
    assert check_dataset_format(ds)["detected_format"] == expected
    assert format_dataset(ds, format_type = "auto")["final_format"] != "pretokenized"


@pytest.mark.parametrize(
    "rows",
    [
        {"input_ids": [["a", "b"]]},
        {"input_ids": [[]]},
        {"input_ids": [[1, 2, 3]], "labels": [[1, 2]]},
        {"tokens": [[1, 2, 3]]},
    ],
)
def test_malformed_token_rows_still_ask_for_mapping(rows):
    ds = Dataset.from_dict(rows)
    assert not is_pretokenized_dataset(ds)
    assert hub_check_dataset_format(ds)["requires_manual_mapping"] is True


def test_prepare_truncates_every_token_column_and_is_idempotent():
    ds = _pretokenized()
    once = prepare_pretokenized_dataset(ds, max_seq_length = 3, vocab_size = 100)
    for column in ("input_ids", "attention_mask", "labels"):
        assert max(len(v) for v in once[column]) == 3
    assert once[0]["labels"] == [-100, 2, 3]
    twice = prepare_pretokenized_dataset(once, max_seq_length = 3, vocab_size = 100)
    assert twice.to_dict() == once.to_dict()


def test_prepare_leaves_short_rows_alone():
    ds = _pretokenized()
    assert prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100) is ds


def test_prepare_rejects_ids_outside_the_vocabulary():
    with pytest.raises(ValueError, match = "outside this model's vocabulary"):
        prepare_pretokenized_dataset(_pretokenized(), max_seq_length = 2048, vocab_size = 4)


def test_prepare_rejects_labels_of_another_length():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2]]})
    with pytest.raises(ValueError, match = "same length"):
        prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)


def test_only_user_labels_turn_off_response_masking():
    assert has_custom_labels(_pretokenized(with_labels = True))
    assert not has_custom_labels(_pretokenized(with_labels = False))
    assert not has_custom_labels(_pretokenized(with_labels = False).to_iterable_dataset())


@pytest.mark.parametrize("format_type", ["auto", "alpaca", "chatml", "sharegpt"])
def test_any_format_but_raw_routes_token_ids(format_type):
    info = format_dataset(_pretokenized(), format_type = format_type)
    assert info["final_format"] == "pretokenized"
    assert info["requires_manual_mapping"] is False


def test_positional_custom_format_mapping_still_binds():
    params = inspect.signature(_format_dataset).parameters
    names = list(params)
    defaults = [params[n].default for n in names[1 : names.index("custom_format_mapping")]]
    assert format_dataset(_pretokenized(), *defaults, None)["final_format"] == "pretokenized"
    mapped = format_dataset(_pretokenized(), *defaults, {"input_ids": "user"})
    assert mapped["final_format"] != "pretokenized"


# Padded rows: TRL's collator rebuilds attention_mask as all ones, so masked padding is trimmed at preparation.
_PADDED = {
    "left": ([0, 0, 4, 5], [0, 0, 1, 1], [-100, -100, 4, 5], [4, 5], [4, 5]),
    "right": ([2, 3, 5, 0, 0], [1, 1, 1, 0, 0], [2, 3, 5, -100, -100], [2, 3, 5], [2, 3, 5]),
    "both": ([0, 2, 3, 0], [0, 1, 1, 0], [-100, -100, 3, -100], [2, 3], [-100, 3]),
}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("side", sorted(_PADDED))
def test_prepare_trims_masked_padding(side, with_labels, streaming):
    ids, mask, labels, want_ids, want_labels = _PADDED[side]
    rows = {"input_ids": [ids, [2, 3]], "attention_mask": [mask, [1, 1]]}
    if with_labels:
        rows["labels"] = [labels, [2, 3]]
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    out = list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))
    assert out[0]["input_ids"] == want_ids
    assert out[0]["attention_mask"] == [1] * len(want_ids)
    if with_labels:
        assert out[0]["labels"] == want_labels
    assert out[1] == {k: v[1] for k, v in rows.items()}


def test_prepare_trims_then_truncates_and_is_idempotent():
    ds = Dataset.from_dict({"input_ids": [[0, 0, 4, 5, 6]], "attention_mask": [[0, 0, 1, 1, 1]]})
    once = prepare_pretokenized_dataset(ds, max_seq_length = 2, vocab_size = 100)
    assert once.to_dict() == {"input_ids": [[4, 5]], "attention_mask": [[1, 1]]}
    assert (
        prepare_pretokenized_dataset(once, max_seq_length = 2, vocab_size = 100).to_dict()
        == once.to_dict()
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "mask, match",
    [
        ([1, 0, 1, 1], "row 1 has 'attention_mask' zeros between attended tokens"),
        ([0, 0, 0, 0], "row 1 has an all-zero 'attention_mask'"),
        ([0, 2, 1, 1], "row 1 has 'attention_mask' values other than 0 and 1"),
        # A negative minimum or no zero at all must not skip the value check.
        ([-1, 0, 1, 1], "row 1 has 'attention_mask' values other than 0 and 1"),
        ([1, 2, 1, 1], "row 1 has 'attention_mask' values other than 0 and 1"),
    ],
)
def test_prepare_refuses_masks_it_cannot_trim(mask, match, streaming):
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 2, "attention_mask": [[1, 1, 1, 1], mask]}
    )
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = match):
        list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "rows, column",
    [
        ({"input_ids": [[1, 2, 3]], "attention_mask": [[0, 1]]}, "attention_mask"),
        ({"input_ids": [[1, 2, 3]], "labels": [[1, 2]]}, "labels"),
        (
            {
                "input_ids": [[0, 2, 3]],
                "attention_mask": [[0, 1, 1]],
                "assistant_masks": [[0, 1, 1, 1]],
            },
            "assistant_masks",
        ),
    ],
)
def test_prepare_refuses_a_padded_mask_of_another_length(rows, column, streaming):
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = f"'{column}' must have the same length"):
        list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))


@pytest.mark.parametrize("streaming", [False, True])
def test_prepare_trims_and_truncates_trl_supervision_masks_with_input_ids(streaming):
    rows = {
        "input_ids": [[0, 0, 4, 5, 6], [2, 3]],
        "attention_mask": [[0, 0, 1, 1, 1], [1, 1]],
        "completion_mask": [[0, 0, 0, 1, 1], [0, 1]],
        "assistant_masks": [[0, 0, 0, 1, 1], [0, 1]],
    }
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    out = list(prepare_pretokenized_dataset(ds, max_seq_length = 2, vocab_size = 100))
    assert out[0] == {
        "input_ids": [4, 5],
        "attention_mask": [1, 1],
        "completion_mask": [0, 1],
        "assistant_masks": [0, 1],
    }
    assert out[1] == {k: v[1] for k, v in rows.items()}


def test_preview_does_not_report_trimmed_padding_as_truncation():
    from core.training import token_preview_worker

    rows = [
        {"input_ids": [0, 0, 4, 5], "attention_mask": [0, 0, 1, 1]},
        {"input_ids": [0, 4, 5, 6, 7], "attention_mask": [0, 1, 1, 1, 1]},
    ]
    ids, _, full_lengths, _ = token_preview_worker._pretokenized(None, "m", rows, 3, {})
    assert ids == [[4, 5], [4, 5, 6]]
    # Row 0 only lost padding; row 1 lost padding and one real token to max_seq_length.
    assert full_lengths == [2, 4]


def test_mlx_training_prepares_pretokenized_train_and_eval_rows():
    # zoo's MLX batcher ignores attention_mask, so the worker must trim padding before MLXTrainer sees the rows.
    import ast

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mlx_training"
    )
    prepared = {
        node.args[0].id: node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "prepare_pretokenized_dataset"
    }
    trainer_line = next(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MLXTrainer"
    )
    assert set(prepared) == {"dataset", "eval_dataset"}
    assert max(prepared.values()) < trainer_line


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("bad", [100, 250, -1, -5])
def test_prepare_rejects_labels_outside_the_vocabulary(bad, streaming):
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[-100, 2, bad]]})
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = f"label {bad} is invalid"):
        list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))


@pytest.mark.parametrize("with_mask", [True, False])
def test_prepare_leaves_unpadded_rows_byte_identical(with_mask):
    rows = {"input_ids": [[1, 2, 3], [4, 5]], "labels": [[-100, 2, 3], [-100, -100]]}
    if with_mask:
        rows["attention_mask"] = [[1, 1, 1], [1, 1]]
    ds = Dataset.from_dict(rows)
    out = prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)
    assert out is ds
    assert out.data.table.equals(Dataset.from_dict(rows).data.table)


def test_trl_collator_never_attends_or_trains_masked_padding():
    collator_cls = pytest.importorskip("trl.trainer.sft_trainer").DataCollatorForLanguageModeling
    ids = [[0, 0, 4, 5], [2, 3, 5, 0], [0, 4, 2, 5]]
    masks = [[0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    ds = prepare_pretokenized_dataset(
        Dataset.from_dict({"input_ids": ids, "attention_mask": masks}),
        max_seq_length = 8,
        vocab_size = 6,
    )
    batch = collator_cls(pad_token_id = 0)([ds[i] for i in range(len(ds))])
    for r, (row, mask) in enumerate(zip(ids, masks)):
        real = [t for t, m in zip(row, mask) if m]
        attended = [
            t
            for t, m in zip(batch["input_ids"][r].tolist(), batch["attention_mask"][r].tolist())
            if m
        ]
        assert attended == real
        assert [x for x in batch["labels"][r].tolist() if x != -100] == real
