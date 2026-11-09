# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized datasets (input_ids / attention_mask / labels) route to training as-is (#7139), and nothing else moves."""

import importlib.util
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


@pytest.mark.parametrize("mask", ["completion_mask", "assistant_masks"])
def test_prepare_truncates_loss_masks_with_the_ids(mask):
    # TRL applies these masks to the labels: left at full length they fail on the first step.
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6]], mask: [[0, 0, 1, 1, 1, 1]]})
    out = prepare_pretokenized_dataset(
        ds, max_seq_length = 4, vocab_size = 100, completion_only_loss = True
    )
    assert out[0]["input_ids"] == [1, 2, 3, 4]
    assert out[0][mask] == [0, 0, 1, 1]


@pytest.mark.parametrize("mask", ["completion_mask", "assistant_masks"])
def test_prepare_rejects_loss_masks_of_another_length(mask):
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], mask: [[0, 1]]})
    with pytest.raises(ValueError, match = f"'{mask}' must have the same length"):
        prepare_pretokenized_dataset(
            ds, max_seq_length = 2048, vocab_size = 100, completion_only_loss = True
        )


def test_prepare_rejects_a_row_without_its_loss_mask():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5]], "completion_mask": [[0, 1, 1], None]})
    with pytest.raises(ValueError, match = "'completion_mask' must have the same length"):
        prepare_pretokenized_dataset(
            ds, max_seq_length = 2048, vocab_size = 100, completion_only_loss = True
        )


def test_prepare_ignores_unused_masks_next_to_user_labels():
    # With labels the masks are not used, so a stale mask column keeps training as before.
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3]], "labels": [[-100, 2, 3]], "completion_mask": [[0, 1]]}
    )
    assert prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100) is ds


def test_prepare_leaves_unused_null_masks_next_to_labels_alone():
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "labels": [[1, 2, 3, 4, 5, 6]],
            "completion_mask": [None],
        }
    )
    out = prepare_pretokenized_dataset(ds, max_seq_length = 4, vocab_size = 100)
    assert out[0]["input_ids"] == [1, 2, 3, 4] and out[0]["labels"] == [1, 2, 3, 4]
    assert out[0]["completion_mask"] is None


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("mask", ["assistant_masks", "completion_mask"])
def test_prepare_truncates_masks_next_to_labels_when_the_collator_applies_them(
    monkeypatch, mask, streamed
):
    # TRL < 1.7's collator applies these masks on top of the row's labels: a 6-token mask beside
    # 4-token labels fails the boolean index on the first step.
    import utils.datasets.pretokenized as pretokenized

    monkeypatch.setattr(pretokenized, "_collator_masks_user_labels", lambda: True, raising = False)
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "labels": [[-100, 2, 3, 4, 5, 6]],
            mask: [[0, 1, 1, 1, 1, 1]],
        }
    )
    if streamed:
        ds = ds.to_iterable_dataset()
    out = list(
        prepare_pretokenized_dataset(
            ds, max_seq_length = 4, vocab_size = 100, completion_only_loss = True
        )
    )
    assert out[0]["labels"] == [-100, 2, 3, 4] and out[0][mask] == [0, 1, 1, 1]
    bad = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]], mask: [[0, 1]]})
    with pytest.raises(ValueError, match = f"'{mask}' must have the same length"):
        prepare_pretokenized_dataset(
            bad, max_seq_length = 2048, vocab_size = 100, completion_only_loss = True
        )


def test_prepare_leaves_masks_next_to_labels_alone_when_the_collator_ignores_them(monkeypatch):
    # TRL 1.7+ trains on the labels as given and never reads the masks beside them.
    import utils.datasets.pretokenized as pretokenized

    monkeypatch.setattr(pretokenized, "_collator_masks_user_labels", lambda: False, raising = False)
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "labels": [[1, 2, 3, 4, 5, 6]],
            "assistant_masks": [[0, 1]],
        }
    )
    out = prepare_pretokenized_dataset(
        ds, max_seq_length = 4, vocab_size = 100, completion_only_loss = True
    )
    assert out[0]["labels"] == [1, 2, 3, 4] and out[0]["assistant_masks"] == [0, 1]


def test_prepare_checks_streamed_loss_masks():
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4, 5, 6]], "completion_mask": [[0, 1]]}
    ).to_iterable_dataset()
    out = prepare_pretokenized_dataset(
        ds, max_seq_length = 4, vocab_size = 100, completion_only_loss = True
    )
    with pytest.raises(ValueError, match = "'completion_mask' must have the same length"):
        list(out)


@pytest.mark.parametrize("completion_only_loss", [None, False])
@pytest.mark.parametrize("streamed", [False, True])
def test_prepare_ignores_completion_masks_the_trainer_will_not_apply(
    completion_only_loss, streamed
):
    # Studio leaves completion_only_loss unset and TRL resolves it False for input_ids rows, so
    # TRL never reads completion_mask: a stale one must not reject or alter a dataset that trains.
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4, 5, 6], [7, 8]], "completion_mask": [[0, 1], None]}
    )
    if streamed:
        ds = ds.to_iterable_dataset()
    out = prepare_pretokenized_dataset(
        ds, max_seq_length = 4, vocab_size = 100, completion_only_loss = completion_only_loss
    )
    rows = list(out)
    assert [r["input_ids"] for r in rows] == [[1, 2, 3, 4], [7, 8]]
    assert [r["completion_mask"] for r in rows] == [[0, 1], None]


def test_prepare_still_checks_assistant_masks_without_completion_only_loss():
    # TRL applies assistant_masks on presence alone.
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "assistant_masks": [[0, 1]]})
    with pytest.raises(ValueError, match = "'assistant_masks' must have the same length"):
        prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)


def test_prepare_keeps_hidden_columns_hidden():
    # A labels column hidden by set_format(columns = ...) is not the user's labels, and the
    # truncation restores the format so it stays hidden.
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6]], "labels": [[1, 2, 3, 4, 5, 6]]})
    ds.set_format(columns = ["input_ids"], output_all_columns = False)
    assert not has_custom_labels(ds)
    out = prepare_pretokenized_dataset(ds, max_seq_length = 4, vocab_size = 100)
    assert list(out[0]) == ["input_ids"]
    assert out[0]["input_ids"] == [1, 2, 3, 4]


def test_prepare_checks_and_truncates_masks_hidden_by_the_format():
    # TRL 0.22-0.26 re-exposes every column via with_transform, and 1.7+ builds labels with
    # map() over dataset.column_names, which ignore set_format: a hidden mask is still applied.
    bad = Dataset.from_dict({"input_ids": [[1, 2, 3]], "assistant_masks": [[0, 1]]})
    bad.set_format(columns = ["input_ids"], output_all_columns = False)
    with pytest.raises(ValueError, match = "'assistant_masks' must have the same length"):
        prepare_pretokenized_dataset(bad, max_seq_length = 2048, vocab_size = 100)
    ok = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4, 5, 6]], "assistant_masks": [[0, 1, 1, 1, 1, 1]]}
    )
    ok.set_format(columns = ["input_ids"], output_all_columns = False)
    out = prepare_pretokenized_dataset(ok, max_seq_length = 4, vocab_size = 100)
    assert list(out[0]) == ["input_ids"]
    assert out.data.column("assistant_masks").to_pylist() == [[0, 1, 1, 1]]


def test_completion_only_loss_resolves_from_the_train_split():
    # TRL resolves an unset completion_only_loss once, from the train split, and applies it to
    # eval too: an input_ids eval split's completion_mask is then applied, so it is checked.
    src = (_BACKEND_ROOT / "core" / "training" / "trainer.py").read_text(encoding = "utf-8")
    assert 'resolve_completion_only_loss(\n                    dataset["dataset"]' in src
    from utils.datasets.pretokenized import resolve_completion_only_loss

    train = Dataset.from_dict({"prompt": ["a"], "completion": ["b"], "input_ids": [[1, 2]]})
    evals = Dataset.from_dict({"input_ids": [[1, 2, 3]], "completion_mask": [[0, 1]]})
    assert resolve_completion_only_loss(train) is True
    assert resolve_completion_only_loss(evals) is False
    assert resolve_completion_only_loss(evals, True) is True
    with pytest.raises(ValueError, match = "'completion_mask' must have the same length"):
        prepare_pretokenized_dataset(
            evals,
            max_seq_length = 2048,
            vocab_size = 100,
            completion_only_loss = resolve_completion_only_loss(train),
        )


def test_prepare_checks_masks_the_format_still_outputs():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "assistant_masks": [[0, 1]]})
    ds.set_format(columns = ["input_ids"], output_all_columns = True)
    with pytest.raises(ValueError, match = "'assistant_masks' must have the same length"):
        prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)


@pytest.mark.parametrize("mask", [[None, None], [1, 0]])
def test_prepare_rejects_non_list_masks_with_the_mask_error(mask):
    # An all-null or scalar mask column has no list kernel in Arrow: still the mask error.
    ds = Dataset.from_dict({"input_ids": [[1, 2], [3]], "assistant_masks": mask})
    with pytest.raises(ValueError, match = "'assistant_masks' must have the same length"):
        prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)


# The torch formatter needs torch, which the CPU-only CI legs do not install.
_TORCH_FMT = pytest.param(
    "torch",
    marks = pytest.mark.skipif(
        importlib.util.find_spec("torch") is None, reason = "torch is not installed"
    ),
)


@pytest.mark.parametrize("fmt", [None, "numpy", _TORCH_FMT])
@pytest.mark.parametrize("mask", [0, 1.0, True])
def test_prepare_rejects_streamed_scalar_masks_with_the_mask_error(mask, fmt, monkeypatch):
    # Streamed rows skip the Arrow type check: a scalar mask must not surface as len()'s TypeError.
    monkeypatch.setattr(
        "datasets.config.TORCHVISION_AVAILABLE", False
    )  # torchvision without VideoReader
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3]], "assistant_masks": [mask]}
    ).to_iterable_dataset()
    if fmt:
        ds = ds.with_format(fmt)
    out = prepare_pretokenized_dataset(ds, max_seq_length = 2, vocab_size = 100)
    with pytest.raises(ValueError, match = "'assistant_masks' must have the same length"):
        list(out)


@pytest.mark.parametrize("fmt", ["numpy", _TORCH_FMT])
def test_prepare_truncates_streamed_array_masks(fmt, monkeypatch):
    monkeypatch.setattr("datasets.config.TORCHVISION_AVAILABLE", False)
    ds = (
        Dataset.from_dict({"input_ids": [[1, 2, 3]], "assistant_masks": [[0, 1, 1]]})
        .to_iterable_dataset()
        .with_format(fmt)
    )
    rows = list(prepare_pretokenized_dataset(ds, max_seq_length = 2, vocab_size = 100))
    assert [list(map(int, r["assistant_masks"])) for r in rows] == [[0, 1]]


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
        # Fractional values sit inside [0, 1], so only an exact {0, 1} check catches them.
        ([0.5, 1.0, 1.0, 1.0], "row 1 has 'attention_mask' values other than 0 and 1"),
        ([0.0, 0.5, 1.0, 1.0], "row 1 has 'attention_mask' values other than 0 and 1"),
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
@pytest.mark.parametrize("column", ["completion_mask", "assistant_masks", "token_type_ids"])
@pytest.mark.parametrize("with_mask", [True, False])
def test_prepare_refuses_an_unpadded_token_column_of_another_length(column, with_mask, streaming):
    # No zero in the mask (or no mask) means nothing is trimmed, but TRL still reads these columns per position.
    rows = {"input_ids": [[1, 2, 3], [4, 5]], column: [[0, 1, 1], [0, 1, 1]]}
    if with_mask:
        rows["attention_mask"] = [[1, 1, 1], [1, 1]]
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = f"'{column}' must have the same length"):
        # completion_mask is applied only under completion_only_loss.
        list(
            prepare_pretokenized_dataset(
                ds, max_seq_length = 2048, vocab_size = 100, completion_only_loss = True
            )
        )


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
    out = list(
        prepare_pretokenized_dataset(
            ds, max_seq_length = 2, vocab_size = 100, completion_only_loss = True
        )
    )
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


def test_mlx_training_prepares_eval_only_when_it_is_pretokenized():
    # A separately supplied eval split is formatted on its own and may resolve to text, with no input_ids to prepare.
    import ast

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mlx_training"
    )
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    call = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "prepare_pretokenized_dataset"
        and node.args[0].id == "eval_dataset"
    )
    guard = parents[call]
    while not isinstance(guard, ast.If):
        guard = parents[guard]
    assert "eval_final_format" in {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    # Set from the eval formatter's own result, not copied from the training split.
    assigned = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "eval_final_format" for t in node.targets)
    ]
    assert any(
        "ev" in {n.id for n in ast.walk(a.value) if isinstance(n, ast.Name)} for a in assigned
    )


def test_mlx_training_prepares_a_pretokenized_eval_split_whatever_the_train_format():
    # A text training split with a pre-tokenized eval split must still trim the eval rows' masked padding.
    import ast

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mlx_training"
    )
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    call = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "prepare_pretokenized_dataset"
        and node.args[0].id == "eval_dataset"
    )
    node = call
    while node is not fn:
        node = parents[node]
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            assert "dataset_final_format" not in names


def test_cuda_training_prepares_each_split_by_its_own_format():
    # A text train split with a pre-tokenized eval split still trims the eval padding, and a text eval split beside a
    # pre-tokenized train split is left alone (it has no input_ids to prepare).
    import ast

    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
            encoding = "utf-8"
        )
    )
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "prepare_pretokenized_dataset"
        and getattr(node.args[0], "id", None) == "eval_dataset"
    )
    guards, node = set(), call
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            guards |= {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    assert "eval_pretokenized" in guards
    assert not guards & {"pretokenized_mode", "dataset_final_format"}
    fn = node = call
    while not isinstance(fn, ast.FunctionDef):
        fn = parents[fn]
    flag = next(
        a
        for a in ast.walk(fn)
        if isinstance(a, ast.Assign) and getattr(a.targets[0], "id", None) == "eval_pretokenized"
    )
    assert "eval_final_format" in {n.id for n in ast.walk(flag.value) if isinstance(n, ast.Name)}
    # The separately formatted eval split records its own resolved format for the training step to read.
    recorded = [
        a
        for a in ast.walk(tree)
        if isinstance(a, ast.Assign)
        and isinstance(a.targets[0], ast.Subscript)
        and getattr(a.targets[0].slice, "value", None) == "eval_final_format"
    ]
    assert any(
        "eval_info" in {n.id for n in ast.walk(a.value) if isinstance(n, ast.Name)}
        for a in recorded
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_prepare_trims_boolean_masks_like_integer_masks(streaming):
    rows = {
        "input_ids": [[0, 2, 3, 0], [4, 5]],
        "attention_mask": [[False, True, True, False], [True, True]],
        "labels": [[-100, 2, 3, -100], [4, 5]],
    }
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    out = list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))
    assert out[0]["input_ids"] == [2, 3]
    assert out[0]["labels"] == [2, 3]
    assert out[0]["attention_mask"] == [True, True]
    assert out[1] == {k: v[1] for k, v in rows.items()}


@pytest.mark.parametrize("streaming", [False, True])
def test_prepare_refuses_interior_false_in_a_boolean_mask(streaming):
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[True, False, True]]})
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = "zeros between attended tokens"):
        list(prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100))


def test_prepare_leaves_an_all_true_boolean_mask_byte_identical():
    rows = {"input_ids": [[1, 2, 3]], "attention_mask": [[True, True, True]]}
    ds = Dataset.from_dict(rows)
    out = prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)
    assert out is ds
    assert out.data.table.equals(Dataset.from_dict(rows).data.table)


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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "rows, column",
    [
        ({"input_ids": [[1, 2], [3, 4]], "attention_mask": [[1, 1], [None, 1]]}, "attention_mask"),
        ({"input_ids": [[1, 2], [3, 4]], "labels": [[-100, 2], [None, 4]]}, "labels"),
    ],
)
def test_prepare_refuses_null_values_in_token_columns(rows, column, streaming):
    # Arrow's any / min_max skip nulls, so [None, 1] used to pass as an all-ones mask and TRL attended the hole.
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    with pytest.raises(ValueError, match = rf"row 1 .*null.*'{column}'"):
        list(prepare_pretokenized_dataset(ds, max_seq_length = 8))


def test_mlx_vocab_size_reads_the_input_embedding_table():
    from types import SimpleNamespace as NS

    from core.training.worker import _mlx_vocab_size

    weight = NS(shape = (32, 8))
    assert _mlx_vocab_size(NS(model = NS(embed_tokens = NS(weight = weight)))) == 32
    # QuantizedEmbedding packs its weight, so the stored row count wins over the packed shape.
    quant = NS(num_embeddings = 64, weight = NS(shape = (64, 1)))
    assert _mlx_vocab_size(NS(model = NS(embed_tokens = quant))) == 64
    vlm = NS(language_model = NS(model = NS(embed_tokens = NS(weight = weight))))
    assert _mlx_vocab_size(vlm) == 32
    lora = NS(embedding = NS(weight = weight))
    assert _mlx_vocab_size(NS(model = NS(embed_tokens = lora))) == 32
    assert _mlx_vocab_size(NS()) is None


def test_mlx_training_passes_the_model_vocab_size_to_both_splits():
    # Without it an out-of-range label reaches the MLX loss, where indexing the vocabulary fails mid-run.
    import ast

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mlx_training"
    )
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "prepare_pretokenized_dataset"
    ]
    assert len(calls) == 2
    for call in calls:
        vocab = next(k.value for k in call.keywords if k.arg == "vocab_size")
        assert isinstance(vocab, ast.Call) and vocab.func.id == "_mlx_vocab_size"
        assert [a.id for a in vocab.args] == ["model"]
        # zoo's MLX reader applies masks by its own policy, not the installed TRL's.
        assert any(k.arg == "mlx" and k.value.value is True for k in call.keywords)


def test_cuda_training_turns_eval_packing_off_for_a_pretokenized_eval_split():
    # TRL packs eval by `packing` unless eval_packing overrides it, and packing drops the eval split's labels.
    import ast

    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
            encoding = "utf-8"
        )
    )
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "eval_pretokenized"
    )
    assigned = [
        a
        for a in ast.walk(guard)
        if isinstance(a, ast.Assign)
        and isinstance(a.targets[0], ast.Subscript)
        and getattr(a.targets[0].value, "id", None) == "config_args"
        and getattr(a.targets[0].slice, "value", None) == "eval_packing"
    ]
    assert assigned and all(a.value.value is False for a in assigned)


def test_eval_packing_is_a_field_on_every_supported_trl():
    # eval_packing exists from TRL 0.22.2 (the floor) on, so setting it never breaks SFTConfig construction.
    SFTConfig = pytest.importorskip("trl").SFTConfig
    assert "eval_packing" in SFTConfig.__dataclass_fields__


@pytest.mark.parametrize("fmt", ["numpy", _TORCH_FMT])
@pytest.mark.parametrize("with_labels", [False, True])
def test_prepare_trims_formatted_streams(fmt, with_labels):
    # A formatted stream hands the mapper arrays, not lists.
    rows = {"input_ids": [[0, 2, 3], [4, 5, 6]], "attention_mask": [[0, 1, 1], [1, 1, 1]]}
    if with_labels:
        rows["labels"] = [[-100, 2, 3], [4, 5, 6]]
    ds = Dataset.from_dict(rows).to_iterable_dataset().with_format(fmt)
    out = list(prepare_pretokenized_dataset(ds, max_seq_length = 2, vocab_size = 100))
    assert [[int(x) for x in r["input_ids"]] for r in out] == [[2, 3], [4, 5]]
    assert [[int(x) for x in r["attention_mask"]] for r in out] == [[1, 1], [1, 1]]
    if with_labels:
        assert [[int(x) for x in r["labels"]] for r in out] == [[2, 3], [4, 5]]


@pytest.mark.parametrize("streaming", [False, True])
def test_mlx_trims_assistant_masks_beside_labels_whatever_the_trl(streaming, monkeypatch):
    # zoo's MLX reader applies assistant_masks beside labels and refuses a length mismatch, even
    # where TRL >= 1.7's collator would ignore them.
    import utils.datasets.pretokenized as pretokenized

    monkeypatch.setattr(pretokenized, "_collator_masks_user_labels", lambda: False)
    rows = {
        "input_ids": [[0, 0, 4, 5]],
        "attention_mask": [[0, 0, 1, 1]],
        "labels": [[-100, -100, 4, 5]],
        "assistant_masks": [[0, 0, 1, 1]],
        "completion_mask": [[0, 0, 1]],
    }
    ds = Dataset.from_dict(rows)
    if streaming:
        ds = ds.to_iterable_dataset()
    out = list(prepare_pretokenized_dataset(ds, max_seq_length = None, vocab_size = 100, mlx = True))
    assert out[0]["input_ids"] == [4, 5]
    assert out[0]["labels"] == [4, 5]
    assert out[0]["assistant_masks"] == [1, 1]
    # completion_mask applies on MLX only for an explicit completion_only_loss=True: left as is.
    assert out[0]["completion_mask"] == [0, 0, 1]
