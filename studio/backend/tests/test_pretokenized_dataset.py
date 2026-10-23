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


@pytest.mark.parametrize("fmt", [None, "numpy", "torch"])
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


@pytest.mark.parametrize("fmt", ["numpy", "torch"])
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
