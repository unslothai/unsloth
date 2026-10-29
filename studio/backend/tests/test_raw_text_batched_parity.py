# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset, IterableDataset

import utils.datasets.raw_text as raw_text
from utils.datasets.raw_text import prepare_raw_text_dataset

EOS = "</s>"

TEXTS = [
    None,
    "",
    "   ",
    "\n\t \r\n",
    "\u3000",  # ideographic space: str.strip() removes it
    "\u200b",  # zero width space: str.strip() keeps it
    "plain",
    "  padded  ",
    "already" + EOS,
    "already" + EOS + "  ",
    EOS,
    "half</",
    "你好，世界",
    "café \U0001f600",
    "line one\nline two\n",
]


def _scalar_oracle(dataset, eos_token):
    """The per-row filter and EOS map this module used before batching."""
    dataset = dataset.filter(lambda ex: isinstance(ex["text"], str) and bool(ex["text"].strip()))

    def _append_eos(ex, _eos = eos_token):
        text = ex["text"]
        return {"text": text if text.endswith(_eos) else text + _eos}

    return dataset.map(_append_eos)


def _rows(texts):
    return [{"text": text, "id": index, "meta": f"m{index}"} for index, text in enumerate(texts)]


def _make(texts, streaming):
    rows = _rows(texts)
    if streaming:
        return IterableDataset.from_generator(lambda: iter(rows))
    return Dataset.from_list(rows)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("repeat", [1, 1500])  # 1500 crosses the 1000-row batch edge
def test_batched_raw_text_matches_scalar_oracle(streaming, repeat):
    texts = TEXTS * repeat
    result = prepare_raw_text_dataset(
        _make(texts, streaming),
        mode_label = "CPT",
        split_name = "train",
        eos_token = EOS,
        append_eos = True,
    )
    expected = _scalar_oracle(_make(texts, streaming), EOS)

    assert list(result.dataset) == list(expected)
    if not streaming:
        assert result.dataset.features == expected.features
        assert result.dataset.column_names == ["text", "id", "meta"]
        dropped = 5 * repeat  # None, "", "   ", mixed whitespace, ideographic space
        assert len(result.dataset) == len(texts) - dropped
        assert [n.message for n in result.notices] == [
            f"CPT: dropped {dropped:,} row(s) with null, non-string or blank 'text' values "
            "from the train split"
        ]


def test_non_string_text_rows_match_scalar_oracle():
    dataset = Dataset.from_dict({"text": [1, 2, 3], "id": [0, 1, 2]})
    with pytest.raises(ValueError, match = "all 3 rows were null, non-string or blank"):
        prepare_raw_text_dataset(dataset, split_name = "train", eos_token = EOS, append_eos = True)
    assert len(_scalar_oracle(dataset, EOS)) == 0

    mixed = [{"text": 5, "id": 0}, {"text": "ok", "id": 1}, {"text": ["x"], "id": 2}]
    streamed = IterableDataset.from_generator(lambda: iter(mixed))
    result = prepare_raw_text_dataset(streamed, eos_token = EOS, append_eos = True)
    assert list(result.dataset) == [{"text": "ok" + EOS, "id": 1}]


def test_large_string_text_column_keeps_its_type():
    from datasets import Features, Value

    features = Features({"text": Value("large_string"), "id": Value("int64")})
    dataset = Dataset.from_dict(
        {"text": ["a", None, "b" + EOS], "id": [0, 1, 2]}, features = features
    )
    result = prepare_raw_text_dataset(dataset, eos_token = EOS, append_eos = True)
    expected = _scalar_oracle(dataset, EOS)
    assert result.dataset.features == expected.features
    assert result.dataset.to_dict() == expected.to_dict()


def test_cache_fingerprint_is_deterministic():
    texts = TEXTS * 3
    first = prepare_raw_text_dataset(_make(texts, False), eos_token = EOS, append_eos = True)
    second = prepare_raw_text_dataset(_make(texts, False), eos_token = EOS, append_eos = True)
    assert first.dataset._fingerprint == second.dataset._fingerprint
    other_eos = prepare_raw_text_dataset(_make(texts, False), eos_token = "<eos>", append_eos = True)
    assert other_eos.dataset._fingerprint != first.dataset._fingerprint


@pytest.mark.parametrize("streaming", [False, True])
def test_map_style_filter_and_eos_passes_run_batched_over_the_text_column(streaming, monkeypatch):
    calls = []
    cls = IterableDataset if streaming else Dataset
    for name in ("filter", "map"):
        original = getattr(cls, name)

        def spy(
            self,
            *args,
            _name = name,
            _original = original,
            **kwargs,
        ):
            calls.append((_name, kwargs.get("batched"), kwargs.get("input_columns")))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, spy)

    raw_text.prepare_raw_text_dataset(_make(TEXTS, streaming), eos_token = EOS, append_eos = True)
    if streaming:
        # Streaming keeps the per-row passes: a batched filter + map pair iterates ~10x slower there.
        assert calls == [("filter", None, None), ("map", None, None)]
    else:
        assert ("filter", True, "text") in calls
        assert ("map", True, "text") in calls
