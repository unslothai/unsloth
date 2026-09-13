#!/usr/bin/env python3
"""Tests for unsloth.dataprep.supervision_audit, without heavy dependencies.

Every counter is checked against hand-built rows whose token ids are written
out explicitly, so an expectation here is a number you can recount by hand.
"""

import dataclasses
import io
import json
import sys
from types import SimpleNamespace

import pytest
from datasets import Dataset

from unsloth.dataprep.supervision_audit import SupervisionAuditReport, audit_supervision


# ── offline tokenizer ──────────────────────────────────────────────────────


class _Encoding(dict):
    """The two accessors callers use on a transformers BatchEncoding: `["input_ids"]` and `.input_ids`."""

    @property
    def input_ids(self):
        return self["input_ids"]


class MockTokenizer:
    """Word-level tokenizer: whitespace-separated words get ids in first-seen order.

    <pad> = 0, <s> = 1 (BOS), </s> = 2 (EOS). `__call__` mirrors the surface
    `train_on_responses_only` touches (`.input_ids` on the result), and `decode`
    joins words with single spaces, so a decoded segment is exactly predictable.
    """

    def __init__(self):
        self.pad_token, self.bos_token, self.eos_token = "<pad>", "<s>", "</s>"
        self.pad_token_id, self.bos_token_id, self.eos_token_id = 0, 1, 2
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
        self.words = {0: "<pad>", 1: "<s>", 2: "</s>"}

    def token_id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
            self.words[self.vocab[word]] = word
        return self.vocab[word]

    def ids(self, text):
        return [self.token_id(word) for word in text.split()]

    def __call__(self, text, add_special_tokens = False, return_tensors = None):
        ids = self.ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return _Encoding(input_ids = ids, attention_mask = [1] * len(ids))

    def decode(self, token_ids, skip_special_tokens = False):
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return " ".join(
            self.words[int(i)]
            for i in token_ids
            if not (skip_special_tokens and int(i) in specials)
        )


class _Tty(io.StringIO):
    def isatty(self):
        return True


class IndexableRows:
    """Rows reachable only through `len()` and integer indexing (no `.iter`, no column access)."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f"integer index expected, got {type(index).__name__}")
        if index < 0 or index >= len(self._rows):
            raise IndexError(index)
        return self._rows[index]


# ── hand-built rows ────────────────────────────────────────────────────────


def chat_row(tok, prompt, response):
    """`<s> user: {prompt} assistant: {response} </s>` with only the response and EOS supervised."""
    prompt_ids = tok.ids(f"<s> user: {prompt} assistant:")
    response_ids = tok.ids(f"{response} </s>")
    return prompt_ids + response_ids, [-100] * len(prompt_ids) + response_ids


def masked(row):
    input_ids, _ = row
    return input_ids, [-100] * len(input_ids)


def columns(rows):
    """(input_ids, labels) rows -> the column dict Dataset.from_dict takes."""
    return {"input_ids": [r[0] for r in rows], "labels": [r[1] for r in rows]}


def labels_rows(tok):
    """Five rows: 34 tokens, 9 supervised (2 / 0 / 4 / 0 / 3 per row), EOS in every row."""
    full = tok.ids("<s> Hello world </s>")
    return [
        chat_row(tok, "What is 2+2?", "4"),  # 8 tokens, 2 supervised
        masked(chat_row(tok, "Hi", "Hello")),  # 6 tokens, 0 supervised, EOS masked
        (full, list(full)),  # 4 tokens, all 4 supervised
        masked(chat_row(tok, "Name a color", "Blue")),  # 8 tokens, 0 supervised, EOS masked
        chat_row(tok, "Say hi", "hi there"),  # 8 tokens, 3 supervised
    ]


def padded_rows(tok):
    """Four right-padded rows: 12 non-padding tokens, 9 supervised, EOS reachable in only two."""
    a, b, c, d, e, g = (tok.token_id(w) for w in "a b c d e g".split())
    return [
        # 4 tokens, 3 supervised; the last non-padding token (EOS) is supervised
        {"input_ids": [1, a, b, 2, 0, 0], "attention_mask": [1, 1, 1, 1, 0, 0], "labels": [-100, a, b, 2, -100, -100]},
        # 3 tokens, all supervised; the label on the padded position must be ignored
        {"input_ids": [1, c, 2, 0], "attention_mask": [1, 1, 1, 0], "labels": [1, c, 2, 7]},
        # 3 tokens, 2 supervised; EOS sits in a masked-out position, so this row has no EOS
        {"input_ids": [1, d, e, 2], "attention_mask": [1, 1, 1, 0], "labels": [-100, d, e, -100]},
        # 2 tokens, 1 supervised; EOS masked out as well
        {"input_ids": [1, g, 2], "attention_mask": [1, 1, 0], "labels": [-100, g, -100]},
    ]


def example_line(lines, row):
    """The decoded text line under the `Example row {row}` header."""
    header = next(i for i, line in enumerate(lines) if line.startswith(f"  Example row {row} "))
    return lines[header + 1]


# ── labels path ────────────────────────────────────────────────────────────


def test_labels_path_counts_every_field():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    report = audit_supervision(ds, tokenizer = tok, verbose = False)

    assert isinstance(report, SupervisionAuditReport)
    assert report.num_rows == 5
    assert report.num_rows_total == 5
    assert report.labels_source == "labels"
    assert report.num_tokens == 34
    assert report.num_supervised_tokens == 9
    assert report.supervised_fraction == pytest.approx(9 / 34)
    assert report.supervised_tokens_per_row == pytest.approx({"min": 0, "median": 2, "mean": 1.8, "max": 4})
    assert report.zero_supervision_rows == 2
    assert report.zero_supervision_row_indices == [1, 3]
    assert report.fully_supervised_rows == 1
    assert report.max_seq_length is None
    assert report.rows_at_max_length is None
    assert report.rows_truncated_mid_response is None
    assert report.eos_token_id == 2
    assert report.rows_with_eos == 5
    assert report.rows_with_supervised_eos == 3
    assert report.bos_token_id == 1
    assert report.rows_with_duplicated_bos == 0
    assert len(report.examples) == 2
    assert all(isinstance(segment, tuple) for segment in report.examples[0])
    assert list(report.examples[0]) == [(False, "<s> user: What is 2+2? assistant:"), (True, "4 </s>")]
    assert list(report.examples[1]) == [(False, "<s> user: Hi assistant: Hello </s>")]
    assert report.warnings == [
        "2 rows (40.0%) have zero supervised tokens and contribute nothing to training.",
        "2 rows (40.0%) contain EOS only in masked (-100) positions, so the model does not learn to stop there.",
    ]
    assert report.ok is False


def test_same_rows_give_the_same_report_whatever_the_container():
    tok = MockTokenizer()
    rows = labels_rows(tok)
    as_dicts = [{"input_ids": ids, "labels": labels} for ids, labels in rows]

    expected = audit_supervision(Dataset.from_dict(columns(rows)), tokenizer = tok, verbose = False).to_dict()

    assert audit_supervision(as_dicts, tokenizer = tok, verbose = False).to_dict() == expected
    assert audit_supervision(IndexableRows(as_dicts), tokenizer = tok, verbose = False).to_dict() == expected
    assert expected["num_supervised_tokens"] == 9


def test_render_layout_without_color():
    tok = MockTokenizer()
    report = audit_supervision(Dataset.from_dict(columns(labels_rows(tok))), tokenizer = tok, verbose = False)

    rendered = report.render(color = False)
    lines = rendered.splitlines()

    assert lines[0] == "Unsloth: Supervision audit of 5 rows (labels from `labels`)"
    expected = [
        "  Supervised tokens: 9 of 34 (26.5%); per row min 0 / median 2 / max 4",
        "  Rows with zero supervised tokens: 2 (40.0%)",
        "  Rows fully supervised (no masking): 1 (20.0%)",
        "  Rows containing EOS: 5 (100.0%); rows with a supervised EOS: 3 (60.0%)",
        "  Rows starting with a duplicated BOS: 0 (0.0%)",
        "  Example row 0 (supervised text in [[double brackets]]):",
        "    <s> user: What is 2+2? assistant:[[4 </s>]]",
        "    <s> user: Hi assistant: Hello </s>",
        "  WARNING: 2 rows (40.0%) have zero supervised tokens and contribute nothing to training.",
        "  WARNING: 2 rows (40.0%) contain EOS only in masked (-100) positions, so the model does not learn to stop there.",
    ]
    for line in expected:
        assert line in lines, f"missing line {line!r} in:\n{rendered}"
    positions = [lines.index(line) for line in expected]
    assert positions == sorted(positions), rendered
    assert example_line(lines, 1) == "    <s> user: Hi assistant: Hello </s>"
    assert "Example row 2" not in rendered
    assert "max_seq_length" not in rendered
    assert "\x1b[" not in rendered
    assert lines[-1].startswith("  WARNING: ")
    assert lines[-2].startswith("  WARNING: ")


def test_render_color_modes(monkeypatch):
    tok = MockTokenizer()
    report = audit_supervision(Dataset.from_dict(columns(labels_rows(tok))), tokenizer = tok, verbose = False)

    ansi = report.render(color = True)
    assert "\x1b[" in ansi
    assert "[[" not in ansi
    assert "4 </s>" in ansi
    assert ansi.splitlines()[0] == "Unsloth: Supervision audit of 5 rows (labels from `labels`)"

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert report.render() == report.render(color = False)
    monkeypatch.setattr(sys, "stdout", _Tty())
    assert report.render() == ansi


def test_to_dict_is_json_serialisable():
    tok = MockTokenizer()
    report = audit_supervision(Dataset.from_dict(columns(labels_rows(tok))), tokenizer = tok, verbose = False)

    as_dict = report.to_dict()

    assert set(as_dict) == {field.name for field in dataclasses.fields(SupervisionAuditReport)}
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert as_dict["examples"][0] == [[False, "<s> user: What is 2+2? assistant:"], [True, "4 </s>"]]
    assert isinstance(as_dict["examples"][0][0], list)
    assert as_dict["num_supervised_tokens"] == 9
    assert as_dict["zero_supervision_row_indices"] == [1, 3]
    assert as_dict["rows_at_max_length"] is None


def test_verbose_controls_printing(capsys):
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    audit_supervision(ds, tokenizer = tok, verbose = False)
    assert capsys.readouterr().out == ""

    report = audit_supervision(ds, tokenizer = tok)
    out = capsys.readouterr().out
    assert out.startswith("Unsloth: Supervision audit of 5 rows")
    assert report.render(color = False) in out
    assert "WARNING: 2 rows (40.0%) have zero supervised tokens" in out


# ── other label sources ────────────────────────────────────────────────────


def test_completion_mask_path_and_labels_precedence():
    tok = MockTokenizer()
    ids0 = tok.ids("<s> user: Hi assistant: Hello </s>")
    ids1 = tok.ids("<s> user: Yo assistant: Hey </s>")
    rows = [
        {"input_ids": ids0, "completion_mask": [False, False, False, False, True, True]},
        {"input_ids": ids1, "completion_mask": [0, 0, 0, 0, 0, 0]},
    ]

    report = audit_supervision(rows, tokenizer = tok, verbose = False)

    assert report.labels_source == "completion_mask"
    assert report.num_tokens == 12
    assert report.num_supervised_tokens == 2
    assert report.zero_supervision_rows == 1
    assert report.zero_supervision_row_indices == [1]
    assert report.fully_supervised_rows == 0
    assert report.rows_with_eos == 2
    assert report.rows_with_supervised_eos == 1
    assert list(report.examples[0]) == [(False, "<s> user: Hi assistant:"), (True, "Hello </s>")]
    assert report.render(color = False).splitlines()[0] == (
        "Unsloth: Supervision audit of 2 rows (labels from `completion_mask`)"
    )

    both = Dataset.from_dict({"input_ids": [ids0], "labels": [ids0], "completion_mask": [[0] * 6]})
    report = audit_supervision(both, tokenizer = tok, verbose = False)
    assert report.labels_source == "labels"
    assert report.num_supervised_tokens == 6
    assert report.fully_supervised_rows == 1
    assert report.warnings == []


def test_no_labels_means_every_token_is_supervised():
    tok = MockTokenizer()
    ds = Dataset.from_dict({"input_ids": [tok.ids("<s> a b </s>"), tok.ids("<s> c </s>")]})

    report = audit_supervision(ds, verbose = False)

    assert report.labels_source == "all_tokens"
    assert report.num_tokens == 7
    assert report.num_supervised_tokens == 7
    assert report.supervised_fraction == 1.0
    assert report.fully_supervised_rows == report.num_rows == 2
    assert report.zero_supervision_rows == 0
    assert report.zero_supervision_row_indices == []
    assert report.supervised_tokens_per_row == pytest.approx({"min": 3, "median": 3.5, "mean": 3.5, "max": 4})
    assert report.eos_token_id is None
    assert report.rows_with_eos is None
    assert report.rows_with_supervised_eos is None
    assert report.bos_token_id is None
    assert report.rows_with_duplicated_bos is None
    assert report.examples == []
    assert report.warnings == []
    assert report.ok is True

    rendered = report.render(color = False)
    lines = rendered.splitlines()
    assert lines[0] == "Unsloth: Supervision audit of 2 rows (labels from `all_tokens`)"
    assert "  Rows fully supervised (no masking): 2 (100.0%)" in lines
    for absent in ("EOS", "BOS", "max_seq_length", "Example row", "WARNING"):
        assert absent not in rendered, rendered

    with_tok = audit_supervision(ds, tokenizer = tok, verbose = False)
    assert with_tok.warnings == []
    assert with_tok.rows_with_eos == 2
    assert with_tok.rows_with_supervised_eos == 2
    assert with_tok.rows_with_duplicated_bos == 0
    assert list(with_tok.examples[0]) == [(True, "<s> a b </s>")]
    assert "    [[<s> a b </s>]]" in with_tok.render(color = False).splitlines()


# ── padding, tensors ───────────────────────────────────────────────────────


def test_attention_mask_excludes_padding_everywhere():
    tok = MockTokenizer()

    report = audit_supervision(padded_rows(tok), tokenizer = tok, max_seq_length = 4, verbose = False)

    assert report.num_rows == 4
    assert report.num_tokens == 12
    assert report.num_supervised_tokens == 9
    assert report.supervised_tokens_per_row == pytest.approx({"min": 1, "median": 2.5, "mean": 2.25, "max": 3})
    assert report.fully_supervised_rows == 1
    assert report.zero_supervision_rows == 0
    # only row 0 has 4 non-padding tokens; its last real token (EOS) is supervised
    assert report.rows_at_max_length == 1
    assert report.rows_truncated_mid_response == 1
    # EOS inside a masked-out position does not count as present
    assert report.rows_with_eos == 2
    assert report.rows_with_supervised_eos == 2
    assert report.rows_with_duplicated_bos == 0
    assert list(report.examples[0]) == [(False, "<s>"), (True, "a b </s>")]
    assert list(report.examples[1]) == [(True, "<s> c </s>")]

    assert len(report.warnings) == 2
    truncated, no_eos = report.warnings
    assert truncated.startswith("1 row")
    assert "(25.0%)" in truncated
    assert "hit max_seq_length = 4 while still inside a supervised span" in truncated
    assert no_eos == (
        "2 rows (50.0%) do not contain the EOS token (id 2), so the model may never learn to stop generating."
    )
    lines = report.render(color = False).splitlines()
    assert "  Rows at max_seq_length = 4: 1 (25.0%), 1 cut off mid-response" in lines
    assert "  Rows containing EOS: 2 (50.0%); rows with a supervised EOS: 2 (50.0%)" in lines


def test_tensor_and_numpy_rows_match_plain_lists():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    tok = MockTokenizer()
    rows = padded_rows(tok)

    expected = audit_supervision(rows, tokenizer = tok, max_seq_length = 4, verbose = False).to_dict()
    assert expected["num_supervised_tokens"] == 9

    as_torch = [{key: torch.tensor(value) for key, value in row.items()} for row in rows]
    as_numpy = [{key: np.array(value) for key, value in row.items()} for row in rows]
    assert audit_supervision(as_torch, tokenizer = tok, max_seq_length = 4, verbose = False).to_dict() == expected
    assert audit_supervision(as_numpy, tokenizer = tok, max_seq_length = 4, verbose = False).to_dict() == expected

    # a torch-formatted padded Dataset hands `.iter()` 2-D tensors per column
    width = max(len(row["input_ids"]) for row in rows)
    square = {
        "input_ids": [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows],
        "attention_mask": [row["attention_mask"] + [0] * (width - len(row["attention_mask"])) for row in rows],
        "labels": [row["labels"] + [-100] * (width - len(row["labels"])) for row in rows],
    }
    formatted = Dataset.from_dict(square).with_format("torch")
    assert audit_supervision(formatted, tokenizer = tok, max_seq_length = 4, verbose = False).to_dict() == expected


# ── iteration, max_rows, trainers ──────────────────────────────────────────


def test_iterable_dataset_streams_and_stops_at_max_rows():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    report = audit_supervision(ds.to_iterable_dataset(), tokenizer = tok, max_rows = 2, verbose = False)

    assert report.num_rows == 2
    assert report.num_rows_total is None
    assert report.num_tokens == 14
    assert report.num_supervised_tokens == 2
    assert report.zero_supervision_row_indices == [1]
    assert report.render(color = False).splitlines()[0] == "Unsloth: Supervision audit of 2 rows (labels from `labels`)"

    everything = audit_supervision(ds.to_iterable_dataset(), tokenizer = tok, max_rows = None, verbose = False)
    sized = audit_supervision(ds, tokenizer = tok, verbose = False)
    assert everything.num_rows == 5
    assert everything.num_rows_total is None
    assert everything.to_dict() == {**sized.to_dict(), "num_rows_total": None}


def test_max_rows_smaller_than_dataset_reports_first_n_of_m():
    tok = MockTokenizer()
    rows = labels_rows(tok)
    ds = Dataset.from_dict(columns(rows))

    report = audit_supervision(ds, tokenizer = tok, max_rows = 2, verbose = False)

    assert report.num_rows == 2
    assert report.num_rows_total == 5
    assert report.num_tokens == 14
    assert report.num_supervised_tokens == 2
    assert report.render(color = False).splitlines()[0] == (
        "Unsloth: Supervision audit of the first 2 of 5 rows (labels from `labels`)"
    )

    as_dicts = [{"input_ids": ids, "labels": labels} for ids, labels in rows]
    partial = audit_supervision(as_dicts, tokenizer = tok, max_rows = 3, verbose = False)
    assert partial.num_rows == 3
    assert partial.num_rows_total == 5

    whole = audit_supervision(ds, tokenizer = tok, max_rows = 5, verbose = False)
    assert whole.num_rows == whole.num_rows_total == 5
    assert whole.render(color = False).splitlines()[0] == "Unsloth: Supervision audit of 5 rows (labels from `labels`)"

    many = [{"input_ids": [7, 8]} for _ in range(1200)]
    first_line = audit_supervision(many, max_rows = 1000, verbose = False).render(color = False).splitlines()[0]
    assert first_line == "Unsloth: Supervision audit of the first 1,000 of 1,200 rows (labels from `all_tokens`)"


def test_trainer_supplies_tokenizer_and_max_seq_length():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))
    trainer = SimpleNamespace(train_dataset = ds, processing_class = tok, args = SimpleNamespace(max_length = 8))

    report = audit_supervision(trainer, verbose = False)

    assert report.num_rows == 5
    assert report.num_supervised_tokens == 9
    assert report.max_seq_length == 8
    assert report.eos_token_id == 2
    assert report.bos_token_id == 1
    assert len(report.examples) == 2
    # rows 0, 3 and 4 have 8 tokens; the last token of rows 0 and 4 is a supervised EOS, row 3 is fully masked
    assert report.rows_at_max_length == 3
    assert report.rows_truncated_mid_response == 2
    assert report.warnings == [
        "2 rows (40.0%) have zero supervised tokens and contribute nothing to training.",
        "2 rows (40.0%) hit max_seq_length = 8 while still inside a supervised span, so their responses are cut off before the end (and before EOS).",
        "2 rows (40.0%) contain EOS only in masked (-100) positions, so the model does not learn to stop there.",
    ]
    assert "  Rows at max_seq_length = 8: 3 (60.0%), 2 cut off mid-response" in report.render(color = False).splitlines()

    explicit = audit_supervision(trainer, max_seq_length = 6, verbose = False)
    assert explicit.max_seq_length == 6
    # rows 0, 1, 3, 4 have >= 6 tokens; rows 1 and 3 are fully masked
    assert explicit.rows_at_max_length == 4
    assert explicit.rows_truncated_mid_response == 2


def test_trainer_tokenizer_and_length_fallbacks():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    old_style = SimpleNamespace(train_dataset = ds, tokenizer = tok, args = SimpleNamespace(max_seq_length = 8))
    report = audit_supervision(old_style, verbose = False)
    assert report.eos_token_id == 2
    assert report.max_seq_length == 8
    assert len(report.examples) == 2

    processor = SimpleNamespace(tokenizer = tok, image_processor = object())
    vlm = SimpleNamespace(
        train_dataset = ds,
        processing_class = processor,
        args = SimpleNamespace(max_length = None, max_seq_length = 6),
    )
    report = audit_supervision(vlm, verbose = False)
    assert report.eos_token_id == 2
    assert report.max_seq_length == 6
    assert list(report.examples[0]) == [(False, "<s> user: What is 2+2? assistant:"), (True, "4 </s>")]

    bare = SimpleNamespace(train_dataset = ds, processing_class = tok, args = SimpleNamespace())
    report = audit_supervision(bare, verbose = False)
    assert report.max_seq_length is None
    assert report.rows_at_max_length is None


# ── truncation, EOS, BOS ───────────────────────────────────────────────────


def test_truncation_counts_only_rows_whose_last_token_is_supervised():
    rows = [
        {"input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8]},  # at max, last token supervised
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, -100]},  # at max, last token masked
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},  # under max
        {"input_ids": [5, 6, 7, 8, 9], "labels": [-100, -100, -100, -100, 9]},  # over max, last token supervised
    ]

    report = audit_supervision(rows, max_seq_length = 4, verbose = False)

    assert report.max_seq_length == 4
    assert report.rows_at_max_length == 3
    assert report.rows_truncated_mid_response == 2
    assert report.zero_supervision_rows == 0
    assert report.warnings == [
        "2 rows (50.0%) hit max_seq_length = 4 while still inside a supervised span, so their responses are cut off before the end (and before EOS).",
    ]
    assert "  Rows at max_seq_length = 4: 3 (75.0%), 2 cut off mid-response" in report.render(color = False).splitlines()

    unlimited = audit_supervision(rows, verbose = False)
    assert unlimited.max_seq_length is None
    assert unlimited.rows_at_max_length is None
    assert unlimited.rows_truncated_mid_response is None
    assert unlimited.warnings == []
    assert "max_seq_length" not in unlimited.render(color = False)


def test_eos_presence_and_supervision():
    tok = MockTokenizer()
    a, b = tok.token_id("a"), tok.token_id("b")
    rows = [
        {"input_ids": [1, a, 2, b, 2], "labels": [-100, -100, -100, b, 2]},  # two EOS, only the second supervised
        {"input_ids": [1, a, b], "labels": [-100, a, b]},  # no EOS at all
        {"input_ids": [1, a, 2], "labels": [-100, a, -100]},  # EOS masked
        {"input_ids": [1, b, 2], "labels": [-100, -100, 2]},  # EOS supervised
    ]

    report = audit_supervision(rows, tokenizer = tok, verbose = False)

    assert report.rows_with_eos == 3
    assert report.rows_with_supervised_eos == 2
    assert report.zero_supervision_rows == 0
    assert len(report.warnings) == 2
    no_eos, masked_eos = report.warnings
    assert no_eos.startswith("1 row")
    assert "(25.0%)" in no_eos
    assert "do not contain the EOS token (id 2)" in no_eos
    assert masked_eos.startswith("1 row")
    assert "contain EOS only in masked (-100) positions" in masked_eos
    assert "  Rows containing EOS: 3 (75.0%); rows with a supervised EOS: 2 (50.0%)" in report.render(color = False).splitlines()


def test_duplicated_bos_uses_the_first_two_non_padding_tokens():
    tok = MockTokenizer()
    a = tok.token_id("a")
    rows = [
        {"input_ids": [1, 1, a, 2], "attention_mask": [1, 1, 1, 1]},  # duplicated
        {"input_ids": [1, a, 1, 2], "attention_mask": [1, 1, 1, 1]},  # two BOS, but not adjacent at the start
        {"input_ids": [0, 0, 1, 1, a, 2], "attention_mask": [0, 0, 1, 1, 1, 1]},  # left padded, duplicated
        {"input_ids": [1, a, 2], "attention_mask": [1, 1, 1]},  # single BOS
    ]

    report = audit_supervision(rows, tokenizer = tok, verbose = False)

    assert report.rows_with_duplicated_bos == 2
    assert report.warnings == [
        "2 rows (50.0%) start with two BOS tokens; your formatting adds BOS on top of the tokenizer's.",
    ]
    assert "  Rows starting with a duplicated BOS: 2 (50.0%)" in report.render(color = False).splitlines()


# ── examples ───────────────────────────────────────────────────────────────


def test_num_examples_controls_decoded_rows():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    none = audit_supervision(ds, tokenizer = tok, num_examples = 0, verbose = False)
    assert none.examples == []
    assert "Example row" not in none.render(color = False)
    assert none.num_supervised_tokens == 9

    assert len(audit_supervision(ds, tokenizer = tok, num_examples = 1, verbose = False).examples) == 1
    assert len(audit_supervision(ds, tokenizer = tok, num_examples = 50, verbose = False).examples) == 5
    assert len(audit_supervision(ds, tokenizer = tok, num_examples = 3, max_rows = 1, verbose = False).examples) == 1

    no_decode = SimpleNamespace(eos_token_id = 2, bos_token_id = None)
    report = audit_supervision(ds, tokenizer = no_decode, verbose = False)
    assert report.examples == []
    assert report.rows_with_eos == 5
    assert report.rows_with_supervised_eos == 3
    assert report.bos_token_id is None
    assert report.rows_with_duplicated_bos is None
    assert "BOS" not in report.render(color = False)

    with pytest.raises(ValueError):
        audit_supervision(ds, tokenizer = tok, num_examples = -1, verbose = False)


def test_render_escapes_control_characters_and_truncates_long_examples():
    tok = MockTokenizer()
    newline, tab, a, b = tok.token_id("\n"), tok.token_id("\t"), tok.token_id("a"), tok.token_id("b")
    short = {"input_ids": [1, a, newline, tab, b, 2], "labels": [-100, -100, -100, -100, b, 2]}
    long = {"input_ids": [1] + [a] * 1498 + [2], "labels": [1] + [a] * 1498 + [2]}
    tiny = {"input_ids": [1, b, 2], "labels": [-100, b, 2]}

    report = audit_supervision([short, long, tiny], tokenizer = tok, max_seq_length = 2048, num_examples = 3, verbose = False)

    rendered = report.render(color = False)
    lines = rendered.splitlines()
    assert lines[0] == "Unsloth: Supervision audit of 3 rows (labels from `labels`)"
    assert "  Supervised tokens: 1,504 of 1,509 (99.7%); per row min 2 / median 2 / max 1,500" in lines
    assert "  Rows at max_seq_length = 2,048: 0 (0.0%), 0 cut off mid-response" in lines

    # control characters are shown escaped, so the example stays on one line
    assert example_line(lines, 0) == "    <s> a \\n \\t[[b </s>]]"
    header = lines.index("  Example row 0 (supervised text in [[double brackets]]):")
    assert lines[header + 2].startswith("  Example row 1 ")

    # the 1,500-token row is cut at 400 characters with an ellipsis, but the report keeps the full text
    long_line = example_line(lines, 1)
    assert long_line.startswith("    [[<s> a a a")
    assert "…" in long_line[-3:], long_line[-20:]
    assert len(long_line) <= 420, len(long_line)
    assert list(report.examples[1]) == [(True, tok.decode(long["input_ids"]))]
    assert len(report.examples[1][0][1]) > 2000
    assert example_line(lines, 2) == "    <s>[[b </s>]]"


# ── edge cases ─────────────────────────────────────────────────────────────


def test_empty_dataset_warns_without_dividing_by_zero():
    tok = MockTokenizer()

    report = audit_supervision(Dataset.from_dict({"input_ids": []}), tokenizer = tok, verbose = False)

    assert report.num_rows == 0
    assert report.num_rows_total == 0
    assert report.num_tokens == 0
    assert report.num_supervised_tokens == 0
    assert report.supervised_fraction == 0.0
    assert report.supervised_tokens_per_row == {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    assert report.zero_supervision_rows == 0
    assert report.zero_supervision_row_indices == []
    assert report.fully_supervised_rows == 0
    assert report.rows_with_eos == 0
    assert report.rows_with_supervised_eos == 0
    assert report.rows_with_duplicated_bos == 0
    assert report.examples == []
    assert report.warnings == ["The dataset is empty."]
    assert report.ok is False

    lines = report.render(color = False).splitlines()
    assert lines[0] == "Unsloth: Supervision audit of 0 rows (labels from `all_tokens`)"
    assert "  WARNING: The dataset is empty." in lines
    json.dumps(report.to_dict())

    bounded = audit_supervision(Dataset.from_dict({"input_ids": []}), max_seq_length = 4, verbose = False)
    assert bounded.rows_at_max_length == 0
    assert bounded.rows_truncated_mid_response == 0
    assert bounded.warnings == ["The dataset is empty."]


def test_rows_of_length_zero_are_counted_but_contribute_no_tokens():
    tok = MockTokenizer()
    a = tok.token_id("a")
    rows = [
        {"input_ids": [], "labels": []},
        {"input_ids": [1, a, 2], "labels": [-100, a, 2]},
    ]

    report = audit_supervision(rows, tokenizer = tok, max_seq_length = 3, verbose = False)

    assert report.num_rows == 2
    assert report.num_tokens == 3
    assert report.num_supervised_tokens == 2
    assert report.zero_supervision_rows == 1
    assert report.zero_supervision_row_indices == [0]
    assert report.supervised_tokens_per_row["min"] == 0
    assert report.supervised_tokens_per_row["max"] == 2
    assert report.rows_with_eos == 1
    assert report.rows_with_supervised_eos == 1
    assert report.rows_with_duplicated_bos == 0
    assert report.rows_at_max_length == 1
    assert report.rows_truncated_mid_response == 1
    assert len(report.examples) == 2
    assert list(report.examples[0]) == []
    assert list(report.examples[1]) == [(False, "<s>"), (True, "a </s>")]
    assert "Example row 0" in report.render(color = False)
    assert audit_supervision(Dataset.from_dict(columns([(r["input_ids"], r["labels"]) for r in rows])),
                             tokenizer = tok, max_seq_length = 3, verbose = False).to_dict() == report.to_dict()

    only_empty = audit_supervision([{"input_ids": []}, {"input_ids": []}], verbose = False)
    assert only_empty.num_rows == 2
    assert only_empty.num_tokens == 0
    assert only_empty.supervised_fraction == 0.0
    assert only_empty.zero_supervision_rows == 2
    assert only_empty.warnings == [
        "All 2 rows have zero supervised tokens, so the training loss will be 0. "
        "If you used train_on_responses_only, its instruction_part / response_part do not match your chat template.",
    ]


def test_missing_input_ids_and_bad_arguments_raise_value_error():
    tok = MockTokenizer()
    ds = Dataset.from_dict(columns(labels_rows(tok)))

    with pytest.raises(ValueError) as excinfo:
        audit_supervision(Dataset.from_dict({"text": ["hello"]}), verbose = False)
    assert "Unsloth: audit_supervision needs an `input_ids` column" in str(excinfo.value)

    with pytest.raises(ValueError, match = "input_ids"):
        audit_supervision([{"labels": [1, 2]}], verbose = False)

    with pytest.raises(ValueError):
        audit_supervision(ds, max_rows = 0, verbose = False)
    with pytest.raises(ValueError):
        audit_supervision(ds, max_rows = -1, verbose = False)


# ── public surface ─────────────────────────────────────────────────────────


def test_public_exports():
    import unsloth.chat_templates as chat_templates
    import unsloth.dataprep as dataprep
    from unsloth.dataprep import supervision_audit as module

    assert module.__all__ == ["audit_supervision", "SupervisionAuditReport"]
    assert chat_templates.audit_supervision is audit_supervision
    assert dataprep.audit_supervision is audit_supervision
    assert dataprep.SupervisionAuditReport is SupervisionAuditReport
    names = chat_templates.__all__
    assert names.index("audit_supervision") == names.index("train_on_responses_only") + 1


# ── integration with the real masking ──────────────────────────────────────


def test_audits_labels_written_by_the_real_train_on_responses_only():
    """Drive unsloth_zoo's masking closure with the offline tokenizer, then audit its labels.

    `train_on_responses_only(None, tokenizer = ..., return_function = True)` only needs
    `tokenizer(text, add_special_tokens = False).input_ids`, so the word-level mock is
    enough to reproduce both the working case and the mismatched-marker case behind the
    all-labels-are-(-100) reports.
    """
    from unsloth.chat_templates import train_on_responses_only

    if train_on_responses_only is None:
        pytest.skip("train_on_responses_only is unavailable on this host (unsloth_zoo.dataset_utils needs torch)")

    tok = MockTokenizer()
    texts = [
        "<s> <user> What is 2+2? <assistant> 4 </s>",
        "<s> <user> Name a color <assistant> Blue </s>",
        "<s> <user> Say hi <assistant> hi there </s>",
    ]
    ds = Dataset.from_dict({"input_ids": [tok.ids(text) for text in texts]})

    mask = train_on_responses_only(
        None,
        tokenizer = tok,
        instruction_part = "<user>",
        response_part = "<assistant>",
        return_function = True,
    )
    labelled = ds.map(mask, batched = True)
    assert labelled[0]["labels"] == [-100] * 6 + tok.ids("4 </s>")

    report = audit_supervision(labelled, tokenizer = tok, verbose = False)
    assert report.labels_source == "labels"
    assert report.zero_supervision_rows == 0
    assert 0 < report.supervised_fraction < 1
    assert report.num_tokens == 24
    assert report.num_supervised_tokens == 7
    assert report.supervised_tokens_per_row == pytest.approx({"min": 2, "median": 2, "mean": 7 / 3, "max": 3})
    assert report.rows_with_eos == 3
    assert report.rows_with_supervised_eos == 3
    assert list(report.examples[0]) == [(False, "<s> <user> What is 2+2? <assistant>"), (True, "4 </s>")]
    assert report.warnings == []
    assert report.ok is True

    wrong = train_on_responses_only(
        None,
        tokenizer = tok,
        instruction_part = "[INST]",
        response_part = "[/INST]",
        return_function = True,
    )
    report = audit_supervision(ds.map(wrong, batched = True), tokenizer = tok, verbose = False)
    assert report.zero_supervision_rows == 3
    assert report.num_supervised_tokens == 0
    assert report.supervised_fraction == 0.0
    assert report.rows_with_eos == 3
    assert report.rows_with_supervised_eos == 0
    assert report.warnings == [
        "All 3 rows have zero supervised tokens, so the training loss will be 0. "
        "If you used train_on_responses_only, its instruction_part / response_part do not match your chat template.",
    ]
    assert report.ok is False
