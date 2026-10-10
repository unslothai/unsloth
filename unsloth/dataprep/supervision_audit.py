# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import dataclasses
from dataclasses import dataclass, field
from itertools import compress, groupby
from statistics import median
from typing import List, Dict, Any, Optional, Tuple

__all__ = [
    "audit_supervision",
    "SupervisionAuditReport",
]

# Label value that the loss ignores; train_on_responses_only writes it over every instruction token.
IGNORE_INDEX = -100

# Columns the audit reads; everything else is left untouched (and, on datasets.Dataset, not even decoded).
_COLUMNS = ("input_ids", "labels", "completion_mask", "assistant_masks", "attention_mask")
_BATCH_SIZE = 256
_ZERO_ROW_INDICES_KEPT = 20
_EXAMPLE_MAX_CHARS = 400
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"

_MISSING_INPUT_IDS = "Unsloth: audit_supervision needs an `input_ids` column – tokenize the dataset (or pass the trainer) first."


@dataclass
class SupervisionAuditReport:
    """What an SFT dataset actually supervises, as counted by audit_supervision"""

    num_rows: int
    num_rows_total: Optional[int]
    labels_source: str
    num_tokens: int
    num_supervised_tokens: int
    supervised_fraction: float
    supervised_tokens_per_row: Dict[str, float]
    zero_supervision_rows: int
    zero_supervision_row_indices: List[int]
    fully_supervised_rows: int
    max_seq_length: Optional[int]
    rows_at_max_length: Optional[int]
    rows_truncated_mid_response: Optional[int]
    eos_token_id: Optional[int]
    rows_with_eos: Optional[int]
    rows_with_supervised_eos: Optional[int]
    bos_token_id: Optional[int]
    rows_with_duplicated_bos: Optional[int]
    examples: List[List[Tuple[bool, str]]] = field(default_factory = list)
    warnings: List[str] = field(default_factory = list)

    @property
    def ok(self) -> bool:
        """True when the audit raised no warning"""
        return not self.warnings

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict; example segments become [is_supervised, text] lists"""
        result = dataclasses.asdict(self)
        result["examples"] = [
            [[bool(is_supervised), text] for is_supervised, text in segments]
            for segments in self.examples
        ]
        return result

    def render(self, color: Optional[bool] = None) -> str:
        """Human-readable report; color = None marks supervised text in green only on a TTY"""
        if color is None:
            color = _stdout_is_tty()
        n = self.num_rows
        if self.num_rows_total is not None and self.num_rows_total > n:
            scope = f"the first {n:,} of {self.num_rows_total:,} rows"
        else:
            scope = f"{n:,} rows"
        per_row = self.supervised_tokens_per_row
        lines = [
            f"Unsloth: Supervision audit of {scope} (labels from `{self.labels_source}`)",
            f"  Supervised tokens: {self.num_supervised_tokens:,} of {self.num_tokens:,} "
            f"({_pct(self.num_supervised_tokens, self.num_tokens)}); per row "
            f"min {_num(per_row.get('min', 0))} / median {_num(per_row.get('median', 0))} / "
            f"max {_num(per_row.get('max', 0))}",
            f"  Rows with zero supervised tokens: {self.zero_supervision_rows:,} "
            f"({_pct(self.zero_supervision_rows, n)})",
            f"  Rows fully supervised (no masking): {self.fully_supervised_rows:,} "
            f"({_pct(self.fully_supervised_rows, n)})",
        ]
        if self.max_seq_length is not None and self.rows_at_max_length is not None:
            lines.append(
                f"  Rows at max_seq_length = {self.max_seq_length:,}: {self.rows_at_max_length:,} "
                f"({_pct(self.rows_at_max_length, n)}), "
                f"{self.rows_truncated_mid_response or 0:,} cut off mid-response"
            )
        if self.rows_with_eos is not None:
            lines.append(
                f"  Rows containing EOS: {self.rows_with_eos:,} ({_pct(self.rows_with_eos, n)}); "
                f"rows with a supervised EOS: {self.rows_with_supervised_eos or 0:,} "
                f"({_pct(self.rows_with_supervised_eos or 0, n)})"
            )
        if self.rows_with_duplicated_bos is not None:
            lines.append(
                f"  Rows starting with a duplicated BOS: {self.rows_with_duplicated_bos:,} "
                f"({_pct(self.rows_with_duplicated_bos, n)})"
            )
        marker = "green" if color else "[[double brackets]]"
        for i, segments in enumerate(self.examples):
            lines.append(f"  Example row {i} (supervised text in {marker}):")
            lines.append("    " + _render_segments(segments, color))
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def audit_supervision(
    dataset,
    tokenizer = None,
    max_seq_length = None,
    completion_only_loss = None,
    max_rows = 10_000,
    num_examples = 2,
    verbose = True,
) -> SupervisionAuditReport:
    """Report what an SFT dataset actually supervises, before the first training step.

    Over the first `max_rows` rows it counts the non-padding tokens that carry a label, the rows that
    contribute no supervised token at all (the train_on_responses_only marker-mismatch
    symptom that otherwise surfaces as "All labels in your dataset are -100"), the rows
    that are fully supervised, the rows at `max_seq_length` whose response is cut off before
    its end (rows longer than the cap are audited cut to the first `max_seq_length` tokens, as
    the trainer prepares them), whether EOS is present and supervised, and whether BOS is
    duplicated.
    The first `num_examples` rows are decoded with their supervised spans marked.

    Typical use, right after masking the instruction tokens:

        from unsloth.chat_templates import train_on_responses_only, audit_supervision
        trainer = train_on_responses_only(trainer, instruction_part = ..., response_part = ...)
        report = audit_supervision(trainer)
        assert report.ok, report.warnings

    A token is supervised when every supervision signal the trainer applies keeps it, the same
    rule as `_supervision_columns` in unsloth/models/rl.py: `labels` (kept where `!= -100`)
    whenever the column exists, `assistant_masks` whenever the column exists, and
    `completion_mask` only under `completion_only_loss`. With a trainer that flag is read as
    the trainer resolved it (a None SFTConfig value means "the rows are prompt/completion");
    without one a present `completion_mask` is applied unless `completion_only_loss = False`.
    Without any signal every non-padding token is supervised.

    `dataset` may be a datasets.Dataset / IterableDataset, a list of row dicts, anything
    indexable with a length, or a trainer (its `train_dataset`, `processing_class`,
    `args.max_length` / `args.max_seq_length` and `completion_only_loss` are picked up).
    Read-only: nothing is modified, and nothing is printed unless `verbose` is True.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError(
            f"Unsloth: audit_supervision max_rows must be None or > 0, got {max_rows}."
        )
    if num_examples < 0:
        raise ValueError(
            f"Unsloth: audit_supervision num_examples must be >= 0, got {num_examples}."
        )

    if hasattr(dataset, "train_dataset"):
        trainer = dataset
        dataset = trainer.train_dataset
        if tokenizer is None:
            tokenizer = getattr(trainer, "processing_class", None)
            if tokenizer is None:
                tokenizer = getattr(trainer, "tokenizer", None)
        if max_seq_length is None:
            args = getattr(trainer, "args", None)
            max_seq_length = getattr(args, "max_length", None)
            if max_seq_length is None:
                max_seq_length = getattr(args, "max_seq_length", None)
        if completion_only_loss is None:
            completion_only_loss = _trainer_completion_only_loss(trainer)
    else:
        trainer = None

    tokenizer = _unwrap_tokenizer(tokenizer)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    decode = getattr(tokenizer, "decode", None) if num_examples > 0 else None

    num_rows_total = _num_rows_total(dataset)
    has_trainer = trainer is not None
    signals = _signals_from_columns(dataset, completion_only_loss, has_trainer)

    num_rows = 0
    num_tokens = 0
    num_supervised_tokens = 0
    supervised_per_row = []
    zero_supervision_rows = 0
    zero_supervision_row_indices = []
    fully_supervised_rows = 0
    rows_at_max_length = 0
    rows_truncated_mid_response = 0
    rows_with_eos = 0
    rows_with_supervised_eos = 0
    rows_with_duplicated_bos = 0
    examples = []

    for index, row in _iter_rows(dataset, max_rows):
        if signals is None:
            signals = _supervision_signals(row.keys(), completion_only_loss, has_trainer)
        input_ids = _to_list(row["input_ids"])
        # (name, column) per active signal; a row missing one of them is judged on the signals it has
        signal_columns = [
            (name, _to_list(row.get(name))) for name in signals if row.get(name) is not None
        ]
        attention_mask = _to_list(row.get("attention_mask"))

        n_all = len(input_ids)
        for name, column in signal_columns + [("attention_mask", attention_mask)]:
            if column is not None and len(column) != n_all:
                raise ValueError(
                    f"Unsloth: audit_supervision row {index} has {n_all} input_ids but "
                    f"{len(column)} {name} – the columns must be aligned token for token."
                )
        # Padding is ignored everywhere: token counts, the last token, EOS and BOS all look at real tokens
        # only. compress() is C-speed; the common unpadded row skips it.
        if attention_mask is not None and 0 in attention_mask:
            input_ids = list(compress(input_ids, attention_mask))
            signal_columns = [
                (name, list(compress(column, attention_mask))) for name, column in signal_columns
            ]

        # Rows longer than max_seq_length are audited as the trainer will see them: cut to the first
        # max_seq_length tokens, so a response that only starts past the cap counts as unsupervised.
        at_max_length = max_seq_length is not None and 0 < max_seq_length <= len(input_ids)
        if at_max_length and len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            signal_columns = [(name, column[:max_seq_length]) for name, column in signal_columns]

        n = len(input_ids)
        if not signal_columns:
            n_supervised = n
        elif len(signal_columns) == 1:
            # list.count() runs in C; a per-token Python comparison would dominate on 10k rows x 2k tokens.
            name, column = signal_columns[0]
            n_supervised = n - column.count(IGNORE_INDEX if name == "labels" else 0)
        else:
            # Several signals are intersected, as the trainer applies them one after another.
            n_supervised = sum(1 for i in range(n) if _is_supervised(signal_columns, i))

        num_rows += 1
        num_tokens += n
        num_supervised_tokens += n_supervised
        supervised_per_row.append(n_supervised)
        if n_supervised == 0:
            zero_supervision_rows += 1
            if len(zero_supervision_row_indices) < _ZERO_ROW_INDICES_KEPT:
                zero_supervision_row_indices.append(index)
        elif n_supervised == n:
            fully_supervised_rows += 1

        if at_max_length:
            rows_at_max_length += 1
            # A supervised final token means the response runs past the cap, unless that token is EOS:
            # a row ending exactly at the cap with a supervised EOS is complete, not cut off.
            ends_with_eos = eos_token_id is not None and input_ids[-1] == eos_token_id
            if not ends_with_eos and _is_supervised(signal_columns, n - 1):
                rows_truncated_mid_response += 1

        if eos_token_id is not None and eos_token_id in input_ids:
            rows_with_eos += 1
            if _any_occurrence_supervised(input_ids, signal_columns, eos_token_id):
                rows_with_supervised_eos += 1

        if (
            bos_token_id is not None
            and n >= 2
            and input_ids[0] == bos_token_id
            and input_ids[1] == bos_token_id
        ):
            rows_with_duplicated_bos += 1

        if decode is not None and len(examples) < num_examples:
            examples.append(_decode_segments(input_ids, signal_columns, decode))

    labels_source = " & ".join(signals) if signals else "all_tokens"

    if supervised_per_row:
        supervised_tokens_per_row = {
            "min": float(min(supervised_per_row)),
            "median": float(median(supervised_per_row)),
            "mean": num_supervised_tokens / num_rows,
            "max": float(max(supervised_per_row)),
        }
    else:
        supervised_tokens_per_row = {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}

    warnings = []
    if num_rows == 0:
        warnings.append("The dataset is empty.")
    elif zero_supervision_rows == num_rows:
        warnings.append(
            f"All {num_rows:,} rows have zero supervised tokens, so the training loss will be 0. "
            "If you used train_on_responses_only, its instruction_part / response_part do not match "
            "your chat template."
        )
    elif zero_supervision_rows > 0:
        warnings.append(
            f"{zero_supervision_rows:,} rows ({_pct(zero_supervision_rows, num_rows)}) have zero "
            "supervised tokens and contribute nothing to training."
        )
    if max_seq_length is not None and rows_truncated_mid_response > 0:
        warnings.append(
            f"{rows_truncated_mid_response:,} rows ({_pct(rows_truncated_mid_response, num_rows)}) hit "
            f"max_seq_length = {max_seq_length:,} while still inside a supervised span, so their responses "
            "are cut off before the end (and before EOS)."
        )
    if eos_token_id is not None and rows_with_eos < num_rows:
        missing = num_rows - rows_with_eos
        warnings.append(
            f"{missing:,} rows ({_pct(missing, num_rows)}) do not contain the EOS token (id {eos_token_id}), "
            "so the model may never learn to stop generating."
        )
    # Skipped when every row is already unsupervised: the all-rows warning above covers it.
    if (
        eos_token_id is not None
        and zero_supervision_rows < num_rows
        and rows_with_supervised_eos < rows_with_eos
    ):
        masked = rows_with_eos - rows_with_supervised_eos
        warnings.append(
            f"{masked:,} rows ({_pct(masked, num_rows)}) contain EOS only in masked (-100) positions, "
            "so the model does not learn to stop there."
        )
    if bos_token_id is not None and rows_with_duplicated_bos > 0:
        warnings.append(
            f"{rows_with_duplicated_bos:,} rows ({_pct(rows_with_duplicated_bos, num_rows)}) start with two "
            "BOS tokens; your formatting adds BOS on top of the tokenizer's."
        )

    report = SupervisionAuditReport(
        num_rows = num_rows,
        num_rows_total = num_rows_total,
        labels_source = labels_source,
        num_tokens = num_tokens,
        num_supervised_tokens = num_supervised_tokens,
        supervised_fraction = (num_supervised_tokens / num_tokens) if num_tokens else 0.0,
        supervised_tokens_per_row = supervised_tokens_per_row,
        zero_supervision_rows = zero_supervision_rows,
        zero_supervision_row_indices = zero_supervision_row_indices,
        fully_supervised_rows = fully_supervised_rows,
        max_seq_length = max_seq_length,
        rows_at_max_length = rows_at_max_length if max_seq_length is not None else None,
        rows_truncated_mid_response = rows_truncated_mid_response
        if max_seq_length is not None
        else None,
        eos_token_id = eos_token_id,
        rows_with_eos = rows_with_eos if eos_token_id is not None else None,
        rows_with_supervised_eos = rows_with_supervised_eos if eos_token_id is not None else None,
        bos_token_id = bos_token_id,
        rows_with_duplicated_bos = rows_with_duplicated_bos if bos_token_id is not None else None,
        examples = examples,
        warnings = warnings,
    )
    if verbose:
        print(report.render())
    return report


def _unwrap_tokenizer(tokenizer):
    """A VLM processor wraps the text tokenizer, which is the object that knows the token ids"""
    if tokenizer is None or hasattr(tokenizer, "eos_token_id"):
        return tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    return tokenizer if inner is None else inner


def _num_rows_total(dataset):
    """len(dataset), or None for streams (IterableDataset has no length)"""
    try:
        return len(dataset)
    except TypeError:
        return None


def _trainer_completion_only_loss(trainer):
    """The trainer's effective completion_only_loss, as the collator sees it: Unsloth's patched
    trainers record the resolved value on args, TRL resolves a None SFTConfig value from the
    dataset shape at construction, and the raw SFTConfig field is the last resort"""
    args = getattr(trainer, "args", None)
    resolved = getattr(args, "_unsloth_resolved_completion_only", None)
    if resolved is None:
        resolved = getattr(trainer, "completion_only_loss", None)
    if resolved is None:
        resolved = getattr(args, "_unsloth_completion_only_loss", None)
    if resolved is None:
        resolved = getattr(args, "completion_only_loss", None)
    return resolved


def _supervision_signals(column_names, completion_only_loss, has_trainer):
    """The columns that decide whether a token is supervised, in the order the trainer applies them.

    Mirrors _supervision_columns in unsloth/models/rl.py: `labels` and `assistant_masks` count on
    presence, `completion_mask` only under completion_only_loss. A None flag is resolved the way TRL
    does it, from prompt/completion rows; when there is no trainer to resolve it against, a present
    completion_mask is applied, since that is the only reason the column exists.
    """
    column_names = set(column_names)
    signals = []
    if "labels" in column_names:
        signals.append("labels")
    if "completion_mask" in column_names:
        if completion_only_loss is None:
            prompt_completion = "prompt" in column_names and "completion" in column_names
            completion_only_loss = prompt_completion or not has_trainer
        if completion_only_loss:
            signals.append("completion_mask")
    if "assistant_masks" in column_names:
        signals.append("assistant_masks")
    return signals


def _signals_from_columns(dataset, completion_only_loss, has_trainer):
    """Supervision signals from the schema when there is one; None defers to the first row"""
    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, (list, tuple)):
        return None
    if "input_ids" not in column_names:
        raise ValueError(_MISSING_INPUT_IDS)
    return _supervision_signals(column_names, completion_only_loss, has_trainer)


def _iter_rows(dataset, max_rows):
    """Yield (index, {column: value}) for the first `max_rows` rows, streaming where the dataset can.

    Dataset.iter() hands out Arrow batches instead of copying whole columns into Python
    objects (see _iter_column in raw_text.py); select_columns() first, so a chat dataset's
    `messages` / `text` columns are never decoded. Anything without iter() (lists, custom
    __getitem__) is indexed or iterated row by row.
    """
    column_names = getattr(dataset, "column_names", None)
    select_columns = getattr(dataset, "select_columns", None)
    if isinstance(column_names, (list, tuple)) and callable(select_columns):
        dataset = select_columns([column for column in _COLUMNS if column in column_names])

    index = 0
    batched = getattr(dataset, "iter", None)
    if callable(batched):
        for batch in batched(batch_size = _BATCH_SIZE):
            if "input_ids" not in batch:
                raise ValueError(_MISSING_INPUT_IDS)
            present = [column for column in _COLUMNS if column in batch]
            for i in range(len(batch["input_ids"])):
                yield index, {column: batch[column][i] for column in present}
                index += 1
                if max_rows is not None and index >= max_rows:
                    return
        return

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        rows = (dataset[i] for i in range(len(dataset)))
    else:
        rows = iter(dataset)
    for row in rows:
        yield index, _row_columns(row)
        index += 1
        if max_rows is not None and index >= max_rows:
            return


def _row_columns(row):
    """The audited columns of one row as a plain dict"""
    result = {}
    for column in _COLUMNS:
        try:
            result[column] = row[column]
        except (KeyError, IndexError, TypeError):
            continue
    if "input_ids" not in result:
        raise ValueError(_MISSING_INPUT_IDS)
    return result


def _to_list(value):
    """Plain Python list from a list, tensor, numpy array or any sequence; None stays None"""
    if value is None or isinstance(value, list):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def _is_supervised(signal_columns, i):
    """Whether token i is kept by every active supervision signal"""
    for name, column in signal_columns:
        value = column[i]
        if (value == IGNORE_INDEX) if name == "labels" else not value:
            return False
    return True


def _any_occurrence_supervised(input_ids, signal_columns, token_id):
    """Whether at least one occurrence of token_id is supervised (list.index is C-speed per hit)"""
    if not signal_columns:
        return True
    start = 0
    while True:
        try:
            i = input_ids.index(token_id, start)
        except ValueError:
            return False
        if _is_supervised(signal_columns, i):
            return True
        start = i + 1


def _decode_segments(input_ids, signal_columns, decode):
    """Decode a row as (is_supervised, text) runs, consecutive tokens with the same status merged"""
    flags = [_is_supervised(signal_columns, i) for i in range(len(input_ids))]
    segments = []
    for is_supervised, group in groupby(zip(input_ids, flags), key = lambda pair: pair[1]):
        text = decode([token for token, _ in group], skip_special_tokens = False)
        segments.append((bool(is_supervised), str(text)))
    return segments


def _render_segments(
    segments,
    color,
    limit = _EXAMPLE_MAX_CHARS,
):
    """One example row as a single escaped line, supervised runs marked, cut at `limit` visible characters"""
    open_marker, close_marker = (_ANSI_GREEN, _ANSI_RESET) if color else ("[[", "]]")
    parts = []
    remaining = limit
    truncated = False
    for is_supervised, text in segments:
        text = _escape(text)
        if not text:
            continue
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        remaining -= len(text)
        parts.append(open_marker + text + close_marker if is_supervised else text)
    line = "".join(parts)
    if truncated:
        line += "…"
    return line


def _escape(text):
    """Show newlines and tabs as \\n and \\t so one example stays on one line"""
    return text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _pct(count, total):
    """`31.2%`; 0.0% when there is nothing to divide by"""
    if not total:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def _num(value):
    """Thousands-separated number; whole values print without decimals"""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _stdout_is_tty():
    """Whether stdout is a terminal (stdout can be None or replaced in notebooks and tests)"""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False
