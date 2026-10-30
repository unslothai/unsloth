# Dataset checks

`unsloth dataset check` audits text training data without loading model weights. It uses
the training formatter, tokenizer and completion masker in a CPU subprocess.

```bash
unsloth dataset check ./train.jsonl --model unsloth/Qwen3-0.6B \
  --train-on-completions --max-seq-length 2048

unsloth dataset check --config train.yaml --full --json > dataset-report.json
```

The default checks the first 1,000 rows. Change that limit with `--max-rows`; the report
states whether the scan is complete. A prefix is not a representative sample or a
guarantee about later rows. `--full` loads a finite dataset and checks every row. Streaming
readers can fetch underlying files larger than the requested row count.

Inputs can be Hub dataset IDs, local JSON/JSONL, CSV, Parquet, or folders created by
`Dataset.save_to_disk`. Config files and their dataset, model, format, context length and
completion-only settings are shared with `unsloth train`. A positional dataset overrides
the configured source. A local tokenizer-only folder also works with `--model`.

Reports include the effective template, token-length distribution, formatter drop count,
overlength rows, and causal loss targets before and after truncation. Position zero of
each unpacked row is excluded from loss-target counts. Rows that lose all supervision to
truncation are distinguished from rows that already had none. Pre-tokenized custom labels
remain authoritative. Example row numbers refer to the prepared dataset, after filtering.
When chat rows also retain `input_ids`, those IDs take precedence over the formatted text,
as they do in training. The report identifies this route and its lack of extra truncation.
Tokenizer loading applies the training repairs, including reuse of an existing padding
token. If padding requires model preparation, the audit reports that it is unsupported.

This first version supports unpacked PyTorch causal language-model training. Apple Silicon's
MLX backend, packed configs and media datasets are refused because their training processors
and sequence boundaries require different statistics. It does not validate IDs against a
model's embedding table or run a training
step. `--format-type raw` selects raw-text CPT explicitly when automatic detection cannot
recognize the input. Malformed files fail at the reader; the audit does not skip bad JSON
lines or estimate how many malformed records remain unread.

Exit codes:

- `0`: no dropped or zero-loss rows in the checked portion. Overlength rows may still be reported.
- `1`: formatting dropped rows or some prepared rows have no causal loss targets.
- `2`: invalid input, unsupported configuration, or an audit failure.

`unsloth train --dry-run` continues to print configuration only.
