"""Autotune replay for flex_attention decode.

Pattern from attention-gym/examples/flex_autotune_replay.py:

1. Run once with `mode="max-autotune-no-cudagraphs"` and
   `TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE` set. Inductor writes a JSON
   log of every kernel config it tried, sorted by wall time per shape.
2. Parse the log for the decode-shape entry (Q_LEN small, large KV).
3. Emit the best fwd_* options as a JSON string that the main flex script
   can accept via --decode_kernel_options.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmarks/flex_autotune_replay.py \
        --log_file logs/flex_autotune.json \
        --max_batch_size 64 \
        --n_prompts 16 \
        --max_new_tokens 64

Writes best decode kernel options to --output_opts (JSON), prints to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run_autotune_pass(log_file: str, args) -> None:
    env = os.environ.copy()
    env["FLEX_COMPILE_MODE"] = "max-autotune-no-cudagraphs"
    # Inductor appends `.json` to this env var value.
    env["TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE"] = log_file.replace(".json", "")

    cmd = [
        sys.executable,
        "-u",
        str(HERE / "qwen3_flex_inference.py"),
        "--n_prompts",
        str(args.n_prompts),
        "--n_rounds",
        "1",
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_batch_size",
        str(args.max_batch_size),
        # NB: autotune in `max-autotune-no-cudagraphs` mode is incompatible with
        # our raw CUDA graph capture path, so we skip --capture_cudagraph here.
        # Goal is only to produce the log, not to benchmark.
        "--stats_path",
        str(HERE / "logs" / "flex_autotune_stats.json"),
    ]
    if args.lora_adapter:
        cmd += ["--lora_adapter", args.lora_adapter]
    print("[autotune] running:", " ".join(cmd))
    print(f"[autotune] logging to {log_file}")
    subprocess.run(cmd, env = env, check = True)


class _SymStub:
    """Pretend-symbolic value so eval() can handle SymPy-ish free vars like `s40`."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Sym({self.name})"


class _SymNamespace(dict):
    """Any unknown name becomes a _SymStub instead of NameError."""

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        # Don't catch obvious builtins.
        if key in ("True", "False", "None"):
            return eval(key)
        return _SymStub(key)

    def __contains__(self, key):
        return True  # satisfies eval's name resolution


def parse_log(log_file: str) -> list[tuple[tuple, dict]]:
    """Return list of (shape_tuple, best_fwd_options_dict) per shape entry."""
    if not Path(log_file).exists():
        raise FileNotFoundError(
            f"Inductor log file missing: {log_file}. "
            f"Did `TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE` fire?"
        )
    with open(log_file) as f:
        data = json.load(f)

    ns = _SymNamespace()
    shapes = []
    for entry in data:
        key, choices = next(iter(entry.items()))
        try:
            parsed = eval(key, {"__builtins__": {}}, ns)
        except Exception:
            parsed = (key,)
        kernel_type = None
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            first = parsed[0]
            if isinstance(first, str):
                kernel_type = first
        best = choices[0]
        opts = {k: v for k, v in best.items() if k not in ("type", "time")}
        shapes.append((parsed, opts, best.get("time"), kernel_type, key))
    return shapes


def pick_decode_shape(shapes):
    """Pick the decode-shape entry.

    Decode has Q_LEN=1. Prefill has Q_LEN large. The shape tuple is
    `('forward', B, H_q, H_kv, Q_LEN, KV_LEN, D_q, D_v)` — so Q_LEN is at
    index 4. When B/KV_LEN are symbolic (`s0`, `s40`), the raw key string
    is the form `('forward', s40, 32, 8, 1, s0, 128, 128)`.
    """
    import re

    def q_len_of(shape_key, parsed):
        # If we successfully parsed and there's a real int at index 4, use it.
        if (
            isinstance(parsed, (list, tuple))
            and len(parsed) > 4
            and isinstance(parsed[4], int)
        ):
            return parsed[4]
        # Else extract from the raw string form, which is always
        # `('forward', <B>, 32, 8, <Q_LEN>, <KV_LEN>, 128, 128)`.
        m = re.match(r"\('forward',\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(\d+)", shape_key)
        if m:
            return int(m.group(1))
        return 10**9

    # (parsed, opts, time, kernel_type) -> plus we need the raw key string.
    # Pass shape_key via _SymNamespace too — actually we'll redo parse_log to
    # include the raw key. Simpler: re-read the file.
    return min(shapes, key = lambda s: q_len_of(s[4] if len(s) > 4 else "", s[0]))


def format_best_opts(best_opts: dict) -> dict:
    """Filter Inductor log keys to those acceptable to FlexKernelOptions as
    fwd_* prefix."""
    from torch.nn.attention.flex_attention import FlexKernelOptions

    annotations = FlexKernelOptions.__annotations__
    out = {}
    for k, v in best_opts.items():
        if k in annotations:
            out[f"fwd_{k}"] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log_file",
        default = "logs/flex_autotune.json",
        help = "Inductor writes the autotune log here. Will have .json appended.",
    )
    p.add_argument(
        "--output_opts",
        default = "logs/flex_best_decode_opts.json",
        help = "Extracted best kernel options go here.",
    )
    p.add_argument("--n_prompts", type = int, default = 16)
    p.add_argument("--max_batch_size", type = int, default = 64)
    p.add_argument("--max_new_tokens", type = int, default = 64)
    p.add_argument("--lora_adapter", default = None)
    p.add_argument(
        "--skip_autotune",
        action = "store_true",
        help = "Skip autotune pass and just parse existing log.",
    )
    args = p.parse_args()

    log_file = args.log_file
    if not log_file.endswith(".json"):
        log_file = log_file + ".json"

    if not args.skip_autotune:
        run_autotune_pass(log_file, args)

    shapes = parse_log(log_file)
    print(f"[autotune] parsed {len(shapes)} shapes from log:")
    for shape, opts, t, kt, key in shapes:
        print(f"  kernel_type={kt!r}  shape={shape}  time={t!r}  opts={opts}")
        print(f"    raw key: {key}")

    if not shapes:
        raise SystemExit("no shapes found in autotune log")

    decode_shape, decode_opts, decode_time, _, _ = pick_decode_shape(shapes)
    print("\n[autotune] selected decode-ish shape:", decode_shape)
    print("[autotune] best decode options:", decode_opts, f"(time={decode_time!r})")

    best = format_best_opts(decode_opts)
    # Always add tuned knobs we already confirmed helpful.
    best.setdefault("PRESCALE_QK", True)
    best.setdefault("USE_TMA", True)
    best.setdefault("BLOCKS_ARE_CONTIGUOUS", True)
    print("\n[autotune] final decode kernel_options:", json.dumps(best, indent = 2))

    Path(args.output_opts).parent.mkdir(parents = True, exist_ok = True)
    with open(args.output_opts, "w") as f:
        json.dump(best, f, indent = 2)
    print(f"[autotune] wrote {args.output_opts}")


if __name__ == "__main__":
    main()
