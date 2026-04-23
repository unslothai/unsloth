"""B=4 parity test so the captured graph bucket actually replays.

Compares flex captured-graph decode vs flex eager-decode (capture
disabled via ``UNSLOTH_FLEX_QWEN3_5_NO_CAPTURE=1``). Exact token-for-token
match is the success criterion — both paths use identical math, only
the kernel-launch wrapper differs.

    CUDA_VISIBLE_DEVICES=2 UNSLOTH_FAST_INFERENCE=1 PYTHONUNBUFFERED=1 \\
        python -u tests/qwen3_5_flex_parity_bs4.py \\
        --model Qwen/Qwen3.5-4B --max_new_tokens 16 \\
        --json_out async_task_outputs/qwen3_5/flex_parity_bs4.json
"""
import argparse
import json
import os
import time


CHAT_PROMPTS = [
    "In one sentence, what is Paris?",
    "What is 23 + 19? Answer in one word.",
    "Continue: The quick brown fox jumps over",
    "Name one primary color.",
]


def _run(model, tok, args, capture: bool):
    class SP:
        def __init__(self, n): self.max_tokens = n; self.temperature = 0.0

    sp = SP(args.max_new_tokens)
    rendered = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in CHAT_PROMPTS
    ]
    os.environ["UNSLOTH_FLEX_QWEN3_5_NO_CAPTURE"] = "0" if capture else "1"
    t0 = time.perf_counter()
    outs = model.fast_generate(rendered, sampling_params=sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    results = []
    for p, o in zip(CHAT_PROMPTS, outs):
        results.append({
            "prompt": p,
            "text": o.outputs[0].text,
            "ids": list(o.outputs[0].token_ids),
        })
    return dt, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--max_batch_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    import torch
    os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    import unsloth  # noqa
    from unsloth import FastLanguageModel

    model, tok_raw = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        fast_inference=True,
        max_batch_size=args.max_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tok = tok_raw.tokenizer if hasattr(tok_raw, "tokenizer") else tok_raw

    # Eager first (so the graph capture doesn't pollute state).
    dt_eager, eager = _run(model, tok, args, capture=False)
    print(f"[eager] gen_s={dt_eager:.2f}")
    for r in eager:
        print(f"[eager] {r['prompt'][:40]!r} -> {r['text'][:80]!r}")

    dt_cap, cap = _run(model, tok, args, capture=True)
    print(f"[cap]   gen_s={dt_cap:.2f}")
    for r in cap:
        print(f"[cap]   {r['prompt'][:40]!r} -> {r['text'][:80]!r}")

    # Token-by-token parity.
    mismatches = []
    for i, (e, c) in enumerate(zip(eager, cap)):
        matches = 0
        for a, b in zip(e["ids"], c["ids"]):
            if a == b:
                matches += 1
            else:
                break
        mismatches.append({
            "prompt_idx": i,
            "prompt": e["prompt"],
            "matches": matches,
            "total": min(len(e["ids"]), len(c["ids"])),
            "eager_text": e["text"],
            "cap_text": c["text"],
        })
        tag = "OK " if matches == min(len(e["ids"]), len(c["ids"])) else "MM "
        print(f"[{tag}] P{i} {matches}/{min(len(e['ids']), len(c['ids']))}")

    summary = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "eager_gen_s": round(dt_eager, 3),
        "capture_gen_s": round(dt_cap, 3),
        "mismatches": mismatches,
    }
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[parity-bs4] wrote {args.json_out}")


if __name__ == "__main__":
    main()
