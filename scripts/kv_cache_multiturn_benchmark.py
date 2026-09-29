"""Multi-turn generate(): re-encode the whole conversation vs continue from a KV cache of the history.

python scripts/kv_cache_multiturn_benchmark.py --model unsloth/Llama-3.2-1B-Instruct --turns 4 8 16 32
"""

import argparse
import time

from unsloth import FastLanguageModel
import torch


def timed(fn, runs):
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return out, sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--turns", type = int, nargs = "+", default = [4, 8, 16, 32])
    parser.add_argument("--max-new-tokens", type = int, default = 32)
    parser.add_argument("--runs", type = int, default = 5)
    parser.add_argument("--load-in-4bit", action = "store_true")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model, max_seq_length = 8192, load_in_4bit = args.load_in_4bit
    )
    FastLanguageModel.for_inference(model)
    gen = dict(
        max_new_tokens = args.max_new_tokens, min_new_tokens = args.max_new_tokens, do_sample = False
    )

    print(
        "| turns | history tok | new tok | full re-encode s | from cache s | speedup | tokens match |"
    )
    print("|---:|---:|---:|---:|---:|---:|:---:|")
    for turns in args.turns:
        history = []
        for i in range(turns):
            history.append(
                {
                    "role": "user",
                    "content": f"Fact {i}: item {i} costs {3 * i + 7} dollars. Please remember it.",
                }
            )
            history.append(
                {"role": "assistant", "content": f"Noted: item {i} costs {3 * i + 7} dollars."}
            )
        question = [{"role": "user", "content": "What does item 2 cost?"}]
        text_h = tokenizer.apply_chat_template(history, tokenize = False)
        text_f = tokenizer.apply_chat_template(
            history + question, tokenize = False, add_generation_prompt = True
        )
        enc = lambda t: tokenizer(t, return_tensors = "pt", add_special_tokens = False).to("cuda")
        hist, full = enc(text_h), enc(text_f)
        n, L = hist.input_ids.shape[1], full.input_ids.shape[1]
        if not torch.equal(full.input_ids[:, :n], hist.input_ids):
            raise SystemExit("chat template does not keep the history as a token prefix")
        with torch.no_grad():
            cache = model(**hist, use_cache = True).past_key_values

        model.generate(**full, max_new_tokens = 2)
        model.generate(**full, past_key_values = cache, max_new_tokens = 2)
        base, t_base = timed(lambda: model.generate(**full, **gen), args.runs)
        kv, t_kv = timed(lambda: model.generate(**full, past_key_values = cache, **gen), args.runs)
        match = torch.equal(base[0, L:], kv[0, L:])
        print(
            f"| {turns} | {n} | {L - n} | {t_base:.4f} | {t_kv:.4f} | {t_base / t_kv:.2f}x | {'YES' if match else 'NO'} |"
        )


if __name__ == "__main__":
    main()
