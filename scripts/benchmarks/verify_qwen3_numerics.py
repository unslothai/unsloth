"""Compare first-token logits between FlexInference._prefill (qwen3 path)
and vanilla model(input_ids) for Qwen3 and Llama-3.2.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/verify_qwen3_numerics.py \
        --model_name unsloth/Qwen3-4B-Base
    CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/verify_qwen3_numerics.py \
        --model_name unsloth/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen3_flex_inference import FlexInference, Sequence  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required = True)
    p.add_argument("--prompt", default = "The quick brown fox jumps over")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype = torch.bfloat16, attn_implementation = "eager"
    ).to("cuda")
    base.eval()

    flex_model = copy.deepcopy(base)

    ids = tok(args.prompt, return_tensors = "pt")["input_ids"].to("cuda")
    print(f"prompt len = {ids.shape[1]}")

    with torch.inference_mode():
        out = base(ids, use_cache = False)
        ref_logits = out.logits[0, -1, :].float()
    print(
        f"vanilla last-token logits: mean {ref_logits.mean().item():.4f}, "
        f"std {ref_logits.std().item():.4f}, argmax {int(ref_logits.argmax())} "
        f"({tok.decode([int(ref_logits.argmax())])!r})"
    )

    inf = FlexInference(
        flex_model,
        tok,
        max_batch_size = 4,
        max_seq_length = 256,
        n_pages = 64,
        page_size = 64,
        max_new_tokens = 1,
        fa4_prefill = False,
    )
    seq = Sequence(text = args.prompt, max_new_tokens = 1)
    inf.tokenize([seq])
    bi = inf.page_table.allocate()
    inf.page_table.reserve(
        bi,
        torch.tensor([bi], device = "cuda", dtype = torch.long),
        seq.total_length,
    )
    seq.batch_idx = bi
    with torch.inference_mode():
        flex_logits = inf._prefill([seq])[0].float()
    print(
        f"flex    last-token logits: mean {flex_logits.mean().item():.4f}, "
        f"std {flex_logits.std().item():.4f}, argmax {int(flex_logits.argmax())} "
        f"({tok.decode([int(flex_logits.argmax())])!r})"
    )

    diff = (flex_logits - ref_logits).abs()
    print(f"max abs diff   = {diff.max().item():.4e}")
    print(f"mean abs diff  = {diff.mean().item():.4e}")
    print(f"argmax match   = {int(ref_logits.argmax()) == int(flex_logits.argmax())}")
    top_ref = set(ref_logits.topk(10).indices.tolist())
    top_flex = set(flex_logits.topk(10).indices.tolist())
    print(f"top-10 overlap = {len(top_ref & top_flex)} / 10")


if __name__ == "__main__":
    main()
