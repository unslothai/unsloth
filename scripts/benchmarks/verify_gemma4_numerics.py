"""Compare first-token logits between `FlexGemma4Inference._prefill` and
vanilla `Gemma4ForCausalLM.forward` on the same prompt. Intended as a
one-shot correctness check; not part of the benchmark matrix.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/benchmarks/verify_gemma4_numerics.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gemma4_flex_inference import (  # noqa: E402
    FlexGemma4Inference,
    Sequence,
    _require_gemma4,
)


def main():
    Gemma4ForCausalLM, Gemma4Config, Gemma4TextConfig = _require_gemma4()
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    from transformers import AutoTokenizer

    name = "unsloth/gemma-4-E2B-it"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- load once (text shell) and keep a pristine copy for the vanilla pass
    full_cfg = Gemma4Config.from_pretrained(name)
    text_cfg = full_cfg.text_config

    full = Gemma4ForConditionalGeneration.from_pretrained(
        name, dtype = torch.bfloat16, attn_implementation = "eager"
    )
    lang = full.model.language_model
    full.model.vision_tower = None
    full.model.audio_tower = None
    full.model.embed_vision = None
    full.model.embed_audio = None

    base = Gemma4ForCausalLM(text_cfg)
    base.model = lang
    base.lm_head.weight = lang.embed_tokens.weight
    base = base.to(torch.bfloat16).to("cuda")
    base.eval()
    del full

    # Deep-copy so Flex's attention patching doesn't mutate the vanilla path.
    flex_model = copy.deepcopy(base)

    prompt = "The quick brown fox jumps over"
    ids = tok(prompt, return_tensors = "pt")["input_ids"].to("cuda")
    print(f"prompt len = {ids.shape[1]}")

    with torch.inference_mode():
        # `Gemma4ForCausalLM.forward` applies `final_logit_softcapping`
        # internally, so these logits are already softcapped.
        out = base(ids, use_cache = False)
        ref_logits = out.logits[0, -1, :].float()
    print(f"vanilla last-token logits: mean {ref_logits.mean().item():.4f}, "
          f"std {ref_logits.std().item():.4f}, "
          f"argmax {int(ref_logits.argmax())} "
          f"({tok.decode([int(ref_logits.argmax())])!r})")

    # Flex path.
    inf = FlexGemma4Inference(
        flex_model,
        tok,
        max_batch_size = 4,
        max_seq_length = 256,
        n_pages = 64,
        page_size = 64,
        max_new_tokens = 1,
        decode_kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16},
        prefill_kernel_options = {
            "FORCE_USE_FLEX_ATTENTION": True,
            "BLOCK_M": 32,
            "BLOCK_N": 32,
        },
        fa4_prefill = False,
    )
    seq = Sequence(text = prompt, max_new_tokens = 1)
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
    print(f"flex    last-token logits: mean {flex_logits.mean().item():.4f}, "
          f"std {flex_logits.std().item():.4f}, "
          f"argmax {int(flex_logits.argmax())} "
          f"({tok.decode([int(flex_logits.argmax())])!r})")

    diff = (flex_logits - ref_logits).abs()
    print(f"max abs diff      = {diff.max().item():.4e}")
    print(f"mean abs diff     = {diff.mean().item():.4e}")
    print(f"argmax match      = {int(ref_logits.argmax()) == int(flex_logits.argmax())}")
    # bf16 ULP is ~1e-2 at magnitude ~5. Report top-10 overlap too.
    top_ref = set(ref_logits.topk(10).indices.tolist())
    top_flex = set(flex_logits.topk(10).indices.tolist())
    print(f"top-10 overlap    = {len(top_ref & top_flex)} / 10")


if __name__ == "__main__":
    main()
