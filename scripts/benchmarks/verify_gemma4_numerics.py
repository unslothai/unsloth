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
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )
    from transformers import AutoTokenizer

    name = "unsloth/gemma-4-E2B-it"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    full_cfg = Gemma4Config.from_pretrained(name)
    text_cfg = full_cfg.text_config

    # Untouched HF reference: `Gemma4ForConditionalGeneration.forward` --
    # the same class everyone else would load via AutoModelForCausalLM
    # for Gemma-4. No patching, no shell, no flex attention.
    ref_raw = Gemma4ForConditionalGeneration.from_pretrained(
        name, dtype = torch.bfloat16, attn_implementation = "eager"
    ).to("cuda")
    ref_raw.eval()

    # Shell copy used by `gemma4_flex_inference.main()`: we deep-copy the
    # loaded multimodal model, drop the vision + audio towers, and move
    # the language_model into a Gemma4ForCausalLM wrapper so PEFT and
    # state-dict hashing treat it as a decoder-only model. The flex path
    # then patches its attention forwards on this shell. We keep the
    # shell around both as (a) the model that gets flex-patched and
    # (b) a sanity check that the shell itself matches the raw HF path.
    full = Gemma4ForConditionalGeneration.from_pretrained(
        name, dtype = torch.bfloat16, attn_implementation = "eager"
    )
    lang = full.model.language_model
    full.model.vision_tower = None
    full.model.audio_tower = None
    full.model.embed_vision = None
    full.model.embed_audio = None

    shell = Gemma4ForCausalLM(text_cfg)
    shell.model = lang
    shell.lm_head.weight = lang.embed_tokens.weight
    shell = shell.to(torch.bfloat16).to("cuda")
    shell.eval()
    del full

    # Deep-copy so Flex's attention patching doesn't mutate the shell.
    flex_model = copy.deepcopy(shell)

    prompt = "The quick brown fox jumps over"
    ids = tok(prompt, return_tensors = "pt")["input_ids"].to("cuda")
    print(f"prompt len = {ids.shape[1]}")

    with torch.inference_mode():
        # `Gemma4ForConditionalGeneration.forward` applies
        # `final_logit_softcapping` internally.
        ref_logits = ref_raw(input_ids = ids, use_cache = False).logits[0, -1, :].float()
        shell_logits = shell(ids, use_cache = False).logits[0, -1, :].float()
    print(
        f"raw   Gemma4ForConditionalGeneration: mean {ref_logits.mean():.4f}, "
        f"std {ref_logits.std():.4f}, argmax {int(ref_logits.argmax())} "
        f"({tok.decode([int(ref_logits.argmax())])!r})"
    )
    print(
        f"shell Gemma4ForCausalLM(text_cfg)    : mean {shell_logits.mean():.4f}, "
        f"std {shell_logits.std():.4f}, argmax {int(shell_logits.argmax())}"
    )
    # Dispose of the raw multimodal model before we build FlexGemma4Inference.
    del ref_raw
    torch.cuda.empty_cache()

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
    print(
        f"flex  last-token logits              : mean {flex_logits.mean():.4f}, "
        f"std {flex_logits.std():.4f}, argmax {int(flex_logits.argmax())} "
        f"({tok.decode([int(flex_logits.argmax())])!r})"
    )

    def report(tag, a, b):
        diff = (a - b).abs()
        top_a = set(a.topk(10).indices.tolist())
        top_b = set(b.topk(10).indices.tolist())
        print(
            f"  {tag:14s} max {diff.max().item():.3e}  mean {diff.mean().item():.3e}  "
            f"argmax={int(a.argmax()) == int(b.argmax())}  top-10={len(top_a & top_b)}/10"
        )

    print("vs raw Gemma4ForConditionalGeneration:")
    report("shell vs raw", shell_logits, ref_logits)
    report("flex  vs raw", flex_logits, ref_logits)
    print("vs shell (Gemma4ForCausalLM wrapper):")
    report("flex  vs shell", flex_logits, shell_logits)


if __name__ == "__main__":
    main()
