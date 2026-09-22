"""End-to-end differential for unslothai/unsloth PR #11468.

Builds a tiny real Llama whose `forward` is narrowed the way
microsoft/Phi-4-reasoning-vision-15B's is -- no `**kwargs`, no
`packed_seq_lengths` -- and drives it through the real patched
`trl.SFTTrainer`. Padding-free is left at its default (None), so Unsloth
auto-enables it, which is the whole point: the user asked for nothing.

Run the same file from a `main` worktree and from the PR head worktree; the
arm is whichever `unsloth` is first on sys.path.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unsloth  # noqa: F401  (must precede transformers / trl)

import torch

import unsloth.trainer as _tm
_hits = []
_real_gate = _tm._forward_accepts_packing_kwargs
def _traced(model):
    out = _real_gate(model)
    _hits.append((type(model).__name__, out))
    return out
_tm._forward_accepts_packing_kwargs = _traced
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer


class NarrowForwardLlama(LlamaForCausalLM):
    """A forward that names neither `packed_seq_lengths` nor `**kwargs`."""

    def forward(self, input_ids = None, attention_mask = None, labels = None, position_ids = None):
        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            position_ids = position_ids,
        )


def main():
    print(f"arm: unsloth from {os.path.dirname(unsloth.__file__)}")

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size = tokenizer.vocab_size,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 512,
    )
    model = NarrowForwardLlama(config).to("cuda").to(torch.bfloat16)

    dataset = Dataset.from_dict({
        "text": [
            "The capital of France is Paris.",
            "Unsloth makes finetuning faster and it uses much less memory overall.",
            "Padding free batching flattens the batch.",
            "Short one.",
        ] * 4
    })

    args = SFTConfig(
        output_dir = "/tmp/repro_11468",
        max_steps = 2,
        per_device_train_batch_size = 2,
        max_length = 128,
        logging_steps = 1,
        report_to = [],
        bf16 = True,
        save_strategy = "no",
    )
    # padding_free deliberately left unset -> Unsloth auto-enables it.

    try:
        trainer = SFTTrainer(
            model = model,
            args = args,
            train_dataset = dataset,
            processing_class = tokenizer,
        )
    except Exception as exc:
        print(f"RESULT: construction raised {type(exc).__name__}: {exc}")
        return

    print(f"after init: args.padding_free={getattr(trainer.args, 'padding_free', None)!r} "
          f"args.packing={getattr(trainer.args, 'packing', None)!r}")

    try:
        out = trainer.train()
    except Exception as exc:
        print(f"RESULT: train() raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    print(f"GATE HITS: {_hits}")
    print(f"RESULT: trained OK, loss={out.training_loss:.4f}, steps={out.global_step}")


if __name__ == "__main__":
    main()
