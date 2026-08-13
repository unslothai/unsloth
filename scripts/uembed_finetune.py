#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Fine-tune the UEmbed (Qwen3.5) multimodal DENSE + SPARSE embedding model with Unsloth.

Target checkpoint: ``Alibaba-NLP/UEmbed-2B`` (backbone ``model_type = qwen3_5``). The script
walks both halves of UEmbed support in order: a dense pass trained with
``MultipleNegativesRankingLoss``, then a short unified pass trained with
``UEmbedUnifiedLoss`` (dense InfoNCE + sparse InfoNCE + FLOPS), then an encode of the same
query as ``dense``, ``sparse`` and ``both``.

How the dense path differs from standard last-token pooling
------------------------------------------------------------
UEmbed does three extra things, and ``FastSentenceTransformer.from_pretrained`` wires all
three automatically when the checkpoint ships a ``sparse_info.json`` with
``num_eos_tokens > 0``:

1. Every input is wrapped in an instruction conversation - a system message carrying
   "Represent the user's input." plus a user message carrying ``video, image, text`` -
   and rendered through the processor's chat template before tokenization.
2. ``num_eos_tokens`` (16 for UEmbed) ``<|endoftext|>`` tokens are appended after the
   content by a ``tokenizers`` post-processor, and padding moves to the right.
3. The dense vector is pooled at ``last_index - num_eos_tokens`` instead of
   ``last_index``, i.e. the position that precedes that EOS block. That is what
   ``pooling_mode = "offset_lasttoken"`` selects; with ``num_eos_tokens = 0`` it is
   byte-for-byte plain ``lasttoken`` pooling, so nothing else regresses.

Pooling at the wrong offset silently returns an EOS filler position instead of the
sentence representation, which is why the gated parity test
(``tests/python/test_uembed_parity.py``) checks cosine >= 0.99 against the upstream
reference rather than merely checking that ``encode()`` returns a vector.

The sparse (SPLADE) path
------------------------
A UEmbed checkpoint ships ``sparse_weights.pt``, and ``from_pretrained`` appends the
``SpladeHead`` to the model's module chain. That head reads the hidden states the dense
pooling just consumed, so ONE forward pass produces ``sentence_embedding`` and
``sparse_embedding`` side by side in the features dict - the sparse half is never a second
forward.

Two consequences the script demonstrates:

- ``UEmbedUnifiedLoss`` reads both keys out of that one dict, so it can optimise the dense
  InfoNCE, the sparse InfoNCE (temperature ``tau_s = 32.0``, not the dense scale, because
  inner products over vocabulary space have a far wider range than cosines) and the FLOPS
  sparsity regulariser together. ``MultipleNegativesRankingLoss`` ignores the sparse
  vector entirely, which is why the dense-only pass leaves the SPLADE head untrained.
- ``model.encode(..., output_mode = "sparse" | "both")`` returns the sparse vector, or
  both vectors, from that same single forward. ``output_mode = "dense"`` (the default) is
  a plain pass-through to sentence-transformers, so nothing existing changes shape.

Backbone constraints (full list in ``unsloth/models/UEMBED_NOTES.md``)
---------------------------------------------------------------------
- bf16 only: ``qwen3_5`` is fp16-blocklisted, so fp16-only GPUs (e.g. T4) cannot run this.
- ``transformers >= 5.2`` (the UEmbed model card asks for ``>= 5.4``).
- ``trust_remote_code = True`` is required to load the checkpoint.
- ``config.json`` has no ``auto_map``, so loading routes through ``AutoModel`` to the base
  ``Qwen3_5Model`` (returns ``last_hidden_state``, which the pooling layer needs).
- Packing / padding-free training is disabled for this backbone (hybrid linear-attention
  GDN layers carry recurrent state across sequence boundaries).
- Save scope: LoRA and merged 16-bit only. GGUF / llama.cpp export is out of scope.

Also needs: a CUDA GPU with bf16 support, ``sentence-transformers >= 5.4``.

Run:
    python scripts/uembed_finetune.py
"""

from __future__ import annotations

# unsloth first: it patches transformers on import.
from unsloth import FastSentenceTransformer
from unsloth.models.uembed_loss import UEmbedUnifiedLoss

import os

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


MODEL_ID = "Alibaba-NLP/UEmbed-2B"

# Opt-in switch for the streamed science-tech slice in scripts/uembed_datasets.py. Unset
# (the default) keeps this script a self-contained smoke that downloads no dataset.
TRAIN_DATASET_ENV_VAR = "UNSLOTH_UEMBED_TRAIN_DATASET"

# One multimodal query, reused by every encode below.
EXAMPLE_QUERY = {
    "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
    "text": "kitten close-up",
}


def build_dataset() -> Dataset:
    """A tiny, self-contained (anchor, positive) contrastive dataset.

    Each column value is either a plain string or a ``{"image": ..., "text": ...}`` dict;
    ``image`` accepts a PIL image, a local path, or a URL. With
    MultipleNegativesRankingLoss every other row's positive acts as an in-batch negative,
    so no explicit negatives are needed. Two rows is enough to prove the multimodal path
    trains end to end - it is a smoke run, not a convergence run.

    -- SCIENCE-TECH DATASET (real convergence run) -------------------------------------
    The domain-adaptation run swaps this inline pair for a streamed science-tech slice,
    opt-in behind the ``UNSLOTH_UEMBED_TRAIN_DATASET`` environment variable:

        multimodal: vidore/arxivqa_test_subsampled  ->  {"anchor": {"text": question},
                                                         "positive": {"image": page_image}}
        text:       SciFact (e.g. allenai/scifact)  ->  {"anchor": {"text": claim},
                                                         "positive": {"text": abstract}}

    Both build the same ``{anchor, positive}`` shape this function returns, so the loss,
    trainer and collator below are unchanged. The loader lives in
    ``scripts/uembed_datasets.py`` (``load_uembed_science_tech_dataset``) and is called
    ONLY from ``main()``, only when that variable is set, so importing this script - or
    that loader - downloads nothing. The few-hundred-step convergence run (loss slope,
    sparsity, recall@k) is executed by Todo 11 on the Brev GPU box; this file deliberately
    keeps the tiny inline dataset as its default so ``python scripts/uembed_finetune.py``
    stays a self-contained smoke.
    ------------------------------------------------------------------------------------
    """
    return Dataset.from_list(
        [
            {
                "anchor": {
                    "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
                    "text": "a photo of a cat",
                },
                "positive": {"text": "a small domesticated feline resting"},
            },
            {
                "anchor": {
                    "image": "https://images.unsplash.com/photo-1543466835-00a7907e9de1",
                    "text": "a photo of a dog",
                },
                "positive": {"text": "a domesticated canine looking at the camera"},
            },
        ]
    )


def main() -> None:
    # 1. Load UEmbed. bf16 is not a preference here - qwen3_5 is fp16-blocklisted.
    #    pooling_mode="offset_lasttoken" selects the UEmbed dense pooling; from_pretrained
    #    reads num_eos_tokens from the checkpoint's sparse_info.json (16 for UEmbed) and
    #    also attaches the EOS post-processor and the instruction formatting.
    model = FastSentenceTransformer.from_pretrained(
        MODEL_ID,
        load_in_16bit = True,
        dtype = torch.bfloat16,
        pooling_mode = "offset_lasttoken",
        processor_kwargs = {"min_pixels": 28 * 28, "max_pixels": 600 * 600},
        trust_remote_code = True,  # required: no auto_map in config.json
    )

    # 2. Attach LoRA to the language tower only; the vision tower stays frozen (cheap,
    #    and the standard contrastive-tuning recipe for a VLM embedder).
    model = FastSentenceTransformer.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = None,
        finetune_vision_layers = False,
        finetune_language_layers = True,
    )

    # 3. Training data. The tiny inline pair is the default. The streamed science-tech
    #    slice is opt-in and is imported HERE rather than at module import, so no dataset
    #    is fetched unless UNSLOTH_UEMBED_TRAIN_DATASET is set. Running the file as
    #    `python scripts/uembed_finetune.py` puts scripts/ on sys.path, which is what makes
    #    the sibling import resolve.
    dataset = build_dataset()
    if os.environ.get(TRAIN_DATASET_ENV_VAR):
        from uembed_datasets import load_uembed_science_tech_dataset
        dataset = load_uembed_science_tech_dataset()
        print("science-tech subsets:", {name: len(rows) for name, rows in dataset.items()})

    # 4. Dense pass: MNRL == InfoNCE with in-batch negatives over the dense vectors only.
    #    It never reads `sparse_embedding`, so the SPLADE head stays untouched here.
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir = "outputs/uembed-dense",
        num_train_epochs = 1,
        per_device_train_batch_size = 2,
        learning_rate = 1e-4,
        bf16 = True,  # fp16 is blocklisted for qwen3_5
        batch_sampler = BatchSamplers.NO_DUPLICATES,  # avoid a positive as its own negative
        logging_steps = 1,
        report_to = "none",
    )

    trainer = SentenceTransformerTrainer(
        model = model,
        args = args,
        train_dataset = dataset,
        loss = loss,
    )
    trainer.train()

    # 5. Unified pass: the same batches, the same single forward, but now the loss also
    #    consumes `sparse_embedding` (dense InfoNCE + lambda * sparse InfoNCE + FLOPS).
    #    The defaults are the paper's: scale = 20.0 (tau_dense = 0.05), tau_s = 32.0,
    #    alpha_q = alpha_d = 0.01. Two steps is a smoke, not a convergence run - raising
    #    lambda_sparse or the alphas is what trades retrieval quality for sparsity.
    #    This requires a checkpoint that ships `sparse_weights.pt`; without one the loss
    #    raises and tells you to use MultipleNegativesRankingLoss instead.
    unified_loss = UEmbedUnifiedLoss(
        model,
        lambda_sparse = 1.0,
        alpha_q = 0.01,
        alpha_d = 0.01,
        scale = 20.0,
        tau_s = 32.0,
    )
    unified_args = SentenceTransformerTrainingArguments(
        output_dir = "outputs/uembed-unified",
        max_steps = 2,
        per_device_train_batch_size = 2,
        learning_rate = 1e-4,
        bf16 = True,
        batch_sampler = BatchSamplers.NO_DUPLICATES,
        logging_steps = 1,
        report_to = "none",
    )
    unified_trainer = SentenceTransformerTrainer(
        model = model,
        args = unified_args,
        train_dataset = dataset,
        loss = unified_loss,
    )
    unified_trainer.train()

    # 6. Encode a multimodal query. The instruction conversation and the trailing EOS block
    #    are applied inside encode(); pass raw content, not a pre-rendered prompt.
    query = model.encode(EXAMPLE_QUERY, normalize_embeddings = True)
    print("dense query embedding shape:", query.shape)

    #    output_mode = "sparse" returns the SPLADE vector from that same forward. It lives
    #    in vocabulary space and is non-negative, so the number that matters is how many
    #    terms survived; it is deliberately NOT normalized, which would flatten the term
    #    weights an inner-product retriever depends on.
    sparse_query = model.encode(EXAMPLE_QUERY, output_mode = "sparse")
    print(
        "sparse query embedding shape:", sparse_query.shape,
        "non-zero terms:", int((sparse_query > 0).sum()),
    )

    #    output_mode = "both" hands back both vectors of the ONE forward pass - the usual
    #    hybrid-retrieval setup, where the sparse vector filters and the dense one reranks.
    hybrid = model.encode(EXAMPLE_QUERY, output_mode = "both")
    print(
        "hybrid dense:", hybrid["sentence_embedding"].shape,
        "hybrid sparse:", hybrid["sparse_embedding"].shape,
    )

    # 7. Save merged bf16 weights (LoRA and merged_16bit are the supported save scopes).
    #    The SPLADE head is written beside them as UEmbed's `sparse_weights.pt` sidecar.
    model.save_pretrained_merged("outputs/uembed-dense-merged", save_method = "merged_16bit")


if __name__ == "__main__":
    main()
