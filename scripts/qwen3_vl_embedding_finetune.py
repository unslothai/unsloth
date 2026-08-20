#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Fine-tune a multimodal (vision) decoder embedding model with Unsloth.

Example target: ``Qwen/Qwen3-VL-Embedding-2B`` (model_type ``qwen3_vl``, modules
``Transformer -> Pooling(last-token) -> Normalize``, 2048-dim). This answers the
feature request "Add Qwen3 VL embedding SFT notebook"
(https://github.com/unslothai/unsloth/issues/4481).

How it works
------------
``FastSentenceTransformer`` loads the base multimodal model via ``AutoModel`` (for
``qwen3_vl`` that is ``Qwen3VLModel``, which returns ``last_hidden_state`` for the
Pooling layer and accepts ``pixel_values`` / ``image_grid_thw``) and swaps in an
``AutoProcessor`` so images become ``pixel_values``. Training rides on the stock
``SentenceTransformerTrainer`` + ``MultipleNegativesRankingLoss`` (in-batch negatives,
i.e. InfoNCE); ``SentenceTransformerDataCollator`` hands each column to
``model.preprocess``, and the module's ``modality_config`` routes
``{"image": ..., "text": ...}`` through the processor, so no custom collator is needed.

Requirements: a CUDA GPU, ``sentence-transformers>=5.4``, ``transformers>=4.57``.

Run:
    python scripts/qwen3_vl_embedding_finetune.py
"""

from __future__ import annotations

from datasets import Dataset

from unsloth import FastSentenceTransformer
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"


def build_dataset() -> Dataset:
    """A tiny (anchor, positive) contrastive dataset.

    Each column value is either a plain string or a ``{"image": ..., "text": ...}``
    dict. ``image`` accepts a PIL image, a local path, or a URL. With
    MultipleNegativesRankingLoss, every other row's positive acts as an in-batch
    negative, so no explicit negatives are required. Replace these with your own
    image/text pairs (e.g. product photo -> description, screenshot -> query).
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
    # 1. Load the VLM embedding model. Keep load_in_16bit for encoder-style speed;
    #    processor_kwargs forwards image-resolution controls to the processor.
    model = FastSentenceTransformer.from_pretrained(
        MODEL_ID,
        load_in_16bit = True,
        pooling_mode = "lasttoken",  # decoder embedders pool the last token
        processor_kwargs = {"min_pixels": 28 * 28, "max_pixels": 600 * 600},
    )

    # 2. Attach LoRA. Default finetune_vision_layers=False keeps the vision tower
    #    frozen and tunes only the language projections (cheap, common recipe).
    model = FastSentenceTransformer.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        finetune_vision_layers = False,
        finetune_language_layers = True,
    )

    dataset = build_dataset()
    loss = MultipleNegativesRankingLoss(model)  # == InfoNCE with in-batch negatives

    args = SentenceTransformerTrainingArguments(
        output_dir = "outputs/qwen3-vl-embedding",
        num_train_epochs = 1,
        per_device_train_batch_size = 2,
        learning_rate = 1e-4,
        bf16 = True,
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

    # 3. Encode a multimodal query and save merged 16-bit weights.
    query = model.encode(
        {
            "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
            "text": "kitten close-up",
        },
        normalize_embeddings = True,
    )
    print("query embedding shape:", query.shape)  # -> (2048,)

    model.save_pretrained_merged("outputs/qwen3-vl-embedding-merged", save_method = "merged_16bit")


if __name__ == "__main__":
    main()
