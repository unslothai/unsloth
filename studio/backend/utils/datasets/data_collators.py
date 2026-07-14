# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom training data collators, particularly for VLM/OCR processing."""

from dataclasses import dataclass
from typing import Any, List, Optional, Union
from loggers import get_logger

logger = get_logger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper speech-to-text training.

    Pads audio input features and text labels separately, masks label padding
    with -100, and strips the leading BOS token. Mirrors the Whisper.ipynb
    notebook collator.
    """

    processor: Any

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors = "pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors = "pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass
class DeepSeekOCRDataCollator:
    """Data collator for DeepSeek OCR VLM training.

    Handles image processing, text tokenization, and label masking for
    instruction fine-tuning.
    """

    processor: Any  # Qwen2VLProcessor or similar
    max_length: int = 2048
    ignore_index: int = -100

    def __call__(self, batch: List[dict]) -> dict:
        """
        Collate a batch of samples.

        Args:
            batch: List of dicts, each with 'messages' containing
                   [{'role': 'user', 'content': [...]}, {'role': 'assistant', 'content': [...]}]

        Returns:
            dict with input_ids, attention_mask, labels, pixel_values, etc.
        """
        from PIL import Image

        all_messages = []
        all_images = []

        for sample in batch:
            messages = sample["messages"]
            all_messages.append(messages)

            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            img = item.get("image")
                            if img is not None and hasattr(img, "size"):  # PIL Image
                                all_images.append(img)

        try:
            texts = [
                self.processor.apply_chat_template(
                    msgs, tokenize = False, add_generation_prompt = False
                )
                for msgs in all_messages
            ]

            inputs = self.processor(
                text = texts,
                images = all_images if all_images else None,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = self.max_length,
            )

            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index

            inputs["labels"] = labels

            return inputs

        except Exception as e:
            logger.info(f"⚠️ DeepSeekOCRDataCollator error: {e}")
            raise


@dataclass
class VLMDataCollator:
    """Generic VLM data collator for various processors (Qwen2VL, LLaVA, etc.)."""

    processor: Any
    max_length: int = 2048
    ignore_index: int = -100
    mask_input_tokens: bool = True  # Mask user tokens in labels

    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of VLM samples."""
        all_messages = []
        all_images = []

        for sample in batch:
            messages = sample.get("messages", [])
            all_messages.append(messages)

            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            img = item.get("image")
                            if img is not None:
                                all_images.append(img)

        texts = [
            self.processor.apply_chat_template(msgs, tokenize = False, add_generation_prompt = False)
            for msgs in all_messages
        ]

        inputs = self.processor(
            text = texts,
            images = all_images if all_images else None,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = self.max_length,
        )

        labels = inputs["input_ids"].clone()

        # Mask padding.
        if hasattr(self.processor, "tokenizer"):
            pad_token_id = self.processor.tokenizer.pad_token_id
        else:
            pad_token_id = self.processor.pad_token_id

        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.ignore_index

        inputs["labels"] = labels

        return inputs
