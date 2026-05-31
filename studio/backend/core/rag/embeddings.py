# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG embedder singleton. Independent of the chat InferenceBackend."""

from __future__ import annotations

import logging
import threading
from typing import Any

from utils.rag.config import RAG_EMBED_BATCH_SIZE, RAG_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any | None = None
_model_name: str | None = None
_embedding_dim: int | None = None


def _load(model_name: str) -> Any:
    logger.info("Loading RAG embedder: %s", model_name)

    # BGE-VL's ST shim breaks across ST versions; load via AutoModel.
    if model_name.startswith("BAAI/BGE-VL"):
        return _BGEVLAdapter(model_name)

    from unsloth import FastSentenceTransformer

    # trust_remote_code: nomic-embed-text-v1.5 needs custom modeling for 8K ctx.
    return FastSentenceTransformer.from_pretrained(
        model_name,
        for_inference = True,
        trust_remote_code = True,
    )


class _BGEVLAdapter:
    """SentenceTransformer-shaped adapter over BGE-VL's AutoModel."""

    def __init__(self, hf_model_name: str):
        from transformers import AutoModel
        import torch

        self._model = AutoModel.from_pretrained(
            hf_model_name,
            trust_remote_code = True,
        )
        # Required: BGE-VL's encode() raises without an installed processor.
        self._model.set_processor(hf_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device).eval()
        self._device = device
        self._dim: int | None = None

    def _normalize(self, tensor):
        import torch.nn.functional as F

        return F.normalize(tensor, p = 2.0, dim = -1)

    # CLIP positional embedding cap; longer text triggers shape mismatch.
    _CLIP_TEXT_MAX_TOKENS = 77

    def encode(
        self,
        inputs,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_ignored,
    ):
        import io

        import numpy as np
        import torch
        from PIL import Image

        if inputs is None or len(inputs) == 0:
            return np.zeros(
                (0, self.get_sentence_embedding_dimension()), dtype = np.float32
            )

        sample = inputs[0]
        is_image = isinstance(sample, Image.Image) or isinstance(
            sample, (bytes, bytearray)
        )

        chunks_out = []
        for start in range(0, len(inputs), batch_size):
            batch = list(inputs[start : start + batch_size])
            if is_image:
                # BGE-VL's data_process re-opens each item via Image.open(...),
                # which needs a file-like (.read()) or path — NOT a pre-opened PIL
                # Image. Pass BytesIO; PIL Images get rebuffered via an in-memory PNG.
                file_likes: list[Any] = []
                for b in batch:
                    if isinstance(b, (bytes, bytearray)):
                        file_likes.append(io.BytesIO(b))
                    elif isinstance(b, Image.Image):
                        buf = io.BytesIO()
                        b.save(buf, format = "PNG")
                        buf.seek(0)
                        file_likes.append(buf)
                    else:
                        file_likes.append(b)
                with torch.no_grad():
                    vecs = self._model.encode(images = file_likes)
            else:
                vecs = self._encode_text_truncated([str(t) for t in batch])
            if normalize_embeddings:
                vecs = self._normalize(vecs)
            chunks_out.append(vecs.detach().cpu())

        out = torch.cat(chunks_out, dim = 0)
        return out.numpy() if convert_to_numpy else out

    def _encode_text_truncated(self, texts: list[str]):
        """Truncate to CLIP's 77-token limit; long text in multimodal mode is lossy."""
        import torch

        tokenizer = self._get_text_tokenizer()
        inputs = tokenizer(
            texts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = self._CLIP_TEXT_MAX_TOKENS,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if any(len(t.split()) > 30 for t in texts):
            logger.info(
                "BGE-VL text encode: truncating chunks to %d tokens (CLIP cap)",
                self._CLIP_TEXT_MAX_TOKENS,
            )
        with torch.no_grad():
            return self._model.get_text_features(**inputs)

    def _get_text_tokenizer(self):
        processor = getattr(self._model, "processor", None)
        if processor is not None:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                return tok
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None:
            return tok
        raise AttributeError("BGE-VL adapter could not locate a text tokenizer")

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            v = self.encode(["dim-probe"], batch_size = 1)
            self._dim = int(v.shape[-1])
        return self._dim

    def tokenize(self, texts):
        return self._get_text_tokenizer()(
            texts,
            return_tensors = "pt",
            padding = True,
        )


def get_embedder(model_name: str | None = None) -> Any:
    global _model, _model_name, _embedding_dim
    target = model_name or RAG_EMBEDDING_MODEL
    with _lock:
        if _model is None or _model_name != target:
            _model = _load(target)
            _model_name = target
            try:
                _embedding_dim = int(_model.get_sentence_embedding_dimension())
            except Exception:
                _embedding_dim = None
        return _model


def get_embedding_dim(model_name: str | None = None) -> int:
    model = get_embedder(model_name)
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = int(model.get_sentence_embedding_dimension())
    return _embedding_dim


def get_active_model_name() -> str | None:
    return _model_name


def encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
    normalize: bool = True,
):
    model = get_embedder(model_name)
    return model.encode(
        texts,
        batch_size = batch_size or RAG_EMBED_BATCH_SIZE,
        normalize_embeddings = normalize,
        convert_to_numpy = True,
        show_progress_bar = False,
    )


def encode_images(
    image_bytes_list: list[bytes],
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
    normalize: bool = True,
):
    """Embed image bytes via a CLIP-family multimodal encoder."""
    from io import BytesIO

    from PIL import Image

    if not image_bytes_list:
        return []
    model = get_embedder(model_name)
    images = [Image.open(BytesIO(b)).convert("RGB") for b in image_bytes_list]
    return model.encode(
        images,
        batch_size = batch_size or RAG_EMBED_BATCH_SIZE,
        normalize_embeddings = normalize,
        convert_to_numpy = True,
        show_progress_bar = False,
    )


def token_counter(model_name: str | None = None):
    """Return a token-count callable backed by the embedder's tokenizer."""
    model = get_embedder(model_name)

    def _count(text: str) -> int:
        try:
            tokens = model.tokenize([text])
            ids = tokens.get("input_ids")
            if ids is None:
                return max(1, len(text) // 4)
            return int(ids.shape[1])
        except Exception:
            return max(1, len(text) // 4)

    return _count

