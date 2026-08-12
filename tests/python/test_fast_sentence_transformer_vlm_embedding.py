# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Tests for multimodal (vision) decoder embedding support in FastSentenceTransformer
(e.g. Qwen/Qwen3-VL-Embedding-2B).

Two layers:
- Fast, no-GPU unit tests for the VLM-detection helper and the get_peft_model vision
  flags. They import unsloth but never touch weights, so they run wherever unsloth is
  importable (skipped otherwise).
- An opt-in, GPU-only end-to-end parity test (UNSLOTH_VLM_EMBEDDING_PARITY_MODEL) that
  loads a real VLM embedding checkpoint and compares against a stock SentenceTransformer.

Design note: for a VLM embedding model, FastSentenceTransformer must keep
auto_model = AutoModel (the base model, e.g. Qwen3VLModel, returns last_hidden_state
which the Pooling layer needs); AutoModelForImageTextToText maps to
*ForConditionalGeneration, which returns logits and corrupts pooling. The image path is
enabled by swapping the text tokenizer for an AutoProcessor, not by changing the model
class.
"""

from __future__ import annotations

import inspect
import os
import types

import pytest


def _import_fast_sentence_transformer():
    """Import FastSentenceTransformer or skip. unsloth pulls in torch and may refuse to
    import without an accelerator; treat any import-time failure as a skip so these
    logic tests never turn a minimal runner red."""
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    try:
        from unsloth import FastSentenceTransformer
    except Exception as exc:  # noqa: BLE001 - unsloth may raise NotImplementedError on CPU-only
        pytest.skip(f"unsloth not importable in this environment: {exc}")
    return FastSentenceTransformer


def _cfg(**attrs):
    return types.SimpleNamespace(**attrs)


def test_is_vlm_embedding_config_detects_vision_config():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    detect = FastSentenceTransformer._is_vlm_embedding_config

    # A Qwen3-VL-Embedding-style config carries a nested vision_config.
    assert detect(_cfg(model_type="qwen3_vl", vision_config=_cfg(depth=24))) is True
    # Architecture-name fallback (no vision_config attribute surfaced).
    assert detect(_cfg(architectures=["Qwen3VLForConditionalGeneration"])) is True


def test_is_vlm_embedding_config_rejects_text_models():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    detect = FastSentenceTransformer._is_vlm_embedding_config

    assert detect(None) is False
    # A plain decoder text embedder (e.g. Qwen3-Embedding) must NOT be treated as VLM.
    assert detect(_cfg(model_type="qwen3", architectures=["Qwen3Model"])) is False
    assert detect(_cfg(model_type="bert", architectures=["BertModel"])) is False


def test_get_peft_model_exposes_vision_lora_flags():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    params = inspect.signature(FastSentenceTransformer.get_peft_model).parameters

    # Vision selectors must exist so VLM embedders can restrict LoRA to the language
    # tower (the cheap, common contrastive-tuning recipe) by default.
    assert params["finetune_vision_layers"].default is False
    assert params["finetune_language_layers"].default is True
    assert params["finetune_attention_modules"].default is True
    assert params["finetune_mlp_modules"].default is True
    assert params["finetune_last_n_layers"].default is None


def test_from_pretrained_accepts_processor_kwargs():
    FastSentenceTransformer = _import_fast_sentence_transformer()
    params = inspect.signature(FastSentenceTransformer.from_pretrained).parameters
    # min_pixels / max_pixels style image controls must be threadable to the processor.
    assert "processor_kwargs" in params
    assert params["processor_kwargs"].default is None


def test_vlm_embedding_matches_stock_st():
    """End-to-end parity: FastSentenceTransformer must match a stock SentenceTransformer
    load of the same VLM embedding checkpoint, and must NOT swap in a logits-returning
    *ForConditionalGeneration model. Opt-in + GPU-only, so default CI is unaffected."""
    model_id = os.environ.get("UNSLOTH_VLM_EMBEDDING_PARITY_MODEL")
    if not model_id:
        pytest.skip(
            "set UNSLOTH_VLM_EMBEDDING_PARITY_MODEL to a vision embedding model "
            "(e.g. Qwen/Qwen3-VL-Embedding-2B) to run the multimodal parity test"
        )

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FastSentenceTransformer requires CUDA; skipping on CPU-only runner")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    texts = ["a photo of a cat", "the capital of France is Paris"]

    # Control FIRST, before importing unsloth, so its global patches never touch the ref.
    ctrl = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": dtype})
    ctrl_emb = np.asarray(
        ctrl.encode(texts, normalize_embeddings=True, batch_size=2), dtype=np.float32
    )

    import unsloth  # noqa: F401
    from unsloth import FastSentenceTransformer

    fast = FastSentenceTransformer.from_pretrained(
        model_id, dtype=dtype, load_in_4bit=False, load_in_16bit=True,
    )

    # Must keep the base multimodal model (returns last_hidden_state), not the LM head.
    inner_name = fast[0].auto_model.__class__.__name__
    assert not inner_name.endswith("ForConditionalGeneration"), (
        f"FastSentenceTransformer loaded {inner_name}; a *ForConditionalGeneration model "
        f"returns logits and corrupts the Pooling layer. Expected the base model."
    )

    fast_emb = np.asarray(
        fast.encode(texts, normalize_embeddings=True, batch_size=2), dtype=np.float32
    )

    cos = (ctrl_emb * fast_emb).sum(1) / (
        np.linalg.norm(ctrl_emb, axis=1) * np.linalg.norm(fast_emb, axis=1)
    )
    assert float(cos.min()) > 0.99, (
        f"VLM embedding parity regressed: min cosine {float(cos.min()):.5f} <= 0.99"
    )
