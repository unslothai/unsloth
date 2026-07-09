# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Regression guard for issue #6881: FastSentenceTransformer must preprocess text
exactly like a stock SentenceTransformer for decoder embedding models.

sentence-transformers 5.x infers a "message" modality for models whose tokenizer
has a chat template (e.g. Qwen/Qwen3-Embedding), so building the module through
`Transformer(model_name, ...)` chat-wraps plain inputs and silently degrades
embeddings. `_create_transformer_module` builds through `Transformer.load(...)`
instead, matching stock ST.

Two layers:
  * test_transformer_load_signature_supports_unsloth_kwargs - fast, runs wherever
    sentence_transformers is importable; fails if ST drops the kwargs the fix passes.
  * test_fast_sentence_transformer_matches_stock_st - end-to-end parity; opt-in via
    UNSLOTH_EMBEDDING_PARITY_MODEL (a HF id or local path), so default CI is unaffected.
"""

from __future__ import annotations

import inspect
import os

import pytest


def test_transformer_load_signature_supports_unsloth_kwargs():
    """Forwards-compat tripwire on the installed ST: Transformer.load must stay a
    callable that accepts the hub kwargs the fix passes (or grow **kwargs)."""
    models = pytest.importorskip("sentence_transformers.models")
    load = getattr(models.Transformer, "load", None)
    assert callable(load), (
        "sentence_transformers Transformer.load is missing; the #6881 fix in "
        "unsloth.models.sentence_transformer._create_transformer_module depends on it."
    )
    params = inspect.signature(load).parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    unsupported = [
        k
        for k in ("token", "cache_folder", "revision", "trust_remote_code")
        if not (accepts_var_kw or k in params)
    ]
    assert not unsupported, (
        f"installed sentence_transformers Transformer.load no longer accepts {unsupported} "
        f"and has no **kwargs; update _create_transformer_module (#6881) before it silently "
        f"falls back to Transformer(...)."
    )


def _probe_texts():
    return [
        "roasted chickpeas in 20 kg bags",
        "The capital of France is Paris.",
        "A fast brown fox jumps over the lazy dog.",
        "recette de tarte aux pommes traditionnelle",
    ]


def test_fast_sentence_transformer_matches_stock_st():
    """End-to-end: FastSentenceTransformer embeddings and tokenization must match a
    stock SentenceTransformer load of the same checkpoint. Opt-in (needs a model)."""
    model_id = os.environ.get("UNSLOTH_EMBEDDING_PARITY_MODEL")
    if not model_id:
        pytest.skip(
            "set UNSLOTH_EMBEDDING_PARITY_MODEL to a chat-template embedding model "
            "(HF id or local path) to run the #6881 parity test"
        )

    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    texts = _probe_texts()
    max_seq_length = 256

    # Control FIRST, before importing unsloth, so its global import patches never
    # touch the stock reference (mirrors the issue's "restart runtime" repro).
    ctrl = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": dtype})
    ctrl.max_seq_length = max_seq_length
    ctrl_ids = ctrl.tokenize([texts[0]])["input_ids"][0].tolist()
    ctrl_emb = np.asarray(ctrl.encode(texts, normalize_embeddings=True, batch_size=8), dtype=np.float32)

    import unsloth  # noqa: F401
    from unsloth import FastSentenceTransformer

    fast = FastSentenceTransformer.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_16bit=use_cuda,
    )
    fast_ids = fast.tokenize([texts[0]])["input_ids"][0].tolist()
    fast_emb = np.asarray(fast.encode(texts, normalize_embeddings=True, batch_size=8), dtype=np.float32)

    # Identical tokenization = no chat-template wrapping slipped in (the #6881 defect).
    assert fast_ids == ctrl_ids, (
        f"tokenization diverged (chat-template wrapping regressed?):\n"
        f"  stock: {ctrl_ids}\n  fast:  {fast_ids}"
    )

    cos = (ctrl_emb * fast_emb).sum(1) / (
        np.linalg.norm(ctrl_emb, axis=1) * np.linalg.norm(fast_emb, axis=1)
    )
    assert float(cos.min()) > 0.99, (
        f"embedding parity regressed: min cosine {float(cos.min()):.5f} <= 0.99 "
        f"(per-text {[round(float(c), 5) for c in cos]})"
    )
