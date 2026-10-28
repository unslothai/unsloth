# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cheaper datasets fingerprints for map functions that close over a tokenizer.

datasets names every map's cache file after a dill hash of the function, so a map closing over a
tokenizer pickles the whole tokenizer through dill's pure-Python pickler. Transformers 5 tokenizers
keep the ``vocab`` / ``merges`` they were built from as ``_vocab`` / ``_merges``: 151k entries each for
Qwen3, 262k for Gemma 3, read only in ``__init__`` to build the Rust ``_tokenizer``, which is pickled
too and carries the same vocabulary. Hashing them is most of the cost: 1.8 s per map for Qwen3,
6.3 s for Gemma 3, against 0.08 s and 0.8 s without them. The reducer below drops just those two
attributes from the hashed state, and only when a Rust tokenizer is present to stand for them, so
the fingerprint still changes with the vocabulary, added tokens and chat template. Tokenizers without
them (transformers 4.x, or ones loaded straight from tokenizer.json) hash exactly as before.

Only hashing changes: datasets registers the reducer on its own Pickler, which pickles nothing else, so
multiprocess workers still receive the whole tokenizer.
"""

_INIT_ONLY_STATE = ("_vocab", "_merges")
_MARKER = "_unsloth_studio_tokenizer_fingerprint"


def _is_rust_tokenizer(value) -> bool:
    from tokenizers import Tokenizer
    return isinstance(value, Tokenizer)


def install_tokenizer_fingerprint() -> bool:
    """Idempotent; False when this datasets has no tokenizer reducer to replace."""
    try:
        from datasets.utils import _dill
        import tokenizers  # noqa: F401
    except Exception:
        return False
    original = getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None)
    pickler = getattr(_dill, "Pickler", None)
    if original is None or pickler is None:
        return False
    if getattr(original, _MARKER, False):
        return True

    def _save_tokenizer(pickler, obj):
        state = obj.__dict__
        if not _is_rust_tokenizer(state.get("_tokenizer")) or not any(
            key in state for key in _INIT_ONLY_STATE
        ):
            return original(pickler, obj)
        # datasets' own reducer still does the hashing, so what it keeps out of the hash (the encode cache,
        # and from 4.7 the backend's truncation / padding and deprecation warnings) stays out; the
        # init-only copies are held off the object only while it runs.
        held = {key: state.pop(key) for key in _INIT_ONLY_STATE if key in state}
        try:
            return original(pickler, obj)
        finally:
            state.update(held)

    setattr(_save_tokenizer, _MARKER, True)
    _save_tokenizer.__wrapped__ = original
    # datasets looks the reducer up by name when it first meets a tokenizer class, and keeps it per class.
    _dill._save_transformersPreTrainedTokenizerBase = _save_tokenizer
    for obj_type, reducer in list(pickler.dispatch.items()):
        if reducer is original:
            pickler.dispatch[obj_type] = _save_tokenizer
    return True
