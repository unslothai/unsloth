# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hash tokenizers for datasets cache names without transformers 5's init-only ``_vocab`` / ``_merges``.

They are read only in ``__init__`` to build the Rust ``_tokenizer``, which is hashed too and carries the
same vocabulary, so dropping them keeps the fingerprint tied to vocab, added tokens and chat template.
Only datasets' hashing Pickler is patched: multiprocess workers still receive the whole tokenizer.
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
        # Delegate so whatever datasets keeps out of the hash (encode cache, 4.7+ truncation / padding) stays out.
        held = {key: state.pop(key) for key in _INIT_ONLY_STATE if key in state}
        try:
            return original(pickler, obj)
        finally:
            state.update(held)

    setattr(_save_tokenizer, _MARKER, True)
    _save_tokenizer.__wrapped__ = original
    # datasets resolves the reducer by name on first sight of a tokenizer class, then caches it per class.
    _dill._save_transformersPreTrainedTokenizerBase = _save_tokenizer
    for obj_type, reducer in list(pickler.dispatch.items()):
        if reducer is original:
            pickler.dispatch[obj_type] = _save_tokenizer
    return True
