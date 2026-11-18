# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hash tokenizers for datasets cache names without transformers 5's init-only ``_vocab`` / ``_merges``.

Read only in ``__init__`` to build the hashed Rust ``_tokenizer``; only datasets' hashing Pickler is patched,
the live tokenizer is never modified, and only transformers' own classes qualify (custom or remote-code
subclasses may read ``_vocab`` while encoding).
"""

_INIT_ONLY_STATE = ("_vocab", "_merges")
_MARKER = "_unsloth_studio_tokenizer_fingerprint"


def _is_rust_tokenizer(value) -> bool:
    from tokenizers import Tokenizer
    return isinstance(value, Tokenizer)


def _is_transformers_class(cls) -> bool:
    # Remote code lives in ``transformers_modules``, which this prefix does not match.
    module = getattr(cls, "__module__", None) or ""
    return module == "transformers" or module.startswith("transformers.")


def _without_init_only(state):
    return {key: value for key, value in state.items() if key not in _INIT_ONLY_STATE}


class _InitStateFilter:
    """Pickler proxy dropping the init-only keys from one object's emitted state, so concurrent readers see it whole."""

    def __init__(self, pickler, target):
        self._pickler = pickler
        self._target = target

    def __getattr__(self, name):
        return getattr(self._pickler, name)

    def save_reduce(self, *args, **kwargs):
        if kwargs.get("obj") is self._target:
            if isinstance(kwargs.get("state"), dict):
                kwargs["state"] = _without_init_only(kwargs["state"])
            elif len(args) > 2 and isinstance(args[2], dict):
                args = args[:2] + (_without_init_only(args[2]),) + args[3:]
        return self._pickler.save_reduce(*args, **kwargs)


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
        if (
            not _is_transformers_class(type(obj))
            or not _is_rust_tokenizer(state.get("_tokenizer"))
            or not any(key in state for key in _INIT_ONLY_STATE)
        ):
            return original(pickler, obj)
        # Delegate so whatever datasets keeps out of the hash (encode cache, 4.7+ truncation / padding) stays out.
        return original(_InitStateFilter(pickler, obj), obj)

    setattr(_save_tokenizer, _MARKER, True)
    _save_tokenizer.__wrapped__ = original
    # datasets resolves the reducer by name on first sight of a tokenizer class, then caches it per class.
    _dill._save_transformersPreTrainedTokenizerBase = _save_tokenizer
    for obj_type, reducer in list(pickler.dispatch.items()):
        if reducer is original:
            pickler.dispatch[obj_type] = _save_tokenizer
    return True
