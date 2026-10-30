# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``tokenizer.decode(ids)`` for a reply that grows one id at a time, without re-decoding every id.

Streaming paths that snapshot the reply by re-decoding all of its ids pay for the whole reply on
every token, which at 8k tokens costs milliseconds per token. This returns the same string per
step by decoding only the ids after the last point where decoding is known to split.

A split before id ``j`` is taken only where ``decode(ids[:k]) == decode(ids[:j]) + tail`` holds
for every continuation, with ``tail`` read off ``decode(ids[j-1:k])`` past the text of
``ids[j-1]`` alone. That holds for Hugging Face fast tokenizers whose decoder is built from
ByteLevel, Metaspace or per-token Replace / ByteFallback / Fuse / Strip(start <= 1) steps, when
both ids at the split are ordinary vocabulary pieces that decode alone to whole characters:

- neither is an added token, so special-token skipping and the added-token chunking of
  ``tokenizers`` stay inside the decoded window;
- neither is a ``<0xNN>`` byte piece, so no ByteFallback run crosses the split;
- each decodes alone without U+FFFD, so the UTF-8 decode restarts at the split;
- the left one decodes to a non-empty string, so a leading-space strip cannot reach the right;
- with ``clean_up_tokenization_spaces`` possibly on, the left one ends in a character no cleanup
  pattern contains, so no cleanup replacement spans the split.

Any other tokenizer, or a decode method overridden anywhere, decodes every id as before.
"""

from __future__ import annotations

import json
import re
import threading
import weakref

__all__ = ["IncrementalDecode", "split_profile"]

# Every character of PreTrainedTokenizerBase.clean_up_tokenization's patterns.
_CLEANUP_CHARS = frozenset(" .?!,'ntmsvre")
_BYTE_PIECE = re.compile(r"<0x[0-9A-Fa-f]{2}>")
_SPLIT_DECODE = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}


def _decoder_is_local(config) -> bool:
    kind = config.get("type") if isinstance(config, dict) else None
    if kind in ("ByteLevel", "Metaspace", "ByteFallback", "Fuse"):
        return True
    if kind == "Replace":
        pattern = config.get("pattern") or {}
        # One character cannot match across two pieces, even after Fuse.
        return isinstance(pattern.get("String"), str) and len(pattern["String"]) == 1
    if kind == "Strip":
        return config.get("stop") == 0 and config.get("start") in (0, 1)
    if kind == "Sequence":
        steps = config.get("decoders")
        return isinstance(steps, list) and all(_decoder_is_local(step) for step in steps)
    return False


def _fast_tokenizer(tokenizer):
    """The Hugging Face fast tokenizer that ``tokenizer.decode`` runs, when nothing overrides its decode path."""
    try:
        from transformers import PreTrainedTokenizerFast
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:  # noqa: BLE001 -- no transformers means no known decode path
        return None
    try:
        bound = tokenizer.decode
    except Exception:  # noqa: BLE001 -- adapters vary
        return None
    inner = getattr(bound, "__self__", None)
    if not isinstance(inner, PreTrainedTokenizerFast):
        return None
    cls = type(inner)
    if (
        getattr(bound, "__func__", None) is not PreTrainedTokenizerBase.decode
        or cls._decode is not PreTrainedTokenizerFast._decode
        or cls.clean_up_tokenization is not PreTrainedTokenizerBase.clean_up_tokenization
        or any(
            name in getattr(inner, "__dict__", {}) for name in ("_decode", "clean_up_tokenization")
        )
    ):
        return None
    return inner


class _SplitProfile:
    """Per-id answers to "may decoding split here", cached for one tokenizer."""

    def __init__(self, tokenizer, added_ids, size):
        self._tokenizer = tokenizer
        self._added = added_ids
        self.size = size
        self._whole = {}
        self._ends_clean = {}
        self._lock = threading.Lock()

    def _whole_piece(self, token_id: int) -> bool:
        whole = self._whole.get(token_id)
        if whole is None:
            whole, text, later = False, "", ""
            if token_id not in self._added:
                try:
                    piece = self._tokenizer.convert_ids_to_tokens(token_id)
                    if isinstance(piece, str) and not _BYTE_PIECE.fullmatch(piece):
                        text = self._tokenizer.decode([token_id], **_SPLIT_DECODE)
                        whole = isinstance(text, str) and bool(text) and "\ufffd" not in text
                    if whole:
                        # The piece also as a later id of its run: Metaspace drops every
                        # replacement char of a run's first id only, so a piece ending in one
                        # ends in "a" alone but in " " mid-reply, where " ." cleanup can span.
                        pair = self._tokenizer.decode([token_id, token_id], **_SPLIT_DECODE)
                        later = pair[len(text) :] if pair.startswith(text) else ""
                        whole = bool(later) and "\ufffd" not in later
                except Exception:  # noqa: BLE001 -- an id the tokenizer cannot name never splits
                    whole = False
            with self._lock:
                self._whole[token_id] = whole
                self._ends_clean[token_id] = (
                    whole and text[-1] not in _CLEANUP_CHARS and later[-1] not in _CLEANUP_CHARS
                )
        return whole

    def splits(self, left: int, right: int, cleanup: bool) -> bool:
        if not (self._whole_piece(left) and self._whole_piece(right)):
            return False
        return not cleanup or self._ends_clean[left]


_PROFILES = weakref.WeakKeyDictionary()
_PROFILES_LOCK = threading.Lock()


def split_profile(tokenizer):
    """The split profile for ``tokenizer``, or None when its decode is not known to split."""
    inner = _fast_tokenizer(tokenizer)
    if inner is None:
        return None
    try:
        size = len(inner)
    except Exception:  # noqa: BLE001
        return None
    with _PROFILES_LOCK:
        profile = _PROFILES.get(inner)
    if profile is not None and profile.size == size:
        return profile or None
    try:
        decoder = inner.backend_tokenizer.decoder
        config = json.loads(decoder.__getstate__()) if decoder is not None else None
        local = _decoder_is_local(config)
        added = frozenset(int(token_id) for token_id in inner.added_tokens_decoder)
    except Exception:  # noqa: BLE001 -- anything unreadable keeps the full decode
        local, added = False, frozenset()
    profile = _SplitProfile(inner, added, size) if local else None
    try:
        with _PROFILES_LOCK:
            # False marks a tokenizer already found not to split.
            _PROFILES[inner] = profile if profile is not None else _NoSplit(size)
    except TypeError:
        pass
    return profile


class _NoSplit:
    def __init__(self, size):
        self.size = size

    def __bool__(self):
        return False


class IncrementalDecode:
    """``tokenizer.decode(ids, **decode_kwargs)`` of the ids fed so far, as ``text``.

    ``fallback(ids)`` replaces that decode for tokenizers that do not split; it must equal it
    wherever they do. Calling the instance with a list that grew by one id feeds that id, so it
    can stand in for a ``decode(ids)`` callable over an append-only list."""

    def __init__(
        self,
        tokenizer,
        decode_kwargs = None,
        *,
        fallback = None,
    ):
        self._tokenizer = tokenizer
        self._kwargs = dict(decode_kwargs or {})
        self._fallback = fallback
        self._profile = split_profile(tokenizer)
        cleanup = self._kwargs.get("clean_up_tokenization_spaces")
        if cleanup is None and self._profile is not None:
            cleanup = getattr(self._profile._tokenizer, "clean_up_tokenization_spaces", True)
        # Unknown counts as on: the barrier test is only stricter.
        self._cleanup = cleanup is not False
        self.reset()

    def reset(self):
        self.ids = []
        self.text = ""
        self._anchor = 0
        self._anchor_text = ""
        self._context_text = ""

    def _decode(self, ids):
        if self._fallback is not None:
            return self._fallback(ids)
        return self._tokenizer.decode(ids, **self._kwargs)

    def feed(self, token_id) -> str:
        ids = self.ids
        ids.append(int(token_id))
        previous = self.text
        if self._profile is None or self._anchor == 0:
            self.text = self._decode(ids)
        else:
            window = self._tokenizer.decode(ids[self._anchor - 1 :], **self._kwargs)
            if window.startswith(self._context_text):
                self.text = self._anchor_text + window[len(self._context_text) :]
            else:
                # Not reachable for a decoder that passed the gate; never guess.
                self._profile = None
                self.text = self._decode(ids)
        count = len(ids)
        if (
            self._profile is not None
            and count >= 2
            and self._profile.splits(ids[-2], ids[-1], self._cleanup)
        ):
            self._anchor = count - 1
            self._anchor_text = previous
            self._context_text = self._tokenizer.decode([ids[-2]], **self._kwargs)
        return self.text

    def __call__(self, ids) -> str:
        count = len(ids)
        if count == len(self.ids) + 1:
            return self.feed(ids[-1])
        if count != len(self.ids):
            self.reset()
            for token_id in ids:
                self.feed(token_id)
        return self.text
