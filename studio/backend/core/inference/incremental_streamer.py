# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``TextIteratorStreamer`` that decodes long lines incrementally instead of re-decoding the whole
line per token; chunks equal upstream ``put``, which short lines and non-splitting tokenizers keep."""

from __future__ import annotations

from transformers import TextIteratorStreamer

from core.inference.incremental_decode import IncrementalDecode, split_profile
from core.inference.native_tool_tokens import NativeToolTokenDecoder

__all__ = ["IncrementalTextIteratorStreamer"]

# Below this many ids upstream's whole-line decode is cheaper than feeding a decoder.
_MIN_LINE_IDS = 64


def _line_decoder(tokenizer, decode_kwargs):
    if isinstance(tokenizer, NativeToolTokenDecoder):
        return tokenizer.incremental() if split_profile(tokenizer._tokenizer) else None
    return IncrementalDecode(tokenizer, decode_kwargs) if split_profile(tokenizer) else None


class IncrementalTextIteratorStreamer(TextIteratorStreamer):
    _decoder = False  # False = unresolved, None = never splits
    _line_ids = None  # token_cache list _decoder consumed; upstream rebinds it on every flush

    def _attach(self) -> bool:
        if len(self.token_cache) < _MIN_LINE_IDS:
            return False
        if self._decoder is False:
            self._decoder = _line_decoder(self.tokenizer, self.decode_kwargs)
        if self._decoder is None:
            return False
        self._decoder.reset()
        for token_id in self.token_cache:
            self._decoder.feed(token_id)
        self._line_ids = self.token_cache
        return True

    def put(self, value):
        if self._line_ids is not self.token_cache and not self._attach():
            return super().put(value)

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        new_ids = value.tolist()
        self.token_cache.extend(new_ids)
        for token_id in new_ids:
            self._decoder.feed(token_id)
        text = self._decoder.text

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            # rfind from print_len slices the same text as upstream's whole-line rfind.
            printable_text = text[self.print_len : text.rfind(" ", self.print_len) + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)
