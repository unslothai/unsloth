# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``TextIteratorStreamer`` whose ``put`` decodes only the ids after the last split point
(``IncrementalDecode``) instead of every id since the last newline on each token, which is
quadratic in line length. Emitted chunks are identical to upstream's ``put`` (transformers 4.57
and 5.x share it); tokenizers that do not split keep upstream's ``put`` unchanged."""

from __future__ import annotations

from transformers import TextIteratorStreamer

from core.inference.incremental_decode import IncrementalDecode, split_profile
from core.inference.native_tool_tokens import NativeToolTokenDecoder

__all__ = ["IncrementalTextIteratorStreamer"]


def _line_decoder_factory(tokenizer, decode_kwargs):
    """A callable returning a fresh decoder with ``feed(id)`` and ``text`` equal to
    ``tokenizer.decode(ids, **decode_kwargs)``, or None when decoding never splits."""
    if isinstance(tokenizer, NativeToolTokenDecoder):
        if split_profile(tokenizer._tokenizer) is None:
            return None
        return tokenizer.incremental
    if split_profile(tokenizer) is None:
        return None
    return lambda: IncrementalDecode(tokenizer, decode_kwargs)


class IncrementalTextIteratorStreamer(TextIteratorStreamer):
    _line_factory = False  # False = not resolved yet; None = upstream put
    _line = None

    def put(self, value):
        factory = self._line_factory
        if factory is False:
            factory = self._line_factory = _line_decoder_factory(self.tokenizer, self.decode_kwargs)
        if factory is None:
            return super().put(value)

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        new_ids = value.tolist()
        line = self._line
        if line is None or not self.token_cache:
            # Upstream starts a fresh token_cache after a newline and in end().
            line = self._line = factory()
            for token_id in self.token_cache:
                line.feed(token_id)
        self.token_cache.extend(new_ids)
        for token_id in new_ids:
            line.feed(token_id)
        text = line.text

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
            self._line = None
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            # rfind from print_len slices the same text as upstream's whole-line rfind.
            printable_text = text[self.print_len : text.rfind(" ", self.print_len) + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)
