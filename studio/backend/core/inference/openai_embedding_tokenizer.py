# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode OpenAI embedding token IDs using the packaged cl100k_base vocabulary."""

import base64
import gzip
import hashlib
from pathlib import Path

# OpenAI's tiktoken 0.13.0 cl100k_base ranks, gzip-compressed without changing the data.
# https://github.com/openai/tiktoken/blob/0.13.0/tiktoken_ext/openai_public.py
# https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
_VOCABULARY = (
    Path(__file__).resolve().parents[2] / "assets" / "tokenizers" / "cl100k_base.tiktoken.gz"
)
_VOCABULARY_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def _load_token_bytes() -> dict[int, bytes]:
    with gzip.open(_VOCABULARY, "rb") as file:
        ranks = file.read()
    if hashlib.sha256(ranks).hexdigest() != _VOCABULARY_SHA256:
        raise ValueError("The packaged cl100k_base vocabulary failed its checksum.")
    tokens = {}
    for line in ranks.splitlines():
        encoded, rank = line.split()
        tokens[int(rank)] = base64.b64decode(encoded)
    tokens.update(
        {
            100257: b"<|endoftext|>",
            100258: b"<|fim_prefix|>",
            100259: b"<|fim_middle|>",
            100260: b"<|fim_suffix|>",
            100276: b"<|endofprompt|>",
        }
    )
    return tokens


# The endpoint imports this module only for OpenAI token arrays. Python's import lock
# initializes it once even when several requests arrive together.
_TOKEN_BYTES = _load_token_bytes()


def decode_tokens(token_ids: list[int]) -> str:
    # A single UTF-8 character can straddle tokens, so decode the joined bytes.
    return b"".join(_TOKEN_BYTES[token] for token in token_ids).decode("utf-8", errors = "replace")
