# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The speech-architecture verdict, as a leaf module with no package baggage.

It lives here rather than beside the rest of the GGUF metadata helpers because
``utils.models.__init__`` eagerly imports ``model_config``, which imports ``yaml``.
``core.inference.llama_cpp`` needs this verdict at import time, and reaching it through
``utils.models`` made the whole models package -- and PyYAML with it -- a hard import
dependency of the chat backend. That took the repo's own Source lint job red, where
``tests/studio/load_freeze/test_load_orchestrator.py`` imports the backend without PyYAML
installed.

Every caller imports from here, so there is exactly ONE definition.
"""

from __future__ import annotations

from typing import Optional

# ``general.architecture`` values naming a speech or neural-codec checkpoint that no Unsloth
# runtime can decode: llama.cpp has no CSM decoder (still an unmerged upstream PR) and no
# media backend reads one either. Published CSM GGUFs do not agree on a spelling, so all four
# on the Hub today are listed: ggml-org "llama-csm", cartesia "csm", cstr "csm-tts", and a
# bundle's Mimi vocoder half "mimi". Named once so the chat gate, the listing classifier and
# the media preflight cannot drift apart.
SPEECH_GGUF_ARCHS = frozenset({"llama-csm", "csm", "csm-tts", "mimi"})

# The Mimi vocoder in ggml-org/sesame-csm-1b-GGUF puts a whole SENTENCE in general.architecture
# rather than an identifier. Matched on the flag, not the full string, so a reword still lands.
_VOCODER_MARKERS = ("--model-vocoder", "cannot be used as llm")


# ``general.architecture`` values whose llama.cpp loader builds no vocabulary output head at
# all. The encoder-only BERT family never produces logits (src/models/bert.cpp,
# nomic-bert.cpp, nomic-bert-moe.cpp, jina-bert-v2.cpp, jina-bert-v3.cpp, modern-bert.cpp,
# neo-bert.cpp, eurobert.cpp create tok_embd and no output tensor), and bitnet.cpp multiplies
# by tok_embd in place instead of duplicating it. For these a missing ``output.weight`` is not
# tying, so nothing extra is allocated and the VRAM budget must not charge for it.
#
# A blocklist and not a decoder whitelist on purpose: an unlisted architecture is charged, so a
# new arch over-counts by one embedding matrix instead of under-counting by one, and
# under-counting weights is what makes the context search promise VRAM the load then takes.
NO_VOCAB_OUTPUT_GGUF_ARCHS = frozenset(
    {
        "bert",
        "bitnet",
        "eurobert",
        "jina-bert-v2",
        "jina-bert-v3",
        "modern-bert",
        "neo-bert",
        "nomic-bert",
        "nomic-bert-moe",
    }
)


def is_no_vocab_output_gguf_architecture(architecture: Optional[str]) -> bool:
    """Whether llama.cpp gives ``general.architecture`` no vocabulary output tensor.

    ``None`` and the empty string are not: an undeclared architecture is unknown, and the one
    caller fails towards charging."""
    if not architecture:
        return False
    return architecture.strip().lower() in NO_VOCAB_OUTPUT_GGUF_ARCHS


def is_speech_gguf_architecture(architecture: Optional[str]) -> bool:
    """Whether ``general.architecture`` names something only a TTS runtime can decode.

    Case- and space-insensitive, like every other architecture comparison here. ``None`` and the
    empty string are NOT speech: a GGUF declaring no architecture is unknown, and every caller
    fails open on unknown."""
    if not architecture:
        return False
    normalized = architecture.strip().lower()
    if normalized in SPEECH_GGUF_ARCHS:
        return True
    return any(marker in normalized for marker in _VOCODER_MARKERS)
