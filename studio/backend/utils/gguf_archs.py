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

# ``general.architecture`` values no Unsloth runtime can decode (llama.cpp has no CSM decoder).
# Published CSM GGUFs disagree on spelling, so all four on the Hub are listed. Named once so the
# chat gate, the listing classifier and the media preflight cannot drift apart.
SPEECH_GGUF_ARCHS = frozenset({"llama-csm", "csm", "csm-tts", "mimi"})

# The Mimi vocoder in ggml-org/sesame-csm-1b-GGUF puts a whole SENTENCE in general.architecture
# rather than an identifier. Matched on the flag, not the full string, so a reword still lands.
_VOCODER_MARKERS = ("--model-vocoder", "cannot be used as llm")


# Architectures whose llama.cpp loader builds no vocabulary output head, so a missing
# ``output.weight`` is not tying and nothing is duplicated (src/models/bert.cpp and its
# nomic/jina/modern/neo/euro siblings create tok_embd alone; bitnet.cpp multiplies by
# tok_embd in place). A blocklist, not a decoder allowlist: an unlisted arch is charged,
# and over-counting by one embedding matrix is the safe direction.
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

    None and "" are not: an undeclared arch is unknown, and the caller fails towards charging."""
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
