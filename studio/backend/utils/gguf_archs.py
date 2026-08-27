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
