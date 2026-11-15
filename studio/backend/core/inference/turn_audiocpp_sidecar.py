# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice mode's end-of-turn model: from how a clip sounds, has the speaker finished?

It only ever ends a turn early. Asked at a short pause, a "finished" shortens the fixed silence
wait; anything else leaves that wait exactly as it was, so a wrong or missing answer costs nothing.
"""

from __future__ import annotations

import threading
from typing import Optional

from core.inference.audio_cpp_clip_sidecar import (
    AudioCppClipSidecar,
    ClipUnavailableError,
    resolve_hub_model,
)
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppModel

# Smart Turn v3.2 (BSD-2-Clause): a Whisper-Tiny encoder with a classifier head, 30 MB.
DEFAULT_TURN_MODEL = f"{AUDIO_CPP_REPO}/Smart-Turn-v3-GGUF"
# The bar a clip must reach to count as finished, by the language spoken. Measured in this runtime
# on the model's published test clips (100 to 700 per language), as the share of unfinished turns
# called finished at each pause: about 3% at 0.6 for these five, where finished speech scores a
# median 0.99 and unfinished 0.006.
_BAR = {"en": 0.6, "de": 0.6, "fr": 0.6, "ja": 0.6, "hi": 0.6}
# Spanish needs 0.9 for the same 3%, and so does a language nobody has measured.
_CAREFUL_BAR = 0.9
# Still 5-7% at 0.9, catching only 40-71% of finished turns: these keep the plain silence wait.
_NOT_TRUSTED = frozenset({"zh", "ar", "vi"})
# How much silence follows speech before the loop asks: as short as a pause between words allows.
# The clip it sends must end at the last voiced frame, since silence after it raises the share of
# unfinished turns called finished (about 4% with none, 8% with 200 ms, 10% with 400 ms).
ASK_AFTER_MS = 250
# The model reads the end of the turn; older audio only costs upload and decode time.
_TAIL_SECONDS = 8.0


def resolve_turn_model(model: Optional[str], *, network: bool = False) -> AudioCppModel:
    identifier = (model or "").strip() or DEFAULT_TURN_MODEL
    return resolve_hub_model(identifier, "turn", "tell the end of a turn with", network = network)


def complete_at(language: Optional[str]) -> Optional[float]:
    """The bar for ``language`` (a BCP 47 tag, or nothing when it is not known); None where the model is not trusted."""
    primary = (language or "").strip().lower().replace("_", "-").split("-")[0]
    if primary in _NOT_TRUSTED:
        return None
    return _BAR.get(primary, _CAREFUL_BAR)


def completion_from_result(payload: dict) -> float:
    """P(the speaker has finished) from the runtime's answer."""
    data = (payload.get("custom_schema_output") or {}).get("data") or {}
    probability = data.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ClipUnavailableError("The end-of-turn model gave no answer.")
    return min(1.0, max(0.0, float(probability)))


class AudioCppTurnSidecar(AudioCppClipSidecar):
    label = "end-of-turn model"

    def _resolve(self, model: Optional[str], *, network: bool) -> AudioCppModel:
        return resolve_turn_model(model, network = network)

    def _cpu_only(self, entry: AudioCppModel) -> bool:
        # About 60 ms a clip on four CPU threads: not worth VRAM, or a move every time training starts.
        return True

    def _known_cpu_id(self, model: Optional[str]) -> Optional[str]:
        return (model or "").strip() or DEFAULT_TURN_MODEL

    def status(self) -> dict:
        return {
            **self.state(),
            "default_model": DEFAULT_TURN_MODEL,
            "ask_after_ms": ASK_AFTER_MS,
        }

    def detect(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict:
        bar = complete_at(language)
        if bar is None:
            # Not asked at all: the loop waits out its silence as it always did.
            return {"p_complete": None, "finished": False, "bar": None, "model": None}
        model_id, payload, _wav = self.run_clip(audio, model, tail_seconds = _TAIL_SECONDS)
        p_complete = completion_from_result(payload)
        return {
            "p_complete": round(p_complete, 4),
            "finished": p_complete >= bar,
            "bar": bar,
            "model": model_id,
        }


_sidecar: Optional[AudioCppTurnSidecar] = None
_sidecar_lock = threading.Lock()


def get_audio_cpp_turn_sidecar() -> AudioCppTurnSidecar:
    global _sidecar
    with _sidecar_lock:
        if _sidecar is None:
            _sidecar = AudioCppTurnSidecar()
        return _sidecar
