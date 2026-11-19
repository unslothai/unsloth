# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import threading
from typing import Optional

from core.inference.audio_cpp_clip_sidecar import (
    AudioCppClipSidecar,
    ClipUnavailableError,
    resolve_hub_model,
)
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppModel

# Smart Turn v3.2 (BSD-2-Clause).
DEFAULT_TURN_MODEL = f"{AUDIO_CPP_REPO}/Smart-Turn-v3-GGUF"
# Measured: 0.6 false-finishes ~3% of unfinished turns for these five; others use 0.9.
_BAR = {"en": 0.6, "de": 0.6, "fr": 0.6, "ja": 0.6, "hi": 0.6}
_CAREFUL_BAR = 0.9
_NOT_TRUSTED = frozenset({"zh", "ar", "vi"})
# Clip ends at the last voiced frame: trailing silence adds false finishes.
ASK_AFTER_MS = 250
_TAIL_SECONDS = 8.0


def resolve_turn_model(
    model: Optional[str],
    *,
    network: bool = False,
    hf_token: Optional[str] = None,
) -> AudioCppModel:
    identifier = (model or "").strip() or DEFAULT_TURN_MODEL
    return resolve_hub_model(
        identifier, "turn", "tell the end of a turn with", network = network, hf_token = hf_token
    )


def complete_at(language: Optional[str]) -> Optional[float]:
    primary = (language or "").strip().lower().replace("_", "-").split("-")[0]
    if primary in _NOT_TRUSTED:
        return None
    return _BAR.get(primary, _CAREFUL_BAR)


def completion_from_result(payload: dict) -> float:
    data = (payload.get("custom_schema_output") or {}).get("data") or {}
    probability = data.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ClipUnavailableError("The end-of-turn model gave no answer.")
    return min(1.0, max(0.0, float(probability)))


class AudioCppTurnSidecar(AudioCppClipSidecar):
    label = "end-of-turn model"

    def _resolve(
        self, model: Optional[str], *, network: bool, hf_token: Optional[str] = None
    ) -> AudioCppModel:
        return resolve_turn_model(model, network = network, hf_token = hf_token)

    def _cpu_only(self, entry: AudioCppModel) -> bool:
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
