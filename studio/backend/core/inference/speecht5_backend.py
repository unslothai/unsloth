# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SpeechT5 TTS backend (microsoft/speecht5_tts and compatible repos).

SpeechT5 is a transformers-native encoder-decoder TTS model (+ HiFi-GAN
vocoder), not a causal speech-LLM and not a GGUF codec model, so it runs
in-process here via the already-installed transformers stack rather than
through the llama-server voice slot. No extra pip dependency and no Python
version gate -- if Studio runs, this runs.
"""

from typing import List, Optional, Tuple

import numpy as np

from core.inference.audio_codecs import _numpy_to_wav_bytes
from loggers import get_logger

logger = get_logger(__name__)

SPEECHT5_SAMPLE_RATE = 16000
# HiFi-GAN vocoder that pairs with microsoft/speecht5_tts. Auto-downloaded on
# first load if not already cached.
VOCODER_REPO = "microsoft/speecht5_hifigan"
# CMU Arctic xvector index used as the default speaker voice -- a clear
# US-English speaker, the same index used throughout the HF SpeechT5 examples.
_DEFAULT_SPEAKER_INDEX = 7306
# SpeechT5 degrades / truncates on very long inputs; keep each synth call under
# roughly this many characters. The chat frontend already chunks by sentence,
# so this only guards against a single pathologically long sentence.
_MAX_CHARS_PER_SEGMENT = 400


def _split_for_speecht5(text: str) -> List[str]:
    """Split overly long text on word boundaries so no single synth call blows
    past SpeechT5's practical input limit."""
    text = text.strip()
    if len(text) <= _MAX_CHARS_PER_SEGMENT:
        return [text] if text else []

    segments: List[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > _MAX_CHARS_PER_SEGMENT:
            segments.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        segments.append(current)
    return segments


class SpeechT5VoiceBackend:
    """In-process SpeechT5 voice slot (independent of the GGUF voice slot and
    the main chat slot). Interface mirrors LlamaCppBackend's audio surface so
    the audio routes can treat it the same way."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embeddings = None
        self._repo_id: Optional[str] = None
        self._is_audio = False
        self._audio_type: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_active(self) -> bool:
        return self.is_loaded

    @property
    def model_identifier(self) -> Optional[str]:
        return self._repo_id

    def _load_speaker_embeddings(self, device):
        import torch

        try:
            from datasets import load_dataset

            ds = load_dataset("Matthijs/cmu-arctic-xvectors", split = "validation")
            xvector = ds[_DEFAULT_SPEAKER_INDEX]["xvector"]
            return torch.tensor(xvector).unsqueeze(0).to(device)
        except Exception as e:
            # Dataset download unavailable (offline etc.): fall back to a fixed
            # deterministic embedding so synthesis still works, just with a less
            # natural default voice.
            logger.warning(
                "Could not load CMU Arctic xvectors (%s); using a deterministic "
                "fallback speaker embedding.",
                e,
            )
            g = torch.Generator().manual_seed(0)
            return torch.randn(1, 512, generator = g).to(device)

    def load_model(self, repo_id: str) -> bool:
        if self.is_loaded and self._repo_id == repo_id:
            return True

        import torch
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SpeechT5 voice model: %s (device=%s)", repo_id, device)
        self._processor = SpeechT5Processor.from_pretrained(repo_id)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(repo_id).to(device)
        self._vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_REPO).to(device)
        self._speaker_embeddings = self._load_speaker_embeddings(device)
        self._repo_id = repo_id
        self._is_audio = True
        self._audio_type = "speecht5"
        return True

    def unload_model(self) -> None:
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embeddings = None
        self._repo_id = None
        self._is_audio = False
        self._audio_type = None

    def generate_audio_response(
        self, text: str, audio_type: Optional[str] = None, **_ignored,
    ) -> Tuple[bytes, int]:
        """Synthesize ``text``; returns (wav_bytes, 16000).

        ``audio_type`` is accepted and ignored, matching the GGUF voice slot's
        generate_audio_response signature."""
        if self._model is None:
            raise RuntimeError("SpeechT5 voice model is not loaded.")

        import torch

        device = next(self._model.parameters()).device
        segments = _split_for_speecht5(text)
        if not segments:
            raise RuntimeError("No text to synthesize.")

        waves: List[np.ndarray] = []
        for segment in segments:
            inputs = self._processor(text = segment, return_tensors = "pt")
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                speech = self._model.generate_speech(
                    input_ids, self._speaker_embeddings, vocoder = self._vocoder
                )
            waves.append(speech.detach().cpu().numpy().astype(np.float32))

        waveform = np.concatenate(waves)
        return _numpy_to_wav_bytes(waveform, SPEECHT5_SAMPLE_RATE), SPEECHT5_SAMPLE_RATE
