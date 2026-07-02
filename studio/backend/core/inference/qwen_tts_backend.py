# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Qwen3-TTS voice backend (Qwen/Qwen3-TTS-12Hz-* repos).

Qwen3-TTS is a transformers-based speech-LLM family (talker + codec tokenizer)
that runs in-process via the `qwen-tts` package, not through llama-server.
Checkpoint kinds differ in how a voice is chosen:

- CustomVoice: premade speakers (Ryan, Aiden, Vivian, ...). Works out of the
  box; the default speaker/language are env-overridable. Best fit for the
  voice-conversation dropdown.
- Base: 3-second voice cloning. Every synth call needs a reference clip and its
  transcript, configured via UNSLOTH_QWEN_TTS_REF_AUDIO / _REF_TEXT.
- VoiceDesign: builds a voice from a text description via
  UNSLOTH_QWEN_TTS_INSTRUCT.

The `qwen-tts` pip package hard-pins transformers/accelerate and drags in
gradio, so it is installed --no-deps against the venv's existing stack. Its
only missing runtime deps are `sox` (imported by the 25Hz tokenizer module at
package import; never *used* on the 12Hz path, and the pysox package imports
fine without the sox binary) -- einops/onnxruntime/librosa/torchaudio are
already in the Studio venv.
"""

import os
import subprocess
import sys
from typing import Optional, Tuple

import numpy as np

from core.inference.audio_codecs import _numpy_to_wav_bytes
from loggers import get_logger

logger = get_logger(__name__)

# Pinned to the version whose imports/API were verified against this backend.
_QWEN_TTS_PIP_SPECS = ("qwen-tts==0.1.1", "sox")

# Premade CustomVoice speaker + language defaults (env-overridable). Ryan is
# one of the two native-English premade speakers.
_DEFAULT_SPEAKER = os.environ.get("UNSLOTH_QWEN_TTS_SPEAKER", "Ryan")
# None -> let the model auto-detect the language from the text.
_DEFAULT_LANGUAGE = os.environ.get("UNSLOTH_QWEN_TTS_LANGUAGE") or None


def _ensure_qwen_tts_installed() -> None:
    # Check presence with find_spec, NOT `import qwen_tts`: importing it eagerly
    # pulls in transformers' AutoProcessor -> torchao, which crashes on Windows
    # ROCm until the stub is installed (done in load_model before the real
    # import). find_spec only resolves the module, it doesn't execute it, so a
    # genuinely-installed-but-not-yet-stubbed package isn't misread as missing.
    import importlib.util

    if importlib.util.find_spec("qwen_tts") is not None:
        return

    logger.info("Installing 'qwen-tts' package (first Qwen3-TTS voice load)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", *_QWEN_TTS_PIP_SPECS],
        capture_output = True,
        text = True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install the 'qwen-tts' package automatically. Install it "
            "manually with: pip install --no-deps qwen-tts sox\n\n"
            + (result.stderr or result.stdout or "")[-2000:]
        )
    if importlib.util.find_spec("qwen_tts") is None:
        raise RuntimeError("'qwen-tts' installed but is not importable.")


def _checkpoint_kind(repo_id: str) -> str:
    """'custom' | 'base' | 'design' from the repo name."""
    lower = repo_id.lower()
    if "customvoice" in lower or "custom-voice" in lower:
        return "custom"
    if "voicedesign" in lower or "voice-design" in lower:
        return "design"
    return "base"


class QwenTTSVoiceBackend:
    """In-process Qwen3-TTS voice slot. Interface mirrors SpeechT5VoiceBackend /
    LlamaCppBackend's audio surface so the audio routes treat them uniformly."""

    def __init__(self) -> None:
        self._model = None
        self._repo_id: Optional[str] = None
        self._kind: Optional[str] = None
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

    def load_model(self, repo_id: str) -> bool:
        if self.is_loaded and self._repo_id == repo_id:
            return True

        _ensure_qwen_tts_installed()

        # qwen_tts imports transformers' AutoProcessor, whose import chain pulls
        # in torchao -- which has no working Windows ROCm build (crashes loading
        # torch's c10d distributed ops). Install the same stub the RAG embedder
        # and training workers use; no-op on other platforms.
        from core._torchao_stub import install_torchao_windows_rocm_stub

        install_torchao_windows_rocm_stub()

        import torch
        from qwen_tts import Qwen3TTSModel

        use_cuda = torch.cuda.is_available()
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32
        logger.info("Loading Qwen3-TTS voice model: %s (device=%s)", repo_id, device_map)
        # Default attention implementation on purpose: flash-attn is not
        # available on Windows/ROCm, and qwen-tts degrades gracefully without it.
        self._model = Qwen3TTSModel.from_pretrained(
            repo_id, device_map = device_map, dtype = dtype
        )
        self._repo_id = repo_id
        self._kind = _checkpoint_kind(repo_id)
        self._is_audio = True
        self._audio_type = "qwen3_tts"

        # Warmup: MIOpen (ROCm) tunes kernels on the first forward pass, which
        # can take tens of seconds; absorb it during load, where the user is
        # already waiting. Only the CustomVoice kind can synthesize without
        # extra inputs, so only it gets warmed.
        if self._kind == "custom":
            try:
                self._model.generate_custom_voice(
                    text = "Warm up.",
                    speaker = _DEFAULT_SPEAKER,
                    language = _DEFAULT_LANGUAGE,
                )
                logger.info("Qwen3-TTS warmup complete")
            except Exception as e:
                logger.warning("Qwen3-TTS warmup failed (non-fatal): %s", e)
        return True

    def unload_model(self) -> None:
        self._model = None
        self._repo_id = None
        self._kind = None
        self._is_audio = False
        self._audio_type = None

    def generate_audio_response(
        self, text: str, audio_type: Optional[str] = None, **_ignored,
    ) -> Tuple[bytes, int]:
        """Synthesize ``text``; returns (wav_bytes, sample_rate).

        ``audio_type`` is accepted and ignored, matching the other voice slots'
        generate_audio_response signature."""
        if self._model is None:
            raise RuntimeError("Qwen3-TTS voice model is not loaded.")

        text = text.strip()
        if not text:
            raise RuntimeError("No text to synthesize.")

        if self._kind == "custom":
            wavs, sr = self._model.generate_custom_voice(
                text = text,
                speaker = _DEFAULT_SPEAKER,
                language = _DEFAULT_LANGUAGE,
            )
        elif self._kind == "design":
            instruct = os.environ.get("UNSLOTH_QWEN_TTS_INSTRUCT")
            if not instruct:
                raise RuntimeError(
                    "This Qwen3-TTS VoiceDesign checkpoint builds its voice from a "
                    "text description. Set UNSLOTH_QWEN_TTS_INSTRUCT to a voice "
                    "description, or use a CustomVoice checkpoint (premade "
                    "speakers) for voice conversation."
                )
            wavs, sr = self._model.generate_voice_design(
                text = text, instruct = instruct, language = _DEFAULT_LANGUAGE
            )
        else:  # base -- voice cloning
            ref_audio = os.environ.get("UNSLOTH_QWEN_TTS_REF_AUDIO")
            ref_text = os.environ.get("UNSLOTH_QWEN_TTS_REF_TEXT")
            if not ref_audio or not ref_text:
                raise RuntimeError(
                    "This Qwen3-TTS Base checkpoint is a voice-cloning model: every "
                    "synthesis needs a ~3s reference clip and its transcript. Set "
                    "UNSLOTH_QWEN_TTS_REF_AUDIO (path/URL to the clip) and "
                    "UNSLOTH_QWEN_TTS_REF_TEXT (what is said in it), or pick the "
                    "CustomVoice checkpoint (premade speakers) instead."
                )
            wavs, sr = self._model.generate_voice_clone(
                text = text,
                language = _DEFAULT_LANGUAGE,
                ref_audio = ref_audio,
                ref_text = ref_text,
            )

        waveform = np.asarray(wavs[0], dtype = np.float32)
        return _numpy_to_wav_bytes(waveform, int(sr)), int(sr)
