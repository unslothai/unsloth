# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Audio codec loading and decoding for TTS inference.
Supports: SNAC (Orpheus), CSM (Sesame), BiCodec (Spark), DAC (OuteTTS)
"""

import io
import json
import os
import re
import wave
import structlog
from loggers import get_logger
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from utils.third_party_source import (
    deactivate_pinned_package,
    ensure_dac_speech_weights,
    ensure_outetts_source,
    ensure_spark_tts_source,
    import_outetts_module,
    import_sparktts_module,
)

logger = get_logger(__name__)
_SPARK_TTS_REPO = "unsloth/Spark-TTS-0.5B"
_MAX_SPARK_EXPORT_METADATA_BYTES = 1_000_000


def _bicodec_assets_complete(repo_path: Path) -> bool:
    try:
        config = repo_path / "BiCodec" / "config.yaml"
        weights = repo_path / "BiCodec" / "model.safetensors"
        return (
            config.is_file()
            and config.stat().st_size > 0
            and weights.is_file()
            and weights.stat().st_size > 0
        )
    except OSError:
        return False


def resolve_bicodec_repo_path(
    model_repo_path: Optional[str] = None,
    *,
    hf_token: Optional[str] | bool = None,
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve and stage the Spark repository that owns the BiCodec assets."""
    from huggingface_hub import snapshot_download
    from utils.hf_cache_settings import active_hf_hub_cache
    from utils.utils import canonical_model_repo_id, hf_env_offline

    source = str(model_repo_path or _SPARK_TTS_REPO).strip()
    local = Path(source).expanduser()
    if local.is_dir():
        root = local.parent if local.name.lower() == "llm" else local
        if _bicodec_assets_complete(root):
            ensure_spark_tts_source(root)
            return os.path.abspath(root)

        metadata = local / "export_metadata.json"
        base_model = None
        try:
            if metadata.is_file() and metadata.stat().st_size <= _MAX_SPARK_EXPORT_METADATA_BYTES:
                value = json.loads(metadata.read_text(encoding = "utf-8-sig"))
                candidate = value.get("base_model") if isinstance(value, dict) else None
                if isinstance(candidate, str) and candidate.strip():
                    base_model = candidate.strip()
        except (OSError, UnicodeError, TypeError, ValueError):
            pass
        if base_model is None:
            raise RuntimeError("The local Spark-TTS model has no complete BiCodec assets")
        source = base_model
        local = Path(source).expanduser()
        if local.is_dir():
            root = local.parent if local.name.lower() == "llm" else local
            if not _bicodec_assets_complete(root):
                raise RuntimeError("The recorded Spark-TTS base has incomplete BiCodec assets")
            ensure_spark_tts_source(root)
            return os.path.abspath(root)

    from utils.security import load_scan_target

    repo_id, _load_subdirs = load_scan_target(canonical_model_repo_id(source), ())
    repo_id = repo_id or source
    root = Path(
        snapshot_download(
            repo_id,
            token = (
                False
                if hf_token is False
                else hf_token.strip()
                if hf_token and hf_token.strip()
                else None
            ),
            cache_dir = cache_dir or active_hf_hub_cache(),
            local_files_only = hf_env_offline() if local_files_only is None else local_files_only,
        )
    )
    if not _bicodec_assets_complete(root):
        raise RuntimeError("The staged Spark-TTS repository has incomplete BiCodec assets")
    ensure_spark_tts_source(root)
    return os.path.abspath(root)


def _numpy_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy waveform to WAV bytes (16-bit PCM)."""
    waveform = waveform.flatten()
    peak = max(abs(waveform.max()), abs(waveform.min()))
    if peak > 1.0:
        waveform = waveform / peak
    pcm = (waveform * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


class AudioCodecManager:
    """Manages loading and caching of audio codec models for TTS decoding."""

    def __init__(self):
        self._snac_model = None
        self._bicodec_tokenizer = None
        self._bicodec_repo_path = None
        self._bicodec_code_dir = None
        self._dac_audio_codec = None
        self._outetts_code_dir = None
        # The loaders reuse a resident codec, so a later request gets the first
        # placement rather than the one it asked for.
        self._codec_devices: dict = {}

    def load_codec(
        self,
        audio_type: str,
        device: str = "cuda",
        model_repo_path: Optional[str] = None,
    ) -> None:
        """Load the appropriate codec for the given audio type."""
        if audio_type == "snac":
            self._load_snac(device, model_repo_path)
        elif audio_type == "bicodec":
            self._load_bicodec(device, model_repo_path)
        elif audio_type == "dac":
            self._load_dac(device, model_repo_path)
        elif audio_type == "csm":
            pass  # CSM decoding is built into the model (output_audio=True)
        else:
            raise ValueError(f"Unknown audio_type: {audio_type}")

    # ── Lazy loaders ─────────────────────────────────────────────

    def _load_snac(self, device: str, model_repo_path: Optional[str] = None) -> None:
        if self._snac_model is not None:
            return
        from snac import SNAC
        from utils.hf_cache_settings import active_hf_hub_cache

        # Route weights to the selected cache; this can run in the main process.
        self._snac_model = (
            SNAC.from_pretrained(
                model_repo_path or "hubertsiuzdak/snac_24khz",
                cache_dir = active_hf_hub_cache(),
            )
            .to(device)
            .eval()
        )
        self._codec_devices["snac"] = device
        logger.info("Loaded SNAC codec (24kHz)")

    def _load_bicodec(
        self,
        device: str,
        model_repo_path: Optional[str] = None,
    ) -> None:
        if self._bicodec_tokenizer is not None:
            return
        spark_code_dir = ensure_spark_tts_source(model_repo_path)
        self._bicodec_code_dir = spark_code_dir
        BiCodecTokenizer = import_sparktts_module(
            "sparktts.models.audio_tokenizer",
            spark_code_dir,
        ).BiCodecTokenizer

        # BiCodecTokenizer needs the MODEL repo path (has BiCodec/ weights)
        tokenizer_path = model_repo_path or spark_code_dir
        self._bicodec_repo_path = tokenizer_path
        self._bicodec_tokenizer = BiCodecTokenizer(tokenizer_path, device)
        self._codec_devices["bicodec"] = device
        logger.info(f"Loaded BiCodec tokenizer from {tokenizer_path}")

    def _load_dac(self, device: str, audio_codec_path: Optional[str] = None) -> None:
        if self._dac_audio_codec is not None:
            return
        outetts_code_dir = ensure_outetts_source()
        self._outetts_code_dir = outetts_code_dir
        AudioProcessor = import_outetts_module(
            "outetts.version.v3.audio_processor",
            outetts_code_dir,
        ).AudioProcessor
        OuteTTSModelConfig = import_outetts_module(
            "outetts.models.config",
            outetts_code_dir,
        ).ModelConfig
        resolved_audio_codec_path = audio_codec_path or ensure_dac_speech_weights()

        dummy_config = OuteTTSModelConfig(
            tokenizer_path = None,
            device = device,
            audio_codec_path = str(resolved_audio_codec_path),
        )
        processor = AudioProcessor(config = dummy_config)
        self._dac_audio_codec = processor.audio_codec
        self._codec_devices["dac"] = device
        logger.info("Loaded DAC audio codec")

    # ── Decoders ─────────────────────────────────────────────────

    def decode_snac(self, generated_ids: torch.Tensor, device: str) -> Tuple[bytes, int]:
        """Decode SNAC tokens (Orpheus) into WAV bytes.

        Finds the START_OF_SPEECH (128257) marker, extracts codes after it,
        strips EOS (128258), redistributes 7-per-frame codes into 3 SNAC layers.
        Returns (wav_bytes, 24000).
        """
        token_indices = (generated_ids == 128257).nonzero(as_tuple = True)
        if len(token_indices[1]) > 0:
            cropped = generated_ids[:, token_indices[1][-1] + 1 :]
        else:
            # Fall back to the entire output if the marker is missing
            logger.warning("No START_OF_SPEECH token (128257) found — using full generated output")
            cropped = generated_ids
        row = cropped[0]

        row = row[row != 128258]

        row = row[: (len(row) // 7) * 7]
        if len(row) == 0:
            raise ValueError("No valid audio codes found after START_OF_SPEECH token")

        codes = [t.item() - 128266 for t in row]

        # Redistribute into 3 SNAC layers (7 codes per frame → 1+2+4)
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(len(codes) // 7):
            layer_1.append(codes[7 * i])
            layer_2.append(codes[7 * i + 1] - 4096)
            layer_3.append(codes[7 * i + 2] - 8192)
            layer_3.append(codes[7 * i + 3] - 12288)
            layer_2.append(codes[7 * i + 4] - 16384)
            layer_3.append(codes[7 * i + 5] - 20480)
            layer_3.append(codes[7 * i + 6] - 24576)

        snac_codes = [
            torch.tensor(layer).unsqueeze(0).to(device) for layer in [layer_1, layer_2, layer_3]
        ]

        with torch.no_grad():
            audio = self._snac_model.decode(snac_codes)

        waveform = audio.squeeze().cpu().numpy()
        return _numpy_to_wav_bytes(waveform, 24000), 24000

    def decode_csm(self, audio_values: torch.Tensor) -> Tuple[bytes, int]:
        """Decode CSM output (already a waveform). Returns (wav_bytes, 24000)."""
        waveform = audio_values[0].to(torch.float32).cpu().numpy()
        return _numpy_to_wav_bytes(waveform, 24000), 24000

    def decode_bicodec(self, generated_text: str, device: str) -> Tuple[bytes, int]:
        """Decode BiCodec tokens (Spark-TTS) from generated text.

        Extracts bicodec_semantic_N and bicodec_global_N tokens via regex.
        Returns (wav_bytes, sample_rate).
        """
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)

        logger.info(
            f"BiCodec decode: {len(global_matches)} global tokens, {len(semantic_matches)} semantic tokens"
        )
        if len(global_matches) < 10:
            logger.info(f"BiCodec generated text (first 500 chars): {generated_text[:500]}")

        if not semantic_matches:
            raise ValueError("No bicodec_semantic tokens found in generated output")

        semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)

        # Speaker encoder expects exactly 32 global tokens (token_num=32); pad with zeros or truncate.
        GLOBAL_TOKEN_NUM = 32
        if global_matches:
            raw = [int(t) for t in global_matches]
        else:
            raw = []
        if len(raw) < GLOBAL_TOKEN_NUM:
            raw = raw + [0] * (GLOBAL_TOKEN_NUM - len(raw))
        raw = raw[:GLOBAL_TOKEN_NUM]
        global_ids = torch.tensor(raw).long().unsqueeze(0)

        self._bicodec_tokenizer.device = device
        self._bicodec_tokenizer.model.to(device)

        wav_np = self._bicodec_tokenizer.detokenize(
            global_ids.to(device),
            semantic_ids.to(device),
        )
        sr = self._bicodec_tokenizer.config.get("sample_rate", 16000)
        return _numpy_to_wav_bytes(wav_np, sr), sr

    def decode_dac(self, generated_text: str, device: str) -> Tuple[bytes, int]:
        """Decode DAC tokens (OuteTTS) from generated text.

        Extracts c1_N and c2_N codec code tokens via regex.
        Returns (wav_bytes, 24000).
        """
        c1 = list(map(int, re.findall(r"<\|c1_(\d+)\|>", generated_text)))
        c2 = list(map(int, re.findall(r"<\|c2_(\d+)\|>", generated_text)))

        if not c1 or not c2:
            raise ValueError("No DAC code tokens (c1/c2) found in generated output")

        t = min(len(c1), len(c2))
        c1 = c1[:t]
        c2 = c2[:t]

        codes = torch.tensor([[c1, c2]], dtype = torch.int64).to(device)
        with torch.no_grad():
            audio = self._dac_audio_codec.decode(codes)

        waveform = audio.squeeze().cpu().numpy()
        return _numpy_to_wav_bytes(waveform, 24000), 24000

    def decode(
        self,
        audio_type: str,
        device: str,
        token_ids: Optional[list] = None,
        text: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        """Unified decode — dispatches to the right codec decoder.

        ``device`` is what the caller would like. Where the codec is actually
        resident wins: input tensors built on another device fail outright for SNAC
        and DAC, and BiCodec would move a CPU-resident codec onto the card, taking
        the VRAM a CPU RAM load promised not to take.
        """
        device = self._codec_devices.get(audio_type, device)
        if audio_type == "snac":
            if not token_ids:
                raise ValueError("SNAC decoding requires token_ids")
            return self.decode_snac(torch.tensor([token_ids], dtype = torch.long), device)
        elif audio_type == "bicodec":
            if not text:
                raise ValueError("BiCodec decoding requires text")
            return self.decode_bicodec(text, device)
        elif audio_type == "dac":
            if not text:
                raise ValueError("DAC decoding requires text")
            return self.decode_dac(text, device)
        raise ValueError(f"Cannot decode audio_type: {audio_type}")

    # ── Cleanup ──────────────────────────────────────────────────

    def unload(self) -> None:
        """Release all codec models from memory."""
        if self._snac_model is not None:
            del self._snac_model
            self._snac_model = None
        if self._bicodec_tokenizer is not None:
            del self._bicodec_tokenizer
            self._bicodec_tokenizer = None
            self._bicodec_repo_path = None
        if self._bicodec_code_dir is not None:
            deactivate_pinned_package("sparktts", self._bicodec_code_dir)
            self._bicodec_code_dir = None
        if self._dac_audio_codec is not None:
            del self._dac_audio_codec
            self._dac_audio_codec = None
        if self._outetts_code_dir is not None:
            deactivate_pinned_package("outetts", self._outetts_code_dir)
            self._outetts_code_dir = None
        self._codec_devices.clear()
        logger.info("Unloaded all audio codecs")
