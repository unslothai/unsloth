# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Audio codec loading and decoding for TTS inference.
Supports: SNAC (Orpheus), CSM (Sesame), BiCodec (Spark), DAC (OuteTTS)
"""

import io
import re
import subprocess
import wave
import structlog
from loggers import get_logger
from typing import Optional, Tuple

import numpy as np
import torch

from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


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
        self._dac_audio_codec = None

    def load_codec(
        self,
        audio_type: str,
        device: str = "cuda",
        model_repo_path: Optional[str] = None,
    ) -> None:
        """Load the appropriate codec for the given audio type."""
        if audio_type == "snac":
            self._load_snac(device)
        elif audio_type == "bicodec":
            self._load_bicodec(device, model_repo_path)
        elif audio_type == "dac":
            self._load_dac(device)
        elif audio_type == "csm":
            pass  # CSM decoding is built into the model (output_audio=True)
        else:
            raise ValueError(f"Unknown audio_type: {audio_type}")

    # ── Lazy loaders ─────────────────────────────────────────────

    def _load_snac(self, device: str) -> None:
        if self._snac_model is not None:
            return
        from snac import SNAC

        self._snac_model = (
            SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        )
        logger.info("Loaded SNAC codec (24kHz)")

    def _load_bicodec(self, device: str, model_repo_path: Optional[str] = None) -> None:
        if self._bicodec_tokenizer is not None:
            return
        import os
        import sys

        # Clone SparkAudio/Spark-TTS GitHub repo for the sparktts Python package
        # (same approach as training — the HF model repos don't contain the package)
        spark_code_dir = os.path.join(
            os.path.dirname(model_repo_path or "."), "Spark-TTS"
        )
        sparktts_pkg = os.path.join(spark_code_dir, "sparktts")
        if not os.path.isdir(sparktts_pkg):
            logger.info(f"Cloning SparkAudio/Spark-TTS to {spark_code_dir}...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/SparkAudio/Spark-TTS",
                    spark_code_dir,
                ],
                check = True,
                **_windows_hidden_subprocess_kwargs(),
            )

        if spark_code_dir not in sys.path:
            sys.path.insert(0, spark_code_dir)

        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        # BiCodecTokenizer needs the MODEL repo path (contains BiCodec/ weights)
        tokenizer_path = model_repo_path or spark_code_dir
        self._bicodec_repo_path = tokenizer_path
        self._bicodec_tokenizer = BiCodecTokenizer(tokenizer_path, device)
        logger.info(f"Loaded BiCodec tokenizer from {tokenizer_path}")

    def _load_dac(self, device: str) -> None:
        if self._dac_audio_codec is not None:
            return
        import os
        import sys

        # Clone OuteTTS repo (same pattern as Spark-TTS / BiCodec)
        # The pip package has problematic dependencies; the notebook clones and
        # removes gguf_model.py, interface.py, __init__.py before importing.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        outetts_code_dir = os.path.join(base_dir, "OuteTTS")
        outetts_pkg = os.path.join(outetts_code_dir, "outetts")
        if not os.path.isdir(outetts_pkg):
            logger.info(f"Cloning edwko/OuteTTS to {outetts_code_dir}...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/edwko/OuteTTS",
                    outetts_code_dir,
                ],
                check = True,
                **_windows_hidden_subprocess_kwargs(),
            )
            # Remove files that pull in heavy / incompatible dependencies
            # (matches notebook: gguf_model.py is under models/, others under outetts/)
            remove_paths = [
                os.path.join(outetts_pkg, "models", "gguf_model.py"),
                os.path.join(outetts_pkg, "interface.py"),
                os.path.join(outetts_pkg, "__init__.py"),
            ]
            for fpath in remove_paths:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    logger.info(f"Removed {fpath}")

        if outetts_code_dir not in sys.path:
            sys.path.insert(0, outetts_code_dir)

        from outetts.version.v3.audio_processor import AudioProcessor
        from outetts.models.config import ModelConfig as OuteTTSModelConfig

        dummy_config = OuteTTSModelConfig(
            tokenizer_path = "OuteAI/Llama-OuteTTS-1.0-1B",
            device = device,
            audio_codec_path = None,
        )
        processor = AudioProcessor(config = dummy_config)
        self._dac_audio_codec = processor.audio_codec
        logger.info("Loaded DAC audio codec")

    # ── Decoders ─────────────────────────────────────────────────

    def decode_snac(
        self, generated_ids: torch.Tensor, device: str
    ) -> Tuple[bytes, int]:
        """
        Decode SNAC tokens (Orpheus) into WAV bytes.

        generated_ids: full model output including prompt tokens.
        Looks for START_OF_SPEECH (128257) marker, extracts codes after it,
        strips EOS (128258), redistributes 7-per-frame codes into 3 SNAC layers.

        Returns (wav_bytes, 24000).
        """
        # Find START_OF_SPEECH token (128257)
        token_indices = (generated_ids == 128257).nonzero(as_tuple = True)
        if len(token_indices[1]) > 0:
            cropped = generated_ids[:, token_indices[1][-1] + 1 :]
        else:
            # Gracefully fall back to using entire output if marker not found
            logger.warning(
                "No START_OF_SPEECH token (128257) found — using full generated output"
            )
            cropped = generated_ids
        row = cropped[0]

        # Remove EOS tokens (128258)
        row = row[row != 128258]

        # Trim to multiple of 7
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
            torch.tensor(layer).unsqueeze(0).to(device)
            for layer in [layer_1, layer_2, layer_3]
        ]

        with torch.no_grad():
            audio = self._snac_model.decode(snac_codes)

        waveform = audio.squeeze().cpu().numpy()
        return _numpy_to_wav_bytes(waveform, 24000), 24000

    def decode_csm(self, audio_values: torch.Tensor) -> Tuple[bytes, int]:
        """
        Decode CSM output (already a waveform from model.generate(output_audio=True)).
        Returns (wav_bytes, 24000).
        """
        waveform = audio_values[0].to(torch.float32).cpu().numpy()
        return _numpy_to_wav_bytes(waveform, 24000), 24000

    def decode_bicodec(self, generated_text: str, device: str) -> Tuple[bytes, int]:
        """
        Decode BiCodec tokens (Spark-TTS) from generated text.
        Extracts bicodec_semantic_N and bicodec_global_N tokens via regex.
        Returns (wav_bytes, sample_rate).
        """
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)

        logger.info(
            f"BiCodec decode: {len(global_matches)} global tokens, {len(semantic_matches)} semantic tokens"
        )
        if len(global_matches) < 10:
            logger.info(
                f"BiCodec generated text (first 500 chars): {generated_text[:500]}"
            )

        if not semantic_matches:
            raise ValueError("No bicodec_semantic tokens found in generated output")

        semantic_ids = (
            torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
        )

        # Speaker encoder expects exactly 32 global tokens (token_num=32 in BiCodec config).
        # Pad with zeros or truncate to 32.
        GLOBAL_TOKEN_NUM = 32
        if global_matches:
            raw = [int(t) for t in global_matches]
        else:
            raw = []
        if len(raw) < GLOBAL_TOKEN_NUM:
            raw = raw + [0] * (GLOBAL_TOKEN_NUM - len(raw))
        raw = raw[:GLOBAL_TOKEN_NUM]
        global_ids = torch.tensor(raw).long().unsqueeze(0)  # (1, 32)

        self._bicodec_tokenizer.device = device
        self._bicodec_tokenizer.model.to(device)

        wav_np = self._bicodec_tokenizer.detokenize(
            global_ids.to(device),
            semantic_ids.to(device),
        )
        sr = self._bicodec_tokenizer.config.get("sample_rate", 16000)
        return _numpy_to_wav_bytes(wav_np, sr), sr

    def decode_dac(self, generated_text: str, device: str) -> Tuple[bytes, int]:
        """
        Decode DAC tokens (OuteTTS) from generated text.
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
        """Unified decode — dispatches to the right codec decoder."""
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
        if self._dac_audio_codec is not None:
            del self._dac_audio_codec
            self._dac_audio_codec = None
        logger.info("Unloaded all audio codecs")
