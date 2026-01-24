"""
Speech-to-Text using Whisper
Optimized for Spanish (Mexican) on Apple Silicon
"""

import asyncio
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf


class SpeechToText:
    """Whisper-based speech-to-text for Spanish"""

    def __init__(self, config: dict):
        self.model_name = config.get("model", "large-v3")
        self.language = config.get("language", "es")
        self.device = config.get("device", "auto")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model - uses faster-whisper for better performance on Mac"""
        try:
            from faster_whisper import WhisperModel

            # Determine compute type based on device
            if self.device == "auto":
                # Use int8 for CPU/Metal, float16 for CUDA
                compute_type = "int8"
                device = "cpu"  # faster-whisper uses CPU but is still fast
            else:
                compute_type = "float16"
                device = self.device

            print(f"    Loading Whisper {self.model_name} on {device}...")
            self.model = WhisperModel(
                self.model_name, device = device, compute_type = compute_type
            )
            print(f"    ✓ Whisper loaded successfully")

        except ImportError:
            # Fallback to standard whisper
            print("    faster-whisper not available, using standard whisper...")
            import whisper

            self.model = whisper.load_model(self.model_name)
            self._use_faster_whisper = False
        else:
            self._use_faster_whisper = True

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text

        Args:
            audio_data: Raw audio bytes (16kHz, mono, 16-bit PCM)

        Returns:
            Transcribed text in Spanish
        """
        # Run transcription in thread pool to not block async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio_data)

    def _transcribe_sync(self, audio_data: bytes) -> str:
        """Synchronous transcription"""
        # Convert bytes to numpy array
        audio_array = (
            np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0
        )

        # Save to temp file (whisper expects file path)
        with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, 16000)

        try:
            if self._use_faster_whisper:
                segments, info = self.model.transcribe(
                    temp_path,
                    language = self.language,
                    beam_size = 5,
                    vad_filter = True,  # Filter out silence
                    vad_parameters = dict(min_silence_duration_ms = 500, speech_pad_ms = 200),
                )
                # Combine all segments
                text = " ".join([segment.text for segment in segments])
            else:
                # Standard whisper
                result = self.model.transcribe(
                    temp_path, language = self.language, fp16 = False
                )
                text = result["text"]

            return text.strip()

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    async def transcribe_stream(self, audio_chunks):
        """
        Transcribe streaming audio chunks

        Args:
            audio_chunks: Async generator of audio bytes

        Yields:
            Partial transcriptions as they become available
        """
        buffer = bytearray()
        chunk_duration_ms = 3000  # Process every 3 seconds

        async for chunk in audio_chunks:
            buffer.extend(chunk)

            # Calculate duration of buffered audio
            # 16kHz, 16-bit mono = 32000 bytes per second
            duration_ms = len(buffer) / 32

            if duration_ms >= chunk_duration_ms:
                text = await self.transcribe(bytes(buffer))
                if text:
                    yield text
                buffer = bytearray()

        # Process remaining audio
        if len(buffer) > 0:
            text = await self.transcribe(bytes(buffer))
            if text:
                yield text
