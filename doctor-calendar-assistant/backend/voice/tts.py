"""
Text-to-Speech using Piper
Mexican Spanish voice synthesis
"""

import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
import struct

import numpy as np


class TextToSpeech:
    """Piper-based text-to-speech for Mexican Spanish"""

    def __init__(self, config: dict):
        self.model_name = config.get("model", "es_MX-claude-medium")
        self.speaker_id = config.get("speaker_id", 0)
        self.sample_rate = config.get("sample_rate", 22050)

        # Piper paths - will be set up by setup script
        self.piper_path = self._find_piper()
        self.model_path = self._find_model()

    def _find_piper(self) -> Path:
        """Find Piper executable"""
        # Check common locations
        locations = [
            Path.home() / ".local" / "bin" / "piper",
            Path("/usr/local/bin/piper"),
            Path("/opt/homebrew/bin/piper"),
            Path(__file__).parent.parent.parent / "models" / "piper" / "piper",
        ]

        for loc in locations:
            if loc.exists():
                return loc

        # Try to find in PATH
        try:
            result = subprocess.run(["which", "piper"], capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass

        print("    ⚠ Piper not found - TTS will use fallback")
        return None

    def _find_model(self) -> Path:
        """Find Piper voice model"""
        models_dir = Path(__file__).parent.parent.parent / "models" / "piper"

        # Look for Mexican Spanish models
        patterns = [
            f"{self.model_name}.onnx",
            "es_MX*.onnx",
            "es-*.onnx",
        ]

        for pattern in patterns:
            matches = list(models_dir.glob(pattern))
            if matches:
                return matches[0]

        # Check if model exists with full name
        model_path = models_dir / f"{self.model_name}.onnx"
        if model_path.exists():
            return model_path

        print(f"    ⚠ Voice model {self.model_name} not found")
        return None

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio

        Args:
            text: Spanish text to speak

        Returns:
            Raw audio bytes (WAV format)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous TTS synthesis"""

        if self.piper_path and self.model_path:
            return self._synthesize_piper(text)
        else:
            return self._synthesize_fallback(text)

    def _synthesize_piper(self, text: str) -> bytes:
        """Use Piper for TTS"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Run Piper
            cmd = [
                str(self.piper_path),
                "--model", str(self.model_path),
                "--output_file", output_path
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout, stderr = process.communicate(input=text.encode("utf-8"))

            if process.returncode != 0:
                print(f"Piper error: {stderr.decode()}")
                return self._synthesize_fallback(text)

            # Read output file
            with open(output_path, "rb") as f:
                return f.read()

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _synthesize_fallback(self, text: str) -> bytes:
        """
        Fallback TTS using macOS say command or generate silence

        On Mac, uses the built-in 'say' command with Spanish voice
        """
        try:
            # Try macOS say command with Spanish voice
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                aiff_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            # Use macOS say command
            subprocess.run([
                "say",
                "-v", "Paulina",  # Mexican Spanish voice on macOS
                "-o", aiff_path,
                text
            ], check=True, capture_output=True)

            # Convert AIFF to WAV using afconvert (macOS)
            subprocess.run([
                "afconvert",
                "-f", "WAVE",
                "-d", "LEI16@22050",
                aiff_path,
                wav_path
            ], check=True, capture_output=True)

            with open(wav_path, "rb") as f:
                audio_data = f.read()

            os.unlink(aiff_path)
            os.unlink(wav_path)

            return audio_data

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Generate short silence as last resort
            print("    ⚠ TTS fallback failed, generating silence")
            return self._generate_silence(1.0)

    def _generate_silence(self, duration: float) -> bytes:
        """Generate silent WAV audio"""
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples, dtype=np.int16)

        # Create WAV header
        wav_data = bytearray()

        # RIFF header
        wav_data.extend(b'RIFF')
        wav_data.extend(struct.pack('<I', 36 + len(samples) * 2))
        wav_data.extend(b'WAVE')

        # fmt chunk
        wav_data.extend(b'fmt ')
        wav_data.extend(struct.pack('<I', 16))  # chunk size
        wav_data.extend(struct.pack('<H', 1))   # PCM format
        wav_data.extend(struct.pack('<H', 1))   # mono
        wav_data.extend(struct.pack('<I', self.sample_rate))
        wav_data.extend(struct.pack('<I', self.sample_rate * 2))  # byte rate
        wav_data.extend(struct.pack('<H', 2))   # block align
        wav_data.extend(struct.pack('<H', 16))  # bits per sample

        # data chunk
        wav_data.extend(b'data')
        wav_data.extend(struct.pack('<I', len(samples) * 2))
        wav_data.extend(samples.tobytes())

        return bytes(wav_data)

    async def synthesize_stream(self, text: str):
        """
        Stream audio synthesis for long text

        Breaks text into sentences and yields audio chunks

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if sentence.strip():
                audio = await self.synthesize(sentence)
                yield audio
