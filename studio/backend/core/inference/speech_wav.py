# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spoken-reply WAV chunks. Torch-free: the routes import it on GGUF-only installs."""

import base64
import io
import wave


def speech_item(samples, sample_rate: int) -> dict:
    """Clipped, not peak-normalised, so chunks of one reply keep the same loudness."""
    import numpy as np

    pcm = (np.clip(samples.reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return {"speech": base64.b64encode(buf.getvalue()).decode("ascii"), "sample_rate": sample_rate}


def join_speech(items: list) -> tuple:
    """``(wav_bytes, sample_rate)`` of a reply's speech items, in order."""
    if len(items) == 1:
        return base64.b64decode(items[0]["speech"]), items[0]["sample_rate"]
    pcm, params = [], None
    for item in items:
        with wave.open(io.BytesIO(base64.b64decode(item["speech"]))) as clip:
            params = params or clip.getparams()
            pcm.append(clip.readframes(clip.getnframes()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(pcm))
    return buf.getvalue(), items[-1]["sample_rate"]
