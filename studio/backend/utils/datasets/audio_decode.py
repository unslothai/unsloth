# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode `datasets` Audio columns with soundfile when torchcodec cannot load.

`datasets` 4.x decodes audio only through torchcodec, and torchcodec needs an
FFmpeg full-shared install to dlopen its native libraries. Windows has none by
default, so `unsloth.import_fixes.disable_torchcodec_if_broken` clears
`datasets.config.TORCHCODEC_AVAILABLE` and every audio column then raises
"To support decoding audio data, please install 'torchcodec'." That blocks the
dataset format check and all six audio trainer paths on an otherwise working
host. Installing a soundfile decoder restores the pre-4.0 output contract,
`{"path", "array", "sampling_rate"}`, which is what those callers already read.
"""

from __future__ import annotations

from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

_installed = False
_ORIGINAL_ENCODE = None


def _decode_with_soundfile(
    self,
    value: dict,
    token_per_repo_id: Optional[dict] = None,
) -> dict:
    """Stand-in for `datasets.Audio.decode_example` that never needs FFmpeg."""
    import io

    import numpy as np
    import soundfile as sf
    from datasets.download.download_config import DownloadConfig
    from datasets.utils.file_utils import is_local_path, xopen

    if not self.decode:
        raise RuntimeError(
            "Decoding is disabled for this feature. Please use Audio(decode=True) instead."
        )
    path, raw = value["path"], value["bytes"]
    if path is None and raw is None:
        raise ValueError(
            f"An audio sample should have one of 'path' or 'bytes' but both are None in {value}."
        )

    if raw is not None:
        source: Any = io.BytesIO(raw)
    elif is_local_path(path):
        source = path
    else:
        # Callers pass a single-repo mapping, so the token needs no URL-to-repo-id parsing.
        token = next(iter(token_per_repo_id.values()), None) if token_per_repo_id else None
        source = xopen(path, "rb", download_config = DownloadConfig(token = token))

    array, sampling_rate = sf.read(source, dtype = "float32", always_2d = False)
    if array.ndim > 1:
        # soundfile returns (frames, channels); torchcodec returns (channels, frames).
        array = np.mean(array, axis = -1)
    target = self.sampling_rate
    if target and sampling_rate != target:
        import librosa
        array = librosa.resample(array, orig_sr = sampling_rate, target_sr = target)
        sampling_rate = target
    return {"path": path, "array": array, "sampling_rate": sampling_rate}


def _encode_with_soundfile(self, value) -> dict:
    """Stand-in for `datasets.Audio.encode_example`, for the array form only.

    The audio VLM path maps without `remove_columns`, so reading `["array"]` writes the
    decoded value back and `cast_storage` re-encodes it. torchcodec's encoder is a hard
    requirement there, which would fail a run the decoder above had just unblocked.
    """
    import io

    import soundfile as sf

    if isinstance(value, dict) and value.get("array") is not None:
        buf = io.BytesIO()
        sf.write(buf, value["array"], value["sampling_rate"], format = "WAV")
        return {"bytes": buf.getvalue(), "path": value.get("path")}
    if isinstance(value, dict) and ("bytes" in value or "path" in value):
        return {"bytes": value.get("bytes"), "path": value.get("path")}
    return _ORIGINAL_ENCODE(self, value)


def ensure_audio_decoding() -> bool:
    """Install the soundfile decoder when torchcodec is unusable. Idempotent.

    Returns True when audio columns can be decoded, either natively or through
    the replacement. False means neither backend is importable, and the caller
    should report that rather than let a decode raise deep inside `datasets`.
    """
    global _installed
    try:
        from datasets import config
        from datasets.features.audio import Audio
    except ImportError:
        return False
    if config.TORCHCODEC_AVAILABLE and not _installed:
        try:
            # config only ran find_spec, and an installed torchcodec whose native libraries
            # cannot dlopen still passes that. The API process never imports unsloth, so
            # disable_torchcodec_if_broken has not corrected the flag here.
            from datasets.features._torchcodec import AudioDecoder  # noqa: F401
        except (ImportError, OSError, RuntimeError) as exc:
            logger.info("torchcodec is installed but unusable (%s)", exc)
            config.TORCHCODEC_AVAILABLE = False
    if config.TORCHCODEC_AVAILABLE:
        return True
    if _installed:
        return True
    try:
        # librosa too: every trainer path casts to a target rate, so a decoder that cannot
        # resample would raise from inside `datasets` exactly where this returns False.
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
    except (ImportError, OSError) as exc:
        logger.warning("No usable audio decoder: torchcodec is broken and %s", exc)
        return False
    global _ORIGINAL_ENCODE
    _ORIGINAL_ENCODE = Audio.encode_example
    Audio.decode_example = _decode_with_soundfile
    Audio.encode_example = _encode_with_soundfile
    _installed = True
    logger.info("torchcodec is unusable; decoding dataset audio with soundfile")
    return True
