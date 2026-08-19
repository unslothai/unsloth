# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decode `datasets` Audio columns with soundfile when torchcodec cannot load.

`datasets` 4.x decodes audio only through torchcodec, which needs an FFmpeg full-shared
install to dlopen its native libraries. Windows has none by default, so
`disable_torchcodec_if_broken` clears `datasets.config.TORCHCODEC_AVAILABLE` and every
audio column raises, blocking the dataset format check and all six audio trainer paths
on an otherwise working host. A soundfile decoder restores the pre-4.0 output contract,
`{"path", "array", "sampling_rate"}`, which is what those callers already read.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

_installed = False
_ORIGINAL_ENCODE = None
# The read-and-patch below must happen once. Two first-time callers (two dataset
# format checks land on the threadpool together) can both pass the _installed
# check, and the second then captures the already-installed shim as
# _ORIGINAL_ENCODE, so its fallback branch recurses into itself until
# RecursionError.
_install_lock = threading.Lock()


def _token_for_url(path: str, token_per_repo_id: Optional[dict]) -> Any:
    """Pick the credential belonging to the repository this URL points at.

    A mapping holds one entry per source repo, and `concatenate_datasets` or
    `interleave_datasets` over streaming splits puts several in it at once, so taking an
    arbitrary value would send one repo's token to another repo's host. Resolved the way
    `datasets.Audio.decode_example` does it, from the repo id embedded in the URL.
    """
    if not token_per_repo_id:
        return None
    from datasets import config
    from datasets.utils.py_utils import string_to_dict

    # A chained URL ("zip://inner::https://outer") names its host in the last segment.
    source_url = path.split("::")[-1]
    pattern = (
        config.HUB_DATASETS_URL
        if source_url.startswith(config.HF_ENDPOINT)
        else config.HUB_DATASETS_HFFS_URL
    )
    try:
        fields = string_to_dict(source_url, pattern)
    except ValueError:
        # Older `datasets` raise here instead of returning None.
        fields = None
    if fields is None:
        # Not a Hub URL, so no repo id to key on. One entry is unambiguous and is the
        # shape every caller in this codebase passes; more than one is not guessable.
        values = list(token_per_repo_id.values())
        return values[0] if len(values) == 1 else None
    return token_per_repo_id.get(fields["repo_id"])


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
        source = xopen(
            path,
            "rb",
            download_config = DownloadConfig(token = _token_for_url(path, token_per_repo_id)),
        )

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
    """Stand-in for `datasets.Audio.encode_example` that never needs FFmpeg.

    The audio VLM path maps without `remove_columns`, so reading `["array"]` writes the
    decoded value back and `cast_storage` re-encodes it through torchcodec's encoder,
    failing a run the decoder above had just unblocked.

    The plain path/bytes forms need no encoder at all, but `datasets` imports
    `torchcodec.encoders` at the top of `encode_example` before it looks at the value, so
    casting a column of file paths raises on a broken host too. Those are handled here
    rather than delegated. Only an `AudioDecoder` value falls through, which genuinely
    needs torchcodec and cannot arrive while this shim is installed.
    """
    import io
    from pathlib import Path

    import soundfile as sf

    if isinstance(value, str):
        return {"bytes": None, "path": value}
    if isinstance(value, Path):
        return {"bytes": None, "path": str(value.absolute())}
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": bytes(value), "path": None}
    if isinstance(value, dict) and value.get("array") is not None:
        buf = io.BytesIO()
        sf.write(buf, value["array"], value["sampling_rate"], format = "WAV")
        return {"bytes": buf.getvalue(), "path": value.get("path")}
    if isinstance(value, dict) and ("bytes" in value or "path" in value):
        return {"bytes": value.get("bytes"), "path": value.get("path")}
    return _ORIGINAL_ENCODE(self, value)


def ensure_audio_decoding() -> bool:
    """Install the soundfile decoder when torchcodec is unusable. Idempotent.

    False means neither backend is importable, and the caller should report that rather
    than let a decode raise deep inside `datasets`.
    """
    global _installed
    try:
        from datasets import config
        from datasets.features.audio import Audio
    except ImportError:
        return False
    # `datasets` < 4 (pyproject still allows >=3.4.1) decodes through soundfile itself and
    # defines no TORCHCODEC_AVAILABLE, so the read below raised AttributeError at the
    # unguarded call site. Nothing to install there, so say so.
    if not hasattr(config, "TORCHCODEC_AVAILABLE"):
        return True
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
    with _install_lock:
        # Re-check under the lock: the loser of the race must not re-capture.
        if _installed:
            return True
        _ORIGINAL_ENCODE = Audio.encode_example
        Audio.decode_example = _decode_with_soundfile
        Audio.encode_example = _encode_with_soundfile
        _installed = True
    logger.info("torchcodec is unusable; decoding dataset audio with soundfile")
    return True
