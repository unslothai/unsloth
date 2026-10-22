# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset image URLs are untrusted row content: fetch them only through the SSRF guard."""

import functools

_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_IMAGE_TIMEOUT_S = 30


def fetch_untrusted_image_bytes(url):
    """Public addresses only, pinned to the resolved IP, every redirect re-checked, size capped."""
    import time

    from core.inference.tools import _fetch_url_raw

    error, body, _ = _fetch_url_raw(
        url,
        timeout = _IMAGE_TIMEOUT_S,
        extra_headers = {"Accept": "image/*"},
        deadline = time.monotonic() + _IMAGE_TIMEOUT_S * 2,
        raw_bytes_max = _MAX_IMAGE_BYTES,
    )
    if error is not None:
        raise OSError(error)
    return body


def install_guarded_image_decode():
    """``datasets.Image`` decodes a bytes-less URL cell by fetching it itself, wherever a row is
    read (detection, previews, conversion, streaming, nested features). Route those fetches through
    the guard; Hub URLs keep datasets' own token-aware fetch. Idempotent."""
    try:
        from datasets import config
        from datasets.features import image as hf_image
    except Exception:
        return
    original = hf_image.Image.decode_example
    if getattr(original, "_unsloth_guarded", False):
        return

    @functools.wraps(original)
    def decode_example(
        self,
        value,
        token_per_repo_id = None,
    ):
        path = value.get("path") if isinstance(value, dict) else None
        if isinstance(path, str) and value.get("bytes") is None:
            source = path.split("::")[-1]
            if source.startswith(("http://", "https://")) and not source.startswith(
                config.HF_ENDPOINT
            ):
                if "::" in path:
                    raise OSError(f"Refusing a remote archive image path: {path}")
                value = {"path": None, "bytes": fetch_untrusted_image_bytes(path)}
        return original(self, value, token_per_repo_id)

    decode_example._unsloth_guarded = True
    hf_image.Image.decode_example = decode_example


install_guarded_image_decode()
