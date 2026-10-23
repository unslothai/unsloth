# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset image URLs are untrusted row content: fetch them only through the SSRF guard."""

import functools
import importlib.abc
import importlib.util
import sys

_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_IMAGE_TIMEOUT_S = 30
# Left to datasets: local paths (no scheme), the Hub, and cloud object stores.
_DATASETS_OWN_SCHEMES = {
    "",
    "file",
    "hf",
    "http",
    "https",
    "s3",
    "gs",
    "gcs",
    "az",
    "abfs",
    "abfss",
    # Archive / compression wrappers datasets chains onto another hop (zip://a.png::<archive>).
    "zip",
    "tar",
    "gzip",
    "bz2",
    "xz",
    "zstd",
    "lz4",
}


def _scheme(url):
    return url.split("://", 1)[0].lower() if "://" in url else ""


def is_network_share(path):
    """UNC and network ``file:`` paths: Windows would open an SMB connection to the host."""
    p = path.replace("\\", "/")
    if p[:5].lower() == "file:":
        p = p[5:]
        if p.startswith("//"):
            host = p[2:].split("/", 1)[0]
            if host.lower() not in ("", "localhost"):
                return True
            p = p[2 + len(host) :]
    if p.startswith(("//?/", "//./")):
        return p[4:].upper().startswith("UNC/")
    return p.startswith("//")


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


def install_guarded_image_decode(hf_image = None):
    """``datasets.Image`` decodes a bytes-less URL cell by fetching it itself, wherever a row is
    read (detection, previews, conversion, streaming, nested features). Route those fetches through
    the guard; Hub URLs keep datasets' own token-aware fetch. Idempotent."""
    try:
        from datasets import config

        if hf_image is None:
            from datasets.features import image as hf_image
    except Exception:
        return
    original = hf_image.Image.decode_example
    if getattr(original, "_unsloth_guarded", False):
        return

    from urllib.parse import urlsplit

    def _origin(url):
        parts = urlsplit(url)
        return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port

    @functools.wraps(original)
    def decode_example(
        self,
        value,
        token_per_repo_id = None,
    ):
        path = value.get("path") if isinstance(value, dict) else None
        if isinstance(path, str) and value.get("bytes") is None:
            # fsspec opens every hop of a chained path, so each one is checked.
            hops = path.split("::")
            for hop in hops:
                if is_network_share(hop):
                    raise OSError(f"Refusing an image path on a network share: {path}")
                if _scheme(hop) not in _DATASETS_OWN_SCHEMES:
                    # ftp://, sftp://, smb:// ... would reach fsspec with no address checks.
                    raise OSError(f"Refusing an image URL with an unsupported scheme: {path}")
            # Exact origin, not a prefix: huggingface.co.attacker.example would match a prefix.
            if any(
                _scheme(hop) in ("http", "https") and _origin(hop) != _origin(config.HF_ENDPOINT)
                for hop in hops
            ):
                if len(hops) > 1:
                    raise OSError(f"Refusing a remote archive image path: {path}")
                value = {"path": None, "bytes": fetch_untrusted_image_bytes(path)}
        return original(self, value, token_per_repo_id)

    decode_example._unsloth_guarded = True
    hf_image.Image.decode_example = decode_example


_IMAGE_MODULE = "datasets.features.image"


class _GuardOnLoad(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        install_guarded_image_decode(module)


class _GuardOnImport(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        name,
        path,
        target = None,
    ):
        if name != _IMAGE_MODULE:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _GuardOnLoad(spec.loader)
        return spec


# Importing datasets here pulled pandas, pyarrow and torch into every Studio start, so the guard is
# applied when datasets' image module loads instead (or now, if something already loaded it).
if _IMAGE_MODULE in sys.modules:
    install_guarded_image_decode(sys.modules[_IMAGE_MODULE])
elif not any(isinstance(finder, _GuardOnImport) for finder in sys.meta_path):
    sys.meta_path.insert(0, _GuardOnImport())
