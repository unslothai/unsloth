# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset image URLs are untrusted row content: fetch them only through the SSRF guard."""

import functools
import importlib.abc
import importlib.util
import os
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


# Same opt-in as unsloth_zoo's media fetch: trusted LAN hosts, shares and archives load as datasets loads them.
ALLOW_PRIVATE_ENV = "UNSLOTH_ALLOW_PRIVATE_URL_FETCH"
_OPT_IN_HINT = f"Set {ALLOW_PRIVATE_ENV}=1 to load images from private hosts or shares you trust."


def private_sources_allowed():
    """Single-user installs only, so a managed account's row cannot reach the owner's LAN; fails closed."""
    if os.environ.get(ALLOW_PRIVATE_ENV) != "1":
        return False
    try:
        from auth.policy import installation_has_managed_accounts
        return not installation_has_managed_accounts()
    except Exception:  # noqa: BLE001 - unreadable account store keeps the guard on
        return False


def _scheme(url):
    return url.split("://", 1)[0].lower() if "://" in url else ""


_LOCAL_SHARE_HOSTS = ("", "localhost", "wsl$", "wsl.localhost")
_WINDOWS = os.name == "nt"


def is_network_share(path):
    """Windows UNC / network ``file:`` paths (SMB); ``\\\\wsl$`` and non-Windows ``//x`` are local."""
    if not _WINDOWS:
        return False
    p = path.replace("\\", "/")
    if p[:5].lower() == "file:":
        p = p[5:]
        if p.startswith("//"):
            host = p[2:].split("/", 1)[0]
            if host.lower() not in _LOCAL_SHARE_HOSTS:
                return True
            p = p[2 + len(host) :]
    if p.startswith(("//?/", "//./")):
        p = p[4:]
        if not p.upper().startswith("UNC/"):
            return False
        p = "//" + p[4:]
    return p.startswith("//") and p[2:].split("/", 1)[0].lower() not in _LOCAL_SHARE_HOSTS


def guard_url(url):
    """Percent-encode the path, query and fragment as a browser would; the host is left for IDNA."""
    from urllib.parse import quote, urlsplit, urlunsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    keep = "/?#[]@!$&'()*+,;=%~:"
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(parts.path, safe = keep),
            quote(parts.query, safe = keep),
            quote(parts.fragment, safe = keep),
        )
    )


def _private_refusal(error):
    lowered = error.lower()
    return error.startswith("Blocked:") and ("non-public" in lowered or "local" in lowered)


def fetch_untrusted_image_bytes(url):
    """Public addresses only, pinned to the resolved IP, every redirect re-checked, size capped."""
    import time

    from core.inference.tools import _fetch_url_raw

    error, body, _ = _fetch_url_raw(
        guard_url(url),
        timeout = _IMAGE_TIMEOUT_S,
        extra_headers = {"Accept": "image/*"},
        deadline = time.monotonic() + _IMAGE_TIMEOUT_S * 2,
        raw_bytes_max = _MAX_IMAGE_BYTES,
    )
    if error is not None:
        raise OSError(f"{error} {_OPT_IN_HINT}" if _private_refusal(error) else error)
    return body


def open_opted_in_url(
    url,
    method = "GET",
    timeout = _IMAGE_TIMEOUT_S,
):
    """The owner's opt-in: no address checks, but quoted, IDNA-encoded and time bounded."""
    import urllib.request
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(guard_url(url))
    userinfo, at, hostport = parts.netloc.rpartition("@")
    if not hostport.isascii():  # never an IPv6 literal, so the first colon starts the port
        name, colon, port = hostport.partition(":")
        hostport = name.encode("idna").decode("ascii") + colon + port
    url = urlunsplit(parts._replace(netloc = userinfo + at + hostport))
    request = urllib.request.Request(url, method = method, headers = {"Accept": "image/*"})
    return urllib.request.urlopen(request, timeout = timeout)


def fetch_opted_in_image_bytes(url):
    """The owner's opt-in fetch, with the guarded fetch's size cap and deadline."""
    import time

    deadline = time.monotonic() + _IMAGE_TIMEOUT_S * 2
    chunks, total = [], 0
    with open_opted_in_url(url) as response:
        if int(response.headers.get("Content-Length") or 0) > _MAX_IMAGE_BYTES:
            total = _MAX_IMAGE_BYTES + 1
        while total <= _MAX_IMAGE_BYTES:
            if time.monotonic() > deadline:
                raise OSError(f"Timed out fetching image: {url}")
            chunk = response.read1(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    if total > _MAX_IMAGE_BYTES:
        raise OSError(f"Image is larger than {_MAX_IMAGE_BYTES >> 20} MB: {url}")
    return b"".join(chunks)


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
        if isinstance(path, str) and value.get("bytes") is None and not private_sources_allowed():
            # fsspec opens every hop of a chained path, so each one is checked.
            hops = path.split("::")
            for hop in hops:
                if is_network_share(hop):
                    raise OSError(
                        f"Refusing an image path on a network share: {path}. {_OPT_IN_HINT}"
                    )
                if _scheme(hop) not in _DATASETS_OWN_SCHEMES:
                    # ftp://, sftp://, smb:// ... would reach fsspec with no address checks.
                    raise OSError(
                        f"Refusing an image URL with an unsupported scheme: {path}. {_OPT_IN_HINT}"
                    )
            # Exact origin, not a prefix: huggingface.co.attacker.example would match a prefix.
            if any(
                _scheme(hop) in ("http", "https") and _origin(hop) != _origin(config.HF_ENDPOINT)
                for hop in hops
            ):
                if len(hops) > 1:
                    raise OSError(f"Refusing a remote archive image path: {path}. {_OPT_IN_HINT}")
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
