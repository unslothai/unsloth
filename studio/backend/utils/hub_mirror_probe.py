# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Log a warning when a custom Hub endpoint redirects to huggingface.co (hf-mirror.com does outside
mainland China), which huggingface_hub otherwise follows silently. No token, no redirect followed."""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional
from urllib.parse import urljoin, urlsplit

import httpx

from loggers import get_logger
from utils.hf_endpoint import is_loopback_host

logger = get_logger(__name__)

PROBE_REPO = "openai-community/gpt2"
PROBE_FILE = "model.safetensors"
_OFFICIAL_DOMAINS = ("huggingface.co", "hf.co")
_TIMEOUT = httpx.Timeout(10.0)
PROBE_ENV = "UNSLOTH_STUDIO_HUB_PROBE"

# Tests swap in an httpx.MockTransport.
transport: Optional[httpx.BaseTransport] = None

_probed: set[str] = set()
_probed_lock = threading.Lock()


def is_official_host(hostname: Optional[str]) -> bool:
    host = (hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in _OFFICIAL_DOMAINS)


def _official_redirect(client: httpx.Client, method: str, url: str) -> Optional[str]:
    """The huggingface.co URL ``url`` redirects to, or None when the endpoint answers itself."""
    response = client.request(method, url)
    location = response.headers.get("location")
    if not response.is_redirect or not location:
        return None
    target = urljoin(url, location)
    return target if is_official_host(urlsplit(target).hostname) else None


def find_official_redirects(endpoint: str) -> list[tuple[str, str]]:
    """``(what, target)`` for each probed route the endpoint sends to huggingface.co."""
    findings = []
    with httpx.Client(
        timeout = _TIMEOUT,
        follow_redirects = False,
        transport = transport,
        headers = {"User-Agent": "unsloth-studio"},
    ) as client:
        for what, method, url in (
            ("Hub API calls", "GET", f"{endpoint}/api/models/{PROBE_REPO}"),
            ("model downloads", "HEAD", f"{endpoint}/{PROBE_REPO}/resolve/main/{PROBE_FILE}"),
        ):
            try:
                target = _official_redirect(client, method, url)
            except httpx.HTTPError as exc:
                logger.debug("Hugging Face endpoint probe of %s failed: %s", url, exc)
                continue
            if target is not None:
                findings.append((what, target))
    return findings


def _probe(endpoint: str) -> None:
    try:
        findings = find_official_redirects(endpoint)
    except Exception as exc:  # noqa: BLE001 - an unreachable mirror is not this probe's finding
        logger.debug("Hugging Face endpoint probe of %s failed: %s", endpoint, exc)
        return
    if not findings:
        return
    logger.warning(
        "Hugging Face endpoint %s redirects %s to %s, so they do not go through the mirror "
        "from this network. They still work while huggingface.co is reachable; if it is not, "
        "use a mirror that serves this region, or ModelScope as the model source.",
        endpoint,
        " and ".join(what for what, _ in findings),
        ", ".join(sorted({urlsplit(target).hostname for _, target in findings})),
    )


def should_probe(endpoint: str) -> bool:
    host = urlsplit(endpoint or "").hostname
    if not host or is_official_host(host) or is_loopback_host(host):
        return False
    off = ("1", "true", "yes", "on")
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in off:
        return False
    return os.environ.get(PROBE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def warn_if_mirror_redirects_to_official(
    endpoint: str, *, start: Callable[[threading.Thread], None] = threading.Thread.start
) -> bool:
    """Probe ``endpoint`` in the background once per process. True when a probe was started."""
    endpoint = (endpoint or "").rstrip("/")
    if not should_probe(endpoint):
        return False
    with _probed_lock:
        if endpoint in _probed:
            return False
        _probed.add(endpoint)
    start(threading.Thread(target = _probe, args = (endpoint,), name = "hub-mirror-probe", daemon = True))
    return True
