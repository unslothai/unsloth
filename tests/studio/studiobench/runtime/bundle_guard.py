# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""React dev builds inflate render cost, so the gate refuses them by two independent checks."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# Small: the marker pair sits within a few hundred bytes in every React build.
_WINDOW = 4000

# Accept backticks: the minified production bundle rewrites short strings as template literals.
_RENDERER_RE = re.compile(r'rendererPackageName\s*:\s*["\'`]react-dom["\'`]')
_BUNDLETYPE_RE = re.compile(r"bundleType\s*:\s*([01])")


@dataclass
class BundleVerdict:
    production: bool
    reason: str
    vite_client_status: Optional[int] = None
    bundle_type: Optional[int] = None
    entry_url: Optional[str] = None
    entry_bytes: Optional[int] = None
    checked: list[str] = field(default_factory = list)

    def as_dict(self) -> dict:
        return {
            "production": self.production,
            "reason": self.reason,
            "vite_client_status": self.vite_client_status,
            "bundle_type": self.bundle_type,
            "entry_url": self.entry_url,
            "entry_bytes": self.entry_bytes,
            "checked": self.checked,
            # Attempted flag: bundle_type None and 0 are opposite findings.
            "bundle_type_attempted": self.entry_url is not None,
            "vite_probe_attempted": self.vite_client_status is not None,
        }


def _get(url: str, timeout: float = 20.0) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, headers = {"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, r.read(), (r.headers.get("Content-Type") or "")
    except urllib.error.HTTPError as exc:
        return exc.code, b"", ""
    except Exception:  # noqa: BLE001
        return 0, b"", ""


def _is_vite_client(status: int, body: bytes, content_type: str) -> bool:
    """A 200 alone proves nothing: the SPA serves index.html for any path, so the body must be script."""
    if status != 200 or not body:
        return False
    head = body[:512].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return False
    if "html" in content_type.lower():
        return False
    return (
        "javascript" in content_type.lower()
        or b"import " in body[:2048]
        or b"vite" in body[:2048].lower()
    )


def _entry_urls(base_url: str, html: str) -> list[str]:
    """Every script the index document loads, most-likely-entry first."""
    out: list[str] = []
    for src in re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html):
        url = src if src.startswith("http") else base_url.rstrip("/") + "/" + src.lstrip("/")
        out.append(url)
    # Vite names the real entry chunk via modulepreload when the script tag is a loader shim.
    for href in re.findall(
        r'<link[^>]+rel=["\']modulepreload["\'][^>]+href=["\']([^"\']+)["\']', html
    ):
        url = href if href.startswith("http") else base_url.rstrip("/") + "/" + href.lstrip("/")
        if url not in out:
            out.append(url)
    return out


def check_bundle(base_url: str) -> BundleVerdict:
    base_url = base_url.rstrip("/")
    checked: list[str] = []

    status, vite_body, vite_ct = _get(f"{base_url}/@vite/client")
    is_dev = _is_vite_client(status, vite_body, vite_ct)
    checked.append(
        f"/@vite/client -> {status} ({vite_ct or 'no content-type'}, "
        f"{len(vite_body)} bytes) -> "
        f"{'a real dev client module' if is_dev else 'not a dev client'}"
    )
    if is_dev:
        return BundleVerdict(
            production = False,
            reason = (
                "/@vite/client served a JavaScript module, so this is a Vite dev server. "
                "React's development build inflates the very axis under investigation by "
                "about 3.2x; a measurement here would confirm any hypothesis."
            ),
            vite_client_status = status,
            checked = checked,
        )

    doc_status, doc, _ = _get(f"{base_url}/chat")
    if doc_status != 200 or not doc:
        doc_status, doc, _ = _get(f"{base_url}/")
    checked.append(f"index document -> {doc_status}, {len(doc)} bytes")
    if doc_status != 200 or not doc:
        return BundleVerdict(
            production = False,
            reason = f"could not fetch the index document ({doc_status})",
            vite_client_status = status,
            checked = checked,
        )

    html = doc.decode("utf-8", "replace")
    urls = _entry_urls(base_url, html)
    checked.append(f"{len(urls)} script assets in the document")
    if not urls:
        return BundleVerdict(
            production = False,
            reason = "the index document loads no script assets to inspect",
            vite_client_status = status,
            checked = checked,
        )

    for url in urls:
        asset_status, raw, _ = _get(url, timeout = 60)
        if asset_status != 200 or not raw:
            checked.append(f"{url} -> {asset_status}")
            continue
        text = raw.decode("utf-8", "replace")
        match = _RENDERER_RE.search(text)
        if match is None:
            checked.append(f"{url}: no rendererPackageName marker, {len(raw)} bytes")
            continue
        # Search near the marker in the same chunk; other packages also carry bundleType.
        lo = max(0, match.start() - _WINDOW)
        hi = min(len(text), match.end() + _WINDOW)
        found = _BUNDLETYPE_RE.search(text, lo, hi)
        if found is None:
            checked.append(
                f"{url}: rendererPackageName found, no bundleType within " f"{_WINDOW} chars of it"
            )
            continue
        bundle_type = int(found.group(1))
        checked.append(f"{url}: react-dom bundleType={bundle_type}")
        if bundle_type == 0:
            return BundleVerdict(
                production = True,
                reason = "react-dom reports bundleType 0",
                vite_client_status = status,
                bundle_type = 0,
                entry_url = url,
                entry_bytes = len(raw),
                checked = checked,
            )
        return BundleVerdict(
            production = False,
            reason = (
                f"react-dom reports bundleType {bundle_type}, which is a DEVELOPMENT build "
                "of React. Its per-render bookkeeping inflates the axis under "
                "investigation by about 3.2x."
            ),
            vite_client_status = status,
            bundle_type = bundle_type,
            entry_url = url,
            entry_bytes = len(raw),
            checked = checked,
        )

    return BundleVerdict(
        production = False,
        reason = (
            'no asset carried a `rendererPackageName: "react-dom"` marker, so the build '
            "mode could not be established. Refusing rather than assuming production."
        ),
        vite_client_status = status,
        checked = checked,
    )
