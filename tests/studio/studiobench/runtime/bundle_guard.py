# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse to benchmark a development build.

A Vite dev server MANUFACTURES the symptom this tool exists to find. React's development build
does per-render bookkeeping the production build does not, and the measured inflation is about
3.2x on exactly the axis under investigation. A number taken against it is not a smaller version
of the truth, it is a different phenomenon with the same units, and it would confirm any
hypothesis you brought to it.

TWO independent checks, because either alone has a hole.

1. **`/@vite/client` answers 200.** Only a dev server serves that path. Cheap, decisive, and it
   catches a dev server even when the entry asset happens to look fine.

2. **`bundleType: 0` in the SAME chunk as `rendererPackageName: "react-dom"`.** React's renderer
   registers itself with the DevTools hook carrying a `bundleType`, 0 for production and 1 for
   development. It has to be read from the same chunk as the `react-dom` marker: a large app
   bundle contains several renderers' worth of strings, and `bundleType:1` from some other
   package proves nothing about react-dom.

**Do NOT grep for `jsxDEV`.** It false-positives. `hast-util-to-jsx-runtime`, which Streamdown
pulls in and which is in every production bundle of this app, ships its own option guard naming
`jsxDEV` as a supported entry point. A `jsxDEV` grep therefore fails a perfectly good production
build, and the natural response to a gate that fails on a correct build is to switch the gate off.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# Kept small: an app entry chunk is megabytes and the marker pair sits within a few hundred bytes
# of each other in every React build, minified or not.
_WINDOW = 4000

# BACKTICKS as well as quotes. Unsloth's production bundle is minified with a pass that rewrites
# short string literals as template literals, so the marker on the shipped build reads
# ``rendererPackageName:`react-dom` `` and a quote-only pattern misses it on every asset -- which
# reads as "build mode could not be established" on a perfectly good production build, i.e. the
# gate fails exactly where it is supposed to pass. Verified against the shipped bundle.
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
            # Attempted flags, because "bundle_type is None" and "bundle_type is 0" are opposite
            # findings and a bare null cannot say which.
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
    """A 200 alone does NOT mean a dev server, and assuming it does breaks the gate.

    Unsloth serves its built frontend as a single-page app, so ANY unknown path returns 200 with
    `index.html` -- measured here against two production Unsloth instances, both of which answered 200 to
    `/@vite/client` and were failed as dev servers by the first version of this check. A gate that
    fails every correct build is a gate someone turns off.

    A real Vite dev server serves that path as a JavaScript module. So the probe is: 200, AND the
    body is script rather than the SPA's HTML fallback.
    """
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
    # A modulepreload is how Vite names the real entry chunk when the script tag points at a
    # loader shim, so it is worth following when the script tags carry nothing.
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
        # The SAME chunk, and within a small window of the marker. A bundle this size carries
        # several `bundleType` occurrences and picking the wrong one is picking a different
        # package's answer to a question about react-dom.
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
