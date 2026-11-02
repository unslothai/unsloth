# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in fetching of a provider site's favicon.

Only a host the user typed into a custom provider's Base URL is ever fetched.
The fetch is strict about egress: http(s) only, every DNS answer must be a
global unicast address (loopback, private, link-local, ULA and cloud metadata
are refused), redirects are capped, and bodies are size-capped. All icons are
validated and re-encoded to PNG before being returned as data URLs.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from core.inference.providers import _metadata_host, _resolve_host
from models.providers import MAX_PROVIDER_ICON_DATA_URL_LENGTH

_MAX_FAVICON_BYTES = 512 * 1024
_MAX_REDIRECTS = 3
_FETCH_TIMEOUT_SECONDS = 5.0
_MAX_ICON_EDGE = 128
# Real favicons top out around 1024 px; 2048^2 RGBA is ~16 MB to decode, while a
# 10 KB PNG can declare 9000^2 and still pass Pillow's own bomb limit.
_MAX_SOURCE_PIXELS = 2048 * 2048
_MAX_LINK_CANDIDATES = 4
# httpx timeouts are per read; this bounds a server that drips bytes.
_OVERALL_DEADLINE_SECONDS = 15.0
# IPv4-compatible (deprecated) and NAT64 well-known prefixes that ip.is_global misses.
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/x-icon",
        "image/vnd.microsoft.icon",
    }
)

# Allowed to be replaced in tests; None means the real network stack.
_transport: httpx.AsyncBaseTransport | None = None


class ProviderFaviconError(Exception):
    """Invalid input or an egress/network failure; carries an HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
    ):
        super().__init__(message)
        self.status_code = status_code


class _NoSiteIcon(Exception):
    """No usable icon exists at the target site; this is not an error."""


class _LinkIconParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs = True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "link":
            return
        attributes = {name.lower(): (value or "").strip() for name, value in attrs}
        href = attributes.get("href", "")
        rel = attributes.get("rel", "").lower().split()
        if not href or not rel:
            return
        if "icon" in rel or "apple-touch-icon" in rel:
            self.hrefs.append(href)


def _parent_host(host: str) -> str | None:
    """api.example.com -> example.com; the one subdomain shape we retry on."""
    if host.lower().startswith("api.") and host.count(".") >= 2:
        return host.split(".", 1)[1]
    return None


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return _is_public_ip(ip.ipv4_mapped)
        if ip in _NAT64:
            # DNS64 networks synthesize these for real sites; vet the embedded IPv4.
            return _is_public_ip(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip in _IPV4_COMPATIBLE:
            return False
    return ip.is_global


def _assert_public_origin(url: str) -> str:
    """Return a vetted IP for ``url``'s host, refusing non-http(s), private, or metadata destinations."""
    if not isinstance(url, str) or not url.strip():
        raise ProviderFaviconError("A base URL is required.")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ProviderFaviconError("The URL is malformed.") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ProviderFaviconError("Only http and https sites are supported.")
    host = (parts.hostname or "").strip(".")
    if not host:
        raise ProviderFaviconError("The URL must contain a hostname.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ProviderFaviconError("The URL is malformed.") from exc

    if _metadata_host(host):
        raise ProviderFaviconError("Cloud metadata endpoints cannot be fetched.")
    try:
        # Literal addresses are classified directly without a lookup.
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        resolved = _resolve_host(host, port, scheme)
        if not resolved:
            raise ProviderFaviconError("The host could not be resolved.", status_code = 502)
        addresses = [ipaddress.ip_address(ip.split("%", 1)[0]) for ip in resolved]
    if not addresses or any(not _is_public_ip(ip) for ip in addresses):
        raise ProviderFaviconError("This address is not a public site.")
    return str(addresses[0])


async def _assert_public_origin_async(url: str) -> str:
    return await asyncio.to_thread(_assert_public_origin, url)


async def _capped_request(
    url: str, *, truncate: bool = False
) -> tuple[int, httpx.Headers, str, bytes]:
    for _ in range(_MAX_REDIRECTS + 1):
        # Every hop is re-validated, including image URLs from page <link> tags.
        vetted_ip = await _assert_public_origin_async(url)
        # Dial the vetted IP so a re-resolve cannot rebind; Host and SNI keep the
        # name, so virtual hosting and TLS hostname verification still apply.
        named = httpx.URL(url)
        async with httpx.AsyncClient(
            transport = _transport,
            timeout = _FETCH_TIMEOUT_SECONDS,
            follow_redirects = False,
            trust_env = False,
            headers = {
                "User-Agent": "unsloth-studio/1.0",
                "Accept": "text/html,application/xhtml+xml,*/*",
            },
        ) as client:
            async with client.stream(
                "GET",
                named.copy_with(host = vetted_ip),
                headers = {"Host": named.netloc.decode("ascii")},
                extensions = {"sni_hostname": named.raw_host.decode("ascii")},
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise _NoSiteIcon("Redirect without a Location header.")
                    url = urljoin(url, location)
                    continue
                declared_length = response.headers.get("content-length")
                if (
                    not truncate
                    and declared_length is not None
                    and declared_length.isdigit()
                    and int(declared_length) > _MAX_FAVICON_BYTES
                ):
                    raise _NoSiteIcon("Favicon exceeds the size cap.")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    remaining = _MAX_FAVICON_BYTES - len(body)
                    body.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        if not truncate:
                            raise _NoSiteIcon("Favicon exceeds the size cap.")
                        break
                return (
                    response.status_code,
                    response.headers,
                    url,
                    bytes(body),
                )
    raise _NoSiteIcon("Too many redirects.")


async def _homepage_links(origin: str) -> list[str]:
    status, _, final_url, fragment = await _capped_request(origin + "/", truncate = True)
    if status != 200:
        raise _NoSiteIcon("Homepage did not respond with 200.")
    parser = _LinkIconParser()
    parser.feed(fragment.decode("utf-8", "replace"))
    return [urljoin(final_url, href) for href in parser.hrefs[:_MAX_LINK_CANDIDATES]]


async def _image_bytes(url: str) -> bytes:
    status, headers, _, content = await _capped_request(url)
    if status != 200:
        raise _NoSiteIcon("Favicon did not respond with 200.")
    content_type = headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in _IMAGE_CONTENT_TYPES:
        raise _NoSiteIcon(f"Unsupported image content type: {content_type}")
    if b"<svg" in content[:1024].lower():
        raise _NoSiteIcon("SVG favicons are not supported.")
    return content


def _ico_frame_sizes(content: bytes) -> list[tuple[int, int]]:
    """Header sizes of every ICO frame; Image.open decodes one, and a frame can exceed its directory entry."""
    from PIL import BmpImagePlugin, IcoImagePlugin, PngImagePlugin

    sizes = []
    for entry in IcoImagePlugin.IcoFile(io.BytesIO(content)).entry:
        frame = io.BytesIO(content[entry.offset :])
        if content[entry.offset : entry.offset + 8] == PngImagePlugin._MAGIC:
            sizes.append(PngImagePlugin.PngImageFile(frame).size)
        else:
            sizes.append(BmpImagePlugin.DibImageFile(frame).size)
    return sizes


def _too_large(sizes: list[tuple[int, int]]) -> bool:
    return any(width * height > _MAX_SOURCE_PIXELS for width, height in sizes)


def _png_data_url(content: bytes) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise _NoSiteIcon("Image decoding is unavailable on this server.") from exc
    try:
        # Check header dimensions before any pixel decode (ICO decodes inside open()).
        if content[:4] == b"\0\0\1\0" and _too_large(_ico_frame_sizes(content)):
            raise _NoSiteIcon("The favicon's dimensions are too large.")
        image = Image.open(io.BytesIO(content))
        if _too_large([image.size]):
            raise _NoSiteIcon("The favicon's dimensions are too large.")
        image.load()
    except Exception as exc:
        raise _NoSiteIcon("The byte stream is not a decodable image.") from exc
    image = image.convert("RGBA")
    width, height = image.size
    scale = min(1.0, _MAX_ICON_EDGE / max(width, height))
    if scale < 1.0:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        image = image.resize((width, height), resample)
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format = "PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        if len(data_url) <= MAX_PROVIDER_ICON_DATA_URL_LENGTH:
            return data_url
        width, height = image.size
        if width <= 16 or height <= 16:
            raise _NoSiteIcon("Could not fit the favicon under the size cap.")
        resample = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max(16, width // 2), max(16, height // 2)), resample)


async def fetch_provider_favicon(base_url: str) -> str | None:
    """Fetch a custom provider site's favicon as a PNG data URL.

    Tries the given host first; if that yields no icon and the host starts
    with ``api.``, it retries once on the parent domain. Returns ``None``
    when the site simply has no usable favicon.
    """
    try:
        return await asyncio.wait_for(_fetch_provider_favicon(base_url), _OVERALL_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        raise ProviderFaviconError("The site took too long to respond.", status_code = 502) from None


async def _fetch_provider_favicon(base_url: str) -> str | None:
    origin = base_url.strip()
    await _assert_public_origin_async(origin)
    parts = urlsplit(origin)
    hosts = [parts.hostname]
    parent = _parent_host(parts.hostname)
    if parent and parent != parts.hostname:
        hosts.append(parent)

    seen: set[str] = set()
    for host in hosts:
        if host in seen:
            continue
        seen.add(host)
        root = f"{parts.scheme}://[{host}]" if ":" in host else f"{parts.scheme}://{host}"
        if parts.port:
            root += f":{parts.port}"
        try:
            for icon_url in await _homepage_links(root):
                try:
                    return await asyncio.to_thread(_png_data_url, await _image_bytes(icon_url))
                except _NoSiteIcon:
                    continue
            return await asyncio.to_thread(_png_data_url, await _image_bytes(root + "/favicon.ico"))
        except (_NoSiteIcon, httpx.HTTPError, OSError):
            continue
    return None
