# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the opt-in provider site favicon fetch (mock HTTP, no real network)."""

import asyncio
import base64
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from core.inference import provider_favicon as favicon
from core.inference.provider_favicon import ProviderFaviconError

_EXT_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}


def _png_bytes(size = (64, 64), fill = (120, 30, 200)) -> bytes:
    image = Image.new("RGBA", size, fill)
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    return buffer.getvalue()


class _Site:
    """In-memory host for the mock transport: routes keyed by (host, path)."""

    def __init__(self):
        self.routes: dict[tuple[str, str], tuple[int, bytes, dict[str, str]]] = {}
        self.requests: list[str] = []
        self.raw: list[httpx.Request] = []

    def route(
        self,
        host,
        path,
        status = 200,
        content = b"",
        headers = None,
    ):
        self.routes[(host, path)] = (status, content, headers or {})

    def handler(self, request: httpx.Request) -> httpx.Response:
        # The client dials the vetted IP; the Host header carries the site name.
        self.raw.append(request)
        named = httpx.URL(
            f"{request.url.scheme}://{request.headers['host']}{request.url.raw_path.decode()}"
        )
        self.requests.append(str(named))
        route = self.routes.get((named.host, named.path))
        if route is None:
            return httpx.Response(404)
        status, content, headers = route
        if "content-type" not in headers:
            content_type = _EXT_CONTENT_TYPES.get(Path(request.url.path).suffix.lower())
            if content_type:
                headers = {**headers, "content-type": content_type}
        return httpx.Response(status, content = content, headers = headers)


@pytest.fixture(autouse = True)
def _mock_transport(monkeypatch: pytest.MonkeyPatch):
    site = _Site()
    monkeypatch.setattr(favicon, "_transport", httpx.MockTransport(site.handler))
    # Keep name-based lookups hermetic; the public-address assertion still applies.
    monkeypatch.setattr(favicon, "_resolve_host", lambda host, port, scheme: ("93.184.216.34",))
    yield site


def test_link_tag_icon_is_parsed_and_returned_as_png(_mock_transport: _Site):
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="icon" href="/icon.png">
        <link rel="apple-touch-icon" href="/apple.png">
        </head></html>""",
    )
    _mock_transport.route("example.com", "/icon.png", content = _png_bytes())

    result = asyncio.run(favicon.fetch_provider_favicon("https://example.com/v1"))

    assert result is not None
    assert result.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[1]))) as image:
        assert image.format == "PNG"
    assert "https://example.com/icon.png" in _mock_transport.requests


def test_shortcut_icon_link_is_found(_mock_transport: _Site):
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="shortcut icon" href="https://cdn.example.com/fav.png">
        </head></html>""",
    )
    _mock_transport.route("cdn.example.com", "/fav.png", content = _png_bytes())

    result = asyncio.run(favicon.fetch_provider_favicon("https://example.com/v1"))

    assert result is not None and result.startswith("data:image/png;base64,")


def test_favicon_ico_fallback(_mock_transport: _Site):
    _mock_transport.route("example.com", "/", content = b"<html><head></head></html>")
    _mock_transport.route(
        "example.com",
        "/favicon.ico",
        content = _png_bytes(),
        headers = {"content-type": "image/x-icon"},
    )

    result = asyncio.run(favicon.fetch_provider_favicon("https://example.com/v1"))

    assert result is not None and result.startswith("data:image/png;base64,")
    assert "https://example.com/favicon.ico" in _mock_transport.requests


def test_api_subdomain_falls_back_to_parent_domain(_mock_transport: _Site):
    # api.example.com has no homepage or icon; the parent domain does.
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="icon" href="/icon.png">
        </head></html>""",
    )
    _mock_transport.route("example.com", "/icon.png", content = _png_bytes())

    result = asyncio.run(favicon.fetch_provider_favicon("https://api.example.com/v1"))

    assert result is not None and result.startswith("data:image/png;base64,")
    assert "https://example.com/icon.png" in _mock_transport.requests
    assert "https://api.example.com/" in _mock_transport.requests


def test_svg_icon_is_rejected(_mock_transport: _Site):
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="icon" href="/favicon.svg">
        </head></html>""",
    )
    _mock_transport.route(
        "example.com",
        "/favicon.svg",
        content = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        headers = {"content-type": "image/svg+xml"},
    )

    assert asyncio.run(favicon.fetch_provider_favicon("https://example.com")) is None


def test_non_image_content_type_is_rejected(_mock_transport: _Site):
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="icon" href="/icon.html">
        </head></html>""",
    )
    _mock_transport.route("example.com", "/icon.html", content = b"<html>not an image</html>")

    assert asyncio.run(favicon.fetch_provider_favicon("https://example.com")) is None


def test_response_size_cap_rejects_huge_body(_mock_transport: _Site):
    _mock_transport.route("example.com", "/", content = b"<html><head></head></html>")
    _mock_transport.route(
        "example.com",
        "/favicon.ico",
        content = b"x" * (512 * 1024 + 42),
        headers = {"content-type": "image/png"},
    )

    assert asyncio.run(favicon.fetch_provider_favicon("https://example.com")) is None


def test_streamed_image_without_content_length_is_capped(monkeypatch):
    class OversizedStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"x" * favicon._MAX_FAVICON_BYTES
            yield b"x"
            pytest.fail("Image stream was consumed past the size cap.")

        async def aclose(self):
            self.closed = True

    stream = OversizedStream()

    def handler(request):
        response = httpx.Response(200, stream = stream, headers = {"content-type": "image/png"})
        assert "content-length" not in response.headers
        return response

    monkeypatch.setattr(favicon, "_transport", httpx.MockTransport(handler))
    with pytest.raises(favicon._NoSiteIcon, match = "size cap"):
        asyncio.run(favicon._image_bytes("https://example.com/icon.png"))
    assert stream.closed


def test_streamed_homepage_is_truncated(monkeypatch):
    class OversizedStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            link = b'<link rel="icon" href="/icon.png">'
            yield link + b" " * (favicon._MAX_FAVICON_BYTES - len(link))
            yield b"x"
            pytest.fail("Homepage stream was consumed past the size cap.")

        async def aclose(self):
            self.closed = True

    stream = OversizedStream()
    monkeypatch.setattr(
        favicon,
        "_transport",
        httpx.MockTransport(lambda request: httpx.Response(200, stream = stream)),
    )
    assert asyncio.run(favicon._homepage_links("https://example.com")) == [
        "https://example.com/icon.png"
    ]
    assert stream.closed


def test_loopback_ip_is_blocked():
    with pytest.raises(ProviderFaviconError, match = "public"):
        asyncio.run(favicon.fetch_provider_favicon("http://127.0.0.1/"))


def test_metadata_ip_is_blocked():
    with pytest.raises(ProviderFaviconError, match = "metadata"):
        asyncio.run(favicon.fetch_provider_favicon("http://169.254.169.254/latest"))


def test_dns_resolving_to_private_address_is_blocked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(favicon, "_resolve_host", lambda host, port, scheme: ("10.0.0.5",))

    with pytest.raises(ProviderFaviconError, match = "public"):
        asyncio.run(favicon.fetch_provider_favicon("https://intranet.example/v1"))


def test_private_scheme_is_rejected():
    with pytest.raises(ProviderFaviconError, match = "http and https"):
        asyncio.run(favicon.fetch_provider_favicon("ftp://example.com"))


def test_icon_link_to_internal_host_is_blocked(_mock_transport: _Site):
    _mock_transport.route(
        "example.com",
        "/",
        content = b"""<html><head>
        <link rel="icon" href="http://169.254.169.254/icon.png">
        </head></html>""",
    )

    with pytest.raises(ProviderFaviconError, match = "metadata"):
        asyncio.run(favicon.fetch_provider_favicon("https://example.com"))
    assert all("169.254.169.254" not in url for url in _mock_transport.requests)


def test_redirect_to_loopback_is_revetted_and_blocked(_mock_transport: _Site):
    _mock_transport.route("example.com", "/", status = 302, headers = {"location": "http://127.0.0.1/"})

    with pytest.raises(ProviderFaviconError, match = "not a public site"):
        asyncio.run(favicon.fetch_provider_favicon("https://example.com"))
    assert all("127.0.0.1" not in url for url in _mock_transport.requests)


def test_request_dials_vetted_ip_with_host_and_sni(_mock_transport: _Site):
    _mock_transport.route("example.com", "/", content = b"<html></html>")

    asyncio.run(favicon.fetch_provider_favicon("https://example.com/v1"))

    request = _mock_transport.raw[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"


def test_sni_hostname_reaches_tls_handshake(monkeypatch):
    # Real transport: the pinned IP is dialed, TLS is told the original name.
    import httpcore

    seen = {}

    class FakeStream:
        async def start_tls(
            self,
            ssl_context,
            server_hostname = None,
            timeout = None,
        ):
            seen["sni"] = server_hostname
            seen["check_hostname"] = ssl_context.check_hostname
            raise httpcore.ConnectError("stop before network")

        async def aclose(self):
            pass

    async def fake_connect_tcp(
        self,
        host,
        port,
        timeout = None,
        local_address = None,
        socket_options = None,
    ):
        seen["dial"] = (host, port)
        return FakeStream()

    monkeypatch.setattr(httpcore.AnyIOBackend, "connect_tcp", fake_connect_tcp)
    monkeypatch.setattr(favicon, "_transport", None)
    monkeypatch.setattr(favicon, "_resolve_host", lambda host, port, scheme: ("93.184.216.34",))
    assert asyncio.run(favicon.fetch_provider_favicon("https://example.com")) is None
    assert seen["dial"] == ("93.184.216.34", 443)
    assert seen["sni"] == "example.com" and seen["check_hostname"] is True


def test_client_ignores_proxy_environment(monkeypatch, _mock_transport: _Site):
    kwargs = []
    real = httpx.AsyncClient

    def spy(*args, **kw):
        kwargs.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(favicon.httpx, "AsyncClient", spy)
    asyncio.run(favicon.fetch_provider_favicon("https://example.com"))
    assert kwargs and all(kw["trust_env"] is False for kw in kwargs)


def test_ipv6_literal_base_url_is_accepted(_mock_transport: _Site):
    _mock_transport.route("2606:4700:4700::1111", "/", content = b"<html></html>")
    _mock_transport.route(
        "2606:4700:4700::1111",
        "/favicon.ico",
        content = _png_bytes(),
        headers = {"content-type": "image/png"},
    )

    result = asyncio.run(favicon.fetch_provider_favicon("https://[2606:4700:4700::1111]/v1"))

    assert result is not None
    assert _mock_transport.raw[0].url.host == "2606:4700:4700::1111"


@pytest.mark.parametrize(
    "url", ["http://[::127.0.0.1]/", "http://[64:ff9b::7f00:1]/", "http://[::ffff:10.0.0.1]/"]
)
def test_embedded_ipv4_ranges_are_refused(url):
    with pytest.raises(ProviderFaviconError, match = "public"):
        asyncio.run(favicon.fetch_provider_favicon(url))


def test_nat64_of_public_ipv4_is_allowed():
    assert favicon._is_public_ip(favicon.ipaddress.ip_address("64:ff9b::5db8:d822"))


def test_large_dimension_tiny_png_is_rejected_before_decode():
    image = Image.new("1", (9000, 9000))
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG", optimize = True)
    assert len(buffer.getvalue()) < 64 * 1024

    def boom(self):
        pytest.fail("load() ran on an oversized image")

    from PIL import ImageFile

    original = ImageFile.ImageFile.load
    ImageFile.ImageFile.load = boom
    try:
        with pytest.raises(favicon._NoSiteIcon) as exc:
            favicon._png_data_url(buffer.getvalue())
        assert "too large" in str(exc.value.__cause__)
    finally:
        ImageFile.ImageFile.load = original


def test_ico_with_oversized_embedded_png_frame_is_rejected():
    import struct

    png = io.BytesIO()
    Image.new("1", (4000, 4000)).save(png, format = "PNG")
    frame = png.getvalue()
    # ICO directory claims 16x16; the embedded PNG is really 4000x4000.
    ico = (
        struct.pack("<HHH", 0, 1, 1)
        + struct.pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, len(frame), 22)
        + frame
    )
    import warnings

    with warnings.catch_warnings():
        # Pillow warns only after decoding a frame that disagrees with its entry.
        warnings.simplefilter("error")
        with pytest.raises(favicon._NoSiteIcon) as exc:
            favicon._png_data_url(ico)
    assert "too large" in str(exc.value.__cause__)


def test_link_candidates_are_capped(_mock_transport: _Site):
    links = "".join(f'<link rel="icon" href="/i{n}.png">' for n in range(50))
    _mock_transport.route("example.com", "/", content = links.encode())

    asyncio.run(favicon.fetch_provider_favicon("https://example.com"))

    icon_requests = [u for u in _mock_transport.requests if "/i" in u]
    assert len(icon_requests) == favicon._MAX_LINK_CANDIDATES


def test_overall_deadline_bounds_a_dripping_server(monkeypatch):
    class DripStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.01)
                yield b" "

    monkeypatch.setattr(favicon, "_OVERALL_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(
        favicon,
        "_transport",
        httpx.MockTransport(lambda request: httpx.Response(200, stream = DripStream())),
    )
    with pytest.raises(ProviderFaviconError, match = "too long") as exc:
        asyncio.run(favicon.fetch_provider_favicon("https://example.com"))
    assert exc.value.status_code == 502
