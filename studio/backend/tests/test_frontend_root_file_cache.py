# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build-root files keep their name across builds: index.html versions its boot scripts, everything else
gets an explicit policy plus working conditional requests."""

import os
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

IMMUTABLE = "public, max-age=31536000, immutable"
ONE_DAY = "public, max-age=86400"

_INDEX = """<!doctype html><html><head>
<link rel="icon" type="image/png" href="/favicon.png" />
<link rel="manifest" href="/manifest.webmanifest" />
<title>shell</title>
<script src="/theme-boot.js"></script>
<script src="/missing-boot.js"></script>
<script src="/theme-boot.js?keep=1"></script>
<script src="//cdn.example/theme-boot.js"></script>
<script src="https://cdn.example/x.js"></script>
<script src="/../outside.js"></script>
<script type="module" crossorigin src="/assets/index-abc123.js"></script>
</head><body></body></html>"""


@pytest.fixture(scope = "module")
def main_module():
    import main as _main
    return _main


def _dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents = True)
    (dist / "fonts").mkdir()
    (dist / "index.html").write_text(_INDEX, encoding = "utf-8")
    (dist / "theme-boot.js").write_text("window.__theme = 'v1';", encoding = "utf-8")
    (dist / "manifest.webmanifest").write_text('{"name": "Unsloth"}', encoding = "utf-8")
    (dist / "favicon.png").write_bytes(b"\x89PNG" + b"\x00" * 64)
    (dist / "fonts" / "Hellix-Regular.woff2").write_bytes(b"wOF2" + b"\x00" * 2048)
    (dist / "assets" / "index-abc123.js").write_text("export {};" * 200, encoding = "utf-8")
    (tmp_path / "outside.js").write_text("// not part of the build", encoding = "utf-8")
    return dist


def _client(main_module, dist, **kwargs):
    app = FastAPI()
    assert main_module.setup_frontend(app, dist, **kwargs)
    app.add_middleware(main_module.SecurityHeadersMiddleware)
    return TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 40000))


def _boot_url(client):
    urls = re.findall(r'src="(/theme-boot\.js\?v=[0-9a-f]+)"', client.get("/").text)
    assert len(urls) == 1
    return urls[0]


def test_index_versions_only_existing_build_root_scripts(tmp_path, main_module):
    shell = _client(main_module, _dist(tmp_path)).get("/")
    assert shell.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert shell.headers["vary"] == "Origin"
    html = shell.text
    assert re.search(r'<script src="/theme-boot\.js\?v=[0-9a-f]{12}"></script>', html)
    for untouched in (
        '<script src="/missing-boot.js">',
        '<script src="/theme-boot.js?keep=1">',
        '<script src="//cdn.example/theme-boot.js">',
        '<script src="https://cdn.example/x.js">',
        '<script src="/../outside.js">',
        '<script type="module" src="/assets/index-abc123.js">',  # crossorigin still stripped
        'href="/favicon.png"',
        'href="/manifest.webmanifest"',
    ):
        assert untouched in html, untouched


def test_versioned_boot_script_is_immutable_and_unversioned_revalidates(tmp_path, main_module):
    client = _client(main_module, _dist(tmp_path))
    versioned = client.get(_boot_url(client))
    assert versioned.status_code == 200
    assert versioned.text == "window.__theme = 'v1';"
    assert versioned.headers["cache-control"] == IMMUTABLE
    assert versioned.headers["content-type"].startswith("text/javascript")

    for url in ("/theme-boot.js", "/theme-boot.js?v=000000000000"):
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        etag, last_modified = response.headers["etag"], response.headers["last-modified"]
        for conditional in ({"If-None-Match": etag}, {"If-Modified-Since": last_modified}):
            again = client.get(url, headers = conditional)
            assert again.status_code == 304
            assert again.content == b""
            assert again.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "path, media_type",
    [
        ("/fonts/Hellix-Regular.woff2", "font/woff2"),
        ("/favicon.png", "image/png"),
        ("/manifest.webmanifest", "application/manifest+json"),
    ],
)
def test_other_root_files_cache_for_a_day_and_answer_304(tmp_path, main_module, path, media_type):
    client = _client(main_module, _dist(tmp_path))
    first = client.get(path)
    assert first.status_code == 200
    assert first.headers["cache-control"] == ONE_DAY
    assert first.headers["content-type"].startswith(media_type)
    for conditional in (
        {"If-None-Match": first.headers["etag"]},
        {"If-Modified-Since": first.headers["last-modified"]},
    ):
        again = client.get(path, headers = conditional)
        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["cache-control"] == ONE_DAY


def test_an_updated_boot_script_gets_a_new_url_on_the_next_page_load(tmp_path, main_module):
    """`unsloth studio update` replaces the dist in place; index.html is never cached, so its new v wins."""
    dist = _dist(tmp_path)
    client = _client(main_module, dist)
    old_url = _boot_url(client)
    old = client.get("/theme-boot.js")
    boot = dist / "theme-boot.js"
    boot.write_text("window.__theme = 'v2, a newer build';", encoding = "utf-8")
    later = os.stat(boot).st_mtime + 60
    os.utime(boot, (later, later))

    new_url = _boot_url(client)
    assert new_url != old_url
    fresh = client.get(new_url)
    assert fresh.text == "window.__theme = 'v2, a newer build';"
    assert fresh.headers["cache-control"] == IMMUTABLE
    # The previous build's URL is no longer vouched for, and old validators no longer match.
    assert client.get(old_url).headers["cache-control"] == "no-cache"
    revalidated = client.get("/theme-boot.js", headers = {"If-None-Match": old.headers["etag"]})
    assert revalidated.status_code == 200 and "v2" in revalidated.text


def test_bootstrap_injection_and_its_nonce_survive_the_rewrite(tmp_path, main_module, monkeypatch):
    from auth import policy

    dist = _dist(tmp_path)
    app = FastAPI()
    assert main_module.setup_frontend(app, dist)
    app.add_middleware(main_module.SecurityHeadersMiddleware)
    app.state.bootstrap_password = "first-run-pw"
    monkeypatch.setattr(main_module, "_should_inject_bootstrap", lambda request: True)
    monkeypatch.setattr(main_module.storage, "requires_password_change", lambda username: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: False)
    monkeypatch.setattr(main_module, "_desktop_owner", lambda: None)
    client = TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 40000))

    shell = client.get("/")
    nonce = re.search(r'<script nonce="([^"]+)">window\.__UNSLOTH_BOOTSTRAP__=', shell.text).group(
        1
    )
    assert "first-run-pw" in shell.text
    assert f"'nonce-{nonce}'" in shell.headers["content-security-policy"]
    assert main_module._CSP_SCRIPT_NONCE_HEADER.lower() not in shell.headers
    assert re.search(r'src="/theme-boot\.js\?v=[0-9a-f]{12}"', shell.text)


def test_hashed_assets_keep_their_policy(tmp_path, main_module):
    client = _client(main_module, _dist(tmp_path))
    asset = client.get("/assets/index-abc123.js")
    assert asset.headers["cache-control"] == IMMUTABLE
    assert client.get("/assets/index-abc123.js?v=x").headers["cache-control"] == IMMUTABLE
    revalidated = client.get(
        "/assets/index-abc123.js", headers = {"If-None-Match": asset.headers["etag"]}
    )
    assert revalidated.status_code == 304
    spa = client.get("/chat/some-thread", headers = {"If-None-Match": "*"})
    assert spa.status_code == 200 and "<title>shell</title>" in spa.text
    assert spa.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert "etag" not in spa.headers


def test_paths_outside_the_build_stay_forbidden(tmp_path, main_module):
    dist = _dist(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the build", encoding = "utf-8")
    try:
        (dist / "leak.txt").symlink_to(secret)
    except OSError:
        pytest.skip("symlinks unavailable")
    client = _client(main_module, dist)
    response = client.get("/leak.txt")
    assert response.status_code == 403
    assert "outside the build" not in response.text


def test_tunnel_only_frontend_still_hides_root_files(tmp_path, main_module):
    app = FastAPI()
    app.state.cloudflare_url = None
    assert main_module.setup_frontend(app, _dist(tmp_path), tunnel_only = True)
    assert TestClient(app).get("/theme-boot.js").status_code == 404
    loopback = TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 40000))
    assert loopback.get("/theme-boot.js").headers["cache-control"] == "no-cache"
    assert loopback.get("/favicon.png").headers["cache-control"] == ONE_DAY
