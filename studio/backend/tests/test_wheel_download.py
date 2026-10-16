# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_wheel fetches a prebuilt wheel once and installs the local file.

uv streams a direct-URL wheel twice when the server's HEAD answer hides range support, which
GitHub release assets do: once for METADATA, once to install (astral-sh/uv#18998). Served
here from loopback, so the request count is the evidence.
"""

import hashlib
import http.server
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import wheel_utils  # noqa: E402

WHEEL = "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
BODY = os.urandom(3 * (1 << 20) + 123)


@pytest.fixture
def server():
    requests: list[tuple[str, str]] = []
    behaviour = {"truncate": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            requests.append(("GET", self.path))
            if not self.path.endswith(".whl"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY[: len(BODY) // 2] if behaviour["truncate"] else BODY)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, requests, behaviour
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_download_is_one_request_with_progress(server, tmp_path):
    base, requests, _ = server
    seen = []
    path = wheel_utils.download_wheel(
        f"{base}/dl/{WHEEL.replace('+', '%2B')}",
        str(tmp_path),
        progress = lambda done, total: seen.append((done, total)),
    )
    assert path == str(tmp_path / WHEEL)
    assert Path(path).read_bytes() == BODY
    assert requests == [("GET", f"/dl/{WHEEL.replace('+', '%2B')}")]
    assert seen[-1] == (len(BODY), len(BODY))
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_sha256_fragment_is_enforced(server, tmp_path):
    base, requests, _ = server
    good = hashlib.sha256(BODY).hexdigest()
    url = f"{base}/{WHEEL}"
    assert wheel_utils.download_wheel(f"{url}#sha256={good}", str(tmp_path)) is not None
    # The fragment is ours to check, never sent.
    assert requests == [("GET", f"/{WHEEL}")]
    assert wheel_utils.download_wheel(f"{url}#sha256={'0' * 64}", str(tmp_path)) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://user:token@mirror.invalid/whl/" + WHEEL,
        "file:///tmp/" + WHEEL,
        "https://mirror.invalid/whl/simple/",
    ],
)
def test_urls_it_cannot_fetch_are_left_to_the_installer(url, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("must not open a connection")

    monkeypatch.setattr(wheel_utils.urllib.request, "urlopen", refuse)
    assert wheel_utils.download_wheel(url, str(tmp_path)) is None


def test_http_error_and_short_body_return_none(server, tmp_path):
    base, _, behaviour = server
    assert wheel_utils.download_wheel(f"{base}/missing", str(tmp_path)) is None
    assert wheel_utils.download_wheel(f"{base}/missing.whl/", str(tmp_path)) is None
    behaviour["truncate"] = True
    assert wheel_utils.download_wheel(f"{base}/{WHEEL}", str(tmp_path)) is None


class _Recorder:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls: list[list[str]] = []
        self.present: list[bool] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.present.append(os.path.isfile(cmd[-1]))
        return subprocess.CompletedProcess(cmd, self.codes.pop(0), stdout = "")


def test_uv_and_pip_both_install_the_one_downloaded_file(server, monkeypatch):
    base, requests, _ = server
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    run = _Recorder([1, 0])
    attempts = wheel_utils.install_wheel(
        f"{base}/{WHEEL}", python_executable = "PY", use_uv = True, run = run
    )
    assert [installer for installer, _ in attempts] == ["uv", "pip"]
    assert len(requests) == 1, "the pip fallback must not fetch the wheel again"
    uv_target, pip_target = run.calls[0][-1], run.calls[1][-1]
    assert uv_target == pip_target and Path(uv_target).name == WHEEL
    assert run.present == [True, True]
    # The temporary copy does not outlive the install.
    assert not Path(uv_target).parent.exists()


def test_unreachable_wheel_falls_back_to_the_url(server, monkeypatch):
    base, _, _ = server
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    run = _Recorder([0])
    url = f"{base}/gone/{WHEEL}"
    monkeypatch.setattr(wheel_utils, "download_wheel", lambda *args, **kwargs: None)
    wheel_utils.install_wheel(url, python_executable = "PY", use_uv = True, run = run)
    assert run.calls == [["uv", "pip", "install", "--python", "PY", "--no-deps", url]]
