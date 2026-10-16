# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_wheel gives uv a direct-URL wheel's metadata, so uv fetches the wheel once.

Without it uv reads METADATA with range requests, and where those fail (a GitHub release asset
behind its cross-origin redirect, astral-sh/uv#18998) it streams the whole wheel for METADATA
and then fetches it again to install: two copies of a 427 MB flash-attn wheel.
"""

import http.server
import io
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import wheel_utils  # noqa: E402

FLASH_ATTN = (
    "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
    "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
)
FLASH_ATTN_TOML = (
    "[[dependency-metadata]]\n"
    'name = "flash-attn"\n'
    'version = "2.8.4+cu13torch2.13cxx11abitrue"\n'
    "requires-dist = []\n"
)


@pytest.fixture
def no_project_config(tmp_path, monkeypatch):
    """Run from a directory with no uv.toml or [tool.uv] above it, and no uv config overrides."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wheel_utils, "_uv_project_config_found", lambda: False)
    monkeypatch.delenv("UV_CONFIG_FILE", raising = False)
    monkeypatch.delenv("UV_NO_CONFIG", raising = False)
    return tmp_path


def test_static_metadata_names_the_wheel(no_project_config, tmp_path):
    out = tmp_path / "cfg"
    out.mkdir()
    assert wheel_utils._uv_static_metadata_config(FLASH_ATTN, str(out)) is True
    assert (out / "uv.toml").read_text(encoding = "utf-8") == FLASH_ATTN_TOML


def test_static_metadata_reads_an_escaped_filename_behind_a_query(no_project_config, tmp_path):
    url = (
        "https://mirror.example/whl/causal_conv1d-1.7.0%2Bcu13torch2.13cxx11abiTRUE"
        "-cp313-cp313-linux_x86_64.whl?token=abc"
    )
    assert wheel_utils._uv_static_metadata_config(url, str(tmp_path)) is True
    text = (tmp_path / "uv.toml").read_text(encoding = "utf-8")
    assert 'name = "causal-conv1d"' in text
    assert 'version = "1.7.0+cu13torch2.13cxx11abitrue"' in text


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/whl/simple/",
        "https://example.test/notawheel-1.0.tar.gz",
        'https://example.test/evil"name-1.0-py3-none-any.whl',
        "https://example.test/too-few-fields.whl",
    ],
)
def test_no_static_metadata_for_what_is_not_a_plain_wheel(url, no_project_config, tmp_path):
    assert wheel_utils._uv_static_metadata_config(url, str(tmp_path)) is False
    assert not (tmp_path / "uv.toml").exists()


@pytest.mark.parametrize("variable", ["UV_CONFIG_FILE", "UV_NO_CONFIG"])
def test_no_static_metadata_where_uv_would_not_read_it(
    variable, no_project_config, tmp_path, monkeypatch
):
    monkeypatch.setenv(variable, "1")
    assert wheel_utils._uv_static_metadata_config(FLASH_ATTN, str(tmp_path)) is False


@pytest.mark.parametrize(
    "name, text, found",
    [
        ("uv.toml", "", True),
        ("pyproject.toml", "[project]\nname = 'x'\n\n[tool.uv]\nnative-tls = true\n", True),
        ("pyproject.toml", "[tool.uv.pip]\nnative-tls = true\n", True),
        ("pyproject.toml", "[project]\nname = 'x'\n", False),
    ],
)
def test_project_config_is_found_from_a_parent(name, text, found, tmp_path, monkeypatch):
    """A config uv would discover from the working directory must not be shadowed."""
    (tmp_path / name).write_text(text, encoding = "utf-8")
    child = tmp_path / "a" / "b"
    child.mkdir(parents = True)
    monkeypatch.chdir(child)
    assert wheel_utils._uv_project_config_found() is found


class _Recorder:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls: list[tuple[list[str], object, str | None]] = []

    def __call__(self, cmd, **kwargs):
        cwd = kwargs.get("cwd")
        toml = Path(cwd, "uv.toml").read_text(encoding = "utf-8") if cwd else None
        self.calls.append((list(cmd), cwd, toml))
        return subprocess.CompletedProcess(cmd, self.codes.pop(0), stdout = "")


def test_uv_runs_beside_the_metadata_and_pip_runs_as_before(no_project_config, monkeypatch):
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    run = _Recorder([1, 0])
    attempts = wheel_utils.install_wheel(FLASH_ATTN, python_executable = "PY", use_uv = True, run = run)
    assert [installer for installer, _ in attempts] == ["uv", "pip"]
    (uv_cmd, uv_cwd, uv_toml), (pip_cmd, pip_cwd, _) = run.calls
    # Still the URL: uv downloads and caches it itself, so a reinstall stays a cache hit.
    assert uv_cmd == ["uv", "pip", "install", "--python", "PY", "--no-deps", FLASH_ATTN]
    assert uv_toml == FLASH_ATTN_TOML
    assert not os.path.exists(uv_cwd), "the config directory outlived the install"
    assert pip_cmd[-1] == FLASH_ATTN and pip_cwd is None


def test_uv_keeps_the_working_directory_when_a_project_config_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(wheel_utils, "_uv_project_config_found", lambda: True)
    run = _Recorder([0])
    wheel_utils.install_wheel(FLASH_ATTN, python_executable = "PY", use_uv = True, run = run)
    assert run.calls[0][1] is None


# -- Against real uv ----------------------------------------------------------------------------


def _wheel_bytes(name: str, version: str) -> bytes:
    """A minimal valid pure-Python wheel."""
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": "VALUE = 1\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "".join(f"{path},,\n" for path in files) + f"{dist}/RECORD,,\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        archive.writestr(f"{dist}/RECORD", record)
    return buffer.getvalue()


@pytest.fixture
def no_range_server():
    """Serves one wheel like the redirected GitHub asset: no Range answer, but ETag revalidation.

    ``gets`` counts the GETs that sent the body.
    """
    body = _wheel_bytes("rangeless_kernel", "1.0.0")
    etag = '"0x1"'
    gets: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _head(self, status = 200):
            self.send_response(status)
            self.send_header("ETag", etag)
            if status == 200:
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()

        def do_HEAD(self):  # noqa: N802
            self._head()

        def do_GET(self):  # noqa: N802
            if self.headers.get("If-None-Match") == etag:
                self._head(304)
                return
            gets.append(self.headers.get("Range") or "")
            self._head()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = httpd.serve_forever, daemon = True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/rangeless_kernel-1.0.0-py3-none-any.whl"
    try:
        yield url, gets
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.skipif(shutil.which("uv") is None, reason = "needs uv on PATH")
@pytest.mark.parametrize("static_metadata, expected_gets", [(False, 2), (True, 1)])
def test_real_uv_fetches_the_wheel_once(
    static_metadata, expected_gets, no_range_server, no_project_config, tmp_path, monkeypatch
):
    url, gets = no_range_server
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", "--python", sys.executable, str(venv)], check = True)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    if not static_metadata:
        monkeypatch.setattr(wheel_utils, "_uv_static_metadata_config", lambda *args: False)
    attempts = wheel_utils.install_wheel(url, python_executable = str(python), use_uv = True)
    assert [(i, r.returncode) for i, r in attempts] == [("uv", 0)], attempts[-1][1].stdout
    assert len(gets) == expected_gets, gets
    # A reinstall is answered from uv's own cache.
    wheel_utils.install_wheel(url, python_executable = str(python), use_uv = True, reinstall = True)
    assert len(gets) == expected_gets, gets
