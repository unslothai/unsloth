# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_wheel gives uv a URL wheel's metadata, so uv fetches it once (astral-sh/uv#18998)."""

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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wheel_utils, "_uv_project_config_found", lambda: False)
    monkeypatch.delenv("UV_CONFIG_FILE", raising = False)
    monkeypatch.delenv("UV_NO_CONFIG", raising = False)
    for name in wheel_utils._UV_PROJECT_DIR_ENV:
        monkeypatch.delenv(name, raising = False)
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


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "f", "n", ""])
def test_a_false_uv_no_config_keeps_static_metadata(
    value, no_project_config, tmp_path, monkeypatch
):
    # uv reads these as false and still discovers config, so the metadata file is honoured.
    monkeypatch.setenv("UV_NO_CONFIG", value)
    assert wheel_utils._uv_static_metadata_config(FLASH_ATTN, str(tmp_path)) is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_a_true_uv_no_config_skips_static_metadata(value, no_project_config, tmp_path, monkeypatch):
    monkeypatch.setenv("UV_NO_CONFIG", value)
    assert wheel_utils._uv_static_metadata_config(FLASH_ATTN, str(tmp_path)) is False


@pytest.mark.parametrize(
    "variable",
    ["UV_CONFIG_FILE", "UV_NO_CONFIG", "UV_PROJECT", "UV_WORKING_DIR", "UV_WORKING_DIRECTORY"],
)
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
        ("pyproject.toml", "[tool]\nuv = { native-tls = true }\n", True),
        ("pyproject.toml", '[tool."uv"]\nnative-tls = true\n', True),
        ("pyproject.toml", "[ tool.uv ]\nnative-tls = true\n", True),
        ("pyproject.toml", "tool.uv.native-tls = true\n", True),
        ("pyproject.toml", "[project]\ndescription = '[tool.uv] docs'\n", False),
        ("pyproject.toml", "[tool.uv\n", True),
    ],
)
def test_project_config_is_found_from_a_parent(name, text, found, tmp_path, monkeypatch):
    (tmp_path / name).write_text(text, encoding = "utf-8")
    child = tmp_path / "a" / "b"
    child.mkdir(parents = True)
    monkeypatch.chdir(child)
    assert wheel_utils._uv_project_config_found() is found


class _Recorder:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls: list[tuple[list[str], str | None, str | None]] = []

    def __call__(self, cmd, **kwargs):
        assert "cwd" not in kwargs, "uv must run from the caller's directory"
        project = cmd[cmd.index("--project") + 1] if "--project" in cmd else None
        toml = Path(project, "uv.toml").read_text(encoding = "utf-8") if project else None
        self.calls.append((list(cmd), project, toml))
        return subprocess.CompletedProcess(cmd, self.codes.pop(0), stdout = "")


def test_uv_reads_the_metadata_and_pip_runs_as_before(no_project_config, monkeypatch):
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    run = _Recorder([1, 0])
    attempts = wheel_utils.install_wheel(FLASH_ATTN, python_executable = "PY", use_uv = True, run = run)
    assert [installer for installer, _ in attempts] == ["uv", "pip"]
    (uv_cmd, project, uv_toml), (pip_cmd, _, _) = run.calls
    assert uv_cmd == [
        "uv",
        "--project",
        project,
        "pip",
        "install",
        "--python",
        "PY",
        "--no-deps",
        FLASH_ATTN,
    ]
    assert uv_toml == FLASH_ATTN_TOML
    assert not os.path.exists(project), "the config directory outlived the install"
    assert pip_cmd == ["PY", "-m", "pip", "install", "--no-deps", FLASH_ATTN]


@pytest.mark.parametrize("broken", ["mkdtemp", "write"])
def test_an_unwritable_temp_dir_still_installs_with_plain_uv(
    broken, no_project_config, monkeypatch
):
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")

    def fail(*args, **kwargs):
        raise PermissionError("denied")

    if broken == "mkdtemp":
        monkeypatch.setattr(wheel_utils.tempfile, "mkdtemp", fail)
    else:
        monkeypatch.setattr("builtins.open", fail)
    run = _Recorder([0])
    attempts = wheel_utils.install_wheel(FLASH_ATTN, python_executable = "PY", use_uv = True, run = run)
    assert [(i, r.returncode) for i, r in attempts] == [("uv", 0)]
    assert run.calls[0][0] == ["uv", "pip", "install", "--python", "PY", "--no-deps", FLASH_ATTN]


def test_uv_runs_as_before_when_a_project_config_applies(monkeypatch):
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(wheel_utils, "_uv_project_config_found", lambda: True)
    run = _Recorder([0])
    wheel_utils.install_wheel(FLASH_ATTN, python_executable = "PY", use_uv = True, run = run)
    assert run.calls[0][0] == ["uv", "pip", "install", "--python", "PY", "--no-deps", FLASH_ATTN]


def _wheel_bytes(name: str, version: str) -> bytes:
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
    """Serves a wheel like a GitHub asset: no Range, ETag honoured; ``gets`` counts bodies sent."""
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
    # Relative on purpose: it must resolve against the cwd, not the deleted config dir.
    monkeypatch.setenv("UV_CACHE_DIR", "relative-uv-cache")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    if not static_metadata:
        monkeypatch.setattr(wheel_utils, "_uv_static_metadata_config", lambda *args: False)
    attempts = wheel_utils.install_wheel(url, python_executable = str(python), use_uv = True)
    assert [(i, r.returncode) for i, r in attempts] == [("uv", 0)], attempts[-1][1].stdout
    assert len(gets) == expected_gets, gets
    assert (no_project_config / "relative-uv-cache").is_dir()
    wheel_utils.install_wheel(url, python_executable = str(python), use_uv = True, reinstall = True)
    assert len(gets) == expected_gets, gets


@pytest.mark.skipif(shutil.which("uv") is None, reason = "needs uv on PATH")
@pytest.mark.parametrize("variable", ["UV_PROJECT", "UV_WORKING_DIR", "UV_WORKING_DIRECTORY"])
def test_real_uv_still_honours_the_users_project_dir(
    variable, no_range_server, no_project_config, tmp_path, monkeypatch
):
    url, _ = no_range_server
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", "--python", sys.executable, str(venv)], check = True)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    chosen = tmp_path / "chosen-project"
    chosen.mkdir()
    # Unsatisfiable, so honouring this config shows as a refusal.
    (chosen / "uv.toml").write_text('required-version = "==0.0.1"\n', encoding = "utf-8")
    monkeypatch.setenv(variable, str(chosen))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    installer, result = wheel_utils.install_wheel(url, python_executable = str(python), use_uv = True)[
        0
    ]
    assert installer == "uv" and "--project" not in result.args, result.args
    # Older uv ignores UV_WORKING_DIR*; plain uv is the reference for what the user's config does.
    plain = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", url],
        capture_output = True,
        text = True,
    )
    assert (result.returncode == 0) == (plain.returncode == 0), (result.stdout, plain.stderr)
    if variable == "UV_PROJECT":
        assert result.returncode != 0 and "0.0.1" in result.stdout, result.stdout
