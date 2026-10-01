# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHELLS = [shell for name in ("powershell", "pwsh") if (shell := shutil.which(name))]
PROGRESS = "Downloading runtime.zip: 25.0% (1.0 MiB/4.0 MiB) at 1.0 MiB/s"
UNKNOWN_PROGRESS = "Downloading runtime.zip: 25.0 MiB downloaded at 1.0 MiB/s"
DIAGNOSTICS = ["resolving release", "Downloading runtime.zip: retrying", "validating runtime"]


def setup_fragment(component):
    source = (ROOT / "studio/setup.ps1").read_text(encoding = "utf-8")
    encoding = source[
        source.index("$_UnslothUtf8NoBom =") : source.index("function Write-StudioLine")
    ]
    helpers = "\n".join(
        re.search(rf"^function {name} \{{.*?^\}}", source, re.M | re.S).group()
        for name in ("Write-StudioLine", "Show-DownloadProgress")
    )
    if component == "node":
        block = re.search(r"\$nodeOut = & .*\n\s*\$nodeExit = \$LASTEXITCODE", source).group()
        if os.name != "nt":
            block = block.replace("$PSScriptRoot\\", "$PSScriptRoot/")
        output, status = "$nodeOut", "$nodeExit"
    else:
        name = "Prebuilt" if component == "llama" else "Whisper"
        start = source.index(f"$prevEAP{name} =")
        end = source.index(f"$ErrorActionPreference = $prevEAP{name}", start)
        block = source[start:end] + f"$ErrorActionPreference = $prevEAP{name}"
        output = "$prebuiltOutput" if component == "llama" else "$whisperOutput"
        status = "$prebuiltExit" if component == "llama" else "$whisperExit"
    return encoding + helpers, block, output, status


def start_installer(tmp_path, shell, component, mode, child_source):
    child = tmp_path / "install_node_prebuilt.py"
    child.write_text(child_source, encoding = "utf-8")
    helpers, block, output, status = setup_fragment(component)
    script = tmp_path / "run.ps1"
    script.write_text(
        "param([string]$Python, [string]$WorkDir)\n"
        "$ErrorActionPreference = 'Stop'\n" + helpers + "\n"
        "$script:UnslothVerbose = ($env:UNSLOTH_VERBOSE -eq '1')\n"
        "$NodeInstallPython = $Python\n$NodeDir = $WorkDir\n"
        "$prebuiltArgs = @((Join-Path $WorkDir 'install_node_prebuilt.py'))\n"
        "$whisperArgs = $prebuiltArgs\n" + block + "\n"
        f"[IO.File]::WriteAllText((Join-Path $WorkDir 'captured.log'), {output})\n"
        f"exit {status}\n",
        encoding = "utf-8",
    )
    env = dict(os.environ, TEMP = str(tmp_path), RELEASE_FILE = str(tmp_path / "release"))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["UNSLOTH_TAURI_UPDATE"] = mode if mode in ("1", "true") else "0"
    env["UNSLOTH_VERBOSE"] = "1" if mode == "verbose" else "0"
    process = subprocess.Popen(
        [
            shell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            sys.executable,
            str(tmp_path),
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        env = env,
        creationflags = subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32" and mode != "verbose"
        else 0,
    )
    lines = queue.Queue()

    def read_lines():
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))

    reader = threading.Thread(target = read_lines, daemon = True)
    reader.start()
    return process, lines, reader


@pytest.mark.parametrize("shell", SHELLS or [None])
@pytest.mark.parametrize("component", ["node", "llama", "whisper"])
@pytest.mark.parametrize("mode", ["1", "true", "quiet", "verbose"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_windows_download_output(tmp_path, shell, component, mode, exit_code):
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    child = (
        "import os, sys, time\nfrom pathlib import Path\n"
        f"print({DIAGNOSTICS[0]!r}, flush=True)\n"
        f"print({PROGRESS!r}, flush=True)\n"
        f"print({DIAGNOSTICS[1]!r}, flush=True)\n"
        f"print({UNKNOWN_PROGRESS!r}, file=sys.{'stdout' if component == 'node' else 'stderr'}, flush=True)\n"
        "deadline = time.monotonic() + 30\n"
        "while not Path(os.environ['RELEASE_FILE']).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        f"print({DIAGNOSTICS[2]!r}, flush=True)\n"
        f"raise SystemExit({exit_code})\n"
    )
    process, lines, reader = start_installer(tmp_path, shell, component, mode, child)
    received = []
    try:
        if mode in ("1", "true"):
            while PROGRESS not in received or UNKNOWN_PROGRESS not in received:
                received.append(lines.get(timeout = 15))
            assert process.poll() is None
    finally:
        (tmp_path / "release").touch()
        process.wait(timeout = 20)
        reader.join(timeout = 5)
        process.stdout.close()
    while not lines.empty():
        received.append(lines.get_nowait())
    assert process.returncode == exit_code, "\n".join(received)
    if mode in ("1", "true"):
        assert sorted(received) == sorted([PROGRESS, UNKNOWN_PROGRESS])
    elif mode == "verbose" and component == "llama":
        assert all(line in "\n".join(received) for line in DIAGNOSTICS)
    else:
        assert received == []
    captured = (tmp_path / "captured.log").read_text(encoding = "utf-8")
    assert all(line in captured for line in [PROGRESS, UNKNOWN_PROGRESS, *DIAGNOSTICS])


@pytest.mark.parametrize("shell", SHELLS or [None])
@pytest.mark.parametrize("component", ["node", "llama", "whisper"])
def test_windows_real_download_reports_progress_while_active(
    tmp_path, shell, component, monkeypatch
):
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    monkeypatch.setenv("UNSLOTH_PROGRESS_PERCENT_STEP", "5")
    payload = b"x" * (20 * 1024 * 1024)
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload[: len(payload) // 5])
            self.wfile.flush()
            release.wait(30)
            self.wfile.write(payload[len(payload) // 5 :])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target = server.serve_forever, daemon = True)
    worker.start()
    archive = tmp_path / "runtime.zip"
    child = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'studio')!r})\n"
        f"import install_{component}_prebuilt as installer\n"
        "installer._LOG_TO_STDOUT = True\n"
        "print('diagnostic before download', flush=True)\n"
        f"installer.download_file('http://127.0.0.1:{server.server_port}/archive', Path({str(archive)!r}))\n"
        "print('diagnostic after download', flush=True)\n"
    )
    process, lines, reader = start_installer(tmp_path, shell, component, "1", child)
    received = []
    try:
        while not any("20.0%" in line for line in received):
            received.append(lines.get(timeout = 15))
        assert [float(line.split(": ")[1].split("%")[0]) for line in received] == [5, 10, 15, 20]
        assert process.poll() is None
        assert not archive.exists()
    finally:
        release.set()
        process.wait(timeout = 20)
        reader.join(timeout = 5)
        process.stdout.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout = 5)
    while not lines.empty():
        received.append(lines.get_nowait())
    assert process.returncode == 0, "\n".join(received)
    assert all(line.startswith("Downloading runtime.zip: ") for line in received)
    assert any("100.0%" in line for line in received)
    assert hashlib.sha256(archive.read_bytes()).digest() == hashlib.sha256(payload).digest()
    captured = (tmp_path / "captured.log").read_text(encoding = "utf-8")
    assert "diagnostic before download" in captured
    assert "diagnostic after download" in captured
