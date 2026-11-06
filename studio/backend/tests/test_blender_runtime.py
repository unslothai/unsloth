# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import io
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from integrations.blender import runtime


def archive_bytes(extra = None):
    buffer = io.BytesIO()
    entries = {
        "root/mcp/blmcp/__init__.py": b"",
        "root/mcp/blmcp/data/prompts.yml": b"initial_instructions: |\n  # Bundled Manuals\n  old\n  # Executing Code\n",
        "root/mcp/blmcp/data/api/ignored.rst": b"not installed",
        "root/mcp/blmcp/tools/search_api_docs.py": b"not installed",
    }
    if extra:
        entries[extra] = b"unsafe"
    with tarfile.open(fileobj = buffer, mode = "w:gz") as archive:
        directory = tarfile.TarInfo("root/mcp/blmcp/data")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def test_failed_download_retry_and_concurrent_offline_cache(tmp_path, monkeypatch):
    data = archive_bytes()
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    download = Mock(side_effect = [b"corrupt", data])
    monkeypatch.setattr(runtime, "_download", download)
    with pytest.raises(ValueError, match = "checksum"):
        runtime.ensure_runtime()
    assert not runtime.runtime_path().exists()
    assert not list(runtime.runtime_path().parent.glob(".install-*"))
    with ThreadPoolExecutor(max_workers = 2) as pool:
        paths = list(pool.map(lambda _: runtime.ensure_runtime(), range(2)))
    assert paths[0] == paths[1] and download.call_count == 2
    assert (paths[0] / ".ready").is_file()
    assert not (paths[0] / "blmcp/data/api").exists()
    assert not (paths[0] / "blmcp/tools/search_api_docs.py").exists()
    assert "Bundled Manuals" not in (paths[0] / "blmcp/data/prompts.yml").read_text()


def test_verified_archive_still_rejects_traversal(tmp_path, monkeypatch):
    data = archive_bytes("root/mcp/blmcp/../../escape.py")
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match = "Unsafe"):
        runtime._extract(data, tmp_path / "runtime")
    assert not (tmp_path / "escape.py").exists()


def test_launcher_runs_v1_fastmcp_runtime_on_installed_mcp(tmp_path):
    package = tmp_path / "blmcp"
    package.mkdir()
    (tmp_path / ".ready").write_text("x")
    (package / "__init__.py").write_text(
        "import asyncio\n"
        "from mcp.server.fastmcp import FastMCP, Image\n"
        "def main():\n"
        "    server = FastMCP('blender-mcp', instructions='x')\n"
        "    @server.tool()\n"
        "    def fail() -> int:\n"
        "        raise RuntimeError('Blender error: NameError')\n"
        "    try:\n"
        "        asyncio.run(server.call_tool('fail', {}))\n"
        "    except Exception as exc:\n"
        "        print(exc)\n"
        "    print(Image(data=b'x', format='png').data)\n"
        "    return 0\n"
    )
    launcher = Path(runtime.__file__).with_name("launcher.py")
    result = subprocess.run(
        [sys.executable, str(launcher), str(tmp_path)], capture_output = True, text = True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Error executing tool fail: Blender error: NameError",
        "b'x'",
    ]
