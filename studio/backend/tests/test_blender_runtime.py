# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import hashlib
import io
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from core.inference import mcp_client, tools
from integrations.blender import runtime, service
from storage import mcp_servers_db


def archive_bytes(extra = None, package = b""):
    buffer = io.BytesIO()
    entries = {
        "root/mcp/blmcp/__init__.py": package,
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


STDIO_SERVER = (
    b"from mcp.server.fastmcp import FastMCP\n"
    b"def main():\n"
    b"    server = FastMCP('blender-mcp')\n"
    b"    @server.tool()\n"
    b"    def get_objects_summary() -> str:\n"
    b"        return 'ok'\n"
    b"    server.run(transport='stdio')\n"
    b"    return 0\n"
)


@pytest.fixture
def blender_row(tmp_path, monkeypatch):
    data = archive_bytes(package = STDIO_SERVER)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    download = Mock(return_value = data)
    monkeypatch.setattr(runtime, "_download", download)
    mcp_servers_db.create_server("blender1", "Blender", "", builtin_id = "blender")
    other_revision = tmp_path / "blender-mcp" / ("0" * 40 + "-runtime-v1")
    other_revision.mkdir(parents = True)
    (other_revision / ".ready").write_text("x")
    return download


def offered_tools():
    return [spec["function"]["name"] for spec in asyncio.run(tools.get_enabled_mcp_tools())]


def test_enabled_row_gets_its_tools_when_only_another_revision_is_cached(blender_row):
    assert offered_tools() == ["mcp__blender__get_objects_summary"]
    assert (runtime.runtime_path() / ".ready").is_file()
    blender_row.assert_called_once()


def test_disabled_row_never_downloads_the_runtime(blender_row):
    mcp_servers_db.update_server("blender1", {"is_enabled": False})
    assert offered_tools() == []
    blender_row.assert_not_called()
    assert service.catalog_item(mcp_servers_db.get_server("blender1")).unavailable_reason is None


def test_failed_runtime_download_is_reported_and_retried(blender_row):
    data = blender_row.return_value
    blender_row.side_effect = [httpx.ConnectError("offline"), data]
    assert offered_tools() == []
    item = service.catalog_item(mcp_servers_db.get_server("blender1"))
    assert item.is_enabled and item.available and "not downloaded" in item.unavailable_reason
    mcp_client.invalidate_tool_cache()
    assert offered_tools() == ["mcp__blender__get_objects_summary"]
    assert service.catalog_item(mcp_servers_db.get_server("blender1")).unavailable_reason is None
