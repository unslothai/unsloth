# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import io
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import httpx
from utils.paths.storage_roots import cache_root

REVISION = "4309a39646e644261624bfcd2bca669b343b7621"
ARCHIVE_SHA256 = "acb68eb4beff27a84ba751931745e62f03ad51b7be50b3a924624153b6c38197"
ARCHIVE_URL = f"https://projects.blender.org/api/v1/repos/lab/blender_mcp/archive/{REVISION}.tar.gz"
_EXCLUDED = {
    "get_python_api_docs.py",
    "search_api_docs.py",
    "search_manual_docs.py",
    "rst_doc_search.py",
    "rst_parse_docs.py",
}


def runtime_path() -> Path:
    return cache_root() / "blender-mcp" / (REVISION + "-runtime-v1")


def _download() -> bytes:
    data = bytearray()
    with httpx.stream("GET", ARCHIVE_URL, follow_redirects = True, timeout = 30) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("Blender MCP download exceeds the size limit")
    return bytes(data)


def _extract(data: bytes, target: Path) -> None:
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Blender MCP download checksum mismatch")
    with tarfile.open(fileobj = io.BytesIO(data), mode = "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 4 or parts[1:3] != ("mcp", "blmcp"):
                continue
            if member.isdir():
                continue
            relative = PurePosixPath(*parts[2:])
            if ".." in parts or "\\" in member.name or ":" in member.name or not member.isfile():
                raise ValueError("Unsafe Blender MCP archive entry")
            if relative.name in _EXCLUDED:
                continue
            if relative.suffix != ".py" and str(relative) != "blmcp/data/prompts.yml":
                continue
            if member.size > 1024 * 1024:
                raise ValueError("Blender MCP archive entry exceeds the size limit")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Missing Blender MCP archive entry")
            destination = target / str(relative)
            destination.parent.mkdir(parents = True, exist_ok = True)
            destination.write_bytes(source.read())
    prompt = target / "blmcp/data/prompts.yml"
    text = prompt.read_text(encoding = "utf-8")
    start, end = text.index("  # Bundled Manuals"), text.index("  # Executing Code")
    prompt.write_text(
        text[:start]
        + "  # Documentation\n\n  Offline documentation tools are not installed.\n\n"
        + text[end:],
        encoding = "utf-8",
    )
    if not (target / "blmcp/__init__.py").is_file():
        raise ValueError("Incomplete Blender MCP runtime")


def ensure_runtime() -> Path:
    from filelock import FileLock

    target = runtime_path()
    target.parent.mkdir(parents = True, exist_ok = True)
    with FileLock(str(target) + ".lock", timeout = 120):
        if (target / ".ready").is_file():
            return target
        if target.exists():
            raise ValueError("Incomplete Blender MCP cache; remove this runtime cache and retry")
        with tempfile.TemporaryDirectory(prefix = ".install-", dir = target.parent) as temporary:
            staged = Path(temporary) / "runtime"
            staged.mkdir()
            _extract(_download(), staged)
            (staged / ".ready").write_text(ARCHIVE_SHA256, encoding = "ascii")
            staged.rename(target)
    return target
