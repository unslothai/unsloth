# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_SINGLE_ENV = _BACKEND_DIR / "requirements" / "single-env"

pytest.importorskip("fastmcp")
pytest.importorskip("packaging")

from packaging.requirements import Requirement  # noqa: E402

_SERVER = """
import sys
from fastmcp import FastMCP

mcp = FastMCP("recipe")

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b

mcp.run(transport = "http", host = "127.0.0.1", port = int(sys.argv[1]), show_banner = False)
"""

_RECIPE_WORKER = """
import asyncio
import sys
import core.data_recipe
from data_designer.engine.mcp.io import ClientSession, streamablehttp_client

async def main():
    async with streamablehttp_client(sys.argv[1], headers = {"x-recipe": "1"}) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print([tool.name for tool in (await session.list_tools()).tools])
            print("modern" if session.discover_result is not None else "legacy")

asyncio.run(main())
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError("MCP server did not start")


def _mcp_requirement(path: Path) -> Requirement:
    lines = path.read_text(encoding = "utf-8").splitlines()
    return next(Requirement(line) for line in lines if re.match(r"mcp\b[<>=]", line))


@pytest.mark.parametrize("name", ["constraints.txt", "data-designer-deps.txt"])
def test_single_env_installs_the_mcp_that_speaks_the_modern_protocol(name):
    requirement = _mcp_requirement(_SINGLE_ENV / name)
    assert requirement.specifier.contains("2.3.0")
    assert not requirement.specifier.contains("1.30.0")


def test_data_designer_mcp_pin_is_relaxed_for_pip_check(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "patch_metadata", _SINGLE_ENV / "patch_metadata.py"
    )
    patch_metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_metadata)
    metadata = tmp_path / "METADATA"
    metadata.write_text("Requires-Dist: mcp<2,>=1.26.0\n", encoding = "utf-8")

    assert patch_metadata.patch_file(metadata) is True

    relaxed = Requirement(metadata.read_text(encoding = "utf-8").split(": ", 1)[1].strip())
    assert relaxed.specifier.contains("2.3.0")


def test_recipe_worker_reaches_mcp_through_data_designers_client_import(tmp_path):
    package = tmp_path / "data_designer" / "engine" / "mcp"
    package.mkdir(parents = True)
    for folder in (package, package.parent, package.parent.parent):
        (folder / "__init__.py").write_text("", encoding = "utf-8")
    (package / "io.py").write_text(
        "from mcp import ClientSession\n"
        "from mcp.client.streamable_http import streamablehttp_client\n",
        encoding = "utf-8",
    )
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_SERVER), str(port)],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        worker = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_RECIPE_WORKER), f"http://127.0.0.1:{port}/mcp"],
            cwd = _BACKEND_DIR,
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(_BACKEND_DIR)])},
            capture_output = True,
            text = True,
            timeout = 120,
        )
    finally:
        server.kill()
        server.wait()
    assert worker.returncode == 0, worker.stderr
    assert "['add']" in worker.stdout
    # A modern-only server rejects the bare initialize handshake data-designer sends.
    assert "modern" in worker.stdout.split()
