# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stdio MCP servers get the managed Node bin dir on PATH.

Run from studio/backend:  python -m pytest tests/test_mcp_stdio_node_path.py -q
"""

import os

import pytest

from core.inference import mcp_client
from utils import node_runtime


@pytest.fixture
def managed_node(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    bin_dir = tmp_path / "studio" / "node" / ("" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(node_runtime, "managed_node_bin_dir", lambda: bin_dir)
    return bin_dir


def test_bin_dir_none_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    assert node_runtime.managed_node_bin_dir() is None


def test_path_prepends_managed_node(managed_node):
    assert node_runtime.path_with_managed_node("/usr/bin") == f"{managed_node}{os.pathsep}/usr/bin"


def test_path_unchanged_when_already_present(managed_node):
    existing = f"{managed_node}{os.pathsep}/usr/bin"
    assert node_runtime.path_with_managed_node(existing) == existing


def test_path_unchanged_without_managed_node(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    assert node_runtime.path_with_managed_node("/usr/bin") == "/usr/bin"


def test_stdio_env_adds_node_to_inherited_path(managed_node, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = mcp_client._stdio_env(None)
    assert env["PATH"] == f"{managed_node}{os.pathsep}/usr/bin"


def test_stdio_env_keeps_server_env_and_extends_its_path(managed_node):
    env = mcp_client._stdio_env({"API_KEY": "sk-1", "PATH": "/opt/bin"})
    assert env["API_KEY"] == "sk-1"
    assert env["PATH"] == f"{managed_node}{os.pathsep}/opt/bin"


def test_stdio_env_is_none_without_managed_node_or_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("PATH", raising = False)
    assert mcp_client._stdio_env(None) is None


def test_client_passes_node_path_to_transport(managed_node, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setenv("PATH", "/usr/bin")
    client = mcp_client._client("npx -y @modelcontextprotocol/server-filesystem /tmp", None)
    assert client.transport.env["PATH"].split(os.pathsep)[0] == str(managed_node)
