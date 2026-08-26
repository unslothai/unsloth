# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-lifetime behavior for repository discovery routes."""

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from core.agent_workspace.common import ProjectWorkspace
from core.agent_workspace import discovery as discovery_module
from core.agent_workspace.discovery import build_repository_map
from routes import agent_workspace as agent_workspace_routes


def test_client_disconnect_cancels_live_repository_map_worker(tmp_path, monkeypatch):
    for index in range(500):
        (tmp_path / f"file-{index:03}.txt").write_text("x", encoding = "utf-8")
    metadata = tmp_path.stat()
    started = threading.Event()
    opened = 0

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            await asyncio.sleep(0)
            return started.is_set()

    original_open = discovery_module._open_beneath

    def slow_open(root_fd, relative):
        nonlocal opened
        opened += 1
        started.set()
        time.sleep(0.005)
        return original_open(root_fd, relative)

    monkeypatch.setattr(
        agent_workspace_routes,
        "project_workspace",
        lambda _project_id: ProjectWorkspace(
            project_id = "project",
            root = tmp_path,
            kind = "folder",
            device_id = metadata.st_dev,
            file_id = metadata.st_ino,
        ),
    )
    monkeypatch.setattr(discovery_module, "_open_beneath", slow_open)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            agent_workspace_routes.repository_map(
                "project",
                DisconnectingRequest(),
                max_paths = 100,
                max_total_bytes = 1024,
                current_subject = "tester",
            )
        )

    assert caught.value.status_code == 499
    assert started.is_set()
    assert opened < 500


def test_repository_map_rejects_root_replaced_after_project_resolution(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    metadata = root.stat()

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    monkeypatch.setattr(
        agent_workspace_routes,
        "project_workspace",
        lambda _project_id: ProjectWorkspace(
            project_id = "project",
            root = root,
            kind = "folder",
            device_id = metadata.st_dev,
            file_id = metadata.st_ino,
        ),
    )

    def replace_then_build(path, **kwargs):
        path.rename(tmp_path / "original-repository")
        path.mkdir()
        (path / "replacement.py").write_text("replacement\n", encoding = "utf-8")
        return build_repository_map(path, **kwargs)

    monkeypatch.setattr(
        agent_workspace_routes,
        "build_repository_map",
        replace_then_build,
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            agent_workspace_routes.repository_map(
                "project",
                ConnectedRequest(),
                max_paths = 100,
                max_total_bytes = 1024,
                current_subject = "tester",
            )
        )

    assert caught.value.status_code == 409
    assert "identity changed" in str(caught.value.detail)
