# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os

import pytest

from core import workspace_files as browser


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    info = root.stat()
    return {"path": str(root), "root_device": info.st_dev, "root_inode": info.st_ino}


def test_lists_assets_without_indexing_filter(folder):
    from pathlib import Path

    root = Path(folder["path"])
    for name in ("main.py", "engine.cpp", "texture.png", "model.blend", "budget.xlsx", "README.md"):
        (root / name).write_bytes(b"asset")
    (root / "src").mkdir()
    result = browser.list_directory(folder)
    assert result["entries"][0]["name"] == "src"
    assert {entry["name"] for entry in result["entries"]} == {
        "src",
        "main.py",
        "engine.cpp",
        "texture.png",
        "model.blend",
        "budget.xlsx",
        "README.md",
    }
    assert not result["truncated"]


@pytest.mark.parametrize(
    "path",
    ["../secret", "/etc/passwd", "C:/Windows", "src/../../secret", "..\\secret", "foo\x00bar"],
)
def test_rejects_escape_paths(folder, path):
    with pytest.raises(ValueError):
        browser.resolve_path(folder, path)


def test_nested_browsing_and_utf8_preview(folder):
    from pathlib import Path

    root = Path(folder["path"])
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text('print("héllo")', encoding="utf-8")
    assert browser.list_directory(folder, "src")["entries"][0]["path"] == "src/main.py"
    assert browser.preview_file(folder, "src/main.py") == {
        "kind": "text",
        "content": 'print("héllo")',
        "truncated": False,
    }


def test_binary_assets_have_no_text_preview(folder):
    from pathlib import Path

    (Path(folder["path"]) / "model.blend").write_bytes(b"BLENDER\x00data")
    assert browser.preview_file(folder, "model.blend")["kind"] == "unsupported"


def test_previews_are_bounded(folder, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(browser, "MAX_TEXT_BYTES", 4)
    (Path(folder["path"]) / "main.py").write_bytes(b"abc\xc3\xa9more")
    assert browser.preview_file(folder, "main.py") == {
        "kind": "text",
        "content": "abc",
        "truncated": True,
    }


def test_large_images_not_read(folder, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(browser, "MAX_IMAGE_BYTES", 4)
    (Path(folder["path"]) / "texture.png").write_bytes(b"12345")
    assert browser.preview_file(folder, "texture.png")["kind"] == "unsupported"


def test_image_preview(folder):
    from pathlib import Path

    (Path(folder["path"]) / "texture.png").write_bytes(b"png")
    assert browser.preview_file(folder, "texture.png") == {
        "kind": "image",
        "mimeType": "image/png",
        "data": "cG5n",
    }


def test_listing_cap_is_visible(folder, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(browser, "MAX_ENTRIES", 2)
    for name in ("a.py", "b.cpp", "c.xlsx"):
        (Path(folder["path"]) / name).touch()
    result = browser.list_directory(folder)
    assert len(result["entries"]) == 2
    assert result["truncated"]


def test_replaced_root_rejected(folder):
    folder["root_inode"] += 1
    with pytest.raises(ValueError, match="changed"):
        browser.list_directory(folder)


def test_symlink_cannot_expose_outside_files(folder, tmp_path):
    from pathlib import Path

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("private")
    link = Path(folder["path"]) / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    assert browser.list_directory(folder)["entries"] == []
    with pytest.raises(ValueError):
        browser.preview_file(folder, "linked/secret.py")


def test_fifo_not_read(folder):
    from pathlib import Path

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not supported")
    os.mkfifo(Path(folder["path"]) / "pipe")
    with pytest.raises(ValueError):
        browser.preview_file(folder, "pipe")


def test_sensitive_folder_not_listed(folder):
    from pathlib import Path

    (Path(folder["path"]) / ".ssh").mkdir()
    assert browser.list_directory(folder)["entries"] == []
    with pytest.raises(ValueError):
        browser.list_directory(folder, ".ssh")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_windows_junction_cannot_expose_outside_files(folder, tmp_path):
    import subprocess
    from pathlib import Path

    outside = tmp_path / "outside-junction"
    outside.mkdir()
    (outside / "secret.py").write_text("private")
    junction = Path(folder["path"]) / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True
    )
    assert result.returncode == 0, result.stderr
    try:
        assert browser.list_directory(folder)["entries"] == []
        with pytest.raises(ValueError, match="junction"):
            browser.preview_file(folder, "junction/secret.py")
    finally:
        junction.rmdir()


def test_file_replaced_before_open_is_rejected(folder, monkeypatch):
    from pathlib import Path

    root = Path(folder["path"])
    target = root / "main.py"
    target.write_text("original")
    replacement = root / "replacement.py"
    replacement.write_text("different file")
    original_open = os.open

    def replacing_open(path, flags):
        replacement.replace(target)
        return original_open(path, flags)

    monkeypatch.setattr(browser.os, "open", replacing_open)
    with pytest.raises(ValueError, match="changed"):
        browser.preview_file(folder, "main.py")


def test_internal_files_hidden_and_unreadable(folder):
    from pathlib import Path

    for name in (".unsloth_sandbox", ".unsloth_sandbox_remap.json"):
        (Path(folder["path"]) / name).write_text("internal")
        with pytest.raises(ValueError):
            browser.preview_file(folder, name)
    assert browser.list_directory(folder)["entries"] == []


def test_search_nested_names(folder):
    from pathlib import Path

    root = Path(folder["path"])
    (root / "src").mkdir()
    (root / "src" / "Main.py").write_text("print(42)")
    assert browser.search_files(folder, "MAIN")["entries"][0]["path"] == "src/Main.py"


def test_routes_without_rag(folder, monkeypatch):
    import importlib.util
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location(
        "test_workspace_routes", Path(__file__).parents[1] / "routes" / "workspace_files.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(folder["path"])
    missing = root / "not-created"
    monkeypatch.setattr(
        module, "workspace_path", lambda session: root if session == "chat" else missing
    )
    app = FastAPI()
    app.include_router(module.router, prefix="/api/workspace-files")
    with TestClient(app) as client:
        assert (
            client.get("/api/workspace-files/files", params={"session": "chat"}).status_code == 401
        )
        app.dependency_overrides[module.get_current_subject] = lambda: "test-user"
        (root / "main.py").write_text("print(42)")
        assert (
            client.get("/api/workspace-files/files", params={"session": "chat"}).json()["entries"][
                0
            ]["name"]
            == "main.py"
        )
        assert (
            client.get(
                "/api/workspace-files/preview", params={"session": "chat", "path": "main.py"}
            ).json()["content"]
            == "print(42)"
        )
        assert (
            client.get(
                "/api/workspace-files/preview", params={"session": "chat", "path": "../outside"}
            ).status_code
            == 400
        )
        assert (
            client.get("/api/workspace-files/files", params={"session": "other"}).json()["entries"]
            == []
        )
        assert not missing.exists()


def test_search_has_total_entry_budget(folder, monkeypatch):
    from pathlib import Path

    root = Path(folder["path"])
    for index in range(4):
        (root / f"file{index}.py").touch()
    monkeypatch.setattr(browser, "MAX_SEARCH_ENTRIES", 2)
    result = browser.search_files(folder, ".py")
    assert len(result["entries"]) == 2
    assert result["truncated"]
