# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
import storage.studio_db as studio_db


@pytest.fixture(autouse = True)
def _denylist_inert(monkeypatch):
    # Keep the denylist inert: macOS tmp_path is under denied /private/var.
    monkeypatch.setattr(studio_db, "is_denied_system_path", lambda _p: False)


def test_resolve_browse_target_returns_allowed_directory(tmp_path):
    allowed = tmp_path / "allowed"
    target = allowed / "models" / "nested"
    target.mkdir(parents = True)

    resolved = models_route._resolve_browse_target(str(target), [allowed])

    assert resolved == target.resolve()


def test_resolve_browse_target_rejects_outside_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    disallowed = tmp_path / "disallowed"
    allowed.mkdir()
    disallowed.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        models_route._resolve_browse_target(str(disallowed), [allowed])

    assert exc_info.value.status_code == 403


def test_resolve_browse_target_rejects_file_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    model_file = allowed / "model.gguf"
    model_file.write_text("gguf")

    with pytest.raises(HTTPException) as exc_info:
        models_route._resolve_browse_target(str(model_file), [allowed])

    assert exc_info.value.status_code == 400


def test_resolve_browse_target_allows_symlink_into_other_allowed_root(tmp_path):
    home_root = tmp_path / "home"
    scan_root = tmp_path / "scan"
    target = scan_root / "nested"
    home_root.mkdir()
    target.mkdir(parents = True)
    (home_root / "scan-link").symlink_to(scan_root, target_is_directory = True)

    resolved = models_route._resolve_browse_target(
        str(home_root / "scan-link" / "nested"),
        [home_root, scan_root],
    )

    assert resolved == target.resolve()


@pytest.mark.skipif(os.altsep is not None, reason = "POSIX-only path semantics")
def test_resolve_browse_target_allows_backslash_in_posix_segment(tmp_path):
    allowed = tmp_path / "allowed"
    target = allowed / r"dir\name"
    target.mkdir(parents = True)

    resolved = models_route._resolve_browse_target(str(target), [allowed])

    assert resolved == target.resolve()


def test_resolve_browse_target_accepts_a_path_spelled_through_a_symlinked_root(tmp_path):
    real = tmp_path / "real"
    (real / "models").mkdir(parents = True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory = True)

    resolved = models_route._resolve_browse_target(str(linked / "models"), [real.resolve()])

    assert resolved == (real / "models").resolve()


def test_resolve_browse_target_symlinked_root_spelling_cannot_escape(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (real / "out").symlink_to(outside, target_is_directory = True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory = True)

    with pytest.raises(HTTPException) as exc_info:
        models_route._resolve_browse_target(str(linked / "out"), [real.resolve()])

    assert exc_info.value.status_code == 403


def test_match_browse_child_falls_back_to_the_only_case_variant(tmp_path):
    (tmp_path / "Documents").mkdir()

    assert models_route._match_browse_child(tmp_path, "Documents").name == "Documents"
    assert models_route._match_browse_child(tmp_path, "documents").name == "Documents"
    assert models_route._match_browse_child(tmp_path, "missing") is None


def test_match_browse_child_refuses_an_ambiguous_case_variant(tmp_path):
    (tmp_path / "Data").mkdir()
    try:
        (tmp_path / "DATA").mkdir()
    except FileExistsError:
        pytest.skip("case-insensitive filesystem")

    assert models_route._match_browse_child(tmp_path, "data") is None


@pytest.fixture
def browse_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(models_route, "_build_browse_allowlist", lambda *_a, **_k: [home.resolve()])
    monkeypatch.setattr(models_route.account_access, "managed_account", lambda: False)
    return home


def test_browse_cuts_after_sorting_and_ignores_files(browse_home, monkeypatch):
    monkeypatch.setattr(models_route, "_BROWSE_ENTRY_CAP", 3)
    for index in range(40):
        (browse_home / f"file-{index:02d}.txt").write_text("x")
    for name in ("delta", "Alpha", "charlie", "bravo", "echo"):
        (browse_home / name).mkdir()

    listing = models_route.browse_folders(str(browse_home), False, "owner", False)

    assert [entry.name for entry in listing.entries] == ["Alpha", "bravo", "charlie"]
    assert listing.truncated is True


def test_browse_without_model_hints_skips_the_probe(browse_home, monkeypatch):
    (browse_home / "models").mkdir()
    (browse_home / "models" / "m.gguf").write_text("gguf")

    def _fail(*_args, **_kwargs):
        raise AssertionError("model probe ran")

    hinted = models_route.browse_folders(str(browse_home), False, "owner", True)
    assert hinted.entries[0].has_models is True

    monkeypatch.setattr(models_route, "_looks_like_model_dir", _fail)
    monkeypatch.setattr(models_route, "_count_model_files", _fail)
    plain = models_route.browse_folders(str(browse_home), False, "owner", False)
    assert [(entry.name, entry.has_models) for entry in plain.entries] == [("models", False)]
    assert plain.model_files_here == 0


def test_create_browse_folder_makes_one_folder_and_lists_it(browse_home):
    payload = models_route.BrowseFolderCreate(parent = str(browse_home), name = "New project")

    created = models_route.create_browse_folder(payload, "owner")

    assert created.current == str((browse_home / "New project").resolve())
    assert (browse_home / "New project").is_dir()
    with pytest.raises(HTTPException) as exc_info:
        models_route.create_browse_folder(payload, "owner")
    assert exc_info.value.status_code == 409


@pytest.mark.parametrize(
    "name",
    ["", " ", "..", "a/b", "a\\b", "bad:name", "trailing.", " lead", "NUL", "con.txt", ".ssh"],
)
def test_create_browse_folder_rejects_unsafe_names(browse_home, name):
    payload = models_route.BrowseFolderCreate(parent = str(browse_home), name = name)

    with pytest.raises(HTTPException) as exc_info:
        models_route.create_browse_folder(payload, "owner")

    assert exc_info.value.status_code == 400
    assert list(browse_home.iterdir()) == []


def test_create_browse_folder_stays_inside_the_allowlist(browse_home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = models_route.BrowseFolderCreate(parent = str(outside), name = "x")

    with pytest.raises(HTTPException) as exc_info:
        models_route.create_browse_folder(payload, "owner")

    assert exc_info.value.status_code == 403
    assert list(outside.iterdir()) == []


def test_create_browse_folder_refuses_studio_data_and_api_keys(browse_home, monkeypatch):
    import utils.paths as paths
    from utils.account_context import OWNER, as_api_caller, run_as

    studio = browse_home / "studio"
    (studio / "outputs").mkdir(parents = True)
    monkeypatch.setattr(paths, "studio_root", lambda: studio)
    monkeypatch.setattr(paths, "outputs_root", lambda: studio / "outputs")
    monkeypatch.setattr(paths, "exports_root", lambda: browse_home / "exports")

    with pytest.raises(HTTPException) as exc_info:
        models_route.create_browse_folder(
            models_route.BrowseFolderCreate(parent = str(studio), name = "x"), "owner"
        )
    assert exc_info.value.status_code == 403
    models_route.create_browse_folder(
        models_route.BrowseFolderCreate(parent = str(studio / "outputs"), name = "run-1"), "owner"
    )
    assert (studio / "outputs" / "run-1").is_dir()

    with pytest.raises(HTTPException) as exc_info:
        run_as(
            as_api_caller(OWNER),
            models_route.create_browse_folder,
            models_route.BrowseFolderCreate(parent = str(browse_home), name = "from-api"),
            "owner",
        )
    assert exc_info.value.status_code == 403
    assert not (browse_home / "from-api").exists()


def test_browse_names_its_places(browse_home, monkeypatch):
    (browse_home / "Documents").mkdir()
    (browse_home / ".lmstudio" / "models").mkdir(parents = True)
    monkeypatch.setattr(models_route.Path, "home", classmethod(lambda cls: browse_home))
    import utils.paths as paths

    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(
        paths, "well_known_model_dirs", lambda: [browse_home / ".lmstudio" / "models"]
    )
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])

    listing = models_route.browse_folders(str(browse_home), False, "owner", False)

    named = {place.label: (place.kind, place.path) for place in listing.places}
    assert named["Home"] == ("home", str(browse_home.resolve()))
    assert named["Documents"][0] == "user"
    assert named["lmstudio/models"][0] == "models"
    assert "Desktop" not in named
    assert listing.suggestions == [place.path for place in listing.places]
