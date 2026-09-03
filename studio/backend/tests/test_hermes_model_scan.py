# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermes Desktop stages its one-click downloads as flat GGUFs in <root>/models.

The staging rules mirror hermes_cli.local_runtime.bootstrap.staged_models: a file
Hermes will not serve is one Studio must not offer to load.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub.services.models.hermes import (  # noqa: E402
    scan_hermes_dir,
    staged_gguf_files,
    staged_model_id,
)
from utils.paths.storage_roots import _hermes_root, hermes_model_dirs  # noqa: E402


def _gguf(path: Path, size: int = 32) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\x00" * size)
    return path


def test_a_single_file_download_is_servable(tmp_path):
    _gguf(tmp_path / "Qwen3-8B-UD-Q4_K_M.gguf")
    assert [p.name for p in staged_gguf_files(tmp_path)] == ["Qwen3-8B-UD-Q4_K_M.gguf"]

    rows = scan_hermes_dir(tmp_path)
    assert len(rows) == 1
    assert rows[0].source == "hermes"
    assert rows[0].model_format == "gguf"
    assert rows[0].display_name == "Qwen3-8B-UD-Q4_K_M"
    assert rows[0].path == str(tmp_path / "Qwen3-8B-UD-Q4_K_M.gguf")


def test_a_complete_split_counts_once_by_its_first_part(tmp_path):
    for index in range(1, 4):
        _gguf(tmp_path / f"Big-Model-0000{index}-of-00003.gguf")

    staged = staged_gguf_files(tmp_path)
    assert [p.name for p in staged] == ["Big-Model-00001-of-00003.gguf"]

    rows = scan_hermes_dir(tmp_path)
    # llama.cpp opens the whole set from part one, so the row points at it and
    # carries the id Hermes knows the model by, not the part's own stem.
    assert len(rows) == 1
    assert rows[0].display_name == "Big-Model"
    assert rows[0].path.endswith("Big-Model-00001-of-00003.gguf")


def test_a_download_still_in_flight_is_not_offered(tmp_path):
    # Parts 1 and 2 of 3 on disk: loading this fails, so it must not be listed.
    _gguf(tmp_path / "Big-Model-00001-of-00003.gguf")
    _gguf(tmp_path / "Big-Model-00002-of-00003.gguf")

    assert staged_gguf_files(tmp_path) == []
    assert scan_hermes_dir(tmp_path) == []


def test_a_continuation_part_is_never_a_row_of_its_own(tmp_path):
    # Only the tail parts, e.g. after the first was deleted: nothing is servable.
    _gguf(tmp_path / "Big-Model-00002-of-00003.gguf")
    _gguf(tmp_path / "Big-Model-00003-of-00003.gguf")

    assert staged_gguf_files(tmp_path) == []


def test_companions_under_assets_are_not_models(tmp_path):
    _gguf(tmp_path / "Vision-Model-Q4_K_M.gguf")
    # Hermes parks vision projectors and spec-decode drafters here precisely so its
    # own router never lists them; a recursive scan would surface both as models.
    _gguf(tmp_path / "assets" / "mmproj-BF16.gguf")
    _gguf(tmp_path / "assets" / "Draft-Model-Q4_K_M.gguf")

    assert [p.name for p in staged_gguf_files(tmp_path)] == ["Vision-Model-Q4_K_M.gguf"]


def test_an_mmproj_left_at_the_top_level_is_still_not_a_model(tmp_path):
    _gguf(tmp_path / "Vision-Model-Q4_K_M.gguf")
    _gguf(tmp_path / "mmproj-Vision-Model-F16.gguf")

    assert [p.name for p in staged_gguf_files(tmp_path)] == ["Vision-Model-Q4_K_M.gguf"]


def test_non_gguf_files_and_a_missing_dir_are_ignored(tmp_path):
    (tmp_path / "presets.ini").write_text("[Qwen3-8B]\n", encoding = "utf-8")
    (tmp_path / "state.json").write_text("{}", encoding = "utf-8")

    assert staged_gguf_files(tmp_path) == []
    assert scan_hermes_dir(tmp_path / "nope") == []


def test_the_row_is_reachable_through_the_scan_limit(tmp_path):
    for name in ("A-Q4_K_M.gguf", "B-Q4_K_M.gguf", "C-Q4_K_M.gguf"):
        _gguf(tmp_path / name)

    assert len(scan_hermes_dir(tmp_path, limit = 2)) == 2


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3-8B-UD-Q4_K_M", "Qwen3-8B-UD-Q4_K_M"),
        ("Big-Model-00001-of-00003", "Big-Model"),
        ("Big-Model-00012-of-00099", "Big-Model"),
    ],
)
def test_staged_model_id_strips_only_a_split_suffix(tmp_path, name, expected):
    assert staged_model_id(tmp_path / f"{name}.gguf") == expected


class TestHermesRoot:
    """The models dir hangs off the ROOT, mirroring Hermes' own resolution."""

    def test_no_env_uses_the_native_home(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising = False)
        if sys.platform != "win32":
            assert _hermes_root() == Path.home() / ".hermes"

    def test_a_profile_under_the_native_home_still_means_the_root(self, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("POSIX home layout")
        monkeypatch.setenv("HERMES_HOME", str(Path.home() / ".hermes" / "profiles" / "coder"))
        # A 20 GB GGUF is a machine asset every profile shares, so it never
        # follows the active profile.
        assert _hermes_root() == Path.home() / ".hermes"

    def test_a_custom_deployment_root_is_used_as_is(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "opt" / "data"))
        assert _hermes_root() == tmp_path / "opt" / "data"

    def test_a_profile_outside_the_native_home_resolves_to_its_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "data" / "profiles" / "coder"))
        assert _hermes_root() == tmp_path / "data"

    def test_the_native_home_is_scanned_even_under_a_session_home(self, monkeypatch, tmp_path):
        # `unsloth start hermes` points HERMES_HOME at a throwaway session dir; the
        # user's real downloads stay under the native home and must still be found.
        if sys.platform == "win32":
            pytest.skip("POSIX home layout")
        native_models = tmp_path / "home" / ".hermes" / "models"
        native_models.mkdir(parents = True)
        session_models = tmp_path / "session" / "models"
        session_models.mkdir(parents = True)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "session"))

        found = {str(p) for p in hermes_model_dirs()}
        assert str(native_models) in found
        assert str(session_models) in found

    def test_only_directories_that_exist_are_returned(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "empty-home"))
        monkeypatch.delenv("HERMES_HOME", raising = False)
        assert hermes_model_dirs() == []
