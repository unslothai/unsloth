# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The ROCm SDPA check child (`python -c`) must not run from the shared temp dir: its cwd is
sys.path[0], so another user's json.py / torch.py there would execute in the child."""

import importlib.util
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "unsloth_rocm_sdpa_check_under_test",
        _REPO_ROOT / "unsloth" / "utils" / "rocm_sdpa_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_child_does_not_import_from_shared_temp(tmp_path, monkeypatch):
    shared = tmp_path / "shared_tmp"
    shared.mkdir()
    marker = tmp_path / "planted_ran"
    for name in ("json.py", "torch.py"):
        (shared / name).write_text(
            f"open({str(marker)!r}, 'a').write({name!r})\nraise SystemExit(7)\n", encoding = "utf-8"
        )
    monkeypatch.setattr(tempfile, "tempdir", str(shared))
    module = _load()
    results = module._run_children(0)
    assert set(results) == {"flash", "efficient"}
    assert not marker.exists(), f"planted module ran in the check child: {marker.read_text()}"
    # The private directory is removed afterwards; only the planted files remain.
    assert sorted(os.listdir(shared)) == ["json.py", "torch.py"]
