# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's folders in a shared system temp dir must be per user and refuse a planted link or foreign owner."""

from __future__ import annotations

import importlib
import os
import stat
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason = "POSIX owner and mode checks")

_SEED_SRC = (
    Path(__file__).resolve().parents[1] / "plugins" / "data-designer-github-repo-seed" / "src"
)


def _plant(tmp: Path, base: str) -> Path:
    """A link at both the old shared name and the per-user name, pointing at a folder we watch."""
    decoy = tmp / "decoy"
    decoy.mkdir(exist_ok = True)
    for name in (base, f"{base}-{os.getuid()}"):
        (tmp / name).symlink_to(decoy, target_is_directory = True)
    return decoy


def _open_dir(tmp: Path, base: str) -> None:
    for name in (base, f"{base}-{os.getuid()}"):
        (tmp / name).mkdir()
        os.chmod(tmp / name, 0o777)


def _private(path) -> bool:
    return stat.S_IMODE(os.lstat(path).st_mode) & 0o077 == 0


@pytest.fixture
def tmp(monkeypatch, tmp_path):
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(root))
    return root


def _ltx():
    from core.inference import video_ltx25
    return video_ltx25.ltx25_transformer_config_dir


def _tmp_root():
    from utils.paths import storage_roots
    return storage_roots.tmp_root


def _flashinfer_lock(monkeypatch):
    from core.inference import diffusion_nvfp4_install as inst

    real_access = os.access
    monkeypatch.setattr(
        inst.os, "access", lambda p, m: False if p == sys.prefix else real_access(p, m)
    )
    return lambda: os.path.dirname(inst._env_lock_path())


def _seed_reader(monkeypatch):
    try:
        importlib.import_module("data_designer.engine.resources.seed_reader")
    except ImportError:

        class SeedReader:
            def __class_getitem__(cls, item):
                return cls

        stubs = {
            "data_designer": {},
            "data_designer.lazy_heavy_imports": {},
            "data_designer.engine": {},
            "data_designer.engine.resources": {},
            "data_designer.engine.resources.seed_reader": {"SeedReader": SeedReader},
            "data_designer_github_repo_seed.config": {"GitHubRepoSeedSource": object},
        }
        for name, attrs in stubs.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, module)
        for name in stubs:
            parent, _, child = name.rpartition(".")
            if parent in stubs:
                setattr(sys.modules[parent], child, sys.modules[name])
    monkeypatch.syspath_prepend(str(_SEED_SRC))
    for name in [m for m in sys.modules if m.startswith("data_designer_github_repo_seed")]:
        if name != "data_designer_github_repo_seed.config":
            monkeypatch.delitem(sys.modules, name)
    impl = importlib.import_module("data_designer_github_repo_seed.impl")
    seen = []

    def materialize(cfg, out_dir):
        out_dir.mkdir(parents = True, exist_ok = True)
        seen.append(out_dir)
        out = out_dir / "rows.jsonl"
        out.write_text("{}\n", encoding = "utf-8")
        return out

    monkeypatch.setattr(impl, "materialize_to_jsonl", materialize)
    monkeypatch.setattr(impl, "_SCRAPE_CACHE", {})
    reader = impl.GitHubRepoSeedReader.__new__(impl.GitHubRepoSeedReader)
    reader.source = types.SimpleNamespace(
        repos = ["o/r"],
        token = "t",
        item_types = ["issues"],
        limit = 1,
        include_comments = False,
        max_comments_per_item = 0,
    )

    def call():
        reader.get_dataset_uri()
        return seen[-1]

    return call


SITES = {
    "ltx25_config": ("unsloth_ltx25_config", lambda mp: _ltx()),
    "tmp_root": ("unsloth-studio", lambda mp: _tmp_root()),
    "flashinfer_lock": ("unsloth-flashinfer-lock", _flashinfer_lock),
    "github_seed": ("studio-github-repo-seed", _seed_reader),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_planted_link_is_refused(site, tmp, monkeypatch):
    base, make = SITES[site]
    call = make(monkeypatch)
    decoy = _plant(tmp, base)
    with pytest.raises(PermissionError):
        call()
    assert list(decoy.iterdir()) == []


@pytest.mark.parametrize("site", sorted(SITES))
def test_folder_of_another_owner_is_refused(site, tmp, monkeypatch):
    base, make = SITES[site]
    call = make(monkeypatch)
    for name in (base, f"{base}-{os.getuid() + 1}"):
        (tmp / name).mkdir(mode = 0o700)
    monkeypatch.setattr(os, "getuid", lambda real = os.getuid(): real + 1)
    with pytest.raises(PermissionError):
        call()


@pytest.mark.parametrize("site", sorted(SITES))
def test_open_modes_are_taken_away(site, tmp, monkeypatch):
    base, make = SITES[site]
    call = make(monkeypatch)
    _open_dir(tmp, base)
    folder = Path(call())
    assert folder.name == f"{base}-{os.getuid()}" and _private(folder)


@pytest.mark.parametrize("site", sorted(SITES))
def test_fresh_folder_is_private_and_per_user(site, tmp, monkeypatch):
    base, make = SITES[site]
    call = make(monkeypatch)
    old = os.umask(0o022)
    try:
        folder = Path(call())
    finally:
        os.umask(old)
    assert folder.parent == tmp and folder.name == f"{base}-{os.getuid()}"
    assert _private(folder)


def test_mode_that_will_not_go_is_refused(tmp, monkeypatch):
    from utils.paths import private_tmp

    _open_dir(tmp, "probe")
    monkeypatch.setattr(private_tmp.os, "chmod", lambda *a, **k: None)
    with pytest.raises(PermissionError):
        private_tmp.private_tmp_dir("probe")
