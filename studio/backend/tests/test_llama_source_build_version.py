# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#12798: a markerless llama.cpp built from a ``git clone --depth 1`` reports build 1, so the
update banner read ``unknown -> <latest>`` forever. The build is recovered from the bNNNN tag
naming the checkout's HEAD, using real shallow clones (file:// so --depth works offline)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_update as upd  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason = "git not installed")

# llama-server path -> what its --version prints; answered in-process so the test runs on Windows too.
_VERSIONS: dict[str, str] = {}
_real_run = subprocess.run


@pytest.fixture(autouse = True)
def _fake_version(monkeypatch):
    def run(args, *a, **k):
        if isinstance(args, list) and args and args[0] in _VERSIONS:
            return subprocess.CompletedProcess(args, 0, stdout = "", stderr = _VERSIONS[args[0]])
        return _real_run(args, *a, **k)

    monkeypatch.setattr(upd.subprocess, "run", run)
    yield
    _VERSIONS.clear()


LATEST_MIX = "b11408-mix-1e24fc5"


def _git(*args, cwd = None) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME = "t",
        GIT_AUTHOR_EMAIL = "t@t",
        GIT_COMMITTER_NAME = "t",
        GIT_COMMITTER_EMAIL = "t@t",
        GIT_CONFIG_GLOBAL = os.devnull,
        GIT_CONFIG_NOSYSTEM = "1",
    )
    return subprocess.run(
        ["git", *args], cwd = cwd, env = env, check = True, capture_output = True, text = True
    ).stdout.strip()


def _upstream(
    tmp_path: Path,
    tag: str | None,
    extra_tag: str | None = None,
) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "-q", "-b", "master", cwd = repo)
    for i in range(3):
        (repo / "f.txt").write_text(str(i))
        _git("add", "f.txt", cwd = repo)
        _git("commit", "-q", "-m", f"c{i}", cwd = repo)
    if tag:
        _git("tag", tag, cwd = repo)
    if extra_tag:
        _git("tag", extra_tag, cwd = repo)
    return repo


def _stub_server(root: Path, version_line: str) -> str:
    exe = root / "build" / "bin" / "llama-server"
    exe.parent.mkdir(parents = True, exist_ok = True)
    exe.write_text("")
    _VERSIONS[str(exe)] = version_line + "\n"
    return str(exe)


def _clone_branch(tmp_path: Path, upstream: Path, ref: str) -> Path:
    # setup.sh / setup.ps1 fresh clone: git clone --depth 1 --branch <ref>
    dest = tmp_path / "llama.cpp"
    _git("clone", "-q", "--depth", "1", "--branch", ref, upstream.as_uri(), str(dest))
    return dest


def _clone_fetch_checkout(tmp_path: Path, upstream: Path, ref: str) -> Path:
    # setup.ps1 reuse path: fetch --depth 1 origin <ref> + checkout -B unsloth-llama-build FETCH_HEAD
    dest = tmp_path / "llama.cpp"
    _git("clone", "-q", "--depth", "1", "--no-tags", upstream.as_uri(), str(dest))
    _git("-C", str(dest), "fetch", "-q", "--depth", "1", "--no-tags", "origin", ref)
    _git("-C", str(dest), "checkout", "-q", "-B", "unsloth-llama-build", "FETCH_HEAD")
    return dest


def _short(dest: Path) -> str:
    return _git("-C", str(dest), "rev-parse", "--short=7", "HEAD")


@pytest.mark.parametrize("layout", [_clone_branch, _clone_fetch_checkout])
@pytest.mark.parametrize("fmt", ["version: 0.6.0 (build 1, commit {c})", "version: 1 ({c})"])
def test_build_recovered_from_shallow_tag(tmp_path, layout, fmt):
    dest = layout(tmp_path, _upstream(tmp_path, "b11408"), "b11408")
    exe = _stub_server(dest, fmt.format(c = _short(dest)))
    assert upd._installed_build_number(exe) == 11408


def test_reported_commit_mismatch_is_unknown(tmp_path):
    dest = _clone_branch(tmp_path, _upstream(tmp_path, "b11408"), "b11408")
    exe = _stub_server(dest, "version: 0.6.0 (build 1, commit deadbee)")
    assert upd._installed_build_number(exe) is None


def test_untagged_checkout_is_unknown(tmp_path):
    dest = _clone_branch(tmp_path, _upstream(tmp_path, None), "master")
    exe = _stub_server(dest, f"version: 1 ({_short(dest)})")
    assert upd._installed_build_number(exe) is None


def test_non_build_tag_is_unknown(tmp_path):
    dest = _clone_branch(tmp_path, _upstream(tmp_path, "v0.6.0"), "v0.6.0")
    exe = _stub_server(dest, f"version: 0.6.0 (build 1, commit {_short(dest)})")
    assert upd._installed_build_number(exe) is None


def test_loose_tag_overrides_stale_packed_tag(tmp_path):
    # A full clone packs b11408 at HEAD; moving the tag writes a loose ref git reads instead.
    dest = tmp_path / "llama.cpp"
    _git("clone", "-q", _upstream(tmp_path, "b11408").as_uri(), str(dest))
    _git("-C", str(dest), "tag", "-f", "b11408", "HEAD~1")
    exe = _stub_server(dest, f"version: 1 ({_short(dest)})")
    assert upd._installed_build_number(exe) is None


def test_real_build_number_does_not_read_git(tmp_path, monkeypatch):
    exe = _stub_server(tmp_path / "llama.cpp", "version: 0.6.0 (build 11420, commit abc1234)")
    monkeypatch.setattr(upd, "_checkout_tag_build", lambda *a: pytest.fail("git metadata read"))
    assert upd._installed_build_number(exe) == 11420


def _status(monkeypatch, exe: str) -> dict:
    res = {
        "prebuilt_available": True,
        "repo": "unslothai/llama.cpp",
        "release_tag": LATEST_MIX,
        "llama_tag": "b11408",
        "asset": "app-b11408-mix-1e24fc5-linux-x64-cuda13.tar.gz",
        "install_kind": "linux-cuda",
    }
    monkeypatch.setattr(upd, "_find_binary", lambda: exe)
    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", lambda **_: dict(res))
    monkeypatch.setattr(upd, "latest_release_assets", lambda *a, **k: {res["asset"]: 1})
    monkeypatch.setattr(upd, "_studio_custom_path_active", lambda: False)
    monkeypatch.setattr(upd, "_active_install_is_local_link", lambda b: False)
    monkeypatch.setattr(upd, "update_checks_disabled", lambda: False)
    return upd._llama_only_status()


@pytest.mark.parametrize(
    "tag, installed, offered",
    [
        ("b11408", "b11408", True),  # same base: the mix still carries extra patches
        ("b11420", "b11420", False),  # source newer than the prebuilt: downgrade guard
        (None, None, True),  # untagged: unchanged, still offered as unknown
    ],
)
def test_source_build_status(tmp_path, monkeypatch, tag, installed, offered):
    dest = _clone_branch(tmp_path, _upstream(tmp_path, tag), tag or "master")
    exe = _stub_server(dest, f"version: 0.6.0 (build 1, commit {_short(dest)})")
    s = _status(monkeypatch, exe)
    assert s["source_build"] is True
    assert s["installed_tag"] == installed
    assert s["latest_tag"] == LATEST_MIX
    assert s["update_available"] is offered
