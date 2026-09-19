# SPDX-License-Identifier: AGPL-3.0-only
"""Integrity controls for the reproducible patched-MXC build input."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


_ROOT = Path(__file__).resolve().parents[2] / "native" / "mxc-runner"
_SPEC = importlib.util.spec_from_file_location(
    "unsloth_mxc_prepare_build", _ROOT / "tools" / "prepare_build.py"
)
assert _SPEC and _SPEC.loader
prepare_build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_build)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "mxc-tests@example.invalid")
    _git(repo, "config", "user.name", "MXC Tests")
    (repo / "value.txt").write_text("before\n", encoding="utf-8")
    _git(repo, "add", "value.txt")
    _git(repo, "commit", "--quiet", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_checked_in_mxc_build_inputs_have_exact_approved_hashes():
    prepare_build.verify_build_inputs(_ROOT)


def test_expected_upstream_commit_is_accepted_and_wrong_commit_is_rejected(tmp_path):
    repo, revision = _repository(tmp_path)
    prepare_build.verify_upstream_checkout(repo, revision)
    with pytest.raises(prepare_build.PreparationError, match="revision mismatch"):
        prepare_build.verify_upstream_checkout(repo, "0" * 40)


def test_dirty_upstream_checkout_is_rejected(tmp_path):
    repo, revision = _repository(tmp_path)
    (repo / "ambient.txt").write_text("not approved", encoding="utf-8")
    with pytest.raises(prepare_build.PreparationError, match="dirty"):
        prepare_build.verify_upstream_checkout(repo, revision)


def test_patch_hash_mismatch_is_rejected_before_git_apply(tmp_path):
    root = tmp_path / "runner"
    (root / "upstream").mkdir(parents=True)
    shutil.copy2(_ROOT / "Cargo.lock", root / "Cargo.lock")
    shutil.copy2(_ROOT / "upstream" / "Cargo.patched.lock", root / "upstream")
    patch = root / "upstream" / "mxc-ca7ea12-no-dacl-tier.patch"
    patch.write_bytes((_ROOT / "upstream" / patch.name).read_bytes() + b"tampered\n")
    with pytest.raises(prepare_build.PreparationError, match="patch.*mismatch"):
        prepare_build.verify_build_inputs(root)


def test_patch_application_failure_is_atomic(tmp_path):
    repo, _ = _repository(tmp_path)
    patch = tmp_path / "broken.patch"
    patch.write_text("not a git patch\n", encoding="utf-8")
    with pytest.raises(prepare_build.PreparationError, match="git apply"):
        prepare_build.apply_approved_patch(repo, patch, "0" * 40)
    assert (repo / "value.txt").read_text(encoding="utf-8") == "before\n"


def test_approved_patch_must_produce_the_complete_expected_tree(tmp_path):
    repo, _ = _repository(tmp_path)
    (repo / "value.txt").write_text("after\n", encoding="utf-8")
    patch = tmp_path / "change.patch"
    patch.write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=repo))
    _git(repo, "add", "--all")
    expected_tree = _git(repo, "write-tree")
    _git(repo, "reset", "--hard", "HEAD")
    prepare_build.apply_approved_patch(repo, patch, expected_tree)
    assert (repo / "value.txt").read_text(encoding="utf-8") == "after\n"

    _git(repo, "reset", "--hard", "HEAD")
    with pytest.raises(prepare_build.PreparationError, match="tree mismatch"):
        prepare_build.apply_approved_patch(repo, patch, "0" * 40)
