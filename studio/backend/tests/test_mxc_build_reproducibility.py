# SPDX-License-Identifier: AGPL-3.0-only
"""Integrity controls for the reproducible patched-MXC build input."""

from __future__ import annotations

import importlib.util
import hashlib
import json
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


def test_bundled_x64_runtime_has_exact_package_allowlist():
    package = _ROOT / "package" / "windows-x86_64"
    trust = json.loads((package / "runtime-package.json").read_text(encoding="utf-8"))
    assert trust == {
        "manifestVersion": prepare_build.RUNTIME_MANIFEST_VERSION,
        "architecture": "x86_64",
        "generation": "mxc-ca7ea12ac6bd-ca42de9160c4afb6",
        "manifestSha256": "c5bd4f47f22ddb35de582b699e4c76d4e6b9f966a0f7d50d24ea372181fa36c9",
        "runnerSha256": "ca42de9160c4afb601267cfd0bbec42c81b12df13965395c244efd95b95af438",
    }
    assert trust["manifestSha256"] == prepare_build.sha256_file(package / "runtime-manifest.json")
    assert trust["runnerSha256"] == prepare_build.sha256_file(package / "unsloth-mxc-runner.exe")


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


def _built_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)
    runner = artifact / "unsloth-mxc-runner.exe"
    runner.write_bytes(b"reproducible runner")
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    manifest = {
        "manifestVersion": prepare_build.RUNTIME_MANIFEST_VERSION,
        "runtimeVersion": "unsloth-mxc-preview-1",
        "generation": f"mxc-{prepare_build.MXC_REVISION[:12]}-{digest[:16]}",
        "architecture": "x86_64",
        "target": prepare_build.TARGET,
        "protocolVersion": prepare_build.RUNNER_PROTOCOL_VERSION,
        "profileId": prepare_build.PROFILE_ID,
        "schemaVersion": prepare_build.MXC_SCHEMA_VERSION,
        "mxcRepository": prepare_build.MXC_REPOSITORY,
        "mxcRevision": prepare_build.MXC_REVISION,
        "mxcPatchSha256": prepare_build.MXC_PATCH_SHA256,
        "mxcPatchedTree": prepare_build.MXC_PATCHED_TREE,
        "cargoLockSha256": prepare_build.PATCHED_LOCK_SHA256,
        "runnerSourceIdentity": hashlib.sha256(b"source").hexdigest(),
        "features": [prepare_build.FEATURES],
        "rustc": "rustc test",
        "artifacts": {
            "runner": {"path": runner.name, "sha256": digest, "size": runner.stat().st_size}
        },
    }
    (artifact / "runtime-manifest.json").write_bytes(
        prepare_build._canonical_json(manifest) + b"\n"
    )
    return artifact


def test_packaging_emits_digest_allowlist_and_rejects_modified_runner(tmp_path):
    artifact = _built_artifact(tmp_path)
    package = prepare_build.package_artifact(artifact=artifact, destination=tmp_path / "package")
    trust = json.loads((package / "runtime-package.json").read_text(encoding="utf-8"))
    assert trust["runnerSha256"] == prepare_build.sha256_file(package / "unsloth-mxc-runner.exe")
    assert trust["manifestSha256"] == prepare_build.sha256_file(package / "runtime-manifest.json")

    broken = _built_artifact(tmp_path / "broken-root")
    broken.joinpath("unsloth-mxc-runner.exe").write_bytes(b"modified")
    with pytest.raises(prepare_build.PreparationError, match="does not match"):
        prepare_build.package_artifact(artifact=broken, destination=tmp_path / "broken-package")
