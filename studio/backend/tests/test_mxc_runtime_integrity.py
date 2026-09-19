# SPDX-License-Identifier: AGPL-3.0-only
"""Adversarial runtime-manifest and immutable-generation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import mxc_runtime


def _write_json(path: Path, value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _artifact(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str, str]:
    root = tmp_path / name
    root.mkdir()
    runner = root / "unsloth-mxc-runner.exe"
    runner.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    generation = f"mxc-{mxc_runtime.MXC_REVISION[:12]}-{digest[:16]}"
    manifest = {
        "manifestVersion": 1,
        "runtimeVersion": mxc_runtime.RUNTIME_VERSION,
        "generation": generation,
        "architecture": "x86_64",
        "target": mxc_runtime.EXPECTED_TARGET,
        "protocolVersion": mxc_runtime.RUNNER_PROTOCOL_VERSION,
        "profileId": mxc_runtime.PROFILE_ID,
        "schemaVersion": mxc_runtime.MXC_SCHEMA_VERSION,
        "mxcRepository": "https://github.com/microsoft/mxc.git",
        "mxcRevision": mxc_runtime.MXC_REVISION,
        "mxcPatchSha256": mxc_runtime.MXC_PATCH_SHA256,
        "mxcPatchedTree": mxc_runtime.MXC_PATCHED_TREE,
        "cargoLockSha256": mxc_runtime.PATCHED_CARGO_LOCK_SHA256,
        "runnerSourceIdentity": hashlib.sha256(b"runner source").hexdigest(),
        "features": ["mxc-no-dacl-api"],
        "rustc": "rustc test toolchain",
        "artifacts": {
            "runner": {
                "path": "unsloth-mxc-runner.exe",
                "sha256": digest,
                "size": len(payload),
            }
        },
    }
    manifest_digest = _write_json(root / "runtime-manifest.json", manifest)
    return root, manifest_digest, digest


@pytest.fixture(autouse=True)
def _runtime_test_state(monkeypatch):
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(mxc_runtime, "_validate_identity", lambda *_args: None)
    mxc_runtime._owners.clear()
    mxc_runtime._retire_pending.clear()
    yield
    mxc_runtime._owners.clear()
    mxc_runtime._retire_pending.clear()


def _install(artifact: tuple[Path, str, str], root: Path):
    directory, manifest_digest, runner_digest = artifact
    return mxc_runtime.install_generation(
        directory,
        approved_manifest_sha256=manifest_digest,
        approved_runner_sha256=runner_digest,
        root=root,
    )


def test_runner_presence_without_trust_manifest_is_not_production_ready(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "unsloth-mxc-runner.exe").write_bytes(b"exists")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="not installed"):
        mxc_runtime.selected_runtime(root=root)


def test_wrong_runner_digest_and_truncation_are_rejected(tmp_path):
    artifact = _artifact(tmp_path, "artifact", b"approved runner")
    directory, manifest_digest, runner_digest = artifact
    (directory / "unsloth-mxc-runner.exe").write_bytes(b"x")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="size|digest"):
        mxc_runtime._validate_runtime_directory(
            directory,
            trust={
                "manifestSha256": manifest_digest,
                "runnerSha256": runner_digest,
            },
            development=False,
        )


def test_manifest_modification_is_rejected_by_trusted_allowlist(tmp_path):
    directory, manifest_digest, runner_digest = _artifact(tmp_path, "artifact", b"runner")
    manifest_path = directory / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rustc"] = "modified after approval"
    _write_json(manifest_path, manifest)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="allowlist"):
        mxc_runtime._validate_runtime_directory(
            directory,
            trust={
                "manifestSha256": manifest_digest,
                "runnerSha256": runner_digest,
            },
            development=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("architecture", "aarch64", "pinned profile"),
        ("protocolVersion", 99, "pinned profile"),
        ("profileId", "other-profile", "pinned profile"),
    ],
)
def test_wrong_architecture_protocol_or_profile_is_rejected(tmp_path, field, value, message):
    directory, _, digest = _artifact(tmp_path, field, b"runner")
    manifest_path = directory / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_digest = _write_json(manifest_path, manifest)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match=message):
        mxc_runtime._validate_runtime_directory(
            directory,
            trust={"manifestSha256": manifest_digest, "runnerSha256": digest},
            development=False,
        )


def test_unextended_runner_identity_reports_precise_unavailability(tmp_path, monkeypatch):
    directory, _, _ = _artifact(tmp_path, "artifact", b"runner")
    monkeypatch.undo()
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    identity = {
        "protocolVersion": 1,
        "profileId": mxc_runtime.PROFILE_ID,
        "profileVersion": 1,
        "schemaVersion": mxc_runtime.MXC_SCHEMA_VERSION,
        "mxcRevision": mxc_runtime.MXC_REVISION,
        "mxcPatchSha256": "unverified",
        "mxcPatchedTree": "unverified",
        "runnerSourceIdentity": "unverified",
        "admissionApi": False,
        "architecture": "x86_64",
    }
    monkeypatch.setattr(
        mxc_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(identity), stderr=""
        ),
    )
    manifest = json.loads((directory / "runtime-manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="required Studio admission API"):
        mxc_runtime._validate_identity(directory / "unsloth-mxc-runner.exe", manifest)


def test_runtime_generation_replacement_does_not_revoke_active_owner(tmp_path):
    root = tmp_path / "installed"
    first = _install(_artifact(tmp_path, "first", b"runner A"), root)
    lease = mxc_runtime.acquire_runtime(root=root)
    second = _install(_artifact(tmp_path, "second", b"runner B"), root)
    assert lease.info.generation == first.generation
    assert mxc_runtime.selected_runtime(root=root).generation == second.generation
    assert lease.info.path.is_file()
    assert mxc_runtime.retire_generation(first.generation, root=root) is False
    assert lease.info.path.is_file()
    lease.release()
    assert not lease.info.path.parent.exists()


def test_concurrent_repair_waits_for_all_generation_owners(tmp_path):
    root = tmp_path / "installed"
    first = _install(_artifact(tmp_path, "first", b"runner A"), root)
    lease_a = mxc_runtime.acquire_runtime(root=root)
    lease_b = mxc_runtime.acquire_runtime(root=root)
    _install(_artifact(tmp_path, "second", b"runner B"), root)
    assert mxc_runtime.retire_generation(first.generation, root=root) is False
    lease_a.release()
    assert first.path.parent.exists()
    lease_b.release()
    assert not first.path.parent.exists()


def test_failed_partial_install_never_becomes_selectable(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    first = _install(_artifact(tmp_path, "first", b"runner A"), root)
    second = _artifact(tmp_path, "second", b"runner B")
    original = mxc_runtime.shutil.copy2
    calls = 0

    def fail_second_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected partial copy")
        return original(*args, **kwargs)

    monkeypatch.setattr(mxc_runtime.shutil, "copy2", fail_second_copy)
    with pytest.raises(OSError, match="partial copy"):
        _install(second, root)
    assert mxc_runtime.selected_runtime(root=root).generation == first.generation
    assert not list((root / "generations").glob("*.tmp"))


def test_malformed_runtime_manifest_is_rejected(tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "runtime-manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="malformed"):
        mxc_runtime._validate_runtime_directory(directory, trust=None, development=True)
