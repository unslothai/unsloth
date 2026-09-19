# SPDX-License-Identifier: AGPL-3.0-only
"""Install, repair, update, rollback, retirement, and uninstall invariants."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import pytest

from core.inference import mxc_runtime


def _write_json(path: Path, value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _package(tmp_path: Path, name: str, payload: bytes) -> Path:
    root = tmp_path / name
    root.mkdir()
    runner = root / "unsloth-mxc-runner.exe"
    runner.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    generation = f"mxc-{mxc_runtime.MXC_REVISION[:12]}-{digest[:16]}"
    manifest = {
        "manifestVersion": mxc_runtime.RUNTIME_MANIFEST_VERSION,
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
        "runnerSourceIdentity": hashlib.sha256(b"source-" + payload).hexdigest(),
        "features": list(mxc_runtime.EXPECTED_FEATURES),
        "rustc": "rustc lifecycle test",
        "artifacts": {"runner": {"path": runner.name, "sha256": digest, "size": len(payload)}},
    }
    manifest_digest = _write_json(root / "runtime-manifest.json", manifest)
    _write_json(
        root / "runtime-package.json",
        {
            "manifestVersion": mxc_runtime.RUNTIME_MANIFEST_VERSION,
            "architecture": "x86_64",
            "generation": generation,
            "manifestSha256": manifest_digest,
            "runnerSha256": digest,
        },
    )
    return root


@pytest.fixture(autouse=True)
def _runtime_state(monkeypatch):
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(mxc_runtime, "_validate_identity", lambda *_args: None)
    mxc_runtime._owners.clear()
    mxc_runtime._retire_pending.clear()
    yield
    mxc_runtime._owners.clear()
    mxc_runtime._retire_pending.clear()


def test_clean_install_publishes_only_fully_verified_generation(tmp_path):
    package = _package(tmp_path, "package-a", b"runner-a")
    root = tmp_path / "installed"
    info = mxc_runtime.install_approved_runtime(package_root=package, root=root)
    assert info.production_ready and not info.development
    assert mxc_runtime.selected_runtime(root=root) == info
    assert json.loads((root / "current.json").read_text())["generation"] == info.generation
    assert (
        mxc_runtime.runtime_status(package_root=package, root=root).state
        is mxc_runtime.RuntimeState.READY
    )
    assert not list((root / "generations").glob("*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    ["flip_runner", "truncate_runner", "delete_runner", "manifest", "current", "trust"],
)
def test_corruption_is_never_selected_and_repair_restages_clean_bytes(tmp_path, mutation):
    package = _package(tmp_path, "package-a", b"approved-runner")
    root = tmp_path / "installed"
    info = mxc_runtime.install_approved_runtime(package_root=package, root=root)
    installed = info.path.parent
    if mutation == "flip_runner":
        installed.joinpath("unsloth-mxc-runner.exe").write_bytes(b"approved-runneX")
    elif mutation == "truncate_runner":
        installed.joinpath("unsloth-mxc-runner.exe").write_bytes(b"x")
    elif mutation == "delete_runner":
        installed.joinpath("unsloth-mxc-runner.exe").unlink()
    elif mutation == "manifest":
        installed.joinpath("runtime-manifest.json").write_text("{}")
    elif mutation == "current":
        root.joinpath("current.json").write_text("{")
    else:
        root.joinpath("runtime-trust.json").write_text("{")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable):
        mxc_runtime.selected_runtime(root=root)
    repaired = mxc_runtime.repair_runtime(package_root=package, root=root)
    assert repaired.runner_sha256 == hashlib.sha256(b"approved-runner").hexdigest()
    assert (
        mxc_runtime.runtime_status(package_root=package, root=root).state
        is mxc_runtime.RuntimeState.READY
    )


def test_failed_update_before_pointer_publication_keeps_current(tmp_path):
    package_a = _package(tmp_path, "package-a", b"runner-a")
    package_b = _package(tmp_path, "package-b", b"runner-b")
    failed_generation = json.loads(
        (package_b / "runtime-package.json").read_text(encoding="utf-8")
    )["generation"]
    root = tmp_path / "installed"
    first = mxc_runtime.install_approved_runtime(package_root=package_a, root=root)

    def fail(stage):
        if stage == "before_pointer_publication":
            raise OSError("injected publication failure")

    with pytest.raises(OSError, match="publication"):
        mxc_runtime.install_approved_runtime(
            package_root=package_b,
            root=root,
            operation="update",
            failure_hook=fail,
        )
    assert mxc_runtime.selected_runtime(root=root).generation == first.generation
    assert not (root / "generations" / failed_generation).exists()
    trust = json.loads((root / "runtime-trust.json").read_text(encoding="utf-8"))
    assert failed_generation not in trust["generations"]


def test_update_and_rollback_do_not_revoke_active_generation(tmp_path):
    package_a = _package(tmp_path, "package-a", b"runner-a")
    package_b = _package(tmp_path, "package-b", b"runner-b")
    root = tmp_path / "installed"
    first = mxc_runtime.install_approved_runtime(package_root=package_a, root=root)
    lease_a = mxc_runtime.acquire_runtime(root=root)
    second = mxc_runtime.update_runtime(package_root=package_b, root=root)
    lease_b = mxc_runtime.acquire_runtime(root=root)
    assert lease_a.info.generation == first.generation
    assert lease_b.info.generation == second.generation
    rolled_back = mxc_runtime.rollback_runtime(root=root)
    assert rolled_back.generation == first.generation
    assert lease_b.info.path.is_file()
    lease_a.release()
    lease_b.release()


def test_gc_keeps_current_previous_and_active_then_retires_old(tmp_path):
    packages = [_package(tmp_path, f"package-{i}", f"runner-{i}".encode()) for i in range(3)]
    root = tmp_path / "installed"
    first = mxc_runtime.install_approved_runtime(package_root=packages[0], root=root)
    lease = mxc_runtime.acquire_runtime(root=root)
    second = mxc_runtime.update_runtime(package_root=packages[1], root=root)
    third = mxc_runtime.update_runtime(package_root=packages[2], root=root)
    assert mxc_runtime.garbage_collect(root=root) == []
    lease.release()
    assert mxc_runtime.garbage_collect(root=root) == [first.generation]
    assert second.path.parent.is_dir()
    assert third.path.parent.is_dir()


def test_uninstall_disables_new_launches_and_waits_for_active_lease(tmp_path):
    package = _package(tmp_path, "package-a", b"runner-a")
    root = tmp_path / "installed"
    mxc_runtime.install_approved_runtime(package_root=package, root=root)
    lease = mxc_runtime.acquire_runtime(root=root)
    assert mxc_runtime.uninstall_runtime(root=root) is False
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable):
        mxc_runtime.selected_runtime(root=root)
    assert lease.info.path.is_file()
    lease.release()
    assert mxc_runtime.uninstall_runtime(root=root) is True
    assert not (root / "generations").exists()


def test_simultaneous_repairs_are_serialized_and_idempotent(tmp_path, monkeypatch):
    package = _package(tmp_path, "package-a", b"runner-a")
    root = tmp_path / "installed"
    active = 0
    peak = 0
    real_copy = mxc_runtime.shutil.copy2

    def observed_copy(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.01)
        try:
            return real_copy(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(mxc_runtime.shutil, "copy2", observed_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: mxc_runtime.repair_runtime(package_root=package, root=root),
                range(2),
            )
        )
    assert results[0].generation == results[1].generation
    assert peak == 1


def test_unlisted_runner_path_and_partial_generation_are_ignored(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    root.mkdir()
    fake_path = tmp_path / "path-bin"
    fake_path.mkdir()
    fake_path.joinpath("unsloth-mxc-runner.exe").write_bytes(b"fake")
    monkeypatch.setenv("PATH", str(fake_path))
    target = tmp_path / "target" / "release"
    target.mkdir(parents=True)
    target.joinpath("unsloth-mxc-runner.exe").write_bytes(b"fake")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="not installed"):
        mxc_runtime.selected_runtime(root=root)


def test_pointer_traversal_and_package_architecture_are_refused(tmp_path, monkeypatch):
    package = _package(tmp_path, "package-a", b"runner-a")
    root = tmp_path / "installed"
    mxc_runtime.install_approved_runtime(package_root=package, root=root)
    _write_json(root / "current.json", {"manifestVersion": 1, "generation": "../escape"})
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="invalid|malformed"):
        mxc_runtime.selected_runtime(root=root)
    trust = json.loads((package / "runtime-package.json").read_text())
    trust["architecture"] = "arm64"
    _write_json(package / "runtime-package.json", trust)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="architecture"):
        mxc_runtime.install_approved_runtime(package_root=package, root=tmp_path / "other")


def test_reparse_package_or_generation_is_refused(tmp_path):
    package = _package(tmp_path, "package-a", b"runner-a")
    package_link = tmp_path / "package-link"
    try:
        package_link.symlink_to(package, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="non-reparse"):
        mxc_runtime.install_approved_runtime(package_root=package_link, root=tmp_path / "installed")
