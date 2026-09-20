# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.inference import mxc_runtime

_WXC = b"official-wxc-test-payload"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "managed" / "windows-x86_64"
    root.mkdir(parents = True)
    (root / "wxc-exec.exe").write_bytes(_WXC)
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(mxc_runtime, "WXC_EXEC_SIZE", len(_WXC))
    monkeypatch.setattr(mxc_runtime, "WXC_EXEC_SHA256", hashlib.sha256(_WXC).hexdigest())
    return root


def test_only_the_fixed_managed_wxc_is_selected(runtime, monkeypatch, tmp_path):
    path_runner = tmp_path / "bin" / "wxc-exec.exe"
    path_runner.parent.mkdir()
    path_runner.write_bytes(b"untrusted")
    monkeypatch.setenv("PATH", str(path_runner.parent))
    info = mxc_runtime.selected_runtime(package_root = runtime)
    assert info.path == (runtime / "wxc-exec.exe").resolve()
    assert info.sha256 == hashlib.sha256(_WXC).hexdigest()


def test_cargo_target_wxc_is_ignored(runtime, monkeypatch, tmp_path):
    cargo_runner = tmp_path / "target" / "release" / "wxc-exec.exe"
    cargo_runner.parent.mkdir(parents = True)
    cargo_runner.write_bytes(_WXC)
    monkeypatch.chdir(cargo_runner.parents[2])
    assert (
        mxc_runtime.selected_runtime(package_root = runtime).path
        == (runtime / "wxc-exec.exe").resolve()
    )


@pytest.mark.parametrize("mutation", [b"modified", b""])
def test_modified_or_truncated_wxc_is_rejected(runtime, mutation):
    (runtime / "wxc-exec.exe").write_bytes(mutation)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "size|digest"):
        mxc_runtime.selected_runtime(package_root = runtime)


def test_missing_wxc_is_rejected(runtime):
    (runtime / "wxc-exec.exe").unlink()
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "missing"):
        mxc_runtime.selected_runtime(package_root = runtime)


def test_wrong_architecture_is_rejected(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime.platform, "machine", lambda: "ARM64")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "x86-64"):
        mxc_runtime.selected_runtime(package_root = runtime)


def test_runtime_lease_holds_and_revalidates_the_wxc_file(runtime, monkeypatch):
    opened = []

    class Guard:
        def close(self):
            opened.append("closed")

    monkeypatch.setattr(
        mxc_runtime,
        "_open_artifact_guard",
        lambda path: opened.append(Path(path).name) or Guard(),
    )
    lease = mxc_runtime.acquire_runtime(package_root = runtime)
    assert opened == ["wxc-exec.exe"]
    assert lease.info.path == (runtime / "wxc-exec.exe").resolve()
    lease.release()
    assert opened[-1] == "closed"


def test_reparse_runtime_directory_is_rejected(runtime, tmp_path):
    link = tmp_path / "runtime-link"
    try:
        link.symlink_to(runtime, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "non-reparse"):
        mxc_runtime.selected_runtime(package_root = link)


def test_release_identity_is_fixed_to_microsoft_v080():
    assert mxc_runtime.RELEASE_URL == (
        "https://github.com/microsoft/mxc/releases/download/v0.8.0/mxc-release-binaries.zip"
    )
    assert mxc_runtime.MXC_REVISION == "7dac1a952f0c9ad13f0a4cb089c4e0e8b3e0013a"
    assert mxc_runtime.RELEASE_ARCHIVE_SIZE == 358_007_638
    assert mxc_runtime.RELEASE_ARCHIVE_SHA256 == (
        "5c3a27073ba18eddf97efb4caad0f8b201c40a18d17b70f3a1e3847fb6232e3c"
    )
    assert mxc_runtime.WXC_EXEC_SIZE == 9_478_968
    assert mxc_runtime.WXC_EXEC_SHA256 == (
        "6049c64723af1173c3739dc6cd6b2f33f6c021bb2832c4216233cba7f71aee9a"
    )


def test_corrupt_managed_wxc_is_never_executed(runtime, monkeypatch):
    from core.inference import mxc_adapter, mxc_policy

    (runtime / "wxc-exec.exe").write_bytes(b"corrupt")
    acquire_runtime = mxc_runtime.acquire_runtime
    monkeypatch.setattr(
        mxc_runtime,
        "acquire_runtime",
        lambda: acquire_runtime(package_root = runtime),
    )
    monkeypatch.setattr(
        mxc_adapter.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("corrupt WXC was executed"),
    )
    config = {
        "fallback": {"allowDaclMutation": False},
        "ui": {"disable": False},
    }
    request = {
        "config": config,
        "configBytes": mxc_policy.canonical_config_bytes(config),
        "policyHash": mxc_policy.compute_policy_hash(config),
    }
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "size|digest"):
        mxc_adapter.spawn(request)
