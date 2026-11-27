# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from core.inference import mxc_runtime

_WXC = b"official-wxc-test-payload"


_REAL_ARCHITECTURE = mxc_runtime._expected_architecture


def _use_pin(
    monkeypatch,
    approved: dict,
    *,
    base = None,
):
    """Make selected_pin() a pin shaped like the legacy one whose approved bytes are ``approved``.

    ``approved`` maps (architecture, kind) to payload bytes, or to a raw (size, sha256) pair.
    """
    base = base or mxc_runtime.selected_pin()
    artifacts = dict(base.artifacts)
    for key, value in approved.items():
        member = mxc_runtime.LEGACY_PIN.artifacts[key][0]
        size, sha256 = (
            value if isinstance(value, tuple) else (len(value), hashlib.sha256(value).hexdigest())
        )
        artifacts[key] = (member, size, sha256)
    pin = mxc_runtime.MxcPin(
        version = base.version,
        revision = base.revision,
        archive_size = base.archive_size,
        archive_sha256 = base.archive_sha256,
        artifacts = artifacts,
    )
    monkeypatch.setattr(mxc_runtime, "selected_pin", lambda: pin)
    return pin


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Pinned to x64 so an arm64 test host (Apple Silicon CI) checks the same pins.
    monkeypatch.setattr(mxc_runtime, "_expected_architecture", lambda: "x86_64")
    monkeypatch.delenv(mxc_runtime.PIN_ENV, raising = False)
    root = tmp_path / "managed" / "windows-x86_64"
    root.mkdir(parents = True)
    (root / "wxc-exec.exe").write_bytes(_WXC)
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    # A supported build whatever the test host runs (Windows Server 2022 CI is build 20348).
    monkeypatch.setattr(
        mxc_runtime.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(build = mxc_runtime.MIN_WINDOWS_BUILD, product_type = 1),
        raising = False,
    )
    _use_pin(monkeypatch, {("x86_64", "exec"): _WXC}, base = mxc_runtime.LEGACY_PIN)
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


def test_unsupported_architecture_is_rejected(runtime, monkeypatch):
    from core.inference import wsl_host

    monkeypatch.setattr(mxc_runtime, "_expected_architecture", _REAL_ARCHITECTURE)
    monkeypatch.setattr(wsl_host, "native_machine", lambda: "x86")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "x64 and ARM64"):
        mxc_runtime.selected_runtime(package_root = runtime)


def test_arm64_pc_selects_only_the_arm64_build(runtime, monkeypatch, tmp_path):
    from core.inference import wsl_host

    arm64 = b"official-wxc-arm64-payload"
    monkeypatch.setattr(mxc_runtime, "_expected_architecture", _REAL_ARCHITECTURE)
    monkeypatch.setattr(wsl_host, "native_machine", lambda: "arm64")
    _use_pin(monkeypatch, {("arm64", "exec"): arm64})
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    assert mxc_runtime._installed_package_root().name == "windows-arm64"
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "size|digest"):
        mxc_runtime.selected_runtime(package_root = runtime)
    (runtime / "wxc-exec.exe").write_bytes(arm64)
    info = mxc_runtime.selected_runtime(package_root = runtime)
    assert info.architecture == "arm64"
    assert mxc_runtime.artifact("exec")[0] == "package/bin/arm64/wxc-exec.exe"


def test_x64_identity_and_folder_are_unchanged(runtime, monkeypatch, tmp_path):
    # Existing installs and probe caches key on these; an x64 PC must see no difference.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    assert mxc_runtime._installed_package_root().name == "windows-x86_64"
    info = mxc_runtime.selected_runtime(package_root = runtime)
    material = {
        "architecture": "x86_64",
        "mxcRevision": mxc_runtime.LEGACY_PIN.revision,
        "releaseTag": "v1.0.1",
        "sha256": hashlib.sha256(_WXC).hexdigest(),
        "size": len(_WXC),
    }
    encoded = json.dumps(material, sort_keys = True, separators = (",", ":")).encode()
    assert info.identity == hashlib.sha256(encoded).hexdigest()


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


def test_each_pin_names_its_own_microsoft_release():
    for pin, tag in ((mxc_runtime.PRIMARY_PIN, "v1.1.0"), (mxc_runtime.LEGACY_PIN, "v1.0.1")):
        assert pin.tag == tag
        assert pin.url == (f"https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{tag[1:]}.tgz")
    # The identity differs per pin, so a probe verdict for one never answers for the other.
    one = mxc_runtime.RuntimeInfo(Path("w"), "a" * 64, "x86_64", mxc_runtime.PRIMARY_PIN)
    two = mxc_runtime.RuntimeInfo(Path("w"), "a" * 64, "x86_64", mxc_runtime.LEGACY_PIN)
    assert one.identity != two.identity


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


_HOST_PREP = b"official-host-prep-test-payload"


@pytest.fixture
def host_prep(runtime, monkeypatch):
    (runtime / "wxc-host-prep.exe").write_bytes(_HOST_PREP)
    _use_pin(monkeypatch, {("x86_64", "prep"): _HOST_PREP})
    return runtime


def test_host_prep_is_pinned_like_wxc(host_prep):
    with mxc_runtime.acquire_host_prep(package_root = host_prep) as lease:
        assert lease.info.path == (host_prep / "wxc-host-prep.exe").resolve()
    (host_prep / "wxc-host-prep.exe").write_bytes(b"y" * len(_HOST_PREP))
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "wxc-host-prep.exe digest"):
        mxc_runtime.acquire_host_prep(package_root = host_prep)


def test_missing_host_prep_never_affects_the_wxc_runtime(runtime):
    mxc_runtime.selected_runtime(package_root = runtime)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "missing"):
        mxc_runtime.selected_host_prep(package_root = runtime)


_BOTH_WARNINGS = [
    "AppContainer + DACL tier selected: ... Run `wxc-host-prep prepare-system-drive` (elevated)",
    "AppContainer + DACL tier selected: ... Run `wxc-host-prep prepare-null-device` (elevated)",
]


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected"),
    [
        (
            json.dumps({"tier": "appcontainer-dacl", "warnings": _BOTH_WARNINGS}),
            0,
            ("prepare-system-drive", "prepare-null-device"),
        ),
        (
            json.dumps({"tier": "appcontainer-dacl", "warnings": _BOTH_WARNINGS[1:]}),
            0,
            ("prepare-null-device",),
        ),
        (json.dumps({"tier": "base-container", "warnings": []}), 0, ()),
        ("not json", 0, None),
        (json.dumps(["warnings"]), 0, None),
        (json.dumps({"warnings": _BOTH_WARNINGS}), 1, None),
    ],
    ids = ["both", "null_device", "prepared", "garbage", "not_object", "failed"],
)
def test_host_prep_probe_reads_wxc_probe_warnings(
    runtime, monkeypatch, stdout, returncode, expected
):
    seen: dict = {}

    def run(argv, **kwargs):
        seen.update(argv = argv, env = kwargs.get("env"), timeout = kwargs.get("timeout"))
        return subprocess.CompletedProcess(argv, returncode, stdout = stdout, stderr = "")

    monkeypatch.setattr(mxc_runtime.subprocess, "run", run)
    env = {"SYSTEMROOT": "C:\\Windows"}
    assert mxc_runtime.probe_host_prep_steps(package_root = runtime, env = env) == expected
    assert seen["argv"] == [str((runtime / "wxc-exec.exe").resolve()), "--probe"]
    assert seen["env"]["SYSTEMROOT"] == "C:\\Windows"
    # --probe reaps orphaned ACEs, so it must read the journal every launch writes.
    assert seen["env"]["MXC_DACL_STATE_DIR"] == str(mxc_runtime.dacl_state_path())
    assert seen["timeout"] == mxc_runtime.HOST_PREP_PROBE_SECONDS


def test_host_prep_probe_timeout_is_not_a_verdict(runtime, monkeypatch):
    def run(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(mxc_runtime.subprocess, "run", run)
    assert mxc_runtime.probe_host_prep_steps(package_root = runtime) is None


@pytest.mark.parametrize("build", [17763, 20348, 22631])
def test_older_windows_never_launches_wxc(runtime, monkeypatch, tmp_path, build):
    # Server 2019 showed wxc-exec loader dialogs on every Settings > Sandbox visit (#16169).
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        mxc_runtime.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(build = build),
        raising = False,
    )
    monkeypatch.setattr(
        mxc_runtime.subprocess, "run", lambda *_a, **_k: pytest.fail("wxc-exec was launched")
    )
    assert mxc_runtime.selected_runtime(package_root = runtime).path.name == "wxc-exec.exe"
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "26100"):
        mxc_runtime.acquire_runtime(package_root = runtime)
    assert mxc_runtime.probe_host_prep_steps(package_root = runtime) is None
    assert mxc_runtime.recover_dacl_state() is False


def test_windows_11_24h2_still_launches_wxc(runtime, monkeypatch):
    monkeypatch.setattr(
        mxc_runtime.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(build = 26100),
        raising = False,
    )
    with mxc_runtime.acquire_runtime(package_root = runtime) as lease:
        assert lease.info.path == (runtime / "wxc-exec.exe").resolve()


def _count_hashes(monkeypatch):
    calls = []
    real = mxc_runtime._sha256_file
    monkeypatch.setattr(
        mxc_runtime, "_sha256_file", lambda path: calls.append(Path(path).name) or real(path)
    )
    return calls


def _no_lock_guard(monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_require_supported_windows", lambda: None)
    monkeypatch.setattr(mxc_runtime, "_open_artifact_guard", lambda path: Path(path).open("rb"))


def test_a_launch_lease_hashes_the_runtime_once_under_its_guard(runtime, monkeypatch):
    _no_lock_guard(monkeypatch)
    order = []
    real_guard = mxc_runtime._open_artifact_guard
    monkeypatch.setattr(
        mxc_runtime, "_open_artifact_guard", lambda path: order.append("guard") or real_guard(path)
    )
    real = mxc_runtime._sha256_file
    monkeypatch.setattr(
        mxc_runtime, "_sha256_file", lambda path: order.append("hash") or real(path)
    )
    with mxc_runtime.acquire_runtime(package_root = runtime) as lease:
        assert lease.info.sha256 == hashlib.sha256(_WXC).hexdigest()
    # The digest is taken while the deny-write guard is held, and only once.
    assert order == ["guard", "hash"]


def test_a_corrupt_runtime_still_fails_the_lease_after_identity_was_cached(
    runtime, monkeypatch, tmp_path
):
    _no_lock_guard(monkeypatch)
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    (runtime / "wxc-exec.exe").write_bytes(b"x" * len(_WXC))
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.acquire_runtime(package_root = runtime)


def test_installation_identity_rehashes_only_when_the_file_changes(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    mxc_runtime.forget_verified_runtime()
    calls = _count_hashes(monkeypatch)
    first = mxc_runtime.installation_identity()
    for _ in range(5):
        assert mxc_runtime.installation_identity() == first
    assert calls == ["wxc-exec.exe"]

    # Same size, new bytes: the stamp (mtime / file id) changes, so it is verified again and refused.
    path = runtime / "wxc-exec.exe"
    path.unlink()
    path.write_bytes(b"y" * len(_WXC))
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.installation_identity()
    assert len(calls) == 2


def test_installation_identity_cache_keys_on_the_approved_pin(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    # A new pin with the same file on disk must not reuse the old verdict.
    _use_pin(monkeypatch, {("x86_64", "exec"): (len(_WXC), "0" * 64)})
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.installation_identity()


def test_status_and_repair_verify_the_bytes_even_with_a_cached_identity(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    monkeypatch.setattr(mxc_runtime, "_file_stamp", lambda path: ("unchanged",))
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    path = runtime / "wxc-exec.exe"
    with open(path, "r+b") as stream:
        stream.write(b"x")
    calls = _count_hashes(monkeypatch)
    mxc_runtime.installation_identity()
    assert calls == []
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.verified_installation_identity()
    from core.inference import sandbox_setup_plan

    assert sandbox_setup_plan.windows_runtime_installed() is False


@pytest.mark.parametrize("acquire", ["acquire_runtime", "acquire_host_prep"])
def test_an_unsupported_host_is_refused_before_any_artifact_is_opened(
    runtime, monkeypatch, acquire
):
    monkeypatch.setattr(mxc_runtime, "_require_supported_windows", lambda: None)
    monkeypatch.setattr(
        mxc_runtime, "_open_artifact_guard", lambda path: pytest.fail(f"opened {path}")
    )

    def unsupported():
        raise mxc_runtime.MxcRuntimeUnavailable("unsupported architecture")

    monkeypatch.setattr(mxc_runtime, "_expected_architecture", unsupported)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "unsupported architecture"):
        getattr(mxc_runtime, acquire)(package_root = runtime)


def test_a_failed_reverification_drops_the_old_verdict(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    path = runtime / "wxc-exec.exe"
    good = path.read_bytes()
    stamp = path.stat()
    path.write_bytes(b"x" * len(good))
    os.utime(path, ns = (stamp.st_atime_ns, stamp.st_mtime_ns + 10**9))
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.installation_identity()
    assert mxc_runtime._identity_cache == {}


def test_a_backend_refresh_forgets_the_runtime_identity(runtime, monkeypatch):
    from core.inference import mxc_probe

    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    assert mxc_runtime._identity_cache
    mxc_probe.invalidate_cache()
    assert mxc_runtime._identity_cache == {}


def test_a_failed_full_verification_also_drops_the_cached_identity(runtime, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: runtime)
    monkeypatch.setattr(mxc_runtime, "_file_stamp", lambda path: ("unchanged",))
    mxc_runtime.forget_verified_runtime()
    mxc_runtime.installation_identity()
    with open(runtime / "wxc-exec.exe", "r+b") as stream:
        stream.write(b"x")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable):
        mxc_runtime.verified_installation_identity()
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "digest"):
        mxc_runtime.installation_identity()


def test_a_path_redirected_while_the_guard_opens_is_refused(runtime, monkeypatch):
    _no_lock_guard(monkeypatch)
    moved = {}
    real_resolve = Path.resolve

    def resolve(self, *args, **kwargs):
        target = real_resolve(self, *args, **kwargs)
        return moved.get(str(target), target)

    def guard(path):
        moved[str(real_resolve(path))] = real_resolve(path).with_name("elsewhere.exe")
        return Path(path).open("rb")

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(mxc_runtime, "_open_artifact_guard", guard)
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match = "changed during acquisition"):
        mxc_runtime.acquire_runtime(package_root = runtime)
