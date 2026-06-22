# SPDX-License-Identifier: AGPL-3.0-only
# Logic tests for studio/install_node_prebuilt.py -- the isolated Node installer.
# No network/GPU: downloads are monkeypatched and archives are built in-memory.

import importlib.util
import io
import json
import os
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_node_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_node_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

HostInfo = M.HostInfo
PrebuiltFallback = M.PrebuiltFallback


def _host(node_os: str, node_arch: str) -> HostInfo:
    ext = ".zip" if node_os == "win" else ".tar.gz"
    return HostInfo(
        system = {"linux": "Linux", "darwin": "Darwin", "win": "Windows"}[node_os],
        machine = node_arch,
        node_os = node_os,
        node_arch = node_arch,
        archive_ext = ext,
        is_windows = node_os == "win",
    )


# ── Host detection (per OS/arch) ──
@pytest.mark.parametrize(
    "system,machine,exp_os,exp_arch,exp_ext",
    [
        ("Linux", "x86_64", "linux", "x64", ".tar.gz"),
        ("Linux", "aarch64", "linux", "arm64", ".tar.gz"),
        ("Darwin", "x86_64", "darwin", "x64", ".tar.gz"),
        ("Darwin", "arm64", "darwin", "arm64", ".tar.gz"),
        ("Windows", "AMD64", "win", "x64", ".zip"),
        ("Windows", "ARM64", "win", "arm64", ".zip"),
    ],
)
def test_detect_host(monkeypatch, system, machine, exp_os, exp_arch, exp_ext):
    monkeypatch.setattr(M.platform, "system", lambda: system)
    monkeypatch.setattr(M.platform, "machine", lambda: machine)
    host = M.detect_host()
    assert (host.node_os, host.node_arch, host.archive_ext) == (exp_os, exp_arch, exp_ext)
    assert host.is_windows == (exp_os == "win")


@pytest.mark.parametrize(
    "system,machine",
    [("Plan9", "x86_64"), ("Linux", "sparc64"), ("Linux", "armv7l"), ("Linux", "armhf")],
)
def test_detect_host_unsupported(monkeypatch, system, machine):
    monkeypatch.setattr(M.platform, "system", lambda: system)
    monkeypatch.setattr(M.platform, "machine", lambda: machine)
    with pytest.raises(PrebuiltFallback):
        M.detect_host()


# ── URL / asset construction (pure) ──
def test_asset_and_url_linux():
    host = _host("linux", "x64")
    assert M.node_asset_name("24.17.0", host) == "node-v24.17.0-linux-x64.tar.gz"
    assert (
        M.node_download_url("24.17.0", M.node_asset_name("24.17.0", host))
        == "https://nodejs.org/dist/v24.17.0/node-v24.17.0-linux-x64.tar.gz"
    )


def test_asset_windows_is_zip():
    host = _host("win", "x64")
    assert M.node_asset_name("24.17.0", host) == "node-v24.17.0-win-x64.zip"


def test_shasums_url():
    assert M.node_shasums_url("24.17.0") == "https://nodejs.org/dist/v24.17.0/SHASUMS256.txt"


def test_binary_layout_is_host_aware():
    # Windows ships node.exe + node_modules\npm at the root; Unix uses bin/ + lib/.
    win = _host("win", "x64")
    nix = _host("linux", "x64")
    assert M.node_binary_path(Path("/n"), win) == Path("/n/node.exe")
    assert M.node_binary_path(Path("/n"), nix) == Path("/n/bin/node")
    assert M.npm_cli_path(Path("/n"), win) == Path("/n/node_modules/npm/bin/npm-cli.js")
    assert M.npm_cli_path(Path("/n"), nix) == Path("/n/lib/node_modules/npm/bin/npm-cli.js")


# ── SHASUMS256.txt parsing ──
def test_expected_sha256_for():
    asset = "node-v24.17.0-linux-x64.tar.gz"
    good = "a" * 64
    text = (
        f"{'b' * 64}  node-v24.17.0-linux-arm64.tar.gz\n"
        f"{good}  {asset}\n"
        f"{'c' * 64}  node-v24.17.0-win-x64.zip\n"
    )
    assert M.expected_sha256_for(text, asset) == good
    assert M.expected_sha256_for(text, "node-v24.17.0-darwin-x64.tar.gz") is None


def test_expected_sha256_rejects_malformed():
    asset = "node-v24.17.0-linux-x64.tar.gz"
    assert M.expected_sha256_for(f"notahex  {asset}\n", asset) is None


# ── Version selection from index.json ──
INDEX = [
    {"version": "v26.3.1", "lts": False},
    {"version": "v24.17.0", "lts": "Krypton"},
    {"version": "v24.9.0", "lts": "Krypton"},
    {"version": "v22.20.0", "lts": "Jod"},
    {"version": "v20.19.0", "lts": "Iron"},
]


def test_select_lts_respects_min_major():
    # Newest LTS at/above 24 -> 24.17.0 (22.x LTS is below the floor).
    assert M.select_node_version(INDEX, channel = "lts", min_major = 24) == "24.17.0"


def test_select_latest_overall():
    assert M.select_node_version(INDEX, channel = "latest", min_major = 24) == "26.3.1"


def test_select_explicit_passthrough():
    assert M.select_node_version(INDEX, channel = "v24.5.0", min_major = 24) == "24.5.0"


def test_select_no_candidate_raises():
    with pytest.raises(PrebuiltFallback):
        M.select_node_version(INDEX, channel = "lts", min_major = 99)


# ── Archive extraction (zip + tar.gz with the npm-style symlink), traversal guard ──
def _add_file(
    tar: tarfile.TarFile,
    name: str,
    data: bytes,
    mode: int = 0o644,
):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tar.addfile(info, io.BytesIO(data))


def _add_symlink(tar: tarfile.TarFile, name: str, target: str):
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tar.addfile(info)


@pytest.mark.skipif(
    os.name == "nt",
    reason = "Node ships a .zip (no symlinks) on Windows; the tar+symlink path is Unix-only",
)
def test_extract_tar_gz_with_npm_symlink(tmp_path: Path):
    # Mirrors the real Node tarball: bin/npm -> ../lib/node_modules/npm/bin/npm-cli.js
    archive = tmp_path / "node.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, "node-v24/bin/node", b"#!/bin/sh\necho v24.17.0\n", mode = 0o755)
        _add_file(tar, "node-v24/lib/node_modules/npm/bin/npm-cli.js", b"// npm")
        _add_symlink(tar, "node-v24/bin/npm", "../lib/node_modules/npm/bin/npm-cli.js")

    dest = tmp_path / "out"
    M.extract_archive(archive, dest)
    npm_link = dest / "node-v24" / "bin" / "npm"
    assert npm_link.is_symlink()
    assert (dest / "node-v24" / "bin" / "node").exists()
    # executable bit preserved
    assert (dest / "node-v24" / "bin" / "node").stat().st_mode & 0o111


def test_extract_zip(tmp_path: Path):
    archive = tmp_path / "node.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("node-v24-win-x64/node.exe", b"MZ")
        zf.writestr("node-v24-win-x64/npm.cmd", b"@echo off")
    dest = tmp_path / "out"
    M.extract_archive(archive, dest)
    assert (dest / "node-v24-win-x64" / "node.exe").exists()


def test_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, "../escape.txt", b"pwn")
    with pytest.raises(PrebuiltFallback):
        M.extract_archive(archive, tmp_path / "out")


# ── Checksum-verified download (accept + reject) ──
def test_download_file_verified_accepts_match(tmp_path: Path, monkeypatch):
    payload = b"real-node-archive"
    sha = M.hashlib.sha256(payload).hexdigest()

    def fake_download(url: str, destination: Path):
        destination.write_bytes(payload)

    monkeypatch.setattr(M, "download_file", fake_download)
    dest = tmp_path / "a.tar.gz"
    M.download_file_verified("http://x/a.tar.gz", dest, expected_sha256 = sha, label = "a")
    assert dest.read_bytes() == payload


def test_download_file_verified_rejects_mismatch(tmp_path: Path, monkeypatch):
    def fake_download(url: str, destination: Path):
        destination.write_bytes(b"tampered")

    monkeypatch.setattr(M, "download_file", fake_download)
    with pytest.raises(PrebuiltFallback):
        M.download_file_verified("http://x/a", tmp_path / "a", expected_sha256 = "0" * 64, label = "a")


# ── Lock liveness probe (Windows must not use os.kill(pid, 0)) ──
def test_pid_is_alive_windows_uses_tasklist_not_os_kill(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "win32")

    def fail_kill(pid, sig):
        raise AssertionError("Windows liveness must not call os.kill(pid, 0)")

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["tasklist", "/FI"]
        assert "PID eq 1234" in cmd
        return types.SimpleNamespace(stdout = '"node.exe","1234","Console","1","12,345 K"\n')

    monkeypatch.setattr(M.os, "kill", fail_kill)
    monkeypatch.setattr(M.subprocess, "run", fake_run)
    assert M._pid_is_alive(1234) is True


def test_pid_is_alive_windows_false_when_tasklist_omits_pid(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(
        M.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            stdout = "INFO: No tasks are running which match the specified criteria.\n"
        ),
    )
    assert M._pid_is_alive(1234) is False


def test_pid_is_alive_windows_assumes_alive_when_tasklist_fails(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "win32")

    def boom(*args, **kwargs):
        raise OSError("tasklist unavailable")

    monkeypatch.setattr(M.subprocess, "run", boom)
    assert M._pid_is_alive(1234) is True


def test_pid_is_alive_posix_signal_zero(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "linux")
    calls = []

    def fake_kill(pid, sig):
        calls.append((pid, sig))
        if pid == 9999:
            raise ProcessLookupError

    monkeypatch.setattr(M.os, "kill", fake_kill)
    assert M._pid_is_alive(1234) is True
    assert M._pid_is_alive(9999) is False
    assert calls == [(1234, 0), (9999, 0)]


# ── existing_install_matches + install_prebuilt short-circuit ──
def test_existing_install_matches_false_without_metadata(tmp_path: Path):
    host = _host("linux", "x64")
    assert M.existing_install_matches(tmp_path, host, version = "24.17.0") is False


def test_existing_install_matches_true_when_version_and_runtime_ok(tmp_path: Path, monkeypatch):
    host = _host("linux", "x64")
    M.write_metadata(tmp_path, version = "24.17.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.17.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    assert M.existing_install_matches(tmp_path, host, version = "24.17.0") is True
    # npm too old -> not a match
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 10)
    assert M.existing_install_matches(tmp_path, host, version = "24.17.0") is False


def test_install_prebuilt_short_circuits_when_version_matches(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    M.write_metadata(install_dir, version = "24.17.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "fetch_json", lambda url: INDEX)
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.17.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)

    def boom(*a, **k):
        raise AssertionError("must not download when the install already matches")

    monkeypatch.setattr(M, "download_file", boom)
    monkeypatch.setattr(M, "download_bytes", boom)

    rc = M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = False)
    assert rc == M.EXIT_SUCCESS


def test_existing_install_usable_is_version_agnostic(tmp_path: Path, monkeypatch):
    host = _host("linux", "x64")
    assert M.existing_install_usable(tmp_path, host) is False  # no metadata
    M.write_metadata(tmp_path, version = "24.17.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.17.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    assert M.existing_install_usable(tmp_path, host) is True
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 10)
    assert M.existing_install_usable(tmp_path, host) is False  # npm below floor
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: None)
    assert M.existing_install_usable(tmp_path, host) is False  # node does not run


def _offline(*a, **k):
    raise OSError("nodejs.org unreachable")


def test_install_prebuilt_keeps_existing_when_index_unreachable(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    M.write_metadata(install_dir, version = "24.17.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.17.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    monkeypatch.setattr(M, "fetch_json", _offline)

    def boom(*a, **k):
        raise AssertionError("must not download when keeping the existing install")

    monkeypatch.setattr(M, "download_file", boom)
    monkeypatch.setattr(M, "download_bytes", boom)

    rc = M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = False)
    assert rc == M.EXIT_SUCCESS


def test_install_prebuilt_reraises_when_index_unreachable_and_no_install(
    tmp_path: Path, monkeypatch
):
    install_dir = tmp_path / "node"  # nothing on disk to fall back to
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "fetch_json", _offline)
    with pytest.raises(OSError):
        M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = False)


def test_install_prebuilt_force_does_not_keep_existing_offline(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    M.write_metadata(install_dir, version = "24.17.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.17.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    monkeypatch.setattr(M, "fetch_json", _offline)
    with pytest.raises(OSError):
        M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = True)


@pytest.mark.parametrize(
    "ver,ok",
    [
        ("20.19.0", True),
        ("20.18.9", False),
        ("22.12.0", True),
        ("22.11.5", False),
        ("23.0.0", True),
        ("24.4.1", True),
        ("21.7.3", False),
        ("24", True),
        ("20", False),
    ],
)
def test_meets_node_floor(ver, ok):
    assert M._meets_node_floor(ver) is ok


def test_install_prebuilt_rejects_explicit_below_floor(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "node"
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))

    def boom(*a, **k):
        raise AssertionError("must not download a below-floor Node")

    monkeypatch.setattr(M, "download_file", boom)
    monkeypatch.setattr(M, "download_bytes", boom)
    with pytest.raises(PrebuiltFallback):
        M.install_prebuilt(install_dir, channel = "20.18.0", min_major = 24, force = False)


def test_install_prebuilt_keeps_existing_when_shasums_fetch_fails(tmp_path: Path, monkeypatch):
    # index.json resolves a newer version, but the later SHASUMS fetch fails and a
    # usable older isolated Node is on disk -> keep it instead of aborting.
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    M.write_metadata(install_dir, version = "24.9.0", asset = "x", sha256 = "y")
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "fetch_json", lambda url: INDEX)  # newest LTS = 24.17.0
    monkeypatch.setattr(M, "installed_node_version", lambda d, h: "24.9.0")
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 11)
    monkeypatch.setattr(M, "download_bytes", _offline)  # SHASUMS fetch fails
    rc = M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = False)
    assert rc == M.EXIT_SUCCESS


def test_install_prebuilt_reraises_shasums_failure_without_existing(tmp_path: Path, monkeypatch):
    install_dir = tmp_path / "node"  # nothing usable on disk
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(M, "fetch_json", lambda url: INDEX)
    monkeypatch.setattr(M, "download_bytes", _offline)
    with pytest.raises(OSError):
        M.install_prebuilt(install_dir, channel = "lts", min_major = 24, force = False)


# ── Isolation invariant: the installer only writes inside its own install_dir ──
def test_run_node_pins_npm_prefix_to_install_dir(tmp_path: Path, monkeypatch):
    # Every node/npm call the installer makes redirects npm's global prefix into
    # the isolated install_dir and drops an inherited NODE_PATH, so a stray `npm
    # -g` can never write to the user's system Node/npm.
    install_dir = tmp_path / "node"
    monkeypatch.setenv("NPM_CONFIG_PREFIX", "/usr/local")  # user's own global prefix
    monkeypatch.setenv("NODE_PATH", "/usr/lib/node_modules")
    captured = {}

    def fake_run(cmd, **kw):
        captured["env"] = kw["env"]
        return types.SimpleNamespace(returncode = 0, stdout = "v24.17.0\n", stderr = "")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    assert M._run_node(install_dir, _host("linux", "x64"), ["-v"]) == "v24.17.0"
    env = captured["env"]
    assert env["NPM_CONFIG_PREFIX"] == str(install_dir)
    assert env["npm_config_prefix"] == str(install_dir)
    assert "NODE_PATH" not in env  # inherited NODE_PATH is dropped, not leaked in


def test_ensure_npm_floor_scopes_upgrade_to_install_dir(tmp_path: Path, monkeypatch):
    # A pinned build shipping npm < 11 self-upgrades, but only inside the isolated
    # prefix: it goes through _run_node against install_dir, never the system.
    install_dir = tmp_path / "node"
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: 10)
    calls = []
    monkeypatch.setattr(M, "_run_node", lambda d, h, args, **kw: calls.append((d, args)) or "")
    M._ensure_npm_floor(install_dir, _host("linux", "x64"))
    assert len(calls) == 1
    target_dir, args = calls[0]
    assert target_dir == install_dir  # upgrade scoped to the isolated dir
    assert args[-3:] == ["install", "-g", f"npm@^{M.NPM_MIN_MAJOR}"]


def test_ensure_npm_floor_noop_when_npm_meets_bar(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(M, "installed_npm_major", lambda d, h: M.NPM_MIN_MAJOR)

    def boom(*a, **k):
        raise AssertionError("must not run an npm upgrade when npm already meets the floor")

    monkeypatch.setattr(M, "_run_node", boom)
    M._ensure_npm_floor(tmp_path / "node", _host("linux", "x64"))
