# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_unsloth_link: nothing that differs from its pin is extracted or run, installs are atomic."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import threading
import urllib.error
import zipfile
from pathlib import Path

import pytest

from studio import install_unsloth_link as M
from studio.cluster_link import tailnet
from studio.cluster_link.tailcat_helper import package

TARGET = "linux-amd64"


def _tar(members) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj = raw, mode = "w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "link":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            else:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if kind != "link" else None)
    return raw.getvalue()


def _zip(members) -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zf:
        for name, data, kind in members:
            info = zipfile.ZipInfo(name)
            if kind == "link":
                info.external_attr = 0o120777 << 16
            zf.writestr(info, data)
    return raw.getvalue()


GOOD = [(M.BINARY, b"#!/bin/sh\necho nodekey:abc\n", "file"), (M.LICENSE, b"BSD", "file")]


def _archived(data: bytes) -> bytes:
    # The installer looks for this host's executable name (unsloth-link.exe on Windows), whatever TARGET says.
    try:
        with tarfile.open(fileobj = io.BytesIO(data), mode = "r:gz") as tar:
            return tar.extractfile(M.BINARY).read()
    except Exception:
        pass
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return zf.read(M.BINARY)
    except Exception:
        return b""  # a malformed archive: refused before its executable matters


@pytest.fixture
def env(tmp_path, monkeypatch):
    rel = tmp_path / "release"
    rel.mkdir()
    root = tmp_path / "root"
    monkeypatch.setattr(M, "install_root", lambda: root)
    monkeypatch.setattr(M, "host_target", lambda *a, **k: TARGET)
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)
    smoked = []
    monkeypatch.setattr(M, "_smoke", lambda binary: smoked.append(binary))
    monkeypatch.setenv(M.ENV_RELEASE_BASE, rel.as_uri())

    def publish(
        data: bytes,
        *,
        name = "unsloth-link-linux-amd64.tar.gz",
        sha = None,
        size = None,
        binary = None,
    ):
        (rel / name).write_bytes(data)
        asset = {
            "name": name,
            "sha256": sha or hashlib.sha256(data).hexdigest(),
            "size": size or len(data),
            "binary_sha256": hashlib.sha256(binary or _archived(data)).hexdigest(),
        }
        return {
            "schema_version": 1,
            "repo": "o/r",
            "tag": "unsloth-link-v9",
            "assets": {TARGET: asset},
        }

    return type("Env", (), {"root": root, "publish": staticmethod(publish), "smoked": smoked})


def test_a_verified_archive_installs_atomically_and_is_found_again(env):
    pins = env.publish(_tar(GOOD))
    binary = M.ensure(pins)
    assert binary == env.root / "unsloth-link-v9" / M.BINARY
    assert len(env.smoked) == 1
    assert os.access(binary, os.X_OK) or os.name == "nt"
    assert M.installed_binary(pins) == binary
    assert not [p for p in env.root.iterdir() if p.name.startswith(".staging-")]
    assert M.ensure(pins) == binary and len(env.smoked) == 1


@pytest.mark.parametrize(
    "tamper, needle",
    [
        ({"sha": "0" * 64}, "pinned sha256"),
        ({"size": 10}, "larger than its pin"),
        ({"size": 10**8}, "its pin says"),
    ],
)
def test_an_archive_that_differs_from_its_pin_is_never_extracted_or_run(env, tamper, needle):
    pins = env.publish(_tar(GOOD), **tamper)
    with pytest.raises(M.InstallError, match = needle):
        M.ensure(pins)
    assert env.smoked == []
    assert M.installed_binary(pins) is None
    assert [p.name for p in env.root.iterdir()] == [".install.lock"] or not list(env.root.iterdir())


@pytest.mark.parametrize(
    "members, fmt",
    [
        ([("../" + M.BINARY, b"x", "file")], "tar"),
        ([("bin/" + M.BINARY, b"x", "file")], "tar"),
        (GOOD + [("extra", b"x", "file")], "tar"),
        ([(M.BINARY, b"", "link")], "tar"),
        ([(M.LICENSE, b"BSD", "file")], "tar"),
        ([("..\\" + M.BINARY, b"x", "file")], "zip"),
        ([(M.BINARY, b"x", "link")], "zip"),
    ],
)
def test_unexpected_archive_entries_are_refused(env, members, fmt):
    data = _zip(members) if fmt == "zip" else _tar(members)
    name = "unsloth-link-linux-amd64." + ("zip" if fmt == "zip" else "tar.gz")
    pins = env.publish(data, name = name)
    with pytest.raises(M.InstallError):
        M.ensure(pins)
    assert env.smoked == [] and M.installed_binary(pins) is None
    assert not (env.root.parent / M.BINARY).exists()


def test_a_failed_smoke_leaves_the_previous_state_untouched(env, monkeypatch):
    pins = env.publish(_tar(GOOD))

    def bad(_binary):
        raise M.InstallError("the downloaded unsloth-link does not run here: nope")

    monkeypatch.setattr(M, "_smoke", bad)
    with pytest.raises(M.InstallError, match = "does not run here"):
        M.ensure(pins)
    assert not (env.root / "unsloth-link-v9").exists()
    assert not [p for p in env.root.iterdir() if p.name.startswith(".staging-")]


def test_a_stale_install_with_a_wrong_marker_is_replaced(env):
    pins = env.publish(_tar(GOOD))
    stale = env.root / "unsloth-link-v9"
    stale.mkdir(parents = True)
    (stale / M.BINARY).write_text("old")
    (stale / M.MARKER).write_text(json.dumps({"sha256": "1" * 64, "target": TARGET}))
    assert M.installed_binary(pins) is None
    binary = M.ensure(pins)
    assert binary.read_bytes() == GOOD[0][1]


def test_concurrent_installs_download_once(env, monkeypatch):
    pins = env.publish(_tar(GOOD))
    fetches, real = [], M._fetch
    gate = threading.Event()

    def slow(url, dest, asset):
        fetches.append(url)
        gate.wait(2)
        real(url, dest, asset)

    monkeypatch.setattr(M, "_fetch", slow)
    results = []
    threads = [threading.Thread(target = lambda: results.append(M.ensure(pins))) for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(30)
    assert len(fetches) == 1 and len(set(results)) == 1 and len(results) == 4


def test_an_unsupported_platform_says_local_pairing_still_works(env, monkeypatch):
    monkeypatch.setattr(M, "host_target", lambda *a, **k: None)
    with pytest.raises(M.InstallError, match = "local network still works"):
        M.ensure(env.publish(_tar(GOOD)))


def test_offline_says_what_to_do(env, monkeypatch):
    pins = env.publish(_tar(GOOD))
    calls = []

    def offline(url, dest, asset):
        calls.append(url)
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(M, "_fetch", offline)
    with pytest.raises(M.InstallError, match = "connect this machine to the internet once"):
        M.ensure(pins)
    assert len(calls) == M.DOWNLOAD_ATTEMPTS


def test_a_404_is_not_retried(env, monkeypatch):
    pins = env.publish(_tar(GOOD))
    calls = []

    def missing(url, dest, asset):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(M, "_fetch", missing)
    with pytest.raises(M.InstallError, match = "not published yet"):
        M.ensure(pins)
    assert len(calls) == 1


def test_release_url_and_override(monkeypatch):
    pins = {"repo": "unslothai/unsloth", "tag": "unsloth-link-v1"}
    asset = {"name": "unsloth-link-linux-amd64.tar.gz"}
    monkeypatch.delenv(M.ENV_RELEASE_BASE, raising = False)
    assert M.asset_url(pins, asset) == (
        "https://github.com/unslothai/unsloth/releases/download/unsloth-link-v1/unsloth-link-linux-amd64.tar.gz"
    )
    monkeypatch.setenv(M.ENV_RELEASE_BASE, "https://staging.example/rel/")
    assert M.asset_url(pins, asset) == "https://staging.example/rel/unsloth-link-linux-amd64.tar.gz"


@pytest.mark.parametrize(
    "system, machine, want",
    [
        ("Linux", "x86_64", "linux-amd64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Darwin", "arm64", "darwin-arm64"),
        ("Windows", "AMD64", "windows-amd64"),
        ("Windows", "ARM64", "windows-arm64"),
        ("Linux", "armv7l", None),
        ("FreeBSD", "amd64", None),
    ],
)
def test_host_targets(system, machine, want):
    assert M.host_target(system, machine) == want


def test_the_shipped_pins_cover_every_built_target():
    pins = M.load_pins()
    assert set(pins["assets"]) == set(package.TARGETS)
    for target, asset in pins["assets"].items():
        assert asset["name"] == package.asset_name(target)
        assert len(asset["binary_sha256"]) == 64 and asset["binary_sha256"] != asset["sha256"]


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "../x.tar.gz", "sha256": "0" * 64, "size": 1, "binary_sha256": "0" * 64},
        {"name": "x.tar.gz", "sha256": "0" * 64, "size": 1},
    ],
    ids = ["path", "no-binary-pin"],
)
def test_a_bad_pins_entry_is_refused(tmp_path, entry):
    bad = {"schema_version": 1, "assets": {TARGET: entry}}
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(M.InstallError, match = "bad entry"):
        M.load_pins(path)


def test_an_installed_executable_that_differs_from_its_pin_is_not_used(env):
    pins = env.publish(_tar(GOOD))
    binary = M.ensure(pins)
    assert M.installed_binary(pins) == binary
    binary.write_bytes(b"#!/bin/sh\necho nodekey:evil\n")  # marker kept, executable swapped
    assert M.installed_binary(pins) is None
    assert M.ensure(pins).read_bytes() == GOOD[0][1] and len(env.smoked) == 2


@pytest.mark.skipif(os.name == "nt", reason = "symlinks need privileges on Windows")
def test_a_symlinked_installed_executable_is_not_used(env, tmp_path):
    pins = env.publish(_tar(GOOD))
    binary = M.ensure(pins)
    real = tmp_path / "elsewhere"
    real.write_bytes(binary.read_bytes())
    binary.unlink()
    binary.symlink_to(real)
    assert M.installed_binary(pins) is None


def test_an_archive_whose_executable_differs_from_its_pin_is_not_run(env):
    pins = env.publish(_tar(GOOD), binary = b"other")
    with pytest.raises(M.InstallError, match = "does not match its pin"):
        M.ensure(pins)
    assert env.smoked == [] and M.installed_binary(pins) is None


def test_the_installed_executable_is_hashed_once_per_process(env, monkeypatch):
    pins = env.publish(_tar(GOOD))
    binary = M.ensure(pins)
    M._VERIFIED.clear()
    opened = []
    real_open = open
    monkeypatch.setattr(
        "builtins.open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k)
    )
    for _ in range(3):
        assert M.installed_binary(pins) == binary
    assert opened.count(str(binary)) == 1


@pytest.mark.skipif(os.name == "nt", reason = "shell-script stand-in for the helper")
@pytest.mark.parametrize(
    "script, ok", [("echo nodekey:abc", True), ("echo boom >&2; exit 3", False)]
)
def test_the_smoke_runs_genkey(tmp_path, script, ok):
    binary = tmp_path / M.BINARY
    binary.write_text(f"#!/bin/sh\n{script}\n")
    binary.chmod(0o755)
    if ok:
        M._smoke(binary)
    else:
        with pytest.raises(M.InstallError, match = "does not run here"):
            M._smoke(binary)


def test_tailnet_prefers_the_override_then_the_verified_install(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / M.BINARY
    fake.parent.mkdir()
    fake.write_text("x")
    monkeypatch.setenv(tailnet.ENV_BIN, str(fake))
    assert tailnet.helper_path() == fake and tailnet.install() == fake
    monkeypatch.setenv(tailnet.ENV_BIN, str(tmp_path / "missing"))
    assert tailnet.helper_path() is None
    with pytest.raises(tailnet.TailnetUnavailable):
        tailnet.install()
    monkeypatch.delenv(tailnet.ENV_BIN)
    monkeypatch.setattr(M, "installed_binary", lambda pins = None: fake)
    assert tailnet.helper_path() == fake


def test_tailnet_install_reports_an_install_failure_as_unavailable(monkeypatch):
    monkeypatch.delenv(tailnet.ENV_BIN, raising = False)
    monkeypatch.setattr(M, "installed_binary", lambda pins = None: None)

    def fail(**_kw):
        raise M.InstallError("could not download unsloth-link (offline)")

    monkeypatch.setattr(M, "ensure", fail)
    with pytest.raises(tailnet.TailnetUnavailable, match = "could not download"):
        tailnet.install()
    assert tailnet.helper_path() is None


def test_helper_path_never_downloads(monkeypatch):
    monkeypatch.delenv(tailnet.ENV_BIN, raising = False)
    monkeypatch.setattr(M, "installed_binary", lambda pins = None: None)
    monkeypatch.setattr(M, "ensure", lambda **_kw: pytest.fail("helper_path must not download"))
    assert tailnet.helper_path() is None and not tailnet.available()


@pytest.mark.skipif(sys.platform != "linux", reason = "packs the linux archive layout")
def test_package_archives_are_reproducible_and_install(tmp_path, env):
    dist = tmp_path / "dist"
    for target in package.TARGETS:
        (dist / target).mkdir(parents = True)
        (dist / target / package.binary_name(target)).write_bytes(
            b"#!/bin/sh\necho nodekey:" + target.encode() + b"\n"
        )
        (dist / target / package.LICENSE).write_text("BSD")
    a = package.package(dist, tmp_path / "a")
    b = package.package(dist, tmp_path / "b")
    assert a == b
    data = (tmp_path / "a" / package.asset_name(TARGET)).read_bytes()
    pins = env.publish(data)
    assert M.ensure(pins).read_bytes().endswith(b"nodekey:linux-amd64\n")


def test_update_pins_records_each_archived_executable(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(package.__file__).parent))
    import update_pins

    dist = tmp_path / "dist"
    for target in package.TARGETS:
        (dist / target).mkdir(parents = True)
        (dist / target / package.binary_name(target)).write_bytes(target.encode())
        (dist / target / package.LICENSE).write_text("BSD")
    package.package(dist, tmp_path / "out")
    assets = update_pins.collect(tmp_path / "out")
    for target in package.TARGETS:
        assert assets[target]["binary_sha256"] == hashlib.sha256(target.encode()).hexdigest()
