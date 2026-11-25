# SPDX-License-Identifier: AGPL-3.0-only

"""The two MXC pins: 1.1.0 where Windows has the built-in container, 1.0.1 (AppContainer + DACL) elsewhere."""

from __future__ import annotations

import json
import os
import re
import urllib.request

import pytest

from core.inference import mxc_runtime

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PRIMARY = mxc_runtime.PRIMARY_PIN
_LEGACY = mxc_runtime.LEGACY_PIN


@pytest.mark.parametrize("pin", [_PRIMARY, _LEGACY], ids = ["primary", "legacy"])
def test_each_pin_is_well_formed(pin):
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin.version)
    assert re.fullmatch(r"[0-9a-f]{40}", pin.revision)
    assert pin.archive_size > 0 and _SHA256.fullmatch(pin.archive_sha256)
    assert pin.url == f"https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{pin.version}.tgz"
    for architecture in ("x86_64", "arm64"):
        member, size, sha256 = pin.artifacts[(architecture, "exec")]
        folder = "x64" if architecture == "x86_64" else "arm64"
        assert member == f"package/bin/{folder}/wxc-exec.exe"
        assert size > 0 and _SHA256.fullmatch(sha256)
    for (architecture, kind), (member, size, sha256) in pin.artifacts.items():
        assert architecture in ("x86_64", "arm64") and kind in ("exec", "prep")
        assert member.endswith("wxc-exec.exe" if kind == "exec" else "wxc-host-prep.exe")


def test_only_the_legacy_pin_ships_host_preparation():
    # 1.1.0 removed the AppContainer + DACL tier and wxc-host-prep (microsoft/mxc#1558).
    assert not _PRIMARY.has_host_prep
    assert _LEGACY.has_host_prep
    assert {key for key in _LEGACY.artifacts if key[1] == "prep"} == {
        ("x86_64", "prep"),
        ("arm64", "prep"),
    }


def test_the_primary_pin_is_newer_than_the_legacy_one():
    def parse(version):
        return tuple(int(part) for part in version.split("."))

    assert parse(_PRIMARY.version) > parse(_LEGACY.version)
    assert mxc_runtime.MXC_SCHEMA_VERSION == "1.0.0"  # the config shape both pins accept


def _windows(
    monkeypatch,
    release,
    *,
    demoted = None,
):
    monkeypatch.delenv(mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: release)
    monkeypatch.setattr(mxc_runtime, "demoted_by_probe", lambda _release = None: bool(demoted))


@pytest.mark.parametrize(
    ("release", "pin"),
    [
        ((1, 26100, 9278), _PRIMARY),  # Windows 11 24H2 at the first update with the container
        ((1, 26100, 9277), _LEGACY),
        ((1, 26200, 9457), _PRIMARY),  # 25H2
        ((1, 26200, 100), _LEGACY),
        ((1, 26300, 9549), _LEGACY),
        ((1, 26300, 9550), _PRIMARY),
        ((1, 28000, 2804), _PRIMARY),
        ((1, 28000, 2803), _LEGACY),
        ((1, 29000, 1), _PRIMARY),  # newer than the table
        ((1, 26150, 1), _PRIMARY),  # Insider build between rows: the installer's probe decides
        ((1, 22631, 4000), _LEGACY),  # Windows 11 23H2
        ((3, 26100, 9999), _LEGACY),  # Windows Server 2025
        ((2, 26100, 9999), _LEGACY),  # domain controller
        ((3, 20348, 2000), _LEGACY),  # Windows Server 2022
        (None, _PRIMARY),  # unreadable: the installer's probe decides
    ],
)
def test_the_pin_follows_windows_built_in_container_support(monkeypatch, release, pin):
    _windows(monkeypatch, release)
    assert mxc_runtime.selected_pin() is pin


def test_a_probe_demotion_moves_a_supported_looking_windows_to_the_legacy_pin(monkeypatch):
    _windows(monkeypatch, (1, 29000, 1), demoted = True)
    assert mxc_runtime.selected_pin() is _LEGACY


@pytest.mark.parametrize("version", [_PRIMARY.version, _LEGACY.version])
def test_the_environment_can_force_either_pin(monkeypatch, version):
    _windows(monkeypatch, (3, 26100, 1))
    monkeypatch.setenv(mxc_runtime.PIN_ENV, version)
    assert mxc_runtime.selected_pin().version == version


def test_an_unknown_forced_version_is_ignored(monkeypatch):
    _windows(monkeypatch, (3, 26100, 1))
    monkeypatch.setenv(mxc_runtime.PIN_ENV, "0.8.0")
    assert mxc_runtime.selected_pin() is _LEGACY


def test_the_demotion_marker_is_keyed_on_the_windows_release(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: (1, 26150, 7))
    assert not mxc_runtime.demoted_by_probe()
    mxc_runtime.record_probe_demotion("tier: None")
    assert mxc_runtime.demoted_by_probe()
    # A Windows update re-opens the question: 1.1.0 is tried again.
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: (1, 26150, 8))
    assert not mxc_runtime.demoted_by_probe()
    (tmp_path / "mxc-runtime" / "pin-demoted.json").write_text("not json", encoding = "utf-8")
    assert not mxc_runtime.demoted_by_probe((1, 26150, 7))


def test_host_preparation_is_never_asked_of_the_primary_pin(monkeypatch):
    monkeypatch.setattr(mxc_runtime, "selected_pin", lambda: _PRIMARY)
    monkeypatch.setattr(
        mxc_runtime, "_run_wxc_probe", lambda *_a, **_k: pytest.fail("probed for host preparation")
    )
    assert mxc_runtime.probe_host_prep_steps() == ()


@pytest.mark.parametrize(
    ("stdout", "code", "tier"),
    [
        ('{"tier": "base-container", "warnings": []}', 0, "base-container"),
        ('{"tier": "appcontainer-dacl"}', 0, "appcontainer-dacl"),
        ('{"warnings": []}', 0, None),  # 1.1.0 on Windows Server: exit 0, no tier
        ("not json", 0, None),
        ('{"tier": "base-container"}', 1, None),
    ],
)
def test_probe_tier_reads_the_tier_wxc_reports(monkeypatch, tmp_path, stdout, code, tier):
    from types import SimpleNamespace
    monkeypatch.setattr(
        mxc_runtime,
        "_run_wxc_probe",
        lambda *_a, **_k: SimpleNamespace(returncode = code, stdout = stdout, stderr = ""),
    )
    assert mxc_runtime.probe_tier(tmp_path) == tier


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_MXC_PIN_NETWORK") != "1",
    reason = "set UNSLOTH_MXC_PIN_NETWORK=1 to compare the pins with the npm registry",
)
def test_the_pins_match_the_npm_registry():
    request = urllib.request.Request(
        "https://registry.npmjs.org/@microsoft/mxc-sdk",
        headers = {"User-Agent": "unsloth-studio-mxc-pin-check"},
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        registry = json.load(response)
    latest = registry["dist-tags"]["latest"]
    assert (
        _PRIMARY.version == latest
    ), f"MXC {latest} is on npm; the primary pin is {_PRIMARY.version}"
    for pin in (_PRIMARY, _LEGACY):
        dist = registry["versions"][pin.version]["dist"]
        assert dist["tarball"] == pin.url
        assert dist.get("unpackedSize", 0) > 0
