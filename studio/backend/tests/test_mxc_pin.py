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
_LEGACY_STILL_INSTALLED = mxc_runtime._legacy_still_installed


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
    # Nothing installed unless a test puts files there: no runtime of the test host's own is seen.
    monkeypatch.setattr(mxc_runtime, "_legacy_still_installed", lambda: False)


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
        ((1, 26100, -1), _PRIMARY),  # only the update revision unreadable: the probe decides
        ((3, 26100, -1), _LEGACY),  # ...but Windows Server is still known
        ((1, 22631, -1), _LEGACY),  # ...and so is a release older than the table
    ],
)
def test_the_pin_follows_windows_built_in_container_support(monkeypatch, release, pin):
    _windows(monkeypatch, release)
    assert mxc_runtime.selected_pin() is pin


def _managed_files(
    monkeypatch,
    tmp_path,
    pin,
    *,
    prep = True,
    architecture = "x86_64",
):
    """Files of ``pin``'s sizes (sparse, never hashed) in a fresh managed folder."""
    monkeypatch.setattr(mxc_runtime, "_expected_architecture", lambda: architecture)
    monkeypatch.setattr(mxc_runtime, "_installed_package_root", lambda: tmp_path)
    monkeypatch.setattr(mxc_runtime, "_legacy_still_installed", _LEGACY_STILL_INSTALLED)
    for kind, name in mxc_runtime._ARTIFACT_NAMES.items():
        if (architecture, kind) in pin.artifacts and (kind == "exec" or prep):
            with open(tmp_path / name, "wb") as handle:
                handle.truncate(pin.artifacts[(architecture, kind)][1])


@pytest.mark.parametrize("architecture", ["x86_64", "arm64"])
def test_mxc_1_0_1_keeps_running_after_windows_gains_the_container(
    monkeypatch, tmp_path, architecture
):
    # A Windows update moved this PC across the floor; its runtime is still 1.0.1 until the next install.
    _windows(monkeypatch, (1, 26200, 9999))
    _managed_files(monkeypatch, tmp_path, _LEGACY, architecture = architecture)
    assert mxc_runtime.target_pin() is _PRIMARY  # what the installer puts here next
    assert mxc_runtime.selected_pin() is _LEGACY  # what Studio runs meanwhile


@pytest.mark.parametrize(
    "installed",
    ["primary", "none", "legacy-without-prep"],
)
def test_only_a_complete_mxc_1_0_1_install_is_kept(monkeypatch, tmp_path, installed):
    _windows(monkeypatch, (1, 26200, 9999))
    if installed == "primary":
        _managed_files(monkeypatch, tmp_path, _PRIMARY)
    elif installed == "legacy-without-prep":
        _managed_files(monkeypatch, tmp_path, _LEGACY, prep = False)
    else:
        _managed_files(monkeypatch, tmp_path, _PRIMARY, prep = False)
        (tmp_path / "wxc-exec.exe").unlink()
    assert mxc_runtime.selected_pin() is _PRIMARY


def test_a_forced_pin_ignores_what_is_installed(monkeypatch, tmp_path):
    _windows(monkeypatch, (1, 26200, 9999))
    _managed_files(monkeypatch, tmp_path, _LEGACY)
    monkeypatch.setenv(mxc_runtime.PIN_ENV, _PRIMARY.version)
    assert mxc_runtime.selected_pin() is _PRIMARY


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
    # A marker left by an older primary pin never demotes a newer one.
    marker = tmp_path / "mxc-runtime" / "pin-demoted.json"
    marker.write_text(json.dumps({"windows": "1:26150.8", "from": "1.0.9"}), encoding = "utf-8")
    assert not mxc_runtime.demoted_by_probe()
    (tmp_path / "mxc-runtime" / "pin-demoted.json").write_text("not json", encoding = "utf-8")
    assert not mxc_runtime.demoted_by_probe((1, 26150, 7))


def test_a_fallback_recorded_on_an_unreadable_release_keeps_applying(monkeypatch, tmp_path):
    monkeypatch.delenv(mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: None)
    assert mxc_runtime.selected_pin() is _PRIMARY
    mxc_runtime.record_probe_demotion("tier: None")
    assert mxc_runtime.selected_pin() is _LEGACY


def test_an_unreadable_update_revision_keeps_the_product_type_and_build(monkeypatch):
    from types import SimpleNamespace
    import builtins

    real_import = builtins.__import__

    def no_winreg(name, *args, **kwargs):
        if name == "winreg":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        mxc_runtime.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(build = 26100, product_type = 3),
        raising = False,
    )
    monkeypatch.setattr(builtins, "__import__", no_winreg)
    mxc_runtime._windows_release.cache_clear()
    try:
        assert mxc_runtime._windows_release() == (3, 26100, -1)
    finally:
        mxc_runtime._windows_release.cache_clear()


def test_the_demotion_marker_is_written_whole(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: (1, 26150, 7))
    mxc_runtime.record_probe_demotion("tier: None")
    assert [path.name for path in (tmp_path / "mxc-runtime").iterdir()] == ["pin-demoted.json"]


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
        ('{"tier": ""}', 0, None),
    ],
)
def test_probe_tier_reads_the_tier_wxc_reports(monkeypatch, tmp_path, stdout, code, tier):
    from types import SimpleNamespace

    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode = code, stdout = stdout, stderr = "")

    monkeypatch.setattr(mxc_runtime, "_run_wxc_probe", run)
    assert mxc_runtime.probe_tier(tmp_path, pin = _PRIMARY) == tier
    # The tier needs no journal, and the installer may be elevated: Studio's is never replayed here.
    assert calls[0]["replay_journal"] is False and calls[0]["pin"] is _PRIMARY


@pytest.mark.parametrize(
    ("stdout", "code"),
    [("not json", 0), ("[]", 0), ('{"tier": "base-container"}', 1), ("", 0xC0000005)],
)
def test_a_probe_without_a_report_is_no_verdict(monkeypatch, tmp_path, stdout, code):
    from types import SimpleNamespace
    monkeypatch.setattr(
        mxc_runtime,
        "_run_wxc_probe",
        lambda *_a, **_k: SimpleNamespace(returncode = code, stdout = stdout, stderr = "boom"),
    )
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable):
        mxc_runtime.probe_tier(tmp_path)


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
