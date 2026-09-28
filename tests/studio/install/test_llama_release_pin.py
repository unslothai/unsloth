# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The default llama.cpp install stops at the tested release in llama_prebuilt_pins.json.

A "latest" request with no release named resolves the newest published release at or
below the pin; an env override lifts it; an install already past the pin is not
downgraded; and when nothing at or below the pin is installable the installer falls
back to the newest release instead of failing.
"""

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_pin", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

FORK = "unslothai/llama.cpp"
UPSTREAM = "ggml-org/llama.cpp"
PIN = "b11160-mix-a6922cc"


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    pins = tmp_path / "llama_prebuilt_pins.json"
    pins.write_text(json.dumps({"schema_version": 1, "release_tag": PIN}), encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_PINS_FILE", str(pins))
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_RELEASE_TAG", raising = False)
    state = getattr(MOD, "_RELEASE_PIN_STATE", None)
    if state is not None:
        monkeypatch.setitem(state, "suspended", False)
        monkeypatch.setitem(state, "installed_release", None)
    return pins


def _bundle(tag):
    return MOD.PublishedReleaseBundle(repo = FORK, release_tag = tag, upstream_tag = tag.split("-")[0])


@pytest.fixture
def fork_listing(monkeypatch):
    """Newest first, as the API lists them: two regressed newer builds, a newer mix of the
    pinned build, the pin, an older build."""
    tags = [
        "b11250-mix-2222222",
        "b11200-mix-ffffff1",
        "b11160-mix-bbbbbbb",
        PIN,
        "b11100-mix-0000000",
    ]
    monkeypatch.setattr(
        MOD,
        "iter_published_release_bundles",
        lambda repo, tag = "": iter([_bundle(t) for t in tags if not tag or t == tag]),
    )
    monkeypatch.setattr(MOD, "validated_checksums_for_bundle", lambda repo, bundle: object())
    monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")
    return tags


def test_the_shipped_pin_names_a_fork_release_and_ships_in_the_wheel():
    payload = json.loads((PACKAGE_ROOT / "studio" / "llama_prebuilt_pins.json").read_text())
    assert re.fullmatch(r"b\d+(-mix-[0-9a-f]+)?", payload["release_tag"])
    pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    assert "llama_prebuilt_pins.json" in pyproject["tool"]["setuptools"]["package-data"]["studio"]


def test_latest_resolves_the_pin_not_a_newer_release(pinned, fork_listing):
    resolved = MOD.resolve_published_release("latest", FORK)
    assert resolved.bundle.release_tag == PIN
    walked = [r.bundle.release_tag for r in MOD.iter_resolved_published_releases("latest", FORK)]
    assert walked == [PIN, "b11100-mix-0000000"]


def test_the_download_host_is_asked_for_the_pin_itself(pinned, monkeypatch):
    asked = []

    def fake_host(repo, tag = ""):
        asked.append(tag)
        return MOD.ResolvedPublishedRelease(
            bundle = _bundle(tag or "b11200-mix-ffffff1"), checksums = object()
        )

    monkeypatch.setattr(MOD, "_download_host_resolved_release", fake_host)
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    first = next(iter(MOD.iter_resolved_published_releases("latest", FORK)))
    assert asked == [PIN] and first.bundle.release_tag == PIN


def test_after_the_pinned_fast_path_older_capped_releases_stay_reachable(
    pinned, fork_listing, monkeypatch
):
    # A pinned bundle with no asset for this host must not jump the install past the pin
    # while an older release at or below it could serve the host.
    monkeypatch.setattr(
        MOD,
        "_download_host_resolved_release",
        lambda repo, tag = "": MOD.ResolvedPublishedRelease(bundle = _bundle(tag), checksums = object()),
    )
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    walked = [
        r.bundle.release_tag
        for r in MOD.iter_resolved_published_releases(
            "latest", FORK, continue_after_fast_path = False
        )
    ]
    assert walked == [PIN, "b11100-mix-0000000"]


@pytest.mark.parametrize(
    "env", [{"UNSLOTH_LLAMA_TAG": "latest"}, {"UNSLOTH_LLAMA_RELEASE_TAG": "b11200-mix-ffffff1"}]
)
def test_an_env_override_lifts_the_pin(pinned, fork_listing, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert MOD.resolve_published_release("latest", FORK).bundle.release_tag == fork_listing[0]


def test_an_explicit_request_is_not_capped(pinned, fork_listing):
    resolved = MOD.resolve_published_release("latest", FORK, "b11200-mix-ffffff1")
    assert resolved.bundle.release_tag == "b11200-mix-ffffff1"
    assert MOD.release_pin_for("b11200", "", FORK) is None
    assert MOD.release_pin_for("latest", "", "someone/llama.cpp") is None


def test_a_missing_or_broken_pins_file_is_todays_behaviour(pinned, fork_listing):
    pinned.write_text("{not json", encoding = "utf-8")
    assert MOD.resolve_published_release("latest", FORK).bundle.release_tag == fork_listing[0]
    pinned.unlink()
    assert MOD.resolve_published_release("latest", FORK).bundle.release_tag == fork_listing[0]


def test_an_install_already_past_the_pin_is_not_downgraded(
    pinned, fork_listing, tmp_path, monkeypatch
):
    install = tmp_path / "llama.cpp"
    install.mkdir()
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b11200-mix-ffffff1", "tag": "b11200"}), encoding = "utf-8"
    )
    seen = {}

    def fake_install(**kw):
        seen["pin"] = MOD.default_release_pin()
        seen["resolved"] = MOD.resolve_published_release("latest", FORK).bundle.release_tag

    monkeypatch.setattr(MOD, "install_prebuilt", fake_install)
    monkeypatch.setattr(sys, "argv", ["install_llama_prebuilt.py", "--install-dir", str(install)])
    MOD.main()
    # Capped at its own release: neither downgraded to the pin nor moved further past it.
    assert seen == {"pin": "b11200-mix-ffffff1", "resolved": "b11200-mix-ffffff1"}
    MOD._RELEASE_PIN_STATE["installed_release"] = "b11030-mix-5ff778e"
    assert MOD.default_release_pin() == PIN
    # Mix names carry no order, so another mix of the pinned build is kept as well.
    MOD._RELEASE_PIN_STATE["installed_release"] = "b11160-mix-bbbbbbb"
    assert MOD.default_release_pin() == "b11160-mix-bbbbbbb"
    MOD._RELEASE_PIN_STATE["installed_release"] = PIN
    assert MOD.default_release_pin() == PIN


def test_nothing_installable_at_or_below_the_pin_falls_back_to_the_newest(pinned, monkeypatch):
    calls = []

    def fake_plans(llama_tag, host, repo, tag, *, max_release_fallbacks):
        pin = MOD.release_pin_for(llama_tag, tag, repo)
        calls.append(pin)
        if pin is not None:
            raise MOD.PrebuiltFallback("no compatible prebuilt asset was found")
        return "latest", ["newest-plan"]

    monkeypatch.setattr(MOD, "_fork_manifest_release_plans", fake_plans)
    assert MOD.resolve_simple_install_release_plans("latest", object(), FORK, "") == (
        "latest",
        ["newest-plan"],
    )
    assert calls == [PIN, None]


def test_an_explicit_request_that_fails_still_fails(pinned, monkeypatch):
    def fake_plans(*args, **kwargs):
        raise MOD.PrebuiltFallback("nope")

    monkeypatch.setattr(MOD, "_fork_manifest_release_plans", fake_plans)
    with pytest.raises(MOD.PrebuiltFallback):
        MOD.resolve_simple_install_release_plans("latest", object(), FORK, "b11200-mix-ffffff1")


def test_the_upstream_scan_is_capped_at_the_pins_build(pinned, monkeypatch):
    releases = [{"tag_name": t} for t in ("b11210", "b11160", "b11100")]
    monkeypatch.setattr(MOD, "iter_release_payloads_by_time", lambda *a: iter(releases))
    monkeypatch.setattr(
        MOD,
        "direct_upstream_release_plan",
        lambda release, host, repo, tag: MOD.InstallReleasePlan(
            requested_tag = tag,
            llama_tag = release["tag_name"],
            release_tag = release["tag_name"],
            attempts = [],
            approved_checksums = None,
        ),
    )
    host = MOD.HostInfo(
        system = "Linux",
        machine = "aarch64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    _tag, plans = MOD.resolve_simple_install_release_plans("latest", host, UPSTREAM, "")
    assert [p.release_tag for p in plans] == ["b11160", "b11100"]


def test_a_source_build_compiles_the_pins_build_without_asking_github(pinned, monkeypatch):
    def offline(*a, **k):
        raise AssertionError("no network call is needed under a pin")

    monkeypatch.setattr(MOD, "fetch_json", offline)
    assert MOD._source_fallback_tag("latest", FORK, "") == "b11160"
    assert MOD.resolve_requested_llama_tag("latest", UPSTREAM) == "b11160"


def test_a_custom_repo_source_fallback_is_not_capped(pinned, monkeypatch):
    def no_release(*a, **k):
        raise MOD.PrebuiltFallback("custom repo has no usable release")

    monkeypatch.setattr(MOD, "resolve_published_release", no_release)
    monkeypatch.setattr(MOD, "fetch_json", lambda url, *a, **k: {"tag_name": "b11223"})
    assert MOD.resolve_requested_llama_tag("latest", "someone/llama.cpp") == "b11223"


def test_the_installed_release_is_read_again_under_the_install_lock(pinned, tmp_path, monkeypatch):
    install = tmp_path / "llama.cpp"
    install.mkdir()
    marker = install / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(json.dumps({"release_tag": "b11100-mix-0000000"}), encoding = "utf-8")
    MOD._RELEASE_PIN_STATE["installed_release"] = "b11100-mix-0000000"
    real_lock = MOD.install_lock

    def lock_after_another_update(path):
        # Another updater finished while this one waited for the lock.
        marker.write_text(json.dumps({"release_tag": "b11200-mix-ffffff1"}), encoding = "utf-8")
        return real_lock(path)

    seen = {}

    def stop(*a, **k):
        seen["pin"] = MOD.default_release_pin()
        raise RuntimeError("stop after planning starts")

    monkeypatch.setattr(MOD, "install_lock", lock_after_another_update)
    monkeypatch.setattr(MOD, "effective_backend_request", stop)
    with pytest.raises(BaseException):
        MOD.install_prebuilt(install, "latest", FORK, "")
    assert seen["pin"] == "b11200-mix-ffffff1"


def test_the_no_listing_current_check_expects_the_pin(pinned, monkeypatch):
    def offline(*a, **k):
        raise AssertionError("the pin names the answer without a request")

    monkeypatch.setattr(MOD, "_download_host_latest_release_tag", offline)
    monkeypatch.setattr(MOD, "_api_newest_release_tag", offline)
    assert MOD._expected_release_tag_without_plan({}, "latest", FORK, "") == PIN
    assert MOD._expected_release_tag_without_plan({}, "latest", UPSTREAM, "") == "b11160"


def test_the_pin_is_not_a_user_pin_for_setup():
    """setup.sh / setup.ps1 still pass "latest" and never UNSLOTH_LLAMA_RELEASE_TAG for the
    default, so the keep-GPU-prebuilt-over-a-CPU-source-build rule (#9255) still applies."""
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text()
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8-sig")
    assert '_DEFAULT_LLAMA_TAG="latest"' in setup_sh
    assert '$DefaultLlamaTag = "latest"' in setup_ps1
    assert "llama_prebuilt_pins" not in re.sub(r"(?m)^\s*#.*$", "", setup_sh)
