# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows CUDA line ordering, the portable fallback attempt, and SM coverage on the two
keep paths that skip the selector (the no-plan fast path and the update-failure keep)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
MODULE_NAME = "studio_install_llama_prebuilt"
if MODULE_NAME in sys.modules:
    m = sys.modules[MODULE_NAME]
else:
    SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert SPEC is not None and SPEC.loader is not None
    m = importlib.util.module_from_spec(SPEC)
    sys.modules[MODULE_NAME] = m
    SPEC.loader.exec_module(m)


def host(
    system="Windows",
    caps=("89",),
    driver=(13, 0),
    **extra,
):
    args = dict(
        system=system,
        machine="AMD64" if system == "Windows" else "x86_64",
        is_windows=system == "Windows",
        is_linux=system == "Linux",
        is_macos=False,
        is_x86_64=True,
        is_arm64=False,
        nvidia_smi="nvidia-smi",
        driver_cuda_version=driver,
        compute_caps=list(caps),
        physical_compute_caps=list(caps),
        visible_cuda_devices=None,
        has_physical_nvidia=True,
        has_usable_nvidia=True,
    )
    args.update(extra)
    return m.HostInfo(**args)


def artifact(
    name,
    line,
    profile,
    sms,
    coverage="targeted",
    rank=100,
    kind="windows-cuda",
):
    caps = [str(s) for s in sms]
    return m.PublishedLlamaArtifact(
        asset_name=name,
        install_kind=kind,
        runtime_line=line,
        coverage_class=coverage,
        supported_sms=caps,
        min_sm=int(caps[0]),
        max_sm=int(caps[-1]),
        bundle_profile=profile,
        rank=rank,
    )


# The fork's app-named bundles: no CUDA minor in the name, so the lines are major-gated.
ARTIFACTS = [
    artifact("llama-app-win-cuda12-legacy-x64.zip", "cuda12", "cuda12-legacy", [50, 61, 70, 75]),
    artifact("llama-app-win-cuda12-older-x64.zip", "cuda12", "cuda12-older", [75, 80, 86, 89]),
    artifact("llama-app-win-cuda13-older-x64.zip", "cuda13", "cuda13-older", [75, 80, 86, 89]),
    artifact("llama-app-win-cuda13-newer-x64.zip", "cuda13", "cuda13-newer", [89, 90, 100, 120]),
    artifact(
        "llama-app-win-cuda13-portable-x64.zip",
        "cuda13",
        "cuda13-portable",
        [75, 80, 86, 89, 90, 100, 120],
        "portable",
        900,
    ),
]


def release(artifacts=ARTIFACTS):
    return m.PublishedReleaseBundle(
        repo="unslothai/llama.cpp",
        release_tag="v1.0",
        upstream_tag="b8508",
        assets={a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        artifacts=list(artifacts),
    )


BUNDLE = release()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for key in (
        "UNSLOTH_LLAMA_CPP_BACKEND",
        "UNSLOTH_PREBUILT_FULL_CHECK",
        "UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE",
        "UNSLOTH_ROCM_GFX_ARCH",
        "UNSLOTH_ROCM_GFX_REMEMBERED",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        m, "detect_torch_cuda_runtime_preference", lambda *a, **k: m.CudaRuntimePreference(None, [])
    )
    monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: ([], {}))


def profiles(h):
    return [a.bundle_profile for a in m.published_windows_cuda_attempts(h, BUNDLE, None)]


class TestDetectedRuntimeDllsOnlyOrderTheLines:
    def test_a_pascal_card_beside_cuda13_dlls_still_gets_the_cuda12_legacy_bundle(
        self, monkeypatch
    ):
        pascal = host(caps=("61",))
        assert profiles(pascal) == ["cuda12-legacy"]
        # torch or another app installed the CUDA 13 runtime; the card is still Pascal.
        monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: (["cuda13"], {}))
        assert profiles(pascal) == ["cuda12-legacy"]

    def test_the_detected_line_is_tried_first_and_the_other_line_still_follows(self, monkeypatch):
        ada = host(caps=("89",))
        assert profiles(ada) == ["cuda13-older", "cuda13-portable", "cuda12-older"]
        monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: (["cuda12"], {}))
        assert profiles(ada) == ["cuda12-older", "cuda13-older", "cuda13-portable"]


class TestTheFastPathAgreesWithTheOrdering:
    def test_every_driver_compatible_windows_line_is_selectable(self, monkeypatch):
        # torch prefers CUDA 12 while only CUDA 13 DLLs are detected: the selector now moves the
        # cuda12 bundle to the front, so the fast path must call that line selectable too.
        monkeypatch.setattr(m, "detected_windows_runtime_lines", lambda: (["cuda13"], {}))
        win = host(caps=("89",))
        assert m._runtime_line_selectable(win, "cuda12") is True
        assert m._runtime_line_selectable(win, "cuda13") is True
        assert m._runtime_line_selectable(host(caps=("89",), driver=(12, 8)), "cuda13") is False
        # Linux still needs the runtime on disk.
        monkeypatch.setattr(m, "detected_linux_runtime_lines", lambda: (["cuda13"], {}))
        assert m._runtime_line_selectable(host("Linux", caps=("89",)), "cuda12") is False


class TestThePortableBundleIsTheFallbackAttempt:
    def test_it_follows_the_targeted_bundle_instead_of_replacing_it(self):
        attempts = m.published_windows_cuda_attempts(host(caps=("89",)), BUNDLE, None)
        cuda13 = [a for a in attempts if a.runtime_line == "cuda13"]
        assert [a.coverage_class for a in cuda13] == ["targeted", "portable"]
        assert cuda13[0].bundle_profile == "cuda13-older"

    def test_it_is_the_only_attempt_when_no_targeted_bundle_covers_the_card(self):
        # sm_100 is in cuda13-newer, so drop that one from the release to force the case.
        thin = release([a for a in ARTIFACTS if a.bundle_profile != "cuda13-newer"])
        attempts = m.published_windows_cuda_attempts(host(caps=("100",)), thin, None)
        assert [a.coverage_class for a in attempts] == ["portable"]


def checksums(choice):
    return m.ApprovedReleaseChecksums(
        repo=BUNDLE.repo,
        release_tag=BUNDLE.release_tag,
        upstream_tag=BUNDLE.upstream_tag,
        artifacts={
            choice.name: m.ApprovedArtifactHash(
                choice.name, "a" * 64, choice.repo, choice.install_kind
            )
        },
    )


def choice_for(h):
    return dataclasses.replace(
        m.published_windows_cuda_attempts(h, BUNDLE, None)[0], expected_sha256="a" * 64
    )


def seed_install(root, h, choice):
    bin_dir = m.install_runtime_dir(root, h)
    bin_dir.mkdir(parents=True, exist_ok=True)
    ext = ".exe" if h.is_windows else ""
    for name in ("llama-server", "llama-quantize"):
        path = bin_dir / (name + ext)
        path.write_bytes(b"placeholder, image checks are stubbed")
        path.chmod(0o755)
    m.write_prebuilt_metadata(
        root,
        host=h,
        requested_tag="latest",
        llama_tag=BUNDLE.upstream_tag,
        release_tag=BUNDLE.release_tag,
        choice=choice,
        approved_checksums=checksums(choice),
        prebuilt_fallback_used=False,
        backend_request="auto",
    )


@pytest.fixture
def healthy_payload(monkeypatch):
    """Coverage policy only: the bytes on disk are placeholders, so the loader checks are stubbed."""
    for name in (
        "_install_tree_is_usable",
        "_kept_install_payload_is_healthy",
        "runtime_payload_is_healthy",
        "_binary_image_runs",
    ):
        monkeypatch.setattr(m, name, lambda *a, **k: True)
    for name in ("confirm_install_tree", "preflight_linux_installed_binaries"):
        monkeypatch.setattr(m, name, lambda *a, **k: None)
    monkeypatch.setattr(m, "_diffusion_visual_server_missing_for_marker", lambda *a, **k: False)


def fast_path(root):
    return m.existing_install_current_without_plan(
        root,
        llama_tag="latest",
        published_repo=BUNDLE.repo,
        published_release_tag="",
        backend_request="auto",
        force_cpu=False,
    )


@pytest.mark.parametrize("system", ["Linux", "Windows"])
class TestTheKeepPathsRequireSmCoverage:
    def test_a_card_swapped_under_a_mask_is_not_kept_by_the_fast_path(
        self, monkeypatch, tmp_path, system, healthy_payload
    ):
        # An empty CUDA_VISIBLE_DEVICES hides the caps from the profile; the physical caps
        # are the only record that an sm_89 card became an sm_120 one.
        masked = host(
            system,
            (),
            physical_compute_caps=["89"],
            has_usable_nvidia=False,
            visible_cuda_devices="",
        )
        swapped = dataclasses.replace(masked, physical_compute_caps=["120"])
        root = tmp_path / "llama"
        seed_install(
            root,
            masked,
            choice_for(dataclasses.replace(masked, compute_caps=["89"], has_usable_nvidia=True)),
        )
        monkeypatch.setattr(m, "_download_host_latest_release_tag", lambda *a: BUNDLE.release_tag)
        assert m.host_profile(masked) == m.host_profile(swapped)
        monkeypatch.setattr(m, "detect_host", lambda **k: masked)
        assert fast_path(root) is True
        monkeypatch.setattr(m, "detect_host", lambda **k: swapped)
        assert fast_path(root) is False

    def test_an_update_failure_keeps_the_install_only_while_it_covers_the_card(
        self, monkeypatch, tmp_path, system, healthy_payload
    ):
        ada = host(system, ("89",))
        blackwell = host(system, ("120",))
        root = tmp_path / "llama"
        seed_install(root, ada, choice_for(ada))
        before = m.load_prebuilt_metadata(root)

        def outage(*a, **k):
            raise m.PrebuiltFallback("simulated release lookup outage")

        monkeypatch.setattr(m, "select_backend_install", outage)
        monkeypatch.setattr(m, "detect_host", lambda **k: ada)
        m.install_prebuilt(root, "latest", BUNDLE.repo, "")
        assert m.load_prebuilt_metadata(root) == before
        monkeypatch.setattr(m, "detect_host", lambda **k: blackwell)
        with pytest.raises(SystemExit) as exit_info:
            m.install_prebuilt(root, "latest", BUNDLE.repo, "")
        assert exit_info.value.code == m.EXIT_FALLBACK
