# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared descriptor-parameterized tests for studio/prebuilt_core.py.

Runs the component-agnostic core against BOTH shipped descriptors -- the real
whisper descriptor exported by install_whisper_prebuilt and a llama-flavored
descriptor built here the way a hypothetical third ggml-family component would
plug in (descriptor only, no installer module). Covers the selection matrix,
checksum fail-closed behavior, extraction guards, the resolver payload, and the
ops monkeypatch seam, so a new component gets this coverage for free.

The llama installer's shipped release-plan machinery is intentionally NOT
routed through the generic flow (its characterization suites pin it); the
llama descriptor here exercises the canonical dialect a future migration
would use, including the "no fallback backend -> report no prebuilt" policy.
"""

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

SPEC = importlib.util.spec_from_file_location(
    "studio_prebuilt_core", STUDIO_DIR / "prebuilt_core.py"
)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)

WSPEC = importlib.util.spec_from_file_location(
    "studio_install_whisper_prebuilt_for_core", STUDIO_DIR / "install_whisper_prebuilt.py"
)
assert WSPEC is not None and WSPEC.loader is not None
iwp = importlib.util.module_from_spec(WSPEC)
sys.modules[WSPEC.name] = iwp
WSPEC.loader.exec_module(iwp)


LLAMA_DESCRIPTOR = core.ComponentDescriptor(
    component = "llama.cpp",
    log_prefix = "llama-prebuilt",
    published_repo = "unslothai/llama.cpp",
    manifest_asset_name = "llama-prebuilt-manifest.json",
    sha256_asset_name = "llama-prebuilt-sha256.json",
    metadata_filename = "UNSLOTH_LLAMA_PREBUILT_INFO.json",
    user_agent = "unsloth-studio-llama-prebuilt",
    # A GPU-selection miss reports "no prebuilt" so the caller can fall back to
    # a source build instead of silently degrading to CPU.
    fallback_backend = None,
    server_binary_name = lambda host: "llama-server",
    runtime_bin_dir = lambda install_dir, host: install_dir / "build" / "bin",
)


class Component:
    """One descriptor under test plus the mutable namespace behind its ops."""

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.namespace = core.component_namespace(descriptor)
        self.namespace["log"] = lambda message: None  # keep test output quiet
        self.ops = core.ModuleOps(self.namespace)

    @property
    def falls_back_to_cpu(self):
        return self.descriptor.fallback_backend == "cpu"


@pytest.fixture(params = ["whisper", "llama"])
def component(request):
    if request.param == "whisper":
        return Component(iwp.DESCRIPTOR)
    return Component(LLAMA_DESCRIPTOR)


def make_host(
    component,
    *,
    os_token = "linux",
    arch_token = "x64",
    is_windows = False,
    is_macos = False,
    is_apple_silicon = False,
    has_usable_nvidia = False,
    has_rocm = False,
    rocm_gfx = None,
    compute_caps = (),
    driver_cuda_version = None,
    torch_runtime_line = None,
    macos_version = None,
):
    if component.descriptor is iwp.DESCRIPTOR:
        return iwp.HostInfo(
            system = {"linux": "Linux", "macos": "Darwin", "windows": "Windows"}[os_token],
            machine = "x86_64" if arch_token == "x64" else "arm64",
            whisper_os = os_token,
            whisper_arch = arch_token,
            archive_ext = ".zip" if is_windows else ".tar.gz",
            is_windows = is_windows,
            is_macos = is_macos,
            is_apple_silicon = is_apple_silicon,
            has_usable_nvidia = has_usable_nvidia,
            has_rocm = has_rocm,
            rocm_gfx = rocm_gfx,
            compute_caps = tuple(compute_caps),
            driver_cuda_version = driver_cuda_version,
            torch_runtime_line = torch_runtime_line,
            macos_version = macos_version,
        )
    # Descriptor-only component: the core default host_platform_tokens hook
    # reads .os_token/.arch_token off a plain host object.
    return SimpleNamespace(
        os_token = os_token,
        arch_token = arch_token,
        is_windows = is_windows,
        is_macos = is_macos,
        is_apple_silicon = is_apple_silicon,
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
        rocm_gfx = rocm_gfx,
        compute_caps = tuple(compute_caps),
        driver_cuda_version = driver_cuda_version,
        torch_runtime_line = torch_runtime_line,
        macos_version = macos_version,
    )


def artifact(
    os_ = "linux",
    arch = "x64",
    backend = "cpu",
    asset = "bundle.tar.gz",
    **extra,
):
    payload = {"os": os_, "arch": arch, "backend": backend, "asset": asset}
    payload.update(extra)
    return payload


def manifest_for(component, artifacts, **extra):
    payload = {
        "schema_version": 1,
        "component": extra.pop("component_name", component.descriptor.component),
        "upstream_tag": "v1.0.0",
        "source_commit": "a" * 40,
        "artifacts": artifacts,
    }
    payload.update(extra)
    return payload


# ── Manifest parsing ──
def test_parse_manifest_normalizes(component):
    manifest = component.ops.parse_manifest(
        manifest_for(component, [artifact(), "not-a-dict", {"os": "linux"}]), label = "m"
    )
    assert manifest["component"] == component.descriptor.component
    assert manifest["upstream_tag"] == "v1.0.0"
    # Non-dict entries and entries without an asset name are dropped.
    assert [a["asset"] for a in manifest["artifacts"]] == ["bundle.tar.gz"]


def test_parse_manifest_rejects_wrong_component(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_manifest(
            manifest_for(component, [artifact()], component_name = "other.cpp"), label = "m"
        )


def test_parse_manifest_rejects_unknown_schema(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_manifest(
            manifest_for(component, [artifact()], schema_version = 99), label = "m"
        )


def test_parse_manifest_rejects_non_object(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_manifest(["nope"], label = "m")
    # An object without an 'artifacts' list is rejected too.
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_manifest(
            {"schema_version": 1, "component": component.descriptor.component}, label = "m"
        )


# ── Selection matrix ──
def test_select_cpu_first_match(component):
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(backend = "cpu", asset = "first-cpu.tar.gz"),
                artifact(backend = "cpu", asset = "second-cpu.tar.gz"),
            ],
        ),
        label = "m",
    )
    host = make_host(component)
    chosen = component.ops.select_artifact(manifest, host, "cpu")
    assert chosen["asset"] == "first-cpu.tar.gz"


def test_select_respects_os_arch(component):
    manifest = component.ops.parse_manifest(
        manifest_for(component, [artifact(os_ = "windows", backend = "cpu")]), label = "m"
    )
    host = make_host(component)
    assert component.ops.select_artifact(manifest, host, "cpu") is None


def _cuda_manifest(component):
    return component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(
                    backend = "cuda",
                    asset = "cuda13-newer.tar.gz",
                    runtime_line = "cuda13",
                    coverage_class = "newer",
                    supported_sms = ["10.0", "12.0"],
                    min_sm = 100,
                    max_sm = 121,
                    rank = 1,
                ),
                artifact(
                    backend = "cuda",
                    asset = "cuda12-older.tar.gz",
                    runtime_line = "cuda12",
                    coverage_class = "older",
                    supported_sms = ["7.5", "8.6", "8.9"],
                    min_sm = 75,
                    max_sm = 89,
                    rank = 1,
                ),
                artifact(
                    backend = "cuda",
                    asset = "cuda12-portable.tar.gz",
                    runtime_line = "cuda12",
                    coverage_class = "portable",
                    supported_sms = ["7.5", "8.6", "8.9", "9.0", "10.0", "12.0"],
                    min_sm = 75,
                    max_sm = 121,
                    rank = 9,
                ),
                artifact(backend = "cpu", asset = "cpu.tar.gz"),
            ],
        ),
        label = "m",
    )


def _with_runtime_lines(component, lines):
    component.namespace["detected_cuda_runtime_lines"] = lambda *, is_windows: list(lines)


def test_select_cuda_blackwell_prefers_covering_highest_major(component):
    # A B200 (sm_100) with a CUDA 13 driver and both runtimes on disk must get
    # the cuda13 Blackwell bundle even though torch reports cuda12.
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
        torch_runtime_line = "cuda12",
    )
    chosen = component.ops.select_artifact(_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda13-newer.tar.gz"


def test_select_cuda_non_blackwell_honors_torch_line(component):
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("8.6",),
        driver_cuda_version = (13, 0),
        torch_runtime_line = "cuda12",
    )
    chosen = component.ops.select_artifact(_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda12-older.tar.gz"


def test_select_cuda_portable_is_per_line_fallback(component):
    # sm_90 is outside every targeted bundle's list; the portable bundle of the
    # usable line covers it and is chosen last.
    _with_runtime_lines(component, ["cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("9.0",),
        driver_cuda_version = (12, 4),
        torch_runtime_line = None,
    )
    chosen = component.ops.select_artifact(_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda12-portable.tar.gz"


def test_select_cuda_unknown_caps_only_portable(component):
    _with_runtime_lines(component, ["cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = (),
        driver_cuda_version = (12, 4),
    )
    chosen = component.ops.select_artifact(_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda12-portable.tar.gz"


def test_select_cuda_requires_on_disk_runtime(component):
    _with_runtime_lines(component, [])  # driver fine, no runtime libs on disk
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    assert component.ops.select_artifact(_cuda_manifest(component), host, "cuda") is None


def test_select_rocm_exact_gfx_match(component):
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(
                    backend = "rocm",
                    asset = "rocm-gfx110X.tar.gz",
                    gfx_target = "gfx110X",
                    mapped_targets = ["gfx1100", "gfx1101"],
                )
            ],
        ),
        label = "m",
    )
    host = make_host(component, has_rocm = True, rocm_gfx = "gfx1100")
    chosen = component.ops.select_artifact(manifest, host, "rocm")
    assert chosen["asset"] == "rocm-gfx110X.tar.gz"
    # In-family but unbuilt arch must not be served the family bundle.
    host_unbuilt = make_host(component, has_rocm = True, rocm_gfx = "gfx1102")
    assert component.ops.select_artifact(manifest, host_unbuilt, "rocm") is None


def test_fallback_policy_differs_per_descriptor(component):
    # No CUDA coverage: whisper degrades to the CPU asset of the same release,
    # the llama-flavored descriptor reports no prebuilt (source-build fallback).
    _with_runtime_lines(component, [])
    manifest = _cuda_manifest(component)
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    if component.falls_back_to_cpu:
        chosen, backend, used_fallback = component.ops.select_artifact_with_fallback(
            manifest, host, "cuda"
        )
        assert (chosen["asset"], backend, used_fallback) == ("cpu.tar.gz", "cpu", True)
    else:
        with pytest.raises(core.PrebuiltFallback):
            component.ops.select_artifact_with_fallback(manifest, host, "cuda")


# ── Full-release CUDA matrix (the seven real linux-x64 CUDA profiles + their
# SM coverage, from the live whisper release manifest; llama publishes the same
# shape). select_artifact must match the host's compute caps + driver.
_CUDA_PROFILES = {
    "cuda12-legacy": ("cuda12", "legacy", ["50", "52", "60", "61"]),
    "cuda12-older": ("cuda12", "older", ["70", "75", "80", "86", "89"]),
    "cuda12-newer": ("cuda12", "newer", ["86", "89", "90", "100", "103", "120"]),
    "cuda12-portable": (
        "cuda12",
        "portable",
        ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
    ),
    "cuda13-older": ("cuda13", "older", ["75", "80", "86", "89"]),
    "cuda13-newer": ("cuda13", "newer", ["86", "89", "90", "100", "103", "120"]),
    "cuda13-portable": ("cuda13", "portable", ["75", "80", "86", "89", "90", "100", "103", "120"]),
}


def _full_cuda_manifest(component):
    artifacts = [artifact(backend = "cpu", asset = "cpu.tar.gz")]
    for name, (line, cls, sms) in _CUDA_PROFILES.items():
        artifacts.append(
            artifact(
                backend = "cuda",
                asset = f"{name}.tar.gz",
                runtime_line = line,
                coverage_class = cls,
                supported_sms = sms,
                min_sm = int(sms[0]),
                max_sm = int(sms[-1]),
            )
        )
    return component.ops.parse_manifest(manifest_for(component, artifacts), label = "m")


@pytest.mark.parametrize(
    "caps,driver,expected",
    [
        (["10.0"], (13, 0), "cuda13-newer"),  # B200 sm_100: legacy/older rejected
        (["10.0"], (12, 8), "cuda12-newer"),  # B200 on a CUDA-12 driver
        (["9.0"], (13, 0), "cuda13-newer"),  # H100 sm_90
        (["8.9"], (13, 0), "cuda13-older"),  # 4090 sm_89: tightest covering (75-89)
        (["8.0"], (13, 0), "cuda13-older"),  # A100 sm_80
        (["7.5"], (13, 0), "cuda13-older"),  # T4 sm_75
        (["7.5"], (12, 4), "cuda12-older"),  # T4 on a CUDA-12 driver
        ([], (13, 0), "cuda13-portable"),  # unknown caps -> portable only
    ],
)
def test_select_cuda_full_release_matrix(component, caps, driver, expected):
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = tuple(caps),
        driver_cuda_version = driver,
    )
    chosen = component.ops.select_artifact(_full_cuda_manifest(component), host, "cuda")
    assert chosen is not None
    assert chosen["asset"] == f"{expected}.tar.gz"


def test_select_cuda_multi_gpu_requires_single_covering_artifact(component):
    # T4 (75) + B200 (100): only a portable bundle covers both.
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("7.5", "10.0"),
        driver_cuda_version = (13, 0),
    )
    chosen = component.ops.select_artifact(_full_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda13-portable.tar.gz"


def test_select_cuda_on_disk_runtime_gates_the_line(component):
    # B200 under a CUDA-13 driver, but only cuda12 runtime libs on disk (e.g.
    # torch-cuda12): the bundles don't ship libcudart/libcublas, so the cuda13
    # line is unusable and selection must fall to cuda12-newer.
    _with_runtime_lines(component, ["cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    chosen = component.ops.select_artifact(_full_cuda_manifest(component), host, "cuda")
    assert chosen["asset"] == "cuda12-newer.tar.gz"


def test_select_cuda_stable_under_manifest_shuffle(component):
    import random

    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    manifest = _full_cuda_manifest(component)
    picks = set()
    for seed in range(8):
        shuffled = dict(manifest)
        arts = list(manifest["artifacts"])
        random.Random(seed).shuffle(arts)
        shuffled["artifacts"] = arts
        picks.add(component.ops.select_artifact(shuffled, host, "cuda")["asset"])
    assert picks == {"cuda13-newer.tar.gz"}


def test_select_cuda_dotted_manifest_sms_are_normalized(component):
    # A manifest using "10.0"-style SMs must still cover a sm_100 host
    # (supported_sms are normalized to bare SM strings before matching).
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(
                    backend = "cuda",
                    asset = "cuda13-newer.tar.gz",
                    runtime_line = "cuda13",
                    coverage_class = "newer",
                    supported_sms = ["8.6", "8.9", "9.0", "10.0"],
                    min_sm = 86,
                    max_sm = 100,
                )
            ],
        ),
        label = "m",
    )
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    assert component.ops.select_artifact(manifest, host, "cuda")["asset"] == "cuda13-newer.tar.gz"


def test_select_cuda_targeted_artifact_missing_sm_metadata_rejected(component):
    # A targeted (non-portable) bundle without supported_sms/min/max coverage
    # metadata can never be proven to cover the host -> rejected, not guessed.
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(
                    backend = "cuda",
                    asset = "cuda13-x.tar.gz",
                    runtime_line = "cuda13",
                    coverage_class = "newer",
                )
            ],
        ),
        label = "m",
    )
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("9.0",),
        driver_cuda_version = (13, 0),
    )
    assert component.ops.select_artifact(manifest, host, "cuda") is None


def test_select_cuda_no_driver_version_none_then_fallback_policy(component):
    # nvidia-smi present but no parseable CUDA version -> no CUDA line -> None,
    # then the descriptor's fallback policy decides (cpu bundle vs no prebuilt).
    _with_runtime_lines(component, ["cuda13", "cuda12"])
    host = make_host(
        component,
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = None,
    )
    manifest = _full_cuda_manifest(component)
    assert component.ops.select_artifact(manifest, host, "cuda") is None
    if component.falls_back_to_cpu:
        chosen, backend, used_fallback = component.ops.select_artifact_with_fallback(
            manifest, host, "cuda"
        )
        assert (chosen["asset"], backend, used_fallback) == ("cpu.tar.gz", "cpu", True)
    else:
        with pytest.raises(core.PrebuiltFallback):
            component.ops.select_artifact_with_fallback(manifest, host, "cuda")


# ── ROCm family matrix (real gfx family -> mapped_targets shapes) ──
_ROCM_FAMILIES = [
    ("gfx103X", ["gfx1030", "gfx1031", "gfx1032"]),
    ("gfx110X", ["gfx1100", "gfx1101", "gfx1102"]),
    ("gfx120X", ["gfx1200", "gfx1201"]),
    ("gfx1151", ["gfx1151"]),
    ("gfx90a", ["gfx90a"]),
    ("gfx908", ["gfx908"]),
]


def _rocm_family_manifest(component):
    artifacts = [artifact(backend = "cpu", asset = "cpu.tar.gz")]
    for label, mapped in _ROCM_FAMILIES:
        artifacts.append(
            artifact(
                backend = "rocm",
                asset = f"rocm-{label}.tar.gz",
                gfx_target = label,
                mapped_targets = mapped,
            )
        )
    return component.ops.parse_manifest(manifest_for(component, artifacts), label = "m")


@pytest.mark.parametrize(
    "gfx,expected",
    [
        ("gfx1030", "gfx103X"),
        ("gfx1100", "gfx110X"),
        ("gfx1201", "gfx120X"),
        ("gfx1151", "gfx1151"),
        ("gfx90a", "gfx90a"),
        ("gfx908", "gfx908"),
        ("gfx110X", "gfx110X"),  # family token forwarded by the updater
    ],
)
def test_select_rocm_family_and_exact_matching(component, gfx, expected):
    host = make_host(component, has_rocm = True, rocm_gfx = gfx)
    chosen = component.ops.select_artifact(_rocm_family_manifest(component), host, "rocm")
    assert chosen["asset"] == f"rocm-{expected}.tar.gz"


def test_macos_min_os_gate(component):
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                artifact(
                    os_ = "macos",
                    arch = "arm64",
                    backend = "metal",
                    asset = "metal-new.tar.gz",
                    min_os = "macos-15.0",
                )
            ],
        ),
        label = "m",
    )
    old_host = make_host(
        component,
        os_token = "macos",
        arch_token = "arm64",
        is_macos = True,
        is_apple_silicon = True,
        macos_version = (14, 7),
    )
    new_host = make_host(
        component,
        os_token = "macos",
        arch_token = "arm64",
        is_macos = True,
        is_apple_silicon = True,
        macos_version = (15, 1),
    )
    assert component.ops.select_artifact(manifest, old_host, "metal") is None
    chosen = component.ops.select_artifact(manifest, new_host, "metal")
    assert chosen["asset"] == "metal-new.tar.gz"


def _metal_artifact(asset, min_os):
    return artifact(
        os_ = "macos",
        arch = "arm64",
        backend = "metal",
        asset = asset,
        min_os = min_os,
    )


def _arm_mac_host(component, macos_version):
    return make_host(
        component,
        os_token = "macos",
        arch_token = "arm64",
        is_macos = True,
        is_apple_silicon = True,
        macos_version = macos_version,
    )


def test_macos_min_os_filters_to_compatible_bundle(component):
    # host 14.0 can't load the 15.0 bundle; the 13.0 bundle is picked instead.
    manifest = component.ops.parse_manifest(
        manifest_for(
            component,
            [
                _metal_artifact("metal-new.tar.gz", "macos-15.0"),
                _metal_artifact("metal.tar.gz", "macos-13.0"),
            ],
        ),
        label = "m",
    )
    host = _arm_mac_host(component, (14, 0))
    assert component.ops.select_artifact(manifest, host, "metal")["asset"] == "metal.tar.gz"


def test_macos_min_os_unknown_host_version_keeps_artifact(component):
    # Unknown host macOS version -> defer to runtime validation, don't reject.
    manifest = component.ops.parse_manifest(
        manifest_for(component, [_metal_artifact("metal-new.tar.gz", "macos-15.0")]), label = "m"
    )
    host = _arm_mac_host(component, None)
    assert component.ops.select_artifact(manifest, host, "metal")["asset"] == "metal-new.tar.gz"


def test_macos_min_os_accepts_bare_version_format(component):
    # A bare "14.0" (no 'macos-' prefix) must still parse, for forward-compat.
    manifest = component.ops.parse_manifest(
        manifest_for(component, [_metal_artifact("metal.tar.gz", "14.0")]), label = "m"
    )
    host = _arm_mac_host(component, (13, 0))
    assert component.ops.select_artifact(manifest, host, "metal") is None  # 13.0 < 14.0


def test_macos_min_os_ok_helper_handles_prefix_and_bare(component):
    host14 = _arm_mac_host(component, (14, 0))
    # The live manifest format is 'macos-<ver>'; the prefix must be stripped.
    assert component.ops.macos_min_os_ok(host14, "macos-14.0") is True
    assert component.ops.macos_min_os_ok(host14, "macos-15.0") is False
    assert component.ops.macos_min_os_ok(host14, "13.3") is True  # bare also parses
    assert component.ops.macos_min_os_ok(host14, None) is True  # unknown -> defer
    host_unknown = _arm_mac_host(component, None)
    assert component.ops.macos_min_os_ok(host_unknown, "macos-15.0") is True


# ── Backend resolution ──
def test_resolve_backend_auto_and_validation(component):
    gpu_host = make_host(component, has_usable_nvidia = True)
    assert component.ops.resolve_backend(gpu_host, "auto", cpu_fallback = False) == "cuda"
    assert component.ops.resolve_backend(gpu_host, "auto", cpu_fallback = True) == "cpu"
    # --cpu-fallback wins over an explicit backend too.
    assert component.ops.resolve_backend(gpu_host, "cuda", cpu_fallback = True) == "cpu"
    # An explicit supported backend passes through untouched.
    assert component.ops.resolve_backend(gpu_host, "vulkan", cpu_fallback = False) == "vulkan"
    mac_host = make_host(
        component, os_token = "macos", arch_token = "arm64", is_macos = True, is_apple_silicon = True
    )
    assert component.ops.resolve_backend(mac_host, None, cpu_fallback = False) == "metal"
    # Intel mac has no Metal bundle in the P0 matrix -> cpu.
    intel_mac = make_host(component, os_token = "macos", arch_token = "x64", is_macos = True)
    assert component.ops.resolve_backend(intel_mac, "auto", cpu_fallback = False) == "cpu"
    rocm_host = make_host(component, has_rocm = True, rocm_gfx = "gfx1100")
    assert component.ops.resolve_backend(rocm_host, "auto", cpu_fallback = False) == "rocm"
    bare_host = make_host(component)
    assert component.ops.resolve_backend(bare_host, None, cpu_fallback = False) == "cpu"
    with pytest.raises(core.PrebuiltFallback):
        component.ops.resolve_backend(gpu_host, "tpu", cpu_fallback = False)


# ── Checksum index: fail closed ──
def _index_for(
    component,
    tag = "v1",
    artifacts = None,
):
    return {
        "schema_version": 1,
        "component": component.descriptor.component,
        "release_tag": tag,
        "artifacts": artifacts
        if artifacts is not None
        else {"bundle.tar.gz": {"sha256": "0" * 64}},
    }


def test_parse_release_checksums_valid(component):
    out = component.ops.parse_release_checksums("r", "v1", _index_for(component))
    assert out == {"bundle.tar.gz": "0" * 64}


@pytest.mark.parametrize(
    "mutation",
    [
        {"component": "other.cpp"},
        {"schema_version": 99},
        {"release_tag": "v2"},
        {"artifacts": {"bundle.tar.gz": {"sha256": "nope"}}},
        {"artifacts": "not-a-map"},
    ],
)
def test_parse_release_checksums_fails_closed(component, mutation):
    payload = _index_for(component)
    payload.update(mutation)
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_release_checksums("r", "v1", payload)


def test_parse_release_checksums_rejects_non_object(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.parse_release_checksums("r", "v1", ["not", "a", "dict"])


def test_expected_sha256_covered_asset_plain_lookup(component):
    assert component.ops.expected_sha256_for({"a.tar.gz": "0" * 64}, "a.tar.gz") == "0" * 64


def test_expected_sha256_missing_asset_fails_closed(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.expected_sha256_for({"a.tar.gz": "0" * 64}, "b.tar.gz")


def test_expected_sha256_manifest_disagreement_fails_closed(component):
    with pytest.raises(core.PrebuiltFallback):
        component.ops.expected_sha256_for(
            {"a.tar.gz": "0" * 64}, "a.tar.gz", manifest_sha256 = "1" * 64
        )
    assert (
        component.ops.expected_sha256_for(
            {"a.tar.gz": "0" * 64}, "a.tar.gz", manifest_sha256 = "0" * 64
        )
        == "0" * 64
    )


# ── Extraction guards ──
def test_extract_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(core.PrebuiltFallback):
        core.extract_archive(archive, tmp_path / "out")


def test_extract_archive_rejects_absolute_member(tmp_path):
    archive = tmp_path / "abs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo("/abs.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(core.PrebuiltFallback):
        core.extract_archive(archive, tmp_path / "out")


def test_extract_archive_rejects_zip_symlink(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "target")
    with pytest.raises(core.PrebuiltFallback, match = "zip archive contained a symlink entry"):
        core.extract_archive(archive, tmp_path / "out")


def test_extract_archive_rejects_absolute_zip_member(tmp_path):
    archive = tmp_path / "abs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/passwd", b"pwn")
    with pytest.raises(core.PrebuiltFallback):
        core.extract_archive(archive, tmp_path / "out")


def test_extract_archive_allows_safe_tar_symlink_chain(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    payload = b"shared-object"
    with tarfile.open(archive, "w:gz") as tar:
        versioned = tarfile.TarInfo("libllama.so.0.0.1")
        versioned.size = len(payload)
        tar.addfile(versioned, io.BytesIO(payload))
        soname = tarfile.TarInfo("libllama.so.0")
        soname.type = tarfile.SYMTYPE
        soname.linkname = "libllama.so.0.0.1"
        tar.addfile(soname)
        linker_name = tarfile.TarInfo("libllama.so")
        linker_name.type = tarfile.SYMTYPE
        linker_name.linkname = "libllama.so.0"
        tar.addfile(linker_name)
    destination = tmp_path / "extract"
    core.extract_archive(archive, destination)
    assert (destination / "libllama.so.0.0.1").read_bytes() == payload
    assert (destination / "libllama.so.0").is_symlink()
    assert (destination / "libllama.so").is_symlink()
    assert (destination / "libllama.so").resolve().read_bytes() == payload


def test_extract_archive_allows_safe_tar_hardlink(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    payload = b"quantize"
    with tarfile.open(archive, "w:gz") as tar:
        target = tarfile.TarInfo("llama-quantize")
        target.size = len(payload)
        tar.addfile(target, io.BytesIO(payload))
        hardlink = tarfile.TarInfo("llama-quantize-copy")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "llama-quantize"
        tar.addfile(hardlink)
    destination = tmp_path / "extract"
    core.extract_archive(archive, destination)
    assert (destination / "llama-quantize-copy").read_bytes() == payload
    assert not (destination / "llama-quantize-copy").is_symlink()


def test_extract_archive_rejects_absolute_tar_symlink_target(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/tmp/libllama.so.0"
        tar.addfile(entry)
    with pytest.raises(core.PrebuiltFallback, match = "archive link used an absolute target"):
        core.extract_archive(archive, tmp_path / "extract")


def test_extract_archive_rejects_escaping_tar_symlink_target(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../outside/libllama.so.0"
        tar.addfile(entry)
    with pytest.raises(core.PrebuiltFallback, match = "archive link escaped destination"):
        core.extract_archive(archive, tmp_path / "extract")


def test_extract_archive_rejects_unresolved_tar_symlink_target(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "libllama.so.0"
        tar.addfile(entry)
    with pytest.raises(core.PrebuiltFallback, match = "unresolved link entries"):
        core.extract_archive(archive, tmp_path / "extract")


def test_extract_archive_rejects_unknown_format(tmp_path):
    archive = tmp_path / "blob.xz"
    archive.write_bytes(b"data")
    with pytest.raises(core.PrebuiltFallback):
        core.extract_archive(archive, tmp_path / "out")


def test_restore_tar_exec_bits(tmp_path):
    payload = tmp_path / "server"
    payload.write_bytes(b"#!/bin/sh\n")
    payload.chmod(0o755)
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname = "bundle/server")
    out = tmp_path / "out"
    core.extract_archive(archive, out)
    extracted = out / "bundle" / "server"
    assert extracted.is_file()
    core.restore_tar_exec_bits(archive, out)
    assert extracted.stat().st_mode & 0o111


# ── Resolver payload ──
def _fake_release(component, artifacts):
    ns = component.namespace
    manifest = component.ops.parse_manifest(manifest_for(component, artifacts), label = "m")
    bundle = core.ReleaseBundle(
        repo = component.descriptor.published_repo,
        release_tag = "v1",
        manifest = manifest,
        asset_urls = {},
    )
    checksums = {str(a["asset"]): "0" * 64 for a in artifacts}
    ns["fetch_release_for_install"] = lambda repo, *, published_release_tag: (bundle, checksums)
    return bundle


def test_resolve_prebuilt_payload_keys(component):
    _fake_release(component, [artifact(backend = "cpu", asset = "cpu.tar.gz")])
    host = make_host(component)
    payload = component.ops.resolve_prebuilt(
        host,
        published_repo = component.descriptor.published_repo,
        published_release_tag = None,
        backend = "cpu",
        cpu_fallback = True,
    )
    assert payload == {
        "prebuilt_available": True,
        "repo": component.descriptor.published_repo,
        "release_tag": "v1",
        "upstream_tag": "v1.0.0",
        "backend": "cpu",
        "requested_backend": "cpu",
        "cpu_fallback": False,
        "asset": "cpu.tar.gz",
        "os": "linux",
        "arch": "x64",
        "runtime_line": None,
    }


def test_resolve_prebuilt_unavailable_payload(component):
    ns = component.namespace

    def boom(repo, *, published_release_tag):
        raise core.PrebuiltFallback("no release")

    ns["fetch_release_for_install"] = boom
    host = make_host(component)
    payload = component.ops.resolve_prebuilt(
        host,
        published_repo = component.descriptor.published_repo,
        published_release_tag = None,
        backend = "cpu",
        cpu_fallback = True,
    )
    assert payload == {"prebuilt_available": False, "repo": component.descriptor.published_repo}


def test_emit_resolver_output_formats(capsys):
    payload = {"prebuilt_available": True, "asset": "a.tar.gz"}
    core.emit_resolver_output(payload, output_format = "json")
    assert json.loads(capsys.readouterr().out) == payload
    core.emit_resolver_output(payload, output_format = "plain")
    assert capsys.readouterr().out.strip() == "a.tar.gz"
    core.emit_resolver_output({"prebuilt_available": False}, output_format = "plain")
    assert json.loads(capsys.readouterr().out) == {"prebuilt_available": False}


# ── Marker / fingerprint ──
def test_install_fingerprint_is_stable_and_sensitive(component):
    kwargs = dict(
        published_repo = component.descriptor.published_repo,
        release_tag = "v1",
        upstream_tag = "v1.0.0",
        source_commit = "a" * 40,
        asset = "cpu.tar.gz",
        asset_sha256 = "0" * 64,
        backend = "cpu",
        runtime_line = None,
        coverage = {},
    )
    first = core.compute_install_fingerprint(**kwargs)
    assert first == core.compute_install_fingerprint(**kwargs)
    changed = core.compute_install_fingerprint(**{**kwargs, "asset_sha256": "1" * 64})
    assert changed != first


def test_write_and_match_marker(component, tmp_path):
    host = make_host(component)
    install_dir = tmp_path / "install"
    manifest = component.ops.parse_manifest(
        manifest_for(component, [artifact(backend = "cpu", asset = "cpu.tar.gz")]), label = "m"
    )
    selection = component.ops.selection_from_artifact(
        published_repo = component.descriptor.published_repo,
        release_tag = "v1",
        manifest = manifest,
        artifact = manifest["artifacts"][0],
        backend = "cpu",
        asset_sha256 = "0" * 64,
    )
    # No marker on disk yet -> not a match.
    assert not component.ops.existing_install_matches(install_dir, host, selection)
    bin_dir = component.ops.runtime_bin_dir(install_dir, host)
    bin_dir.mkdir(parents = True)
    component.ops.write_prebuilt_metadata(install_dir, selection)
    # Marker alone is not enough: the server binary must exist too.
    assert not component.ops.existing_install_matches(install_dir, host, selection)
    (bin_dir / component.ops.server_binary_name(host)).write_bytes(b"bin")
    marker = json.loads((install_dir / component.descriptor.metadata_filename).read_text())
    assert marker["component"] == component.descriptor.component
    assert marker["install_fingerprint"] == selection.fingerprint()
    assert component.ops.existing_install_matches(install_dir, host, selection)
    # A different selection (new sha) must force a reinstall.
    other = core.InstallSelection(
        **{
            **selection.__dict__,
            "asset_sha256": "1" * 64,
        }
    )
    assert not component.ops.existing_install_matches(install_dir, host, other)


# ── Host/GPU token helpers (component-independent core functions) ──
# Value tables moved verbatim from the llama characterization suite; these are
# pure functions with no descriptor sensitivity, so they run unparameterized.
@pytest.mark.parametrize(
    "value,expected",
    [
        ("8.6", "86"),
        ("07.05", "75"),
        ("75", "75"),
        (86, "86"),
        ("", None),
        ("  ", None),
        ("x.y", None),
        ("8.6.0", None),
        ("9.0", "90"),
    ],
)
def test_normalize_compute_cap(value, expected):
    assert core.normalize_compute_cap(value) == expected


@pytest.mark.parametrize(
    "values,expected",
    [
        (["8.6", "86", "8.6"], ["86"]),  # deduplication
        (["9.0", "7.5", "8.6"], ["75", "86", "90"]),  # numeric sort
        (["8.6", "bad", "", "7.5"], ["75", "86"]),  # drops invalid
        ([], []),
    ],
)
def test_normalize_compute_caps(values, expected):
    assert core.normalize_compute_caps(values) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", []),
        ("-1", []),
        ("0", ["0"]),
        ("0,1,2", ["0", "1", "2"]),
        (" 0 , 1 ", ["0", "1"]),
    ],
)
def test_parse_cuda_visible_devices(value, expected):
    assert core.parse_cuda_visible_devices(value) == expected


@pytest.mark.parametrize(
    "visible,expected",
    [
        (["0", "1", "2"], True),
        (["GPU-abc123"], True),
        (None, False),
        ([], False),
        (["0", "MIG-device"], False),
    ],
)
def test_supports_explicit_visible_device_matching(visible, expected):
    assert core.supports_explicit_visible_device_matching(visible) is expected


_GPU_ROWS = [
    ("0", "GPU-aaa", "8.6"),
    ("1", "GPU-bbb", "7.5"),
    ("2", "GPU-ccc", "8.9"),
]


@pytest.mark.parametrize(
    "visible,expected_indices",
    [
        (None, [0, 1, 2]),  # no filter returns all
        ([], []),
        (["0", "2"], [0, 2]),  # filter by index
        (["gpu-bbb"], [1]),  # UUID match is case insensitive
        (["0", "0"], [0]),  # same device requested twice is deduplicated
        (["99"], []),  # unknown token matches nothing
    ],
)
def test_select_visible_gpu_rows(visible, expected_indices):
    expected = [_GPU_ROWS[i] for i in expected_indices]
    assert core.select_visible_gpu_rows(_GPU_ROWS, visible) == expected


@pytest.mark.parametrize(
    "driver,expected",
    [
        (None, []),
        ((11, 8), []),
        ((12, 4), ["cuda12"]),
        ((13, 0), ["cuda13", "cuda12"]),
        ((14, 0), ["cuda14", "cuda13", "cuda12"]),  # future major derives lines
    ],
)
def test_compatible_linux_runtime_lines(driver, expected):
    host = SimpleNamespace(driver_cuda_version = driver)
    assert core.compatible_linux_runtime_lines(host) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12.6", "cuda12"),
        ("13.0", "cuda13"),
        ("11.8", None),
        (None, None),
        ("", None),
    ],
)
def test_runtime_line_from_cuda_version(value, expected):
    assert core.runtime_line_from_cuda_version(value) == expected


def _caps_host(caps):
    return SimpleNamespace(compute_caps = list(caps))


def test_host_is_blackwell_includes_datacenter_parts():
    assert core.host_is_blackwell(_caps_host(["10.0"])) is True  # B200 sm_100
    assert core.host_is_blackwell(_caps_host(["10.3"])) is True  # B300 sm_103
    assert core.host_is_blackwell(_caps_host(["12.0"])) is True  # RTX 50 sm_120
    assert core.host_is_blackwell(_caps_host(["12.1"])) is True  # DGX Spark sm_121
    assert core.host_is_blackwell(_caps_host(["9.0"])) is False  # Hopper
    assert core.host_is_blackwell(_caps_host(["8.0"])) is False  # Ampere
    assert core.host_is_blackwell(_caps_host(["9.0", "10.0"])) is True  # highest cap wins


def test_blackwell_min_toolkit_is_sm_aware():
    # Family floor is 12.8; sm_103/sm_121 (no native target before 12.9) lift it.
    f = core.blackwell_min_toolkit_for_host
    assert f(_caps_host(["10.0"])) == (12, 8)  # B200
    assert f(_caps_host(["12.0"])) == (12, 8)  # RTX 50
    assert f(_caps_host(["10.3"])) == (12, 9)  # B300
    assert f(_caps_host(["12.1"])) == (12, 9)  # DGX Spark
    assert f(_caps_host(["10.0", "10.3"])) == (12, 9)  # max across SMs wins


# ── The ops seam ──
def test_module_ops_prefers_module_globals_over_core_defaults(component):
    ns = dict(component.namespace)
    calls = []

    def fake_download_file(url, destination):
        calls.append(url)
        destination.write_bytes(b"data")

    ns["download_file"] = fake_download_file
    ops = core.ModuleOps(ns)
    assert ops.download_file is fake_download_file
    # Core defaults still resolve (and come back bound) for everything else.
    assert callable(ops.fetch_json)
    with pytest.raises(AttributeError):
        _ = ops.does_not_exist_anywhere
