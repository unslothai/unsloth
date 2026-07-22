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


# ── Backend resolution ──
def test_resolve_backend_auto_and_validation(component):
    gpu_host = make_host(component, has_usable_nvidia = True)
    assert component.ops.resolve_backend(gpu_host, "auto", cpu_fallback = False) == "cuda"
    assert component.ops.resolve_backend(gpu_host, "auto", cpu_fallback = True) == "cpu"
    mac_host = make_host(
        component, os_token = "macos", arch_token = "arm64", is_macos = True, is_apple_silicon = True
    )
    assert component.ops.resolve_backend(mac_host, None, cpu_fallback = False) == "metal"
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
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "target")
    with pytest.raises(core.PrebuiltFallback):
        core.extract_archive(archive, tmp_path / "out")


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
    bin_dir = component.ops.runtime_bin_dir(install_dir, host)
    bin_dir.mkdir(parents = True)
    (bin_dir / component.ops.server_binary_name(host)).write_bytes(b"bin")
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
    component.ops.write_prebuilt_metadata(install_dir, selection)
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
