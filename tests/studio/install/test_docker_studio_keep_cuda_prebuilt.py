# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Docker Studio layer must keep the base image's CUDA llama.cpp bundle.

docker/Dockerfile.studio symlinks Studio's llama.cpp dir at the base's baked CUDA bundle
and then runs install.sh --local. No GPU is visible during an image build, so setup.sh's
hardware detection resolves the CPU bundle for the host arch and installs it over the CUDA
one: the published unsloth/unsloth:latest carries whiteouts for libggml-cuda.so on BOTH
arm64 and amd64, and its UNSLOTH_PREBUILT_INFO.json reads backend: cpu.

Part one pins that root cause on both arches with the real selector. Part two drives the
fix, setup.sh's _keep_installed_gpu_prebuilt, against real markers on a stubbed aarch64
host with no working nvidia-smi. Part three pins the wiring that turns the knob on.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
DOCKERFILE_STUDIO = PACKAGE_ROOT / "docker" / "Dockerfile.studio"
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"

SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

HostInfo = INSTALL_LLAMA_PREBUILT.HostInfo
PublishedLlamaArtifact = INSTALL_LLAMA_PREBUILT.PublishedLlamaArtifact
PublishedReleaseBundle = INSTALL_LLAMA_PREBUILT.PublishedReleaseBundle
_linux_published_attempts = INSTALL_LLAMA_PREBUILT._linux_published_attempts

RELEASE_TAG = "b10840-mix-d5c17a0"
LLAMA_TAG = "b10840"
FORK = "unslothai/llama.cpp"


def _usable_bash():
    exe = shutil.which("bash")
    if exe is None:
        return None
    try:
        probe = subprocess.run(
            [exe, "-c", "printf ok"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            errors = "replace",
            timeout = 60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if probe.stdout.strip() == "ok" else None


BASH = _usable_bash()
requires_bash = pytest.mark.skipif(BASH is None, reason = "a working bash is required")


# ── part one: why the CUDA bundle is replaced, on both arches ──


def _host(machine):
    return HostInfo(
        system = "Linux",
        machine = machine,
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = machine == "x86_64",
        is_arm64 = machine == "aarch64",
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )


def _artifact(asset_name, install_kind, **overrides):
    defaults = dict(
        asset_name = asset_name,
        install_kind = install_kind,
        runtime_line = None,
        coverage_class = None,
        supported_sms = [],
        min_sm = None,
        max_sm = None,
        bundle_profile = None,
        rank = 100,
    )
    defaults.update(overrides)
    return PublishedLlamaArtifact(**defaults)


def _release(artifacts):
    return PublishedReleaseBundle(
        repo = FORK,
        release_tag = RELEASE_TAG,
        upstream_tag = RELEASE_TAG,
        assets = {a.asset_name: f"https://example.invalid/{a.asset_name}" for a in artifacts},
        artifacts = artifacts,
    )


# (arch, the CUDA bundle the base image bakes, the CPU bundle detection lands on)
_ARCH_CASES = [
    (
        "aarch64",
        _artifact(
            f"app-{RELEASE_TAG}-linux-arm64-cuda13-portable.tar.gz",
            "linux-arm64-cuda",
            runtime_line = "cuda13",
            coverage_class = "portable",
            bundle_profile = "cuda13-portable",
            supported_sms = ["90", "100", "103", "120", "121"],
            min_sm = 90,
            max_sm = 121,
            rank = 60,
        ),
        _artifact(f"app-{RELEASE_TAG}-linux-arm64-cpu.tar.gz", "linux-arm64"),
    ),
    (
        "x86_64",
        _artifact(
            f"app-{RELEASE_TAG}-linux-x64-cuda12-portable.tar.gz",
            "linux-cuda",
            runtime_line = "cuda12",
            coverage_class = "portable",
            bundle_profile = "cuda12-portable",
            supported_sms = ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
            min_sm = 70,
            max_sm = 120,
            rank = 60,
        ),
        _artifact(f"app-{RELEASE_TAG}-linux-x64-cpu.tar.gz", "linux-cpu"),
    ),
]


@pytest.mark.parametrize(
    ("machine", "cuda_artifact", "cpu_artifact"), _ARCH_CASES, ids = ["arm64", "amd64"]
)
def test_a_gpuless_build_host_resolves_the_cpu_bundle(machine, cuda_artifact, cpu_artifact):
    """The defect: with no GPU visible the selector picks CPU even though CUDA is published.

    This is what install.sh --local runs inside the Docker build, and it is why the CUDA
    bundle the base baked is overwritten. amd64 is NOT exempt.
    """
    attempts = _linux_published_attempts(_host(machine), _release([cuda_artifact, cpu_artifact]))
    assert [choice.name for choice in attempts] == [cpu_artifact.asset_name]
    assert [choice.install_kind for choice in attempts] == [cpu_artifact.install_kind]


# ── part two: the fix, driven out of setup.sh ──

_BASE_IMAGE_MARKER = {
    "upstream_tag": RELEASE_TAG,
    "source_repo": FORK,
    "platform": "linux-arm64-cuda",
    "bundle_profile": "cuda13-portable",
    "runtime_line": "cuda13",
    "coverage_class": "portable",
    "supported_sms": ["90", "100", "103", "120", "121"],
    "tag": LLAMA_TAG,
    "release_tag": RELEASE_TAG,
    "published_repo": FORK,
}

_INSTALLER_CUDA_MARKER = {
    "requested_tag": RELEASE_TAG,
    "tag": LLAMA_TAG,
    "release_tag": RELEASE_TAG,
    "published_repo": FORK,
    "asset": f"app-{RELEASE_TAG}-linux-arm64-cuda13-portable.tar.gz",
    "force_cpu": False,
    "backend": "cuda",
}

_SHIPPED_CPU_MARKER = {
    "requested_tag": RELEASE_TAG,
    "tag": LLAMA_TAG,
    "release_tag": RELEASE_TAG,
    "published_repo": FORK,
    "asset": f"app-{RELEASE_TAG}-linux-arm64-cpu.tar.gz",
    "force_cpu": False,
    "backend": "cpu",
    "supported_sms": [],
}


OTHER_MIX = "b10840-mix-0000000"


def _marker(**overrides):
    payload = dict(_BASE_IMAGE_MARKER)
    payload.update(overrides)
    return payload


_SH_HARNESS = """
set -u
_FN_FILE="$1"
_INSTALL_DIR="$2"
_REQUESTED_TAG="$3"
_REPO="$4"
. "$_FN_FILE"
if _keep_installed_gpu_prebuilt "$_INSTALL_DIR" "$_REQUESTED_TAG" "$_REPO" "${UNSLOTH_LLAMA_RELEASE_TAG:-}"; then
    printf 'KEEP'
else
    printf 'REPLACE'
fi
"""


def _sliced_sh_functions(tmp_path):
    """_has_local_llama_server + _keep_installed_gpu_prebuilt, sliced out of setup.sh.

    setup.sh runs install steps at load, so the functions are sliced rather than sourced.
    """
    text = SETUP_SH.read_text(encoding = "utf-8")
    body = ""
    for name in ("_has_local_llama_server() {", "_keep_installed_gpu_prebuilt() {"):
        start = text.index(name)
        end = text.index("\n}\n", start) + len("\n}\n")
        body += text[start:end] + "\n"
    assert "UNSLOTH_LLAMA_KEEP_PREBUILT" in body, "sliced the wrong block out of setup.sh"
    assert body.count("<<'PY'") == 1, "expected exactly one heredoc in the sliced block"
    path = tmp_path / "keep_prebuilt_fn.sh"
    path.write_text(body, encoding = "utf-8")
    return path


def _stub_bin(tmp_path):
    """A GPU-less aarch64 build host: uname says aarch64 and nvidia-smi cannot answer."""
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir(exist_ok = True)
    python_shim = stub_dir / "python"
    python_shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding = "utf-8")
    (stub_dir / "uname").write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ]; then echo aarch64; else echo Linux; fi\n',
        encoding = "utf-8",
    )
    (stub_dir / "nvidia-smi").write_text(
        '#!/bin/sh\necho "NVIDIA-SMI has failed because no NVIDIA driver is running." >&2\n'
        "exit 9\n",
        encoding = "utf-8",
    )
    for entry in stub_dir.iterdir():
        entry.chmod(0o755)
    return stub_dir


def _run_keep_decision(
    tmp_path,
    *,
    marker,
    server,
    requested_tag,
    repo = FORK,
    env = None,
):
    install_dir = tmp_path / "llama.cpp"
    (install_dir / "build" / "bin").mkdir(parents = True, exist_ok = True)
    if marker is not None:
        (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps(marker), encoding = "utf-8"
        )
    if server:
        for path in (install_dir / "llama-server", install_dir / "build" / "bin" / "llama-server"):
            path.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
            path.chmod(0o755)
    script = tmp_path / "drive.sh"
    script.write_text(_SH_HARNESS, encoding = "utf-8")
    stub_dir = _stub_bin(tmp_path)
    run_env = dict(os.environ)
    run_env.pop("UNSLOTH_LLAMA_KEEP_PREBUILT", None)
    run_env.pop("UNSLOTH_LLAMA_RELEASE_TAG", None)
    run_env["PATH"] = f"{stub_dir}{os.pathsep}{run_env.get('PATH', '')}"
    run_env.update(env or {})
    proc = subprocess.run(
        [
            BASH,
            str(script),
            str(_sliced_sh_functions(tmp_path)),
            str(install_dir),
            requested_tag,
            repo,
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = run_env,
        timeout = 120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


ON = {"UNSLOTH_LLAMA_KEEP_PREBUILT": "1"}

# (id, marker, llama-server present, requested tag, extra env, expected verdict)
_KEEP_CASES = [
    ("base_image_cuda_marker_pinned_tag", _BASE_IMAGE_MARKER, True, RELEASE_TAG, ON, "KEEP"),
    ("base_image_cuda_marker_base_build", _BASE_IMAGE_MARKER, True, LLAMA_TAG, ON, "KEEP"),
    ("base_image_cuda_marker_latest", _BASE_IMAGE_MARKER, True, "latest", ON, "KEEP"),
    ("installer_written_cuda_marker", _INSTALLER_CUDA_MARKER, True, RELEASE_TAG, ON, "KEEP"),
    (
        "knob_true",
        _BASE_IMAGE_MARKER,
        True,
        RELEASE_TAG,
        {"UNSLOTH_LLAMA_KEEP_PREBUILT": "true"},
        "KEEP",
    ),
    ("rocm_bundle", _marker(platform = "linux-rocm"), True, RELEASE_TAG, ON, "KEEP"),
    # a full mix pin names one bundle: only that exact mix may be kept
    ("full_mix_pin_exact_match", _BASE_IMAGE_MARKER, True, RELEASE_TAG, ON, "KEEP"),
    ("full_mix_pin_other_mix_same_base", _BASE_IMAGE_MARKER, True, OTHER_MIX, ON, "REPLACE"),
    (
        "full_mix_pin_marker_holds_other_mix",
        _marker(release_tag = OTHER_MIX, upstream_tag = OTHER_MIX),
        True,
        RELEASE_TAG,
        ON,
        "REPLACE",
    ),
    # a bare base build pin still accepts any mix cut from that build
    (
        "bare_base_pin_accepts_any_mix",
        _marker(release_tag = OTHER_MIX, upstream_tag = OTHER_MIX),
        True,
        LLAMA_TAG,
        ON,
        "KEEP",
    ),
    # UNSLOTH_LLAMA_RELEASE_TAG is checked against the marker's release_tag on its own
    (
        "release_tag_pin_match",
        _BASE_IMAGE_MARKER,
        True,
        LLAMA_TAG,
        {**ON, "UNSLOTH_LLAMA_RELEASE_TAG": RELEASE_TAG},
        "KEEP",
    ),
    (
        "release_tag_pin_mismatch",
        _BASE_IMAGE_MARKER,
        True,
        LLAMA_TAG,
        {**ON, "UNSLOTH_LLAMA_RELEASE_TAG": OTHER_MIX},
        "REPLACE",
    ),
    (
        "release_tag_pin_but_marker_records_none",
        _marker(release_tag = None),
        True,
        LLAMA_TAG,
        {**ON, "UNSLOTH_LLAMA_RELEASE_TAG": RELEASE_TAG},
        "REPLACE",
    ),
    # the shipped defect: a CPU bundle must still be replaced
    ("shipped_cpu_marker", _SHIPPED_CPU_MARKER, True, RELEASE_TAG, ON, "REPLACE"),
    (
        "cpu_platform_no_backend_key",
        _marker(platform = "linux-arm64"),
        True,
        RELEASE_TAG,
        ON,
        "REPLACE",
    ),
    ("deliberate_force_cpu", _marker(force_cpu = True), True, RELEASE_TAG, ON, "REPLACE"),
    # stale trees must still be replaced
    (
        "stale_release_tag",
        _marker(tag = "b10700", release_tag = "b10700-mix-aaaaaaa", upstream_tag = "b10700-mix-aaaaaaa"),
        True,
        RELEASE_TAG,
        ON,
        "REPLACE",
    ),
    (
        "other_fork",
        _marker(published_repo = "someone-else/llama.cpp"),
        True,
        RELEASE_TAG,
        ON,
        "REPLACE",
    ),
    ("no_marker_at_all", None, True, RELEASE_TAG, ON, "REPLACE"),
    ("marker_is_not_json", "not json", True, RELEASE_TAG, ON, "REPLACE"),
    ("no_llama_server_binary", _BASE_IMAGE_MARKER, False, RELEASE_TAG, ON, "REPLACE"),
    # the knob is opt-in, and an explicit backend request still wins
    ("knob_unset", _BASE_IMAGE_MARKER, True, RELEASE_TAG, {}, "REPLACE"),
    (
        "knob_zero",
        _BASE_IMAGE_MARKER,
        True,
        RELEASE_TAG,
        {"UNSLOTH_LLAMA_KEEP_PREBUILT": "0"},
        "REPLACE",
    ),
    (
        "explicit_backend_request",
        _BASE_IMAGE_MARKER,
        True,
        RELEASE_TAG,
        {**ON, "_explicit_llama_source_backend": "vulkan"},
        "REPLACE",
    ),
]


@requires_bash
@pytest.mark.parametrize(
    ("marker", "server", "requested_tag", "env", "expected"),
    [case[1:] for case in _KEEP_CASES],
    ids = [case[0] for case in _KEEP_CASES],
)
def test_keep_decision(tmp_path, marker, server, requested_tag, env, expected):
    if marker == "not json":
        install_dir = tmp_path / "llama.cpp"
        install_dir.mkdir(parents = True, exist_ok = True)
        (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text("{ broken", encoding = "utf-8")
        marker = None
    verdict = _run_keep_decision(
        tmp_path, marker = marker, server = server, requested_tag = requested_tag, env = env
    )
    assert verdict == expected


# ── part three: the wiring that turns the knob on ──


def test_setup_sh_keeps_the_bundle_instead_of_installing_a_prebuilt():
    text = SETUP_SH.read_text(encoding = "utf-8")
    skip = text.index('elif [ "${_SKIP_PREBUILT_INSTALL:-false}" = true ]; then')
    keep = text.index('elif _keep_installed_gpu_prebuilt "$LLAMA_CPP_DIR"')
    install = text.index('    substep "installing prebuilt llama.cpp..."')
    assert skip < keep < install, (
        "the keep branch must sit after the local-dir / force-compile / PR branches "
        "and before the prebuilt install, or it would hijack an explicit request"
    )
    assert "_LLAMA_KEEP_PREBUILT_ACTIVE=true" in text
    call = text[keep : text.index("\n", keep)]
    assert (
        "${UNSLOTH_LLAMA_RELEASE_TAG:-}" in call
    ), "an explicit published release pin must reach the keep decision"


def test_the_arm64_cpu_last_resort_cannot_undo_a_kept_bundle():
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("# ── arm64 Linux GPU: CPU prebuilt as a last resort ──")
    block = text[start : text.index("_ARM64_CPU_CMD=(", start)]
    assert '[ "$_LLAMA_KEEP_PREBUILT_ACTIVE" != true ]' in block, (
        "the arm64 CPU last resort must not install app-<tag>-linux-arm64-cpu.tar.gz "
        "over a bundle the keep branch deliberately preserved"
    )


def test_dockerfile_studio_sets_the_knob_and_asserts_the_cuda_backend():
    text = DOCKERFILE_STUDIO.read_text(encoding = "utf-8")
    knob = text.index("UNSLOTH_LLAMA_KEEP_PREBUILT=1 \\")
    install = text.index("bash install.sh --local")
    assert knob < install, "the knob must be in install.sh's environment"
    assert "/opt/unsloth/llama.cpp/libggml-cuda.so" in text
    assert "/opt/unsloth/llama.cpp/build/bin/libggml-cuda.so" in text
    assertion = text.index("/opt/unsloth/llama.cpp/libggml-cuda.so")
    assert install < assertion, "the CUDA backend assertion must run AFTER install.sh"
    assert (
        "exit 1" in text[assertion : assertion + 600]
    ), "a missing CUDA backend library must fail the build, not just print"
