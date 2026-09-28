# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh must not replace a working GPU llama.cpp prebuilt with a CPU source build (#9255).

A prebuilt update that fails (network, a GitHub limit, a bad release) restores the existing
install and then source-builds. Without nvcc that build is CPU-only, and the swap replaced
the restored CUDA prebuilt for good while the installer reported "built". Part one drives
the keep decision, sliced out of setup.sh, against real markers. Part two pins the wiring:
the decision runs before the compile and again at the swap, and the footer names the outcome.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
SETUP_TEXT = SETUP_SH.read_text(encoding = "utf-8")

BASH = shutil.which("bash")
requires_bash = pytest.mark.skipif(BASH is None, reason = "a working bash is required")

_FUNCTIONS = (
    "_has_local_llama_server() {",
    "_installed_prebuilt_ref_matches() {",
    "_installed_prebuilt_backend() {",
    "_gpu_prebuilt_to_keep_over_cpu_build() {",
)

_HARNESS = """
set -u
. "$1"
# Stubbed: the real ones read this host's /sys/class/drm and start llama-server.
_setup_has_intel_gpu() { [ "${_setup_intel_gpu:-false}" = true ]; }
_installed_prebuilt_runs() { [ "${_setup_prebuilt_runs:-true}" = true ]; }
if _kept="$(_gpu_prebuilt_to_keep_over_cpu_build "$2")"; then
    printf 'KEEP %s' "$_kept"
else
    printf 'REPLACE'
fi
"""


def _sliced_functions(tmp_path):
    """The three helpers, sliced out: setup.sh runs install steps at load."""
    body = ""
    for name in _FUNCTIONS:
        start = SETUP_TEXT.index(name)
        end = SETUP_TEXT.index("\n}\n", start) + len("\n}\n")
        body += SETUP_TEXT[start:end] + "\n"
    path = tmp_path / "keep_fns.sh"
    path.write_text(body, encoding = "utf-8")
    return path


def _install(
    tmp_path,
    marker,
    *,
    server = True,
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir(exist_ok = True)
    if server:
        exe = install_dir / "llama-server"
        exe.write_text("#!/bin/sh\n", encoding = "utf-8")
        exe.chmod(0o755)
    if marker is not None:
        text = marker if isinstance(marker, str) else json.dumps(marker)
        (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(text, encoding = "utf-8")
    return install_dir


def _decide(tmp_path, install_dir, **env):
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(exist_ok = True)
    shim = stub_bin / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding = "utf-8")
    shim.chmod(0o755)
    run_env = {
        **os.environ,
        "PATH": f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "_LLAMA_FORCE_COMPILE": "0",
        "_LLAMA_PR": "",
        # The pin comparison imports install_llama_prebuilt.py from beside setup.sh.
        "SCRIPT_DIR": str(SETUP_SH.parent),
        "_setup_nvidia_physical": "false",
        "_setup_amd_detected": "false",
    }
    run_env.update(env)
    result = subprocess.run(
        [BASH, "-c", _HARNESS, "harness", str(_sliced_functions(tmp_path)), str(install_dir)],
        env = run_env,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ── part one: the keep decision ──


@requires_bash
class TestTheKeepDecision:
    def test_a_cuda_prebuilt_on_an_nvidia_host_is_kept(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": "cuda"})
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "KEEP cuda"

    @pytest.mark.parametrize("backend", ["rocm", "vulkan"])
    def test_an_amd_prebuilt_on_an_amd_host_is_kept(self, tmp_path, backend):
        install_dir = _install(tmp_path, {"backend": backend})
        assert _decide(tmp_path, install_dir, _setup_amd_detected = "true") == f"KEEP {backend}"

    def test_a_rocm_prebuilt_needs_the_amd_gpu_not_just_any_gpu(self, tmp_path):
        # An AMD-to-NVIDIA swap: the ROCm binary cannot run here, so the CPU build wins.
        install_dir = _install(tmp_path, {"backend": "rocm"})
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"

    def test_a_vulkan_prebuilt_is_kept_for_every_vendor(self, tmp_path):
        # The Intel-only host is the one the Vulkan route exists for.
        install_dir = _install(tmp_path, {"backend": "vulkan"})
        assert _decide(tmp_path, install_dir, _setup_intel_gpu = "true") == "KEEP vulkan"
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "KEEP vulkan"
        assert _decide(tmp_path, install_dir) == "REPLACE"

    def test_the_marker_backend_is_read_case_insensitively(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": " CUDA "})
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "KEEP cuda"

    def test_a_cuda_prebuilt_is_replaced_once_the_gpu_is_gone(self, tmp_path):
        # The GPU the marker names left the machine: a CPU build is the honest install now.
        install_dir = _install(tmp_path, {"backend": "cuda"})
        assert _decide(tmp_path, install_dir) == "REPLACE"
        assert _decide(tmp_path, install_dir, _setup_amd_detected = "true") == "REPLACE"

    def test_an_explicit_version_pin_is_not_satisfied_by_the_old_install(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": "cuda"})
        nvidia = {"_setup_nvidia_physical": "true"}
        assert _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "b7000", **nvidia) == "REPLACE"
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_RELEASE_TAG = "b7000-mix", **nvidia)
            == "REPLACE"
        )
        assert _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "latest", **nvidia) == "KEEP cuda"

    def test_a_version_pin_the_old_install_already_satisfies_keeps_it(self, tmp_path):
        install_dir = _install(
            tmp_path, {"backend": "cuda", "tag": "b8508", "release_tag": "b8508-mix"}
        )
        nvidia = {"_setup_nvidia_physical": "true"}
        assert _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "b8508", **nvidia) == "KEEP cuda"
        assert _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "b8509", **nvidia) == "REPLACE"
        # The installer's own matching: a short commit pin names the recorded full commit.
        install_dir = _install(
            tmp_path,
            {
                "backend": "cuda",
                "tag": "0123456789abcdef0123456789abcdef01234567",
                "release_tag": "b8508-mix",
            },
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "0123456789ab", **nvidia)
            == "KEEP cuda"
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "fedcba987654", **nvidia) == "REPLACE"
        )
        # The marker writer records a commit pin in source_commit beside the upstream build tag.
        install_dir = _install(
            tmp_path,
            {
                "backend": "cuda",
                "tag": "b8508",
                "release_tag": "b8508-mix",
                "requested_source_ref": "0123456789abcdef0123456789abcdef01234567",
                "resolved_source_ref": "0123456789abcdef0123456789abcdef01234567",
                "source_commit": "0123456789abcdef0123456789abcdef01234567",
            },
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "0123456789ab", **nvidia)
            == "KEEP cuda"
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_TAG = "fedcba987654", **nvidia) == "REPLACE"
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_RELEASE_TAG = "b8508-mix", **nvidia)
            == "KEEP cuda"
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_RELEASE_TAG = "b8509-mix", **nvidia)
            == "REPLACE"
        )
        # A published release tag is a name: a hex-looking prefix of the recorded one is not it.
        install_dir = _install(
            tmp_path, {"backend": "cuda", "release_tag": "0123456789abcdef0123456789abcdef01234567"}
        )
        assert (
            _decide(tmp_path, install_dir, UNSLOTH_LLAMA_RELEASE_TAG = "0123456789ab", **nvidia)
            == "REPLACE"
        )

    def test_a_cpu_prebuilt_is_not_worth_keeping(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": "cpu"})
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"

    @pytest.mark.parametrize(
        "marker",
        [None, "{not json", "[1, 2]", {"asset": "x"}, {"backend": None}, {"backend": 3}],
        ids = ["absent", "malformed", "not-a-dict", "no-backend", "null", "non-string"],
    )
    def test_an_unreadable_marker_does_not_keep(self, tmp_path, marker):
        install_dir = _install(tmp_path, marker)
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"

    @pytest.mark.parametrize(
        ("marker", "backend"),
        [
            (
                {"llama_backend": "vulkan", "asset": "llama-b7001-bin-ubuntu-vulkan-x64.tar.gz"},
                "vulkan",
            ),
            ({"llama_backend": "hip", "asset": "app-b9001-linux-x64-rocm-gfx110X.tar.gz"}, "rocm"),
            ({"asset": "app-b6210-linux-x64-cuda12.tar.gz"}, "cuda"),
            ({"asset": "app-b9001-linux-x64-rocm-gfx110X.tar.gz"}, "rocm"),
            ({"llama_backend": "auto", "asset": "app-b1-linux-x64-cuda13-newer.tar.gz"}, "cuda"),
        ],
        ids = ["llama_backend", "hip-spelling", "asset-cuda", "asset-rocm", "auto-request"],
    )
    def test_a_legacy_marker_names_its_backend_elsewhere(self, tmp_path, marker, backend):
        # Shapes from before #8520 (tests/studio/install/test_keep_install_backcompat_9979.py).
        install_dir = _install(tmp_path, marker)
        both = {"_setup_nvidia_physical": "true", "_setup_amd_detected": "true"}
        assert _decide(tmp_path, install_dir, **both) == f"KEEP {backend}"

    def test_a_legacy_cpu_marker_is_replaced(self, tmp_path):
        install_dir = _install(tmp_path, {"asset": "llama-b6099-bin-ubuntu-x64.tar.gz"})
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"
        install_dir = _install(
            tmp_path, {"backend": "sycl", "asset": "app-b1-linux-x64-cuda12.tar.gz"}
        )
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"

    def test_a_prebuilt_that_no_longer_runs_is_replaced(self, tmp_path):
        # A quarantined library leaves the marker and the executable behind.
        install_dir = _install(tmp_path, {"backend": "cuda"})
        assert (
            _decide(
                tmp_path, install_dir, _setup_nvidia_physical = "true", _setup_prebuilt_runs = "false"
            )
            == "REPLACE"
        )

    def test_a_tree_without_a_server_is_not_worth_keeping(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": "cuda"}, server = False)
        assert _decide(tmp_path, install_dir, _setup_nvidia_physical = "true") == "REPLACE"

    def test_a_build_asked_for_by_hand_still_runs(self, tmp_path):
        install_dir = _install(tmp_path, {"backend": "cuda"})
        nvidia = {"_setup_nvidia_physical": "true"}
        assert _decide(tmp_path, install_dir, _LLAMA_FORCE_COMPILE = "1", **nvidia) == "REPLACE"
        assert _decide(tmp_path, install_dir, _LLAMA_PR = "12345", **nvidia) == "REPLACE"


# ── part two: the wiring ──


def _between(start_marker, end_marker):
    start = SETUP_TEXT.index(start_marker)
    return SETUP_TEXT[start : SETUP_TEXT.index(end_marker, start)]


def test_a_failed_update_keeps_the_gpu_prebuilt_before_any_source_build():
    # The first decision is at the failed update itself: no source build is started.
    window = _between('[ "$_PREBUILT_STATUS" -eq 2 ]', "_NEED_LLAMA_SOURCE_BUILD=true")
    assert (
        '_LLAMA_KEPT_GPU_PREBUILT="$(_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR")"'
        in window
    )
    assert "the next update will retry" in window
    # The reason is read before the helper's log is deleted.
    assert window.index('_llama_update_fail_reason "$_PREBUILT_LOG"') < window.index(
        'rm -f "$_PREBUILT_LOG"'
    )


@requires_bash
@pytest.mark.parametrize(
    ("log", "reason"),
    [
        ("HTTP Error 429: Too Many Requests", "GitHub rate limit"),
        ("API rate limit exceeded for 1.2.3.4", "GitHub rate limit"),
        ("urlopen error timed out", "network error"),
        ("Temporary failure in name resolution", "network error"),
        ("HTTP Error 503: Service Unavailable", "network error"),
        ("checksum mismatch for app-b1-linux-x64-cuda12.tar.gz", "download failed"),
        ("", "download failed"),
    ],
)
def test_the_failure_reason_is_a_few_words(tmp_path, log, reason):
    fn = _between("_llama_update_fail_reason() {", "\n}\n") + "\n}\n"
    path = tmp_path / "prebuilt.log"
    path.write_text(log, encoding = "utf-8")
    result = subprocess.run(
        [BASH, "-c", fn + '\n_llama_update_fail_reason "$1"', "reason", str(path)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 60,
    )
    assert result.stdout.strip() == reason, result.stderr


def test_the_decision_runs_before_the_compile():
    # Deciding after a 20 minute compile would be correct and pointless.
    window = _between('_BUILD_DESC="building (CPU)"', 'run_quiet_no_exit "cmake llama.cpp"')
    assert (
        '_LLAMA_KEPT_GPU_PREBUILT="$(_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR")"'
        in window
    )
    assert "BUILD_OK=false" in window
    # The configure must honour that verdict: it sat in the same BUILD_OK block.
    assert '[ "$BUILD_OK" = true ] && ! run_quiet_no_exit "cmake llama.cpp"' in SETUP_TEXT


def test_the_decision_runs_again_at_the_swap():
    # A CUDA build that fell back to CPU on the way only shows at the swap.
    window = _between("caught here, after the fact", "# Swap only after build succeeds")
    assert '[ -z "$GPU_BACKEND" ] && [ "$_TRY_METAL_CPU_FALLBACK" != true ]' in window
    assert '_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR"' in window
    assert "_LLAMA_CPU_ONLY_ON_GPU_HOST=true" in window


def test_the_kept_tree_is_checked_offline_the_way_the_updater_checks_it():
    keep = _between("_gpu_prebuilt_to_keep_over_cpu_build() {", "\n}\n")
    assert '_installed_prebuilt_runs "$install_dir" || return 1' in keep
    runs = _between("_installed_prebuilt_runs() {", "\n}\n")
    # Not --validate-install: that downloads its probe model, and the update just failed
    # for want of a download.
    assert '--check-installed "$1"' in runs and "--validate-install" not in runs


def _load_ilp():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "keep_check_ilp", PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
    )
    ilp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ilp
    spec.loader.exec_module(ilp)
    return ilp


def _linux_host(ilp, **fields):
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    base.update(fields)
    return ilp.HostInfo(**base)


class TestTheKeptBundleMustStillCoverTheCard:
    """A same-vendor card swap passes the vendor check and --version; the marker does not."""

    def test_a_cuda_bundle_is_kept_only_for_the_sms_it_was_built_for(self):
        ilp = _load_ilp()
        marker = {"backend": "cuda", "supported_sms": ["7.5", "8.6", "8.9"]}
        assert ilp._kept_install_covers_host(marker, _linux_host(ilp, compute_caps = ["8.9"]))
        assert not ilp._kept_install_covers_host(marker, _linux_host(ilp, compute_caps = ["12.0"]))
        # Two cards: every one must be covered.
        assert not ilp._kept_install_covers_host(
            marker, _linux_host(ilp, compute_caps = ["8.9", "12.0"])
        )

    def test_a_masked_card_is_checked_by_its_physical_sms(self):
        ilp = _load_ilp()
        marker = {"backend": "cuda", "supported_sms": ["8.9"]}
        masked = _linux_host(ilp, compute_caps = [], physical_compute_caps = ["12.0"])
        assert not ilp._kept_install_covers_host(marker, masked)
        assert ilp._kept_install_covers_host(
            marker, _linux_host(ilp, compute_caps = [], physical_compute_caps = ["8.9"])
        )

    def test_a_cuda_bundle_is_kept_only_while_the_driver_runs_its_runtime(self):
        ilp = _load_ilp()
        marker = {"backend": "cuda", "runtime_line": "cuda13", "supported_sms": ["8.9"]}
        host = lambda v: _linux_host(ilp, compute_caps = ["8.9"], driver_cuda_version = v)
        assert ilp._kept_install_covers_host(marker, host((13, 1)))
        assert not ilp._kept_install_covers_host(marker, host((12, 8)))
        # cuda12 runs on a 13 driver; an unknown driver cannot tell.
        assert ilp._kept_install_covers_host({**marker, "runtime_line": "cuda12"}, host((13, 1)))
        assert ilp._kept_install_covers_host(marker, host(None))

    def test_a_rocm_bundle_is_kept_for_its_mapped_targets_or_family(self):
        ilp = _load_ilp()
        marker = {
            "backend": "rocm",
            "gfx_target": "gfx110X",
            "mapped_targets": ["gfx1100", "gfx1101", "gfx1102"],
        }
        for gfx in ("gfx1100", "GFX1101", "gfx110X"):
            assert ilp._kept_install_covers_host(marker, _linux_host(ilp, rocm_gfx_target = gfx))
        assert not ilp._kept_install_covers_host(
            marker, _linux_host(ilp, rocm_gfx_target = "gfx1201")
        )

    def test_a_marker_without_coverage_cannot_tell_and_passes(self):
        ilp = _load_ilp()
        host = _linux_host(ilp, compute_caps = ["12.0"], rocm_gfx_target = "gfx1201")
        for marker in (
            None,
            {"backend": "cuda"},
            {"backend": "rocm", "mapped_targets": []},
            {"backend": "vulkan"},
            {"llama_backend": "cpu"},
        ):
            assert ilp._kept_install_covers_host(marker, host)
        # An unknown host SM (masked) cannot be checked either.
        assert ilp._kept_install_covers_host(
            {"backend": "cuda", "supported_sms": ["8.9"]}, _linux_host(ilp)
        )

    def test_check_installed_refuses_a_bundle_that_no_longer_covers_the_card(
        self, tmp_path, monkeypatch
    ):
        ilp = _load_ilp()
        monkeypatch.setattr(ilp, "detect_host", lambda **k: _linux_host(ilp, compute_caps = ["12.0"]))
        monkeypatch.setattr(ilp, "_existing_install_runs", lambda d, h: True)
        monkeypatch.setattr(
            ilp,
            "load_prebuilt_metadata",
            lambda d: {"backend": "cuda", "supported_sms": ["8.9"]},
        )
        monkeypatch.setattr(
            sys, "argv", ["install_llama_prebuilt.py", "--check-installed", str(tmp_path)]
        )
        assert ilp.main() == 2


@requires_bash
def test_check_installed_answers_with_the_updaters_own_check(tmp_path, monkeypatch):
    ilp = _load_ilp()
    seen = {}
    monkeypatch.setattr(ilp, "detect_host", lambda **k: "host")

    def runs(install_dir, host):
        seen["args"] = (install_dir, host)
        return seen["answer"]

    monkeypatch.setattr(ilp, "_existing_install_runs", runs)
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--check-installed", str(tmp_path)]
    )
    seen["answer"] = True
    assert ilp.main() == 0
    assert seen["args"] == (tmp_path, "host")
    seen["answer"] = False
    assert ilp.main() == 2
    # Nothing about a check may download: no fetch helper is reached.
    monkeypatch.setattr(ilp, "download_file", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(
        ilp, "_existing_install_runs", lambda d, h: (_ for _ in ()).throw(OSError("x"))
    )
    assert ilp.main() == 2


def test_an_intel_host_on_cpu_is_named_too():
    window = _between("caught here, after the fact", "# Swap only after build succeeds")
    assert "_setup_has_intel_gpu" in window


def test_the_arm64_cpu_prebuilt_fallback_is_named_in_the_footer_too():
    window = _between('step "llama.cpp" "arm64 CPU prebuilt installed', "_STUDIO_OWNED_MARKER")
    assert "_LLAMA_CPU_ONLY_ON_GPU_HOST=true" in window


def test_a_kept_prebuilt_is_not_reported_as_a_failed_build():
    window = _between(
        'step "llama.cpp" "binary not found after build"', 'step "llama.cpp" "build failed"'
    )
    assert 'elif [ -n "$_LLAMA_KEPT_GPU_PREBUILT" ]' in window


def test_every_footer_names_the_outcome():
    assert (
        SETUP_TEXT.count("_print_llama_gpu_notes\n") == 3
    ), "llama-only, Colab and the default footer"
    notes = _between("_print_llama_gpu_notes() {", "\n}\n")
    assert "_LLAMA_KEPT_GPU_PREBUILT" in notes and "_LLAMA_CPU_ONLY_ON_GPU_HOST" in notes


def test_the_flags_are_initialised_for_set_u():
    assert '_LLAMA_KEPT_GPU_PREBUILT=""' in SETUP_TEXT
    assert "_LLAMA_CPU_ONLY_ON_GPU_HOST=false" in SETUP_TEXT


def test_the_source_build_reads_capabilities_from_the_driver_library_too():
    """setup.sh's CUDA source build turned CUDA off without nvidia-smi (#5854); the probe
    module beside it lists the same capabilities."""
    start = SETUP_TEXT.index(
        'CUDA_ARCHS="$(_resolve_cuda_archs "$_raw_caps" "${UNSLOTH_LLAMA_CUDA_ARCHS:-}")"'
    )
    end = SETUP_TEXT.index('if [ -n "$CUDA_ARCHS" ]; then', start)
    between = SETUP_TEXT[start:end]
    # After the first resolution, so an nvidia-smi answering N/A falls back as an absent one does.
    assert "_probe_compute_caps" in between and '[ -z "$CUDA_ARCHS" ]' in between
    assert "UNSLOTH_NVIDIA_LIBRARY_PROBE" in between


@requires_bash
@pytest.mark.parametrize(
    "listing, expected",
    [
        ("GPU 0: A (compute 8.9)\nGPU 1: B (compute 12.0)\n", "8.9\n12.0\n"),
        # One GPU without a readable capability voids the list, as the Python side does.
        ("GPU 0: A (compute 8.9)\nGPU 1: B (compute )\n", ""),
        ("", ""),
    ],
)
def test_the_probe_capabilities_are_all_or_nothing(tmp_path, listing, expected):
    start = SETUP_TEXT.index("_probe_compute_caps() {")
    body = SETUP_TEXT[start : SETUP_TEXT.index("\n}\n", start) + 3]
    (tmp_path / "python3").write_text(f"#!/bin/sh\nprintf '%b' {listing!r}\n", encoding = "utf-8")
    (tmp_path / "python3").chmod(0o755)
    (tmp_path / "nvidia_probe.py").write_text("", encoding = "utf-8")
    script = '_setup_run_smi() { "$@"; }\n' + body + "\n_probe_compute_caps\n"
    result = subprocess.run(
        [BASH, "-c", script],
        env = {
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
            "SCRIPT_DIR": str(tmp_path),
        },
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
