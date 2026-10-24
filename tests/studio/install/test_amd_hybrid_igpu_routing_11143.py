# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""_pick_rocm_gfx_target took the first arch listed, and a Vulkan request was erased to auto."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_11143", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ILP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ILP
_SPEC.loader.exec_module(ILP)

_VISIBILITY_ENV = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _rocminfo(*arches: str) -> str:
    """rocminfo as the real tool prints it: one agent section per GPU, arch repeated."""
    return "".join(
        f"***\nAgent {index}\n***\n  Name: {arch}\n  ISA: amdgcn-amd-amdhsa--{arch}\n"
        for index, arch in enumerate(arches, start = 1)
    )


# Autouse: _pick_rocm_gfx_target reads visibility masks set on GPU hosts.
@pytest.fixture(autouse = True)
def unmasked(monkeypatch):
    for name in _VISIBILITY_ENV:
        monkeypatch.delenv(name, raising = False)
    return monkeypatch


def test_a_leading_igpu_does_not_strand_the_discrete_card(unmasked):
    """The reported host: Raphael gfx1036 driving the display enumerates ahead of the
    RX 9060 XT. One arch picks the bundle, so the discrete card must win."""
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1200")) == "gfx1200"


def test_every_shadowing_arch_defers_to_a_discrete_card(unmasked):
    """The table is the contract, not the one arch from the report."""
    for igpu in sorted(ILP.SHADOWING_INTEGRATED_GFX):
        assert ILP._pick_rocm_gfx_target(_rocminfo(igpu, "gfx1100")) == "gfx1100", igpu


def test_a_non_shadowing_first_arch_is_untouched(unmasked):
    """A discrete card that enumerates first already decided; the preference must not
    reorder a host it has no business reordering."""
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1100", "gfx1036")) == "gfx1100"
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1200")) == "gfx1200"


def test_an_all_integrated_host_keeps_its_own_arch(unmasked):
    """Nothing to prefer. The old code returned the first token here and still must:
    returning None would drop a host that has a perfectly good APU bundle."""
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1103")) == "gfx1036"
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036")) == "gfx1036"


def test_gfx906_is_never_the_repick(unmasked):
    """install.sh excludes it because naming it on a mixed host strands BOTH cards."""
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx906", "gfx1200")) == "gfx1200"
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx906")) == "gfx1036"


def test_rocminfo_output_is_not_masked_by_rocr_twice(unmasked):
    """rocminfo already applied ROCR_VISIBLE_DEVICES=1,0: its survivors are R9700, iGPU."""
    out = _rocminfo("gfx1201", "gfx1036")
    unmasked.setenv("ROCR_VISIBLE_DEVICES", "1,0")
    assert ILP._pick_rocm_gfx_target(out, rocr_filtered = True) == "gfx1201"
    unmasked.setenv("HIP_VISIBLE_DEVICES", "1")
    assert ILP._pick_rocm_gfx_target(out, rocr_filtered = True) == "gfx1036"
    unmasked.delenv("HIP_VISIBLE_DEVICES")
    assert ILP._pick_rocm_gfx_target(out) == "gfx1036"


def test_rocr_over_filtered_output_is_still_an_explicit_choice(unmasked):
    """ROCR=0,1 leaves the iGPU first; it was chosen, so the discrete repick must not undo it."""
    out = _rocminfo("gfx1036", "gfx1200")
    unmasked.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    assert ILP._pick_rocm_gfx_target(out, rocr_filtered = True) == "gfx1036"
    unmasked.setenv("ROCR_VISIBLE_DEVICES", "-1")
    assert ILP._pick_rocm_gfx_target(out, rocr_filtered = True) is None


def test_the_rocminfo_probe_says_its_output_is_rocr_filtered():
    import ast

    tree = ast.parse(Path(ILP.__file__).read_text(encoding = "utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "detect_host"
    )
    calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_pick_rocm_gfx_target"
    ]
    assert any(k.arg == "rocr_filtered" for c in calls for k in c.keywords)


@pytest.mark.parametrize("env", _VISIBILITY_ENV)
def test_an_explicit_visibility_mask_still_wins(monkeypatch, env):
    """HIP_VISIBLE_DEVICES=0 on the hybrid host selects the iGPU on purpose -- it is the
    documented workaround for #7624 / #7669 -- and the preference must not undo it."""
    for name in _VISIBILITY_ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv(env, "0")
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1200")) == "gfx1036"
    monkeypatch.setenv(env, "1")
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1200")) == "gfx1200"


@pytest.mark.parametrize("value", ["", "-1"])
def test_a_mask_selecting_no_device_still_means_no_gpu(unmasked, value):
    unmasked.setenv("HIP_VISIBLE_DEVICES", value)
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1200")) is None


@pytest.mark.parametrize("value", ["GPU-abc123", "7"])
def test_a_mask_this_resolver_cannot_read_is_still_an_explicit_choice(unmasked, value):
    """UUID-style and out-of-range masks fell through to device 0 before, and must keep
    doing so: the user selected a device, so this is not the place to second-guess it."""
    unmasked.setenv("HIP_VISIBLE_DEVICES", value)
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1036", "gfx1200")) == "gfx1036"


def _choice(install_kind: str) -> "ILP.AssetChoice":
    return ILP.AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "b9925",
        name = "bundle.tar.gz",
        url = "https://example/bundle",
        source_label = "published",
        install_kind = install_kind,
    )


def test_a_vulkan_request_that_landed_rocm_keeps_the_request():
    """The reported symptom: "settings were already configured for Vulkan" and ROCm kept
    coming back. Recording "auto" destroyed the choice, so every later update re-detected."""
    landed = _choice("linux-rocm")
    assert ILP.persisted_marker_backend_request("vulkan", landed) == "vulkan"
    assert ILP.marker_backend_request_was_satisfied("vulkan", landed) is False


def test_a_request_the_bundle_honours_is_not_flagged():
    landed = _choice("linux-vulkan")
    assert ILP.persisted_marker_backend_request("vulkan", landed) == "vulkan"
    assert ILP.marker_backend_request_was_satisfied("vulkan", landed) is True


@pytest.mark.parametrize("request_backend", [None, "auto"])
def test_detection_is_never_flagged_unsatisfied(request_backend):
    """ "auto" asks for whatever the host resolves to, so nothing can contradict it."""
    landed = _choice("linux-rocm")
    assert ILP.persisted_marker_backend_request(request_backend, landed) == "auto"
    assert ILP.marker_backend_request_was_satisfied(request_backend, landed) is True


def test_an_old_marker_without_the_field_reads_as_satisfied():
    """Back-compat is the whole reason the flag is absent-means-satisfied: every marker
    ever written lacks it, and each one described an install that honoured its request."""
    assert ILP.marker_records_unsatisfied_backend_request(None) is False
    assert ILP.marker_records_unsatisfied_backend_request({}) is False
    assert (
        ILP.marker_records_unsatisfied_backend_request(
            {"backend_request": "vulkan", "backend": "vulkan"}
        )
        is False
    )
    assert (
        ILP.marker_records_unsatisfied_backend_request(
            {"backend_request": "vulkan", "backend": "rocm", "backend_request_unsatisfied": True}
        )
        is True
    )


def _linux_host() -> "ILP.HostInfo":
    return ILP.HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_macos = False,
        is_linux = True,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )


def test_only_a_flagged_marker_may_disagree_with_its_own_backend():
    """_marker_backend_fits_host reads the pair as self-consistency evidence. A
    disagreement the installer recorded on purpose is consistent; one with no flag is
    still a marker this installer did not write, and keeps taking the full path."""
    host = _linux_host()
    honest = {"backend_request": "vulkan", "backend": "vulkan"}
    unflagged = {"backend_request": "vulkan", "backend": "rocm"}
    flagged = dict(unflagged, backend_request_unsatisfied = True)
    assert ILP._marker_backend_fits_host(honest, host) is True
    assert ILP._marker_backend_fits_host(unflagged, host) is False
    assert ILP._marker_backend_fits_host(flagged, host) is True


def test_an_unserved_preferred_arch_source_builds_rather_than_using_the_igpus_bundle():
    """An unserved discrete arch must source build, since the iGPU's bundle may be for a masked device."""
    assert ILP._pick_rocm_gfx_target(_rocminfo("gfx1103", "gfx1010")) == "gfx1010"


def test_the_fast_path_keeps_the_callers_forwarded_rocm_detection():
    """Re-derived route must reuse the forwarded ROCm detection, or AMD hosts always miss the fast path."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ILP.install_prebuilt))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "existing_install_current_without_plan"
        ):
            passed = {kw.arg for kw in node.keywords}
            missing = {"override_has_rocm", "override_rocm_gfx", "route"} - passed
            assert not missing, (
                f"install_prebuilt calls existing_install_current_without_plan without "
                f"{sorted(missing)}, so its re-derived route describes a different host"
            )
            return
    raise AssertionError("no existing_install_current_without_plan call found")


def test_an_explicit_false_flag_is_canonicalised_away_on_reuse():
    """Absent is the only spelling of satisfied. A marker already carrying an explicit
    false has to lose the key, or readers get two spellings of one state."""
    marker = {
        "backend_request": "vulkan",
        "backend": "vulkan",
        ILP.MARKER_BACKEND_REQUEST_UNSATISFIED: False,
    }
    patch = ILP._marker_selection_patch(
        marker,
        choice = _choice("linux-vulkan"),
        backend_request = "vulkan",
        persist_force_cpu = False,
        persist_llama_backend = None,
        ggml_tree = None,
        rocm_gfx = None,
    )
    assert ILP.MARKER_BACKEND_REQUEST_UNSATISFIED in patch
    assert patch[ILP.MARKER_BACKEND_REQUEST_UNSATISFIED] is None


def test_a_satisfied_marker_without_the_flag_is_left_alone():
    """The negative control for the above: no key, nothing to canonicalise."""
    patch = ILP._marker_selection_patch(
        {"backend_request": "vulkan", "backend": "vulkan"},
        choice = _choice("linux-vulkan"),
        backend_request = "vulkan",
        persist_force_cpu = False,
        persist_llama_backend = None,
        ggml_tree = None,
        rocm_gfx = None,
    )
    assert ILP.MARKER_BACKEND_REQUEST_UNSATISFIED not in patch


def test_a_single_build_platform_never_records_an_unfulfillable_request():
    """On single-build platforms record auto, since a kept request has no picker option."""
    mac = _choice("macos-arm64")
    for request in ("cpu", "vulkan", "rocm", "cuda"):
        assert ILP.persisted_marker_backend_request(request, mac) == "auto"
        assert ILP.marker_backend_request_was_satisfied(request, mac) is True
    assert ILP.backend_for_install_kind("macos-arm64") == "metal"
    assert "metal" not in ILP.REQUESTABLE_BACKENDS
    assert ILP.persisted_marker_backend_request("vulkan", _choice("linux-rocm")) == "vulkan"
    assert ILP.marker_backend_request_was_satisfied("vulkan", _choice("linux-rocm")) is False
