# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for AMD marketing-name and torch fast-path reconciliation."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
STACK_PATH = REPO_ROOT / "studio" / "install_python_stack.py"

STACK_SPEC = importlib.util.spec_from_file_location("amd_reconciliation_stack", STACK_PATH)
assert STACK_SPEC is not None and STACK_SPEC.loader is not None
stack = importlib.util.module_from_spec(STACK_SPEC)
sys.modules[STACK_SPEC.name] = stack
STACK_SPEC.loader.exec_module(stack)

# rocminfo lists the CPU before the discrete GPU.
ROCMINFO_DISCRETE = """\
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 9950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 9950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
"""

# The APU fallback feeds name-based gfx inference when rocminfo has no gfx token.
ROCMINFO_APU_NO_GFX = """\
*******
Agent 1
*******
  Name:                    AMD Ryzen AI Max+ 395 w/ Radeon 8060S
  Marketing Name:          AMD Ryzen AI Max+ 395 w/ Radeon 8060S
  Device Type:             CPU
"""

ROCMINFO_MULTI_GPU = """\
*******
Agent 1
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon 8060S
  Device Type:             GPU
"""

ROCMINFO_MULTI_GPU_BLANK_FIRST = """\
*******
Agent 1
*******
  Name:                    gfx1100
  Marketing Name:
  Device Type:             GPU
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon 8060S
  Device Type:             GPU
"""


def _extract_function(script: pathlib.Path, name: str) -> str:
    src = script.read_text(encoding = "utf-8")
    start = src.find(f"{name}() {{")
    assert start >= 0, f"{name} missing from {script.name}"
    end = src.find("\n}", start)
    assert end > start, f"{name} in {script.name} is not brace-terminated"
    return src[start : end + 2]


def _run_gpu_records(tmp_path, script: pathlib.Path, name: str, rocminfo: str) -> str:
    probe = tmp_path / "probe.sh"
    probe.write_text(_extract_function(script, name) + f"\n{name}\n")
    return subprocess.run(
        ["bash", str(probe)],
        input = rocminfo,
        capture_output = True,
        text = True,
        timeout = 30,
    ).stdout.strip()


RECORD_HELPERS = [
    (INSTALL_SH, "_rocminfo_gpu_records"),
    (SETUP_SH, "_setup_rocminfo_gpu_records"),
]
RECORD_IDS = ["install.sh", "setup.sh"]


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_discrete_host_reports_the_gpu_not_the_cpu(tmp_path, script, fn):
    """#7307: the GPU line must name the card, not the processor in front of it."""
    assert _run_gpu_records(tmp_path, script, fn, ROCMINFO_DISCRETE) == (
        "gfx1201|AMD Radeon AI PRO R9700"
    )


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_apu_without_a_gfx_agent_keeps_the_processor_name(tmp_path, script, fn):
    """The fallback feeds name-based arch inference; losing it would drop APUs to CPU torch."""
    assert (
        _run_gpu_records(tmp_path, script, fn, ROCMINFO_APU_NO_GFX)
        == "|AMD Ryzen AI Max+ 395 w/ Radeon 8060S"
    )


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_no_rocminfo_output_yields_no_name(tmp_path, script, fn):
    assert _run_gpu_records(tmp_path, script, fn, "") == ""


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_multi_gpu_probe_preserves_one_name_per_device(tmp_path, script, fn):
    assert _run_gpu_records(tmp_path, script, fn, ROCMINFO_MULTI_GPU).splitlines() == [
        "gfx1100|AMD Radeon RX 7900 XTX",
        "gfx1151|AMD Radeon 8060S",
    ]


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_blank_marketing_name_preserves_its_device_slot(tmp_path, script, fn):
    assert _run_gpu_records(tmp_path, script, fn, ROCMINFO_MULTI_GPU_BLANK_FIRST).splitlines() == [
        "gfx1100|",
        "gfx1151|AMD Radeon 8060S",
    ]


@pytest.mark.parametrize(
    "script, fn",
    [
        (INSTALL_SH, "_select_visible_gpu_line"),
        (SETUP_SH, "_setup_select_visible_line"),
    ],
    ids = RECORD_IDS,
)
def test_visible_device_index_selects_the_matching_name(tmp_path, script, fn):
    probe = tmp_path / "select.sh"
    probe.write_text(_extract_function(script, fn) + f"\nprintf '%s\\n' first second | {fn} 1\n")
    result = subprocess.run(["bash", str(probe)], capture_output = True, text = True, timeout = 30)
    assert result.stdout.strip() == "second"


@pytest.mark.parametrize(
    "script, fn",
    [
        (INSTALL_SH, "_select_visible_gpu_line"),
        (SETUP_SH, "_setup_select_visible_line"),
    ],
    ids = RECORD_IDS,
)
def test_visible_device_selection_keeps_a_record_with_a_blank_name(tmp_path, script, fn):
    probe = tmp_path / "select-blank.sh"
    probe.write_text(
        _extract_function(script, fn)
        + f"\nprintf '%s\\n' 'gfx1100|' 'gfx1151|AMD Radeon 8060S' | {fn} 0\n"
    )
    result = subprocess.run(["bash", str(probe)], capture_output = True, text = True, timeout = 30)
    assert result.stdout.strip() == "gfx1100|"


# The dependency fast path delegates to the same Python implementation that performs repair.


def _repair_needed(
    monkeypatch,
    *,
    version = "2.11.0+cpu",
    hip = "",
    **state,
):
    defaults = {
        "NO_TORCH": False,
        "IS_LINUX": True,
        "_TORCH_BACKEND": "",
        "machine": "x86_64",
        "unknown_pin": None,
        "rocm_pin": None,
        "nvidia": False,
        "amd": True,
        "assume_amd_detected": False,
        "assume_nvidia_detected": False,
        "detected_gfx_devices": [],
        "detected_gfx_probe": None,
        "inferred_gfx": None,
        "rocm_version": (7, 2),
        "installed_rocm_family": None,
    }
    defaults.update(state)
    monkeypatch.setattr(stack, "NO_TORCH", defaults["NO_TORCH"])
    monkeypatch.setattr(stack, "IS_LINUX", defaults["IS_LINUX"])
    monkeypatch.setattr(stack, "_TORCH_BACKEND", defaults["_TORCH_BACKEND"])
    monkeypatch.setattr(stack.platform, "machine", lambda: defaults["machine"])
    monkeypatch.setattr(
        stack, "_explicit_unknown_family_torch_index_url", lambda: defaults["unknown_pin"]
    )
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: defaults["rocm_pin"])
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: defaults["nvidia"])
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: defaults["amd"])
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: defaults["inferred_gfx"])
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: defaults["rocm_version"])
    monkeypatch.setattr(
        stack, "_installed_rocm_wheel_family", lambda: defaults["installed_rocm_family"]
    )
    monkeypatch.setattr(stack, "_installed_torch_build_metadata", lambda: (version, hip))
    return stack._rocm_fast_path_needs_repair(
        assume_amd_detected = defaults["assume_amd_detected"],
        assume_nvidia_detected = defaults["assume_nvidia_detected"],
        detected_gfx_devices = defaults["detected_gfx_devices"],
        detected_gfx_probe = defaults["detected_gfx_probe"],
    )


@pytest.mark.parametrize("version", ["2.11.0+cpu", "2.10.0", "2.10.0+cu128"])
def test_non_rocm_wheel_on_an_amd_host_forces_the_pass(monkeypatch, version):
    assert _repair_needed(monkeypatch, version = version)


def test_untagged_hip_build_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(monkeypatch, version = "2.11.0", hip = "7.2.0")


@pytest.mark.parametrize(
    "state",
    [
        {"NO_TORCH": True},
        {"machine": "aarch64"},
        {"nvidia": True},
        {"amd": False},
        {"unknown_pin": "https://mirror/whl/rocm-private"},
    ],
    ids = ["no-torch", "aarch64", "nvidia", "no-amd", "custom-pin"],
)
def test_unrepairable_or_user_selected_states_keep_the_fast_path(monkeypatch, state):
    assert not _repair_needed(monkeypatch, **state)


def test_unreadable_rocm_without_an_inferred_arch_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(monkeypatch, rocm_version = None)


def test_inferred_arch_allows_repair_without_a_rocm_version(monkeypatch):
    assert _repair_needed(monkeypatch, amd = False, rocm_version = None, inferred_gfx = "gfx1151")


def test_setup_can_reuse_its_amd_vendor_verdict(monkeypatch):
    """The delegated decision must not repeat NVIDIA or AMD hardware probes."""
    assert _repair_needed(monkeypatch, assume_amd_detected = True, nvidia = True, amd = False)


def test_setup_nvidia_verdict_skips_unpinned_rocm_repair(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        amd = False,
        nvidia = False,
        assume_nvidia_detected = True,
    )


def test_missing_or_unreadable_torch_metadata_forces_the_pass(monkeypatch):
    assert _repair_needed(monkeypatch, version = "", hip = "")


def test_explicit_rocm_pin_repairs_without_a_hardware_probe(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        amd = False,
        nvidia = True,
        version = "2.11.0+cpu",
        rocm_pin = "https://download.pytorch.org/whl/rocm7.2",
    )


def test_explicit_rocm_family_mismatch_forces_the_pass(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        version = "2.10.0+rocm6.4",
        rocm_pin = "https://download.pytorch.org/whl/rocm7.2",
    )


def test_matching_explicit_rocm_family_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        version = "2.10.0+rocm6.4",
        rocm_pin = "https://download.pytorch.org/whl/rocm6.4",
    )


def test_explicit_gfx_sibling_family_forces_the_pass(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.13.0",
        hip = "7.13.0",
        rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
        installed_rocm_family = "gfx1150",
    )


def test_matching_explicit_gfx_family_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.13.0",
        hip = "7.13.0",
        rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
        installed_rocm_family = "gfx1151",
    )


def test_unknown_explicit_gfx_family_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.13.0",
        hip = "7.13.0",
        rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
        installed_rocm_family = None,
    )


def test_strix_generic_rocm_wheel_forces_the_pass(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.2",
        hip = "7.2.0",
        detected_gfx_devices = ["gfx1151"],
    )


def test_strix_matching_arch_wheel_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.13.0",
        hip = "7.13.0",
        detected_gfx_devices = ["gfx1151"],
        installed_rocm_family = "gfx1151",
    )


def test_strix_sibling_arch_wheel_forces_the_pass(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.13.0",
        hip = "7.13.0",
        detected_gfx_devices = ["gfx1151"],
        installed_rocm_family = "gfx1150",
    )


def test_visible_non_strix_gpu_does_not_trigger_present_strix_repair(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert not _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.2",
        hip = "7.2.0",
        detected_gfx_devices = ["gfx1100", "gfx1151"],
        detected_gfx_probe = "amd-smi",
    )


def test_gfx906_newer_rocm_wheel_forces_the_pass(monkeypatch):
    assert _repair_needed(
        monkeypatch,
        version = "2.11.0+rocm7.2",
        hip = "7.2.0",
        detected_gfx_devices = ["gfx906"],
    )


def test_gfx906_legacy_rocm_wheel_keeps_the_fast_path(monkeypatch):
    assert not _repair_needed(
        monkeypatch,
        version = "2.7.0+rocm6.3",
        hip = "6.3.0",
        detected_gfx_devices = ["gfx906"],
    )


def test_active_interpreter_metadata_ignores_a_stale_python_tree(tmp_path, monkeypatch):
    active = tmp_path / "python3.12" / "site-packages"
    stale = tmp_path / "python3.10" / "site-packages"
    (active / "torch").mkdir(parents = True)
    (stale / "torch").mkdir(parents = True)
    (active / "torch" / "version.py").write_text(
        "from typing import Optional\n__version__ = '2.11.0'\nhip: Optional[str] = '7.2.0'\n"
    )
    (stale / "torch" / "version.py").write_text("__version__ = '2.10.0+cpu'\n")
    monkeypatch.setattr(stack.sysconfig, "get_path", lambda _name: str(active))
    assert stack._installed_torch_build_metadata() == ("2.11.0", "7.2.0")


def test_escape_precedes_the_dependency_pass_it_controls():
    """The vendor probe must run before the fast path, or the escape reads unset variables."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    probe = src.find("if _setup_has_usable_nvidia_gpu; then")
    fast_path = src.find("# ── Check if Python deps need updating ──")
    decision = src.find('if [ "$_SKIP_PYTHON_DEPS" = false ]; then')
    assert 0 <= probe < fast_path < decision, (
        "setup.sh must classify the host before the dependency fast path decides "
        "whether to reinstall PyTorch"
    )


def test_fast_path_probe_reads_metadata_without_importing_torch():
    source = STACK_PATH.read_text(encoding = "utf-8")
    start = source.index("def _rocm_fast_path_needs_repair(")
    end = source.index("\ndef _ensure_rocm_torch()", start)
    assert "import torch" not in source[start:end]


def test_shell_bounds_and_diagnoses_the_delegated_probe():
    source = SETUP_SH.read_text(encoding = "utf-8")
    start = source.index("            _setup_rocm_fast_path_probe()")
    block = source[start : start + 3500]
    assert 'timeout 45 "$VENV_DIR/bin/python"' in block
    assert "--rocm-fast-path-needs-repair" in block
    assert "--amd-detected --amd-gfx" in block
    assert "--nvidia-detected" in block
    assert (
        'if [ "$_SKIP_PYTHON_DEPS" = true ] && [ "${_setup_nvidia_usable:-}" != true ]'
        not in source
    )
    assert 'verbose_substep "ROCm reconciliation probe failed' in block
    stack_source = STACK_PATH.read_text(encoding = "utf-8")
    assert "else 3" in stack_source


# ── setup.sh: the GPU summary must report the runtime answer ──


def test_summary_probes_torch_and_has_a_warning_arm():
    src = SETUP_SH.read_text(encoding = "utf-8")
    start = src.find("    _setup_rocm_torch_ok=unknown")
    assert start >= 0, "setup.sh lost the AMD torch runtime probe"
    block = src[start : start + 2500]
    assert (
        "torch.cuda.is_available()" in block
    ), "the AMD summary must ask torch whether it can use the GPU it just announced"
    assert (
        "signal.alarm(60)" in block and "timeout 60" in block
    ), "the probe must be bounded: a faulted HIP runtime hangs inside `import torch`"
    assert (
        'step "gpu" "AMD ROCm ($_setup_gfx, PyTorch cannot use it)" "$C_WARN"' in block
    ), "a GPU torch cannot use must not be reported as a healthy ROCm install"


def test_summary_reports_an_intentional_all_hidden_mask_separately():
    src = SETUP_SH.read_text(encoding = "utf-8")
    assert "HIP_VISIBLE_DEVICES+x" in src
    assert "ROCR_VISIBLE_DEVICES+x" in src
    assert "CUDA_VISIBLE_DEVICES+x" in src
    assert 'substep "$_setup_vis_name intentionally hides every AMD device from PyTorch."' in src
