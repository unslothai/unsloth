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


def _plan(
    monkeypatch,
    *,
    imports_as_rocm = False,
    version = "2.11.0+cpu",
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
        "gfx_devices": [],
        "gfx_probe": None,
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
    monkeypatch.setattr(stack, "_probe_rocm_torch", lambda: (imports_as_rocm, version))
    monkeypatch.setattr(
        stack, "_detect_amd_gfx_codes", lambda **_kwargs: list(defaults["gfx_devices"])
    )
    monkeypatch.setattr(stack, "_LAST_AMD_GFX_PROBE", defaults["gfx_probe"])
    return stack._linux_rocm_torch_plan()[0]


@pytest.mark.parametrize("version", ["2.11.0+cpu", "2.10.0", "2.10.0+cu128"])
def test_non_rocm_wheel_on_an_amd_host_forces_the_pass(monkeypatch, version):
    assert _plan(monkeypatch, version = version) is not None


def test_untagged_hip_build_keeps_the_fast_path(monkeypatch):
    assert _plan(monkeypatch, imports_as_rocm = True, version = "2.11.0") is None


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
    assert _plan(monkeypatch, **state) is None


def test_unreadable_rocm_without_an_inferred_arch_keeps_the_fast_path(monkeypatch):
    assert _plan(monkeypatch, rocm_version = None) is None


def test_inferred_arch_allows_repair_without_a_rocm_version(monkeypatch):
    assert _plan(monkeypatch, amd = False, rocm_version = None, inferred_gfx = "gfx1151")


@pytest.mark.parametrize("version", ["", "2.11.0+rocm7.2"])
def test_missing_or_corrupt_torch_import_forces_the_pass(monkeypatch, version):
    """A tag on disk cannot make an unimportable torch healthy."""
    assert _plan(monkeypatch, imports_as_rocm = False, version = version) is not None


def test_explicit_rocm_pin_repairs_without_a_hardware_probe(monkeypatch):
    assert _plan(
        monkeypatch,
        amd = False,
        nvidia = True,
        version = "2.11.0+cpu",
        rocm_pin = "https://download.pytorch.org/whl/rocm7.2",
    )


def test_explicit_rocm_family_mismatch_forces_the_pass(monkeypatch):
    assert _plan(
        monkeypatch,
        imports_as_rocm = True,
        version = "2.10.0+rocm6.4",
        rocm_pin = "https://download.pytorch.org/whl/rocm7.2",
    )


def test_matching_explicit_rocm_family_keeps_the_fast_path(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.10.0+rocm6.4",
            rocm_pin = "https://download.pytorch.org/whl/rocm6.4",
        )
        is None
    )


def test_explicit_gfx_sibling_family_forces_the_pass(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.13.0",
            rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
            installed_rocm_family = "gfx1150",
        )
        is not None
    )


def test_matching_explicit_gfx_family_keeps_the_fast_path(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.13.0",
            rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
            installed_rocm_family = "gfx1151",
        )
        is None
    )


def test_unknown_explicit_gfx_family_keeps_the_fast_path(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.13.0",
            rocm_pin = "https://repo.amd.com/rocm/whl/gfx1151",
            installed_rocm_family = None,
        )
        is None
    )


def test_strix_generic_rocm_wheel_forces_the_pass(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.2",
            gfx_devices = ["gfx1151"],
        )
        is not None
    )


def test_strix_matching_arch_wheel_keeps_the_fast_path(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.13.0",
            gfx_devices = ["gfx1151"],
            installed_rocm_family = "gfx1151",
        )
        is None
    )


def test_strix_sibling_arch_wheel_forces_the_pass(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.13.0",
            gfx_devices = ["gfx1151"],
            installed_rocm_family = "gfx1150",
        )
        is not None
    )


def test_visible_non_strix_gpu_does_not_trigger_present_strix_repair(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.2",
            gfx_devices = ["gfx1100", "gfx1151"],
            gfx_probe = "amd-smi",
        )
        is None
    )


def test_gfx906_newer_rocm_wheel_forces_the_pass(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.11.0+rocm7.2",
            gfx_devices = ["gfx906"],
        )
        is not None
    )


def test_gfx906_legacy_rocm_wheel_keeps_the_fast_path(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = "2.7.0+rocm6.3",
            gfx_devices = ["gfx906"],
        )
        is None
    )


def test_display_probe_does_not_duplicate_the_fast_path_decision():
    src = SETUP_SH.read_text(encoding = "utf-8")
    probe = src.find("if _setup_has_usable_nvidia_gpu; then")
    fast_path = src.find("# ── Check if Python deps need updating ──")
    decision = src.find('if [ "$_SKIP_PYTHON_DEPS" = false ]; then')
    assert 0 <= fast_path < decision < probe


def test_fast_path_checks_the_active_torch_import_with_a_timeout():
    source = STACK_PATH.read_text(encoding = "utf-8")
    start = source.index("def _probe_rocm_torch()")
    end = source.index("\ndef _rocm_packages_for_index", start)
    block = source[start:end]
    assert "import torch" in block
    assert "timeout = 90" in block


def test_fast_path_and_repair_use_the_same_plan():
    source = STACK_PATH.read_text(encoding = "utf-8")
    ensure = source[source.index("def _ensure_rocm_torch()") :]
    main = source[source.index('if __name__ == "__main__":') :]
    assert "plan, rocm_torch_ready, _runtime_is_gfx906 = _linux_rocm_torch_plan()" in ensure
    assert "_linux_rocm_torch_plan()[0] is not None" in main


def test_shell_bounds_and_diagnoses_the_delegated_probe():
    source = SETUP_SH.read_text(encoding = "utf-8")
    start = source.index("            _setup_rocm_fast_path_probe()")
    block = source[start : start + 3500]
    assert 'timeout --kill-after=5 180 "$VENV_DIR/bin/python"' in block
    assert "--rocm-fast-path-needs-repair" in block
    assert "--amd-detected" not in block
    assert "--nvidia-detected" not in block
    assert 'verbose_substep "ROCm reconciliation probe failed' in block
    stack_source = STACK_PATH.read_text(encoding = "utf-8")
    assert "else 3" in stack_source or "else 3)" in stack_source


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
        "signal.alarm(60)" in block and "timeout --kill-after=5 60" in block
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


def test_summary_reports_an_explicit_non_rocm_backend_separately():
    src = SETUP_SH.read_text(encoding = "utf-8")
    assert 'case "$_setup_non_rocm_backend" in cpu|cuda|xpu)' in src
    assert "PyTorch GPU use is disabled by the explicit" in src
    assert "amdgpu kernel driver" in src
    assert src.index("PyTorch GPU use is disabled by the explicit") < src.index(
        "Check that the amdgpu kernel driver"
    )
