# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for AMD marketing-name and torch fast-path reconciliation."""

from __future__ import annotations

import dataclasses
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

AMD_SMI_MULTI_GPU_BLANK_FIRST = """\
GPU: 0
    ASIC:
        MARKET_NAME:
        TARGET_GRAPHICS_VERSION: gfx1100
GPU: 1
    ASIC:
        MARKET_NAME: AMD Radeon 8060S
        TARGET_GRAPHICS_VERSION: gfx1151
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

AMD_SMI_RECORD_HELPERS = [
    (INSTALL_SH, "_amd_smi_marketing_records"),
    (SETUP_SH, "_setup_amd_smi_marketing_records"),
]


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


@pytest.mark.parametrize("script, fn", AMD_SMI_RECORD_HELPERS, ids = RECORD_IDS)
def test_amd_smi_blank_marketing_name_preserves_its_device_slot(tmp_path, script, fn):
    assert _run_gpu_records(tmp_path, script, fn, AMD_SMI_MULTI_GPU_BLANK_FIRST).splitlines() == [
        "|",
        "|AMD Radeon 8060S",
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


def _plan_tuple(
    monkeypatch,
    *,
    imports_as_rocm = False,
    version = "2.11.0+cpu",
    importable = None,
    **state,
):
    # Unset means "the version string alone decides": a blank one is what an
    # unimportable torch reports.
    if importable is None:
        importable = imports_as_rocm or bool(version)
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
        "selected_strix_result": None,
        "recorded_attempt": None,
        "recorded_pin": None,
        "cvd_hides_nvidia": False,
        "physical_nvidia": False,
    }
    defaults.update(state)
    monkeypatch.setattr(
        stack.install_manifest,
        "recorded_rocm_repair_attempt",
        lambda *_a, **_k: defaults["recorded_attempt"],
    )
    monkeypatch.setattr(
        stack.install_manifest,
        "recorded_torch_index_url",
        lambda *_a, **_k: defaults["recorded_pin"],
    )
    # Captured at import, before the dependency pass drops the manifest.
    monkeypatch.setattr(stack, "_RECORDED_TORCH_INDEX_URL", defaults["recorded_pin"])
    monkeypatch.setattr(stack, "_cvd_hides_nvidia", lambda: defaults["cvd_hides_nvidia"])
    monkeypatch.setattr(stack, "_has_physical_nvidia_gpu", lambda: defaults["physical_nvidia"])
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
    monkeypatch.setattr(stack, "_probe_rocm_torch", lambda: (imports_as_rocm, version, importable))
    monkeypatch.setattr(
        stack, "_detect_amd_gfx_codes", lambda **_kwargs: list(defaults["gfx_devices"])
    )
    monkeypatch.setattr(stack, "_LAST_AMD_GFX_PROBE", defaults["gfx_probe"])
    if defaults["selected_strix_result"] is not None:
        monkeypatch.setattr(
            stack,
            "_selected_linux_strix_gfx",
            lambda *_args, **_kwargs: defaults["selected_strix_result"],
        )
    return stack._linux_rocm_torch_plan()


def _plan(monkeypatch, **kwargs):
    return _plan_tuple(monkeypatch, **kwargs)[0]


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


@pytest.mark.parametrize("version", ["2.12.0+rocm7.13.0", "3.0.0+rocm7.13.0"])
def test_strix_matching_arch_wheel_outside_supported_release_forces_the_pass(monkeypatch, version):
    assert (
        _plan(
            monkeypatch,
            imports_as_rocm = True,
            version = version,
            gfx_devices = ["gfx1151"],
            installed_rocm_family = "gfx1151",
        )
        is not None
    )


def test_matching_strix_wheel_with_a_confirmed_spoof_is_clear_only(monkeypatch):
    plan = _plan(
        monkeypatch,
        imports_as_rocm = True,
        version = "2.11.0+rocm7.13.0",
        installed_rocm_family = "gfx1151",
        selected_strix_result = ("gfx1151", "gfx1151", "gfx1151", {"gfx1151"}),
    )
    assert plan is not None
    assert plan.install_torch is False
    assert plan.clear_hsa_spoof_gfx == "gfx1151"
    assert stack._linux_rocm_fast_path_exit_code(plan) == 4


def test_linux_rocm_fast_path_encodes_install_and_clear_independently():
    plan = stack._LinuxRocmTorchPlan(
        index_url = "https://repo.amd.com/rocm/whl/gfx1151",
        packages = ("torch", "torchvision", "torchaudio"),
        label = "test",
        reason = "test",
    )
    assert stack._linux_rocm_fast_path_exit_code(None) == 3
    assert stack._linux_rocm_fast_path_exit_code(plan) == 0
    assert (
        stack._linux_rocm_fast_path_exit_code(
            dataclasses.replace(plan, install_torch = False, clear_hsa_spoof_gfx = "gfx1151")
        )
        == 4
    )
    assert (
        stack._linux_rocm_fast_path_exit_code(
            dataclasses.replace(plan, clear_hsa_spoof_gfx = "gfx1151")
        )
        == 5
    )


def test_clear_only_plan_does_not_reinstall_torch(monkeypatch):
    plan = stack._LinuxRocmTorchPlan(
        index_url = "https://repo.amd.com/rocm/whl/gfx1151",
        packages = ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"),
        label = "ROCm torch (Strix arch-specific)",
        reason = "matching wheel",
        install_torch = False,
        clear_hsa_spoof_gfx = "gfx1151",
    )
    cleared = []
    installs = []
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_linux_rocm_torch_plan", lambda: (plan, True, False))
    monkeypatch.setattr(stack, "_clear_confirmed_hsa_spoof", cleared.append)
    monkeypatch.setattr(stack, "_bnb_rocm_prerelease_url", lambda: None)
    monkeypatch.setattr(stack, "_bnb_rocm_arch_has_binary", lambda: True)
    monkeypatch.setattr(stack, "pip_install", lambda *args, **kwargs: installs.append(args))

    stack._ensure_rocm_torch()

    assert cleared == ["gfx1151"]
    assert installs
    assert all("--index-url" not in call for call in installs)
    assert all(call[0] != plan.label for call in installs)


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
    assert "_linux_rocm_fast_path_exit_code(_linux_rocm_torch_plan()[0])" in main


def test_shell_bounds_and_diagnoses_the_delegated_probe():
    source = SETUP_SH.read_text(encoding = "utf-8")
    assert "--kill-after" not in source
    assert source.count("timeout -k 5") == 3
    start = source.index("_setup_rocm_fast_path_probe() {")
    block = source[start : source.index("_SKIP_PYTHON_DEPS=false\n_SKIP_VERSION_CHECK", start)]
    assert 'timeout -k 5 180 "$VENV_DIR/bin/python"' in block
    assert "--rocm-fast-path-needs-repair" in block
    assert "--amd-detected" not in block
    assert "--nvidia-detected" not in block
    assert "unset HSA_OVERRIDE_GFX_VERSION" in block
    assert '"$_setup_rocm_repair_rc" -eq 4' in block
    assert '"$_setup_rocm_repair_rc" -eq 5' in block
    decision = source[
        source.index('if [ "$_SKIP_PYTHON_DEPS" = true ]; then\n            _setup_run') :
    ]
    assert 'verbose_substep "ROCm reconciliation probe failed' in decision[:1200]
    stack_source = STACK_PATH.read_text(encoding = "utf-8")
    assert "def _linux_rocm_fast_path_exit_code" in stack_source
    assert "return 3" in stack_source


def test_a_forced_dependency_pass_still_applies_the_hsa_clear_in_this_shell():
    """install_python_stack can only unset the override inside its own process, so the
    per-architecture wheel it just installed would still be read through the spoofed ISA."""
    source = SETUP_SH.read_text(encoding = "utf-8")
    start = source.index('if [ "$_SKIP_PYTHON_DEPS" = false ]; then\n    install_python_stack')
    block = source[start : source.index("\nelse", start)]
    assert '[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]' in block
    assert "_setup_run_rocm_fast_path_probe" in block
    assert "_setup_apply_rocm_hsa_clear" in block
    # Defined unconditionally: the fast-path block that used to own it never runs here.
    assert source.index("_setup_rocm_fast_path_probe() {") < start
    assert source.index("_setup_apply_rocm_hsa_clear() {") < start


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
        "signal.alarm(60)" in block and "timeout -k 5 60" in block
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


# ── An unimportable torch must not re-arm the repair forever (#8473) ──

ROCMINFO_APU_GPU_WITHOUT_GFX = """\
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
  Marketing Name:          AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    AMD Radeon Graphics
  Marketing Name:          AMD Radeon 780M Graphics
  Vendor Name:             AMD
  Device Type:             GPU
"""

ROCMINFO_NAME_WITH_EMBEDDED_COLON = """\
*******
Agent 1
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X OAM: 750W SKU
  Vendor Name:             AMD
  Device Type:             GPU
"""


def _probe_with(
    monkeypatch,
    *,
    returncode = 0,
    stdout = b"",
    raises = None,
):
    def fake_run(*_args, **_kwargs):
        if raises is not None:
            raise raises
        result = dataclasses.make_dataclass("R", ["returncode", "stdout"])
        return result(returncode, stdout)

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    return stack._probe_rocm_torch()


def test_probe_reports_a_healthy_rocm_torch_as_importable(monkeypatch):
    assert _probe_with(monkeypatch, stdout = b"6.3.42131|2.7.0+rocm6.3\n") == (
        True,
        "2.7.0+rocm6.3",
        True,
    )


def test_probe_reports_a_cpu_wheel_as_importable_but_not_rocm(monkeypatch):
    assert _probe_with(monkeypatch, stdout = b"|2.9.0+cpu\n") == (False, "2.9.0+cpu", True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"returncode": -6},
        {"returncode": 1},
        {"raises": OSError("no interpreter")},
        {"raises": subprocess.TimeoutExpired(cmd = "python", timeout = 90)},
    ],
    ids = ["segfault", "import-error", "oserror", "timeout"],
)
def test_probe_reports_an_unrunnable_torch_as_not_importable(monkeypatch, kwargs):
    """A torch that aborts, raises or hangs is BROKEN, not the wrong wheel."""
    assert _probe_with(monkeypatch, **kwargs)[2] is False


def test_an_unimportable_torch_is_repaired_on_the_first_attempt(monkeypatch):
    plan = _plan(monkeypatch, version = "", importable = False)
    assert plan is not None and plan.install_torch is True
    assert plan.repair_key


def test_the_same_failed_repair_is_not_repeated(monkeypatch):
    """Without this the fast path force-reinstalls GB on every `studio update`, forever."""
    first = _plan(monkeypatch, version = "", importable = False)
    repeat = _plan(monkeypatch, version = "", importable = False, recorded_attempt = first.repair_key)
    assert repeat is not None
    assert repeat.blocked is True
    assert repeat.install_torch is False
    assert stack._linux_rocm_fast_path_exit_code(repeat) == 3


def test_a_blocked_repair_is_not_reported_as_a_converged_one(monkeypatch):
    """torch_ready would clear the ledger, re-arming the same multi-GB reinstall."""
    first = _plan(monkeypatch, version = "", importable = False)
    _, torch_ready, _ = _plan_tuple(
        monkeypatch, version = "", importable = False, recorded_attempt = first.repair_key
    )
    assert torch_ready is False


def test_a_blocked_repair_does_not_clear_a_confirmed_hsa_spoof(monkeypatch):
    """The un-repaired generic wheel may run only through that override."""
    strix = ("gfx1151", "gfx1151", "gfx1151", {"gfx1151"})
    first = _plan(
        monkeypatch,
        version = "2.10.0+rocm6.4",
        imports_as_rocm = True,
        installed_rocm_family = "gfx1150",
        selected_strix_result = strix,
    )
    assert first.install_torch is True and first.clear_hsa_spoof_gfx == "gfx1151"
    repeat = _plan(
        monkeypatch,
        version = "2.10.0+rocm6.4",
        imports_as_rocm = True,
        installed_rocm_family = "gfx1150",
        selected_strix_result = strix,
        recorded_attempt = first.repair_key,
    )
    assert repeat.blocked is True
    assert repeat.clear_hsa_spoof_gfx is None
    assert stack._linux_rocm_fast_path_exit_code(repeat) == 3


def test_a_repair_that_changed_the_installed_torch_is_still_attempted(monkeypatch):
    """The key covers the observed state, so a moved-on host is never blocked by it."""
    stale = _plan(monkeypatch, version = "2.9.0+cpu").repair_key
    fresh = _plan(monkeypatch, version = "2.10.0+cpu", recorded_attempt = stale)
    assert fresh is not None and fresh.install_torch is True and fresh.blocked is False


def test_an_unrelated_recorded_attempt_does_not_block_a_repair(monkeypatch):
    plan = _plan(monkeypatch, version = "", importable = False, recorded_attempt = "0" * 32)
    assert plan is not None and plan.install_torch is True and plan.blocked is False


def test_the_repair_key_is_not_the_raw_index_url(monkeypatch):
    """It is written to disk and a pinned index may carry credentials."""
    plan = _plan(monkeypatch, version = "")
    assert "://" not in plan.repair_key and plan.index_url not in plan.repair_key


def test_the_attempt_is_recorded_before_the_install_runs():
    """pip_install exits the process on failure; an unrecorded attempt is a repeated one."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    block = source[source.index("def _ensure_rocm_torch()") :]
    assert block.index("record_rocm_repair_attempt(plan.repair_key)") < block.index(
        "pip_install(\n"
    )


def _run_ensure_rocm_torch(monkeypatch, plan, torch_ready):
    cleared = []
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_linux_rocm_torch_plan", lambda: (plan, torch_ready, False))
    monkeypatch.setattr(stack, "_clear_confirmed_hsa_spoof", lambda *_a: None)
    monkeypatch.setattr(stack, "_bnb_rocm_prerelease_url", lambda: None)
    monkeypatch.setattr(stack, "_bnb_rocm_arch_has_binary", lambda: True)
    monkeypatch.setattr(stack, "pip_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(stack, "pip_install_try", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        stack.install_manifest,
        "clear_rocm_repair_attempt",
        lambda *_a, **_k: cleared.append(True),
    )
    stack._ensure_rocm_torch()
    return cleared


def test_a_converged_host_forgets_the_last_attempt(monkeypatch):
    """Otherwise a later, genuinely new breakage in the same state stays unrepairable."""
    assert _run_ensure_rocm_torch(monkeypatch, None, True) == [True]


def test_a_blocked_repair_keeps_the_ledger(monkeypatch):
    """Clearing it here is what recreated the loop the ledger exists to stop."""
    blocked = stack._LinuxRocmTorchPlan(
        index_url = "https://download.pytorch.org/whl/rocm7.2",
        packages = ("torch", "torchvision", "torchaudio"),
        label = "ROCm torch (rocm7.2)",
        reason = "ROCm 7.2",
        install_torch = False,
        clear_hsa_spoof_gfx = None,
        repair_key = "a" * 32,
        blocked = True,
    )
    assert _run_ensure_rocm_torch(monkeypatch, blocked, False) == []


def test_the_repair_ledger_survives_the_manifest_being_dropped(tmp_path):
    """It records an attempt made DURING the pass that deletes the manifest."""
    import install_manifest

    install_manifest.record_rocm_repair_attempt("abc123", root = tmp_path)
    install_manifest.write_manifest(root = tmp_path, req_root = tmp_path)
    install_manifest.remove_manifest(root = tmp_path)
    assert install_manifest.recorded_rocm_repair_attempt(root = tmp_path) == "abc123"
    install_manifest.clear_rocm_repair_attempt(root = tmp_path)
    assert install_manifest.recorded_rocm_repair_attempt(root = tmp_path) is None


# ── A deliberate non-ROCm pin must outlive the run that set it ──


@pytest.mark.parametrize(
    "pin",
    [
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu128",
        "https://mirror.internal/simple",
    ],
    ids = ["cpu", "cuda", "custom"],
)
def test_a_recorded_non_rocm_pin_survives_a_later_update(monkeypatch, pin):
    """UNSLOTH_TORCH_INDEX_URL is the user's, so `studio update` never sees it."""
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    assert _plan(monkeypatch, recorded_pin = pin) is None


@pytest.mark.parametrize(
    "pin",
    ["https://download.pytorch.org/whl/rocm6.3", "https://download.pytorch.org/whl/gfx1151"],
    ids = ["rocm", "gfx"],
)
def test_a_recorded_rocm_pin_still_allows_the_repair(monkeypatch, pin):
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    assert _plan(monkeypatch, recorded_pin = pin) is not None


def test_this_runs_rocm_pin_overrides_a_recorded_cpu_one(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/rocm6.3")
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    assert _plan(monkeypatch, recorded_pin = "https://download.pytorch.org/whl/cpu") is not None


def test_write_manifest_records_an_explicit_pin(tmp_path):
    import install_manifest
    install_manifest.write_manifest(
        root = tmp_path,
        req_root = tmp_path,
        torch_index_url = "https://download.pytorch.org/whl/cpu",
    )
    assert (
        install_manifest.recorded_torch_index_url(root = tmp_path)
        == "https://download.pytorch.org/whl/cpu"
    )


def test_write_manifest_omits_an_absent_pin(tmp_path):
    """Absent means unknown; a recorded value must never be a guess."""
    import install_manifest

    install_manifest.write_manifest(root = tmp_path, req_root = tmp_path)
    manifest = install_manifest.read_manifest(root = tmp_path)
    assert "torch_index_url" not in manifest
    assert install_manifest.recorded_torch_index_url(root = tmp_path) is None


def test_the_installer_records_the_explicit_pin_and_not_the_detected_backend():
    """install.sh invents UNSLOTH_TORCH_BACKEND from autodetection; freezing it would
    outlive the host it described."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    assert "torch_index_url = _manifest_torch_index_url()," in source
    manifest_source = (REPO_ROOT / "studio" / "install_manifest.py").read_text(encoding = "utf-8")
    assert "torch_backend" not in manifest_source


# ── The recorded pin must survive the pass that drops the manifest ──


def test_the_recorded_pin_is_read_before_the_dependency_pass_drops_the_manifest(
    monkeypatch, tmp_path
):
    """A lazy read finds nothing: remove_manifest() runs first, so the CPU or CUDA index
    the user chose would be silently replaced with ROCm on the first update."""
    import install_manifest

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    pin = "https://download.pytorch.org/whl/cpu"
    install_manifest.write_manifest(root = tmp_path, req_root = tmp_path, torch_index_url = pin)
    monkeypatch.setattr(stack, "_RECORDED_TORCH_INDEX_URL", pin)
    install_manifest.remove_manifest(root = tmp_path)
    assert stack._recorded_non_rocm_torch_pin() == pin
    assert stack._manifest_torch_index_url() == pin


def test_this_runs_explicit_pin_replaces_the_recorded_one(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "rocm7.2")
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.setattr(stack, "_RECORDED_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
    assert stack._manifest_torch_index_url() == "https://download.pytorch.org/whl/rocm7.2"


@pytest.mark.parametrize(
    "pin, stored",
    [
        ("https://user:tok@mirror.internal/whl/cpu", "https://mirror.internal/whl/cpu"),
        ("https://mirror.internal/whl/cpu?token=abc", "https://mirror.internal/whl/cpu"),
    ],
    ids = ["userinfo", "query"],
)
def test_the_recorded_pin_carries_no_credentials(monkeypatch, pin, stored):
    """The manifest is written with the ordinary umask; reconciliation only needs the family."""
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", pin)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.setattr(stack, "_RECORDED_TORCH_INDEX_URL", None)
    recorded = stack._manifest_torch_index_url()
    assert recorded == stored
    assert "tok" not in recorded and "abc" not in recorded


# ── A hidden NVIDIA GPU must not cost the user a working CUDA torch ──


def test_the_physical_probe_ignores_the_visibility_mask(monkeypatch):
    monkeypatch.setattr(stack.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda *_a, **_k: dataclasses.make_dataclass("R", ["returncode", "stdout"])(
            0, "GPU 0: NVIDIA Fake (UUID: GPU-x)\n"
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert stack._has_physical_nvidia_gpu() is True
    # The mask still hides it from the routing question, which #6183 pinned.
    assert stack._has_usable_nvidia_gpu() is False


def test_a_hidden_nvidia_gpu_keeps_a_working_cuda_torch(monkeypatch):
    """CUDA_VISIBLE_DEVICES=-1 is the ordinary "run on CPU" idiom, not consent to
    replace several GB of working CUDA wheels with ROCm ones."""
    assert (
        _plan(
            monkeypatch,
            version = "2.9.0+cu128",
            cvd_hides_nvidia = True,
            physical_nvidia = True,
        )
        is None
    )


def test_a_hidden_nvidia_gpu_does_not_protect_an_unimportable_torch(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            version = "",
            importable = False,
            cvd_hides_nvidia = True,
            physical_nvidia = True,
        )
        is not None
    )


def test_a_hidden_nvidia_gpu_does_not_protect_a_cpu_wheel(monkeypatch):
    """#6183's mixed-host routing is untouched: only a CUDA build is defended."""
    assert (
        _plan(
            monkeypatch,
            version = "2.9.0+cpu",
            cvd_hides_nvidia = True,
            physical_nvidia = True,
        )
        is not None
    )


def test_cuda_torch_on_a_host_with_no_nvidia_hardware_is_still_repaired(monkeypatch):
    """A venv poisoned with CUDA wheels on an AMD-only box must still be fixed."""
    assert (
        _plan(
            monkeypatch,
            version = "2.9.0+cu128",
            cvd_hides_nvidia = True,
            physical_nvidia = False,
        )
        is not None
    )


def test_an_unmasked_host_is_unaffected(monkeypatch):
    assert _plan(monkeypatch, version = "2.9.0+cu128", physical_nvidia = True) is not None


# ── The marketing-name fallback must never name the CPU (#7307) ──


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_a_gpu_agent_without_a_gfx_token_beats_the_cpu_agent(tmp_path, script, fn):
    """rocminfo lists the CPU first, so latching the FIRST marketing name reports the
    processor as the GPU -- the very bug this file exists to fix."""
    assert _run_gpu_records(tmp_path, script, fn, ROCMINFO_APU_GPU_WITHOUT_GFX) == (
        "|AMD Radeon 780M Graphics"
    )


@pytest.mark.parametrize("script, fn", RECORD_HELPERS, ids = RECORD_IDS)
def test_a_marketing_name_containing_a_colon_is_not_truncated(tmp_path, script, fn):
    assert _run_gpu_records(tmp_path, script, fn, ROCMINFO_NAME_WITH_EMBEDDED_COLON) == (
        "gfx942|AMD Instinct MI300X OAM: 750W SKU"
    )


def test_both_copies_of_the_record_helper_stay_identical():
    install_body = _extract_function(INSTALL_SH, "_rocminfo_gpu_records")
    setup_body = _extract_function(SETUP_SH, "_setup_rocminfo_gpu_records")
    assert install_body.split("{", 1)[1] == setup_body.split("{", 1)[1]


def test_the_record_helper_does_not_split_on_the_first_colon_space():
    for script, fn in RECORD_HELPERS:
        body = _extract_function(script, fn)
        assert "awk -F': '" not in body, f"{script.name}: -F': ' truncates an embedded ': '"
        assert "sub(/^[^:]*:[[:space:]]*/" in body


# ── The repair and the report must see the same hardware ──


def test_the_rocm_probe_environment_is_seeded_before_the_repair_probe():
    """On WSL2 ROCDXG a later export leaves the repairing half and the reporting half
    disagreeing about the same machine."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    export = src.index('export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"')
    path_append = src.index('PATH="$PATH:/opt/rocm/bin"')
    probe = src.index("--rocm-fast-path-needs-repair")
    summary = src.index("# ── GPU detection summary")
    assert export < probe < summary
    assert path_append < probe
