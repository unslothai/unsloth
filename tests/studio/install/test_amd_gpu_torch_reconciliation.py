# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for AMD marketing-name and torch fast-path reconciliation."""

from __future__ import annotations

import pathlib
import subprocess

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"

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


def _extract_function(script: pathlib.Path, name: str) -> str:
    src = script.read_text(encoding = "utf-8")
    start = src.find(f"{name}() {{")
    assert start >= 0, f"{name} missing from {script.name}"
    end = src.find("\n}", start)
    assert end > start, f"{name} in {script.name} is not brace-terminated"
    return src[start : end + 2]


def _run_marketing_name(tmp_path, script: pathlib.Path, name: str, rocminfo: str) -> str:
    probe = tmp_path / "probe.sh"
    probe.write_text(_extract_function(script, name) + f"\n{name}\n")
    return subprocess.run(
        ["bash", str(probe)],
        input = rocminfo,
        capture_output = True,
        text = True,
        timeout = 30,
    ).stdout.strip()


MARKETING_HELPERS = [
    (INSTALL_SH, "_rocminfo_gpu_marketing_name"),
    (SETUP_SH, "_setup_rocminfo_gpu_marketing_name"),
]
MARKETING_IDS = ["install.sh", "setup.sh"]


@pytest.mark.parametrize("script, fn", MARKETING_HELPERS, ids = MARKETING_IDS)
def test_discrete_host_reports_the_gpu_not_the_cpu(tmp_path, script, fn):
    """#7307: the GPU line must name the card, not the processor in front of it."""
    assert _run_marketing_name(tmp_path, script, fn, ROCMINFO_DISCRETE) == "AMD Radeon AI PRO R9700"


@pytest.mark.parametrize("script, fn", MARKETING_HELPERS, ids = MARKETING_IDS)
def test_apu_without_a_gfx_agent_keeps_the_processor_name(tmp_path, script, fn):
    """The fallback feeds name-based arch inference; losing it would drop APUs to CPU torch."""
    assert (
        _run_marketing_name(tmp_path, script, fn, ROCMINFO_APU_NO_GFX)
        == "AMD Ryzen AI Max+ 395 w/ Radeon 8060S"
    )


@pytest.mark.parametrize("script, fn", MARKETING_HELPERS, ids = MARKETING_IDS)
def test_no_rocminfo_output_yields_no_name(tmp_path, script, fn):
    assert _run_marketing_name(tmp_path, script, fn, "") == ""


# ── setup.sh: the dependency fast path must repair a non-ROCm wheel on an AMD host ──


def _extract_amd_escape() -> str:
    """Extract the ROCm leaf predicate and its fast-path arm."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    start = src.find("        _setup_rocm_family_leaf() {")
    assert start >= 0, "setup.sh lost the AMD escape's ROCm-family predicate"
    arm = src.find('substep "AMD GPU detected but installed PyTorch is not a ROCm build', start)
    assert arm > start, "setup.sh lost the AMD dependency-pass escape"
    end = src.find("\n        fi\n", arm)
    assert end > arm, "the AMD escape is not fi-terminated at its own indent"
    return src[start : end + len("\n        fi\n")]


def _run_amd_escape(tmp_path, **state) -> str:
    """Run the escape with the fast path's inputs preset; report the resulting decision."""
    defaults = {
        "_SKIP_PYTHON_DEPS": "true",
        "_setup_pin_leaf": "",
        "_setup_pin_ver": "2.11.0+cpu",
        "_setup_pin_is_xpu": "false",
        "_setup_nvidia_usable": "false",
        "_setup_amd_detected": "true",
    }
    defaults.update(state)
    assigns = "\n".join(f"{k}='{v}'" for k, v in defaults.items())
    probe = tmp_path / "escape.sh"
    probe.write_text(
        "substep() { :; }\n"
        + assigns
        + "\n"
        + _extract_amd_escape()
        + '\necho "$_SKIP_PYTHON_DEPS"\n'
    )
    return subprocess.run(
        ["bash", str(probe)], capture_output = True, text = True, timeout = 30
    ).stdout.strip()


@pytest.mark.parametrize(
    "torch_version",
    ["2.11.0+cpu", "2.10.0", "2.10.0+cu128"],
    ids = ["cpu-wheel", "pypi-cuda-wheel", "cuda-wheel"],
)
def test_non_rocm_wheel_on_an_amd_host_forces_the_pass(tmp_path, torch_version):
    """#8473 / #7275: only the dependency pass installs ROCm torch, so it has to run."""
    assert _run_amd_escape(tmp_path, _setup_pin_ver = torch_version) == "false"


def test_rocm_wheel_keeps_the_fast_path(tmp_path):
    assert _run_amd_escape(tmp_path, _setup_pin_ver = "2.11.0+rocm7.2.3") == "true"


def test_absent_torch_keeps_the_fast_path(tmp_path):
    """A GGUF-only (no-torch) venv has no wheel to repair and must not pay a pass every update."""
    assert _run_amd_escape(tmp_path, _setup_pin_ver = "") == "true"


def test_no_amd_gpu_keeps_the_fast_path(tmp_path):
    assert _run_amd_escape(tmp_path, _setup_amd_detected = "false") == "true"


def test_usable_nvidia_keeps_the_fast_path(tmp_path):
    """A mixed host classified as NVIDIA takes the CUDA route; this escape must not fire."""
    assert _run_amd_escape(tmp_path, _setup_nvidia_usable = "true") == "true"


@pytest.mark.parametrize("leaf", ["cpu", "cu128", "xpu"], ids = ["cpu", "cu128", "xpu"])
def test_explicit_non_rocm_pin_is_respected(tmp_path, leaf):
    """An index pin is the user's answer for this host, not something to repair."""
    assert _run_amd_escape(tmp_path, _setup_pin_leaf = leaf) == "true"


@pytest.mark.parametrize(
    "leaf",
    ["simple", "rocm-current", "rocm7.2-private", "rocm7.", "rocm7.2.1", "gfx-private"],
    ids = ["simple", "rocm-current", "rocm-private", "rocm-partial", "rocm-triple", "gfx-private"],
)
def test_unknown_family_pin_does_not_loop_the_dependency_pass(tmp_path, leaf):
    """_ensure_rocm_torch declines a verbatim pin, so forcing the pass would repair nothing."""
    assert _run_amd_escape(tmp_path, _setup_pin_leaf = leaf) == "true"


@pytest.mark.parametrize(
    "leaf",
    ["", "rocm7.2", "rocm6", "gfx1201", "gfx120x-all"],
    ids = ["unpinned", "rocm-dotted", "rocm-major", "gfx", "gfx-suffixed"],
)
def test_rocm_pins_still_repair(tmp_path, leaf):
    assert _run_amd_escape(tmp_path, _setup_pin_leaf = leaf) == "false"


def test_xpu_wheel_is_left_to_the_xpu_escapes(tmp_path):
    assert _run_amd_escape(tmp_path, _setup_pin_is_xpu = "true") == "true"


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


def test_escape_reads_the_wheel_not_the_runtime():
    """Reinstalling cannot fix a driver or permission fault, so the escape must not key on it."""
    assert "torch.cuda.is_available()" not in _extract_amd_escape()


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
