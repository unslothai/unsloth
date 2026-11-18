# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mixed NVIDIA + AMD host on the CUDA default: install.sh says the AMD card was skipped (#12248)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_INSTALL_SH = Path(__file__).resolve().parents[3] / "install.sh"
_CUDA = "https://download.pytorch.org/whl/cu130"
_ROCM = "https://download.pytorch.org/whl/rocm7.0"
_FLAG = "UNSLOTH_FORCE_ROCM_TORCH=1"


def _shell_function(name: str) -> str:
    """The text of one function from install.sh, by brace matching."""
    lines = _INSTALL_SH.read_text(encoding = "utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated function {name}")


def _fake_rocminfo(devices: "list[str]") -> str:
    blocks = [
        "Agent 1",
        "*******",
        "  Name:                    AMD Ryzen 9",
        "  Device Type:             CPU",
    ]
    for _i, _gfx in enumerate(devices):
        blocks += [
            "*******",
            f"Agent {_i + 2}",
            "*******",
            f"  Name:                    {_gfx}",
            "  Marketing Name:          AMD Radeon Graphics",
            "  Device Type:             GPU",
        ]
    return "rocminfo() { cat <<'ROCMINFO_EOF'\n" + "\n".join(blocks) + "\nROCMINFO_EOF\n}"


def _host(
    *,
    nvidia: bool,
    amd: "list[str]",
    rocm_sees_amd: bool = True,
    closed_nodes: "list[str] | None" = None,
) -> str:
    """Hardware probes pinned; everything that decides lives in the lifted install.sh text."""
    rocm = rocm_sees_amd and bool(amd)
    probe = "printf '%s\\n' " + " ".join(amd) if rocm else ":"
    return "\n".join(
        [
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            # The real veto is not under test here; the notice always asks with ignore-nvidia.
            f'_has_amd_rocm_gpu() {{ [ "${{1:-}}" = ignore-nvidia ] || return 1; return {0 if rocm else 1}; }}',
            f"_amd_gpu_present_via_pci() {{ return {0 if amd else 1}; }}",
            f"_probe_amd_gfx_arch() {{ {probe}; }}",
            f"_kfd_gfx_targets() {{ {probe}; }}",
            f"_infer_linux_amd_gfx_arch() {{ {'echo ' + amd[0] if amd else 'return 1'}; }}",
            "_detect_rocm_version_tag() { echo rocm7.0; }",
            "_ensure_rocm_probe_env() { :; }",
            # No rocminfo at all when ROCm is not installed: the notice asks `command -v`.
            _fake_rocminfo(amd) if rocm else "",
            f"_amd_nodes_closed_to_this_user() {{ printf '%s\\n' {' '.join(closed_nodes or [])}; }}"
            if closed_nodes
            else "_amd_nodes_closed_to_this_user() { :; }",
            "_amd_node_repairs() { printf 'join:render\\njoin:video\\n'; }",
            "_amd_render_node_present() { return 0; }",
            "_an_amd_render_node_is_open() { return 1; }",
            "_run_may_open_kfd() { return 1; }",
        ]
    )


_LIFTED = [
    "_rocm_torch_explicitly_requested",
    "_torch_index_url_leaf",
    "_amd_hardware_corroborated",
    "_amd_arch_index_family_for_gfx",
    "_amd_gfx_has_wheel_route",
    "_amd_generic_tag_carries_gfx",
    "_amd_visible_masks_select_no_gpu",
    "_amd_mask_survivors",
    "_amd_runtime_gfx_target",
    "_amd_gfx_is_shadowing_integrated",
    "_amd_gfx_is_strix_apu",
    "_amd_prefer_discrete_gfx",
    "_rocminfo_gpu_records",
    "_amd_ordered_gfx_devices",
    "_amd_request_has_a_wheel_route",
    "substep",
    "_shell_quote",
    "_print_closed_amd_node_repairs",
    "_notice_if_cuda_skipped_amd",
]


def _notice(
    host: str,
    url: str,
    *,
    env: str = "",
    tauri: bool = False,
) -> "subprocess.CompletedProcess[str]":
    script = "\n".join(
        [
            "set -e",
            f"TAURI_MODE={'true' if tauri else 'false'}",
            "C_DIM='' C_RST='' C_WARN=''",
            *(_shell_function(name) for name in _LIFTED),
            host,
            f"{env} _notice_if_cuda_skipped_amd '{url}'",
        ]
    )
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("VISIBLE_DEVICES")
        and k not in ("UNSLOTH_ROCM_GFX_ARCH", "UNSLOTH_FORCE_ROCM_TORCH")
    }
    # Keep a real rocminfo / amd-smi on a ROCm host out of the "not installed" cases.
    clean["PATH"] = "/usr/bin:/bin"
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = clean)
    assert out.returncode == 0, out.stderr
    return out


_R9700 = ["gfx1201"]


def test_a_mixed_host_on_cuda_is_told_about_the_amd_card():
    """Fails before the fix: install.sh picked CUDA torch and said nothing about the R9700."""
    out = _notice(_host(nvidia = True, amd = _R9700), _CUDA)
    assert "An AMD GPU (gfx1201) was also found" in out.stderr
    assert _FLAG in out.stderr
    assert "drops out of training" in out.stderr
    assert "Settings > System > GGUF inference engine > Compute backend" in out.stderr
    assert "ROCm is not installed for the AMD GPU" not in out.stderr
    assert out.stdout == ""


def test_a_mixed_host_without_rocm_is_told_to_install_it():
    """The second report's box before ROCm went on: the card is on the bus, rocminfo is absent."""
    out = _notice(_host(nvidia = True, amd = _R9700, rocm_sees_amd = False), _CUDA)
    assert _FLAG in out.stderr
    assert "ROCm is not installed for the AMD GPU" in out.stderr


def test_unsloth_desktop_gets_it_on_stdout():
    """The Tauri installer shows stdout lines and only logs stderr."""
    out = _notice(_host(nvidia = True, amd = _R9700), _CUDA, tauri = True)
    assert _FLAG in out.stdout
    assert out.stderr == ""


def test_the_flag_already_set_prints_nothing_new():
    """Under the flag get_torch_index_url prints its own FORCE lines, or the CUDA-fallback warning."""
    for url in (_CUDA, _ROCM):
        out = _notice(_host(nvidia = True, amd = _R9700), url, env = _FLAG)
        assert out.stdout == "" and out.stderr == ""


@pytest.mark.parametrize(
    "nvidia, amd, url",
    [
        (True, [], _CUDA),
        (False, _R9700, _ROCM),
        (True, _R9700, _ROCM),
        (True, _R9700, "https://download.pytorch.org/whl/cpu"),
    ],
    ids = ["nvidia-only", "amd-only", "mixed-host-on-rocm", "mixed-host-on-cpu"],
)
def test_nothing_new_off_the_mixed_cuda_case(nvidia, amd, url):
    out = _notice(_host(nvidia = nvidia, amd = amd), url)
    assert out.stdout == "" and out.stderr == ""


def test_a_ryzen_igpu_next_to_an_nvidia_card_is_not_advertised():
    """A Ryzen 7000 desktop's own iGPU (gfx1036) is AMD display class on every RTX box."""
    out = _notice(_host(nvidia = True, amd = ["gfx1036"]), _CUDA)
    assert out.stdout == "" and out.stderr == ""


def test_a_discrete_card_beside_the_igpu_still_is():
    out = _notice(_host(nvidia = True, amd = ["gfx1036", "gfx1201"]), _CUDA)
    assert "An AMD GPU (gfx1201) was also found" in out.stderr


def test_an_amd_arch_with_no_wheels_is_not_offered_the_flag():
    """The flag would keep CUDA there anyway (get_torch_index_url's wheel-route gate)."""
    out = _notice(_host(nvidia = True, amd = ["gfx1010"]), _CUDA)
    assert out.stdout == "" and out.stderr == ""


def test_a_strix_apu_next_to_an_nvidia_card_is_not_advertised():
    """Strix Point / Halo are iGPUs too; _amd_gfx_is_strix_apu is the table _amd_prefer_discrete_gfx uses."""
    for gfx in ("gfx1150", "gfx1151"):
        out = _notice(_host(nvidia = True, amd = [gfx]), _CUDA)
        assert out.stdout == "" and out.stderr == "", gfx


def test_closed_device_nodes_get_the_group_repair():
    """rocminfo is there but /dev/kfd is not openable: the installer's own repair text, reused."""
    out = _notice(_host(nvidia = True, amd = _R9700, closed_nodes = ["/dev/kfd"]), _CUDA)
    assert _FLAG in out.stderr
    assert "cannot open its device nodes" in out.stderr
    assert "/dev/kfd" in out.stderr
    assert "sudo usermod -a -G render,video " in out.stderr
    assert "ROCm is not installed for the AMD GPU" not in out.stderr


def test_open_device_nodes_get_no_repair():
    out = _notice(_host(nvidia = True, amd = _R9700), _CUDA)
    assert "usermod" not in out.stderr


@pytest.mark.parametrize("mask", ["HIP_VISIBLE_DEVICES=0", "ROCR_VISIBLE_DEVICES=0"])
def test_an_amd_mask_without_the_flag_is_told_the_flag_is_needed(mask):
    out = _notice(_host(nvidia = True, amd = _R9700), _CUDA, env = mask)
    assert "only choose among AMD GPUs" in out.stderr
    assert _FLAG in out.stderr


def test_no_amd_mask_no_mask_line():
    out = _notice(_host(nvidia = True, amd = _R9700), _CUDA)
    assert "only choose among AMD GPUs" not in out.stderr


def test_it_is_one_notice():
    out = _notice(
        _host(nvidia = True, amd = _R9700, closed_nodes = ["/dev/kfd"]),
        _CUDA,
        env = "HIP_VISIBLE_DEVICES=0",
    )
    assert out.stderr.count("was also found") == 1
    assert out.stderr.count("Compute backend") == 1


def test_the_installer_calls_it_once_after_the_index_is_final():
    """After the AMD diagnostics it reuses, never for a pinned index or --no-torch."""
    body = _INSTALL_SH.read_text(encoding = "utf-8")
    calls = [
        line.strip()
        for line in body.splitlines()
        if "_notice_if_cuda_skipped_amd " in line and not line.lstrip().startswith("#")
    ]
    assert calls == ['_notice_if_cuda_skipped_amd "$TORCH_INDEX_URL"']
    call = body.index(calls[0])
    guard = body.rindex("\nif ", 0, call)
    assert (
        body[guard:call]
        .strip()
        .startswith('if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_pinned" = false ]; then')
    )
    # The repair function is defined (top-level, so in order) before the call runs.
    assert body.index("\n_print_closed_amd_node_repairs() {") < call
    assert body.index("_print_closed_amd_node_repairs\nelif") < call
