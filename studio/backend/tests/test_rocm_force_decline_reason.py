# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_FORCE_ROCM_TORCH=1 that keeps CUDA torch says why, on install and on a standalone update."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_rocm_torch_explicit_request import (  # noqa: E402
    _clean_env,
    _fake_rocminfo,
    _mixed_host,
    _shell_function,
    _wheel_route_defs,
)

_STACK = Path(__file__).resolve().parents[2] / "install_python_stack.py"
_INSTALL_SH = Path(__file__).resolve().parents[3] / "install.sh"
_CU = "https://download.pytorch.org/whl/cu130"
_CPU = "https://download.pytorch.org/whl/cpu"
_ROCM = "https://repo.amd.com/rocm/whl/gfx120X-all/"


@pytest.fixture
def stack():
    spec = importlib.util.spec_from_file_location("_unsloth_install_python_stack_h", _STACK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shell_function_or_noop(name: str) -> str:
    try:
        return _shell_function(name)
    except StopIteration:
        return f"{name}() {{ :; }}"


def _fallback_block() -> str:
    """install.sh's FORCE-kept-CUDA block, lifted verbatim from its first line to its last fi."""
    text = _INSTALL_SH.read_text(encoding = "utf-8")
    head = 'if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \\\n   _rocm_torch_explicitly_requested && _has_usable_nvidia_gpu; then\n'
    start = text.index(head)
    end = text.index(
        '\nif [ "$SKIP_TORCH" = false ]; then\n    _warn_if_cuda_mask_hides_amd', start
    )
    return text[start:end]


def _install(
    host: "tuple[str, str]",
    *,
    index: str = _CU,
    env: "dict | None" = None,
    machine: str = "x86_64",
    os_name: str = "linux",
    wsl_runtime: bool = False,
    wsl_gpu: str = "",
) -> "tuple[str, str]":
    """Run the block on a stubbed mixed host whose first get_torch_index_url returned ``index``."""
    wsl_name = (
        f"_wsl_amd_gpu_name() {{ printf '%s' '{wsl_gpu}'; }}"
        if wsl_gpu
        else "_wsl_amd_gpu_name() { return 1; }"
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f'uname() {{ case "${{1:-}}" in -m) echo {machine} ;; *) echo Linux ;; esac; }}',
            _shell_function("_rocm_torch_explicitly_requested"),
            "_has_usable_nvidia_gpu() { return 0; }",
            host[0],
            *_wheel_route_defs(rocminfo = host[1]),
            _shell_function("_torch_index_url_leaf"),
            _shell_function("_torch_index_url_is_rocm"),
            f"_rocm_wsl_runtime_present() {{ return {0 if wsl_runtime else 1}; }}",
            wsl_name,
            # Not on base: no-ops there so the block itself runs and the test is red on behaviour.
            *(
                _shell_function_or_noop(name)
                for name in ("_amd_gfx_is_strix_apu", "_amd_route_answer", "_notice_mixed_host_amd")
            ),
            # Closed device nodes are test_rocm_mixed_host_cuda_notice.py's to pin.
            "_amd_nodes_closed_to_this_user() { :; }",
            f"get_torch_index_url() {{ echo {_CU}; }}",
            f"OS={os_name}",
            "_torch_index_pinned=false",
            "SKIP_TORCH=false",
            f"TORCH_INDEX_URL={index}",
            '_force_rocm_kept_cuda=""',
            _fallback_block(),
            # What install.sh runs after the AMD diagnostics; the reason is printed there.
            '_notice_mixed_host_amd "$TORCH_INDEX_URL" "$_force_rocm_kept_cuda"',
            'echo "INDEX=$TORCH_INDEX_URL"',
        ]
    )
    run_env = _clean_env({"UNSLOTH_FORCE_ROCM_TORCH": "1", **(env or {})})
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = run_env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip(), out.stderr


def _host(
    probe: str = "",
    *,
    inferred: str = "",
    pci: bool = True,
    kfd: str = "",
    devices = (),
) -> "tuple[str, str]":
    """(probe stubs, rocminfo) for a stubbed host; rocminfo goes through _wheel_route_defs."""
    stubs = "\n".join(
        [
            f"_probe_amd_gfx_arch() {{ printf '%s' '{probe}'; }}",
            f"_kfd_gfx_targets() {{ printf '%s' '{kfd}'; }}",
            f"_infer_linux_amd_gfx_arch() {{ printf '%s' '{inferred}'; }}",
            f"_amd_gpu_present_via_pci() {{ return {0 if pci else 1}; }}",
        ]
    )
    return stubs, _fake_rocminfo(list(devices)) if devices else "rocminfo() { return 1; }"


_REASON = "[WARN] UNSLOTH_FORCE_ROCM_TORCH is set, but ROCm PyTorch can't be used here: "


@pytest.mark.parametrize(
    "host, env, reason",
    [
        (_host(pci = False), {}, "no AMD GPU was found."),
        (_host("gfx1201"), {"ROCR_VISIBLE_DEVICES": ""}, "ROCR_VISIBLE_DEVICES"),
        (
            _host(pci = False),
            {"UNSLOTH_ROCM_GFX_ARCH": "gfx1201"},
            "UNSLOTH_ROCM_GFX_ARCH=gfx1201 is set, but no AMD GPU was found.",
        ),
        (
            _host("gfx1010"),
            {"UNSLOTH_ROCM_GFX_ARCH": "gfx1010"},
            "there are no ROCm PyTorch wheels for gfx1010 (UNSLOTH_ROCM_GFX_ARCH).",
        ),
        (_host("gfx1010"), {}, "there are no ROCm PyTorch wheels for gfx1010."),
        (_host(inferred = ""), {}, "an AMD GPU was found, but its gfx arch could not be read."),
        (
            _host("gfx1100\ngfx1201"),
            {},
            "AMD GPUs of different archs (gfx1100 gfx1201) and no device order",
        ),
        (
            _host("gfx906\ngfx1100", devices = ["gfx906", "gfx1100"]),
            {},
            "gfx906 only has ROCm 6.3 wheels",
        ),
    ],
    ids = [
        "no-amd-gpu",
        "mask-hides-amd",
        "declared-arch-no-card",
        "declared-arch-no-wheels",
        "probed-arch-no-wheels",
        "arch-unreadable",
        "mixed-archs-no-order",
        "gfx906-beside-another-arch",
    ],
)
def test_install_says_why_the_request_kept_cuda(host, env, reason):
    """Red on base: a declining route returned CUDA straight from get_torch_index_url, so the
    fallback block saw nothing to replace and printed nothing at all."""
    stdout, stderr = _install(host, env = env)
    line = next((l for l in stderr.splitlines() if l.startswith(_REASON)), "")
    assert reason in line, stderr
    assert len([l for l in stderr.splitlines() if l.startswith(_REASON)]) == 1, stderr
    assert stdout.endswith(f"INDEX={_CU}")


def test_a_gate_after_the_route_names_the_arch_and_keeps_cuda():
    """The route picked gfx1201, then a get_torch_index_url gate sent it to CPU: the arch is named
    and the existing CPU-to-CUDA swap still happens."""
    stdout, stderr = _install(_host("gfx1201"), index = _CPU)
    assert f"{_REASON}ROCm PyTorch was ruled out for gfx1201 (see the warning above)." in stderr
    assert "Keeping the CUDA build" in stderr
    assert stdout.endswith(f"INDEX={_CU}")


def test_non_x86_hosts_say_the_wheels_are_x86_only():
    _, stderr = _install(_host("gfx1201"), index = _CPU, machine = "aarch64")
    assert f"{_REASON}ROCm PyTorch wheels are x86_64 only." in stderr


def test_a_viable_request_prints_nothing_new():
    """The control: the route won, the index is ROCm, so the block stays silent."""
    stdout, stderr = _install(_host("gfx1201"), index = _ROCM)
    assert "UNSLOTH_FORCE_ROCM_TORCH" not in stderr, stderr
    assert stdout.endswith(f"INDEX={_ROCM}")


def test_wsl_without_librocdxg_says_the_setup_was_skipped():
    _, stderr = _install(_host(pci = False), os_name = "wsl", wsl_gpu = "AMD Radeon AI PRO R9700")
    assert f"{_REASON}no AMD GPU was found." in stderr
    assert (
        "Windows reports AMD Radeon AI PRO R9700, but ROCm-on-WSL (librocdxg) isn't installed"
        in stderr
    )
    assert "scripts/install_rocm_wsl_strixhalo.sh" in stderr


@pytest.mark.parametrize(
    "wsl_runtime, wsl_gpu, os_name",
    [(True, "AMD Radeon AI PRO R9700", "wsl"), (False, "", "wsl"), (False, "AMD Radeon", "linux")],
    ids = ["runtime-present", "windows-sees-no-amd", "not-wsl"],
)
def test_the_wsl_hint_needs_wsl_an_amd_adapter_and_no_runtime(wsl_runtime, wsl_gpu, os_name):
    _, stderr = _install(
        _host(pci = False), os_name = os_name, wsl_runtime = wsl_runtime, wsl_gpu = wsl_gpu
    )
    assert f"{_REASON}no AMD GPU was found." in stderr
    assert "ROCm-on-WSL" not in stderr


# Standalone update: _TORCH_BACKEND is "" so _ensure_rocm_torch re-decides on its own.


class _Stop(BaseException):
    pass


def _update(
    stack,
    monkeypatch,
    capsys,
    *,
    wsl_gpu: "str | None" = None,
    **host,
) -> str:
    _mixed_host(stack, monkeypatch, **host)
    monkeypatch.setattr(stack, "_wsl_rocm_runtime_present", lambda: False)
    monkeypatch.setattr(stack, "_wsl_amd_adapter_name", lambda: wsl_gpu)

    def _refuse(*a, **k):
        pytest.fail("the update reached an install")

    monkeypatch.setattr(stack, "pip_install", _refuse)
    monkeypatch.setattr(stack, "pip_install_try", _refuse)

    def _probe(*a, **k):
        raise _Stop

    # Past every vendor gate: the first thing after the ROCm version (which the route test reads).
    monkeypatch.setattr(stack, "_probe_torch_runtime", _probe)
    # The update runs the repair twice on Linux; the reason prints once.
    for _ in range(2):
        try:
            stack._ensure_rocm_torch()
        except _Stop:
            pass
    return capsys.readouterr().out


_PY_REASON = "UNSLOTH_FORCE_ROCM_TORCH is set, but ROCm PyTorch can't be used here: "


@pytest.mark.parametrize(
    "host, reason",
    [
        (dict(rocm_gpu = False, display = False, archs = []), "no AMD GPU was found."),
        (dict(archs = ["gfx1010"]), "there are no ROCm PyTorch wheels for gfx1010."),
        (
            dict(archs = ["gfx1100"], masks = {"ROCR_VISIBLE_DEVICES": ""}),
            "hides every AMD GPU",
        ),
    ],
    ids = ["no-amd-gpu", "no-wheels", "mask-hides-amd"],
)
def test_update_says_why_the_request_kept_cuda(stack, monkeypatch, capsys, host, reason):
    """Red on base: _ensure_rocm_torch returned on a non-viable route without a word."""
    out = _update(stack, monkeypatch, capsys, **host)
    lines = [l for l in out.splitlines() if _PY_REASON in l]
    assert len(lines) == 1, out
    assert reason in lines[0], out
    assert "Keeping CUDA torch for the NVIDIA GPU." in lines[0]


def test_update_on_wsl_without_librocdxg_says_the_setup_was_skipped(stack, monkeypatch, capsys):
    out = _update(
        stack,
        monkeypatch,
        capsys,
        wsl_gpu = "AMD Radeon AI PRO R9700",
        rocm_gpu = False,
        display = False,
        archs = [],
        wsl = True,
    )
    assert _PY_REASON in out
    assert (
        "Windows reports AMD Radeon AI PRO R9700, but ROCm-on-WSL (librocdxg) isn't installed"
        in out
    )


def test_update_on_an_nvidia_only_wsl_host_gives_no_rocm_setup_hint(stack, monkeypatch, capsys):
    """Windows reports no AMD adapter: the reason still prints, the setup advice does not."""
    out = _update(
        stack, monkeypatch, capsys, wsl_gpu = None, rocm_gpu = False, display = False, archs = [], wsl = True
    )
    assert _PY_REASON in out
    assert "ROCm-on-WSL" not in out


def test_a_viable_request_on_update_prints_nothing_new(stack, monkeypatch, capsys):
    """The control: gfx1100 has wheels, so the repair proceeds to the version probe silently."""
    out = _update(stack, monkeypatch, capsys, archs = ["gfx1100"])
    assert "UNSLOTH_FORCE_ROCM_TORCH" not in out, out


def test_update_under_install_sh_stays_quiet(stack, monkeypatch, capsys):
    """install.sh already said why and exported the CUDA backend; the stack must not repeat it."""
    out = _update(stack, monkeypatch, capsys, archs = ["gfx1010"], backend = "cuda")
    assert "UNSLOTH_FORCE_ROCM_TORCH" not in out, out
