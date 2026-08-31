# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for setup.sh's AMD torch fast-path escape."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location("studio_ips_amd_fastpath", _STACK_PATH)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack
_STACK_SPEC.loader.exec_module(stack)

_ROUTING_ENV = (
    "UNSLOTH_TORCH_INDEX_URL",
    "UNSLOTH_TORCH_INDEX_FAMILY",
    "UNSLOTH_TORCH_BACKEND",
    "UNSLOTH_NO_TORCH",
    "UNSLOTH_ROCM_GFX_ARCH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def _host(
    monkeypatch,
    *,
    torch = ("2.9.0", ""),
    ran = True,
    importable = True,
    gfx = ("gfx1151",),
    rocm_gpu = True,
    rocm_ver = (6, 4),
    inferred = None,
    nvidia = False,
    backend = "",
    machine = "x86_64",
    linux = True,
    no_torch = False,
    env = None,
):
    """A ROCm host with a CPU wheel by default; each argument moves one thing."""
    for name in _ROUTING_ENV:
        monkeypatch.delenv(name, raising = False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(stack, "IS_LINUX", linux)
    monkeypatch.setattr(stack, "NO_TORCH", no_torch)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", backend)
    monkeypatch.setattr(stack.platform, "machine", lambda: machine)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: nvidia)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: rocm_gpu and not nvidia)
    monkeypatch.setattr(stack, "_detect_amd_gfx_codes", lambda dedup = True: list(gfx))
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: rocm_ver)
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: inferred)
    version, hip = torch
    monkeypatch.setattr(stack, "_probe_torch_runtime", lambda: (ran, importable, version, hip, ""))

    # Nothing here may install: the dependency pass it unlocks owns that.
    def _no_installs(*_a, **_k):
        raise AssertionError("the fast-path probe must not install anything")

    monkeypatch.setattr(stack, "pip_install", _no_installs)
    monkeypatch.setattr(stack, "pip_install_try", _no_installs)


def _needs_pass():
    return stack._amd_torch_needs_dependency_pass()


# Wrong wheel


@pytest.mark.parametrize(
    "version",
    ["2.9.0+cpu", "2.9.0+cu128", "2.9.0", "2.8.0a0+34c6371d24.nv25.08"],
)
def test_a_non_rocm_wheel_on_a_rocm_host_forces_the_pass(monkeypatch, version):
    _host(monkeypatch, torch = (version, ""))
    assert _needs_pass() is True


def test_an_explicit_rocm_pin_answers_without_a_hardware_probe(monkeypatch):
    """The pin commits to ROCm wheels headless, as _ensure_rocm_torch already does."""
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        rocm_gpu = False,
        gfx = (),
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm6.4"},
    )
    assert _needs_pass() is True


def test_a_real_device_selection_is_not_a_hidden_host(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = {"HIP_VISIBLE_DEVICES": "1"})
    assert _needs_pass() is True


# Correct wheel


@pytest.mark.parametrize(
    "torch",
    [
        ("2.9.0+rocm6.4", "6.4"),
        ("2.9.0+rocm6.4", ""),
        ("2.11.0+rocm7.13.0", "7.13"),
        # AMD and source builds carry torch.version.hip with no +rocm local version.
        ("2.5.0a0+git1234567", "6.2.41134"),
    ],
)
def test_a_rocm_wheel_keeps_the_fast_path(monkeypatch, torch):
    _host(monkeypatch, torch = torch)
    assert _needs_pass() is False


# Uncertain classification


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"ran": False}, id = "probe-did-not-run"),
        pytest.param({"importable": False}, id = "torch-does-not-import"),
        pytest.param({"torch": (None, "")}, id = "no-version-reported"),
        pytest.param({"torch": ("", "")}, id = "empty-version-reported"),
    ],
)
def test_an_unreadable_torch_keeps_the_fast_path(monkeypatch, kwargs):
    _host(monkeypatch, **kwargs)
    assert _needs_pass() is False


def test_a_mixed_arch_host_keeps_the_fast_path(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), gfx = ("gfx1151", "gfx1100"))
    assert _needs_pass() is False


def test_two_cards_of_one_arch_are_not_ambiguous(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), gfx = ("gfx1100", "gfx1100"))
    assert _needs_pass() is True


@pytest.mark.parametrize(
    "mask", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
@pytest.mark.parametrize("value", ["", "-1", " "])
def test_a_mask_that_hides_every_device_keeps_the_fast_path(monkeypatch, mask, value):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = {mask: value})
    assert _needs_pass() is False


@pytest.mark.parametrize(
    "env",
    [
        # ROCr hides everything, then HIP names a device out of the empty set.
        {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "-1"},
        {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""},
        {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": ""},
    ],
)
def test_stacked_masks_with_a_hidden_layer_keep_the_fast_path(monkeypatch, env):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = env)
    assert _needs_pass() is False


def test_stacked_masks_that_all_select_are_not_a_hidden_host(monkeypatch):
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        env = {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"},
    )
    assert _needs_pass() is True


@pytest.mark.parametrize("cuda", ["", "-1"])
def test_a_set_hip_mask_shadows_the_cuda_alias(monkeypatch, cuda):
    """CUDA_VISIBLE_DEVICES is HIP's alias, not a layer under it (_pick_visible_index).

    Hiding NVIDIA with CUDA_VISIBLE_DEVICES=-1 while pinning HIP is a real mixed-host
    pattern; the HIP mask wins, so the GPU is visible and the wheel is repairable.
    """
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        env = {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": cuda},
    )
    assert _needs_pass() is True


def test_a_hidden_hip_mask_wins_over_a_selecting_cuda_alias(monkeypatch):
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        env = {"HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert _needs_pass() is False


def test_an_nvidia_host_keeps_the_fast_path(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), nvidia = True)
    assert _needs_pass() is False


def test_an_nvidia_host_with_an_inferable_amd_arch_keeps_the_fast_path(monkeypatch):
    """_infer_linux_amd_gfx_arch never checks NVIDIA, so this gate carries the host.

    Without it the preflight forces a pass that _ensure_rocm_torch then refuses.
    """
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        nvidia = True,
        rocm_gpu = False,
        gfx = (),
        inferred = "gfx1151",
    )
    assert _needs_pass() is False


def test_a_host_with_no_amd_gpu_keeps_the_fast_path(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), rocm_gpu = False, gfx = ())
    assert _needs_pass() is False


@pytest.mark.parametrize("backend", ["cpu", "cuda", "xpu"])
def test_a_resolved_non_rocm_backend_keeps_the_fast_path(monkeypatch, backend):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), backend = backend)
    assert _needs_pass() is False


@pytest.mark.parametrize(
    "pin",
    [
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/xpu",
        "https://mirror.internal.example/simple",
    ],
)
def test_a_non_rocm_pin_keeps_the_fast_path(monkeypatch, pin):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = {"UNSLOTH_TORCH_INDEX_URL": pin})
    assert _needs_pass() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"no_torch": True}, id = "gguf-only-install"),
        pytest.param({"linux": False}, id = "not-linux"),
        pytest.param({"machine": "aarch64"}, id = "no-rocm-wheels-for-this-arch"),
    ],
)
def test_a_host_without_rocm_wheels_keeps_the_fast_path(monkeypatch, kwargs):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), **kwargs)
    assert _needs_pass() is False


# CLI


def _run_cli(
    *args,
    env = None,
    safe_path = False,
):
    # PINNED, and UNSLOTH_NO_TORCH is pinned for the same reason PATH and HOME already are:
    # it is an ambient input to the answer under test that the caller does not intend to vary.
    # Left unset, `_infer_no_torch` falls through to `install_manifest.recorded_no_torch()`,
    # which reads `.unsloth-no-torch` and `unsloth_install_manifest.json` out of `sys.prefix` --
    # one path, shared by every xdist worker. Three other test modules drive the real
    # `install_python_stack()` in process and leave that marker behind with no cleanup, so
    # whether it exists when this child starts is a race between workers. It resolves the FIRST
    # line of `_amd_torch_needs_dependency_pass`, which returns False and exits 1 before any
    # wheel-family logic runs, and the probe sends the child's stderr to DEVNULL, so the failure
    # arrives as rc=1 with empty stdout and empty stderr and names nothing.
    # Measured on origin/main with this file byte-identical: marker absent 8/8 pass, marker
    # present 8/8 fail. A test that reports its subject broken on the strength of a file another
    # test left lying around is not measuring its subject.
    # The `**(env or {})` below still wins, so the cases that set it deliberately are unaffected.
    child = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "UNSLOTH_NO_TORCH": "0",
        **(env or {}),
    }
    if safe_path:
        child["PYTHONSAFEPATH"] = "1"
    return subprocess.run(
        [sys.executable, str(_STACK_PATH), *args],
        env = child,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        timeout = 180,
    )


def _decision(result):
    """The CLI's own account of which input produced its exit code.

    Exit 1 alone is five states (no-torch venv, resolved backend, non-ROCm pin, absent or
    masked AMD host, unreadable torch), so the code cannot say which one it saw. Asserting
    the line back against the code keeps the diagnostic from drifting off the decision.
    """
    stdout = result.stdout.decode(errors = "replace")
    marked = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith(stack._AMD_FASTPATH_DECISION_MARKER)
    ]
    assert len(marked) == 1, f"expected one decision line, got {marked!r} in {stdout!r}"
    line = marked[0]
    expected = "needs_pass=True" if result.returncode == 0 else "needs_pass=False"
    assert expected in line, f"decision line disagrees with exit {result.returncode}: {line}"
    return line


@pytest.mark.parametrize("env_name", ["UNSLOTH_NO_TORCH", "UNSLOTH_TORCH_BACKEND"])
@pytest.mark.parametrize("safe_path", [False, True])
def test_the_cli_reports_keep_the_fast_path_as_a_non_zero_exit(env_name, safe_path):
    env = {"UNSLOTH_NO_TORCH": "1", "UNSLOTH_TORCH_BACKEND": "cpu"}
    result = _run_cli(
        "--amd-torch-needs-dependency-pass",
        env = {env_name: env[env_name]},
        safe_path = safe_path,
    )
    stderr = result.stderr.decode(errors = "replace")
    # An import failure also exits 1, so exit 1 alone does not prove the gate ran.
    assert not stderr, stderr
    # Read as a STATEMENT, not in an assert message: a `_decision(result)` that appears only
    # after the comma runs once the assertion has already failed, so it checks nothing on the
    # passing path while reading exactly like it does.
    decision = _decision(result)
    assert result.returncode == 1, decision
    # ...and the gate that answered must be the one this case names, not whichever other
    # exit-1 state the host happened to be in, or the case passes on any host that keeps
    # the fast path for an unrelated reason.
    expected_field = {
        "UNSLOTH_NO_TORCH": "no_torch=True",
        "UNSLOTH_TORCH_BACKEND": "backend='cpu'",
    }[env_name]
    assert expected_field in decision


@pytest.mark.parametrize(
    "version,hip,expected",
    [("2.9.0+cpu", None, 0), ("2.9.0+rocm6.4", "6.4.43483", 1)],
)
def test_the_cli_answers_end_to_end_over_a_stub_torch(tmp_path, version, hip, expected):
    """Exit 0 is the only side that moves setup.sh, so drive it for real.

    A ROCm pin skips the hardware gates, leaving the wheel family as the only input.
    """
    (tmp_path / "torch.py").write_text(
        "import types\n"
        f"__version__ = {version!r}\n"
        f"version = types.SimpleNamespace(hip = {hip!r}, cuda = None)\n"
    )
    result = _run_cli(
        "--amd-torch-needs-dependency-pass",
        env = {
            "UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm6.4",
            "PYTHONPATH": str(tmp_path),
        },
    )
    stderr = result.stderr.decode(errors = "replace")
    assert not stderr, stderr
    # The decision line, not the bare code: exit 1 is five states here, and a bare
    # `assert 1 == 0` with both streams empty is what this case used to report. A statement,
    # so it is checked on the passing path too, not only when the next line fails.
    decision = _decision(result)
    assert result.returncode == expected, f"{decision}\n{stderr}"
    # The wheel family must be the input that decided it, so the case cannot pass on a host
    # that answered before the probe was reached.
    assert f"'{version}'" in decision, decision


@pytest.mark.parametrize(
    "argv",
    [
        ["--amd-torch-needs-dependency-pass", "--amd-torch-needs-dependency-pass"],
        ["--amd-torch-needs-dependency-passs"],
        ["--amd-torch-needs-dependency-pass", "extra"],
    ],
)
def test_a_malformed_probe_call_never_falls_through_to_the_installer(argv):
    result = _run_cli(*argv)
    assert result.returncode == 2, result.stdout.decode(errors = "replace")


# Repair parity
# Both directions are asserted, so every row must be one the two sides agree on. The
# conservative divergences (unreadable torch, hidden mask, mixed arch, wrong ROCm
# family) are covered above and do not belong here.


def _repair_installs(monkeypatch):
    """Torch index URLs _ensure_rocm_torch installs, under the same stubs."""
    installed = []

    def _record(_label, *args, **_kw):
        if "--index-url" in args:
            installed.append(args[args.index("--index-url") + 1])

    monkeypatch.setattr(stack, "pip_install", _record)
    monkeypatch.setattr(stack, "pip_install_try", lambda *a, **k: True)
    monkeypatch.setattr(stack, "_clear_confirmed_hsa_spoof", lambda _g: None)
    monkeypatch.setattr(stack, "_bnb_rocm_prerelease_url", lambda: None)
    monkeypatch.setattr(stack, "_bitsandbytes_installed", lambda: False)
    monkeypatch.setattr(stack, "_install_bnb_windows_rocm", lambda: True)
    stack._ensure_rocm_torch()
    return installed


@pytest.mark.parametrize(
    "label,host,expected",
    [
        ("a visible GPU with a readable ROCm version", dict(rocm_ver = (6, 4), inferred = None), True),
        (
            "a Strix host below the per-arch floor",
            dict(rocm_ver = (7, 0), inferred = None, gfx = ("gfx1151",)),
            True,
        ),
        (
            "a gfx906 host on a newer ROCm",
            dict(rocm_ver = (6, 4), inferred = None, gfx = ("gfx906",)),
            True,
        ),
        # #7301: no runtime enumerates anything, but the arch is known.
        (
            "UNSLOTH_ROCM_GFX_ARCH naming the arch with no runtime",
            dict(rocm_ver = None, inferred = "gfx1151", rocm_gpu = False, gfx = ()),
            True,
        ),
        (
            "an arch inferred from the CPU model with no runtime",
            dict(rocm_ver = None, inferred = "gfx1151", rocm_gpu = False, gfx = ()),
            True,
        ),
        # Only UNSLOTH_ROCM_GFX_ARCH carries this: a visible GPU defeats the row
        # above's "no runtime" disjunct.
        (
            "UNSLOTH_ROCM_GFX_ARCH rescuing a visible GPU with an unreadable ROCm",
            dict(rocm_ver = None, inferred = "gfx1151", gfx = ("gfx1151",)),
            True,
        ),
        # The generic download.pytorch.org arm, the one _generic_pytorch_rocm_tag feeds.
        (
            "a non-Strix GPU on a ROCm with a published wheel family",
            dict(rocm_ver = (6, 4), inferred = None, gfx = ("gfx1030",)),
            True,
        ),
        # The repair prints "skipping torch reinstall" and returns for all three.
        (
            "a visible GPU whose ROCm version cannot be read",
            dict(rocm_ver = None, inferred = None),
            False,
        ),
        (
            "a ROCm older than any published wheel family",
            dict(rocm_ver = (5, 0), inferred = None, gfx = ("gfx1030",)),
            False,
        ),
        (
            "an inferred arch AMD publishes no per-arch index for",
            dict(rocm_ver = None, inferred = "gfx900", rocm_gpu = False, gfx = ()),
            False,
        ),
    ],
)
def test_the_preflight_and_the_repair_agree(monkeypatch, label, host, expected):
    env = {"UNSLOTH_ROCM_GFX_ARCH": host["inferred"]} if "GFX_ARCH" in label else None
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = env, **host)
    preflight = stack._amd_torch_needs_dependency_pass()
    assert preflight is expected, label

    _host(monkeypatch, torch = ("2.9.0+cpu", ""), env = env, **host)
    installs = _repair_installs(monkeypatch)
    if preflight:
        assert installs, f"{label}: preflight forced a pass the repair declined"
    else:
        assert not installs, f"{label}: the repair acts but the preflight kept the fast path"
