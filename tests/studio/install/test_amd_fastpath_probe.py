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
    [("2.9.0+rocm6.4", "6.4"), ("2.9.0+rocm6.4", ""), ("2.11.0+rocm7.13.0", "7.13")],
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


def test_an_nvidia_host_keeps_the_fast_path(monkeypatch):
    _host(monkeypatch, torch = ("2.9.0+cpu", ""), nvidia = True)
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


@pytest.mark.parametrize("env_name", ["UNSLOTH_NO_TORCH", "UNSLOTH_TORCH_BACKEND"])
def test_the_cli_reports_keep_the_fast_path_as_a_non_zero_exit(env_name):
    env = {"UNSLOTH_NO_TORCH": "1", "UNSLOTH_TORCH_BACKEND": "cpu"}
    result = subprocess.run(
        [sys.executable, str(_STACK_PATH), "--amd-torch-needs-dependency-pass"],
        env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", env_name: env[env_name]},
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        timeout = 180,
    )
    assert result.returncode == 1, result.stderr.decode(errors = "replace")


# Repair parity
# A True preflight must make the repair install. False may be deliberately conservative.


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
    # The invariant: a forced pass always has work to do.
    if preflight:
        assert installs, f"{label}: preflight forced a pass the repair declined"
    else:
        assert not installs, f"{label}: the repair acts but the preflight kept the fast path"
