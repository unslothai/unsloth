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
    probe_source = "kfd",
    installed_family = None,
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

    def _detect(dedup = True, **_kw):
        # The real probe records which tool answered;
        stack._LAST_AMD_GFX_PROBE = probe_source if gfx else None
        return list(dict.fromkeys(gfx)) if dedup else list(gfx)

    monkeypatch.setattr(stack, "_detect_amd_gfx_codes", _detect)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [], raising = False)
    # No AMD per-arch wheel is installed unless a case says so:
    monkeypatch.setattr(stack, "_installed_rocm_wheel_family", lambda: installed_family)
    monkeypatch.setattr(stack, "_torch_requires_rocm_sdk", lambda: installed_family is not None)
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: rocm_ver)
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: inferred)
    version, hip = torch
    monkeypatch.setattr(stack, "_probe_torch_runtime", lambda: (ran, importable, version, hip, ""))

    # Nothing here may install:
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
    # gfx1100 rather than the default gfx1151:
    _host(monkeypatch, torch = torch, gfx = ("gfx1100",))
    assert _needs_pass() is False


@pytest.mark.parametrize("gfx", ["gfx1150", "gfx1151"])
def test_a_strix_host_on_a_wheel_without_its_kernels_forces_the_pass(monkeypatch, gfx):
    """rocm6.4 does not carry gfx1150/gfx1151 (AMD dates support to 7.1), and
    _ensure_rocm_torch already reroutes such a host on a fresh install. Keeping the fast path
    is the preflight declining a repair the repair itself performs."""
    _host(monkeypatch, torch = ("2.9.0+rocm6.4", "6.4"), gfx = (gfx,))
    assert _needs_pass() is True
    # A generic tag that does carry them is still not the build Strix wants: the reroute prefers AMD's 7.13 fixes over
    # every generic index below the floor, so the pass is due until torch IS that build.
    _host(monkeypatch, torch = ("2.11.0+rocm7.2", "7.2"), gfx = (gfx,), rocm_ver = (7, 2))
    assert _needs_pass() is True
    _host(
        monkeypatch,
        torch = ("2.11.0+rocm7.13.0", "7.13"),
        gfx = (gfx,),
        rocm_ver = (7, 2),
        installed_family = gfx,
    )
    assert _needs_pass() is False




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


def test_a_mixed_arch_host_keeps_the_fast_path_only_while_it_is_ambiguous(monkeypatch):
    """Being mixed is not being unreadable. Unpinned, probe and mask order can disagree with
    nothing to resolve it, so the fast path stands. Pinned, _runtime_gfx_target composes both
    layers and names a card whose generic wheel can still lack kernels for it."""
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        gfx = ("gfx1151", "gfx1103"),
        probe_source = "amd-smi",
        env = {"HIP_VISIBLE_DEVICES": "1"},
    )
    assert _needs_pass() is False
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        gfx = ("gfx1151", "gfx1103"),
        env = {"ROCR_VISIBLE_DEVICES": "GPU-8d1f2e3a4b5c6d7e"},
    )
    assert _needs_pass() is False
    # Not ambiguous:
    # Ambiguous: amd-smi enumerates in discovery order and a mask indexes HIP order.
    # Ambiguous: a UUID names a device but no position in a list of arches.
    # Not ambiguous: KFD node order IS the order the mask indexes, so this names gfx1103, whose generic wheel carries
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        gfx = ("gfx1100", "gfx1103"),
        env = {"HIP_VISIBLE_DEVICES": "1"},
    )
    assert _needs_pass() is True


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
    pattern: the HIP mask wins, so the GPU is visible and the wheel repairable.
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




def _run_cli(
    *args,
    env = None,
    safe_path = False,
):
    # PINNED, and UNSLOTH_NO_TORCH is pinned for the same reason PATH and HOME already are: it is an ambient input to
    # the answer under test that the caller does not intend to vary.
    # Left unset, `_infer_no_torch` falls through to `install_manifest.recorded_no_torch()`, which reads
    # `.unsloth-no-torch` and `unsloth_install_manifest.json` out of `sys.prefix`
    # Three other test modules drive the real `install_python_stack()` in process and leave that marker behind with no
    # cleanup, so whether it exists when this child starts is a race between workers.
    # It resolves the FIRST line of `_amd_torch_needs_dependency_pass`, which returns False and exits 1 before any
    # wheel-family logic runs, and the probe sends the child's stderr to DEVNULL, so the failure arrives as rc=1 with
    # empty stdout and empty stderr and names nothing.
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
    masked AMD host, unreadable torch), so the code cannot say which it saw. Asserting the
    line back against the code keeps the diagnostic from drifting.
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
    assert not stderr, stderr
    # Read as a STATEMENT, not in an assert message:
    decision = _decision(result)
    assert result.returncode == 1, decision
    # ...and the gate that answered must be the one this case names, not whichever other exit-1 state the host happened
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
    # An import failure also exits 1, so exit 1 alone does not prove the gate ran.
    assert not stderr, stderr
    # The decision line, not the bare code:
    decision = _decision(result)
    assert result.returncode == expected, f"{decision}\n{stderr}"
    # The wheel family must be the input that decided it, so the case cannot pass on a host
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




# conservative divergences (unreadable torch, hidden mask, mixed arch, wrong ROCm
# Repair parity Both directions are asserted, so every row must be one the two sides agree on.
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
        (
            "UNSLOTH_ROCM_GFX_ARCH rescuing a visible GPU with an unreadable ROCm",
            dict(rocm_ver = None, inferred = "gfx1151", gfx = ("gfx1151",)),
            True,
        ),
        (
            "a non-Strix GPU on a ROCm with a published wheel family",
            dict(rocm_ver = (6, 4), inferred = None, gfx = ("gfx1030",)),
            True,
        ),
        (
            "a visible Strix GPU whose ROCm version cannot be read",
            dict(rocm_ver = None, inferred = None),
            True,
        ),
        (
            "a visible GPU whose ROCm version cannot be read",
            dict(rocm_ver = None, inferred = None, gfx = ("gfx1100",)),
            False,
        ),
        (
            "a ROCm older than any published wheel family",
            dict(rocm_ver = (5, 0), inferred = None, gfx = ("gfx1030",)),
            False,
        ),
        # Only UNSLOTH_ROCM_GFX_ARCH carries this:
        # The generic download.pytorch.org arm, the one _generic_pytorch_rocm_tag feeds.
        # The default host here is Strix, whose per-arch index needs no host ROCm version, so an unreadable one is the
        # A non-Strix arch the generic wheel does carry:
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




# Right wheel family, wrong architecture
def _rocm_torch(
    monkeypatch,
    *,
    family,
    owns_sdk = None,
    **kw,
):
    """A host whose torch IS a ROCm build.

    ``family`` is the per-arch family it reads back as; None means unknowable.
    ``owns_sdk`` is whether torch requires AMD's rocm[libraries], separating a generic wheel
    (False, no family to read) from a per-arch install whose family will not read back.
    """
    _host(monkeypatch, torch = ("2.11.0+rocm7.13.0", "7.13"), **kw)
    _owns = family is not None if owns_sdk is None else owns_sdk
    monkeypatch.setattr(stack, "_torch_requires_rocm_sdk", lambda: _owns)
    monkeypatch.setattr(stack, "_installed_rocm_wheel_family", lambda: family)


def test_a_generic_wheel_without_kernels_for_this_gpu_forces_the_pass(monkeypatch):
    """The reported host on an update. Torch is a ROCm build, which used to end the probe
    for all of them, so the repair was reachable on a fresh install and never on update."""
    _rocm_torch(monkeypatch, family = None, gfx = ("gfx1103",))
    assert _needs_pass() is True


def test_a_per_arch_wheel_built_for_another_gpu_forces_the_pass(monkeypatch):
    """A per-arch install outlives the card it was made for: the gfx110X-all wheels carry
    no gfx1200 kernels, and no update would ever notice."""
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx1200",))
    assert _needs_pass() is True


def test_that_repair_is_reachable_with_no_readable_rocm_version(monkeypatch):
    """Both per-arch repairs are version-independent, and a bundled-runtime host has no
    system ROCm to read a version from -- the very host they exist for."""
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx1200",), rocm_ver = None)
    assert _needs_pass() is True


@pytest.mark.parametrize(
    "family, gfx",
    [
        ("gfx1152", ("gfx1152",)),
        ("gfx110x-all", ("gfx1103",)),
        (None, ("gfx1100",)),
    ],
)
def test_a_healthy_rocm_install_still_keeps_the_fast_path(monkeypatch, family, gfx):
    """The cost of a wrong True is only a dependency pass, but an update that runs one every
    time is the reason this probe exists."""
    _rocm_torch(monkeypatch, family = family, gfx = gfx)
    assert _needs_pass() is False


def test_an_unknowable_family_keeps_the_fast_path(monkeypatch):
    """_installed_rocm_wheel_family answers None for two runtimes with no `rocm` to
    arbitrate. _ensure_rocm_torch cannot skip on a family it never read, so forcing the pass
    would reinstall the multi-GB stack on every single update rather than once."""
    _rocm_torch(monkeypatch, family = None, owns_sdk = True, gfx = ("gfx1152",))
    assert _needs_pass() is False


def test_a_matching_family_below_its_torch_floor_forces_the_pass(monkeypatch):
    """The family is the right SHAPE, not a working build: below 2.11 these leaves carry the
    _grouped_mm bug, and _already_on_leaf keeps a matching family only above that floor.
    Keeping the fast path here leaves the broken build in place on every update."""
    _rocm_torch(monkeypatch, family = "gfx1152", gfx = ("gfx1152",))
    monkeypatch.setattr(
        stack, "_probe_torch_runtime", lambda: (True, True, "2.10.0+rocm7.13.0", "7.13", "")
    )
    assert _needs_pass() is True


def test_a_leaf_with_no_torch_floor_keeps_the_fast_path_below_211(monkeypatch):
    """Only the leaves in _ROCM_GFX_TORCH211_LEAVES have that bug; the rest ship <2.11 builds
    and are correct as they are, so the floor must not be applied to them."""
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx1103",))
    monkeypatch.setattr(
        stack, "_probe_torch_runtime", lambda: (True, True, "2.10.0+rocm7.13.0", "7.13", "")
    )
    assert _needs_pass() is False


def test_a_named_arch_resolves_a_host_no_ordinal_can(monkeypatch):
    """The decline message offers UNSLOTH_ROCM_GFX_ARCH as the way through, so it has to be
    one: the user naming the arch outright is the reading no ordinal or UUID can contradict.
    Without it the same host stays ambiguous and keeps the fast path."""
    _host(
        monkeypatch,
        torch = ("2.9.0+cpu", ""),
        gfx = ("gfx1151", "gfx1103"),
        probe_source = "amd-smi",
        env = {"HIP_VISIBLE_DEVICES": "1", "UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
    )
    assert _needs_pass() is True


def test_a_rocm_pin_is_not_asked_a_hardware_question(monkeypatch):
    """A pin commits to an index regardless of the visible GPU, which is why every hardware
    gate above it is skipped. Asked under a pin, the family question answers for whatever card
    the probing machine has -- how the end-to-end CLI case began failing on an AMD box."""
    _rocm_torch(
        monkeypatch,
        family = "gfx110x-all",
        gfx = ("gfx1151",),
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm7.13"},
    )
    assert _needs_pass() is False
    _rocm_torch(
        monkeypatch,
        family = "gfx110x-all",
        gfx = ("gfx1151",),
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm6.4"},
    )
    assert _needs_pass() is True
    # The same stale family without a pin is exactly what the repair is for.
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx1151",))
    assert _needs_pass() is True


def test_a_stale_family_no_index_can_repair_keeps_the_fast_path(monkeypatch):
    """The preflight must not promise a repair the repair declines. gfx1010 has no AMD leaf
    and no generic kernels, so _ensure_rocm_torch refuses on the same routability question,
    and answering True here buys a dependency pass per update and never a working torch."""
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx1010",))
    assert _needs_pass() is False
    # An empty leaf is not the test:
    for _gfx in ("gfx942", "gfx950"):
        _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = (_gfx,))
        assert _needs_pass() is True, _gfx


def test_a_mixed_host_gfx906_keeps_the_fast_path(monkeypatch):
    """gfx906 answers the abstract routability question yes, but its one usable route is the
    rocm6.3 tag, which opens only when it is the sole detected arch. _ensure_rocm_torch
    declines the demotion there, so asking the abstract question buys a pass and no repair."""
    _rocm_torch(
        monkeypatch,
        family = "gfx110x-all",
        gfx = ("gfx1100", "gfx906"),
        env = {"HIP_VISIBLE_DEVICES": "1"},
    )
    assert _needs_pass() is False
    # Alone, gfx906 does have a route, and the same stale family is worth the pass.
    _rocm_torch(monkeypatch, family = "gfx110x-all", gfx = ("gfx906",))
    assert _needs_pass() is True


def test_a_confirmed_spoof_forces_the_pass_even_on_a_matching_family(monkeypatch):
    """_clear_confirmed_hsa_spoof runs only inside _ensure_rocm_torch, so a fast path taken
    on the wheel question alone leaves HSA_OVERRIDE_GFX_VERSION set: ROCr keeps presenting an
    ISA the installed wheels have no code for (#7331), however well the family matches."""
    # The pin names the family torch already carries, so nothing is due for it either.
    _rocm_torch(
        monkeypatch,
        family = "gfx1152",
        gfx = (),
        rocm_ver = None,
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: ["gfx1152"], raising = False)
    assert _needs_pass() is True


def test_a_sole_gfx906_above_its_legacy_tag_forces_the_pass(monkeypatch):
    """The gfx906 reroute is a compatibility reroute, not a missing-kernel one, so the wheel
    question never saw it and the rocm6.3 downgrade was unreachable from `studio update`."""
    _rocm_torch(monkeypatch, family = None, gfx = ("gfx906",), rocm_ver = (6, 4))
    assert _needs_pass() is True
    _host(monkeypatch, torch = ("2.9.0+rocm6.3", "6.3"), gfx = ("gfx906",), rocm_ver = (6, 4))
    assert _needs_pass() is False


def test_a_repin_to_another_per_arch_leaf_forces_the_pass(monkeypatch):
    """Both leaves are torch 2.11 with a three-part +rocm7.13.0 tag, so the pin arm's version
    comparison sees no change and would keep the fast path over an edited pin -- the update
    that was supposed to move the card onto its own wheels."""
    # A pin that no longer matches the installed wheel is the pin's own question, not a hardware one, and
    _rocm_torch(
        monkeypatch,
        family = "gfx110x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx120X-all/"},
    )
    assert _needs_pass() is True


def test_a_pin_naming_the_installed_leaf_keeps_the_fast_path(monkeypatch):
    """The same comparison the other way: nothing to apply, so nothing to pass for."""
    _rocm_torch(
        monkeypatch,
        family = "gfx120x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx120X-all/"},
    )
    assert _needs_pass() is False


def test_a_pin_at_a_non_floor_leaf_already_satisfied_keeps_the_fast_path(monkeypatch):
    """The pin arm now answers the preflight, so a heuristic that calls a valid environment
    stale costs a dependency pass on every update as well as the reinstall behind it."""
    _rocm_torch(
        monkeypatch,
        family = "gfx110x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx110X-all/"},
    )
    assert _needs_pass() is False
