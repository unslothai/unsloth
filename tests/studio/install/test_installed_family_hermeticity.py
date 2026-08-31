# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A test that reaches the reinstall skip without pinning the installed-wheel pair reads
the machine running pytest, so it passes on a bare laptop and fails on any host that
already has AMD per-arch torch. Three cases in TestEnsureRocmTorch did.

_installed_rocm_wheel_family and _torch_requires_rocm_sdk go to importlib.metadata for the
RUNNING interpreter, which is the venv being repaired at install time and the test runner
here. This file fakes the metadata layer, never the functions, so their parsing still runs.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_hermeticity", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_MARK = stack_mod._TORCH_PROBE_MARKER

# `rocm` Requires-Dist, torch Requires-Dist, distributions on disk. Two are pip behaviours,
# not hypotheticals: installing a generic wheel over a per-arch one drops rocm[libraries]
# from torch but leaves `rocm` naming the old family, and a family switch upgrades `rocm`
# in place without uninstalling the superseded runtime.
AMBIENT_STATES: "dict[str, tuple]" = {
    "bare": (None, ["filelock"], []),
    "perarch-gfx1151": (
        ["rocm-sdk-libraries-gfx1151==7.13.0; extra == 'libraries'"],
        ["rocm[libraries]==7.13.0"],
        ["rocm", "rocm-sdk-libraries-gfx1151"],
    ),
    "perarch-gfx110X-all": (
        ["rocm-sdk-libraries-gfx110X-all==7.13.0; extra == 'libraries'"],
        ["rocm[libraries]==7.13.0"],
        ["rocm", "rocm-sdk-libraries-gfx110X-all"],
    ),
    "perarch-gfx120X-all": (
        ["rocm-sdk-libraries-gfx120X-all==7.13.0; extra == 'libraries'"],
        ["rocm[libraries]==7.13.0"],
        ["rocm", "rocm-sdk-libraries-gfx120X-all"],
    ),
    "generic-with-orphan-rocm": (
        ["rocm-sdk-libraries-gfx110X-all==7.13.0; extra == 'libraries'"],
        ["filelock"],
        ["rocm", "rocm-sdk-libraries-gfx110X-all"],
    ),
    "two-families-after-switch": (
        None,
        ["filelock"],
        ["rocm-sdk-libraries-gfx110X-all", "rocm-sdk-libraries-gfx1151"],
    ),
}


class _Dist:
    """The one attribute _installed_rocm_wheel_family reads off a distribution."""

    def __init__(self, name: str):
        self.metadata = {"Name": name}


@contextmanager
def ambient(state: str):
    """Make importlib.metadata describe ``state`` instead of this interpreter."""
    from importlib import metadata

    rocm_reqs, torch_reqs, dists = AMBIENT_STATES[state]
    real_requires = metadata.requires

    def _requires(name):
        if name == "rocm":
            if rocm_reqs is None:
                raise metadata.PackageNotFoundError("rocm")
            return list(rocm_reqs)
        if name == "torch":
            return list(torch_reqs)
        return real_requires(name)

    with (
        patch.object(metadata, "requires", _requires),
        patch.object(metadata, "distributions", lambda *a, **kw: [_Dist(n) for n in dists]),
    ):
        yield


def _route(
    gfx: str,
    torch_line: str,
    env: "dict | None" = None,
    family: "str | None" = None,
    torch_owns_rocm: bool = False,
) -> str:
    """Every pip argument _ensure_rocm_torch() produced, as one string. ``family`` /
    ``torch_owns_rocm`` pin the installed-wheel pair; the defaults are "no ROCm installed"."""
    pip, pip_try = MagicMock(), MagicMock(return_value = True)
    probe = MagicMock(returncode = 0, stdout = _MARK + torch_line + "\n")
    buf = io.StringIO()

    stack_mod._invalidate_torch_runtime_probe()
    with (
        patch.dict(os.environ, env or {}, clear = False),
        patch.object(stack_mod, "IS_WINDOWS", False),
        patch.object(stack_mod, "IS_MACOS", False),
        patch.object(stack_mod, "_TORCH_BACKEND", ""),
        patch.object(stack_mod.platform, "machine", return_value = "x86_64"),
        patch.object(stack_mod, "pip_install", pip),
        patch.object(stack_mod, "pip_install_try", pip_try),
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack_mod, "_has_rocm_gpu", return_value = True),
        patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = None),
        patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = [gfx]),
        patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1)),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = []),
        patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = family),
        patch.object(stack_mod, "_torch_requires_rocm_sdk", return_value = torch_owns_rocm),
        patch.object(stack_mod.os.path, "isdir", return_value = True),
        patch.object(stack_mod.subprocess, "run", return_value = probe),
    ):
        for _stale in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "UNSLOTH_ROCM_GFX_ARCH",
            "UNSLOTH_TORCH_INDEX_URL",
            "UNSLOTH_AMD_ROCM_MIRROR",
            "UNSLOTH_ROCM_TORCH_INSTALLED",
        ):
            if _stale not in (env or {}):
                os.environ.pop(_stale, None)
        stack_mod._invalidate_torch_runtime_probe()
        import contextlib

        with contextlib.redirect_stdout(buf):
            stack_mod._ensure_rocm_torch()
    stack_mod._invalidate_torch_runtime_probe()
    return str(pip.call_args_list) + str(pip_try.call_args_list)


# The generic-ROCm hosts are the interesting ones: that is where the skip arms live.
HOSTS = {
    "gfx1103-cpu-torch": ("gfx1103", "2.10.0+cpu||"),
    "gfx1103-generic-rocm": ("gfx1103", "2.10.0+rocm7.1|7.1|"),
    "gfx1100-generic-rocm": ("gfx1100", "2.10.0+rocm7.1|7.1|"),
}


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_pinning_the_pair_makes_this_machine_irrelevant(host):
    """Pinning the pair is SUFFICIENT isolation. Not "routing ignores what is installed" --
    at install time it must not; the claim is that these two functions are the only door."""
    gfx, torch_line = HOSTS[host]
    with ambient("bare"):
        expected = _route(gfx, torch_line)
    for state in AMBIENT_STATES:
        with ambient(state):
            got = _route(gfx, torch_line)
        assert got == expected, (
            f"{host} routes differently when the interpreter running the tests is in the "
            f"{state!r} state, even though the installed-wheel pair is pinned. Something in "
            f"_ensure_rocm_torch now reads this venv by a path those two functions do not "
            f"cover, so the suite's verdict depends on who runs it.\n"
            f"  bare: {expected}\n  {state}: {got}"
        )


@pytest.mark.parametrize(
    "family, torch_owns_rocm, expect_reinstall",
    [
        (None, False, True),
        ("gfx110x-all", True, False),
        # Family matches but torch does not own it: the orphan, so the running torch is the
        # generic build with no gfx1103 kernels and the skip must not fire.
        ("gfx110x-all", False, True),
        ("gfx120x-all", True, True),
    ],
    ids = ["no-family", "correct-family", "orphan-metapackage", "wrong-family"],
)
def test_the_installed_family_decides_the_skip(family, torch_owns_rocm, expect_reinstall):
    """The other half: without this, pinning both to None would satisfy the test above while
    the feature they gate quietly stopped working."""
    with ambient("bare"):
        calls = _route(
            "gfx1103", "2.10.0+rocm7.13.0|7.13|", family = family, torch_owns_rocm = torch_owns_rocm
        )
    rerouted = "repo.amd.com/rocm/whl/gfx110X-all/" in calls
    assert rerouted is expect_reinstall, (
        f"family={family!r} torch_owns_rocm={torch_owns_rocm} should "
        f"{'reinstall' if expect_reinstall else 'keep the installed wheels'}; got {calls}"
    )


def test_the_states_are_distinguishable():
    """Non-vacuity: if the fake metadata said nothing, the test above would pass emptily."""
    with ambient("perarch-gfx1151"):
        assert stack_mod._installed_rocm_wheel_family() == "gfx1151"
        assert stack_mod._torch_requires_rocm_sdk() is True
    with ambient("bare"):
        assert stack_mod._installed_rocm_wheel_family() is None
        assert stack_mod._torch_requires_rocm_sdk() is False
    with ambient("generic-with-orphan-rocm"):
        assert stack_mod._installed_rocm_wheel_family() == "gfx110x-all"
        assert stack_mod._torch_requires_rocm_sdk() is False
    with ambient("two-families-after-switch"):
        # Two runtimes, no `rocm` to arbitrate: unknowable, not a guess.
        assert stack_mod._installed_rocm_wheel_family() is None
