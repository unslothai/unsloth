# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reroute decision must not depend on what is installed in the interpreter running it.

_installed_rocm_wheel_family() and _torch_requires_rocm_sdk() read importlib.metadata for
the RUNNING interpreter. At install time that is exactly right: install_python_stack.py is
executed by the venv's own python, so the venv it reads is the venv it is repairing. Under
pytest it is exactly wrong: the interpreter is whatever ran the suite, so a test that
reaches those functions without pinning them asks the developer's laptop a question it
means to ask a mock, and the answer decides whether a skip arm fires.

That is not hypothetical. On the gfx1151 AMD runner, whose Studio venv holds AMD per-arch
torch, three cases in TestEnsureRocmTorch flipped: the skip arms read a real `gfx1151`
family and declined reinstalls those cases assert. They passed everywhere else, so nothing
short of running the suite ON an AMD machine could show it.

This file is the standing check. It fakes the metadata layer (never the functions, so the
real lookup logic still runs) into each install state a user can actually be in, and
asserts the routing verdicts do not move. A future helper that learns to read the venv
fails here rather than on someone's hardware.
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

# The states a venv can be in when the installer next runs, as `rocm`'s Requires-Dist,
# torch's Requires-Dist, and the distributions on disk. Each is a real shape:
#
#   bare                 a laptop, and every hosted CI runner. The state that hid the bug.
#   perarch-*            AMD per-arch torch, which is what the gfx1151 runner has.
#   generic-with-orphan  pip drops rocm[libraries] from torch when a generic wheel is
#                        force-reinstalled over a per-arch one, but leaves `rocm` behind
#                        still naming the old family. _torch_requires_rocm_sdk exists for
#                        this; the family alone would read the orphan as live.
#   two-families         a family switch upgrades `rocm` in place and never uninstalls the
#                        superseded runtime, so two are on disk and neither is decisive.
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
    """Make importlib.metadata describe ``state`` instead of this interpreter.

    Patched at the metadata layer, not at the functions, so _installed_rocm_wheel_family
    and _torch_requires_rocm_sdk still execute their own parsing -- which is half of what
    is being protected.
    """
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
    """Every pip argument _ensure_rocm_torch() produced for this host, as one string.

    ``family`` / ``torch_owns_rocm`` pin the installed-wheel pair the way a test that
    describes its host by mocks must. Leaving them at the "nothing ROCm installed"
    defaults is what every case here does: the point is that the answer must then come
    from those pins and not from the interpreter running pytest.
    """
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


# A CPU torch on a gfx1103 host has no ROCm build to compare a family against, so every
# state must reach the same repair; a generic ROCm torch is the interesting one, because
# that is where the skip arms live.
HOSTS = {
    "gfx1103-cpu-torch": ("gfx1103", "2.10.0+cpu||"),
    "gfx1103-generic-rocm": ("gfx1103", "2.10.0+rocm7.1|7.1|"),
    "gfx1100-generic-rocm": ("gfx1100", "2.10.0+rocm7.1|7.1|"),
}


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_pinning_the_pair_makes_this_machine_irrelevant(host):
    """With the installed-wheel pair pinned, the ambient venv must not reach the decision.

    Not "routing never depends on what is installed" -- at install time it must, and that
    is the reinstall skip this change adds. The claim is narrower and is the one that keeps
    the suite portable: pinning those two functions is SUFFICIENT isolation. A future helper
    that reaches importlib.metadata by some other route would satisfy neither, and fails
    here instead of on an AMD user's machine.
    """
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
        # Nothing ROCm-owned on disk: no family to trust, so repair.
        (None, False, True),
        # The right family, owned by torch: the skip this change adds.
        ("gfx110x-all", True, False),
        # The orphan `rocm` pip leaves behind when a generic wheel is installed over a
        # per-arch one. The family matches but torch does not own it, so the skip must not
        # fire -- the running torch really is the generic build with no gfx1103 kernels.
        ("gfx110x-all", False, True),
        # A per-arch install that outlived its GPU.
        ("gfx120x-all", True, True),
    ],
    ids = ["no-family", "correct-family", "orphan-metapackage", "wrong-family"],
)
def test_the_installed_family_decides_the_skip(family, torch_owns_rocm, expect_reinstall):
    """The behaviour the pin above is standing in for, asserted directly.

    This is the other half: the pair is not merely isolatable, it is what the skip reads.
    Without this, pinning both to None would satisfy the hermeticity test while the feature
    they gate quietly stopped working.
    """
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
        # `rocm` still names the old family, but torch no longer owns it. Skips that read
        # the family alone would preserve wheels the running torch is not using.
        assert stack_mod._installed_rocm_wheel_family() == "gfx110x-all"
        assert stack_mod._torch_requires_rocm_sdk() is False
    with ambient("two-families-after-switch"):
        # Two runtimes and no `rocm` to arbitrate: unknowable, not a guess.
        assert stack_mod._installed_rocm_wheel_family() is None
