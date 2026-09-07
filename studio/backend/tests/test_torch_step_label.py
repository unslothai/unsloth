# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _torch_step_label in install_python_stack.py.

The label is the only place a standalone `unsloth studio update` states which
torch backend it is working on. On Windows the ROCm probe reads rocminfo and
amd-smi, which ship with the HIP SDK and not with AMD's bundled-runtime wheels,
so a working ROCm host printed "torch check (cpu)" on the same line-block where
the next step correctly reported Windows ROCm.

The second thing this file pins is the COST of answering that. _torch_step_label
runs before the first pip step of its block, where the memoized _TORCH_RUNTIME_PROBE
is cold, and on the affected host none of the four _ensure_* calls that follow reach
the probe either. Answering with `import torch` there spent up to the probe's 90s
timeout before _progress() emitted anything -- on exactly the wedged-driver hosts
that timeout exists to rescue -- so the verdict is read off disk instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"


def _load_module(monkeypatch):
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


# -- the hardware half of the matrix -------------------------------------------------
# Each fixture is what the four detectors report on that machine. "amd_bundled" is the
# reported Strix Halo case: AMD's bundled-runtime wheels carry no rocminfo and no
# amd-smi, so only torch's own version.py knows the build is ROCm.
_HARDWARE = {
    "nvidia": dict(nvidia = True, rocm_probe = False, hip = "", label = "2.9.1+cu128"),
    "amd_tooling": dict(nvidia = False, rocm_probe = True, hip = "6.4.43483", label = "2.8.0+rocm6.4"),
    "amd_bundled": dict(
        nvidia = False, rocm_probe = False, hip = "6.4.43483-a1", label = "2.8.0a0+rocmsdk20250901"
    ),
    "xpu": dict(nvidia = False, rocm_probe = False, hip = "", label = "2.9.1+xpu"),
    "cpu": dict(nvidia = False, rocm_probe = False, hip = "", label = "2.9.1+cpu"),
    "no_torch": dict(nvidia = False, rocm_probe = False, hip = "", label = ""),
}

# -- the platform half ---------------------------------------------------------------
_PLATFORMS = {
    "windows": dict(is_windows = True, is_macos = False, is_wsl = False),
    "linux": dict(is_windows = False, is_macos = False, is_wsl = False),
    "wsl": dict(is_windows = False, is_macos = False, is_wsl = True),
    "macos": dict(is_windows = False, is_macos = True, is_wsl = False),
}


def _prepare(
    monkeypatch,
    *,
    platform_name,
    hardware_name,
    known_backend = "",
    warm_probe = None,
):
    """Load the module with one (platform, hardware) cell of the matrix in place.

    Everything is stubbed at a boundary that exists on BOTH sides of this change --
    the four detectors, the two file reads, and _probe_torch_runtime -- so the same
    file discriminates rather than erroring out on a symbol one tree lacks.

    Returns the module and the list _probe_torch_runtime calls are recorded into.
    Reaching that function is the expensive event: it is what spawns `import torch`
    under a 90s timeout, so an empty list is the assertion that a label cost nothing.
    """
    mod = _load_module(monkeypatch)
    plat = _PLATFORMS[platform_name]
    hw = _HARDWARE[hardware_name]

    monkeypatch.setattr(mod, "_TORCH_BACKEND", known_backend)
    monkeypatch.setattr(mod, "IS_WINDOWS", plat["is_windows"])
    monkeypatch.setattr(mod, "IS_MACOS", plat["is_macos"])
    monkeypatch.setattr(mod, "IS_LINUX", not plat["is_windows"] and not plat["is_macos"])
    monkeypatch.setattr(mod, "_is_wsl", lambda: plat["is_wsl"])
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: hw["nvidia"])
    monkeypatch.setattr(mod, "_has_rocm_gpu", lambda: hw["rocm_probe"])
    # The OFF-DISK reads. raising = False because _torch_hip_version_on_disk does not
    # exist on the pre-fix tree; there the same facts arrive via _probe_torch_runtime.
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: hw["hip"], raising = False)
    monkeypatch.setattr(mod, "_installed_torch_version_label", lambda: hw["label"])
    monkeypatch.setattr(mod, "_TORCH_RUNTIME_PROBE", warm_probe)

    probe_calls = []
    probe_result = (
        warm_probe
        if warm_probe is not None
        else (True, bool(hw["label"]), hw["label"] or None, hw["hip"], "")
    )

    def _recording_probe():
        # Mirror the real function's memo exactly: a warm memo returns instantly and costs
        # nothing, so only a COLD entry is an interpreter start worth recording.
        if mod._TORCH_RUNTIME_PROBE is not None:
            return mod._TORCH_RUNTIME_PROBE
        probe_calls.append(probe_result)
        mod._TORCH_RUNTIME_PROBE = probe_result
        return probe_result

    monkeypatch.setattr(mod, "_probe_torch_runtime", _recording_probe)
    # Belt and braces: nothing in this path may shell out either.
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: pytest.fail(f"a progress label must not start a subprocess: {a!r}"),
    )
    return mod, probe_calls


# ==== the platform x hardware matrix ================================================
# Expected label per cell with UNSLOTH_TORCH_BACKEND unset, i.e. every standalone
# `unsloth studio update`. Only Windows consults torch's own ROCm build, because
# _installed_torch_is_windows_rocm_cheap returns False on `not IS_WINDOWS` before doing
# any work -- that is what keeps Linux, WSL and macOS on their pre-existing answers.
_MATRIX = {
    ("windows", "nvidia"): "cuda",
    ("windows", "amd_tooling"): "rocm",
    ("windows", "amd_bundled"): "rocm",  # the regression this PR fixes
    ("windows", "xpu"): "cpu",
    ("windows", "cpu"): "cpu",
    ("windows", "no_torch"): "cpu",
    ("linux", "nvidia"): "cuda",
    ("linux", "amd_tooling"): "rocm",
    ("linux", "amd_bundled"): "cpu",  # no rocminfo on Linux means no ROCm claim
    ("linux", "xpu"): "cpu",
    ("linux", "cpu"): "cpu",
    ("linux", "no_torch"): "cpu",
    ("wsl", "nvidia"): "cuda",
    ("wsl", "amd_tooling"): "rocm",
    ("wsl", "amd_bundled"): "cpu",
    ("wsl", "xpu"): "cpu",
    ("wsl", "cpu"): "cpu",
    ("wsl", "no_torch"): "cpu",
    # macOS has no NVIDIA or AMD ROCm story; these rows pin the function's actual
    # behaviour rather than claiming the configurations are supported.
    ("macos", "nvidia"): "cuda",
    ("macos", "amd_tooling"): "rocm",
    ("macos", "amd_bundled"): "cpu",
    ("macos", "xpu"): "cpu",
    ("macos", "cpu"): "cpu",
    ("macos", "no_torch"): "cpu",
}


@pytest.mark.parametrize(("platform_name", "hardware_name"), sorted(_MATRIX))
def test_label_over_the_platform_and_hardware_matrix(monkeypatch, platform_name, hardware_name):
    mod, _calls = _prepare(monkeypatch, platform_name = platform_name, hardware_name = hardware_name)
    expected = _MATRIX[(platform_name, hardware_name)]
    assert mod._torch_step_label("check") == f"torch check ({expected})"


@pytest.mark.parametrize(("platform_name", "hardware_name"), sorted(_MATRIX))
@pytest.mark.parametrize("suffix", ["check", "final", "flavor"])
def test_every_suffix_keeps_the_same_backend(monkeypatch, platform_name, hardware_name, suffix):
    """All three call sites (:7166, :7371, :7383) share one backend verdict."""
    mod, _calls = _prepare(monkeypatch, platform_name = platform_name, hardware_name = hardware_name)
    expected = _MATRIX[(platform_name, hardware_name)]
    assert mod._torch_step_label(suffix) == f"torch {suffix} ({expected})"


# ==== an explicit backend wins over every probe =====================================


@pytest.mark.parametrize("known_backend", ["cuda", "rocm", "cpu", "xpu", "gfx1151-custom"])
@pytest.mark.parametrize("platform_name", sorted(_PLATFORMS))
def test_an_explicit_backend_wins_over_every_probe(monkeypatch, known_backend, platform_name):
    """install.sh's resolved backend is authoritative, verbatim, on every platform."""
    mod, _calls = _prepare(
        monkeypatch,
        platform_name = platform_name,
        # deliberately contradictory hardware: nothing below may override the pin
        hardware_name = "amd_bundled",
        known_backend = known_backend,
    )
    assert mod._torch_step_label("check") == f"torch check ({known_backend})"


def test_an_explicit_backend_consults_no_detector(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_TORCH_BACKEND", "cuda")
    for name in (
        "_has_usable_nvidia_gpu",
        "_has_rocm_gpu",
        "_installed_torch_is_windows_rocm",
        "_installed_torch_is_windows_rocm_cheap",
        "_torch_hip_version_on_disk",
        "_installed_torch_version_label",
    ):
        monkeypatch.setattr(
            mod,
            name,
            lambda *_a, **_k: pytest.fail(f"{name} must not run for a pinned backend"),
            raising = False,
        )
    assert mod._torch_step_label("check") == "torch check (cuda)"


# ==== precedence ====================================================================


def test_nvidia_still_takes_priority(monkeypatch):
    """A mixed NVIDIA+AMD Windows host stays CUDA even with a ROCm torch installed."""
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "nvidia")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "6.4.43483")
    assert mod._torch_step_label("check") == "torch check (cuda)"


def test_the_rocm_probe_still_answers(monkeypatch):
    """rocminfo/amd-smi remain the primary signal; nothing about them changed."""
    mod, _calls = _prepare(monkeypatch, platform_name = "linux", hardware_name = "amd_tooling")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "")
    monkeypatch.setattr(mod, "_installed_torch_version_label", lambda: "")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_windows_rocm_torch_is_rocm_even_with_no_rocm_tooling(monkeypatch):
    """The regression: rocminfo and amd-smi are absent, torch's own version.py is not."""
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_windows_rocm_torch_is_recognised_by_version_string_alone(monkeypatch):
    """A build whose version.py carries no hip= line but whose __version__ says rocm."""
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_host_with_neither_is_still_cpu(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "cpu")
    assert mod._torch_step_label("check") == "torch check (cpu)"


# ==== the cost of the answer ========================================================


@pytest.mark.parametrize("platform_name", sorted(_PLATFORMS))
@pytest.mark.parametrize("hardware_name", sorted(_HARDWARE))
def test_the_label_never_runs_the_torch_probe(monkeypatch, platform_name, hardware_name):
    """Every cell of the matrix, with a COLD memo: zero `import torch` subprocesses.

    _probe_torch_runtime is the 90s-bounded interpreter start. Formatting a progress
    line must never be what pays for it.
    """
    mod, probe_calls = _prepare(
        monkeypatch, platform_name = platform_name, hardware_name = hardware_name
    )
    mod._torch_step_label("check")
    assert probe_calls == []


def test_the_label_leaves_the_probe_memo_cold(monkeypatch):
    """A cold _TORCH_RUNTIME_PROBE stays cold, so the label pays for nothing.

    The discriminating assertion for the affected host: pre-fix, the label reached
    _probe_torch_runtime, populating the memo and spending up to its 90s timeout before
    _progress() had printed anything -- and pip_install invalidates that memo three
    statements later, so the cost was not even amortised.
    """
    mod, probe_calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    assert mod._TORCH_RUNTIME_PROBE is None
    assert mod._torch_step_label("check") == "torch check (rocm)"
    assert probe_calls == []
    assert mod._TORCH_RUNTIME_PROBE is None


def test_the_label_reuses_a_warm_probe_instead_of_the_disk(monkeypatch):
    """When something else already paid for the probe, prefer its richer answer."""
    warm = (True, True, "2.8.0a0+rocmsdk20250901", "6.4.43483", "")
    mod, probe_calls = _prepare(
        monkeypatch,
        platform_name = "windows",
        hardware_name = "cpu",  # disk says CPU
        warm_probe = warm,
    )
    disk_reads = []
    monkeypatch.setattr(
        mod,
        "_torch_hip_version_on_disk",
        lambda: disk_reads.append("hip") or "",
        raising = False,
    )
    assert mod._torch_step_label("check") == "torch check (rocm)"
    assert probe_calls == []  # a warm memo is reused, never re-probed
    assert disk_reads == []  # and the disk is not consulted behind it


def test_a_warm_negative_probe_is_believed(monkeypatch):
    """A warm probe saying "not ROCm" is authoritative over the disk heuristics."""
    warm = (True, True, "2.9.1+cpu", "", "")
    mod, probe_calls = _prepare(
        monkeypatch,
        platform_name = "windows",
        hardware_name = "amd_bundled",
        warm_probe = warm,
    )
    assert mod._torch_step_label("check") == "torch check (cpu)"
    assert probe_calls == []


def test_an_inconclusive_warm_probe_is_not_read_as_rocm(monkeypatch):
    """The wedged-driver tuple (ran=False) must not manufacture a ROCm claim."""
    mod, probe_calls = _prepare(
        monkeypatch,
        platform_name = "windows",
        hardware_name = "cpu",
        warm_probe = (False, False, None, "", ""),
    )
    assert mod._torch_step_label("check") == "torch check (cpu)"
    assert probe_calls == []


@pytest.mark.parametrize("platform_name", ["linux", "wsl", "macos"])
def test_non_windows_never_touches_the_torch_build(monkeypatch, platform_name):
    """Linux, WSL and macOS short-circuit before any disk or probe read.

    This is what makes the change incapable of altering the answer on those platforms.
    """
    mod, probe_calls = _prepare(
        monkeypatch, platform_name = platform_name, hardware_name = "amd_bundled"
    )
    touched = []
    monkeypatch.setattr(
        mod,
        "_torch_hip_version_on_disk",
        lambda: touched.append("hip") or "",
        raising = False,
    )
    monkeypatch.setattr(
        mod, "_installed_torch_version_label", lambda: touched.append("label") or ""
    )
    # Not asserted here: whether the Windows helper is CALLED off Windows. Both trees
    # return False on `not IS_WINDOWS` before doing any work, so the call is free; what
    # matters is that no probe and no disk read happen behind it.
    assert mod._torch_step_label("check") == "torch check (cpu)"
    assert touched == []
    assert probe_calls == []


# ==== the off-disk reader itself ====================================================
# Unit tests for the helper this change introduces. They are skipped, not failed, on a
# tree that predates it: their job is to hold the new reader, not to discriminate.


def _requires_hip_reader(mod):
    if not hasattr(mod, "_torch_hip_version_on_disk"):
        pytest.skip("_torch_hip_version_on_disk does not exist on this tree")


_VERSION_PY_ROCM = """\
from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'xpu']
__version__ = '2.8.0a0+rocmsdk20250901'
debug = False
cuda: Optional[str] = None
git_version = 'deadbeef'
hip: Optional[str] = '6.4.43483-a1b2c3d'
xpu: Optional[str] = None
"""

_VERSION_PY_CUDA = """\
from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'xpu']
__version__ = '2.9.1+cu128'
debug = False
cuda: Optional[str] = '12.8'
git_version = 'deadbeef'
hip: Optional[str] = None
xpu: Optional[str] = None
"""

# The un-annotated form older torch builds wrote.
_VERSION_PY_ROCM_UNANNOTATED = """\
__version__ = '2.7.0+rocm6.3'
hip = '6.3.42131'
cuda = None
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_VERSION_PY_ROCM, "6.4.43483-a1b2c3d"),
        (_VERSION_PY_ROCM_UNANNOTATED, "6.3.42131"),
        (_VERSION_PY_CUDA, ""),  # hip = None must not read as a HIP string
        ("", ""),
        ('hip = "6.9.0"\n', "6.9.0"),  # double-quoted
        ("__version__ = '2.9.1'\n", ""),
        ("# hip = '1.0'\n", ""),  # not at the start of a line after ^ anchoring
    ],
)
def test_the_hip_reader_matches_only_a_quoted_value(monkeypatch, tmp_path, text, expected):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text("", encoding = "utf-8")
    (torch_dir / "version.py").write_text(text, encoding = "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert mod._torch_hip_version_on_disk() == expected


def test_the_hip_reader_survives_a_missing_torch(monkeypatch, tmp_path):
    """An absent torch is "no answer", never an exception into a progress label."""
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    monkeypatch.setattr(
        mod.importlib.util, "find_spec", lambda _name: (_ for _ in ()).throw(ImportError("x"))
    )
    assert mod._torch_hip_version_on_disk() == ""


def test_the_hip_reader_survives_an_unreadable_version_py(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text("", encoding = "utf-8")
    # version.py is a directory: the read raises OSError, which must be swallowed.
    (torch_dir / "version.py").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    assert mod._torch_hip_version_on_disk() == ""


def test_the_hip_reader_starts_no_subprocess(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: pytest.fail("reading version.py must not start a subprocess"),
    )
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text("", encoding = "utf-8")
    (torch_dir / "version.py").write_text(_VERSION_PY_ROCM, encoding = "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert mod._torch_hip_version_on_disk() == "6.4.43483-a1b2c3d"


# ==== this machine, unsimulated =====================================================


def test_the_label_on_this_real_host(monkeypatch):
    """One unsimulated observation: whatever this box is, the label costs no subprocess.

    Deliberately asserts the shape rather than a fixed backend, so it holds on the CPU
    CI runner and on a GPU box alike.
    """
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_TORCH_BACKEND", "")
    monkeypatch.setattr(mod, "_TORCH_RUNTIME_PROBE", None)
    real_run = subprocess.run
    calls = []

    def _record(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", _record)
    label = mod._torch_step_label("check")
    assert label.startswith("torch check (") and label.endswith(")")
    assert label[len("torch check (") : -1] in {"cuda", "rocm", "cpu"}
    # _has_usable_nvidia_gpu/_has_rocm_gpu may shell out to nvidia-smi or rocminfo; the
    # `import torch` probe (which is the expensive one) must not be among the calls.
    assert not any("import torch" in str(a) for a in calls), calls
    assert mod._TORCH_RUNTIME_PROBE is None
