# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _torch_step_label: which backend it names, and what answering costs.

rocminfo and amd-smi ship with the HIP SDK, not with AMD's bundled-runtime wheels, so a
working Windows ROCm host printed "torch check (cpu)". Answering with `import torch`
instead cost up to the probe's 90s timeout before _progress() emitted anything.
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


# "amd_bundled" is the Strix Halo case: no rocminfo, no amd-smi, only version.py knows.
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
    """One matrix cell, stubbed only at names both trees have, so this file discriminates.

    Returns the recorded _probe_torch_runtime calls; empty means the label cost nothing.
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
    # raising = False: _torch_hip_version_on_disk does not exist on the pre-fix tree.
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
        # Mirror the real memo: only a COLD entry is an interpreter start.
        if mod._TORCH_RUNTIME_PROBE is not None:
            return mod._TORCH_RUNTIME_PROBE
        probe_calls.append(probe_result)
        mod._TORCH_RUNTIME_PROBE = probe_result
        return probe_result

    monkeypatch.setattr(mod, "_probe_torch_runtime", _recording_probe)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: pytest.fail(f"a progress label must not start a subprocess: {a!r}"),
    )
    return mod, probe_calls


# UNSLOTH_TORCH_BACKEND unset. Only Windows consults torch's own build, which is what
# keeps Linux, WSL and macOS on their pre-existing answers.
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
    # macOS rows pin actual behaviour, not supported configurations.
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
    mod, _calls = _prepare(monkeypatch, platform_name = platform_name, hardware_name = hardware_name)
    expected = _MATRIX[(platform_name, hardware_name)]
    assert mod._torch_step_label(suffix) == f"torch {suffix} ({expected})"


@pytest.mark.parametrize("known_backend", ["cuda", "rocm", "cpu", "xpu", "gfx1151-custom"])
@pytest.mark.parametrize("platform_name", sorted(_PLATFORMS))
def test_an_explicit_backend_wins_over_every_probe(monkeypatch, known_backend, platform_name):
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


def test_nvidia_still_takes_priority(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "nvidia")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "6.4.43483")
    assert mod._torch_step_label("check") == "torch check (cuda)"


def test_the_rocm_probe_still_answers(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "linux", hardware_name = "amd_tooling")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "")
    monkeypatch.setattr(mod, "_installed_torch_version_label", lambda: "")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_windows_rocm_torch_is_rocm_even_with_no_rocm_tooling(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_windows_rocm_torch_is_recognised_by_version_string_alone(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    monkeypatch.setattr(mod, "_torch_hip_version_on_disk", lambda: "")
    assert mod._torch_step_label("check") == "torch check (rocm)"


def test_a_host_with_neither_is_still_cpu(monkeypatch):
    mod, _calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "cpu")
    assert mod._torch_step_label("check") == "torch check (cpu)"


@pytest.mark.parametrize("platform_name", sorted(_PLATFORMS))
@pytest.mark.parametrize("hardware_name", sorted(_HARDWARE))
def test_the_label_never_runs_the_torch_probe(monkeypatch, platform_name, hardware_name):
    mod, probe_calls = _prepare(
        monkeypatch, platform_name = platform_name, hardware_name = hardware_name
    )
    mod._torch_step_label("check")
    assert probe_calls == []


def test_the_label_leaves_the_probe_memo_cold(monkeypatch):
    """The discriminating case: pre-fix the label probed here, and pip_install then
    invalidated the memo three statements later, so the 90s was not even amortised.
    """
    mod, probe_calls = _prepare(monkeypatch, platform_name = "windows", hardware_name = "amd_bundled")
    assert mod._TORCH_RUNTIME_PROBE is None
    assert mod._torch_step_label("check") == "torch check (rocm)"
    assert probe_calls == []
    assert mod._TORCH_RUNTIME_PROBE is None


def test_the_label_reuses_a_warm_probe_instead_of_the_disk(monkeypatch):
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
    """Linux, WSL and macOS short-circuit, so the change cannot alter their answer."""
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
    # Not whether the helper is called off Windows (free either way), but that nothing
    # happens behind it.
    assert mod._torch_step_label("check") == "torch check (cpu)"
    assert touched == []
    assert probe_calls == []


# Skipped, not failed, on a tree without the helper: these hold it, not discriminate.


def _requires_hip_reader(mod):
    if not hasattr(mod, "_torch_hip_version_on_disk"):
        pytest.skip("_torch_hip_version_on_disk does not exist on this tree")


def _fake_torch_on_path(
    monkeypatch,
    tmp_path,
    version_py,
    *,
    as_directory = False,
):
    """Put a stand-in torch package where find_spec will see it.

    find_spec returns sys.modules[name].__spec__ for an already-imported module, so on a
    runner that has torch, prepending sys.path alone leaves the fixture ignored.
    """
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir(exist_ok = True)
    (torch_dir / "__init__.py").write_text("", encoding = "utf-8")
    if as_directory:
        (torch_dir / "version.py").mkdir()
    else:
        (torch_dir / "version.py").write_text(version_py, encoding = "utf-8")
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    monkeypatch.delitem(sys.modules, "torch.version", raising = False)
    monkeypatch.syspath_prepend(str(tmp_path))
    return torch_dir


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
    _fake_torch_on_path(monkeypatch, tmp_path, text)
    assert mod._torch_hip_version_on_disk() == expected


def test_the_hip_reader_survives_a_missing_torch(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    monkeypatch.setattr(
        mod.importlib.util, "find_spec", lambda _name: (_ for _ in ()).throw(ImportError("x"))
    )
    assert mod._torch_hip_version_on_disk() == ""


def test_the_hip_reader_survives_an_unreadable_version_py(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    # version.py is a directory: the read raises OSError, which must be swallowed.
    _fake_torch_on_path(monkeypatch, tmp_path, "", as_directory = True)
    assert mod._torch_hip_version_on_disk() == ""


def test_the_hip_reader_starts_no_subprocess(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _requires_hip_reader(mod)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: pytest.fail("reading version.py must not start a subprocess"),
    )
    _fake_torch_on_path(monkeypatch, tmp_path, _VERSION_PY_ROCM)
    assert mod._torch_hip_version_on_disk() == "6.4.43483-a1b2c3d"


def test_the_label_on_this_real_host(monkeypatch):
    """One unsimulated observation. Asserts the shape, so it holds on CPU and GPU alike."""
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
    # nvidia-smi/rocminfo may run; the expensive `import torch` must not.
    assert not any("import torch" in str(a) for a in calls), calls
    assert mod._TORCH_RUNTIME_PROBE is None
