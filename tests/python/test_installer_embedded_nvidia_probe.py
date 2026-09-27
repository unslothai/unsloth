# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Behaviour of the ctypes NVIDIA probe embedded in install.ps1 and studio/setup.ps1.

The probe recovers the CUDA version AND the per-device compute capabilities on hosts where the
installer cannot emit a P/Invoke type (Constrained Language Mode, WDAC, Dynamic Code Security).
Capabilities are the part WMI cannot supply: they feed ``$CudaArch`` into
``-DCMAKE_CUDA_ARCHITECTURES`` for the llama.cpp source build (#5854) and the pre-Turing cap.

``tests/studio/test_nvidia_python_probe_parity.ps1`` pins the probe's TEXT against
``studio/nvidia_probe.py`` and against the other copy. This file pins its BEHAVIOUR, by running
it against a fake NVML / CUDA library. The two are complementary: a defect edited identically
into both copies is invisible to a parity check and caught here.
"""

from __future__ import annotations

import ctypes
import io
import ntpath
import os
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason = "the library names below are the POSIX ones"
)
LINUX_NVML = "libnvidia-ml.so.1"
LINUX_CUDA = "libcuda.so.1"
NVML_OUT = "nvml;12;6;8.9,8.9"
CUDA_OUT = "cuda;12;6;9.0,9.0"


def _embedded(path: Path) -> str:
    # Anchored on the function: both files carry other here-strings, and the first one is a
    # different probe.
    text = path.read_text(encoding = "utf-8")
    at = text.index("function Read-NvidiaLibraryRawViaPython")
    match = re.search(r"\$probeSource = @'\n(.*?)\n'@\n", text[at:], re.S)
    assert match, f"no probe here-string in {path}"
    return match.group(1)


INSTALL_PROBE = _embedded(ROOT / "install.ps1")
SETUP_PROBE = _embedded(ROOT / "studio" / "setup.ps1")


def test_both_copies_are_byte_identical():
    # Leading whitespace IS Python syntax, so this is a byte comparison, not a stripped one.
    assert INSTALL_PROBE == SETUP_PROBE


def test_the_probe_is_stdlib_only():
    imports = set(re.findall(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", INSTALL_PROBE))
    assert imports <= {"ctypes", "os", "sys"}, imports


def test_the_probe_compiles():
    compile(INSTALL_PROBE, "embedded_probe.py", "exec")


class _Fn:
    # The probe sets .restype / .argtypes on every export.
    def __init__(self, impl):
        self._impl, self.restype, self.argtypes = impl, None, None

    def __call__(self, *args):
        return self._impl(*args)


class _Lib:
    def __init__(self, table):
        self._table = table
        self.calls: list[str] = []

    def __getattr__(self, name):
        if name not in self._table:
            raise AttributeError(name)
        return _Fn(lambda *args: (self.calls.append(name), self._table[name](*args))[1])


def _run(
    libs,
    hints = ("", ""),
    loaded = None,
) -> str:
    """Execute the probe with ctypes.CDLL replaced, and return what it wrote to stdout."""

    def fake_cdll(name):
        if loaded is not None:
            loaded.append(name)
        if libs.get(name) is None:
            raise OSError(f"cannot load {name}")
        return libs[name]

    shim = types.ModuleType("ctypes")
    for attr in ("c_int", "c_uint", "c_void_p", "POINTER"):
        setattr(shim, attr, getattr(ctypes, attr))
    shim.CDLL = fake_cdll
    # Identity byref, so a fake export can write through to the caller's c_int.
    shim.byref = lambda obj: obj
    buffer = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, redirect_stdout(buffer):
        mp.setitem(sys.modules, "ctypes", shim)
        mp.setenv("UNSLOTH_NVML_HINT", hints[0])
        mp.setenv("UNSLOTH_CUDA_HINT", hints[1])
        exec(compile(INSTALL_PROBE, "embedded_probe.py", "exec"), {})
    return buffer.getvalue()


def _put(value):
    def write(out):
        out.value = value
        return 0

    return write


def _nvml(
    count = 2,
    packed = 12060,
    caps = ((8, 9), (8, 9)),
    bad_handle = -1,
    bad_cap = -1,
    init = 0,
):
    """A fake libnvidia-ml. bad_handle / bad_cap make that device index fail."""

    def get_handle(index, out):
        if index == bad_handle:
            return 999
        out.value = 0x1000 + index
        return 0

    def get_cap(handle, major, minor):
        index = handle.value - 0x1000 if handle.value else 0
        if index == bad_cap:
            return 999
        major.value, minor.value = caps[index]
        return 0

    return _Lib(
        {
            "nvmlInit_v2": lambda: init,
            "nvmlShutdown": lambda: 0,
            "nvmlDeviceGetCount_v2": _put(count),
            "nvmlSystemGetCudaDriverVersion_v2": _put(packed),
            "nvmlDeviceGetHandleByIndex_v2": get_handle,
            "nvmlDeviceGetCudaComputeCapability": get_cap,
        }
    )


def _cuda(
    count = 2,
    packed = 12060,
    caps = ((9, 0), (9, 0)),
    init = 0,
    seen = None,
    attrs = None,
):
    def cu_init(flags):
        if seen is not None:
            # What the driver would see: the mask at the moment cuInit reads it.
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return init

    def get_device(out, index):
        out.value = index
        return 0

    def get_attr(out, attr, device):
        if attrs is not None:
            attrs.append(attr)
        out.value = caps[device.value][0 if attr == 75 else 1]
        return 0

    return _Lib(
        {
            "cuInit": cu_init,
            "cuDriverGetVersion": _put(packed),
            "cuDeviceGetCount": _put(count),
            "cuDeviceGet": get_device,
            "cuDeviceGetAttribute": get_attr,
        }
    )


@POSIX_ONLY
@pytest.mark.parametrize(
    "nvml, cuda, expected",
    [
        pytest.param({}, None, NVML_OUT, id = "healthy_two_gpu_nvml"),
        pytest.param(
            {"packed": 13010}, None, "nvml;13;1;8.9,8.9", id = "version_is_major_1000_minor_10"
        ),
        # A partial list would misreport the lowest capability and so the pre-Turing cap.
        pytest.param({"bad_handle": 1}, None, "", id = "unreadable_handle_voids_source"),
        pytest.param({"bad_cap": 1}, None, "", id = "unreadable_capability_voids_source"),
        pytest.param({"count": 0}, None, "", id = "no_devices"),
        # Below 1000 the packed form cannot encode a real CUDA version.
        pytest.param({"packed": 999}, None, "", id = "nonsense_driver_version"),
        pytest.param({"init": 1}, {}, CUDA_OUT, id = "failed_nvml_init_falls_through"),
        pytest.param(None, {}, CUDA_OUT, id = "nvml_absent_uses_driver_api"),
        pytest.param({}, {}, NVML_OUT, id = "nvml_preferred_when_both_answer"),
        pytest.param(None, None, "", id = "neither_library"),
        pytest.param(None, {"init": 1}, "", id = "failed_cuinit"),
    ],
)
def test_probe_output(nvml, cuda, expected):
    libs = {
        LINUX_NVML: None if nvml is None else _nvml(**nvml),
        LINUX_CUDA: None if cuda is None else _cuda(**cuda),
    }
    assert _run(libs) == expected


@POSIX_ONLY
def test_a_partial_read_still_shuts_nvml_down():
    lib = _nvml(bad_cap = 1)
    _run({LINUX_NVML: lib, LINUX_CUDA: None})
    assert "nvmlShutdown" in lib.calls


@POSIX_ONLY
@pytest.mark.parametrize("value, nvml_first", [("1", False), ("0", True)])
def test_the_skip_nvml_switch(monkeypatch, value, nvml_first):
    # "1" is the CUDA-only retry after a child NVML held past its deadline: NVML is never loaded
    # or initialised, and the driver API is. Any other value still reads NVML first.
    monkeypatch.setenv("UNSLOTH_NVIDIA_PROBE_SKIP_NVML", value)
    nvml, cuda = _nvml(), _cuda()
    out = _run({LINUX_NVML: nvml, LINUX_CUDA: cuda})
    assert ("nvmlInit_v2" in nvml.calls) is nvml_first
    assert out == (NVML_OUT if nvml_first else CUDA_OUT)
    if not nvml_first:
        assert "cuInit" in cuda.calls and "cuDriverGetVersion" in cuda.calls


@POSIX_ONLY
@pytest.mark.parametrize(
    "mask", ["1", None], ids = ["mask_hidden_then_restored", "absent_mask_not_invented"]
)
def test_cuinit_sees_the_physical_inventory(monkeypatch, mask):
    # A hidden pre-Turing card still caps the family, so cuInit must see no mask.
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    seen: list = []
    _run({LINUX_NVML: None, LINUX_CUDA: _cuda(seen = seen)})
    assert seen == [None]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == mask


@POSIX_ONLY
def test_the_capability_attributes_are_75_and_76():
    # A wrong attribute number returns a plausible integer rather than an error.
    attrs: list = []
    _run({LINUX_NVML: None, LINUX_CUDA: _cuda(attrs = attrs)})
    assert set(attrs) == {75, 76}


@POSIX_ONLY
def test_the_output_matches_what_the_powershell_side_parses():
    """Get-NvidiaLibraryInventory splits on ';' and wants exactly four fields."""
    parts = _run({LINUX_NVML: _nvml()}).split(";")
    assert len(parts) == 4
    assert parts[0] in ("nvml", "cuda")
    assert int(parts[1]) >= 1
    assert all(re.match(r"^\d+\.\d+$", cap) for cap in parts[3].split(","))


def _run_windows(monkeypatch, libs, hints, loaded):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "path", ntpath)
    try:
        return _run(libs, hints = hints, loaded = loaded)
    finally:
        monkeypatch.undo()


SYSTEM32_NVML = r"C:\Windows\System32\nvml.dll"
SYSTEM32_CUDA = r"C:\Windows\System32\nvcuda.dll"


def test_windows_loads_only_the_driver_paths_it_is_handed(monkeypatch):
    order: list[str] = []
    _run_windows(monkeypatch, {}, (SYSTEM32_NVML, SYSTEM32_CUDA), order)
    assert order == [SYSTEM32_NVML, SYSTEM32_CUDA]


def test_a_cuda_stand_in_beside_python_is_not_an_nvidia_gpu(monkeypatch):
    # ZLUDA ships nvcuda.dll and nvml.dll that answer as a driver on an AMD host (#11736):
    # without the driver in System32 the probe must say nothing, not report a GPU.
    order: list[str] = []
    stand_in = {"nvml.dll": _nvml(), "nvcuda.dll": _cuda()}
    assert _run_windows(monkeypatch, stand_in, (SYSTEM32_NVML, SYSTEM32_CUDA), order) == ""
    assert "nvml.dll" not in order and "nvcuda.dll" not in order


def test_a_relative_hint_is_never_loaded(monkeypatch):
    order: list[str] = []
    stand_in = {"nvml.dll": _nvml(), "nvcuda.dll": _cuda()}
    assert _run_windows(monkeypatch, stand_in, ("nvml.dll", "nvcuda.dll"), order) == ""
    assert order == []


def test_the_real_driver_in_system32_still_answers(monkeypatch):
    order: list[str] = []
    out = _run_windows(monkeypatch, {SYSTEM32_NVML: None, SYSTEM32_CUDA: _cuda()}, (SYSTEM32_NVML, SYSTEM32_CUDA), order)
    assert out == CUDA_OUT
