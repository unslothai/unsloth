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
import os
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _embedded(path: Path) -> str:
    """The here-string body inside Read-NvidiaLibraryRawViaPython.

    Anchored on the function: both files carry other here-strings, and matching the first one
    silently returns a different probe.
    """
    text = path.read_text(encoding="utf-8")
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
    """A fake exported function. Carries .restype / .argtypes because the probe sets them."""

    def __init__(self, impl):
        self._impl = impl
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._impl(*args)


class _Lib:
    def __init__(self, table):
        self._table = table
        self.calls: list[str] = []

    def __getattr__(self, name):
        if name not in self._table:
            raise AttributeError(name)

        def record(*args, _name=name):
            self.calls.append(_name)
            return self._table[_name](*args)

        return _Fn(record)


def _run(probe_source: str, libs: dict[str, _Lib | None], hints: tuple[str, str] = ("", "")) -> str:
    """Execute the probe with ctypes.CDLL replaced, and return what it wrote to stdout."""
    loaded: list[str] = []

    def fake_cdll(name):
        loaded.append(name)
        lib = libs.get(name)
        if lib is None:
            raise OSError(f"cannot load {name}")
        return lib

    shim = types.ModuleType("ctypes")
    for attr in ("c_int", "c_uint", "c_void_p", "POINTER"):
        setattr(shim, attr, getattr(ctypes, attr))
    shim.CDLL = fake_cdll
    # Identity byref, so a fake export can write through to the caller's c_int.
    shim.byref = lambda obj: obj

    namespace: dict = {}
    saved_module = sys.modules.get("ctypes")
    saved_hints = {k: os.environ.get(k) for k in ("UNSLOTH_NVML_HINT", "UNSLOTH_CUDA_HINT")}
    sys.modules["ctypes"] = shim
    os.environ["UNSLOTH_NVML_HINT"], os.environ["UNSLOTH_CUDA_HINT"] = hints
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(probe_source, "embedded_probe.py", "exec"), namespace)
    finally:
        for key, value in saved_hints.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_module is None:
            del sys.modules["ctypes"]
        else:
            sys.modules["ctypes"] = saved_module
    return buffer.getvalue()


def _nvml(count=2, packed=12060, caps=((8, 9), (8, 9)), bad_handle=-1, bad_cap=-1, init=0):
    """A fake libnvidia-ml. bad_handle / bad_cap make that device index fail."""

    def get_count(out):
        out.value = count
        return 0

    def get_version(out):
        out.value = packed
        return 0

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

    return _Lib({
        "nvmlInit_v2": lambda: init,
        "nvmlShutdown": lambda: 0,
        "nvmlDeviceGetCount_v2": get_count,
        "nvmlSystemGetCudaDriverVersion_v2": get_version,
        "nvmlDeviceGetHandleByIndex_v2": get_handle,
        "nvmlDeviceGetCudaComputeCapability": get_cap,
    })


def _cuda(count=2, packed=12060, caps=((9, 0), (9, 0)), init=0, seen=None):
    def cu_init(flags):
        if seen is not None:
            # What the driver would see: the mask at the moment cuInit reads it.
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return init

    def get_count(out):
        out.value = count
        return 0

    def get_version(out):
        out.value = packed
        return 0

    def get_device(out, index):
        out.value = index
        return 0

    def get_attr(out, attr, device):
        out.value = caps[device.value][0 if attr == 75 else 1]
        return 0

    return _Lib({
        "cuInit": cu_init,
        "cuDriverGetVersion": get_version,
        "cuDeviceGetCount": get_count,
        "cuDeviceGet": get_device,
        "cuDeviceGetAttribute": get_attr,
    })


LINUX_NVML = "libnvidia-ml.so.1"
LINUX_CUDA = "libcuda.so.1"


@pytest.mark.skipif(os.name == "nt", reason="the library names below are the POSIX ones")
class TestNvmlRung:
    def test_a_healthy_two_gpu_host_reports_version_and_both_capabilities(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml()})
        assert out == "nvml;12;6;8.9,8.9"

    def test_the_version_is_unpacked_as_major_1000_plus_minor_10(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml(packed=13010)})
        assert out.startswith("cuda;") is False
        assert out == "nvml;13;1;8.9,8.9"

    def test_one_unreadable_handle_voids_the_whole_source(self):
        # A partial list would misreport the lowest capability and so the pre-Turing cap.
        lib = _nvml(count=2, bad_handle=1)
        out = _run(INSTALL_PROBE, {LINUX_NVML: lib, LINUX_CUDA: None})
        assert out == ""

    def test_one_unreadable_capability_voids_the_whole_source(self):
        lib = _nvml(count=2, bad_cap=1)
        out = _run(INSTALL_PROBE, {LINUX_NVML: lib, LINUX_CUDA: None})
        assert out == ""

    def test_a_partial_read_still_shuts_nvml_down(self):
        lib = _nvml(count=2, bad_cap=1)
        _run(INSTALL_PROBE, {LINUX_NVML: lib, LINUX_CUDA: None})
        assert "nvmlShutdown" in lib.calls

    def test_no_devices_is_no_answer(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml(count=0), LINUX_CUDA: None})
        assert out == ""

    def test_a_nonsense_driver_version_is_rejected(self):
        # Below 1000 the packed form cannot encode a real CUDA version.
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml(packed=999), LINUX_CUDA: None})
        assert out == ""

    def test_a_failed_init_falls_through_rather_than_raising(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml(init=1), LINUX_CUDA: _cuda()})
        assert out == "cuda;12;6;9.0,9.0"


@pytest.mark.skipif(os.name == "nt", reason="the library names below are the POSIX ones")
class TestCudaRung:
    def test_nvml_absent_falls_to_the_driver_api(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: _cuda()})
        assert out == "cuda;12;6;9.0,9.0"

    def test_nvml_is_preferred_when_both_answer(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml(), LINUX_CUDA: _cuda()})
        assert out.startswith("nvml;")

    def test_neither_library_is_an_empty_answer_not_an_error(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: None})
        assert out == ""

    def test_cuinit_sees_no_visible_device_mask(self):
        # The inventory must be PHYSICAL: a hidden pre-Turing card still caps the family.
        seen: list = []
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        try:
            _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: _cuda(seen=seen)})
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert seen == [None]

    def test_the_mask_is_restored_afterwards(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        try:
            _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: _cuda()})
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_an_absent_mask_is_not_invented(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: _cuda()})
        assert "CUDA_VISIBLE_DEVICES" not in os.environ

    def test_a_failed_cuinit_is_no_answer(self):
        out = _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: _cuda(init=1)})
        assert out == ""

    def test_the_capability_attributes_are_75_and_76(self):
        # A wrong attribute number returns a plausible integer rather than an error.
        recorded: list = []

        lib = _cuda()
        original = lib._table["cuDeviceGetAttribute"]

        def spy(out, attr, device):
            recorded.append(attr)
            return original(out, attr, device)

        lib._table["cuDeviceGetAttribute"] = spy
        _run(INSTALL_PROBE, {LINUX_NVML: None, LINUX_CUDA: lib})
        assert set(recorded) == {75, 76}


@pytest.mark.skipif(os.name == "nt", reason="the library names below are the POSIX ones")
def test_the_output_matches_what_the_powershell_side_parses():
    """Get-NvidiaLibraryInventory splits on ';' and wants exactly four fields."""
    out = _run(INSTALL_PROBE, {LINUX_NVML: _nvml()})
    parts = out.split(";")
    assert len(parts) == 4
    assert parts[0] in ("nvml", "cuda")
    assert int(parts[1]) >= 1
    assert all(re.match(r"^\d+\.\d+$", cap) for cap in parts[3].split(","))


def test_the_windows_hint_is_tried_before_the_bare_library_name():
    """On Windows the installer passes the nvml.dll it already located."""
    order: list[str] = []

    def fake_cdll(name):
        order.append(name)
        raise OSError(name)

    shim = types.ModuleType("ctypes")
    for attr in ("c_int", "c_uint", "c_void_p", "POINTER"):
        setattr(shim, attr, getattr(ctypes, attr))
    shim.CDLL = fake_cdll
    shim.byref = lambda obj: obj

    saved_module = sys.modules.get("ctypes")
    saved_hints = {k: os.environ.get(k) for k in ("UNSLOTH_NVML_HINT", "UNSLOTH_CUDA_HINT")}
    saved_name = os.name
    sys.modules["ctypes"] = shim
    os.environ["UNSLOTH_NVML_HINT"] = r"C:\Windows\System32\nvml.dll"
    os.environ["UNSLOTH_CUDA_HINT"] = "nvcuda.dll"
    try:
        os.name = "nt"  # type: ignore[misc]
        with redirect_stdout(io.StringIO()):
            exec(compile(INSTALL_PROBE, "embedded_probe.py", "exec"), {})
    finally:
        os.name = saved_name  # type: ignore[misc]
        for key, value in saved_hints.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_module is None:
            del sys.modules["ctypes"]
        else:
            sys.modules["ctypes"] = saved_module

    assert order[0] == r"C:\Windows\System32\nvml.dll"
    assert "nvml.dll" in order
    assert "nvcuda.dll" in order
