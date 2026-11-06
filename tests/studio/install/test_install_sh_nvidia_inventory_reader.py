# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""install.sh's inline driver-library reader (_nvidia_library_inventory) against stub
libraries: an NVML that raises or lacks the versioned entry points must not stop the
CUDA driver API from answering."""

from __future__ import annotations

import io
import os
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[3] / "install.sh"


def _reader_source() -> str:
    text = INSTALL_SH.read_text(encoding = "utf-8")
    m = re.search(r"_nvidia_library_inventory\(\) \{.*?<<'PY'\n(.*?)\nPY\n", text, re.S)
    assert m, "the inline reader was not found"
    return m.group(1)


class _Ok:
    """A C function stub: writes the values it was given to its byref arguments."""

    def __init__(self, *values):
        self.values = values

    def __call__(self, *args):
        refs = [a for a in args if hasattr(a, "_obj")]
        for ref, value in zip(refs, self.values):
            ref._obj.value = value
        return 0


def _lib(**symbols):
    return types.SimpleNamespace(**symbols)


def _cuda_lib(version = 13010, cap = (8, 9)):
    return _lib(
        cuInit = _Ok(),
        cuDeviceGetCount = _Ok(1),
        cuDriverGetVersion = _Ok(version),
        cuDeviceGet = _Ok(0),
        cuDeviceGetAttribute = _Attr(cap),
    )


class _Attr:
    def __init__(self, cap):
        self.cap = cap

    def __call__(self, ref, attribute, dev):
        ref._obj.value = self.cap[0] if attribute == 75 else self.cap[1]
        return 0


def _run(
    monkeypatch,
    libraries: dict[str, object],
    reader: str = "",
) -> tuple[int, str]:
    """Run the reader with ctypes.CDLL answering from `libraries`; (exit code, stdout).
    `reader` is the argument install.sh passes to run one reader under its own deadline."""
    import ctypes as real_ctypes

    monkeypatch.setattr(sys, "argv", ["-", reader] if reader else ["-"])

    def cdll(name):
        if name in libraries:
            return libraries[name]
        raise OSError(f"{name}: cannot open shared object file")

    stub = types.SimpleNamespace(
        CDLL = cdll,
        c_uint = real_ctypes.c_uint,
        c_int = real_ctypes.c_int,
        c_void_p = real_ctypes.c_void_p,
        byref = real_ctypes.byref,
    )
    monkeypatch.setitem(sys.modules, "ctypes", stub)
    out = io.StringIO()
    code = 0
    try:
        with redirect_stdout(out):
            exec(compile(_reader_source(), str(INSTALL_SH), "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        code = int(e.code or 0)
    return code, out.getvalue().strip()


def test_the_cuda_driver_api_answers_when_nvml_lacks_the_versioned_symbols(monkeypatch):
    # An older NVML: nvmlInit_v2 is missing, the unversioned nvmlInit exists but nothing else does.
    nvml = _lib(nvmlInit = _Ok(), nvmlShutdown = _Ok())
    code, out = _run(monkeypatch, {"libnvidia-ml.so.1": nvml, "libcuda.so.1": _cuda_lib()})
    assert (code, out) == (0, "13.1 8.9")


def test_a_reader_that_raises_yields_to_the_next(monkeypatch):
    class _Broken:
        def __getattr__(self, name):
            raise AttributeError(name)

    code, out = _run(monkeypatch, {"libnvidia-ml.so.1": _Broken(), "libcuda.so.1": _cuda_lib()})
    assert (code, out) == (0, "13.1 8.9")


def test_the_cuda_reader_sees_the_physical_cards(monkeypatch):
    """The driver API honours CUDA_VISIBLE_DEVICES; the inventory must not."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    lib = _cuda_lib()

    def cu_init(flags):
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        return 0

    lib.cuInit = cu_init
    code, out = _run(monkeypatch, {"libcuda.so.1": lib})
    assert (code, out) == (0, "13.1 8.9")


def test_no_library_at_all_is_no_inventory(monkeypatch):
    code, out = _run(monkeypatch, {})
    assert (code, out) == (1, "")


def test_nvml_answers_first_when_it_can(monkeypatch):
    nvml = _lib(
        nvmlInit_v2 = _Ok(),
        nvmlShutdown = _Ok(),
        nvmlDeviceGetCount_v2 = _Ok(2),
        nvmlSystemGetCudaDriverVersion_v2 = _Ok(12080),
        nvmlDeviceGetHandleByIndex_v2 = _Ok(1),
        nvmlDeviceGetCudaComputeCapability = _Ok(6, 1),
    )
    code, out = _run(monkeypatch, {"libnvidia-ml.so.1": nvml, "libcuda.so.1": _cuda_lib()})
    assert (code, out) == (0, "12.8 6.1,6.1")


def _nvml_lib():
    return _lib(
        nvmlInit_v2 = _Ok(),
        nvmlShutdown = _Ok(),
        nvmlDeviceGetCount_v2 = _Ok(1),
        nvmlSystemGetCudaDriverVersion_v2 = _Ok(12080),
        nvmlDeviceGetHandleByIndex_v2 = _Ok(1),
        nvmlDeviceGetCudaComputeCapability = _Ok(10, 0),
    )


def test_each_reader_runs_alone_under_its_own_deadline(monkeypatch):
    """install.sh runs NVML and the CUDA driver API as separate bounded processes, so a slow
    NVML no longer uses up the deadline the driver API needed (8x B200: NVML took ~23s)."""
    libs = {"libnvidia-ml.so.1": _nvml_lib(), "libcuda.so.1": _cuda_lib(cap = (10, 0))}
    assert _run(monkeypatch, libs, "nvml") == (0, "12.8 10.0")
    assert _run(monkeypatch, libs, "cuda") == (0, "13.1 10.0")
    assert _run(monkeypatch, {"libcuda.so.1": _cuda_lib()}, "nvml") == (1, "")
    assert _run(monkeypatch, {"libnvidia-ml.so.1": _nvml_lib()}, "cuda") == (1, "")


def test_install_sh_gives_each_reader_its_own_bound():
    text = INSTALL_SH.read_text(encoding = "utf-8")
    body = text[text.index("_nvidia_library_inventory() {") :]
    body = body[: body.index("\n}\n")]
    assert "for _nli_reader in nvml cuda; do" in body
    assert '_run_bounded --secs "$_nli_secs" "$_nli_py" -I - "$_nli_reader"' in body
