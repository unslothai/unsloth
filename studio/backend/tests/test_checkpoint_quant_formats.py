# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Hub's Recommended filter lists FP8 / NVFP4 checkpoints only where every card runs them
natively. Hermetic: nvidia-smi is stubbed."""

from __future__ import annotations

import subprocess
import types

import pytest
from utils.hardware import nvidia


def _smi(
    monkeypatch,
    stdout: str = "",
    returncode: int = 0,
    raises = None,
) -> list:
    calls: list = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if raises is not None:
            raise raises
        return types.SimpleNamespace(returncode = returncode, stdout = stdout)

    monkeypatch.setattr(nvidia.gpu_query, "run_nvidia_smi", fake_run)
    return calls


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("8.6\n", []),  # RTX 3090
        ("8.9\n", ["fp8"]),  # RTX 4090
        ("9.0\n", ["fp8"]),  # H100
        ("10.0\n", ["fp8", "nvfp4"]),  # B200
        ("12.0\n", ["fp8", "nvfp4"]),  # RTX 5090
        ("12.0\n8.9\n", ["fp8"]),  # mixed host answers for the least capable card
        ("12.0\n8.0\n", []),
        # A device with no compute capability (the NPU nvidia-smi lists beside an integrated GPU) is skipped.
        ("12.1\n[N/A]\n", ["fp8", "nvfp4"]),
        ("[N/A]\n12.1\n", ["fp8", "nvfp4"]),
        ("N/A\n8.9\n", ["fp8"]),
        ("[N/A]\n", []),  # nothing with a capability: none
        ("12.0\ngarbage\n", []),  # an unparseable row still counts as unreadable
        ("", []),
    ],
)
def test_formats_follow_the_least_capable_card(monkeypatch, stdout, expected):
    calls = _smi(monkeypatch, stdout)
    assert nvidia.get_checkpoint_quant_formats() == expected
    assert "--query-gpu=compute_cap" in calls[0]


def test_no_nvidia_smi_or_a_failed_query_lists_nothing(monkeypatch):
    _smi(monkeypatch, raises = FileNotFoundError("nvidia-smi"))
    assert nvidia.get_checkpoint_quant_formats() == []
    _smi(monkeypatch, raises = subprocess.TimeoutExpired("nvidia-smi", 5))
    assert nvidia.get_checkpoint_quant_formats() == []
    _smi(monkeypatch, stdout = "9.0\n", returncode = 6)
    assert nvidia.get_checkpoint_quant_formats() == []


def test_compute_cap_is_a_cached_static_query():
    assert (
        nvidia.gpu_query.classify(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        )
        == nvidia.gpu_query.STATIC
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("cuda", ["fp8", "nvfp4"]), ("cpu", []), ("rocm", []), ("xpu", []), ("mlx", [])],
)
def test_system_route_reports_formats_only_on_a_cuda_backend(monkeypatch, backend, expected):
    """A CPU-only torch or a mask hiding every card still leaves nvidia-smi seeing them."""
    import main
    from utils.hardware import hardware

    _smi(monkeypatch, "12.0\n")
    monkeypatch.setattr(hardware, "_backend_label", lambda _device: backend)
    assert main._checkpoint_quant_formats() == expected


def _fake_core(tmp_path, monkeypatch, backends):
    """An installed core library whose kernels/fp8.py names ``backends``; None means not installed."""
    import importlib.util

    if backends is None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        return
    root = tmp_path / "unsloth"
    (root / "kernels").mkdir(parents = True)
    (root / "kernels" / "fp8.py").write_text(
        f"_FP8_ROWWISE_BACKENDS = {tuple(backends)!r}\n", encoding = "utf-8"
    )
    spec = types.SimpleNamespace(submodule_search_locations = [str(root)])
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "unsloth" else None
    )


@pytest.mark.parametrize(
    ("platform", "backends", "expected"),
    [
        # Linux torch has the row-wise kernel: FP8 runs natively whatever the core library ships.
        ("linux", None, []),
        ("linux", ("fbgemm", "scaled_mm", "dequant"), []),
        # Windows torch has no row-wise kernel: native only with a tensorwise fallback.
        ("win32", ("fbgemm", "scaled_mm", "dequant"), ["fp8_rowwise"]),
        ("win32", None, ["fp8_rowwise"]),
        ("win32", ("fbgemm", "scaled_mm", "tensorwise", "dequant"), []),
    ],
)
def test_fp8_is_dequant_only_without_a_rowwise_or_tensorwise_gemm(
    monkeypatch, tmp_path, platform, backends, expected
):
    import sys

    # Nothing is hidden: the format list itself is untouched.
    _smi(monkeypatch, "12.0\n")
    assert nvidia.get_checkpoint_quant_formats() == ["fp8", "nvfp4"]
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delitem(sys.modules, "core.inference.diffusion_fp8_unit_scale", raising = False)
    _fake_core(tmp_path, monkeypatch, backends)
    assert nvidia.get_checkpoint_quant_dequant_only(["fp8", "nvfp4"]) == expected


def test_an_integrated_gpu_beside_an_npu_keeps_fp8_and_its_dequant_marker(monkeypatch, tmp_path):
    """The device row order a Windows-on-Arm part reports: the GPU, then an NPU with no compute capability."""
    import sys

    _smi(monkeypatch, "12.1\n[N/A]\n")
    formats = nvidia.get_checkpoint_quant_formats()
    assert formats == ["fp8", "nvfp4"]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "core.inference.diffusion_fp8_unit_scale", raising = False)
    _fake_core(tmp_path, monkeypatch, None)
    assert nvidia.get_checkpoint_quant_dequant_only(formats) == ["fp8_rowwise"]


def test_a_probed_rowwise_verdict_beats_the_platform(monkeypatch, tmp_path):
    import sys

    _fake_core(tmp_path, monkeypatch, ("fbgemm", "scaled_mm", "dequant"))
    probed = types.SimpleNamespace(_ROWWISE = {0: False})
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_fp8_unit_scale", probed)
    monkeypatch.setattr(sys, "platform", "linux")
    assert nvidia.get_checkpoint_quant_dequant_only(["fp8"]) == ["fp8_rowwise"]
    probed._ROWWISE = {0: True, 1: None}
    monkeypatch.setattr(sys, "platform", "win32")
    assert nvidia.get_checkpoint_quant_dequant_only(["fp8"]) == []


def test_no_fp8_format_is_never_dequant_only(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    assert nvidia.get_checkpoint_quant_dequant_only([]) == []
    assert nvidia.get_checkpoint_quant_dequant_only(["nvfp4"]) == []


def test_system_route_reports_dequant_only_beside_the_formats(monkeypatch, tmp_path):
    import sys

    import main
    from utils.hardware import hardware

    _smi(monkeypatch, "12.0\n")
    monkeypatch.setattr(hardware, "_backend_label", lambda _device: "cuda")
    formats = main._checkpoint_quant_formats()
    assert formats == ["fp8", "nvfp4"]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "core.inference.diffusion_fp8_unit_scale", raising = False)
    _fake_core(tmp_path, monkeypatch, None)
    assert main._checkpoint_quant_dequant_only(formats) == ["fp8_rowwise"]
    # The route ships both keys from one formats read.
    src = open(main.__file__, encoding = "utf-8").read()
    assert '"checkpoint_quant_dequant_only": _checkpoint_quant_dequant_only(_quant_formats)' in src
