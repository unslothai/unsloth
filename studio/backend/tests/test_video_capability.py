# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for video generation capability gating.

Video runs through the diffusers pipelines in core/inference/video.py, so it is supported on
CUDA and XPU, and on Apple Silicon whenever torch exposes a Metal device -- which is independent
of MLX, and so of whether the host can train. Everything else gets a reason: macos_unsupported /
pytorch_not_installed / mps_unavailable / no_accelerator / detection_failed. Mirrors
test_export_capability.py: the matrix mocks the hardware probes, wiring is checked with ast, so
it runs on CPU.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

import utils.hardware.hardware as hw

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel):
    return (_BACKEND / rel).read_text(encoding = "utf-8")


def _func_src(rel, name):
    src = _src(rel)
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


def _patch(
    monkeypatch,
    *,
    torch: bool,
    device,
    apple: bool,
    chat_only_reason = "no_gpu",
    system = None,
    mps = None,
    torch_import_error = None,
):
    monkeypatch.setattr(hw, "_has_torch", lambda: torch)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", chat_only_reason)
    # Pinned rather than left ambient: a real broken-torch host would otherwise leak into every
    # case here, and the Apple branch reads it to tell "installed but broken" from "absent".
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", torch_import_error)
    monkeypatch.setattr(hw, "_torch_mps_available", lambda: torch if mps is None else mps)
    # Pin the OS too. video_capability() asks platform.system() directly so an Intel Mac
    # is covered, which means the non-Mac cases below would flip to macos_unsupported
    # when this suite runs on a macOS runner. Default follows `apple` so existing callers
    # keep the host they were written for.
    monkeypatch.setattr(hw.platform, "system", lambda: system or ("Darwin" if apple else "Linux"))


# -- capability matrix --------------------------------------------------------------------------


def test_cuda_supports_video(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CUDA, apple = False)
    cap = hw.video_capability()
    assert cap["video_supported"] is True
    assert cap["video_unsupported_reason"] is None
    assert cap["video_unsupported_message"] is None


def test_xpu_supports_video(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.XPU, apple = False)
    assert hw.video_capability()["video_supported"] is True


def test_apple_silicon_with_mps_supports_video(monkeypatch):
    """Apple Silicon runs the same device-neutral diffusers pipelines on Metal. The host may
    report MLX or plain CPU depending on what else is installed; neither changes the answer."""
    for device in (hw.DeviceType.MLX, hw.DeviceType.CPU):
        _patch(monkeypatch, torch = True, device = device, apple = True)
        cap = hw.video_capability()
        assert cap["video_supported"] is True
        assert cap["video_unsupported_reason"] is None
        assert cap["video_unsupported_message"] is None


def test_apple_silicon_without_torch_reports_pytorch_missing(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.MLX, apple = True)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "pytorch_not_installed"


@pytest.mark.parametrize(
    "device, chat_only_reason",
    [
        # Both states detect_hardware() can actually publish for a broken torch on Apple Silicon.
        # MLX needs no torch, so a healthy stack still reports MLX with no chat-only reason; a
        # broken one falls back to CPU and records mlx_unavailable. Pairing MLX with
        # mlx_unavailable would let the production check be narrowed to either and still pass.
        (hw.DeviceType.MLX, None),
        (hw.DeviceType.CPU, "mlx_unavailable"),
    ],
)
def test_apple_silicon_with_broken_torch_is_not_told_to_install_it(
    monkeypatch, device, chat_only_reason
):
    # A wheel with unresolved native libs raises from torch's own __init__, so _has_torch() reads
    # False for it exactly as it does for an absent one, and neither reason above routes this host
    # to the detection_failed branch. Without the explicit check it is told to install the
    # PyTorch already sitting there broken.
    _patch(
        monkeypatch,
        torch = False,
        device = device,
        apple = True,
        chat_only_reason = chat_only_reason,
        torch_import_error = "OSError('broken native library')",
    )
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "detection_failed"
    assert "fails to import" in cap["video_unsupported_message"]
    assert "not installed" not in cap["video_unsupported_message"]


def test_apple_silicon_without_a_metal_device_is_not_supported(monkeypatch):
    # Apple Silicon alone does not imply MPS: a torch built without it leaves the pipelines
    # with nowhere to run, and claiming support would fail at load instead of at the gate.
    _patch(monkeypatch, torch = True, device = hw.DeviceType.MLX, apple = True, mps = False)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "mps_unavailable"


def test_mlx_device_is_apple_even_without_the_apple_probe(monkeypatch):
    # MLX only exists on Apple, so an is_apple_silicon() that fails to answer must not
    # reclassify the host as a CPU box missing a GPU. With torch + Metal it is supported.
    _patch(monkeypatch, torch = True, device = hw.DeviceType.MLX, apple = False)
    assert hw.video_capability()["video_supported"] is True


def test_no_torch_non_apple_reports_pytorch_missing(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.CPU, apple = False)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "pytorch_not_installed"
    assert "PyTorch is not installed" in cap["video_unsupported_message"]


def test_cpu_with_torch_reports_no_accelerator(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CPU, apple = False)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "no_accelerator"
    # Must not tell a user who has PyTorch to install PyTorch.
    assert "PyTorch is not installed" not in cap["video_unsupported_message"]


def test_a_failed_detection_is_reported_as_such(monkeypatch):
    """Same rule as export: a broken probe leaves the host looking CPU-only, so reporting
    no_accelerator would point the remediation at hardware that may be fine."""
    _patch(
        monkeypatch,
        torch = True,
        device = hw.DeviceType.CPU,
        apple = False,
        chat_only_reason = "detection_failed",
    )
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "detection_failed"
    assert "detection failed" in cap["video_unsupported_message"].lower()


# -- endpoint / package wiring (ast) ------------------------------------------------------------


def test_main_endpoints_expose_video_capability():
    m = _src("main.py")
    # Both system endpoints spread video_capability() into their response, as they do for export.
    assert m.count("**video_capability()") >= 2
    assert '"/api/system/hardware"' in m and '"/api/system"' in m


# -- the Metal probe itself ---------------------------------------------------------------------
#
# The matrix above stubs _torch_mps_available, so nothing there executes it. These drive the real
# helper against a fake torch: without them, changing its predicate is invisible to the suite.


def _fake_torch(monkeypatch, backends):
    torch = types.ModuleType("torch")
    torch.backends = backends
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)


def test_mps_probe_reads_availability_not_whether_torch_was_built_with_it(monkeypatch):
    class _Mps:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

        def is_built(self):
            return True

    # is_built() is true on any Metal-capable build, including one that cannot reach a device
    # here, so a probe on that predicate would promise video the host cannot run.
    _fake_torch(monkeypatch, types.SimpleNamespace(mps = _Mps(available = False)))
    assert hw._torch_mps_available() is False
    _fake_torch(monkeypatch, types.SimpleNamespace(mps = _Mps(available = True)))
    assert hw._torch_mps_available() is True


def test_mps_probe_treats_a_torch_without_the_backend_as_no_metal(monkeypatch):
    _fake_torch(monkeypatch, types.SimpleNamespace())
    assert hw._torch_mps_available() is False


def test_mps_probe_reports_no_metal_rather_than_raising(monkeypatch):
    class _Boom:
        def is_available(self):
            raise RuntimeError("Metal probe blew up")

    _fake_torch(monkeypatch, types.SimpleNamespace(mps = _Boom()))
    assert hw._torch_mps_available() is False


def test_mps_probe_reports_no_metal_when_torch_is_absent(monkeypatch):
    # Only the answer is asserted, not that the import was skipped: an import of the None below
    # raises into the same except, so both paths return False and no assertion can tell them apart.
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert hw._torch_mps_available() is False


def test_hardware_package_reexports_video_capability():
    init = _src("utils/hardware/__init__.py")
    assert "def video_capability()" in init
    assert '"video_capability"' in init


def test_video_capability_separates_apple_silicon_from_intel_macs():
    cap = _func_src("utils/hardware/hardware.py", "video_capability")
    assert "DeviceType.CUDA, DeviceType.XPU" in cap
    assert "_has_torch()" in cap
    # Strip comments before asserting on the gate. This assertion used to name
    # is_apple_silicon(), and when the check widened to every Darwin host it kept passing on the
    # word surviving in a comment, which is not a test of anything.
    code = "\n".join(
        line.split("#", 1)[0] for line in cap.splitlines() if not line.strip().startswith("#")
    )
    assert "_torch_mps_available()" in code, (
        "Apple Silicon must be admitted on a measured Metal device, not on the platform alone, "
        "or a torch without MPS is promised video it cannot run"
    )
    assert 'platform.system() == "Darwin"' in code, (
        "the remaining macOS branch must key on Darwin, or an Intel Mac falls through to be "
        "told to install PyTorch or buy a GPU for something it cannot do either way"
    )
    assert "DeviceType.MLX" in code


def test_frontend_reads_the_new_fields():
    hook = (_BACKEND.parent / "frontend" / "src" / "hooks" / "use-hardware-info.ts").read_text(
        encoding = "utf-8"
    )
    for field in ("video_supported", "video_unsupported_reason", "video_unsupported_message"):
        assert field in hook, f"{field} is not consumed by use-hardware-info.ts"


def test_intel_mac_is_macos_unsupported_not_a_missing_gpu(monkeypatch):
    # An Intel Mac detects as plain CPU and is_apple_silicon() is False, so the reason
    # used to come out as pytorch_not_installed or no_accelerator. Both tell the user to
    # install PyTorch or add a GPU, and neither enables video on a machine with no Metal
    # device. Installing torch does not change that, so the answer holds either way.
    for torch_present in (True, False):
        _patch(
            monkeypatch,
            torch = torch_present,
            device = hw.DeviceType.CPU,
            apple = False,
            system = "Darwin",
            mps = False,
        )
        cap = hw.video_capability()
        assert cap["video_supported"] is False
        assert cap["video_unsupported_reason"] == "macos_unsupported", (
            f"Intel Mac with torch={torch_present} reported " f"{cap['video_unsupported_reason']!r}"
        )
        assert "Apple Silicon" in cap["video_unsupported_message"]
        assert "GPU" not in cap["video_unsupported_message"]


def test_a_broken_probe_still_beats_the_macos_branch(monkeypatch):
    # detection_failed is checked first on purpose: a Mac whose probe fell over should be
    # told detection failed, not that video is coming soon, because the verdict is unknown.
    _patch(
        monkeypatch,
        torch = True,
        device = hw.DeviceType.CPU,
        apple = True,
        chat_only_reason = "detection_failed",
        system = "Darwin",
    )
    assert hw.video_capability()["video_unsupported_reason"] == "detection_failed"
