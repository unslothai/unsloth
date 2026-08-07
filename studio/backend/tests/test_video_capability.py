# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for video generation capability gating.

Video runs through the diffusers pipelines in core/inference/video.py, which have no Apple path,
so it is supported iff ``get_device() in {CUDA, XPU}`` with a reason otherwise (macos_unsupported
/ pytorch_not_installed / no_accelerator / detection_failed). Mirrors test_export_capability.py:
the matrix mocks the hardware probes, wiring is checked with ast, so it runs on CPU.
"""

import ast
from pathlib import Path

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
):
    monkeypatch.setattr(hw, "_has_torch", lambda: torch)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: apple)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", chat_only_reason)
    # Pin the OS too. video_capability() asks platform.system() directly so an Intel Mac
    # is covered, which means the non-Mac cases below would flip to macos_unsupported
    # when this suite runs on a macOS runner. Default follows `apple` so existing callers
    # keep the host they were written for.
    monkeypatch.setattr(
        hw.platform, "system", lambda: system or ("Darwin" if apple else "Linux")
    )


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


def test_apple_silicon_reports_coming_soon(monkeypatch):
    """A healthy Apple Silicon host is not chat-only, so the tab is enabled and would just fail
    at load. Say what is actually true instead: there is no Apple video path yet."""
    for device in (hw.DeviceType.MLX, hw.DeviceType.CPU):
        _patch(monkeypatch, torch = True, device = device, apple = True)
        cap = hw.video_capability()
        assert cap["video_supported"] is False
        assert cap["video_unsupported_reason"] == "macos_unsupported"
        assert "coming soon" in cap["video_unsupported_message"].lower()


def test_mlx_device_is_unsupported_even_without_the_apple_probe(monkeypatch):
    # MLX only exists on Apple, so an is_apple_silicon() that fails to answer must not
    # reclassify the host as a CPU box missing a GPU.
    _patch(monkeypatch, torch = False, device = hw.DeviceType.MLX, apple = False)
    cap = hw.video_capability()
    assert cap["video_supported"] is False
    assert cap["video_unsupported_reason"] == "macos_unsupported"


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


def test_hardware_package_reexports_video_capability():
    init = _src("utils/hardware/__init__.py")
    assert "def video_capability()" in init
    assert '"video_capability"' in init


def test_video_capability_excludes_the_apple_backends():
    cap = _func_src("utils/hardware/hardware.py", "video_capability")
    # Supported set is CUDA + XPU only; MLX is named solely on the unsupported side.
    assert "DeviceType.CUDA, DeviceType.XPU" in cap
    assert "_has_torch()" in cap
    # Strip comments and docstrings before asserting on the gate. This assertion used to
    # name is_apple_silicon(), and when the check widened to every Darwin host it kept
    # passing on the word surviving in a comment, which is not a test of anything.
    code = "\n".join(
        line.split("#", 1)[0] for line in cap.splitlines() if not line.strip().startswith("#")
    )
    assert 'platform.system() == "Darwin"' in code, (
        "the macOS branch must key on Darwin, not Apple Silicon, or an Intel Mac is told "
        "to install PyTorch or buy a GPU for something macOS cannot do at all"
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
    # install PyTorch or add a GPU, and neither enables video: the pipelines have no
    # supported macOS path at all. Same answer on both Macs.
    for torch_present in (True, False):
        _patch(
            monkeypatch,
            torch = torch_present,
            device = hw.DeviceType.CPU,
            apple = False,
            system = "Darwin",
        )
        cap = hw.video_capability()
        assert cap["video_supported"] is False
        assert cap["video_unsupported_reason"] == "macos_unsupported", (
            f"Intel Mac with torch={torch_present} reported "
            f"{cap['video_unsupported_reason']!r}"
        )
        assert "coming soon" in cap["video_unsupported_message"].lower()
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
