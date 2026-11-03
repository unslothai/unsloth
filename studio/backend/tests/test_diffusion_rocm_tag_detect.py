# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD SDK / Radeon torch wheels leave ``torch.version.hip`` unset and carry the ROCm tag only in ``__version__``.

Studio's device target already reads them as ROCm (``rocm_bf16.is_rocm_torch``), but the diffusion and video fast-path
gates read ``torch.version.hip`` directly, so on such a wheel the ROCm-only paths (fused FLUX RoPE, VAE attention
chunking, Qwen-Image-2.1 query chunks) declined and the NVIDIA-only ones (render thread, NVENC, NVFP4, int8 GEMM)
treated the AMD card as NVIDIA. Every gate now goes through the one tag-aware helper.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")

INFERENCE = _BACKEND / "core" / "inference"
# Mentions that are not gates: a cache key / label recording the build, and comments or docstrings.
_ALLOWED = {
    "diffusion_probe_cache.py",  # records str(torch.version.hip) in a cache key
    "diffusion_measured_peaks.py",  # labels the toolkit version
}


@pytest.fixture
def amd_sdk_wheel(monkeypatch):
    """torch as an AMD SDK wheel reports itself: no version.hip, no version.cuda, the tag in __version__."""
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.13.0", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    for env in (
        "UNSLOTH_DIFFUSION_FUSED_ROPE",
        "UNSLOTH_DIFFUSION_RENDER_THREAD",
        "UNSLOTH_DIFFUSION_VAE_ATTN_CHUNK",
    ):
        monkeypatch.delenv(env, raising = False)
    return torch


def test_fused_flux_rope_switch_is_on_for_a_tag_only_rocm_wheel(amd_sdk_wheel):
    from core.inference import diffusion_rocm_fused as rf

    assert rf._is_rocm() is True
    assert rf._is_nvidia_linux() is False
    assert rf._switch_on(rf.FUSED_ROPE_ENV) is True


def test_render_thread_stays_off_on_a_tag_only_rocm_wheel(amd_sdk_wheel):
    from core.inference import diffusion_render_thread as rt
    assert rt.enabled() is False


def test_vae_attention_chunking_installs_on_a_tag_only_rocm_wheel(amd_sdk_wheel):
    from core.inference import diffusion_vae_attn_chunked as vc
    vae = SimpleNamespace(modules = lambda: iter(()))
    assert vc.wants_install(vae, SimpleNamespace(backend = "rocm")) is True


def test_nvidia_only_paths_decline_on_a_tag_only_rocm_wheel(amd_sdk_wheel, monkeypatch):
    from core.inference import (
        diffusion_compile_config,
        diffusion_flux2_rope,
        diffusion_nvfp4_install,
        diffusion_render_thread,
        video_nvenc,
    )

    monkeypatch.setenv(video_nvenc.ENCODER_ENV, "nvenc")
    assert video_nvenc.nvenc_gpu() is None
    assert diffusion_compile_config._device_capability() is None
    assert diffusion_flux2_rope.wanted(torch.float16, "cuda") is False
    reason = diffusion_nvfp4_install._host_refusal("cuda")
    assert reason is not None and "ROCm" in reason
    assert diffusion_render_thread.enabled() is False


def test_cuda_build_keeps_the_nvidia_answers(monkeypatch):
    """The helper must not read a CUDA build as ROCm: same gates, NVIDIA identity."""
    from core.inference import diffusion_render_thread as rt
    from core.inference import diffusion_rocm_fused as rf

    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_RENDER_THREAD", raising = False)
    assert rf._is_rocm() is False
    assert rf._is_nvidia_linux() is True
    assert rt.enabled() is True


def _is_version_lookup(node: ast.AST) -> bool:
    """``torch.version`` or ``getattr(torch, "version", ...)``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "version"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "version"
    )


def _raw_hip_reads(path: Path) -> list[int]:
    """Lines of ``torch.version.hip`` / ``getattr(torch.version, "hip", ...)`` reads in code (not comments)."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    # Inside an f-string the read labels a cache key with the build string; it gates nothing.
    labels = {id(n) for js in ast.walk(tree) if isinstance(js, ast.JoinedStr) for n in ast.walk(js)}
    lines = []
    for node in ast.walk(tree):
        if id(node) in labels:
            continue
        if isinstance(node, ast.Attribute) and node.attr == "hip":
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "version":
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "hip"
            and _is_version_lookup(node.args[0])
        ):
            lines.append(node.lineno)
    return sorted(set(lines))


def test_no_diffusion_or_video_gate_reads_version_hip_directly():
    offenders = {}
    for path in sorted(INFERENCE.glob("*.py")):
        if not path.name.startswith(("diffusion", "video")) or path.name in _ALLOWED:
            continue
        lines = _raw_hip_reads(path)
        if lines:
            offenders[path.name] = lines
    assert offenders == {}, (
        "use core.inference.rocm_bf16.is_rocm_torch(torch): AMD SDK / Radeon wheels leave torch.version.hip "
        f"unset. Raw reads: {offenders}"
    )


def test_the_detector_sees_every_spelling_of_a_hip_read(tmp_path):
    src = tmp_path / "gates.py"
    src.write_text(
        "def a(torch):\n    return torch.version.hip\n"
        "def b(torch):\n    return getattr(torch.version, 'hip', None)\n"
        "def c(torch):\n    return getattr(getattr(torch, 'version', None), 'hip', None)\n"
        "def d(torch):\n    return f\"hip{getattr(torch.version, 'hip', None)}\"\n",
        encoding = "utf-8",
    )
    assert _raw_hip_reads(src) == [2, 4, 6]


def test_fp16_wan_fused_kernels_decline_on_a_tag_only_rocm_wheel(amd_sdk_wheel, monkeypatch):
    from core.inference import video_wan_fused

    monkeypatch.setattr(video_wan_fused, "_kernels", lambda: {"stub": True})
    monkeypatch.setattr(video_wan_fused, "disabled", lambda: False)
    assert video_wan_fused.wanted(torch.float16, "cuda") is False
