# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""H4: never emit --no-mmproj-offload to a build that would reject it.

The existing pin tests all run against /fake/llama-server, whose --help probe
cannot answer, so every one of them exercises only the INCONCLUSIVE branch.
These drive the launch path with each of the three probe outcomes explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch  # noqa: E402,F401

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_PROJECTOR_BYTES = 900 * 1024 * 1024


def _caps(*, answered: bool, supports: bool) -> dict:
    """A capability dict shaped like probe_server_capabilities' own output."""
    return {
        "found": True,
        "mtp_probe_inconclusive": not answered,
        "supports_no_mmproj_offload": supports,
        "supports_flash_attn": True,
        "flash_attn_takes_value": True,
        "supports_no_context_shift": True,
        "supports_jinja": True,
        "supports_mtp": False,
        "spec_draft_ngl_flag": None,
        "flags": {},
        "switch_flags": [],
    }


def _tight_vision(tmp_path, monkeypatch, caps: dict):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 6_000, 8_000)])
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: _PROJECTOR_BYTES
    backend._mmproj_matches_model_family = lambda *a, **k: True
    backend._get_gguf_size_bytes = lambda _path: 4_500 * 1024 * 1024
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * 1024 * 1024
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, b: dict(caps))
    )
    return backend, gguf


@pytest.mark.parametrize(
    "answered, supports, expect_flag",
    [
        # (a) probe answered, flag present: pin.
        (True, True, True),
        # (b) probe answered, flag genuinely absent: MUST NOT emit it. An old
        # binary exits on an unknown argument, so the server never starts.
        (True, False, False),
        # (c) probe unanswered: fail open. --no-mmproj-offload is b5178 and the
        # base argv unconditionally emits `--flash-attn on`, whose value form is
        # b6325, so any build that survives the base argv already has the flag.
        (False, False, True),
    ],
)
def test_pin_respects_the_capability_probe(tmp_path, monkeypatch, answered, supports, expect_flag):
    backend, gguf = _tight_vision(
        tmp_path, monkeypatch, _caps(answered = answered, supports = supports)
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert ("--no-mmproj-offload" in cmd) is expect_flag
    # Vision is never silently dropped to avoid the flag.
    assert "--mmproj" in cmd


def test_the_unanswered_fail_open_is_backed_by_the_base_argv(tmp_path, monkeypatch):
    """The fail-open's whole justification, asserted rather than asserted-in-prose.

    Emitting --no-mmproj-offload to an unprobed build is only safe because the
    same argv already carries a strictly NEWER flag. If Studio ever stops
    emitting `--flash-attn on` on an unanswered probe, the projector pin becomes
    the oldest thing in the argv and the fail-open is no longer free.
    """
    backend, gguf = _tight_vision(tmp_path, monkeypatch, _caps(answered = False, supports = False))

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert "--no-mmproj-offload" in cmd
    i = cmd.index("--flash-attn")
    assert cmd[i + 1] == "on", "the b6325 value form is what makes the b5178 pin free"


def test_conclusively_unsupported_build_keeps_the_projector_on_the_gpu(tmp_path, monkeypatch):
    """The (b) case is a degradation, not an outage: no flag, projector stays."""
    backend, gguf = _tight_vision(tmp_path, monkeypatch, _caps(answered = True, supports = False))

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert "--no-mmproj-offload" not in cmd
    assert "--mmproj" in cmd
    assert backend.is_vision is True
    assert backend.vision_on_cpu is False
