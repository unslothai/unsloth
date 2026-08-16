# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto placement of the vision projector, and the disable_vision toggle.

The policy: the projector runs once per image, model layers once per token, so
when the projector's VRAM is what pushes layers onto the CPU, Auto moves the
projector to host RAM instead. It never disables vision to save VRAM -- that
would turn a pasted screenshot into a confident text-only answer.

Most of what is worth pinning down here is where the pin must NOT fire, since a
projector wrongly moved to the CPU is a silent 3.6x on every image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuses the placement harness, which also installs the module stubs the backend
# import needs. Import it before anything from core.inference.
from test_llama_cpp_placement import _backend, _launch, _write_gguf  # noqa: E402,F401

from core.inference.llama_cpp import (  # noqa: E402
    _extra_args_set_mmproj_offload,
    _paravirtual_mmproj_pinnable,
)

_PROJECTOR_BYTES = 900 * 1024 * 1024


def _vision_backend(
    tmp_path: Path,
    *,
    memory,
    mmproj_bytes: int = _PROJECTOR_BYTES,
):
    """A backend whose model resolves a projector of a known size."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _pinned(cmd: list) -> bool:
    return "--no-mmproj-offload" in cmd


def _tight_vision_backend(tmp_path: Path, *, free_mib: int, model_bytes: int):
    """A card where the model's fit turns on the projector's bytes.

    The compute-buffer estimate is pinned small on purpose. Against the stub GGUF
    the real estimator returns several gigabytes, which swamps every other term
    and makes nothing fit on any card -- so a pin test built on it would pass
    whatever the policy did.
    """
    backend, gguf = _vision_backend(tmp_path, memory = [(0, free_mib, free_mib + 2_000)])
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * 1024 * 1024
    return backend, gguf


# --------------------------------------------------------------------------
# The pure predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extras, expected",
    [
        (None, False),
        ([], False),
        (["--top-k", "5"], False),
        (["--no-mmproj-offload"], True),
        (["--mmproj-offload"], True),
        (["--top-k", "5", "--mmproj-offload"], True),
    ],
)
def test_user_offload_flag_is_detected(extras, expected):
    assert _extra_args_set_mmproj_offload(extras) is expected


def test_a_prefix_match_is_not_the_flag():
    # --mmproj and --mmproj-url start with the same characters; neither places
    # the projector, so neither may suppress the automatic pin.
    assert _extra_args_set_mmproj_offload(["--mmproj", "/some/path"]) is False
    assert _extra_args_set_mmproj_offload(["--mmproj-url", "http://x"]) is False


def test_unanswered_probe_still_allows_the_pin():
    # A --help probe that failed is not evidence the flag is missing, and every
    # build that can start at all has carried it since b5178. Dropping the pin
    # there would be a self-inflicted outage.
    assert _paravirtual_mmproj_pinnable({"mtp_probe_inconclusive": True}) is True


def test_a_probed_build_without_the_flag_is_not_pinnable():
    # The probe answered and the flag is genuinely absent: emitting it would be
    # an unknown argument, so the pin must degrade to leaving it on the GPU.
    assert _paravirtual_mmproj_pinnable({}) is False


def test_a_probed_build_with_the_flag_is_pinnable():
    assert _paravirtual_mmproj_pinnable({"supports_no_mmproj_offload": True}) is True


# --------------------------------------------------------------------------
# Where the pin must NOT fire
# --------------------------------------------------------------------------


def test_no_pin_when_everything_fits(tmp_path):
    # Tier 1. A card with room to spare keeps the projector on the GPU, where
    # image encoding is 3.6x faster and costs nothing it needs back.
    backend, gguf = _vision_backend(tmp_path, memory = [(0, 40_000, 48_000)])

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert not _pinned(cmd)
    assert "--mmproj" in cmd


def test_no_pin_on_a_cpu_only_box(tmp_path):
    # No GPU enumerated: the projector is already on the CPU and there are no
    # layers to displace, so the flag would be noise. Metal reaches here too.
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 0, model_bytes = 4_500 * 1024 * 1024)
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert not _pinned(cmd)


def test_no_pin_in_manual_placement(tmp_path):
    # Manual means the user owns the offload. Studio does not second-guess it.
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True, gpu_memory_mode = "manual", gpu_layers = 10)["cmd"]

    assert not _pinned(cmd)


# Studio's own count, excluding the user's token. `<= 1` cannot tell the
# --mmproj-offload case apart from Studio adding its own flag exactly once, so
# it passed with the guard deleted.
@pytest.mark.parametrize(
    "spelling, studio_emits", [("--mmproj-offload", 0), ("--no-mmproj-offload", 1)]
)
def test_no_automatic_pin_when_the_user_named_the_placement(tmp_path, spelling, studio_emits):
    # llama.cpp is last-wins and Studio appends its own flags first, so racing
    # the user for the flag would either be silently overridden or fight a
    # deliberate choice.
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True, extra_args = [spelling])["cmd"]

    assert cmd.count("--no-mmproj-offload") == studio_emits
    # The user's own token survives to the argv either way.
    assert cmd.count(spelling) == 1


def test_no_pin_without_a_projector(tmp_path):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 1_000, 48_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert not _pinned(cmd)
    assert "--mmproj" not in cmd


# --------------------------------------------------------------------------
# disable_vision
# --------------------------------------------------------------------------


def test_disable_vision_drops_the_projector_entirely(tmp_path):
    backend, gguf = _vision_backend(tmp_path, memory = [(0, 40_000, 48_000)])

    cmd = _launch(backend, gguf, is_vision = True, disable_vision = True)["cmd"]

    assert "--mmproj" not in cmd
    # Not --no-mmproj either: nothing was resolved, so there is nothing to
    # place, and the runtime is_vision echo is what tells the UI.
    assert not _pinned(cmd)
    assert backend.is_vision is False


def test_disable_vision_defaults_off_and_changes_nothing(tmp_path):
    backend, gguf = _vision_backend(tmp_path, memory = [(0, 40_000, 48_000)])

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert "--mmproj" in cmd
    assert backend._disable_vision is False


def test_disable_vision_reports_itself(tmp_path):
    backend, gguf = _vision_backend(tmp_path, memory = [(0, 40_000, 48_000)])

    _launch(backend, gguf, is_vision = True, disable_vision = True)

    assert backend._disable_vision is True
    assert backend.vision_on_cpu is False


# --------------------------------------------------------------------------
# vision_on_cpu reporting
# --------------------------------------------------------------------------


def test_vision_on_cpu_is_false_without_a_projector(tmp_path):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 40_000, 48_000)])

    _launch(backend, gguf)

    assert backend.vision_on_cpu is False


def test_vision_on_cpu_never_true_while_is_vision_is_false(tmp_path):
    # The property ANDs with is_vision, so a stale pin flag can never report a
    # CPU projector on a text-only load.
    backend, gguf = _vision_backend(tmp_path, memory = [(0, 40_000, 48_000)])
    _launch(backend, gguf, is_vision = True, disable_vision = True)

    backend._vision_on_cpu = True

    assert backend.is_vision is False
    assert backend.vision_on_cpu is False


# --------------------------------------------------------------------------
# Where the pin MUST fire
# --------------------------------------------------------------------------


def test_pins_when_the_projector_would_displace_layers(tmp_path):
    # The model alone nearly fills the card; the projector is what tips it over.
    # Its VRAM is worth more as layers, which run per token rather than per image.
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _pinned(cmd)
    # Vision is kept, not dropped: the projector still loads, on the CPU.
    assert "--mmproj" in cmd
    assert backend.is_vision is True
    assert backend.vision_on_cpu is True


def test_the_flag_is_emitted_exactly_once(tmp_path):
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count("--no-mmproj-offload") == 1


def test_disable_vision_beats_the_pin(tmp_path):
    # Nothing to place when the projector was never resolved.
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True, disable_vision = True)["cmd"]

    assert "--mmproj" not in cmd
    assert not _pinned(cmd)
    assert backend.vision_on_cpu is False


def test_no_pin_when_the_model_cannot_fit_either_way(tmp_path):
    # Deliberately narrow. On a stack that is mostly CPU-resident already, the
    # projector's bytes buy a couple of percent per token, paid for with a silent
    # 3.6x on every image. The pin only fires where it buys full residency.
    backend, gguf = _tight_vision_backend(
        tmp_path, free_mib = 6_000, model_bytes = 60_000 * 1024 * 1024
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert not _pinned(cmd)
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is False
