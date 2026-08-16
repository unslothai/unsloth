# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two things the pin policy asserts about itself but does not test.

1. The pin's entire justification is that it buys FULL GPU residency. If it
   fires and the model still lands on --fit, the user paid a 3.6x image encode
   for nothing.
2. ``disable_vision``'s documented contract is "ignored for models that have no
   vision projector". A projector is also what carries AUDIO input.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _launch  # noqa: E402
from test_mmproj_cpu_pin_policy import _tight_vision_backend, _vision_backend  # noqa: E402


def test_the_pin_actually_buys_full_residency(tmp_path):
    """Otherwise the trade the policy comment describes was never made."""
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert "--no-mmproj-offload" in cmd
    # Full offload, not llama.cpp's adaptive --fit: the pin freed exactly the
    # bytes that were keeping layers on the host.
    assert cmd[cmd.index("-ngl") + 1] == "-1"
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_without_the_pin_the_same_card_falls_back_to_fit(tmp_path):
    """The control for the test above: the pin is what changes the placement."""
    backend, gguf = _tight_vision_backend(tmp_path, free_mib = 6_000, model_bytes = 4_500 * 1024 * 1024)

    # The user owning the placement suppresses the automatic pin, leaving the
    # projector charged -- which is exactly the pre-feature planner.
    cmd = _launch(backend, gguf, is_vision = True, extra_args = ["--mmproj-offload"])["cmd"]

    assert "-ngl" not in cmd or cmd[cmd.index("-ngl") + 1] != "-1"
    assert cmd[cmd.index("--fit") + 1] == "on"


def test_disable_vision_is_inert_for_a_model_with_no_projector(tmp_path):
    """H6: the toggle is a no-op on a text-only GGUF.

    Byte-identical argv, and the by-user echo stays False so the client falls
    through to the generic "cannot accept images" rather than pointing at a
    toggle that changed nothing.
    """
    from test_llama_cpp_placement import _backend

    (tmp_path / "on").mkdir(exist_ok = True)
    (tmp_path / "off").mkdir(exist_ok = True)
    on_backend, on_gguf = _backend(tmp_path / "on", vulkan = False, memory = [(0, 40_000, 48_000)])
    off_backend, off_gguf = _backend(tmp_path / "off", vulkan = False, memory = [(0, 40_000, 48_000)])

    on = _launch(on_backend, on_gguf, disable_vision = True)["cmd"]
    off = _launch(off_backend, off_gguf)["cmd"]

    def _scrub(cmd):
        # Both separators, not just POSIX's. The two launches differ only in the
        # tmp directory their GGUF sits in, so on Windows -- where the argv
        # carries ``...\\on\\model.gguf`` against ``...\\off\\model.gguf`` -- a
        # forward-slash-only scrub leaves the paths in and this compares the tmp
        # layout instead of the argv, which is a failure on the runner and
        # invisible here.
        return [
            "<X>" if ("/" in str(a) or "\\" in str(a) or str(a).isdigit()) else a for a in cmd
        ]

    assert _scrub(on) == _scrub(off)
    assert on_backend.vision_disabled_by_user is False
    assert on_backend.vision_on_cpu is False
