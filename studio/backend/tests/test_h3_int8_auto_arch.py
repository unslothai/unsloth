# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video: Auto keeps INT8 attention off on sm89 (L4: the audio moved past the bf16 noise floor) and on the
unmeasured sm86; explicit On, the env switch and the other arches are as before. The load block is checked by source
order, as the other H3 INT8 load tests do."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.inference import video_minimax_h3_attn as A

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("capability", [(8, 9), (8, 6)])
def test_auto_stays_off_on_the_listed_arches_and_says_why(capability):
    why = A.h3_int8_auto_off_reason(capability)
    assert why is not None
    assert f"sm{capability[0]}{capability[1]}" in why
    assert "audio" in why and "int8_attention='on'" in why


@pytest.mark.parametrize("capability", [(8, 0), (12, 0), (9, 0), (10, 0), (7, 5), None])
def test_auto_may_race_elsewhere(capability):
    assert A.h3_int8_auto_off_reason(capability) is None


def test_off_arches_are_the_measured_ones():
    assert A.H3_INT8_AUTO_OFF_ARCHS == frozenset({(8, 6), (8, 9)})


def _h3_block() -> str:
    src = (BACKEND_DIR / "core" / "inference" / "video.py").read_text(encoding = "utf-8")
    start = src.index("h3_attn_backend = select_attention_backend(")
    end = src.index("from .video_minimax_h3_flashinfer import install_h3_flashinfer", start)
    return src[start:end]


def test_h3_load_gates_auto_not_forced_by_arch_and_reports_it():
    block = _h3_block()
    gate = block[
        block.index("h3_int8_arch_off = h3_int8_auto_off_reason(") : block.index(
            "h3_torch_device ="
        )
    ]
    # the card the load runs on
    assert 'cuda_capability(str(getattr(umem_target, "torch_device", None) or device))' in gate
    assert "int8_auto = int8_auto_config and h3_int8_arch_off is None" in gate
    # explicit On (or the env switch 1) is decided before and never looks at the arch
    forced = block[block.index("int8_forced = ") : block.index("int8_auto = (")]
    assert "h3_int8_arch_off" not in forced
    assert (
        "if int8_forced:"
        in block[block.index("if strided_installed and (int8_forced or int8_auto):") :]
    )
    # status: not engaged, with the arch reason (only when every other Auto gate held)
    reason = block[block.index("h3_int8_reason = (") :]
    reason = reason[: reason.index("if h3_int8_modules:")]
    assert "else h3_int8_arch_off\n" in reason
    assert "if int8_auto_config and h3_int8_arch_off is not None" in reason
