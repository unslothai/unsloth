# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Both compare components accept both persisted thread shapes.

A compare pair is stored as chat_threads.model_type = model1/model2 by the
generalized compare and base/lora by the LoRA compare. Which component mounts
is decided from async global state (the loaded checkpoint against the LoRA
list), so either component can be handed a pair the other one persisted:
`loras` starts empty, GeneralCompareContent mounts first and hydrates, and if
the checkpoint then turns out to be a LoRA export, React swaps in
LoraCompareContent over the same pair. When that resolver matched only its own
base/lora shape, it set both thread ids to undefined and blanked both panes
one frame after they had painted (#9823).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAT_PAGE = (REPO / "studio/frontend/src/features/chat/chat-page.tsx").read_text(
    encoding = "utf-8"
)


def test_the_lora_compare_resolver_accepts_a_generalized_pair():
    assert '(t) => t.modelType === "base" || t.modelType === "model1"' in CHAT_PAGE
    assert '(t) => t.modelType === "lora" || t.modelType === "model2"' in CHAT_PAGE


def test_the_general_compare_resolver_accepts_a_lora_pair():
    assert '(t) => t.modelType === "model1" || t.modelType === "base"' in CHAT_PAGE
    assert '(t) => t.modelType === "model2" || t.modelType === "lora"' in CHAT_PAGE


def test_no_resolver_matches_a_single_shape_only():
    # A bare single-shape find would reintroduce the blanking for whichever
    # pair the other component persisted.
    for lonely in (
        'threads.find((t) => t.modelType === "lora")',
        'threads.find((t) => t.modelType === "model1")',
        'threads.find((t) => t.modelType === "model2")',
    ):
        assert lonely not in CHAT_PAGE, lonely
