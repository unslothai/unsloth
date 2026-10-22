# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ninth-round cells for the DSpark/DFlash sidecar reserve.

A draft pinned to any card of a Studio-chosen Vulkan list is colocated with the target (only
--split-mode none narrows it to the main card), a draft missing either tensor borrows it, and a
draft with its own head keeps its logits graph on the card its own split gives that head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_FLOOR = 1024 + 8 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)


@pytest.mark.parametrize(
    "extras, pin, reserved",
    [
        ([], "Vulkan1", True),
        ([], "Vulkan0", True),
        (["--split-mode", "none"], "Vulkan1", False),
        (["--split-mode", "none"], "Vulkan0", True),
    ],
)
def test_a_pin_on_any_studio_chosen_vulkan_card_is_on_the_target(tmp_path, extras, pin, reserved):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO, vulkan = True)
    cmd = _r2._placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_ids = [0, 1],
        extra_args = [*extras, "--device-draft", pin],
    )["cmd"]
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


def _draft(monkeypatch, tmp_path, *, tensors):
    path = tmp_path / "draft.gguf"
    path.write_bytes(b"draft")
    import utils.models.gguf_metadata as gm
    import utils.models.model_config as mc

    monkeypatch.setattr(mc, "colocated_split_shards", lambda p: ([p], True))
    monkeypatch.setattr(gm, "_gguf_shard_has_named_tensor", lambda shard, name: name in tensors)
    return str(path)


@pytest.mark.parametrize(
    "tensors, borrows, borrows_output",
    [
        # Upstream DFlash borrows the target's head when a draft ships none, and no marker
        # tells a build that ties it to the draft's embeddings apart, so it stays borrowed.
        ({"token_embd.weight"}, True, True),
        ({"output.weight"}, True, False),
        (set(), True, True),
        ({"token_embd.weight", "output.weight"}, False, False),
    ],
)
def test_a_draft_missing_a_tensor_borrows_it(
    monkeypatch, tmp_path, tensors, borrows, borrows_output
):
    path = _draft(monkeypatch, tmp_path, tensors = tensors)
    assert _lc._sidecar_borrows_target_tensors(path) is borrows
    assert _lc._sidecar_borrows_output(path) is borrows_output


def test_an_unread_draft_borrows_the_output():
    assert _lc._sidecar_borrows_output(None) is True


def test_a_draft_with_its_own_head_keeps_its_graph_on_its_own_output_card(tmp_path, monkeypatch):
    """A 3:2 split puts all three draft blocks on card 0 and the draft's own output layer
    (3 / 4 = 0.75) on card 1. Borrowing the one-block target's head would put the graph on
    card 0 (r7); owning it moves the graph to card 1."""
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(_lc, "_sidecar_borrows_output", lambda path: False)
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    backend._mtp_draft_compute_bytes = lambda *a, **k: 3 * 1024**3
    _r7._three_block_drafter(backend)
    backend._n_layers = 1
    used, out = _r7._margins(
        _r7._dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"])
    )
    assert out >= 1024 + 3 * 1024
