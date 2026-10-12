# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""create_stopping_criteria must not assume CUDA.

Before the fix these failed with a cuda/cpu device mismatch, and at construction on a
non-CUDA build.
"""

import types

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch

from unsloth.chat_templates import create_stopping_criteria


class _FakeTokenizer:
    """Just enough surface for create_stopping_criteria; no network, no files."""

    def __init__(self, eos_token_id = 2):
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        texts,
        add_special_tokens = False,
        return_tensors = "pt",
    ):
        # "\n" + stop_word -> [newline, stop, stop]; the caller drops the newline.
        return types.SimpleNamespace(input_ids = torch.tensor([[13, 100, 101]]))


def test_eos_criteria_builds_and_runs_on_cpu():
    criteria = create_stopping_criteria(_FakeTokenizer(eos_token_id = 2))

    assert criteria[0].stop_token.device.type == "cpu"
    assert criteria[0](torch.tensor([9, 9, 2]), None) is True
    assert criteria[0](torch.tensor([9, 9, 3]), None) is False


def test_multi_token_criteria_builds_and_runs_on_cpu():
    criteria = create_stopping_criteria(_FakeTokenizer(), stop_word = "<|stop|>")

    assert criteria[0].length == 2
    assert criteria[0].single_match is False
    assert criteria[0].stop_token.device.type == "cpu"
    assert criteria[0](torch.tensor([7, 100, 101]), None) is True
    assert criteria[0](torch.tensor([100, 101, 7]), None) is False


# Neither torch.cuda.is_available(), which the spoof patches True process-wide, nor
# has_real_accelerator(), which is true on an XPU-only host: the body names cuda three times.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device")
def test_stop_token_follows_the_input_device():
    criteria = create_stopping_criteria(_FakeTokenizer(eos_token_id = 2))

    assert criteria[0](torch.tensor([9, 9, 2], device = "cuda"), None) is True
    assert criteria[0].stop_token.device.type == "cuda"
    pointer = criteria[0].stop_token.data_ptr()

    # cached, so generation does not pay a host-to-device copy per token
    assert criteria[0](torch.tensor([9, 9, 2], device = "cuda"), None) is True
    assert criteria[0].stop_token.data_ptr() == pointer

    assert criteria[0](torch.tensor([9, 9, 2]), None) is True
    assert criteria[0].stop_token.device.type == "cpu"


@pytest.mark.parametrize(
    "stop_word, rows, expected",
    [
        ("eos_token", [[7, 8, 2], [7, 8, 3]], [True, False]),
        ("eos_token", [[7, 8, 3], [7, 8, 2]], [False, True]),
        ("<|stop|>", [[7, 100, 101], [7, 100, 8]], [True, False]),
        ("<|stop|>", [[7, 100, 8], [7, 100, 101]], [False, True]),
        ("<|stop|>", [[100], [101]], [False, False]),
    ],
)
def test_stopping_criteria_matches_each_sequence(stop_word, rows, expected):
    criteria = create_stopping_criteria(_FakeTokenizer(), stop_word = stop_word)
    # Exercise Transformers' public aggregation, as generate() does.
    result = criteria(torch.tensor(rows), scores = None)
    torch.testing.assert_close(result, torch.tensor(expected))
