# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_RowStream`` reproduces the snapshot streams it was extracted from."""

import random

import pytest

from core.inference.chat_template_helpers import ReasoningChannelNormalizer
from core.inference.mlx_inference import _RowStream, _mlx_stop_cut


MARKERS = ("<|channel|>analysis<|message|>", "<|end|>")


_FRAGMENTS = [
    "hello",
    " world",
    "\n",
    "ST",
    "OP",
    "STOP",
    "<|end",
    "|>",
    "END",
    "EN",
    "�",
    "a",
    "  ",
    "<|channel|>analysis<|message|>",
    "thinking",
    "<|end|>",
    "answer",
    "S",
    "T",
    "O",
    "P",
    "<",
    "|",
    "e",
    "n",
    "d",
]
_BYTE_FRAGMENTS = [
    b"hello",
    b" world",
    b"\n",
    b"ST",
    b"OP",
    b"STOP",
    b"END",
    b"EN",
    b"\xe2\x82",
    b"\xac",
    b"\xc3",
    b"\xa9",
    b"caf\xc3",
    b"\xf0\x9f",
    b"\x99\x82",
    b"a",
    b"  ",
    b"answer",
    b"S",
    b"T",
    b"O",
    b"P",
]
_SEQUENCE_SETS = [(), ("STOP",), ("STOP", "END"), ("<|end|>",), ("ST", "OPQ")]

