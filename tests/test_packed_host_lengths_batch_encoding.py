# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The host packed-lengths carry must also work for a BatchEncoding, whose .to() moves the batch in place."""

import pathlib

import pytest
import torch

accelerate_data_loader = pytest.importorskip("accelerate.data_loader")
from transformers.tokenization_utils_base import BatchEncoding

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_batch_encoding_moved_in_place_carries_host_lengths(monkeypatch):
    import unsloth.utils.packing as packing

    assert (
        pathlib.Path(packing.__file__).resolve()
        == (ROOT / "unsloth" / "utils" / "packing.py").resolve()
    )
    original = getattr(
        accelerate_data_loader.send_to_device, "__wrapped__", accelerate_data_loader.send_to_device
    )
    monkeypatch.setattr(accelerate_data_loader, "send_to_device", original)
    packing._install_host_lengths_carry()
    batch = BatchEncoding(
        {
            "input_ids": torch.ones(1, 9, dtype = torch.long),
            "packed_seq_lengths": torch.tensor([3, 2, 4], dtype = torch.int32),
        }
    )
    moved = accelerate_data_loader.send_to_device(batch, torch.device("meta"))
    assert moved is batch  # BatchEncoding.to() is in place
    lengths = moved["packed_seq_lengths"]
    assert lengths.device.type == "meta"
    assert getattr(lengths, "_unsloth_lengths_tuple", None) == (3, 2, 4)
