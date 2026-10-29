# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""decode_snac copies the codes to the host once and builds the same three SNAC layers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inference.audio_codecs import AudioCodecManager  # noqa: E402


def _reference_layers(generated_ids):
    """The previous per-token .item() unpacking, kept verbatim as the oracle."""
    token_indices = (generated_ids == 128257).nonzero(as_tuple = True)
    if len(token_indices[1]) > 0:
        cropped = generated_ids[:, token_indices[1][-1] + 1 :]
    else:
        cropped = generated_ids
    row = cropped[0]
    row = row[row != 128258]
    row = row[: (len(row) // 7) * 7]
    codes = [t.item() - 128266 for t in row]
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(len(codes) // 7):
        layer_1.append(codes[7 * i])
        layer_2.append(codes[7 * i + 1] - 4096)
        layer_3.append(codes[7 * i + 2] - 8192)
        layer_3.append(codes[7 * i + 3] - 12288)
        layer_2.append(codes[7 * i + 4] - 16384)
        layer_3.append(codes[7 * i + 5] - 20480)
        layer_3.append(codes[7 * i + 6] - 24576)
    return [torch.tensor(layer).unsqueeze(0) for layer in [layer_1, layer_2, layer_3]]


class _RecordingSnac:
    def __init__(self):
        self.codes = None

    def decode(self, codes):
        self.codes = codes
        return torch.zeros(1, 1, 16)


def _frames(rng, n_frames):
    offsets = 128266 + 4096 * torch.arange(7)
    return (torch.randint(0, 4096, (n_frames, 7), generator = rng) + offsets).reshape(-1)


def _cases():
    rng = torch.Generator().manual_seed(0)
    body = _frames(rng, 40)
    m, eos = torch.tensor([128257]), torch.tensor([128258])
    yield "marker", torch.cat([torch.tensor([1, 2, 128257]), body]).unsqueeze(0)
    yield "two_markers_last_wins", torch.cat([m, _frames(rng, 3), m, body]).unsqueeze(0)
    yield "no_marker", body.unsqueeze(0)
    yield (
        "eos_inside_and_tail",
        torch.cat([m, body[:20], eos, body[20:], eos, torch.tensor([5, 6])]).unsqueeze(0),
    )
    yield (
        "out_of_range",
        torch.cat(
            [m, torch.tensor([0, 1, 2, 3, 4, 5, 6, 10**9, -3, 128258, 7, 8, 9, 10])]
        ).unsqueeze(0),
    )
    yield "int32", torch.cat([m, body]).unsqueeze(0).to(torch.int32)
    yield "float32", torch.cat([m, body]).unsqueeze(0).to(torch.float32)
    yield "batch_of_two", torch.stack([torch.cat([m, body]), torch.cat([m, body.flip(0)])])
    yield "non_contiguous", torch.cat([m, body, body]).reshape(1, -1)[:, ::2]


@pytest.mark.parametrize("name,generated", list(_cases()), ids = [c[0] for c in _cases()])
def test_layers_match_the_per_token_path_without_item(name, generated, monkeypatch):
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac()
    calls = []
    real_item = torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "item", lambda self: calls.append(1) or real_item(self))
    _, sr = mgr.decode_snac(generated, "cpu")
    monkeypatch.setattr(torch.Tensor, "item", real_item)

    assert calls == []
    assert sr == 24000
    expected = _reference_layers(generated)
    assert len(mgr._snac_model.codes) == 3
    for got, exp in zip(mgr._snac_model.codes, expected):
        assert got.dtype == exp.dtype and got.device.type == "cpu"
        assert torch.equal(got, exp)


@pytest.mark.parametrize(
    "generated",
    [
        torch.tensor([[128257, 1, 2, 3]]),
        torch.tensor([[128257, 128258, 128258]]),
        torch.tensor([[5, 6, 128257]]),
    ],
)
def test_too_few_codes_raise(generated):
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac()
    with pytest.raises(ValueError, match = "No valid audio codes found after START_OF_SPEECH token"):
        mgr.decode_snac(generated, "cpu")


def test_decode_entry_point_passes_the_same_codes():
    token_ids = [128257] + _frames(torch.Generator().manual_seed(1), 5).tolist()
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac()
    mgr.decode("snac", "cpu", token_ids = token_ids)
    for got, exp in zip(mgr._snac_model.codes, _reference_layers(torch.tensor([token_ids]))):
        assert torch.equal(got, exp) and got.dtype == exp.dtype
