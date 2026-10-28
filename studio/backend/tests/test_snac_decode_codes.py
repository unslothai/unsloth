# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SNAC decode copies the generated codes to the host once and still builds the same three layers and WAV bytes."""

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inference import audio_codecs  # noqa: E402
from core.inference.audio_codecs import AudioCodecManager, _numpy_to_wav_bytes  # noqa: E402


def _reference_layers(generated_ids):
    """The per-token .item() redistribution, kept verbatim as the oracle."""
    token_indices = (generated_ids == 128257).nonzero(as_tuple = True)
    if len(token_indices[1]) > 0:
        cropped = generated_ids[:, token_indices[1][-1] + 1 :]
    else:
        cropped = generated_ids
    row = cropped[0]
    row = row[row != 128258]
    row = row[: (len(row) // 7) * 7]
    if len(row) == 0:
        raise ValueError("No valid audio codes found after START_OF_SPEECH token")
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


def _reference_wav(waveform, sample_rate):
    waveform = waveform.flatten()
    peak = max(abs(waveform.max()), abs(waveform.min()))
    if peak > 1.0:
        waveform = waveform / peak
    pcm = (waveform * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class _RecordingSnac:
    def __init__(self, waveform):
        self.waveform = waveform
        self.codes = None

    def decode(self, codes):
        self.codes = codes
        return self.waveform


def _frames(rng, n_frames):
    offsets = 128266 + 4096 * torch.arange(7)
    return (torch.randint(0, 4096, (n_frames, 7), generator = rng) + offsets).reshape(-1)


def _cases():
    rng = torch.Generator().manual_seed(0)
    body = _frames(rng, 40)
    yield "marker", torch.cat([torch.tensor([1, 2, 128257]), body]).unsqueeze(0)
    yield (
        "two_markers_last_wins",
        torch.cat(
            [torch.tensor([128257]), _frames(rng, 3), torch.tensor([128257]), body]
        ).unsqueeze(0),
    )
    yield "no_marker", body.unsqueeze(0)
    yield (
        "eos_inside_and_tail",
        torch.cat(
            [
                torch.tensor([128257]),
                body[:20],
                torch.tensor([128258]),
                body[20:],
                torch.tensor([128258, 5, 6]),
            ]
        ).unsqueeze(0),
    )
    yield (
        "out_of_range_codes",
        torch.cat(
            [
                torch.tensor([128257]),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 10**9, -3, 128257 + 1, 7, 8, 9, 10]),
            ]
        ).unsqueeze(0),
    )
    yield "int32", torch.cat([torch.tensor([128257]), body]).unsqueeze(0).to(torch.int32)
    yield "float32", torch.cat([torch.tensor([128257]), body]).unsqueeze(0).to(torch.float32)
    yield (
        "batch_of_two",
        torch.stack(
            [
                torch.cat([torch.tensor([128257]), body]),
                torch.cat([torch.tensor([128257]), body.flip(0)]),
            ]
        ),
    )
    yield "non_contiguous", torch.cat([torch.tensor([128257]), body, body]).reshape(1, -1)[:, ::2]


@pytest.mark.parametrize("name,generated", list(_cases()), ids = [c[0] for c in _cases()])
def test_layers_and_wav_match_the_per_token_path(name, generated, monkeypatch):
    waveform = torch.linspace(-1.5, 1.25, 960).reshape(1, 1, -1)
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac(waveform)

    item_calls = []
    real_item = torch.Tensor.item

    def _counting_item(self):
        item_calls.append(1)
        return real_item(self)

    monkeypatch.setattr(torch.Tensor, "item", _counting_item)
    wav, sr = mgr.decode_snac(generated, "cpu")
    monkeypatch.setattr(torch.Tensor, "item", real_item)

    expected = _reference_layers(generated)
    got = mgr._snac_model.codes
    assert len(got) == 3
    for g, e in zip(got, expected):
        assert g.dtype == e.dtype and g.shape == e.shape and g.device.type == "cpu"
        assert torch.equal(g, e)
    assert sr == 24000
    assert wav == _reference_wav(waveform.squeeze().numpy(), 24000)
    # The codes reach the host in one copy, not one .item() per token
    assert item_calls == []


@pytest.mark.parametrize(
    "generated",
    [
        torch.tensor([[128257, 1, 2, 3]]),
        torch.tensor([[128257, 128258, 128258]]),
        torch.tensor([[5, 6, 128257]]),
    ],
)
def test_too_few_codes_raise_the_same_error(generated):
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac(torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match = "No valid audio codes found after START_OF_SPEECH token"):
        mgr.decode_snac(generated, "cpu")
    with pytest.raises(ValueError, match = "No valid audio codes found after START_OF_SPEECH token"):
        _reference_layers(generated)


def test_decode_entry_point_passes_the_same_codes():
    token_ids = [128257] + _frames(torch.Generator().manual_seed(1), 5).tolist()
    mgr = AudioCodecManager()
    mgr._snac_model = _RecordingSnac(torch.zeros(1, 1, 16))
    mgr.decode("snac", "cpu", token_ids = token_ids)
    expected = _reference_layers(torch.tensor([token_ids], dtype = torch.long))
    for g, e in zip(mgr._snac_model.codes, expected):
        assert torch.equal(g, e) and g.dtype == e.dtype


def _wav_inputs():
    rng = np.random.default_rng(0)
    yield "quiet", (rng.standard_normal(4000) * 0.2).astype(np.float32)
    yield "loud", (rng.standard_normal(4000) * 3).astype(np.float32)
    yield "peak_exactly_one", np.array([1.0, -0.5, 0.25], dtype = np.float32)
    yield "neg_peak", np.array([-2.0, 0.5, 1.5], dtype = np.float32)
    yield "float64", rng.standard_normal(500) * 2
    yield "float16", (rng.standard_normal(500) * 2).astype(np.float16)
    yield "three_d", (rng.standard_normal((1, 1, 777)) * 1.2).astype(np.float32)
    yield (
        "two_d_fortran",
        np.asfortranarray((rng.standard_normal((3, 50)) * 1.7).astype(np.float32)),
    )
    yield "non_contiguous", (rng.standard_normal(1000) * 1.3).astype(np.float32)[::3]
    yield "zero_d", np.float32(0.75) * np.ones((), dtype = np.float32)
    yield "nan", np.array([0.1, np.nan, -0.3], dtype = np.float32)
    yield "inf", np.array([0.1, np.inf, -0.3], dtype = np.float32)
    yield "neg_inf", np.array([0.1, -np.inf, 2.0], dtype = np.float32)
    yield "zeros", np.zeros(64, dtype = np.float32)


@pytest.mark.parametrize("name,arr", list(_wav_inputs()), ids = [c[0] for c in _wav_inputs()])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_wav_bytes_identical_and_input_untouched(name, arr):
    before = arr.copy()
    for sr in (16000, 24000):
        assert _numpy_to_wav_bytes(arr, sr) == _reference_wav(arr.copy(), sr)
    np.testing.assert_array_equal(arr, before)


def test_empty_waveform_raises_like_before():
    with pytest.raises(ValueError) as new:
        _numpy_to_wav_bytes(np.zeros(0, dtype = np.float32), 24000)
    with pytest.raises(ValueError) as old:
        _reference_wav(np.zeros(0, dtype = np.float32), 24000)
    assert str(new.value) == str(old.value)


def test_module_still_exposes_the_helpers():
    assert audio_codecs._numpy_to_wav_bytes is _numpy_to_wav_bytes
