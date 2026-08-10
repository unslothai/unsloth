# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio datasets stay readable when torchcodec cannot load its FFmpeg libraries."""

from __future__ import annotations

import io

import pytest

from utils.datasets import audio_decode

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
datasets = pytest.importorskip("datasets")


def _wav_bytes(samples = 1600, sampling_rate = 16000):
    buf = io.BytesIO()
    sf.write(
        buf,
        np.linspace(-0.5, 0.5, samples, dtype = "float32"),
        sampling_rate,
        format = "WAV",
    )
    return buf.getvalue()


@pytest.fixture
def broken_torchcodec(monkeypatch):
    """What disable_torchcodec_if_broken leaves behind on a host with no FFmpeg."""
    from datasets import config
    from datasets.features.audio import Audio

    monkeypatch.setattr(config, "TORCHCODEC_AVAILABLE", False)
    monkeypatch.setattr(Audio, "decode_example", Audio.decode_example)
    monkeypatch.setattr(audio_decode, "_installed", False)


def test_a_broken_torchcodec_makes_datasets_refuse_the_column(broken_torchcodec):
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio": [{"path": "a.wav", "bytes": _wav_bytes()}]})
    ds = ds.cast_column("audio", Audio(sampling_rate = 24000))
    with pytest.raises(ImportError, match = "torchcodec"):
        ds[0]["audio"]


def test_the_soundfile_decoder_resamples_to_the_cast_rate(broken_torchcodec):
    from datasets import Audio, Dataset

    assert audio_decode.ensure_audio_decoding() is True
    ds = Dataset.from_dict({"audio": [{"path": "a.wav", "bytes": _wav_bytes()}]})
    ds = ds.cast_column("audio", Audio(sampling_rate = 24000))
    decoded = ds[0]["audio"]

    assert decoded["sampling_rate"] == 24000
    # 1600 samples at 16 kHz is 0.1 s, so 24 kHz gives 2400 back.
    assert len(decoded["array"]) == pytest.approx(2400, abs = 4)
    assert decoded["path"] == "a.wav"


def test_a_stereo_source_is_averaged_to_mono(broken_torchcodec):
    from datasets import Audio, Dataset

    buf = io.BytesIO()
    sf.write(buf, np.zeros((800, 2), dtype = "float32"), 16000, format = "WAV")
    audio_decode.ensure_audio_decoding()
    ds = Dataset.from_dict({"audio": [{"path": "s.wav", "bytes": buf.getvalue()}]})
    ds = ds.cast_column("audio", Audio())

    assert np.asarray(ds[0]["audio"]["array"]).ndim == 1


def test_ensure_audio_decoding_reports_failure_without_soundfile(monkeypatch, broken_torchcodec):
    import builtins

    real_import = builtins.__import__

    def no_soundfile(name, *args, **kwargs):
        if name == "soundfile":
            raise OSError("libsndfile not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_soundfile)
    assert audio_decode.ensure_audio_decoding() is False


def test_a_working_torchcodec_is_left_alone(monkeypatch):
    from datasets import config
    from datasets.features.audio import Audio

    monkeypatch.setattr(config, "TORCHCODEC_AVAILABLE", True)
    monkeypatch.setattr(audio_decode, "_installed", False)
    before = Audio.decode_example
    assert audio_decode.ensure_audio_decoding() is True
    assert Audio.decode_example is before


def test_audio_array_and_rate_reads_a_torchcodec_decoder():
    """The decoder supports subscripting but has no ``.get``, which callers used to use."""

    class _Decoder:
        def __getitem__(self, key):
            return {"array": np.zeros(4, dtype = "float32"), "sampling_rate": 22050}[key]

    array, rate = audio_decode.audio_array_and_rate(_Decoder(), 16000)
    assert rate == 22050 and len(array) == 4
    assert not hasattr(_Decoder(), "get")


@pytest.mark.parametrize(
    "value",
    [None, {}, {"array": None, "sampling_rate": 16000}, object()],
)
def test_audio_array_and_rate_falls_back_on_an_unreadable_cell(value):
    assert audio_decode.audio_array_and_rate(value, 16000) == (None, 16000)
