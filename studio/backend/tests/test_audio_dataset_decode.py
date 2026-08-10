# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio datasets stay readable when torchcodec cannot load its FFmpeg libraries."""

from __future__ import annotations

import io

import pytest

from utils.datasets import audio_decode

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
# The shim needs both: it resamples through librosa, so without it every
# ensure_audio_decoding() below correctly returns False and the tests fail.
pytest.importorskip("librosa")
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
    # encode_example is patched too, so it needs restoring as well: leaving the
    # shim installed made the next test capture it as _ORIGINAL_ENCODE.
    monkeypatch.setattr(Audio, "encode_example", Audio.encode_example)
    monkeypatch.setattr(audio_decode, "_installed", False)
    monkeypatch.setattr(audio_decode, "_ORIGINAL_ENCODE", None)


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
    import sys
    import types

    from datasets import config
    from datasets.features.audio import Audio

    # Stub the decoder the guard probes for, so the assertion holds on hosts
    # that have no torchcodec installed at all rather than a broken one.
    module = sys.modules.get("datasets.features._torchcodec")
    if module is None:
        module = types.ModuleType("datasets.features._torchcodec")
        module.AudioDecoder = object
        monkeypatch.setitem(sys.modules, "datasets.features._torchcodec", module)
    monkeypatch.setattr(config, "TORCHCODEC_AVAILABLE", True)
    monkeypatch.setattr(audio_decode, "_installed", False)
    before = Audio.decode_example
    assert audio_decode.ensure_audio_decoding() is True
    assert Audio.decode_example is before


def test_a_stereo_source_keeps_its_frames(broken_torchcodec):
    """soundfile returns (frames, channels), torchcodec (channels, frames): the wrong
    axis collapsed a clip to one sample per channel and trained on near-silence."""
    from datasets import Audio, Dataset

    buf = io.BytesIO()
    sf.write(buf, np.zeros((800, 2), dtype = "float32"), 16000, format = "WAV")
    audio_decode.ensure_audio_decoding()
    ds = Dataset.from_dict({"audio": [{"path": "s.wav", "bytes": buf.getvalue()}]})
    ds = ds.cast_column("audio", Audio())

    assert len(ds[0]["audio"]["array"]) == 800


def test_a_decoder_that_cannot_resample_reports_unusable(monkeypatch, broken_torchcodec):
    """Every trainer cast names a target rate, so soundfile alone is not enough."""
    import builtins

    real_import = builtins.__import__

    def no_librosa(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("no librosa in no-torch mode")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_librosa)
    assert audio_decode.ensure_audio_decoding() is False


def test_the_dataset_format_check_installs_the_decoder():
    """The preview reads audio rows, so the wiring is the fix, not the module."""
    import inspect

    from hub.services.datasets import formatting

    source = inspect.getsource(formatting.check_format_response)
    assert "ensure_audio_decoding()" in source


def test_the_audio_trainer_paths_install_the_decoder():
    # Read the source rather than import it: this asserts a wiring contract, and
    # importing the trainer drags in the whole torch/unsloth stack for it.
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py"
    ).read_text(encoding = "utf-8")
    assert "ensure_audio_decoding()" in source
    # Guarded so a text-only run never pays for the probe.
    assert "if self._audio_type or self.is_audio_vlm:" in source


def test_a_concurrent_first_install_captures_the_original_encode_once(monkeypatch, broken_torchcodec):
    """Two first-time callers must not both capture Audio.encode_example.

    The loser used to capture the already-installed shim as _ORIGINAL_ENCODE, so
    its fallback branch recursed into itself until RecursionError, breaking
    dataset encoding for the whole process.
    """
    import threading

    from datasets.features.audio import Audio

    from utils.datasets import audio_decode

    original = Audio.encode_example
    monkeypatch.setattr(audio_decode, "_installed", False, raising = False)
    monkeypatch.setattr(audio_decode, "_ORIGINAL_ENCODE", None, raising = False)

    start = threading.Barrier(4)
    errors: list[BaseException] = []

    def install():
        try:
            start.wait(timeout = 10)
            audio_decode.ensure_audio_decoding()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = install) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)

    assert not errors, errors[:2]
    assert audio_decode._ORIGINAL_ENCODE is original
    assert audio_decode._ORIGINAL_ENCODE is not audio_decode._encode_with_soundfile
