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


def _m4a_bytes(seconds = 1.0, sampling_rate = 22050):
    """An AAC file in an MP4 container: what torchcodec decoded and libsndfile cannot."""
    av = pytest.importorskip("av")
    t = np.arange(int(seconds * sampling_rate)) / sampling_rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buf = io.BytesIO()
    try:
        with av.open(buf, "w", format = "mp4") as container:
            stream = container.add_stream("aac", rate = sampling_rate)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(tone[np.newaxis, :], format = "flt", layout = "mono")
            frame.sample_rate = sampling_rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this PyAV cannot encode AAC: {exc}")
    return buf.getvalue()


def test_an_m4a_row_decodes_through_pyav_when_torchcodec_is_broken(broken_torchcodec):
    # soundfile refuses the container, so before this the row raised on every host whose
    # torchcodec cannot load, and the installer told people to install FFmpeg for a
    # format PyAV's bundled FFmpeg already reads.
    from datasets import Audio, Dataset

    raw = _m4a_bytes()
    with pytest.raises(Exception):
        sf.read(io.BytesIO(raw))
    assert audio_decode.ensure_audio_decoding() is True
    ds = Dataset.from_dict({"audio": [{"path": "tone.m4a", "bytes": raw}]})
    ds = ds.cast_column("audio", Audio(sampling_rate = 16000))
    decoded = ds[0]["audio"]

    assert decoded["sampling_rate"] == 16000
    array = np.asarray(decoded["array"])
    assert array.ndim == 1 and array.dtype == np.float32
    # One second of a 440 Hz tone at 0.5 amplitude, resampled to 16 kHz; AAC adds priming
    # samples and the encoder rounds to frame boundaries, so bound rather than pin.
    assert 15000 <= len(array) <= 17500
    assert 0.4 <= float(np.abs(array).max()) <= 0.6
    assert decoded["path"] == "tone.m4a"


def _two_stream_m4a_bytes(rate = 22050, freqs = (440, 880)):
    """One MP4 holding two AAC tracks, a tone per stream; the second is the one a stream_index must reach."""
    av = pytest.importorskip("av")
    t = np.arange(rate) / rate
    buf = io.BytesIO()
    try:
        with av.open(buf, "w", format = "mp4") as container:
            streams = []
            for hz in freqs:
                stream = container.add_stream("aac", rate = rate)
                stream.layout = "mono"
                streams.append(stream)
            for stream, hz in zip(streams, freqs):
                tone = (0.5 * np.sin(2 * np.pi * hz * t)).astype("float32")
                frame = av.AudioFrame.from_ndarray(tone[np.newaxis, :], format = "flt", layout = "mono")
                frame.sample_rate = rate
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this PyAV cannot encode AAC: {exc}")
    return buf.getvalue()


def _dominant_hz(array, rate):
    spectrum = np.abs(np.fft.rfft(np.asarray(array, dtype = "float64")))
    return float(np.fft.rfftfreq(len(array), 1.0 / rate)[int(np.argmax(spectrum))])


def test_the_stream_index_selects_the_track(broken_torchcodec):
    # datasets.Audio(stream_index=1) must reach the second track, as torchcodec would.
    from datasets import Audio, Dataset

    raw = _two_stream_m4a_bytes()
    assert audio_decode.ensure_audio_decoding() is True
    rows = {"audio": [{"path": "two.m4a", "bytes": raw}]}
    first = Dataset.from_dict(rows).cast_column("audio", Audio())[0]["audio"]
    second = Dataset.from_dict(rows).cast_column("audio", Audio(stream_index = 1))[0]["audio"]
    assert abs(_dominant_hz(first["array"], first["sampling_rate"]) - 440) < 20
    assert abs(_dominant_hz(second["array"], second["sampling_rate"]) - 880) < 20
    with pytest.raises(Exception, match = "stream"):
        audio_decode._read_mono(io.BytesIO(raw), stream_index = 5)


def test_a_missing_index_picks_the_default_track_not_the_first(broken_torchcodec, tmp_path):
    # torchcodec resolves stream_index=None through av_find_best_stream, where a default
    # disposition beats position; a file whose second track is the default must decode it.
    import shutil
    import subprocess

    av = pytest.importorskip("av")
    if shutil.which("ffmpeg") is None or not hasattr(av.container.streams.StreamContainer, "best"):
        pytest.skip("needs the ffmpeg CLI to set dispositions and PyAV >= 13 for streams.best")
    path = tmp_path / "second_is_default.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1:sample_rate=22050",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1:sample_rate=22050",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "aac",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            str(path),
        ],
        check = True,
    )
    assert audio_decode.ensure_audio_decoding() is True
    array, rate = audio_decode._read_mono(str(path))
    assert abs(_dominant_hz(array, rate) - 880) < 20
    array, rate = audio_decode._read_mono(str(path), stream_index = 0)
    assert abs(_dominant_hz(array, rate) - 440) < 20


def test_a_channel_first_array_round_trips_through_the_encoder(broken_torchcodec):
    # torchcodec hands decoded audio out as (channels, samples); libsndfile writes (frames, channels).
    # Written as is, a (2, 1600) clip became two frames of 1600 channels, or failed outright.
    from datasets import Audio, Dataset

    assert audio_decode.ensure_audio_decoding() is True
    stereo = np.stack([np.linspace(-0.5, 0.5, 1600, dtype = "float32")] * 2)
    assert stereo.shape == (2, 1600)
    ds = Dataset.from_dict({"audio": [{"array": stereo, "sampling_rate": 16000, "path": "s.wav"}]})
    decoded = ds.cast_column("audio", Audio(sampling_rate = 16000))[0]["audio"]
    assert len(decoded["array"]) == 1600 and decoded["sampling_rate"] == 16000


def test_an_m4a_path_decodes_through_pyav(broken_torchcodec, tmp_path):
    # The path form goes to av.open as a filename, the bytes form as a buffer.
    from datasets import Audio, Dataset

    path = tmp_path / "tone.m4a"
    path.write_bytes(_m4a_bytes())
    audio_decode.ensure_audio_decoding()
    ds = Dataset.from_dict({"audio": [str(path)]}).cast_column("audio", Audio())
    decoded = ds[0]["audio"]
    assert decoded["sampling_rate"] == 22050
    assert 20000 <= len(np.asarray(decoded["array"])) <= 24000


def test_an_undecodable_row_names_both_decoders(broken_torchcodec):
    from datasets import Audio, Dataset

    audio_decode.ensure_audio_decoding()
    ds = Dataset.from_dict({"audio": [{"path": "junk.bin", "bytes": b"not audio at all"}]})
    ds = ds.cast_column("audio", Audio())
    with pytest.raises(RuntimeError, match = "soundfile .* PyAV"):
        ds[0]["audio"]


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

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
        encoding = "utf-8"
    )
    assert "ensure_audio_decoding()" in source
    # Guarded so a text-only run never pays for the probe.
    assert "if self._audio_type or self.is_audio_vlm:" in source


def test_a_concurrent_first_install_captures_the_original_encode_once(
    monkeypatch, broken_torchcodec
):
    """Two first-time callers must not both capture Audio.encode_example.

    The loser captured the already-installed shim as _ORIGINAL_ENCODE, so its fallback
    branch recursed into itself until RecursionError.
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


def test_a_multi_repo_mapping_picks_the_token_of_the_source_repo():
    """Interleaved or concatenated streaming splits carry one token per source repo.

    Handing an arbitrary one to xopen sends a private repo's credential to a different
    repo's host, so the repo id has to come from the URL being opened.
    """
    from datasets import config

    tokens = {"org/first": "token-first", "org/second": "token-second"}
    url = "hf://datasets/org/second@main/data/train-00000.wav"
    assert audio_decode._token_for_url(url, tokens) == "token-second"

    resolve = f"{config.HF_ENDPOINT}/datasets/org/first/resolve/main/data/train-00000.wav"
    assert audio_decode._token_for_url(resolve, tokens) == "token-first"


def test_a_chained_url_is_keyed_on_the_repo_it_actually_fetches():
    """Compressed streaming shards arrive as "zip://inner::https://outer" chains."""
    from datasets import config

    tokens = {"org/first": "token-first", "org/second": "token-second"}
    outer = f"{config.HF_ENDPOINT}/datasets/org/second/resolve/main/audio.zip"
    assert audio_decode._token_for_url(f"zip://clip.wav::{outer}", tokens) == "token-second"


def test_an_unknown_host_gets_no_token_when_the_mapping_is_ambiguous():
    tokens = {"org/first": "token-first", "org/second": "token-second"}
    assert audio_decode._token_for_url("https://example.com/clip.wav", tokens) is None
    # The single-repo mapping every caller in this codebase passes still works, which is
    # what the previous next(iter(...)) did for all of them.
    assert audio_decode._token_for_url("https://example.com/clip.wav", {"org/x": "t"}) == "t"
    assert audio_decode._token_for_url("https://example.com/clip.wav", {}) is None
    assert audio_decode._token_for_url("https://example.com/clip.wav", None) is None


def test_a_repo_absent_from_the_mapping_sends_no_credential():
    """A public repo mixed in with private ones must not borrow their token."""
    tokens = {"org/private": "token-private"}
    url = "hf://datasets/org/public@main/data/clip.wav"
    assert audio_decode._token_for_url(url, tokens) is None
