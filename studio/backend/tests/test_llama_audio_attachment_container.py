# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import sys
import types

import numpy as np
import pytest
import routes.inference as inference_route


def _id3_prefix() -> bytes:
    # ID3v2.4, no payload and no footer.
    return b"ID3\x04\x00\x00\x00\x00\x00\x00"


def test_audio_container_sniffer_distinguishes_mpeg_layers_and_adts():
    # MPEG-1 Layer III, MPEG-1 Layer II, and ADTS AAC respectively.
    assert inference_route._sniff_audio_container(b"\xff\xfb\x90\x64") == "mp3"
    assert inference_route._sniff_audio_container(b"\xff\xfd\x90\x64") is None
    assert inference_route._sniff_audio_container(b"\xff\xf1\x50\x80") is None


def test_audio_container_sniffer_checks_the_frame_after_id3():
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xfb\x90\x64") == "mp3"
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xfd\x90\x64") is None
    assert inference_route._sniff_audio_container(_id3_prefix() + b"\xff\xf1\x50\x80") is None


def test_aac_and_mp2_are_transcoded_instead_of_forwarded_as_mp3(monkeypatch):
    decoded = []
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda raw: (decoded.append(raw) or object(), 16_000),
    )
    monkeypatch.setattr(
        inference_route,
        "_fit_transcoded_audio_to_wav_cap",
        lambda array, sample_rate: (array, sample_rate),
    )
    monkeypatch.setattr(
        inference_route,
        "_mono_f32_to_wav_bytes",
        lambda _array, _sample_rate: b"transcoded wav",
    )

    for raw in (b"\xff\xfd\x90\x64", b"\xff\xf1\x50\x80"):
        encoded, format_name = inference_route._prepare_audio_for_llama(
            base64.b64encode(raw).decode("ascii")
        )
        assert format_name == "wav"
        assert base64.b64decode(encoded) == b"transcoded wav"

    assert decoded == [b"\xff\xfd\x90\x64", b"\xff\xf1\x50\x80"]


def _encode_wma() -> bytes:
    # PyAV is optional in a backend install, so only the fixtures that need it skip.
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    with av.open(buffer, mode = "w", format = "asf") as output:
        stream = output.add_stream("wmav2", rate = 44_100)
        stream.layout = "stereo"
        stream.bit_rate = 128_000
        indexes = np.arange(8_192)
        mono = np.sin(indexes * 2 * np.pi * 440 / 44_100).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(
            np.stack([mono, mono]),
            format = "fltp",
            layout = "stereo",
        )
        frame.sample_rate = 44_100
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def _encode_amr() -> bytes:
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    with av.open(buffer, mode = "w", format = "amr") as output:
        stream = output.add_stream("amr_nb", rate = 8_000)
        stream.layout = "mono"
        stream.bit_rate = 12_200
        indexes = np.arange(3_200)
        samples = (np.sin(indexes * 2 * np.pi * 440 / 8_000) * 16_000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format = "s16",
            layout = "mono",
        )
        frame.sample_rate = 8_000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def test_wma_and_amr_decode_with_no_librosa(monkeypatch):
    # Match the GGUF-only installation: PyAV is present, librosa is not. Force
    # libsndfile out too so the test proves the PyAV fallback owns both formats.
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)

    for raw, expected_rate in ((_encode_wma(), 44_100), (_encode_amr(), 8_000)):
        samples, sample_rate = inference_route._decode_audio_mono(raw)
        assert sample_rate == expected_rate
        assert samples.ndim == 1
        assert samples.dtype == np.float32
        assert np.max(np.abs(samples)) > 0


def test_pyav_duration_limit_stops_before_concatenating(monkeypatch):
    raw = _encode_amr()
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 0)

    def concatenate_after_limit(*_args, **_kwargs):
        raise AssertionError("duration-limited audio must not be concatenated")

    monkeypatch.setattr(np, "concatenate", concatenate_after_limit)
    try:
        inference_route._decode_audio_mono(raw)
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected PyAV to stop at the decoded-duration limit")


def test_soundfile_duration_limit_stops_before_concatenating(monkeypatch):
    decoded_blocks = []

    class FakeSoundFile:
        samplerate = 4

        def __init__(self, _source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def blocks(self, **_kwargs):
            for block in (np.ones((4, 2), dtype = np.float32), np.ones((1, 2), dtype = np.float32)):
                decoded_blocks.append(len(block))
                yield block

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        types.SimpleNamespace(SoundFile = FakeSoundFile),
    )
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)

    def concatenate_after_limit(*_args, **_kwargs):
        raise AssertionError("duration-limited audio must not be concatenated")

    monkeypatch.setattr(np, "concatenate", concatenate_after_limit)
    try:
        inference_route._decode_audio_mono(b"bounded soundfile input")
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected SoundFile to stop at the decoded-duration limit")

    assert decoded_blocks == [4, 1]


def test_audio_input_decode_is_bounded_by_the_duration_limit(monkeypatch):
    """The non-GGUF audio-input path decodes with torchaudio, which used to
    materialize the whole waveform before any duration check. A low-bitrate AMR
    or WMA under the 25 MB upload cap can still hold hours of PCM."""
    loads: list[dict] = []

    class _FakeTorchaudio:
        @staticmethod
        def info(_path):
            return types.SimpleNamespace(sample_rate = 16_000, num_frames = 16_000 * 60 * 60)

        @staticmethod
        def load(path, **kwargs):
            loads.append(kwargs)
            raise AssertionError("an over-limit file must not be decoded")

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 30 * 60)

    try:
        inference_route._decode_audio_base64(base64.b64encode(b"long recording").decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the decoded-duration limit to be enforced")
    assert loads == []


def test_audio_input_decode_streams_a_container_that_hides_its_length(monkeypatch):
    """A header that does not report num_frames cannot be bounded by num_frames:
    with several channels the read would still outrun the ceiling, and trimming it
    would silently truncate a file that is inside the limit. Stream it instead."""
    streamed: list[bytes] = []

    class _FakeTorchaudio:
        @staticmethod
        def info(_path):
            return types.SimpleNamespace(sample_rate = 16_000, num_frames = 0, num_channels = 2)

        @staticmethod
        def load(*_a, **_k):
            raise AssertionError("an unreported length must not be loaded whole")

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    def _bounded(raw):
        streamed.append(raw)
        return np.zeros(16_000, dtype = np.float32), 16_000

    monkeypatch.setattr(inference_route, "_decode_audio_mono", _bounded)

    out = inference_route._decode_audio_base64(base64.b64encode(b"endless stream").decode())
    assert out.shape == (16_000,)
    assert streamed == [b"endless stream"]


def test_a_wide_container_inside_the_clock_is_streamed_not_refused(monkeypatch):
    """30 minutes of high-rate or multichannel audio is within the advertised
    limit, so it must still decode; it just cannot be held at once."""
    streamed: list[bytes] = []

    class _FakeTorchaudio:
        @staticmethod
        def info(_path):
            # Inside the 30-minute clock, past the ceiling once channels count.
            return types.SimpleNamespace(
                sample_rate = 48_000,
                num_frames = 48_000 * 20 * 60,
                num_channels = 2,
            )

        @staticmethod
        def load(*_a, **_k):
            raise AssertionError("a wide container must not be materialized")

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    def _bounded(raw):
        streamed.append(raw)
        return np.zeros(16_000, dtype = np.float32), 16_000

    monkeypatch.setattr(inference_route, "_decode_audio_mono", _bounded)

    out = inference_route._decode_audio_base64(base64.b64encode(b"wide input").decode())
    assert out.shape == (16_000,)
    assert streamed == [b"wide input"]


def test_audio_input_decode_passes_a_short_recording_through(monkeypatch):
    import torch

    class _FakeTorchaudio:
        transforms = types.SimpleNamespace()

        @staticmethod
        def info(_path):
            return types.SimpleNamespace(sample_rate = 16_000, num_frames = 16_000)

        @staticmethod
        def load(_path, **_kwargs):
            return torch.ones((2, 16_000)), 16_000

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)
    out = inference_route._decode_audio_base64(base64.b64encode(b"short").decode())
    assert out.shape == (16_000,)
    assert out.dtype == np.float32


def test_a_load_is_capped_by_the_sample_ceiling_not_only_by_the_clock(monkeypatch):
    """num_frames is the value being distrusted, so a container that understates
    it must not license a read up to the rate-relative limit: at 192 kHz that is
    four times the sample ceiling."""
    import torch

    loads: list[dict] = []
    ceiling = 64

    class _FakeTorchaudio:
        transforms = types.SimpleNamespace()

        @staticmethod
        def info(_path):
            # Claims 8 frames; the file actually holds far more.
            return types.SimpleNamespace(sample_rate = 192_000, num_frames = 8, num_channels = 2)

        @staticmethod
        def load(_path, **kwargs):
            loads.append(kwargs)
            frames = kwargs["num_frames"]
            return torch.ones((2, frames)), 192_000

    monkeypatch.setattr(inference_route, "_MAX_DECODED_SAMPLES", ceiling)
    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    try:
        inference_route._decode_audio_base64(base64.b64encode(b"understated").decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the sample ceiling to be enforced")
    # One frame past ceiling / channels, not past rate * seconds.
    assert loads == [{"num_frames": ceiling // 2 + 1}]


def test_an_honest_length_is_still_read_in_full(monkeypatch):
    """Capping by both limits must not truncate a file that fits: the read window
    stays at or above what info() reported."""
    import torch

    loads: list[dict] = []

    class _FakeTorchaudio:
        transforms = types.SimpleNamespace()

        @staticmethod
        def info(_path):
            return types.SimpleNamespace(sample_rate = 48_000, num_frames = 48_000 * 60, num_channels = 1)

        @staticmethod
        def load(_path, **kwargs):
            loads.append(kwargs)
            return torch.ones((1, 48_000 * 60)), 48_000

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    def _must_not_stream(_raw):
        raise AssertionError("a file that fits must not be streamed")

    monkeypatch.setattr(inference_route, "_decode_audio_mono", _must_not_stream)

    class _FakeResample:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, waveform):
            return waveform[..., :16_000]

    _FakeTorchaudio.transforms.Resample = _FakeResample
    out = inference_route._decode_audio_base64(base64.b64encode(b"one minute").decode())
    assert out.shape == (16_000,)
    assert loads[0]["num_frames"] >= 48_000 * 60


def test_audio_input_decode_stays_bounded_when_the_probe_fails(monkeypatch):
    """info() failing does not license an unbounded load: a container it cannot
    read is still handed to the bounded decoder rather than materialized whole."""
    bounded: list[bytes] = []

    class _FakeTorchaudio:
        transforms = types.SimpleNamespace()

        @staticmethod
        def info(_path):
            raise RuntimeError("unknown container")

        @staticmethod
        def load(*_a, **_k):
            raise AssertionError("a probe failure must not fall back to a whole load")

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    def _bounded(raw):
        bounded.append(raw)
        return np.zeros(16_000, dtype = np.float32), 16_000

    monkeypatch.setattr(inference_route, "_decode_audio_mono", _bounded)

    out = inference_route._decode_audio_base64(base64.b64encode(b"amr bytes").decode())
    assert out.shape == (16_000,)
    assert bounded == [b"amr bytes"]


def test_the_probe_failure_path_still_enforces_the_cap(monkeypatch):
    """The bounded decoder raises at the cap, and that error is not swallowed."""

    class _FakeTorchaudio:
        @staticmethod
        def info(_path):
            raise RuntimeError("unknown container")

        @staticmethod
        def load(*_a, **_k):
            raise AssertionError("a probe failure must not fall back to a whole load")

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)

    def _too_long(_raw):
        raise inference_route._DecodedAudioTooLongError("decoded audio exceeds the limit")

    monkeypatch.setattr(inference_route, "_decode_audio_mono", _too_long)

    try:
        inference_route._decode_audio_base64(base64.b64encode(b"hours of amr").decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the duration cap to be enforced")


def test_a_high_rate_container_is_capped_by_samples_not_only_by_clock(monkeypatch):
    """The duration cap is rate-relative, so 30 minutes at a high rate retains far
    more memory than the same speech at 16 kHz. The absolute ceiling bounds it."""
    decoded_blocks = []

    class FakeSoundFile:
        samplerate = 192_000
        channels = 1

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def blocks(self, **_kwargs):
            # Well inside the 30-minute clock at this rate, past the sample cap.
            block = np.ones(ceiling + 1, dtype = np.float32)
            decoded_blocks.append(len(block))
            yield block

    # A small ceiling: the check is what is under test, and the production
    # constant would allocate 346 MB on every run of this file.
    ceiling = 64
    monkeypatch.setattr(inference_route, "_MAX_DECODED_SAMPLES", ceiling)
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(SoundFile = FakeSoundFile))

    def concatenate_after_limit(*_args, **_kwargs):
        raise AssertionError("an over-ceiling decode must not be concatenated")

    monkeypatch.setattr(np, "concatenate", concatenate_after_limit)
    try:
        inference_route._decode_audio_mono_with_soundfile(b"high rate input")
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the decoded-sample ceiling to be enforced")
    assert decoded_blocks == [ceiling + 1]


def test_the_ceiling_never_bites_before_the_advertised_duration(monkeypatch):
    """An ordinary rate keeps the full 30 minutes: the ceiling only refuses rates
    above 48 kHz, so the documented limit still means what it says."""
    assert inference_route._MAX_DECODED_SAMPLES >= 48_000 * inference_route._MAX_AUDIO_SECONDS
    for rate in (8_000, 16_000, 44_100, 48_000):
        assert rate * inference_route._MAX_AUDIO_SECONDS <= inference_route._MAX_DECODED_SAMPLES


def test_the_librosa_fallback_reads_only_up_to_the_cap(monkeypatch):
    """audioread would otherwise materialize the whole waveform and let the check
    after it run too late."""
    seen: dict = {}

    class _FakeLibrosa:
        @staticmethod
        def get_samplerate(_path):
            return 192_000

        @staticmethod
        def load(
            _path,
            sr = None,
            mono = True,
            duration = None,
        ):
            seen["duration"] = duration
            return np.zeros(16_000, dtype = np.float32), 192_000

    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", _FakeLibrosa)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono_with_av",
        lambda raw: (_ for _ in ()).throw(RuntimeError("av cannot read this")),
    )

    inference_route._decode_audio_mono(b"container only librosa reads")
    # Bounded by the sample ceiling at this rate, not by the 30-minute clock.
    assert seen["duration"] is not None
    assert seen["duration"] <= inference_route._MAX_DECODED_SAMPLES / 192_000 + 1
    assert seen["duration"] < inference_route._MAX_AUDIO_SECONDS


def test_an_unprobeable_rate_is_refused_rather_than_read_unbounded(monkeypatch):
    """Without a sample rate there is no window that bounds the read: 30 minutes
    is gigabytes at a high rate. Refusing beats loading it to find out."""

    class _FakeLibrosa:
        @staticmethod
        def get_samplerate(_path):
            raise RuntimeError("cannot probe this container")

        @staticmethod
        def load(*_a, **_k):
            raise AssertionError("an unbounded read must not be attempted")

    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", _FakeLibrosa)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono_with_av",
        lambda raw: (_ for _ in ()).throw(RuntimeError("av cannot read this")),
    )

    try:
        inference_route._decode_audio_mono(b"container with no reported rate")
    except RuntimeError as error:
        assert "sample rate" in str(error)
    else:
        raise AssertionError("expected an unprobeable rate to be refused")


def test_the_block_size_counts_channels_not_only_frames(monkeypatch):
    """blocks() sizes its buffer as frames * channels and yields a copy, so a
    frame-only budget let a 255-channel container reach 49 MB twice over from an
    upload capped at 25."""
    sizes = []

    class FakeSoundFile:
        samplerate = 48_000
        channels = 255

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def blocks(self, **kwargs):
            sizes.append(kwargs["blocksize"])
            yield np.zeros((kwargs["blocksize"], self.channels), dtype = np.float32)

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(SoundFile = FakeSoundFile))
    samples, rate = inference_route._decode_audio_mono_with_soundfile(b"many channels")
    assert rate == 48_000
    assert samples.ndim == 1
    block = sizes[0]
    assert block * FakeSoundFile.channels <= inference_route._MAX_DECODE_BLOCK_SAMPLES
    assert block >= 1


def test_ordinary_audio_keeps_its_block_size(monkeypatch):
    """The divisor only bites past about 20 channels, so nothing common moves."""
    sizes = []

    class FakeSoundFile:
        samplerate = 48_000
        channels = 1

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def blocks(self, **kwargs):
            sizes.append(kwargs["blocksize"])
            yield np.zeros(kwargs["blocksize"], dtype = np.float32)

    for channels in (1, 2, 6, 8):
        FakeSoundFile.channels = channels
        monkeypatch.setitem(
            sys.modules, "soundfile", types.SimpleNamespace(SoundFile = FakeSoundFile)
        )
        inference_route._decode_audio_mono_with_soundfile(b"ordinary audio")
    assert sizes == [48_000, 48_000, 48_000, 48_000]
