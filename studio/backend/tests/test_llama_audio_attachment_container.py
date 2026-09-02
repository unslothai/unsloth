# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import pathlib
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


def test_pyav_output_is_not_held_twice(monkeypatch):
    """np.concatenate keeps the block list and its result live at once, so a
    30-minute upload at the sample ceiling held two 346 MB arrays. Each block is
    written into the output as it arrives, in decode order, and nothing joins a
    list at the end."""
    raw = _encode_amr()
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the PyAV path must not concatenate its blocks")

    monkeypatch.setattr(np, "concatenate", refuse)
    samples, sample_rate = inference_route._decode_audio_mono(raw)
    assert sample_rate == 8_000
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
    # torch is only a source of fake tensors for the fake torchaudio below, so
    # a machine without it should skip rather than fail, as the av cases do.
    torch = pytest.importorskip("torch")

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
    # torch is only a source of fake tensors for the fake torchaudio below, so
    # a machine without it should skip rather than fail, as the av cases do.
    torch = pytest.importorskip("torch")

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
    # torch is only a source of fake tensors for the fake torchaudio below, so
    # a machine without it should skip rather than fail, as the av cases do.
    torch = pytest.importorskip("torch")

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


def _wav_header(
    sample_rate: int,
    channels: int,
    bits: int,
    data_bytes: int,
    byte_rate: int = 0,
    format_tag: int = 0x0001,
    sub_format: int = 0,
) -> bytes:
    block_align = channels * bits // 8
    byte_rate = byte_rate or sample_rate * block_align
    # WAVE_FORMAT_EXTENSIBLE needs its 22-byte extension to name a sub-format.
    extension = b""
    if sub_format:
        extension = (
            (22).to_bytes(2, "little")
            + bits.to_bytes(2, "little")
            + (0).to_bytes(4, "little")
            + sub_format.to_bytes(2, "little")
            + bytes(14)
        )
    fmt_size = 16 + len(extension)
    return (
        b"RIFF"
        + (20 + fmt_size + data_bytes).to_bytes(4, "little")
        + b"WAVEfmt "
        + fmt_size.to_bytes(4, "little")
        + format_tag.to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + bits.to_bytes(2, "little")
        + extension
        + b"data"
        + data_bytes.to_bytes(4, "little")
    )


def _mp3_frames(seconds: float, bitrate_kbps: int = 8) -> bytes:
    """MPEG-2.5 Layer III frames at 8 kHz: 576 samples and 72 bytes each at 8 kbps."""
    version_bits = 0x00 << 3  # MPEG 2.5
    layer_bits = 0x01 << 1  # Layer III
    bitrate_index = (0, 8, 16, 24, 32, 40, 48, 56, 64).index(bitrate_kbps)
    third = (bitrate_index << 4) | (0x02 << 2)  # 8 kHz, no padding
    header = bytes([0xFF, 0xE0 | version_bits | layer_bits | 0x01, third, 0x00])
    length = (576 // 8) * bitrate_kbps * 1000 // 8_000
    frame = header + b"\x00" * (length - 4)
    return frame * max(1, round(seconds / (576 / 8_000)))


def test_a_forwarded_wav_is_bounded_by_its_own_header(monkeypatch):
    """Passthrough returns before every bounded decoder, so the duration cap has
    to come from the header."""
    # A small cap, so the fixture can carry the PCM its header declares rather
    # than allocating the 24 MB that 50 minutes at 8 kHz would really take.
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)
    long_wav = _wav_header(8_000, 1, 8, 8_000 * 2) + b"\x80" * (8_000 * 2)
    try:
        inference_route._prepare_audio_for_llama(base64.b64encode(long_wav).decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected a forwarded wav to be held to the limit")

    # One inside the limit is still forwarded untouched.
    short_wav = _wav_header(8_000, 1, 8, 4_000) + b"\x80" * 4_000
    encoded, container = inference_route._prepare_audio_for_llama(
        base64.b64encode(short_wav).decode()
    )
    assert container == "wav"
    assert base64.b64decode(encoded) == short_wav


def test_a_forwarded_mp3_is_bounded_by_its_frame_headers(monkeypatch):
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 60)
    try:
        inference_route._prepare_audio_for_llama(base64.b64encode(_mp3_frames(120)).decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected a forwarded mp3 to be held to the limit")

    encoded, container = inference_route._prepare_audio_for_llama(
        base64.b64encode(_mp3_frames(10)).decode()
    )
    assert container == "mp3"


def test_the_frame_walk_stops_at_the_cap(monkeypatch):
    """Proving a file too long must not cost the whole walk."""
    hours = _mp3_frames(3 * 60 * 60)
    assert len(hours) > 1_000_000
    seconds = inference_route._mp3_seconds(hours, 60.0)
    assert seconds is not None
    # Just past the cap, not the file's full three hours.
    assert 60.0 < seconds < 61.0


def test_a_container_that_cannot_say_reports_nothing():
    """None means the walk established no length, not that there is none."""
    assert inference_route._wav_seconds(b"RIFF\x00\x00\x00\x00WAVE") is None
    assert inference_route._mp3_seconds(b"not a frame at all", 60.0) is None
    # A tag with no frames behind it reads as no audio, not as a duration.
    assert inference_route._mp3_seconds(_id3_prefix(), 60.0) is None


def _decode_instead_of_forwarding(monkeypatch):
    """Replace the decoder with a marker, so a transcode is visible as one."""
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda _raw: (np.zeros(8_000, dtype = np.float32), 8_000),
    )


def test_a_container_that_cannot_say_is_decoded_rather_than_forwarded(monkeypatch):
    """Forwarding an unreadable header applied the cap only to files honest
    enough to describe themselves, which is the wrong way round. A transcode
    costs a decode and puts the file back under both ceilings."""
    _decode_instead_of_forwarding(monkeypatch)
    unreadable = _mp3_frames(5) + b"\x01\x02\x03\x04"
    assert inference_route._mp3_seconds(unreadable, 60.0) is None
    _encoded, container = inference_route._prepare_audio_for_llama(
        base64.b64encode(unreadable).decode()
    )
    assert container == "wav"


def test_junk_between_frames_cannot_shorten_a_forwarded_mp3(monkeypatch):
    """A decoder resynchronises past junk and plays the rest. Counting only the
    frames before it let four stray bytes present two hours as two seconds."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 60)
    _decode_instead_of_forwarding(monkeypatch)
    one_frame = len(_mp3_frames(576 / 8_000))
    spiked = _mp3_frames(120)
    spiked = spiked[: one_frame * 10] + b"\x00\x00\x00\x00" + spiked[one_frame * 10 :]
    assert inference_route._mp3_seconds(spiked, 60.0) is None
    _encoded, container = inference_route._prepare_audio_for_llama(
        base64.b64encode(spiked).decode()
    )
    assert container == "wav"


def test_a_compressed_wav_is_not_measured_with_pcm_arithmetic(monkeypatch):
    """Only WAVE_FORMAT_PCM fixes nBlockAlign as one sample frame.

    An IMA ADPCM block holds around 505 frames, so rate * blockAlign overstates
    its byte rate 505-fold and made three hours of audio read as twenty-one
    seconds, which is short enough to forward. Nothing here can read a codec's
    own header, so a non-PCM tag reports no duration and is decoded instead.
    """
    rate, block_align, samples_per_block = 8_000, 256, 505
    byte_rate = round(rate / samples_per_block * block_align)
    payload = b"\x00" * (byte_rate * 120)
    adpcm = _wav_header(rate, 1, 4, len(payload), byte_rate = byte_rate, format_tag = 0x11) + payload
    assert inference_route._wav_seconds(adpcm) is None

    _decode_instead_of_forwarding(monkeypatch)
    _encoded, container = inference_route._prepare_audio_for_llama(base64.b64encode(adpcm).decode())
    assert container == "wav"

    # Uncompressed tags keep their header arithmetic, extensible included.
    for tag in (0x0001, 0x0003):
        pcm = _wav_header(8_000, 1, 8, 8_000, format_tag = tag) + b"\x80" * 8_000
        assert inference_route._wav_seconds(pcm) == pytest.approx(1.0, rel = 1e-6)


def test_an_extensible_wav_is_measured_by_its_sub_format():
    """WAVE_FORMAT_EXTENSIBLE carries the real tag in its SubFormat GUID, and a
    file claiming the tag without the extension has not said what it holds."""
    payload = b"\x80" * 8_000
    pcm_sub = _wav_header(8_000, 1, 8, len(payload), format_tag = 0xFFFE, sub_format = 0x0001) + payload
    assert inference_route._wav_seconds(pcm_sub) == pytest.approx(1.0, rel = 1e-6)

    adpcm_sub = (
        _wav_header(8_000, 1, 8, len(payload), format_tag = 0xFFFE, sub_format = 0x0011) + payload
    )
    assert inference_route._wav_seconds(adpcm_sub) is None

    bare = _wav_header(8_000, 1, 8, len(payload), format_tag = 0xFFFE) + payload
    assert inference_route._wav_seconds(bare) is None


def test_the_last_resort_decoder_reads_a_bounded_range(monkeypatch):
    """torchaudio 2.9's load() is load_with_torchcodec, which calls
    get_all_samples() and only then slices to num_frames, so the argument bounds
    the return value and not the allocation. The fallback asks torchcodec for
    the metadata and reads a range instead."""
    # torch is only a source of fake tensors for the fake torchaudio below, so
    # a machine without it should skip rather than fail, as the av cases do.
    torch = pytest.importorskip("torch")

    ranges: list = []

    class _Metadata:
        sample_rate = 16_000
        num_channels = 1
        duration_seconds = 5.0

    class _Samples:
        data = torch.zeros(1, 8_000)

    class _AudioDecoder:
        def __init__(self, _path):
            self.metadata = _Metadata()

        def get_samples_played_in_range(self, start, stop):
            ranges.append((start, stop))
            return _Samples()

    monkeypatch.setitem(sys.modules, "torchcodec", types.SimpleNamespace(decoders = None))
    monkeypatch.setitem(
        sys.modules,
        "torchcodec.decoders",
        types.SimpleNamespace(AudioDecoder = _AudioDecoder),
    )
    waveform, rate = inference_route._decode_audio_with_torchcodec("ignored")
    assert rate == 16_000
    assert waveform.shape[-1] == 8_000
    assert len(ranges) == 1
    start, stop = ranges[0]
    assert start == 0.0
    assert stop <= inference_route._MAX_AUDIO_SECONDS + 1


def test_the_last_resort_decoder_refuses_an_overlong_file(monkeypatch):
    class _Metadata:
        sample_rate = 16_000
        num_channels = 1
        duration_seconds = inference_route._MAX_AUDIO_SECONDS + 60

    class _AudioDecoder:
        def __init__(self, _path):
            self.metadata = _Metadata()

        def get_samples_played_in_range(self, _start, _stop):
            raise AssertionError("an over-limit file must not be decoded")

    monkeypatch.setitem(sys.modules, "torchcodec", types.SimpleNamespace(decoders = None))
    monkeypatch.setitem(
        sys.modules,
        "torchcodec.decoders",
        types.SimpleNamespace(AudioDecoder = _AudioDecoder),
    )
    try:
        inference_route._decode_audio_with_torchcodec("ignored")
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the duration limit to refuse this file")


def test_the_last_resort_decoder_refuses_without_torchcodec(monkeypatch):
    """No bounded reader means no read, not an unbounded one."""
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    try:
        inference_route._decode_audio_with_torchcodec("ignored")
    except RuntimeError as error:
        assert "could not be decoded" in str(error)
    else:
        raise AssertionError("expected a refusal with no bounded reader")


def test_a_free_format_frame_does_not_authorise_a_passthrough():
    """Bitrate index 0 carries no length in the header, so the walk cannot
    establish one and the file has to be bounded by decoding it instead."""
    free = bytes([0xFF, 0xE0 | (0x00 << 3) | (0x01 << 1) | 0x01, 0x00, 0x00])
    assert inference_route._mp3_seconds(free + _mp3_frames(5), 60.0) is None


def test_a_forged_byte_rate_cannot_shorten_a_forwarded_wav(monkeypatch):
    """nAvgBytesPerSec is redundant with the fields around it, so it is the one
    that can be moved alone. Multiplied by ten thousand it made a long recording
    read as a fraction of a second, and a short file is forwarded untouched."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)
    payload = b"\x80" * (8_000 * 2)
    honest = _wav_header(8_000, 1, 8, len(payload)) + payload
    assert inference_route._wav_seconds(honest) > 1

    forged = _wav_header(8_000, 1, 8, len(payload), byte_rate = 8_000 * 10_000) + payload
    assert inference_route._wav_seconds(forged) == inference_route._wav_seconds(honest)
    try:
        inference_route._prepare_audio_for_llama(base64.b64encode(forged).decode())
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("a forged byte rate walked past the duration limit")


def test_an_id3v1_trailer_does_not_hide_the_duration():
    """A trailing tag is metadata; the frames before it still count."""
    tagged = _mp3_frames(10) + b"TAG" + b"\x00" * 125
    seconds = inference_route._mp3_seconds(tagged, 60.0)
    assert seconds is not None
    assert 9.0 < seconds < 11.0


def test_a_switch_preflight_answers_an_overlong_upload_like_the_serving_path(monkeypatch):
    """The same file must not read as 413 when the model is loaded and 400 when a
    swap happens to be running."""
    import asyncio

    from fastapi import HTTPException

    def _too_long(_b64):
        raise inference_route._DecodedAudioTooLongError("decoded audio exceeds the limit")

    monkeypatch.setattr(inference_route, "_prepare_audio_for_llama", _too_long)
    monkeypatch.setattr(inference_route, "_decode_audio_base64", _too_long)
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: True)

    for target_is_gguf in (True, False):
        try:
            asyncio.run(inference_route._preflight_audio_for_switch({"b64": "x"}, target_is_gguf))
        except HTTPException as error:
            assert error.status_code == 413, target_is_gguf
            assert error.detail == inference_route._audio_too_long_detail()
        else:
            raise AssertionError(f"expected a 413 for target_is_gguf={target_is_gguf}")


def test_a_switch_preflight_still_reports_undecodable_audio_as_a_bad_value(monkeypatch):
    """Only the duration case moves; a file that really cannot be read keeps 400."""
    import asyncio

    from fastapi import HTTPException

    def _broken(_b64):
        raise ValueError("not audio at all")

    monkeypatch.setattr(inference_route, "_prepare_audio_for_llama", _broken)
    monkeypatch.setattr(inference_route, "_decode_audio_base64", _broken)
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: True)

    for target_is_gguf in (True, False):
        try:
            asyncio.run(inference_route._preflight_audio_for_switch({"b64": "x"}, target_is_gguf))
        except HTTPException as error:
            assert error.status_code == 400, target_is_gguf
        else:
            raise AssertionError(f"expected a 400 for target_is_gguf={target_is_gguf}")


def test_every_path_names_the_limit_the_same_way(monkeypatch):
    """One message, so a client sees one answer wherever the limit was found."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 90 * 60)
    assert inference_route._audio_too_long_detail() == "Audio is too long (max 90 minutes)."
    # Explicitly UTF-8: read_text() follows the locale encoding, which is cp1252
    # on a stock Windows runner, and the module holds bytes cp1252 cannot decode.
    source = (pathlib.Path(inference_route.__file__)).read_text(encoding = "utf-8")
    assert source.count('f"Audio is too long') == 1


def _old_resample(arr, source_rate, target_rate):
    """The whole-array pass this replaced, kept so the output can be compared."""
    duration = len(arr) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num = len(arr), endpoint = False)
    target_x = np.linspace(0.0, duration, num = target_len, endpoint = False)
    return np.interp(target_x, source_x, arr).astype(np.float32)


@pytest.mark.parametrize(
    ("count", "source_rate", "target_rate"),
    [
        (48_000, 48_000, 16_000),
        (2_500_000, 48_000, 47_999),
        ((1 << 20) + 1, 48_000, 44_100),
        (100_000, 16_000, 48_000),
        (3, 8_000, 4_000),
    ],
)
def test_the_sliced_resampler_returns_what_the_whole_array_pass_did(
    count, source_rate, target_rate
):
    """Slicing is a memory change, not an audio one: same samples, same dtype."""
    arr = np.random.default_rng(11).standard_normal(count).astype(np.float32)
    fitted = inference_route._resample_mono_linear(arr, source_rate, target_rate)
    expected = _old_resample(arr, source_rate, target_rate)
    assert fitted.dtype == expected.dtype
    assert np.array_equal(fitted, expected)


def test_resampling_does_not_allocate_grids_the_length_of_the_recording():
    """The two float64 grids were 8 bytes per input sample each, so a 30-minute
    upload spent more on the coordinates than on the audio."""
    import tracemalloc

    arr = np.zeros(4_000_000, dtype = np.float32)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        fitted = inference_route._resample_mono_linear(arr, 48_000, 16_000)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    # The result is a third of the input; anything past it plus a slice means a
    # grid was materialized whole.
    assert peak < fitted.nbytes + 16 * 1024 * 1024


def test_the_wav_encoder_writes_the_bytes_it_always_wrote():
    """In place beyond the first copy, and the same file out."""
    rng = np.random.default_rng(3)
    for scale in (0.5, 3.0):
        arr = (rng.standard_normal(5_000) * scale).astype(np.float32)
        arr[7], arr[8], arr[9] = np.nan, np.inf, -np.inf
        scrubbed = np.nan_to_num(
            np.asarray(arr, dtype = np.float32).flatten(), posinf = 0.0, neginf = 0.0
        )
        peak = float(np.abs(scrubbed).max())
        if peak > 1.0:
            scrubbed = scrubbed / peak
        expected = (scrubbed * 32767.0).astype(np.int16).tobytes()
        written = inference_route._mono_f32_to_wav_bytes(arr, 16_000)
        assert written[inference_route._WAV_HEADER_BYTES :] == expected
        assert len(written) == inference_route._WAV_HEADER_BYTES + arr.size * 2


def test_the_wav_encoder_leaves_its_caller_s_array_alone():
    """Scaling in place must happen on the copy, not on the decoded audio."""
    arr = np.array([0.5, -2.0, 0.25], dtype = np.float32)
    inference_route._mono_f32_to_wav_bytes(arr, 8_000)
    assert np.array_equal(arr, np.array([0.5, -2.0, 0.25], dtype = np.float32))


class _CountingSoundFile:
    """A libsndfile stand-in whose header count can disagree with what it yields."""

    samplerate = 8_000
    channels = 1
    blocks_yielded = 4
    declared_frames = 4
    block_len = 1_000
    allocations: list = []

    def __init__(self, _source):
        self.frames = type(self).declared_frames * type(self).block_len

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def blocks(self, **_kwargs):
        for index in range(type(self).blocks_yielded):
            yield np.full(type(self).block_len, float(index), dtype = np.float32)


def _decode_with(monkeypatch, **attributes):
    for name, value in attributes.items():
        setattr(_CountingSoundFile, name, value)
    monkeypatch.setitem(
        sys.modules, "soundfile", types.SimpleNamespace(SoundFile = _CountingSoundFile)
    )
    joins = []
    real_concatenate = np.concatenate

    def counted(arrays, *args, **kwargs):
        joins.append(len(list(arrays)) if isinstance(arrays, list) else 1)
        return real_concatenate(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "concatenate", counted)
    arr, rate = inference_route._decode_audio_mono(b"blocks")
    return arr, rate, joins


def test_an_honest_header_decodes_into_one_array(monkeypatch):
    """Holding every block and then joining them copied the whole recording twice."""
    arr, rate, joins = _decode_with(
        monkeypatch, blocks_yielded = 4, declared_frames = 4, block_len = 1_000
    )
    assert rate == 8_000
    assert joins == []
    assert len(arr) == 4_000
    assert np.array_equal(arr[::1_000], np.array([0.0, 1.0, 2.0, 3.0], dtype = np.float32))


def test_a_header_claiming_more_than_it_holds_buys_no_buffer(monkeypatch):
    """A buffer sized from an unread header is how a small file asks for a large
    allocation, so the count is only trusted once the decode has met half of it."""
    sizes = []
    real_empty = np.empty

    def watched(shape, *args, **kwargs):
        if isinstance(shape, int):
            sizes.append(shape)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", watched)
    arr, _rate, _joins = _decode_with(
        monkeypatch, blocks_yielded = 1, declared_frames = 200, block_len = 1_000
    )
    assert len(arr) == 1_000
    assert all(size <= 2_000 for size in sizes), sizes


def test_a_header_that_undercounts_still_returns_every_sample(monkeypatch):
    """The buffer takes what it can hold and the rest joins the ordinary way."""
    arr, _rate, joins = _decode_with(
        monkeypatch, blocks_yielded = 5, declared_frames = 3, block_len = 1_000
    )
    assert len(arr) == 5_000
    assert joins != []
    assert np.array_equal(arr[::1_000], np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype = np.float32))


def test_a_header_past_the_duration_cap_is_not_preallocated(monkeypatch):
    """The cap refuses the audio anyway; it must not buy the array first."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)
    sizes = []
    real_empty = np.empty

    def watched(shape, *args, **kwargs):
        if isinstance(shape, int):
            sizes.append(shape)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", watched)
    try:
        _decode_with(monkeypatch, blocks_yielded = 4, declared_frames = 4, block_len = 3_000)
    except inference_route._DecodedAudioTooLongError:
        pass
    else:
        raise AssertionError("expected the duration cap to refuse this decode")
    assert sizes == [], sizes


def test_a_header_that_undercounts_keeps_the_samples_in_order(monkeypatch):
    """The buffer is closed by the first block that does not fit it.

    Re-testing the fit for every block let a short final block drop back into
    the unused tail and be emitted ahead of the blocks that overflowed before
    it. The length stayed right, so nothing downstream noticed, and what
    reached the model was the same audio with a piece of it moved.
    """

    class _VaryingSoundFile:
        samplerate = 8_000
        channels = 1
        frames = 2_500

        def __init__(self, _source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def blocks(self, **_kwargs):
            at = 0
            for length in (1_000, 1_000, 1_000, 500):
                yield np.arange(at, at + length, dtype = np.float32)
                at += length

    monkeypatch.setitem(
        sys.modules, "soundfile", types.SimpleNamespace(SoundFile = _VaryingSoundFile)
    )
    arr, rate = inference_route._decode_audio_mono_with_soundfile(b"undercounted")
    assert rate == 8_000
    # A ramp in, so a reorder is a descent out.
    assert np.array_equal(arr, np.arange(3_500, dtype = np.float32))


def test_torchaudio_alone_can_still_decode_audio(monkeypatch):
    """Backwards compatibility for an install that predates PyAV.

    Before the bounded readers existed this path called torchaudio.load()
    outright, so torchaudio on its own was enough. torchaudio 2.9 removed
    info(), which is what sends such an install to the streaming chain, and
    that chain needs libsndfile, PyAV or librosa. Losing audio on upgrade is
    not an acceptable way to gain a memory bound, so it falls back to the
    bounded torchcodec reader that 2.9's own load() decodes through.
    """
    # torch is only a source of fake tensors for the fake torchaudio below, so
    # a machine without it should skip rather than fail, as the av cases do.
    torch = pytest.importorskip("torch")

    class _Metadata:
        sample_rate = 16_000
        num_channels = 1
        duration_seconds = 0.5

    class _Samples:
        data = torch.zeros(1, 8_000)

    class _AudioDecoder:
        def __init__(self, _path):
            self.metadata = _Metadata()

        def get_samples_played_in_range(self, _start, _stop):
            return _Samples()

    class _FakeTorchaudio:
        class transforms:
            pass

    monkeypatch.setitem(sys.modules, "torchaudio", _FakeTorchaudio)
    monkeypatch.setitem(sys.modules, "torchcodec", types.SimpleNamespace(decoders = None))
    monkeypatch.setitem(
        sys.modules,
        "torchcodec.decoders",
        types.SimpleNamespace(AudioDecoder = _AudioDecoder),
    )
    for absent in ("soundfile", "av", "librosa"):
        monkeypatch.setitem(sys.modules, absent, None)

    payload = b"\x80" * 8_000
    raw = _wav_header(8_000, 1, 8, len(payload)) + payload
    decoded = inference_route._decode_audio_base64(base64.b64encode(raw).decode())
    assert decoded.shape[0] == 8_000


def test_a_tag_in_the_middle_cannot_shorten_a_concatenated_mp3(monkeypatch):
    """`cat one.mp3 two.mp3 > both.mp3` is how people join MP3s, and it leaves
    the first file's 128-byte ID3v1 tag sitting between the two streams.

    Matching a trailer by its magic alone accepted that tag as the end of the
    recording, so an hour of audio behind it reported the first file's few
    seconds and was forwarded untouched. A trailer now has to account for its
    own length and reach EOF, and one that does not means the walk never read
    the whole file.
    """
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 60)
    joined = _mp3_frames(5) + b"TAG" + bytes(125) + _mp3_frames(300)
    assert inference_route._mp3_seconds(joined, 60.0) is None

    _decode_instead_of_forwarding(monkeypatch)
    _encoded, container = inference_route._prepare_audio_for_llama(
        base64.b64encode(joined).decode()
    )
    assert container == "wav"


def test_a_trailer_that_reaches_eof_still_ends_the_count():
    """The counterpart: real trailers must not cost a needless transcode."""
    frames = _mp3_frames(10)
    ape_size = 64
    ape = (
        b"APETAGEX"
        + (2000).to_bytes(4, "little")
        + ape_size.to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + (0x80000000).to_bytes(4, "little")
        + bytes(8)
        + bytes(ape_size)
    )
    for trailer in (b"", b"TAG" + bytes(125), ape, b"TAG" + bytes(125) + ape):
        seconds = inference_route._mp3_seconds(frames + trailer, 60.0)
        assert seconds is not None and 9.9 < seconds < 10.1, len(trailer)


def test_pyav_writes_into_one_buffer_rather_than_collecting_blocks(monkeypatch):
    """Dropping each block after copying it does not bound anything.

    The blocks are small, so freeing them returns them to the allocator's free
    lists and not to the OS: resident memory for a 30-minute 48 kHz upload
    measured 2.0x the waveform whether the join was np.concatenate or a fill
    loop. Only writing each block into the output as it arrives holds one copy,
    so the decoder must never build a list of the whole decode.
    """
    raw = _encode_amr()
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "librosa", None)
    expected, rate = inference_route._decode_audio_mono(raw)

    # An under-reported duration has to grow the buffer instead of appending to
    # a list, and the samples must survive the growth copies in decode order.
    monkeypatch.setattr(inference_route, "_av_expected_samples", lambda *_args: 1)
    grown, grown_rate = inference_route._decode_audio_mono(raw)
    assert grown_rate == rate
    assert np.array_equal(grown, expected)


def test_a_forged_container_duration_cannot_ask_for_a_huge_buffer(monkeypatch):
    """The duration only sizes the first allocation, so a lie costs a growth
    copy, never a wrong result and never an allocation the decode would refuse."""

    class _Stream:
        duration = None
        time_base = None

    class _Container:
        duration = 10**18
        streams = types.SimpleNamespace(audio = [_Stream()])

    assert inference_route._av_expected_samples(_Container(), 48_000) == (
        inference_route._MAX_DECODED_SAMPLES + 1
    )

    class _Undeclared(_Container):
        duration = 0

    # Nothing declared starts at a minute and grows from there.
    assert inference_route._av_expected_samples(_Undeclared(), 48_000) == 60 * 48_000 + 1
