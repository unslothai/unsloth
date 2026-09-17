# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
"""`datasets` Audio columns keep decoding after a broken torchcodec is disabled (#8642):
soundfile for what libsndfile reads, PyAV's bundled FFmpeg for the rest, no system FFmpeg.
GPU-free; the decoder is installed straight from import_fixes."""

from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
datasets = pytest.importorskip("datasets")
pytest.importorskip(
    "torch"
)  # importing unsloth needs torch; Studio's no-torch venvs skip this file

from unsloth import import_fixes  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_STUDIO_SHIM = _REPO / "studio" / "backend" / "utils" / "datasets" / "audio_decode.py"


def _wav_bytes(samples = 1600, rate = 16000):
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.5, 0.5, samples, dtype = "float32"), rate, format = "WAV")
    return buf.getvalue()


def _m4a_bytes(seconds = 1.0, rate = 22050):
    av = pytest.importorskip("av")
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buf = io.BytesIO()
    try:
        with av.open(buf, "w", format = "mp4") as container:
            stream = container.add_stream("aac", rate = rate)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(tone[np.newaxis, :], format = "flt", layout = "mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this PyAV cannot encode AAC: {exc}")
    return buf.getvalue()


@pytest.fixture
def broken_torchcodec(monkeypatch):
    """What disable_torchcodec_if_broken leaves behind, with the class restored afterwards."""
    from datasets import config
    from datasets.features.audio import Audio

    if not hasattr(config, "TORCHCODEC_AVAILABLE"):
        pytest.skip("datasets < 4 decodes through soundfile itself")
    monkeypatch.setattr(config, "TORCHCODEC_AVAILABLE", False)
    monkeypatch.setattr(Audio, "decode_example", Audio.decode_example)
    monkeypatch.setattr(Audio, "encode_example", Audio.encode_example)
    monkeypatch.setattr(Audio, "_unsloth_audio_fallback", False, raising = False)


def test_a_wav_row_decodes_and_resamples(broken_torchcodec):
    from datasets import Audio, Dataset

    assert import_fixes.patch_datasets_audio_decoding_without_torchcodec() is True
    ds = Dataset.from_dict({"audio": [{"path": "a.wav", "bytes": _wav_bytes()}]})
    ds = ds.cast_column("audio", Audio(sampling_rate = 24000))
    decoded = ds[0]["audio"]
    assert decoded["sampling_rate"] == 24000
    assert len(decoded["array"]) == pytest.approx(2400, abs = 4)
    assert decoded["path"] == "a.wav"


def test_an_m4a_row_decodes_through_pyav(broken_torchcodec):
    from datasets import Audio, Dataset

    raw = _m4a_bytes()
    with pytest.raises(Exception):
        sf.read(io.BytesIO(raw))  # libsndfile cannot, so this row needs the PyAV leg
    import_fixes.patch_datasets_audio_decoding_without_torchcodec()
    ds = Dataset.from_dict({"audio": [{"path": "tone.m4a", "bytes": raw}]})
    ds = ds.cast_column("audio", Audio(sampling_rate = 16000))
    decoded = ds[0]["audio"]
    array = np.asarray(decoded["array"])
    assert decoded["sampling_rate"] == 16000 and array.ndim == 1
    assert 15000 <= len(array) <= 17500
    assert 0.4 <= float(np.abs(array).max()) <= 0.6


@pytest.mark.parametrize("shape", ["absent", "broken"])
def test_resampling_works_without_a_usable_librosa(broken_torchcodec, monkeypatch, tmp_path, shape):
    import sys

    pytest.importorskip("av")
    if shape == "absent":
        monkeypatch.setitem(sys.modules, "librosa", None)  # `import librosa` now raises ImportError
    else:
        # An old librosa beside numpy 2 raises AttributeError at import; that must reach PyAV too.
        (tmp_path / "librosa.py").write_text(
            "raise AttributeError('np.complex was removed')\n", encoding = "utf-8"
        )
        monkeypatch.delitem(sys.modules, "librosa", raising = False)
        monkeypatch.syspath_prepend(str(tmp_path))
    out = import_fixes._audio_resample(np.zeros(1600, dtype = np.float32), 16000, 24000)
    assert len(out) == pytest.approx(2400, abs = 8)


def test_a_working_torchcodec_is_left_alone(monkeypatch):
    from datasets import config
    from datasets.features.audio import Audio

    if not hasattr(config, "TORCHCODEC_AVAILABLE"):
        pytest.skip("datasets < 4")
    monkeypatch.setattr(config, "TORCHCODEC_AVAILABLE", True)
    before = Audio.decode_example
    assert import_fixes.patch_datasets_audio_decoding_without_torchcodec() is False
    assert Audio.decode_example is before


def test_the_disabler_installs_the_decoder():
    # Read the source: importing a real torchcodec would decide this by the host, not the code.
    src = ast.parse((_REPO / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(src)
        if isinstance(n, ast.FunctionDef) and n.name == "disable_torchcodec_if_broken"
    )
    calls = [
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "patch_datasets_audio_decoding_without_torchcodec" in calls


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
    assert import_fixes.patch_datasets_audio_decoding_without_torchcodec() is True
    rows = {"audio": [{"path": "two.m4a", "bytes": raw}]}
    first = Dataset.from_dict(rows).cast_column("audio", Audio())[0]["audio"]
    second = Dataset.from_dict(rows).cast_column("audio", Audio(stream_index = 1))[0]["audio"]
    assert abs(_dominant_hz(first["array"], first["sampling_rate"]) - 440) < 20
    assert abs(_dominant_hz(second["array"], second["sampling_rate"]) - 880) < 20
    with pytest.raises(Exception, match = "stream"):
        import_fixes._audio_read_mono(io.BytesIO(raw), stream_index = 5)


def test_a_wheel_that_raises_anything_at_import_is_disabled(
    broken_torchcodec, monkeypatch, tmp_path
):
    # A damaged torchcodec need not raise ImportError or RuntimeError; the installer already calls
    # every failure "broken", so the library must disable it and seat the fallback the same way.
    import sys

    pkg = tmp_path / "torchcodec"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding = "utf-8")
    (pkg / "decoders.py").write_text("raise AttributeError('damaged wheel')\n", encoding = "utf-8")
    for name in [n for n in sys.modules if n == "torchcodec" or n.startswith("torchcodec.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(import_fixes, "_torchcodec_version_mismatch_hint", lambda: None)
    monkeypatch.setattr(import_fixes, "_torchcodec_provenance_hint", lambda: None)
    import warnings

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        import_fixes.disable_torchcodec_if_broken()
    from datasets.features.audio import Audio

    assert sys.modules["torchcodec"] is None
    assert getattr(Audio, "_unsloth_audio_fallback", False) is True
    # The one report a user gets, now that the installers say nothing: what failed and what decodes.
    said = [str(w.message) for w in caught if "torchcodec is installed but" in str(w.message)]
    assert said and "soundfile and PyAV" in said[0] and "reinstall torchcodec" in said[0]


def test_the_audio_extras_carry_the_fallback_decoders():
    # `unsloth[audio-torch2xx]` must install what decodes when torchcodec cannot load, or the
    # fallback re-raises libsndfile's error on exactly the containers it exists for.
    tomllib = pytest.importorskip("tomllib")
    with open(_REPO / "pyproject.toml", "rb") as fh:
        extras = tomllib.load(fh)["project"]["optional-dependencies"]
    audio = {k: v for k, v in extras.items() if k.startswith("audio-torch")}
    assert audio
    for name, specs in audio.items():
        names = {re.split(r"[ ;<>=!~\[]", spec, 1)[0] for spec in specs}
        assert {"torchcodec", "soundfile", "av"} <= names, name


def test_the_load_failure_is_classified_for_the_warning(monkeypatch):
    # The import-time warning names the cause it can establish: FFmpeg off the loader path,
    # FFmpeg present so the cause is elsewhere, or a failure that never reached libtorchcodec.
    def raised(msg, cls = RuntimeError):
        try:
            raise cls(msg)
        except cls as exc:
            return exc

    ffmpeg_gone = raised(
        "Could not load libtorchcodec. Likely causes: 1. FFmpeg is not properly installed"
    )
    monkeypatch.setattr(import_fixes, "_ffmpeg_on_loader_path", lambda: False)
    assert import_fixes._torchcodec_load_failure(ffmpeg_gone) == "ffmpeg"
    monkeypatch.setattr(import_fixes, "_ffmpeg_on_loader_path", lambda: True)
    assert import_fixes._torchcodec_load_failure(ffmpeg_gone) == "native"
    assert (
        import_fixes._torchcodec_load_failure(
            raised("DLL load failed while importing _core", ImportError)
        )
        == "broken"
    )
    assert set(import_fixes._TORCHCODEC_FALLBACK_NOTES) == {"ffmpeg", "native", "broken"}


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
    assert import_fixes.patch_datasets_audio_decoding_without_torchcodec() is True
    array, rate = import_fixes._audio_read_mono(str(path))
    assert abs(_dominant_hz(array, rate) - 880) < 20
    array, rate = import_fixes._audio_read_mono(str(path), stream_index = 0)
    assert abs(_dominant_hz(array, rate) - 440) < 20


def test_a_channel_first_array_round_trips_through_the_encoder(broken_torchcodec):
    # torchcodec hands decoded audio out as (channels, samples); libsndfile writes (frames, channels).
    # Written as is, a (2, 1600) clip became two frames of 1600 channels, or failed outright.
    from datasets import Audio, Dataset

    assert import_fixes.patch_datasets_audio_decoding_without_torchcodec() is True
    stereo = np.stack([np.linspace(-0.5, 0.5, 1600, dtype = "float32")] * 2)
    assert stereo.shape == (2, 1600)
    ds = Dataset.from_dict({"audio": [{"array": stereo, "sampling_rate": 16000, "path": "s.wav"}]})
    decoded = ds.cast_column("audio", Audio(sampling_rate = 16000))[0]["audio"]
    assert len(decoded["array"]) == 1600 and decoded["sampling_rate"] == 16000


def _normalized(path: Path, name: str, rename: dict) -> str:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.name = "f"
    fn.returns = None  # Studio annotates, the library does not; the bodies are what must match
    for arg in fn.args.args:
        arg.annotation = None
    fn.body = [
        b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
    ]
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in rename:
            n.id = rename[n.id]
    return ast.dump(fn)


@pytest.mark.parametrize(
    ("library", "studio"),
    [("_audio_decode_with_av", "_decode_with_av"), ("_audio_read_mono", "_read_mono")],
)
def test_the_library_and_studio_decoders_do_not_drift(library, studio):
    # Studio's API process never imports unsloth, so it carries its own copy of the decoder.
    if not _STUDIO_SHIM.exists():
        pytest.skip("no studio checkout")
    rename = {"_audio_decode_with_av": "_decode_with_av"}
    assert _normalized(_REPO / "unsloth" / "import_fixes.py", library, rename) == _normalized(
        _STUDIO_SHIM, studio, {}
    )
