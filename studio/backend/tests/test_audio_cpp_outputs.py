# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stems out of a saved ``/v1/tasks/run`` separation answer, decoded from disk."""

from __future__ import annotations

import base64
import io
import json
import struct
import tracemalloc
import wave

import pytest

from core.inference.audio_cpp_outputs import (
    SeparationOutputError,
    extract_named_outputs,
    riff_info,
)

RATE = 44100


def _wav(
    frames = 4410,
    channels = 2,
    rate = RATE,
    fill = b"\x01\x00",
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(fill * channels * frames)
    return buf.getvalue()


def _float_wav(
    frames,
    channels = 2,
    rate = RATE,
) -> bytes:
    data = b"\x00\x00\x00\x00" * frames * channels
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * channels * 4, channels * 4, 32)
    fmt += struct.pack("<H", 0)  # cbSize: an 18-byte fmt chunk
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _answer(
    tmp_path,
    payload,
    *,
    indent = None,
    raw = None,
):
    path = tmp_path / "response.json"
    if raw is not None:
        path.write_bytes(raw)
    else:
        path.write_text(json.dumps(payload, indent = indent), encoding = "utf-8")
    out = tmp_path / "out"
    out.mkdir(exist_ok = True)
    return path, out


def _stems(ids, frames = 4410):
    return [
        {"id": stem, "audio": _b64(_wav(frames)), "sample_rate": RATE, "channels": 2}
        for stem in ids
    ]


@pytest.mark.parametrize(
    "ids",
    [
        ["drums", "bass", "other", "vocals"],
        ["drums", "bass", "other", "vocals", "guitar", "piano"],
    ],
)
def test_every_named_stem_lands_in_its_own_wav(tmp_path, ids):
    path, out = _answer(
        tmp_path, {"named_audio_outputs": _stems(ids), "timing": {"wall_ms": 365.5}}
    )
    outputs = extract_named_outputs(path, out)
    assert [o["id"] for o in outputs] == ids
    for output in outputs:
        assert set(output) == {"id", "path", "sample_rate", "channels", "duration_s"}
        assert output["path"] == str(out / f"{output['id']}.wav")
        assert (output["sample_rate"], output["channels"], output["duration_s"]) == (RATE, 2, 0.1)
        assert riff_info(out / f"{output['id']}.wav")["frames"] == 4410
        with wave.open(output["path"]) as w:
            assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (RATE, 2, 4410)


def test_key_order_and_pretty_printing_do_not_matter(tmp_path):
    stems = [{"audio": _b64(_wav()), "channels": 2, "id": "vocals", "sample_rate": RATE}]
    path, out = _answer(tmp_path, {"timing": {}, "named_audio_outputs": stems}, indent = 4)
    assert [o["id"] for o in extract_named_outputs(path, out)] == ["vocals"]
    spaced = (
        '{ "named_audio_outputs" :\n [ { "id" : "drums" ,\n "audio"  :  "'
        + _b64(_wav())
        + '" } ] }'
    ).encode()
    path, out = _answer(tmp_path, None, raw = spaced)
    assert [o["id"] for o in extract_named_outputs(path, out)] == ["drums"]


def test_escaped_slashes_inside_base64_decode(tmp_path):
    # A run of 0xff bytes encodes to "////", which a JSON writer may escape as "\/".
    data = _wav(fill = b"\xff\xff")
    encoded = _b64(data)
    assert "/" in encoded
    raw = (
        '{"named_audio_outputs":[{"id":"vocals","audio":"' + encoded.replace("/", "\\/") + '"}]}'
    ).encode()
    path, out = _answer(tmp_path, None, raw = raw)
    (output,) = extract_named_outputs(path, out, chunk = 1001)
    assert (out / "vocals.wav").read_bytes() == data
    assert output["duration_s"] == 0.1


def test_a_top_level_audio_is_not_counted_twice(tmp_path):
    first = _wav()
    payload = {
        "audio": _b64(first),
        "sample_rate": RATE,
        "named_audio_outputs": [{"id": "audio_0", "audio": _b64(first)}],
    }
    path, out = _answer(tmp_path, payload)
    outputs = extract_named_outputs(path, out)
    assert [o["id"] for o in outputs] == ["audio_0"]
    assert sorted(p.name for p in out.iterdir()) == ["audio_0.wav"]


@pytest.mark.parametrize(
    "payload",
    [
        {"audio": _b64(_wav())},
        {"named_audio_outputs": []},
        {"named_audio_outputs": [{"id": "vocals", "audio": _b64(b"not a wav at all")}]},
        {"named_audio_outputs": [{"id": "vocals"}]},
        {"named_audio_outputs": [{"audio": _b64(_wav())}]},
    ],
)
def test_no_decodable_stem_is_an_error(tmp_path, payload):
    path, out = _answer(tmp_path, payload)
    with pytest.raises(SeparationOutputError, match = "returned no stems"):
        extract_named_outputs(path, out)
    assert list(out.iterdir()) == []


def test_an_empty_or_broken_answer_is_an_error(tmp_path):
    for raw in (b"", b"{not json", b'{"named_audio_outputs":[{"id":"x","audio":"QUJD'):
        path, out = _answer(tmp_path, None, raw = raw)
        with pytest.raises(SeparationOutputError):
            extract_named_outputs(path, out)


def test_a_bad_stem_after_a_good_one_removes_both(tmp_path):
    payload = {
        "named_audio_outputs": [
            {"id": "vocals", "audio": _b64(_wav())},
            {"id": "drums", "audio": _b64(b"garbage!")},
        ]
    }
    path, out = _answer(tmp_path, payload)
    with pytest.raises(SeparationOutputError):
        extract_named_outputs(path, out)
    assert list(out.iterdir()) == []


def test_duplicate_and_unsafe_ids_become_safe_unique_names(tmp_path):
    payload = {"named_audio_outputs": _stems(["vocals", "vocals", "../x", "a b/c"])}
    path, out = _answer(tmp_path, payload)
    ids = [o["id"] for o in extract_named_outputs(path, out)]
    assert ids == ["vocals", "vocals_2", "___x", "a_b_c"]
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{i}.wav" for i in ids)
    assert not (tmp_path / "x.wav").exists()


def test_a_float_wav_reports_its_duration(tmp_path):
    payload = {"named_audio_outputs": [{"id": "vocals", "audio": _b64(_float_wav(22050))}]}
    path, out = _answer(tmp_path, payload)
    (output,) = extract_named_outputs(path, out)
    assert (output["sample_rate"], output["channels"], output["duration_s"]) == (RATE, 2, 0.5)


def test_a_data_uri_prefix_is_skipped(tmp_path):
    data = _wav()
    payload = {
        "named_audio_outputs": [{"id": "vocals", "audio": "data:audio/wav;base64," + _b64(data)}]
    }
    path, out = _answer(tmp_path, payload)
    extract_named_outputs(path, out)
    assert (out / "vocals.wav").read_bytes() == data


def test_a_large_answer_decodes_in_bounded_memory(tmp_path):
    # Two ~30 MB base64 stems (~46 MB of audio): never held whole in memory.
    frames = 5_700_000
    stem = _wav(frames, fill = b"\x12\x34")
    encoded = base64.b64encode(stem)
    path = tmp_path / "response.json"
    with open(path, "wb") as f:
        f.write(b'{"named_audio_outputs":[')
        for index, name in enumerate((b"vocals", b"instrumental")):
            if index:
                f.write(b",")
            f.write(b'{"id":"' + name + b'","audio":"')
            f.write(encoded)
            f.write(b'","sample_rate":44100,"channels":2}')
        f.write(b'],"timing":{"wall_ms":1}}')
    assert path.stat().st_size > 60_000_000
    del stem, encoded
    out = tmp_path / "out"
    out.mkdir()
    tracemalloc.start()
    try:
        outputs = extract_named_outputs(path, out)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024 * 1024, peak
    assert [o["id"] for o in outputs] == ["vocals", "instrumental"]
    assert riff_info(out / "vocals.wav")["frames"] == frames
