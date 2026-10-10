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


def _wav(frames = 4410, fill = b"\x01\x00") -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(fill * 2 * frames)
    return buf.getvalue()


def _float_wav(frames, channels = 2) -> bytes:
    data = b"\x00\x00\x00\x00" * frames * channels
    fmt = struct.pack("<HHIIHH", 3, channels, RATE, RATE * channels * 4, channels * 4, 32)
    fmt += struct.pack("<H", 0)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _answer(tmp_path, payload):
    """``payload`` is raw bytes or a JSON-able answer; returns (answer path, out dir)."""
    path = tmp_path / "response.json"
    path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    out = tmp_path / "out"
    out.mkdir(exist_ok = True)
    return path, out


def _stems(ids):
    return [{"id": i, "audio": _b64(_wav()), "sample_rate": RATE, "channels": 2} for i in ids]


def test_stem_ids_differing_only_by_case_get_their_own_files(tmp_path):
    path, out = _answer(tmp_path, {"named_audio_outputs": _stems(["Vocals", "vocals", "VOCALS"])})
    outputs = extract_named_outputs(path, out)
    assert [o["id"] for o in outputs] == ["Vocals", "vocals_2", "VOCALS_3"]
    assert len({o["path"].lower() for o in outputs}) == 3


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
        with wave.open(output["path"]) as w:
            assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (RATE, 2, 4410)


_FF = _wav(fill = b"\xff\xff")  # base64 holds "////", which a JSON writer may escape as "\/"
_PRETTY = {"timing": {}, "named_audio_outputs": [{"audio": _b64(_wav()), "channels": 2, "id": "v"}]}


@pytest.mark.parametrize(
    "raw, stem, data, duration",
    [
        pytest.param(json.dumps(_PRETTY, indent = 4), "v", _wav(), 0.1, id = "key_order_and_indent"),
        pytest.param(
            '{ "named_audio_outputs" :\n [ { "id" : "drums" ,\n "audio"  :  "'
            + _b64(_wav())
            + '" } ] }',
            "drums",
            _wav(),
            0.1,
            id = "spaced",
        ),
        pytest.param(
            '{"named_audio_outputs":[{"id":"v","audio":"' + _b64(_FF).replace("/", "\\/") + '"}]}',
            "v",
            _FF,
            0.1,
            id = "escaped_slashes",
        ),
        pytest.param(
            json.dumps(
                {
                    "named_audio_outputs": [
                        {"id": "v", "audio": "data:audio/wav;base64," + _b64(_wav())}
                    ]
                }
            ),
            "v",
            _wav(),
            0.1,
            id = "data_uri_prefix",
        ),
        pytest.param(
            json.dumps({"named_audio_outputs": [{"id": "v", "audio": _b64(_float_wav(22050))}]}),
            "v",
            _float_wav(22050),
            0.5,
            id = "float_wav_duration",
        ),
        pytest.param(
            json.dumps(
                {
                    "audio": _b64(_wav()),
                    "named_audio_outputs": [{"id": "audio_0", "audio": _b64(_wav())}],
                }
            ),
            "audio_0",
            _wav(),
            0.1,
            id = "top_level_audio_not_counted_twice",
        ),
    ],
)
def test_answer_shapes_decode_to_one_stem(tmp_path, raw, stem, data, duration):
    path, out = _answer(tmp_path, raw.encode())
    (output,) = extract_named_outputs(path, out, chunk = 1001)
    assert (output["id"], output["sample_rate"], output["channels"]) == (stem, RATE, 2)
    assert output["duration_s"] == duration
    assert [p.name for p in out.iterdir()] == [f"{stem}.wav"]
    assert (out / f"{stem}.wav").read_bytes() == data


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"audio": _b64(_wav())}, "returned no stems"),
        ({"named_audio_outputs": []}, "returned no stems"),
        (
            {"named_audio_outputs": [{"id": "v", "audio": _b64(b"not a wav at all")}]},
            "returned no stems",
        ),
        ({"named_audio_outputs": [{"id": "vocals"}]}, "returned no stems"),
        ({"named_audio_outputs": [{"audio": _b64(_wav())}]}, "returned no stems"),
        (b"", None),
        (b"{not json", None),
        (b'{"named_audio_outputs":[{"id":"x","audio":"QUJD', None),
        pytest.param(
            {
                "named_audio_outputs": [
                    {"id": "v", "audio": _b64(_wav())},
                    {"id": "d", "audio": _b64(b"garbage!")},
                ]
            },
            None,
            id = "a_bad_stem_after_a_good_one_removes_both",
        ),
    ],
)
def test_a_broken_answer_is_an_error_and_leaves_nothing(tmp_path, payload, match):
    path, out = _answer(tmp_path, payload)
    with pytest.raises(SeparationOutputError, match = match):
        extract_named_outputs(path, out)
    assert list(out.iterdir()) == []


def test_duplicate_and_unsafe_ids_become_safe_unique_names(tmp_path):
    payload = {"named_audio_outputs": _stems(["vocals", "vocals", "../x", "a b/c"])}
    path, out = _answer(tmp_path, payload)
    ids = [o["id"] for o in extract_named_outputs(path, out)]
    assert ids == ["vocals", "vocals_2", "___x", "a_b_c"]
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{i}.wav" for i in ids)
    assert not (tmp_path / "x.wav").exists()


def test_a_large_answer_decodes_in_bounded_memory(tmp_path):
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
