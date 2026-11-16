# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#4539: S3 audio datasets download beside their manifest and the manifest points at them.
boto3 is faked as in test_s3_dataset.py."""

import csv
import importlib.util
import io
import json
import os
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _BACKEND / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3_dataset = _load("s3_dataset_audio_tests", "core/training/s3_dataset.py")


class _FakePaginator:
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix")
        contents = [{"Key": k} for k in self._keys if prefix is None or k.startswith(prefix)]
        mid = len(contents) // 2
        yield {"Contents": contents[:mid]}
        yield {"Contents": contents[mid:]}


class _FakeS3Client:
    """The house fake, plus per-key content so a manifest can hold real rows."""

    def __init__(
        self,
        keys,
        contents = None,
    ):
        self._keys = list(keys)
        self._contents = contents or {}
        self.downloaded = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self._keys)

    def download_file(self, bucket, key, local_path, **kwargs):
        self.downloaded.append((bucket, key, local_path))
        callback = kwargs.get("Callback")
        if callback is not None:
            callback(1)
        content = self._contents.get(key, f"content-of:{key}")
        if key.lower().endswith(".parquet") and isinstance(content, str):
            content = _parquet_bytes([{"text": content}])
        if isinstance(content, bytes):
            Path(local_path).write_bytes(content)
        else:
            with open(local_path, "w", encoding = "utf-8", newline = "") as f:
                f.write(content)


def _parquet_bytes(rows, schema = None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema = schema), buffer)
    return buffer.getvalue()


def _parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).read().to_pylist()


def _install(
    monkeypatch,
    keys,
    contents = None,
):
    client = _FakeS3Client(keys, contents)
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda cfg: client)
    return client


def _cfg(**overrides):
    base = {
        "bucket": "my-bucket",
        "region": "us-east-1",
        "prefix": "datasets/",
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "secret",
        "use_iam_role": False,
    }
    base.update(overrides)
    return base


def _rows(jsonl_path):
    with open(jsonl_path, encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_audio_files_download_beside_the_manifest_preserving_structure(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        [
            "datasets/metadata.jsonl",
            "datasets/audio/a.wav",
            "datasets/audio/b.mp3",
            "datasets/notes.txt",
        ],
        contents = {"datasets/metadata.jsonl": '{"audio": "audio/a.wav", "text": "hi"}\n'},
    )

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    assert [os.path.basename(f) for f in files] == ["metadata.jsonl"]
    assert (tmp_path / "audio" / "a.wav").exists()
    assert (tmp_path / "audio" / "b.mp3").exists()
    assert not (tmp_path / "notes.txt").exists()


def test_a_manifest_beside_audio_is_not_a_mixed_format_error(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        ["datasets/metadata.csv", "datasets/a.wav", "datasets/b.mp3"],
        contents = {"datasets/metadata.csv": "audio,text\na.wav,hi\n"},
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["metadata.csv"]


def test_two_manifest_formats_beside_audio_is_still_a_mixed_format_error(monkeypatch, tmp_path):
    client = _install(
        monkeypatch,
        ["datasets/metadata.csv", "datasets/extra.parquet", "datasets/a.wav"],
    )
    with pytest.raises(ValueError, match = "mixed dataset formats"):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert client.downloaded == []


def test_audio_with_no_manifest_names_whats_missing(monkeypatch, tmp_path):
    _install(monkeypatch, ["datasets/audio/a.wav", "datasets/audio/b.mp3"])
    with pytest.raises(ValueError, match = "manifest"):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids = ["lf", "crlf"])
def test_s3_json_lines_preserves_unicode_transcript(monkeypatch, tmp_path, suffix, newline):
    rows = [
        {"audio": "audio/a.wav", "text": "first\u0085second\u2028third\u2029fourth"},
        {"audio": "audio/a.wav", "text": "last"},
    ]
    manifest = "".join(json.dumps(row, ensure_ascii = False) + newline for row in rows)
    key = f"datasets/train{suffix}"
    _install(monkeypatch, [key, "datasets/audio/a.wav"], contents = {key: manifest})

    result = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))

    audio_path = tmp_path / "audio" / "a.wav"
    assert audio_path.is_file()
    assert _rows(result.files[0]) == [{**row, "audio": str(audio_path)} for row in rows]


def test_jsonl_audio_references_are_rewritten_to_local_paths(monkeypatch, tmp_path):
    manifest = "\n".join(
        [
            json.dumps({"audio": "audio/a.wav", "text": "one"}),
            json.dumps({"audio": "s3://my-bucket/datasets/audio/b.mp3", "text": "two"}),
            json.dumps({"audio": {"path": "audio/a.wav"}, "text": "three"}),
            json.dumps({"audio": "https://example.com/c.wav", "text": "four"}),
            json.dumps({"audio": [{"path": "audio/a.wav"}, "audio/b.mp3"], "text": "five"}),
        ]
    )
    _install(
        monkeypatch,
        ["datasets/metadata.jsonl", "datasets/audio/a.wav", "datasets/audio/b.mp3"],
        contents = {"datasets/metadata.jsonl": manifest + "\n"},
    )

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    rows = _rows(files[0])

    local_a = str(tmp_path / "audio" / "a.wav")
    local_b = str(tmp_path / "audio" / "b.mp3")
    assert rows[0]["audio"] == local_a
    assert rows[1]["audio"] == local_b
    assert rows[2]["audio"]["path"] == local_a
    assert rows[3]["audio"] == "https://example.com/c.wav"
    assert rows[4]["audio"] == [{"path": local_a}, local_b]
    assert all(os.path.isabs(p) for p in (rows[0]["audio"], rows[1]["audio"]))
    assert rows[0]["text"] == "one"


def test_json_lines_in_a_json_manifest_are_rewritten(monkeypatch, tmp_path):
    manifest = "".join(
        json.dumps({"audio": name, "text": "hi"}) + "\n" for name in ("a.wav", "b.wav")
    )
    _install(
        monkeypatch,
        ["datasets/train.json", "datasets/a.wav", "datasets/b.wav"],
        contents = {"datasets/train.json": manifest},
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert [row["audio"] for row in _rows(files[0])] == [
        str(tmp_path / "a.wav"),
        str(tmp_path / "b.wav"),
    ]


@pytest.mark.parametrize("layout", ["object", "jsonl", "array"])
@pytest.mark.parametrize("audio_value", ["a.wav", {"path": "a.wav", "bytes": None}])
def test_single_record_manifest_rewrites_audio(monkeypatch, tmp_path, layout, audio_value):
    row = {"audio": audio_value, "text": "hello", "speaker": "speaker-1"}
    suffix = ".jsonl" if layout == "jsonl" else ".json"
    key = "datasets/train" + suffix
    payload = [row] if layout == "array" else row
    _install(
        monkeypatch,
        [key, "datasets/a.wav"],
        contents = {key: json.dumps(payload) + "\n"},
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    loaded = json.loads(Path(files[0]).read_text(encoding = "utf-8"))
    result = loaded[0] if layout == "array" else loaded
    local_audio = str(tmp_path / "a.wav")
    expected_audio = (
        {"path": local_audio, "bytes": None} if isinstance(audio_value, dict) else local_audio
    )
    assert result == {**row, "audio": expected_audio}
    assert Path(local_audio).is_file()


def test_json_manifest_non_record_layout_is_unchanged(tmp_path):
    # A column-oriented object of indexed columns is not a row of audio paths.
    payload = {"audio": {"0": "a.wav"}, "text": {"0": "hello"}}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(payload), encoding = "utf-8")
    s3_dataset._rewrite_json_manifest(
        str(path), lambda value: "/local/a.wav" if value == "a.wav" else None
    )
    assert json.loads(path.read_text(encoding = "utf-8")) == payload


def test_manifest_relative_references_resolve_against_the_manifest_dir(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        ["datasets/train/metadata.jsonl", "datasets/train/clips/a.wav"],
        contents = {
            "datasets/train/metadata.jsonl": json.dumps({"audio": "clips/a.wav", "text": "hi"})
            + "\n"
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    rows = _rows(files[0])
    assert rows[0]["audio"] == str(tmp_path / "train" / "clips" / "a.wav")


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".csv"])
@pytest.mark.parametrize(
    "reference,root_audio,nested_audio,expected",
    [
        ("clips/a.wav", True, True, None),
        ("./clips/a.wav", True, True, "train/clips/a.wav"),
        ("s3://my-bucket/datasets/train/clips/a.wav", True, True, "train/clips/a.wav"),
        ("s3://my-bucket/datasets/clips/a.wav", True, True, "clips/a.wav"),
        ("clips/a.wav", False, True, "train/clips/a.wav"),
        ("clips/a.wav", True, False, "clips/a.wav"),
        ("missing.wav", True, True, "missing.wav"),
    ],
)
def test_audio_reference_ambiguity(
    monkeypatch, tmp_path, suffix, reference, root_audio, nested_audio, expected
):
    manifest_key = "datasets/train/rows" + suffix
    row = {"audio": reference, "text": "transcript for selected clip"}
    if suffix == ".csv":
        content = f"audio,text\n{reference},{row['text']}\n"
    else:
        content = json.dumps([row] if suffix == ".json" else row) + "\n"
    keys = [manifest_key]
    if root_audio:
        keys.append("datasets/clips/a.wav")
    if nested_audio:
        keys.append("datasets/train/clips/a.wav")
    client = _install(monkeypatch, keys, contents = {manifest_key: content})

    if expected is None:
        with pytest.raises(ValueError, match = "Ambiguous audio reference") as error:
            s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
        assert reference in str(error.value)
        assert manifest_key in str(error.value)
        assert "./" in str(error.value) and "s3://" in str(error.value)
        assert (tmp_path / "train" / ("rows" + suffix)).read_text() == content
        assert [key for _, key, _ in client.downloaded] == [manifest_key]
        return

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    with open(files[0], encoding = "utf-8") as handle:
        if suffix == ".csv":
            result = next(csv.DictReader(handle))
        else:
            result = json.load(handle)
            if suffix == ".json":
                result = result[0]
    expected_path = expected if expected == "missing.wav" else str(tmp_path / expected)
    assert result == {"audio": expected_path, "text": row["text"]}


def test_prefix_and_manifest_relative_candidates_can_be_the_same_file(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        ["datasets/metadata.jsonl", "datasets/clips/a.wav"],
        contents = {"datasets/metadata.jsonl": '{"audio": "clips/a.wav", "text": "hi"}\n'},
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert _rows(files[0])[0]["audio"] == str(tmp_path / "clips/a.wav")


def test_csv_audio_references_are_rewritten_to_local_paths(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        ["datasets/metadata.csv", "datasets/audio/a.wav"],
        contents = {
            "datasets/metadata.csv": "audio,text\naudio/a.wav,hello there\nmissing.wav,kept\n"
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    with open(files[0], encoding = "utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["audio"] == str(tmp_path / "audio" / "a.wav")
    assert rows[0]["text"] == "hello there"
    assert rows[1]["audio"] == "missing.wav"


def test_tabular_only_prefixes_keep_the_flat_layout(monkeypatch, tmp_path):
    _install(monkeypatch, ["datasets/sub/train.parquet", "datasets/other/train.parquet"])
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert sorted(os.path.basename(f) for f in files) == ["train.parquet", "train_1.parquet"]
    assert not (tmp_path / "sub").exists()


def test_a_key_with_dotdot_segments_cannot_escape_the_download_dir(monkeypatch, tmp_path):
    client = _install(
        monkeypatch,
        ["datasets/metadata.jsonl", "datasets/../../evil.wav"],
        contents = {"datasets/metadata.jsonl": '{"audio": "a.wav", "text": "hi"}\n'},
    )
    target = tmp_path / "download"
    target.mkdir()
    with pytest.raises(ValueError, match = "\\.\\."):
        s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(target))
    escaped = tmp_path / "evil.wav"
    assert not escaped.exists()
    assert all("evil" not in key for (_b, key, _p) in client.downloaded)


@pytest.mark.parametrize(
    "parts",
    [
        ["C:evil.txt"],
        ["C:\\Users\\victim\\Startup\\evil.bat"],
        ["\\\\server\\share\\evil.wav"],
        ["audio", "..\\..\\evil.wav"],
        ["\\evil.wav"],
        ["datasets", ".. ", "evil.wav"],
        ["datasets", ". .", "evil.wav"],
        ["datasets\\.. \\evil.wav"],
    ],
)
def test_windows_shaped_keys_cannot_escape_the_download_dir(parts):
    import ntpath
    with pytest.raises(ValueError, match = "S3 key"):
        s3_dataset._contained_local_path("C:\\data\\download", parts, pathmod = ntpath)


@pytest.mark.parametrize(
    "parts", [["train.parquet"], ["audio", "a.wav"], ["audio", "x..y.wav"], ["a\\b.wav"]]
)
def test_ordinary_keys_stay_inside_the_download_dir(parts, tmp_path):
    import ntpath
    assert s3_dataset._contained_local_path(str(tmp_path), parts) == os.path.join(
        str(tmp_path), *parts
    )
    if "\\" not in parts[-1]:
        win = s3_dataset._contained_local_path("C:\\data\\download", parts, pathmod = ntpath)
        assert win == ntpath.join("C:\\data\\download", *parts)


@pytest.mark.parametrize("manifest_format", ["jsonl", "json", "csv"])
@pytest.mark.parametrize(
    "names",
    [
        ("audio/clip.wav", "audio//clip.wav"),
        ("audio/clip.wav", "audio/./clip.wav"),
        ("audio/A.wav", "audio/a.wav"),
        ("audio/caf\u00e9.wav", "audio/cafe\u0301.wav"),
        ("Audio/clip.wav", "audio/clip.wav"),
        ("caf\u00e9/clip.wav", "cafe\u0301/clip.wav"),
    ],
)
def test_audio_aliases_keep_each_recording_and_its_transcript(
    monkeypatch, tmp_path, names, manifest_format
):
    keys = [f"datasets/{name}" for name in names]
    records = [
        {"audio": f"s3://my-bucket/{key}", "text": f"transcript-{i}, heard"}
        for i, key in enumerate(keys)
    ]
    if manifest_format == "csv":
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames = ["audio", "text"])
        writer.writeheader()
        writer.writerows(records)
        manifest = handle.getvalue()
    elif manifest_format == "json":
        manifest = json.dumps(records)
    else:
        manifest = "\n".join(map(json.dumps, records))
    manifest_key = f"datasets/train.{manifest_format}"
    _install(
        monkeypatch,
        [manifest_key, *keys],
        contents = {
            manifest_key: manifest + "\n",
            keys[0]: "FIRST",
            keys[1]: "SECOND",
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    if manifest_format == "csv":
        with open(files[0], encoding = "utf-8", newline = "") as handle:
            rows = list(csv.DictReader(handle))
    elif manifest_format == "json":
        rows = json.loads(Path(files[0]).read_text())
    else:
        rows = _rows(files[0])
    paths = [row["audio"] for row in rows]
    assert len({unicodedata.normalize("NFC", path).casefold() for path in paths}) == 2
    assert [Path(path).read_text() for path in paths] == ["FIRST", "SECOND"]
    assert [row["text"] for row in rows] == ["transcript-0, heard", "transcript-1, heard"]


@pytest.mark.parametrize(
    "names",
    [("train.jsonl", "./train.jsonl"), ("TRAIN.jsonl", "train.jsonl")],
)
def test_manifest_aliases_keep_all_training_rows(monkeypatch, tmp_path, names):
    keys = [f"datasets/{name}" for name in names]
    _install(
        monkeypatch,
        [*keys, "datasets/audio/clip.wav"],
        contents = {
            key: json.dumps({"audio": "audio/clip.wav", "text": f"transcript-{i}"}) + "\n"
            for i, key in enumerate(keys)
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert len({unicodedata.normalize("NFC", path).casefold() for path in files}) == 2
    assert [row["text"] for path in files for row in _rows(path)] == [
        "transcript-0",
        "transcript-1",
    ]
    assert all(Path(_rows(path)[0]["audio"]).is_file() for path in files)


def test_audio_alias_suffixes_do_not_overwrite_another_object(monkeypatch, tmp_path):
    keys = ["datasets/audio/clip.wav", "datasets/audio//clip.wav", "datasets/audio/clip_1.wav"]
    _install(
        monkeypatch,
        ["datasets/metadata.jsonl", *keys],
        contents = {
            "datasets/metadata.jsonl": "\n".join(
                json.dumps({"audio": f"s3://my-bucket/{key}"}) for key in keys
            )
            + "\n",
            **{key: f"recording-{i}" for i, key in enumerate(keys)},
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert [Path(row["audio"]).read_text() for row in _rows(files[0])] == [
        "recording-0",
        "recording-1",
        "recording-2",
    ]


@pytest.mark.parametrize("existing_name", ["metadata.jsonl", "audio/clip.wav"])
def test_structured_download_keeps_existing_local_files(monkeypatch, tmp_path, existing_name):
    existing = tmp_path / existing_name
    existing.parent.mkdir(parents = True, exist_ok = True)
    existing.write_text("KEEP")
    _install(
        monkeypatch,
        ["datasets/metadata.jsonl", "datasets/audio/clip.wav"],
        contents = {
            "datasets/metadata.jsonl": '{"audio":"audio/clip.wav","text":"new sample"}\n',
            "datasets/audio/clip.wav": "NEW",
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert existing.read_text() == "KEEP"
    assert Path(_rows(files[0])[0]["audio"]).read_text() == "NEW"


@pytest.mark.parametrize("names", [("clip.wav", "/clip.wav"), ("A.wav", "a.wav")])
def test_parquet_audio_aliases_download_every_object(monkeypatch, tmp_path, names):
    keys = [f"datasets/audio/{name}" for name in names]
    _install(
        monkeypatch,
        ["datasets/train.parquet", *keys],
        contents = {keys[0]: "FIRST", keys[1]: "SECOND"},
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert files == [str(tmp_path / "train.parquet")]
    audio = sorted(path.read_text() for path in (tmp_path / "audio").iterdir())
    assert audio == ["FIRST", "SECOND"]


def test_parquet_download_keeps_existing_audio(monkeypatch, tmp_path):
    existing = tmp_path / "clip.wav"
    existing.write_text("KEEP")
    _install(monkeypatch, ["datasets/train.parquet", "datasets/clip.wav"])
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert files == [str(tmp_path / "train.parquet")]
    assert existing.read_text() == "KEEP"
    assert (tmp_path / "clip_1.wav").read_text() == "content-of:datasets/clip.wav"


@pytest.mark.parametrize("second_key", ["./train.parquet", "TRAIN.parquet"])
def test_parquet_manifest_aliases_keep_both_shards(monkeypatch, tmp_path, second_key):
    _install(
        monkeypatch,
        ["datasets/train.parquet", f"datasets/{second_key}", "datasets/clip.wav"],
        contents = {
            "datasets/train.parquet": "FIRST",
            f"datasets/{second_key}": "SECOND",
        },
    )
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert [_parquet_rows(path)[0]["text"] for path in files] == ["FIRST", "SECOND"]
    assert (tmp_path / "clip.wav").is_file()


def test_parquet_audio_without_collisions_keeps_its_paths(monkeypatch, tmp_path):
    _install(monkeypatch, ["datasets/train.parquet", "datasets/audio/clip.wav"])
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert files == [str(tmp_path / "train.parquet")]
    assert (tmp_path / "audio" / "clip.wav").read_text() == "content-of:datasets/audio/clip.wav"


@pytest.mark.parametrize("layout", ["string", "struct", "list"])
@pytest.mark.parametrize(
    "reference", ["audio/a.wav", "./audio/a.wav", "s3://my-bucket/datasets/train/audio/a.wav"]
)
def test_parquet_audio_references_resolve_to_the_downloaded_recording(
    monkeypatch, tmp_path, layout, reference
):
    value = reference if layout == "string" else {"path": reference, "bytes": None}
    if layout == "list":
        value = [value, None]
    row = {"audio": value, "text": "transcript", "speaker_id": 7}
    _install(
        monkeypatch,
        ["datasets/train/rows.parquet", "datasets/train/audio/a.wav"],
        contents = {"datasets/train/rows.parquet": _parquet_bytes([row])},
    )
    download = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    path = str(tmp_path / "train" / "audio" / "a.wav")
    expected = path if layout == "string" else {"path": path, "bytes": None}
    if layout == "list":
        expected = [expected, None]
    assert _parquet_rows(download.files[0]) == [{**row, "audio": expected}]
    assert Path(path).is_file()


def test_parquet_rewrite_preserves_hf_audio_schema_and_embedded_bytes(monkeypatch, tmp_path):
    import struct
    import wave

    import datasets
    import pyarrow.parquet as pq

    features = datasets.Features(
        {"audio": datasets.Audio(decode = False), "text": datasets.Value("string")}
    )
    rows = [
        {"audio": {"path": "a.wav", "bytes": None}, "text": "external"},
        {"audio": {"path": None, "bytes": b"embedded recording"}, "text": "embedded"},
        {"audio": None, "text": "missing"},
    ]
    schema = features.arrow_schema.with_metadata(
        {**features.arrow_schema.metadata, b"source": b"keep metadata"}
    )
    recording = io.BytesIO()
    samples = struct.pack("<4h", 42, 2048, -42, -2048)
    with wave.open(recording, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(samples)
    _install(
        monkeypatch,
        ["datasets/rows.parquet", "datasets/a.wav"],
        contents = {
            "datasets/rows.parquet": _parquet_bytes(rows, schema),
            "datasets/a.wav": recording.getvalue(),
        },
    )
    download = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    assert pq.ParquetFile(download.files[0]).schema_arrow.equals(schema, check_metadata = True)
    loaded = datasets.load_dataset("parquet", data_files = download.files, split = "train")
    assert loaded.features == features
    assert loaded[0]["audio"]["path"] == str(tmp_path / "a.wav")
    with wave.open(loaded[0]["audio"]["path"], "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.readframes(4) == samples
    assert loaded[1] == rows[1] and loaded[2] == rows[2]


def test_parquet_rewrite_uses_the_allocated_names_for_colliding_audio(monkeypatch, tmp_path):
    keys = ["datasets/audio/clip.wav", "datasets/audio//clip.wav"]
    rows = [{"audio": f"s3://my-bucket/{key}", "text": key} for key in keys]
    _install(
        monkeypatch,
        ["datasets/rows.parquet", *keys],
        contents = {
            "datasets/rows.parquet": _parquet_bytes(rows),
            keys[0]: "FIRST",
            keys[1]: "SECOND",
        },
    )
    download = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    loaded = _parquet_rows(download.files[0])
    assert loaded[0]["audio"] != loaded[1]["audio"]
    assert [Path(row["audio"]).read_text() for row in loaded] == ["FIRST", "SECOND"]
    assert [row["text"] for row in loaded] == keys


def test_parquet_without_matching_references_stays_byte_identical(monkeypatch, tmp_path):
    rows = [
        {"audio": None, "text": "keep"},
        {"audio": "https://example.com/a.wav", "text": "remote"},
    ]
    original = _parquet_bytes(rows)
    _install(
        monkeypatch,
        ["datasets/rows.parquet", "datasets/a.wav"],
        contents = {"datasets/rows.parquet": original},
    )
    download = s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    assert Path(download.files[0]).read_bytes() == original


def test_parquet_ambiguity_in_a_later_batch_preserves_the_manifest(monkeypatch, tmp_path):
    rows = [{"audio": "s3://my-bucket/datasets/a.wav", "text": "safe"}] * 1100
    rows.append({"audio": "a.wav", "text": "ambiguous"})
    original = _parquet_bytes(rows)
    client = _install(
        monkeypatch,
        ["datasets/train/rows.parquet", "datasets/a.wav", "datasets/train/a.wav"],
        contents = {"datasets/train/rows.parquet": original},
    )
    with pytest.raises(ValueError, match = "Ambiguous audio reference"):
        s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    manifest = tmp_path / "train" / "rows.parquet"
    assert manifest.read_bytes() == original
    assert list(manifest.parent.iterdir()) == [manifest]
    assert [key for _, key, _ in client.downloaded] == ["datasets/train/rows.parquet"]


def test_parquet_write_failure_preserves_the_manifest_and_removes_staging(monkeypatch, tmp_path):
    import errno
    import pyarrow.parquet as pq

    original = _parquet_bytes([{"audio": "a.wav", "text": "keep on failure"}])
    client = _install(
        monkeypatch,
        ["datasets/rows.parquet", "datasets/a.wav"],
        contents = {"datasets/rows.parquet": original},
    )

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(pq.ParquetWriter, "write_table", full_disk)
    with pytest.raises(OSError, match = "disk full"):
        s3_dataset.prepare_s3_dataset_download(_cfg(), dest_dir = str(tmp_path))
    manifest = tmp_path / "rows.parquet"
    assert manifest.read_bytes() == original
    assert list(tmp_path.iterdir()) == [manifest]
    assert [key for _, key, _ in client.downloaded] == ["datasets/rows.parquet"]
