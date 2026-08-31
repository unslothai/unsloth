# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""S3 audio datasets: audio files download beside their manifest and stay reachable.

#4539: a Whisper finetune's dataset is audio files on S3 plus a transcription
manifest. The tabular-only loader (#6222) filtered the audio keys out of the
listing, flattened every download to its basename, and called a manifest+audio
prefix a mixed-format error -- so the one layout this feature exists for could
never load. Pinned here: audio keys are downloaded preserving their structure,
the single-format rule judges only the manifest files, and the manifest's audio
references are rewritten to the materialized local paths (prefix-relative keys,
s3:// URIs, and manifest-relative paths; anything unmatched is left alone).

Same harness as test_s3_dataset.py: boto3 may be absent in CI, so the client is
faked and no network or credentials are involved.
"""

import csv
import importlib.util
import json
import os
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

    def __init__(self, keys, contents = None):
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
        with open(local_path, "w", encoding = "utf-8") as f:
            f.write(self._contents.get(key, f"content-of:{key}"))


def _install(monkeypatch, keys, contents = None):
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
            "datasets/notes.txt",  # still unsupported, still skipped
        ],
        contents = {"datasets/metadata.jsonl": '{"audio": "audio/a.wav", "text": "hi"}\n'},
    )

    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))

    # The loader consumes manifests only; audio materializes beside them.
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


def test_jsonl_audio_references_are_rewritten_to_local_paths(monkeypatch, tmp_path):
    manifest = "\n".join(
        [
            # A prefix-relative key.
            json.dumps({"audio": "audio/a.wav", "text": "one"}),
            # A full S3 URI to a downloaded key.
            json.dumps({"audio": "s3://my-bucket/datasets/audio/b.mp3", "text": "two"}),
            # The HF undecoded-audio dict shape.
            json.dumps({"audio": {"path": "audio/a.wav"}, "text": "three"}),
            # Unmatched references are somebody else's contract: left alone.
            json.dumps({"audio": "https://example.com/c.wav", "text": "four"}),
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
    assert all(os.path.isabs(p) for p in (rows[0]["audio"], rows[1]["audio"]))
    assert rows[0]["text"] == "one"


def test_manifest_relative_references_resolve_against_the_manifest_dir(monkeypatch, tmp_path):
    # The manifest lives in a subdirectory and references a sibling by name.
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
    # An unmatched reference stays; it may be absolute on the training host.
    assert rows[1]["audio"] == "missing.wav"


def test_tabular_only_prefixes_keep_the_flat_layout(monkeypatch, tmp_path):
    # #6222's contract: no audio in the listing means basenames in the target
    # dir, name collisions deduplicated -- unchanged by the audio path.
    _install(monkeypatch, ["datasets/sub/train.parquet", "datasets/other/train.parquet"])
    files = s3_dataset.download_s3_dataset(_cfg(), dest_dir = str(tmp_path))
    assert sorted(os.path.basename(f) for f in files) == ["train.parquet", "train_1.parquet"]
    assert not (tmp_path / "sub").exists()


def test_a_key_with_dotdot_segments_cannot_escape_the_download_dir(monkeypatch, tmp_path):
    # ".." is a legal literal in an S3 key, and the structured layout joins
    # keys into filesystem paths. Bucket contents are external input: a key
    # aimed above the temp dir must stop the download, not write there.
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
