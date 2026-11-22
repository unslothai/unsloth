# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in live test on the bucket named by UNSLOTH_S3_LIVE_BUCKET; uploads are deleted afterwards."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import uuid
import wave
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

BUCKET = os.environ.get("UNSLOTH_S3_LIVE_BUCKET", "")
pytestmark = pytest.mark.skipif(not BUCKET, reason = "UNSLOTH_S3_LIVE_BUCKET not set")


def _wav(frames: int = 800) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(
            b"".join(struct.pack("<h", 3000 if i % 20 < 10 else -3000) for i in range(frames))
        )
    return buf.getvalue()


@pytest.fixture
def uploaded():
    boto3 = pytest.importorskip("boto3")
    client = boto3.client("s3", region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    prefix = f"unsloth-live-{uuid.uuid4().hex[:8]}/"
    manifest = "\n".join(
        [
            json.dumps({"audio": "audio/a.wav", "text": "one"}),
            json.dumps({"audio": f"s3://{BUCKET}/{prefix}audio/sub/b.wav", "text": "two"}),
            json.dumps({"audio": {"path": "audio/a.wav"}, "text": "three"}),
            json.dumps({"audio": "https://example.com/c.wav", "text": "four"}),
        ]
    )
    objects = {
        f"{prefix}metadata.jsonl": (manifest + "\n").encode(),
        f"{prefix}audio/a.wav": _wav(),
        f"{prefix}audio/sub/b.wav": _wav(),
    }
    for key, body in objects.items():
        client.put_object(Bucket = BUCKET, Key = key, Body = body)
    try:
        yield prefix
    finally:
        client.delete_objects(Bucket = BUCKET, Delete = {"Objects": [{"Key": key} for key in objects]})


def test_audio_dataset_round_trips_through_a_real_bucket(uploaded, tmp_path):
    from core.training import s3_dataset

    cfg = {
        "bucket": BUCKET,
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "prefix": uploaded,
        "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "use_iam_role": not os.environ.get("AWS_ACCESS_KEY_ID"),
    }
    files = s3_dataset.download_s3_dataset(cfg, dest_dir = str(tmp_path))

    assert [os.path.relpath(f, tmp_path) for f in files] == ["metadata.jsonl"]
    a = str(tmp_path / "audio" / "a.wav")
    b = str(tmp_path / "audio" / "sub" / "b.wav")
    assert os.path.isfile(a) and os.path.isfile(
        b
    ), "audio lands beside the manifest, structure kept"
    for path in (a, b):
        with wave.open(path) as w:
            assert w.getnframes() == 800, "bytes survive the trip"

    rows = [json.loads(line) for line in open(files[0], encoding = "utf-8") if line.strip()]
    assert rows[0]["audio"] == a
    assert rows[1]["audio"] == b
    assert rows[2]["audio"]["path"] == a
    assert rows[3]["audio"] == "https://example.com/c.wav"
