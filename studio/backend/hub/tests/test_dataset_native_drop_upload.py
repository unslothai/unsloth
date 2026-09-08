# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from hub.services.datasets import local
import utils.native_path_leases as leases

SECRET = b"d" * 32


@pytest.fixture(autouse = True)
def _lease_secret(monkeypatch):
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(
    path: Path,
    *,
    operation = "dataset-import",
    path_kind = "dataset",
) -> str:
    stat = path.stat()
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": path_kind,
        "path_type": "file",
        "source_kind": "drop",
        "token_id_hash": hashlib.sha256(b"dataset_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": os.urandom(16).hex(),
        "display_label": path.name,
        "size_bytes": stat.st_size,
        "modified_ms": int(stat.st_mtime_ns // 1_000_000),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(
        SECRET,
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64(signature)}"


def test_signed_dataset_drop_is_copied_to_upload_storage(monkeypatch, tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"hello"}\n', encoding = "utf-8")
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", upload_root)

    response = local._native_upload_dataset_response(_sign(source))

    stored_path = Path(response.stored_path)
    assert response.filename == source.name
    assert stored_path.parent == upload_root
    assert stored_path.read_bytes() == source.read_bytes()
    assert stored_path.resolve() != source.resolve()


def test_async_dataset_drop_offloads_native_copy(monkeypatch, tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", tmp_path / "uploads")
    offloaded = []

    async def run_offloaded(function, *args):
        offloaded.append(function)
        return function(*args)

    monkeypatch.setattr(local.asyncio, "to_thread", run_offloaded)

    response = asyncio.run(local.upload_dataset_response(None, _sign(source)))

    assert Path(response.stored_path).read_bytes() == source.read_bytes()
    assert offloaded == [local._native_upload_dataset_response]


@pytest.mark.parametrize(
    "operation,path_kind",
    [("attach", "dataset"), ("dataset-import", "attachment")],
)
def test_dataset_drop_rejects_grants_for_other_purposes(
    monkeypatch, tmp_path, operation, path_kind
):
    source = tmp_path / "train.csv"
    source.write_text("text\nhello\n", encoding = "utf-8")
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", tmp_path / "uploads")

    with pytest.raises(HTTPException) as excinfo:
        local._native_upload_dataset_response(
            _sign(source, operation = operation, path_kind = path_kind)
        )

    assert excinfo.value.status_code == 400


def test_dataset_drop_rejects_unsupported_extensions(monkeypatch, tmp_path):
    source = tmp_path / "payload.exe"
    source.write_bytes(b"MZ")
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", tmp_path / "uploads")

    with pytest.raises(HTTPException) as excinfo:
        local._native_upload_dataset_response(_sign(source))

    assert excinfo.value.status_code == 400
