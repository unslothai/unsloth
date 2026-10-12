# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native GGUF selection is authorized on the lease's canonical path, not its display label.

The desktop shell mints the signed lease; on a multi-account install the account signed in
inside that shell is a managed one, and its ``model_path`` is the file's display label.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import utils.native_path_leases as leases
from auth import policy
from hub.services.models import account_access as access
from models.inference import LoadRequest, ValidateModelRequest
from routes import inference as inference_routes
from utils.account_context import AccountContext, arun_as, run_as
from utils.paths.storage_roots import workspace_root

ALICE = AccountContext("a" * 32, "alice")
SECRET = base64.urlsafe_b64encode(b"native-lease-secret-for-tests-32").decode()


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv(leases.LEASE_SECRET_ENV, SECRET)
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_public_repos", {})
    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        ),
    )


def _mint(path, operation: str, label: str) -> str:
    st = os.lstat(path)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": "model",
        "path_type": "file",
        "source_kind": "drop",
        "token_id_hash": "token-hash",
        "display_label": label,
        "issued_at_ms": int(time.time() * 1000) - 1000,
        "expires_at_ms": int(time.time() * 1000) + 600_000,
        "nonce": base64.urlsafe_b64encode(os.urandom(12)).decode(),
        "size_bytes": st.st_size,
        "modified_ms": int(st.st_mtime_ns // 1_000_000),
        "device_id": format(st.st_dev, "x"),
        "file_id": format(st.st_ino, "x"),
    }
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(
        base64.urlsafe_b64decode(SECRET), body.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{body}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"


def _gguf(directory, name = "private-model.gguf"):
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / name
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


def _validate(request):
    return asyncio.run(arun_as(ALICE, inference_routes.validate_model(request, None, "alice")))


def test_validate_accepts_a_native_gguf_inside_the_accounts_own_workspace():
    gguf = _gguf(run_as(ALICE, workspace_root) / "models")
    request = ValidateModelRequest(
        model_path = gguf.name,
        native_path_lease = _mint(gguf, "validate-model", gguf.name),
    )
    response = _validate(request)
    assert response.valid is True


def test_validate_refuses_a_native_gguf_outside_the_accounts_roots(tmp_path):
    gguf = _gguf(tmp_path / "elsewhere")
    request = ValidateModelRequest(
        model_path = gguf.name,
        native_path_lease = _mint(gguf, "validate-model", gguf.name),
    )
    with pytest.raises(HTTPException) as excinfo:
        _validate(request)
    assert excinfo.value.status_code == 404


def test_load_refuses_a_native_gguf_outside_the_accounts_roots(tmp_path):
    gguf = _gguf(tmp_path / "elsewhere")
    request = LoadRequest(
        model_path = gguf.name,
        native_path_lease = _mint(gguf, "load-model", gguf.name),
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(arun_as(ALICE, inference_routes._load_model_impl(request, None, "alice")))
    assert excinfo.value.status_code == 404
