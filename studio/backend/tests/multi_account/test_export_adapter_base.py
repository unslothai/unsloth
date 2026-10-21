# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An export load authorizes the adapter's base with the caller's account and pins it for the worker."""

import json

import pytest
from fastapi import HTTPException

from routes import export as route
from utils.account_context import AccountContext, OWNER, run_as
from utils.paths import outputs_root

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


def _adapter(account, name, base):
    path = run_as(account, lambda: outputs_root() / name)
    path.mkdir(parents = True, exist_ok = True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base})
    )
    return path


def test_foreign_adapter_base_is_refused(isolated_auth):
    foreign = run_as(BOB, lambda: outputs_root() / "base")
    foreign.mkdir(parents = True)
    adapter = _adapter(ALICE, "adapter", str(foreign))
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, route._authorized_adapter_base, str(adapter))
    assert exc.value.status_code == 404


def test_own_adapter_base_is_pinned_into_the_worker_config(isolated_auth, monkeypatch):
    from core.export.orchestrator import ExportOrchestrator

    own = run_as(ALICE, lambda: outputs_root() / "base")
    own.mkdir(parents = True)
    adapter = _adapter(ALICE, "adapter", str(own))
    assert run_as(ALICE, route._authorized_adapter_base, str(adapter)) == str(own)
    assert run_as(OWNER, route._authorized_adapter_base, str(adapter)) is None

    backend = ExportOrchestrator()
    configs = []
    monkeypatch.setattr(backend, "_spawn_subprocess", configs.append)
    monkeypatch.setattr(backend, "_wait_response", lambda *a, **k: {"success": True, "message": ""})
    monkeypatch.setattr(backend, "_ensure_subprocess_alive", lambda: False)
    run_as(ALICE, backend.load_checkpoint, str(adapter), base_model = str(own))
    assert configs[0]["base_model"] == str(own)
