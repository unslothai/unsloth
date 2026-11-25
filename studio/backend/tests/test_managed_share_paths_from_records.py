# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's library item id or diffusion run record must not make Studio touch a
share path (on Windows any stat of one is an outbound SMB sign-in with the service account)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core import library
from core.training import diffusion_checkpoint, diffusion_training_service as dts
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
SHARE = "\\\\attacker.example\\share\\m"


def test_imports_this_tree():
    here = Path(__file__).resolve().parents[1]
    assert Path(library.__file__).resolve().is_relative_to(here)
    assert Path(dts.__file__).resolve().is_relative_to(here)


@pytest.fixture
def share_probes(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    touched: list[str] = []
    real_realpath, real_exists = os.path.realpath, os.path.exists

    def realpath(p, *a, **k):
        if "attacker.example" in str(p):
            touched.append(str(p))
        return real_realpath(p, *a, **k)

    def exists(p):
        if "attacker.example" in str(p):
            touched.append(str(p))
        return real_exists(p)

    monkeypatch.setattr(os.path, "realpath", realpath)
    monkeypatch.setattr(os.path, "exists", exists)
    return touched


def test_managed_model_id_share_never_touched(share_probes):
    item = f"model:training:{SHARE}"
    assert run_as(ALICE, library.fingerprint, item) is None
    assert run_as(ALICE, library.item_exists, item) is False
    with pytest.raises(ValueError):
        run_as(ALICE, library._delete_item, item, None)
    assert share_probes == []


def test_owner_model_id_unchanged(share_probes):
    # The owner path keeps its old behaviour (it resolves, then fails the root check).
    assert run_as(OWNER, library.fingerprint, f"model:training:{SHARE}") is None
    assert share_probes


def test_managed_run_record_share_not_scanned(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    seen: list = []

    def describe(output_dir, **kwargs):
        seen.append((output_dir, kwargs.get("source_checkpoint")))
        return {"can_resume": True, "checkpoint_step": 3, "checkpoint_path": "x"}

    monkeypatch.setattr(diffusion_checkpoint, "describe_resume_state", describe)
    for record_dir in (SHARE, str(tmp_path / "elsewhere")):
        out = run_as(ALICE, dts._resume_fields, record_dir, status = "error")
        assert out["can_resume"] is False
    assert seen == []
    # Owner records are scanned exactly as before.
    assert run_as(OWNER, dts._resume_fields, SHARE, status = "error")["can_resume"] is True
    assert seen == [(SHARE, None)]
