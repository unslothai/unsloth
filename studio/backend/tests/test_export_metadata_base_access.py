# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local merged vision export builds its processor from the base_model in export_metadata.json,
so a managed load must be allowed that base too, as it must for an adapter's base."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.inference as routes


def _export(tmp_path, base):
    folder = tmp_path / "mine"
    folder.mkdir(parents = True)
    (folder / "export_metadata.json").write_text(json.dumps({"base_model": base}))
    return SimpleNamespace(is_local = True, is_lora = False, path = str(folder), base_model = None)


def _deny_owner_paths(reference, *args, **kwargs):
    if str(reference).startswith("/srv/owner"):
        raise HTTPException(status_code = 404, detail = "Model not found")


def test_managed_load_refuses_a_foreign_export_metadata_base(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(routes.account_access, "require_model_access", _deny_owner_paths)
    with pytest.raises(HTTPException) as err:
        routes._require_resolved_base_access(_export(tmp_path, "/srv/owner/exports/support-vlm"))
    assert err.value.status_code == 404


def test_reachable_bases_and_the_owner_are_unchanged(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(
        routes.account_access, "require_model_access", lambda ref, *a, **k: seen.append(ref)
    )
    routes._require_resolved_base_access(_export(tmp_path, "unsloth/Qwen2-VL-2B-Instruct"))
    assert seen == ["unsloth/Qwen2-VL-2B-Instruct"]
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(routes.account_access, "require_model_access", _deny_owner_paths)
    routes._require_resolved_base_access(_export(tmp_path / "o", "/srv/owner/exports/x"))
