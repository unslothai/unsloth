# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from auth import model_policy


def test_blocklist_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    assert model_policy.get_blocked_models() == []
    assert model_policy.set_blocked_models(["Org/Big", "org/big", " x/y ", ""]) == ["Org/Big", "x/y"]
    assert model_policy.is_model_blocked("ORG/big")
    assert not model_policy.is_model_blocked("a/b")
    assert not model_policy.is_model_blocked(None)
