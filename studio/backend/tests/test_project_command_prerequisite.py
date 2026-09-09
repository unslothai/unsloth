# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from core.agent_workspace import execution


def test_missing_edit_boundary_refuses_before_native_probe(monkeypatch):
    monkeypatch.setattr(execution, "EDIT_BOUNDARY_AVAILABLE", False)
    monkeypatch.setattr(
        execution, "_probe_backend", lambda *args: pytest.fail("Native probe must not run")
    )
    status = execution.execution_boundary_status("linux")
    assert not status.available
    assert "file edit support" in status.reason
