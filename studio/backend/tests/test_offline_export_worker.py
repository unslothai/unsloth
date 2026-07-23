# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def test_offline_window_forces_in_process_hub_flags(monkeypatch):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")

    import huggingface_hub.constants as hub_constants
    import transformers.utils.hub as transformers_hub
    import utils.transformers_version as transformers_version
    from core.export.worker import _offline_window_if_unreachable

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: False)
    monkeypatch.setattr(
        transformers_version,
        "hf_endpoint_unreachable",
        lambda timeout = 3: True,
    )
    monkeypatch.setattr(
        hub_constants,
        "HF_HUB_OFFLINE",
        False,
        raising = False,
    )
    if hasattr(transformers_hub, "_is_offline_mode"):
        monkeypatch.setattr(
            transformers_hub,
            "_is_offline_mode",
            False,
            raising = False,
        )
    if hasattr(transformers_hub, "OFFLINE"):
        monkeypatch.setattr(
            transformers_hub,
            "OFFLINE",
            False,
            raising = False,
        )

    with _offline_window_if_unreachable(step = "test"):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert hub_constants.HF_HUB_OFFLINE is True
        if hasattr(transformers_hub, "_is_offline_mode"):
            assert transformers_hub._is_offline_mode is True
        if hasattr(transformers_hub, "OFFLINE"):
            assert transformers_hub.OFFLINE is True

    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_offline_window_falls_back_when_force_context_enter_fails(monkeypatch):
    import transformers  # noqa: F401 - ensures the in-process force path is selected
    import utils.transformers_version as transformers_version
    import core.export.worker as worker

    class _BrokenContext:
        def __enter__(self):
            raise RuntimeError("offline force unavailable")

        def __exit__(self, *_args):
            raise AssertionError("__exit__ should not run when __enter__ failed")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: False)
    monkeypatch.setattr(
        transformers_version,
        "hf_endpoint_unreachable",
        lambda timeout = 3: True,
    )
    monkeypatch.setattr(
        worker,
        "_force_hf_offline_window",
        lambda: _BrokenContext(),
    )

    with worker._offline_window_if_unreachable(step = "test"):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None
