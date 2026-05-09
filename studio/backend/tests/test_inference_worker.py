# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import worker
import utils.models as models


def test_build_model_config_forwards_trust_remote_code(monkeypatch):
    seen: dict[str, object] = {}

    class DummyModelConfig:
        @staticmethod
        def from_identifier(**kwargs):
            seen.update(kwargs)
            return object()

    monkeypatch.setattr(models, "ModelConfig", DummyModelConfig)

    worker._build_model_config(
        {
            "model_name": "org/custom-code-vlm",
            "trust_remote_code": True,
            "hf_token": "",
            "gguf_variant": None,
        }
    )

    assert seen["model_id"] == "org/custom-code-vlm"
    assert seen["trust_remote_code"] is True
