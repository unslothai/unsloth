# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MLX load's response answers trust_remote_code from the files the worker opened, not the Hub."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import routes.inference as routes
from core.inference import model_ids, worker
from core.inference.orchestrator import _mirrored_model_entry

resolve = routes._resolve_loaded_trust_remote_code


def test_the_loaded_answer_replaces_the_hub_read_only(monkeypatch):
    monkeypatch.setattr(routes, "_auto_map_trust_remote_code", lambda *_a: 1 / 0)
    assert resolve("org/m", None, {}, loaded_auto_map = True) is True
    assert resolve("org/m", None, {}, loaded_auto_map = False) is False
    assert resolve("org/m", None, {}, trust_remote_code_used = True, loaded_auto_map = False) is True
    stored = {"requires_trust_remote_code": False}
    assert resolve("org/m", stored, {}, loaded_auto_map = True) is False
    monkeypatch.setattr(routes, "_auto_map_trust_remote_code", lambda *_a: True)
    assert resolve("org/m", None, {}) is True
    assert _mirrored_model_entry({"loaded_auto_map": False}, "org/m")["loaded_auto_map"] is False
    assert hasattr(worker._handle_load, "__wrapped__")


@pytest.mark.parametrize(
    "config, device, fields, base, answer",
    [
        ('{"auto_map": {"AutoModel": "m.M"}}', "mlx", {}, None, True),
        ('{"model_type": "qwen3"}', "mlx", {}, None, False),
        ('{"model_type": "qwen3"}', "mlx", {"is_lora": True}, None, None),
        ('{"model_type": "qwen3"}', "mlx", {"gguf_variant": "4bit"}, None, None),
        ('{"model_type": "qwen3"}', "mlx", {}, "org/base", None),
    ],
)
def test_the_worker_answers_only_for_its_own_files(
    tmp_path, monkeypatch, config, device, fields, base, answer
):
    monkeypatch.setattr(model_ids, "mlx_host_bnb_base_repo", lambda _name: base)
    (tmp_path / "config.json").write_text(config)
    backend = SimpleNamespace(device = device, _model = SimpleNamespace(_src_path = str(tmp_path)))
    mc = SimpleNamespace(identifier = "org/m", **{"is_lora": False, **fields})
    assert worker._loaded_auto_map(backend, mc) is answer
