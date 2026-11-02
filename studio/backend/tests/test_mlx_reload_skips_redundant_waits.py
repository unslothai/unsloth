# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One MLX load asks the Hub each question once."""

from __future__ import annotations

import inspect

import pytest

from core.inference import worker
from utils import utils as uu
from utils.models import model_config as mc


def test_a_supplied_verdict_answers_without_a_probe(monkeypatch):
    monkeypatch.setattr("utils.transformers_version.hf_endpoint_unreachable", lambda *a, **k: 1 / 0)
    uu.reset_hf_reachability_cache()
    with uu.pinned_hf_reachability(True):
        assert uu.hf_unreachable() is True


def test_nested_scopes_join_and_the_scope_ends_with_the_outermost(monkeypatch):
    reads = []
    monkeypatch.setattr("huggingface_hub.model_info", lambda repo, **_k: reads.append(repo) or repo)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    with mc.shared_hub_model_info():
        mc._hub_model_info("org/m")
        with mc.shared_hub_model_info():
            mc._hub_model_info("org/m")
        mc._hub_model_info("org/m")
    assert reads == ["org/m"]
    assert mc._hub_model_info_scope.get() is None


def test_only_the_mlx_load_holds_the_answers():
    source = inspect.getsource(worker.run_inference_process)
    scope = source.index("hub_answers.enter_context(pinned_hf_reachability(_probed_offline))")
    assert (
        scope
        < source.index("mlx_refusal(model_name")
        < source.index("_handle_load(backend, config")
    )
    closed = source.index("hub_answers.close()")
    assert (
        source.index("_handle_load(backend, config")
        < closed
        < source.index("batch = _ResidentBatch")
    )
