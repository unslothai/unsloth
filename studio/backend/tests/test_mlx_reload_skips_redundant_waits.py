# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One MLX load asks the Hub each question once, and a reload does not wait for memory nothing reads yet."""

from __future__ import annotations

import inspect
import urllib.error
from types import SimpleNamespace

import pytest

from core.inference import orchestrator, worker
from utils import utils as uu
from utils.models import model_config as mc


class _PastTheSettle(BaseException):
    pass


def _settle_calls(
    monkeypatch,
    device,
    expected = None,
    settled = True,
):
    calls = []
    monkeypatch.setattr(orchestrator, "get_device", lambda: SimpleNamespace(value = device))
    monkeypatch.setattr(orchestrator, "prepare_gpu_selection", lambda *a, **k: (None, {}))
    monkeypatch.setattr("utils.transformers_version.needs_transformers_5", lambda *_a: False)
    monkeypatch.setattr("utils.transformers_version.sidecar_swap_kind", lambda: None)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _s: None)

    def released():
        raise _PastTheSettle

    backend = orchestrator.InferenceOrchestrator.__new__(orchestrator.InferenceOrchestrator)
    vars(backend).update(
        loading_models = set(),
        models = {},
        active_model_name = None,
        _managed_engine = None,
        _proc = object(),
        _ensure_subprocess_alive = lambda: True,
        _cancel_generation = lambda: None,
        _shutdown_subprocess = lambda *a, **k: True,
        _wait_for_worker_vram_settle = lambda _since, *, expected_free_gb = None, **_: (
            calls.append(expected_free_gb) or True
        ),
    )
    with pytest.raises(_PastTheSettle):
        backend.load_model(
            SimpleNamespace(identifier = "org/m", gguf_variant = None),
            on_prior_worker_released = released,
            post_handoff_expected_free_gb = expected,
        )
    return calls


@pytest.mark.parametrize(
    "device, expected, calls",
    [("mlx", None, []), ("mlx", {0: 8.0}, [{0: 8.0}]), ("cuda", None, [None])],
)
def test_only_mlx_with_no_figure_skips_the_settle(monkeypatch, device, expected, calls):
    assert _settle_calls(monkeypatch, device, expected) == calls


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


def test_a_shared_answer_replays_its_token_rejection_into_a_later_scope(monkeypatch):
    from hub.utils.hf_tokens import saved_token_rejected, token_rejection_scope

    def model_info(repo, *, token, **_k):
        if token:
            raise urllib.error.HTTPError(repo, 401, "expired", None, None)
        return repo

    monkeypatch.setattr("huggingface_hub.model_info", model_info)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    with mc.shared_hub_model_info():
        mc._hub_model_info("org/m", "hf_expired")
        with token_rejection_scope():
            assert not saved_token_rejected("hf_expired")
            assert mc._hub_model_info("org/m", "hf_expired") == "org/m"
            assert saved_token_rejected("hf_expired")


def test_only_the_mlx_load_holds_the_answers():
    source = inspect.getsource(worker.run_inference_process)
    scope = source.index("hub_answers.enter_context(pinned_hf_reachability(_probed_offline))")
    assert (
        scope
        < source.index("mlx_refusal(model_name")
        < source.index("_handle_load(backend, config")
    )
    closed = source.rindex("hub_answers.close()")
    assert (
        source.index("read_hub_model_info_ahead(model_name")
        < source.index("_hw.detect_hardware()")
        < source.index("hub_answers.close()")
        < source.index("model_info_ahead.join()")
        < source.index("mlx_refusal(model_name")
    )
    assert (
        source.index("_handle_load(backend, config")
        < closed
        < source.index("batch = _ResidentBatch")
    )
