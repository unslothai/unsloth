# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selected warmup groups run only their stages and required shared prerequisites."""

from contextlib import nullcontext
import threading

import pytest


@pytest.fixture
def selected_warm(monkeypatch):
    from utils import torch_warmup as warm

    calls = []
    monkeypatch.delenv(warm.DISABLE_ENV_VAR, raising = False)
    monkeypatch.setattr(warm, "_thread", None)
    monkeypatch.setattr(warm, "_thread_epoch", None)
    monkeypatch.setattr(warm, "_status", {"started": False, "finished": False, "stages": {}})
    monkeypatch.setattr(warm, "_detection_epoch", lambda: 7)
    monkeypatch.setattr(warm, "_owning_epoch", lambda epoch: nullcontext())
    monkeypatch.setattr(
        warm,
        "_STAGES",
        tuple(
            (name, lambda name = name: calls.append(name))
            for name in ("hardware", "inference_backend", "transformers", "datasets")
        ),
    )
    monkeypatch.setattr(warm, "_kick_early_quant_probe", lambda: calls.append("quantization"))
    yield warm, calls
    warm.join_background_warm(5)


_FEATURES = ("hardware", "text", "datasets", "media", "quantization")
_ALL_SELECTIONS = [
    frozenset(feature for bit, feature in enumerate(_FEATURES) if mask & (1 << bit))
    for mask in range(32)
]


@pytest.mark.parametrize(
    "features",
    _ALL_SELECTIONS,
    ids = lambda features: "+".join(feature for feature in _FEATURES if feature in features)
    or "none",
)
def test_every_selection_runs_only_its_stages_and_dependencies(selected_warm, features):
    warm, calls = selected_warm
    expected = []
    if features & {"hardware", "text", "media", "quantization"}:
        expected.append("hardware")
    if features & {"text", "media", "quantization"}:
        expected.append("inference_backend")
    if "quantization" in features:
        expected.append("quantization")
    if "text" in features:
        expected.append("transformers")
    if "datasets" in features:
        expected.append("datasets")
    assert warm.start_background_warm(features = features) is bool(features)
    assert warm.join_background_warm(5) is True
    assert calls == expected
    assert warm.warm_status()["finished"] is bool(features)


def test_empty_selection_starts_no_worker(selected_warm):
    warm, calls = selected_warm
    assert warm.start_background_warm(features = set()) is False
    assert warm.warm_status()["started"] is False
    assert warm._thread is None
    assert not calls


def test_selection_is_captured_before_waiting_for_the_first_response(selected_warm, monkeypatch):
    warm, calls = selected_warm
    monkeypatch.setattr(warm, "START_GATE_MAX_WAIT_S", 10)
    gate = threading.Event()
    selection = {"text"}
    assert warm.start_background_warm(gate = gate, features = selection) is True
    selection.update({"datasets", "media", "quantization"})
    gate.set()
    assert warm.join_background_warm(5) is True
    assert calls == ["hardware", "inference_backend", "transformers"]


def test_legacy_explicit_call_keeps_the_full_warm(selected_warm):
    warm, calls = selected_warm
    assert warm.start_background_warm() is True
    assert warm.join_background_warm(5) is True
    assert calls == ["hardware", "inference_backend", "quantization", "transformers", "datasets"]


def test_legacy_operator_disable_overrides_every_selection(selected_warm, monkeypatch):
    warm, calls = selected_warm
    monkeypatch.setenv(warm.DISABLE_ENV_VAR, "1")
    assert warm.start_background_warm(features = {"text", "quantization", "datasets"}) is False
    assert not calls


@pytest.mark.parametrize("has_model", [False, True])
def test_quantization_fallback_is_independent_of_early_timing_and_diffusers(
    selected_warm, monkeypatch, has_model
):
    warm, calls = selected_warm
    monkeypatch.setenv(warm.EARLY_PROBE_ENV_VAR, "0")
    monkeypatch.setenv(warm.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", lambda: has_model)
    monkeypatch.setattr(warm, "_prewarm_quant_probe", lambda: calls.append("probe"))
    warm.prewarm_quant_probe_if_image_models_exist()
    assert calls == (["probe"] if has_model else [])


def test_quantization_fallback_honors_global_disable(selected_warm, monkeypatch):
    warm, calls = selected_warm
    monkeypatch.setenv(warm.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(
        warm,
        "_a_local_model_would_load_through_diffusers",
        lambda: pytest.fail("disabled quantization scanned local models"),
    )
    monkeypatch.setattr(warm, "_prewarm_quant_probe", lambda: calls.append("probe"))
    warm.prewarm_quant_probe_if_image_models_exist()
    assert not calls


def test_restart_hands_the_new_selection_to_the_successor(selected_warm, monkeypatch):
    warm, calls = selected_warm
    epoch = [7]
    entered = threading.Event()
    release = threading.Event()

    def hardware():
        calls.append("hardware")
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(warm, "_detection_epoch", lambda: epoch[0])
    monkeypatch.setattr(
        warm,
        "_STAGES",
        (
            ("hardware", hardware),
            ("transformers", lambda: calls.append("transformers")),
            ("datasets", lambda: calls.append("datasets")),
        ),
    )
    assert warm.start_background_warm(features = {"text"})
    try:
        assert entered.wait(5)
        epoch[0] = 8
        assert warm.start_background_warm(features = {"datasets"})
    finally:
        release.set()
    assert warm.join_background_warm(5)
    assert calls == ["hardware", "datasets"]


def test_a_retired_warm_does_not_launch_its_selected_quantization_check(selected_warm, monkeypatch):
    warm, calls = selected_warm
    epoch = [7]
    monkeypatch.setattr(warm, "_detection_epoch", lambda: epoch[0])

    def retire_during_import():
        calls.append("inference_backend")
        epoch[0] += 1

    monkeypatch.setattr(
        warm,
        "_STAGES",
        (
            ("hardware", lambda: calls.append("hardware")),
            ("inference_backend", retire_during_import),
        ),
    )
    assert warm.start_background_warm(features = {"quantization"})
    assert warm.join_background_warm(5)
    assert calls == ["hardware", "inference_backend"]
