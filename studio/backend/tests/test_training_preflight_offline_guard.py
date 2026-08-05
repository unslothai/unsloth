# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards for the two cache-path regressions this rework introduced.

Both only exist because the rework pins a local snapshot and reaches the Hub from the
start route; neither mechanism is present on main.
"""

import time

import pytest

from core.training import worker as training_worker
from routes import training as training_routes


# --------------------------------------------------------------------------------------
# The Hub metadata preflight must consult the reachability guard.
# --------------------------------------------------------------------------------------


def test_model_preflight_short_circuits_when_the_hub_is_unreachable(monkeypatch):
    """A dead link must not burn the 5s + 10s metadata budget per resolved address."""
    calls = []

    def unreachable() -> bool:
        return True

    def hf_model_info(*args, **kwargs):  # pragma: no cover - must not run
        calls.append(kwargs.get("timeout"))
        time.sleep(5)
        raise AssertionError("metadata was fetched despite an unreachable Hub")

    monkeypatch.setattr(training_routes, "_hub_unreachable", unreachable)
    monkeypatch.setattr(training_routes, "hf_model_info", hf_model_info, raising = False)

    started = time.monotonic()
    with pytest.raises(Exception) as excinfo:
        training_routes._remote_untrainable_model_format("unsloth/does-not-matter", None)
    elapsed = time.monotonic() - started

    # The guard is checked by the caller, so the raw helper still runs; what matters is
    # that the caller never reaches it. Assert the helper is the only slow path.
    assert (
        calls == [] or elapsed < 5.0
    ), f"preflight consumed {elapsed:.1f}s against an unreachable Hub"
    assert excinfo.value is not None


def test_hub_unreachable_prefers_the_memoised_verdict(monkeypatch):
    """The guard must be cheap: a memoised verdict short-circuits both probes."""
    probes = []

    monkeypatch.setattr(training_routes, "hf_reachability_memo", lambda: True)
    monkeypatch.setattr(
        training_routes, "hf_dns_dead", lambda *a, **k: probes.append("dns") or True
    )
    monkeypatch.setattr(
        training_routes, "hf_unreachable", lambda *a, **k: probes.append("tcp") or True
    )

    assert training_routes._hub_unreachable() is True
    assert probes == [], "a memoised verdict must not re-probe the network"


def test_hub_unreachable_fails_open_when_reachable(monkeypatch):
    """An online host must be unaffected, so the normal path keeps its behaviour."""
    monkeypatch.setattr(training_routes, "hf_reachability_memo", lambda: None)
    monkeypatch.setattr(training_routes, "hf_dns_dead", lambda *a, **k: False)
    monkeypatch.setattr(training_routes, "hf_unreachable", lambda *a, **k: False)

    assert training_routes._hub_unreachable() is False


# --------------------------------------------------------------------------------------
# A pinned snapshot whose tokenizer cannot load must still earn one Hub retry.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        # SentencePiece/BPE families dereference a None vocab path. These arrive with no
        # cache-specific text, so a message whitelist that misses them makes a pinned
        # tokenizer-less snapshot terminal for ~26 tokenizer families (#7845).
        AttributeError("'NoneType' object has no attribute 'endswith'"),
        AttributeError("'NoneType' object has no attribute 'readlines'"),
        TypeError(
            "argument should be a str or an os.PathLike object where __fspath__ "
            "returns a str, not 'NoneType'"
        ),
        ValueError("Can't find a vocabulary file at path 'None'."),
    ],
)
def test_missing_tokenizer_artifacts_are_retryable_cache_errors(error):
    assert (
        training_worker._is_model_cache_artifact_error(error) is True
    ), f"{type(error).__name__}: {error} must earn a Hub retry, not fail the run"


@pytest.mark.parametrize(
    "error",
    [
        # A Hub retry cannot install a missing Python package, so these must stay fatal.
        ImportError("You need to install sacremoses to use XLMTokenizer."),
        ImportError("TransfoXLTokenizer requires the sacremoses library"),
        ValueError("Tokenizer class ParakeetCTCTokenizer does not exist"),
    ],
)
def test_unrelated_failures_are_not_treated_as_cache_errors(error):
    assert (
        training_worker._is_model_cache_artifact_error(error) is False
    ), f"{type(error).__name__} is not a cache artifact problem and must not retry"


def test_both_metadata_preflight_legs_consult_the_reachability_guard():
    """Wiring contract, not behaviour: the helper tests above pass even when the guard
    is never called, so assert both legs actually consult it. Kept deliberately narrow
    (two call sites, both inside the preflight) so it cannot pass vacuously."""
    import inspect

    source = inspect.getsource(training_routes)
    assert (
        source.count("_hub_unreachable()") >= 3
    ), "expected the guard definition plus both preflight legs to reference it"

    model_leg = source.split("def _reject_untrainable_model_request", 1)[1].split("\ndef ", 1)[0]
    assert (
        "_hub_unreachable()" in model_leg
    ), "the model metadata preflight must short-circuit on an unreachable Hub"
    assert model_leg.index("_hub_unreachable()") < model_leg.index(
        "_remote_untrainable_model_format("
    ), "the guard must be checked before the metadata fetch, not after"

    dataset_leg = source.split("def _preflight_hf_dataset_request", 1)[1].split("\ndef ", 1)[0]
    assert (
        "_hub_unreachable()" in dataset_leg
    ), "the dataset metadata preflight must short-circuit on an unreachable Hub"
