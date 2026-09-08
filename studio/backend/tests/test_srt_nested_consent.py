# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Controlled eligibility and token-custody seams, not platform qualification."""

import subprocess
from unittest.mock import Mock
import pytest
from core.inference import srt_nested, srt_probe
from core.inference.srt_diagnostics import ProbeReason
from core.inference.tool_isolation import LimitedGrantStore, LimitedGrantError


@pytest.mark.parametrize(
    "code",
    [
        "dependency_missing",
        "runtime_missing",
        "policy_invalid",
        "policy_oversized",
        "probe_timeout",
        "enforcement_failed",
        "probe_failed",
    ],
)
def test_unrelated_failures_never_offer_nested(monkeypatch, code):
    control = Mock(side_effect = AssertionError("no differential control warranted"))
    monkeypatch.setattr(srt_nested, "_proc_control", control)
    assert not srt_nested.eligibility((False, ProbeReason(code)))
    control.assert_not_called()


@pytest.mark.parametrize(
    "results,expected", [([1, 0], True), ([0], False), ([1, 1], False), ([-9], False)]
)
def test_only_proc_differential_offers_nested(monkeypatch, results, expected):
    monkeypatch.setattr(srt_nested, "container_context", lambda: True)
    monkeypatch.setattr(srt_nested.os, "access", lambda *a: True)
    control = Mock(side_effect = results)
    monkeypatch.setattr(srt_nested, "_proc_control", control)
    assert srt_nested.eligibility((False, ProbeReason("operation_unsupported"))) is expected
    assert control.call_args_list[0].kwargs == {"nested": False}
    if len(results) > 1:
        assert control.call_args_list[1].kwargs == {"nested": True}


def test_missing_bwrap_and_timeouts_cannot_offer_nested(monkeypatch):
    monkeypatch.setattr(srt_nested, "container_context", lambda: True)
    control = Mock(side_effect = subprocess.TimeoutExpired("owned control", 3))
    monkeypatch.setattr(srt_nested, "_proc_control", control)
    monkeypatch.setattr(srt_nested.os, "access", lambda *a: False)
    assert not srt_nested.eligibility((False, ProbeReason("operation_unsupported")))
    control.assert_not_called()
    monkeypatch.setattr(srt_nested.os, "access", lambda *a: True)
    assert not srt_nested.eligibility((False, ProbeReason("operation_unsupported")))


def test_nested_and_limited_tokens_are_not_interchangeable():
    nested = srt_nested.NestedGrantStore()
    limited = LimitedGrantStore()
    scope = {
        "current_subject": "actor",
        "tool_ui_session_id": "page",
        "probe_generation": "generation",
    }
    nested_grant = nested.issue(**scope)
    limited_grant = limited.issue(**scope)
    assert nested_grant.mode == "container_isolation"
    nested.validate(nested_grant.token, **scope)
    with pytest.raises(LimitedGrantError):
        nested.validate(limited_grant.token, **scope)
    with pytest.raises(LimitedGrantError):
        limited.validate(nested_grant.token, **scope, requested_mode = "limited")
    for changed in [
        {"current_subject": "other"},
        {"tool_ui_session_id": "reload"},
        {"probe_generation": "new-runtime"},
    ]:
        with pytest.raises(LimitedGrantError):
            nested.validate(nested_grant.token, **{**scope, **changed})


def test_probe_cache_is_partitioned_by_variant(monkeypatch):
    monkeypatch.setattr(srt_probe, "_cache", {})
    native = Mock(return_value = (True, "controlled probe"))
    monkeypatch.setattr(srt_probe, "_native_probe", native)
    srt_probe.probe()
    srt_probe.probe(isolation_variant = "nested")
    srt_probe.probe()
    srt_probe.probe(isolation_variant = "nested")
    assert native.call_count == 2
    assert native.call_args_list[1].kwargs["isolation_variant"] == "nested"
