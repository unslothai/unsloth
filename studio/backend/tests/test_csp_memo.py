# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The nonce-free Content-Security-Policy is built once per set of inputs, and any input change rebuilds it."""

import pytest


@pytest.fixture
def main_module(monkeypatch):
    import main as _main

    monkeypatch.setattr(_main, "_CSP_CACHE", {})
    for name in ("HF_ENDPOINT", "HF_DATASETS_SERVER"):
        monkeypatch.delenv(name, raising = False)
    return _main


def _counting(monkeypatch, main_module):
    calls = []
    original = main_module._build_csp_uncached

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "_build_csp_uncached", spy)
    return calls, original


def test_repeat_responses_reuse_the_policy(monkeypatch, main_module):
    calls, original = _counting(monkeypatch, main_module)
    first = [main_module._build_csp(), main_module._build_csp(docs = True)]
    for _ in range(5):
        assert [main_module._build_csp(), main_module._build_csp(docs = True)] == first
    assert len(calls) == 2
    assert first == [original(), original(docs = True)]


def test_a_changed_endpoint_rebuilds_the_policy(monkeypatch, main_module):
    calls, original = _counting(monkeypatch, main_module)
    before = main_module._build_csp()
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.internal:8443")
    after = main_module._build_csp()
    assert after == original()
    assert "hub.internal:8443" in after and "hub.internal:8443" not in before
    monkeypatch.delenv("HF_ENDPOINT")
    assert main_module._build_csp() == before
    assert len(calls) == 2


def test_nonce_policies_are_never_kept(monkeypatch, main_module):
    calls, _ = _counting(monkeypatch, main_module)
    assert "'nonce-AAA'" in main_module._build_csp("AAA")
    assert "'nonce-BBB'" in main_module._build_csp("BBB")
    assert len(calls) == 2
    assert main_module._CSP_CACHE == {}


def test_a_settings_save_during_the_build_is_not_cached_under_the_old_key(monkeypatch, main_module):
    original = main_module._build_csp_uncached

    def build_while_a_save_lands(*args, **kwargs):
        policy = original(*args, **kwargs)
        monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.com")
        return policy

    monkeypatch.setattr(main_module, "_build_csp_uncached", build_while_a_save_lands)
    main_module._build_csp()
    monkeypatch.setattr(main_module, "_build_csp_uncached", original)
    monkeypatch.delenv("HF_ENDPOINT")
    assert main_module._CSP_CACHE == {}
    assert main_module._build_csp() == original()
