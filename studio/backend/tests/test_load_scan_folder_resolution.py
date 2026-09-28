# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repo id a registered local root holds must load from disk, not re-download into the hub."""

import core.inference.local_model_resolver as resolver
import routes.inference as inf
from models.inference import LoadRequest


def _request(**kwargs):
    kwargs.setdefault("model_path", "unsloth/Muse-Glimmer-30B-GGUF")
    return LoadRequest(**kwargs)


def test_repo_id_resolves_to_local_scan_folder_copy(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (
            "C:/scan-folder/models--x/snapshots/abc",
            "Q4_K_XL",
            "unsloth/Muse-Glimmer-30B-GGUF",
        )

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request())
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF"]
    assert rewritten.model_path == "C:/scan-folder/models--x/snapshots/abc"
    assert rewritten.gguf_variant == "Q4_K_XL"


def test_pinned_variant_resolves_by_repo_and_quant(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return ("C:/models/snapshots/abc", "Q8_0", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request(gguf_variant = "Q8_0"))
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:Q8_0"]
    assert rewritten.model_path == "C:/models/snapshots/abc"
    assert rewritten.gguf_variant == "Q8_0"


def test_local_miss_keeps_the_request_untouched(monkeypatch):
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda wanted, **kwargs: None)
    original = _request()
    assert inf._as_local_scan_folder_request(original) is original


def test_bare_name_stays_remote_for_the_hub(monkeypatch):
    def fail(wanted, **kwargs):
        raise AssertionError("resolver consulted for a bare name")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    request = _request(model_path = "Muse-Glimmer-30B-GGUF")
    assert inf._as_local_scan_folder_request(request) is request


def test_resolved_variant_replaces_the_requested_one(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        # A legacy alias (e.g. a retired quant label) resolves to the current on-disk quant.
        return ("C:/models/snapshots/abc", "Q4_K_XL", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request(gguf_variant = "BF16"))
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:BF16"]
    assert rewritten.gguf_variant == "Q4_K_XL"


def test_paths_and_manifest_refs_never_consult_the_resolver(monkeypatch):
    def fail(wanted, **kwargs):
        raise AssertionError("resolver consulted for a non-repo-id path")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    for path in (
        "C:/scan-folder/models--x/snapshots/abc",
        "/home/me/model.gguf",
        "ollama-manifest:sha256:deadbeef",
    ):
        request = _request(model_path = path)
        assert inf._as_local_scan_folder_request(request) is request


def test_resolver_failure_or_ollama_target_keeps_the_request(monkeypatch):
    def boom(wanted, **kwargs):
        raise OSError("scan root unreadable")

    monkeypatch.setattr(resolver, "resolve_local_gguf", boom)
    original = _request()
    assert inf._as_local_scan_folder_request(original) is original

    monkeypatch.setattr(
        resolver,
        "resolve_local_gguf",
        lambda wanted, **kwargs: ("ollama-manifest:sha256:deadbeef", None, "loader"),
    )
    tagged = _request()
    assert inf._as_local_scan_folder_request(tagged) is tagged


def test_managed_account_keeps_the_requested_id(monkeypatch):
    def fail(wanted, **kwargs):
        raise AssertionError("resolver consulted for a managed account")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    original = _request()
    assert inf._as_local_scan_folder_request(original) is original


def test_native_path_lease_keeps_the_requested_id(monkeypatch):
    def fail(wanted, **kwargs):
        raise AssertionError("resolver consulted under a native lease")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    original = _request(native_path_lease = "lease-token")
    assert inf._as_local_scan_folder_request(original) is original
