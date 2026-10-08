# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repo id a registered local root holds must load from disk, not re-download into the hub."""

import pytest

import core.inference.local_model_resolver as resolver
import routes.inference as inf
from models.inference import LoadRequest


_SNAPSHOT = "C:/scan-folder/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/abc"


def _request(**kwargs):
    kwargs.setdefault("model_path", "unsloth/Muse-Glimmer-30B-GGUF")
    return LoadRequest(**kwargs)


def test_repo_id_resolves_to_local_scan_folder_copy(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (_SNAPSHOT, "Q4_K_XL", "unsloth/Muse-Glimmer-30B-GGUF")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request(), True)
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF"]
    assert rewritten.model_path == _SNAPSHOT
    assert rewritten.gguf_variant == "Q4_K_XL"


def test_pinned_variant_resolves_by_repo_and_quant(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (_SNAPSHOT, "Q8_0", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request(gguf_variant = "Q8_0"), True)
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:Q8_0"]
    assert rewritten.model_path == _SNAPSHOT
    assert rewritten.gguf_variant == "Q8_0"


def test_local_miss_keeps_the_request_untouched(monkeypatch):
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda wanted, **kwargs: None)
    original = _request()
    assert inf._as_local_scan_folder_request(original, True) is original


def test_bare_name_stays_remote_for_the_hub(monkeypatch):
    def fail(wanted, **kwargs):
        pytest.fail("resolver consulted for a bare name")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    request = _request(model_path = "Muse-Glimmer-30B-GGUF")
    assert inf._as_local_scan_folder_request(request, True) is request


def test_resolved_variant_replaces_the_requested_one(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (_SNAPSHOT, "Q4_K_XL", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    rewritten = inf._as_local_scan_folder_request(_request(gguf_variant = "BF16"), True)
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:BF16"]
    assert rewritten.gguf_variant == "Q4_K_XL"


def test_paths_and_manifest_refs_never_consult_the_resolver(monkeypatch):
    def fail(wanted, **kwargs):
        pytest.fail("resolver consulted for a non-repo-id path")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    for path in (
        "C:/scan-folder/models--x/snapshots/abc",
        "/home/me/model.gguf",
        "ollama-manifest:sha256:deadbeef",
    ):
        request = _request(model_path = path)
        assert inf._as_local_scan_folder_request(request, True) is request


def test_resolver_failure_or_ollama_target_keeps_the_request(monkeypatch):
    def boom(wanted, **kwargs):
        raise OSError("scan root unreadable")

    monkeypatch.setattr(resolver, "resolve_local_gguf", boom)
    original = _request()
    assert inf._as_local_scan_folder_request(original, True) is original

    monkeypatch.setattr(
        resolver,
        "resolve_local_gguf",
        lambda wanted, **kwargs: ("ollama-manifest:sha256:deadbeef", None, "loader"),
    )
    tagged = _request()
    assert inf._as_local_scan_folder_request(tagged, True) is tagged


def test_a_caller_other_than_the_owner_session_keeps_the_requested_id(monkeypatch):
    seen = []
    monkeypatch.setattr(
        resolver, "resolve_local_gguf", lambda wanted, **kwargs: seen.append(wanted)
    )
    original = _request()
    assert inf._as_local_scan_folder_request(original, False) is original
    assert seen == []


def test_native_path_lease_keeps_the_requested_id(monkeypatch):
    def fail(wanted, **kwargs):
        pytest.fail("resolver consulted under a native lease")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    original = _request(native_path_lease = "lease-token")
    assert inf._as_local_scan_folder_request(original, True) is original


# Real resolver index below (no mock).

_REPO = "unsloth/Tiny-Probe-GGUF"


def _cache_repo(root, files):
    repo_dir = root / ("models--" + _REPO.replace("/", "--"))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("abc")
    snapshot = repo_dir / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    for name in files:
        (snapshot / name).write_bytes(b"GGUF" + b"\0" * 64)
    return snapshot


@pytest.fixture
def roots(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    active = tmp_path / "hub"
    active.mkdir()
    scan = tmp_path / "scan"
    scan.mkdir()
    monkeypatch.setattr("routes.models._resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.hermes_model_dirs", lambda: [], raising = False)
    monkeypatch.setattr("utils.paths.ollama_model_dirs", lambda: [], raising = False)
    monkeypatch.setattr("utils.paths.omlx_model_dirs", lambda: [], raising = False)
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active])
    monkeypatch.chdir(tmp_path)
    connection = studio_db.get_connection()
    with connection:
        connection.execute(
            "INSERT INTO scan_folders (path, created_at) VALUES (?, datetime('now'))",
            (str(scan),),
        )
    connection.close()
    resolver.invalidate_index()
    yield active, scan
    resolver.invalidate_index()


def test_scan_folder_only_copy_loads_from_disk(roots):
    _active, scan = roots
    snapshot = _cache_repo(scan, ["Tiny-Probe-UD-Q5_K_XL.gguf"])
    rewritten = inf._as_local_scan_folder_request(_request(model_path = _REPO), True)
    assert (rewritten.model_path, rewritten.gguf_variant) == (str(snapshot), "UD-Q5_K_XL")
    pinned = inf._as_local_scan_folder_request(
        _request(model_path = _REPO, gguf_variant = "UD-Q5_K_XL"), True
    )
    assert pinned.model_path == str(snapshot)
    # A quant the copy does not hold stays remote rather than serving other weights.
    missing = inf._as_local_scan_folder_request(
        _request(model_path = _REPO, gguf_variant = "Q8_0"), True
    )
    assert (missing.model_path, missing.gguf_variant) == (_REPO, "Q8_0")


@pytest.mark.parametrize("in_scan_folder", [False, True])
def test_active_hub_cache_copy_keeps_the_repo_id(roots, in_scan_folder):
    active, scan = roots
    _cache_repo(active, ["Tiny-Probe-Q4_K_M.gguf", "Tiny-Probe-Q8_0.gguf"])
    if in_scan_folder:
        _cache_repo(scan, ["Tiny-Probe-Q4_K_M.gguf"])
    for variant in (None, "Q8_0"):
        request = _request(model_path = _REPO, gguf_variant = variant)
        assert inf._as_local_scan_folder_request(request, True) is request


def test_weightless_hub_skeleton_does_not_hide_the_scan_folder_copy(roots):
    active, scan = roots
    _cache_repo(active, [])
    snapshot = _cache_repo(scan, ["Tiny-Probe-Q4_K_M.gguf"])
    rewritten = inf._as_local_scan_folder_request(_request(model_path = _REPO), True)
    assert rewritten.model_path == str(snapshot)


def test_validate_reads_the_scan_folder_copy_offline(roots, monkeypatch):
    import asyncio

    from models.inference import ValidateModelRequest

    _active, scan = roots
    _cache_repo(scan, ["Tiny-Probe-Q4_K_M.gguf"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(inf, "_owner_session", lambda fastapi_request: True)
    result = asyncio.run(inf.validate_model(ValidateModelRequest(model_path = _REPO), None, "owner"))
    assert result.valid and result.is_gguf
    # An API key caller may not read the owner's copy, so it keeps the remote refusal.
    monkeypatch.setattr(inf, "_owner_session", lambda fastapi_request: False)
    with pytest.raises(inf.HTTPException):
        asyncio.run(inf.validate_model(ValidateModelRequest(model_path = _REPO), None, "owner"))


@pytest.mark.parametrize(
    "hit",
    [
        ("C:/lmstudio/models/unsloth/Muse-Glimmer-30B-GGUF", "Q4_K_XL", "loader"),
        (_SNAPSHOT, None, "loader"),
        ("C:/scan-folder/models--unsloth--Other-GGUF/snapshots/abc", "Q4_K_XL", "loader"),
    ],
    ids = ["non_hf_layout", "non_gguf", "other_repo"],
)
def test_only_a_gguf_snapshot_of_the_requested_repo_is_taken(monkeypatch, hit):
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda wanted, **kwargs: hit)
    request = _request()
    assert inf._as_local_scan_folder_request(request, True) is request


def test_scan_folder_nested_inside_the_hub_cache_still_loads_from_disk(roots, monkeypatch):
    from storage import studio_db

    active, _scan = roots
    nested = active / "archive"
    nested.mkdir()
    connection = studio_db.get_connection()
    with connection:
        connection.execute(
            "INSERT INTO scan_folders (path, created_at) VALUES (?, datetime('now'))",
            (str(nested),),
        )
    connection.close()
    resolver.invalidate_index()
    snapshot = _cache_repo(nested, ["Tiny-Probe-Q4_K_M.gguf"])
    rewritten = inf._as_local_scan_folder_request(_request(model_path = _REPO), True)
    assert rewritten.model_path == str(snapshot)


def test_a_rewritten_load_keeps_repo_keyed_chat_templates(roots, monkeypatch):
    import asyncio

    _active, scan = roots
    _cache_repo(scan, ["Tiny-Probe-Q4_K_M.gguf"])
    monkeypatch.setattr(inf, "_owner_session", lambda fastapi_request: True)
    seen = []

    class _Stop(BaseException):
        pass

    def _record(*, model_identifier, user_override):
        seen.append(model_identifier)
        raise _Stop

    monkeypatch.setattr(inf, "resolve_effective_chat_template_override", _record)
    with pytest.raises(_Stop):
        asyncio.run(inf._load_model_impl(_request(model_path = _REPO), None, "owner"))
    assert seen == [_REPO]
