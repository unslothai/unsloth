# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repo id a registered local root holds must load from disk, not re-download into the hub."""

from pathlib import Path

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


def test_a_managed_account_session_keeps_the_requested_id(monkeypatch):
    # Managed accounts keep remote semantics: the rewrite consults the resolver never.
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
    monkeypatch.setattr(inf, "_rewrite_local_session", lambda fastapi_request: True)
    result = asyncio.run(inf.validate_model(ValidateModelRequest(model_path = _REPO), None, "owner"))
    assert result.valid and result.is_gguf
    # The call site gates on the rewrite predicate, mocked True here: a locally managed
    # session is served offline. That an API key qualifies is pinned by
    # test_the_rewrite_serves_api_key_sessions_too; the owner-only warning routing below
    # does not gate the rewrite.
    monkeypatch.setattr(inf, "_owner_session", lambda fastapi_request: False)
    assert asyncio.run(
        inf.validate_model(ValidateModelRequest(model_path = _REPO), None, "owner")
    ).valid
    # A managed account may not read the owner's copy, so it keeps the remote refusal.
    monkeypatch.setattr(inf, "_rewrite_local_session", lambda fastapi_request: False)
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


_MUSE_FILE_ID = "unsloth/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
_PROBE_FILE_ID = f"{_REPO}/Tiny-Probe-UD-Q5_K_XL.gguf"


def test_repo_file_id_resolves_to_local_scan_folder_copy(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (_SNAPSHOT, "UD-Q4_K_XL", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    monkeypatch.setattr(inf, "_load_dir_offers_file", lambda load_dir, name, variant: True)
    rewritten = inf._as_local_scan_folder_request(_request(model_path = _MUSE_FILE_ID), True)
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL"]
    assert rewritten.model_path == _SNAPSHOT
    assert rewritten.gguf_variant == "UD-Q4_K_XL"


def test_repo_file_id_prefers_the_file_over_a_pinned_variant(monkeypatch):
    seen = []

    def fake_resolve(wanted, **kwargs):
        seen.append(wanted)
        return (_SNAPSHOT, "UD-Q4_K_XL", "loader")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fake_resolve)
    monkeypatch.setattr(inf, "_load_dir_offers_file", lambda load_dir, name, variant: True)
    request = _request(model_path = _MUSE_FILE_ID, gguf_variant = "Q8_0")
    rewritten = inf._as_local_scan_folder_request(request, True)
    assert seen == ["unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL"]
    assert rewritten.gguf_variant == "UD-Q4_K_XL"


def test_repo_file_id_local_miss_keeps_the_request_untouched(monkeypatch):
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda wanted, **kwargs: None)
    original = _request(model_path = _MUSE_FILE_ID)
    assert inf._as_local_scan_folder_request(original, True) is original


def test_repo_file_id_not_in_snapshot_stays_remote(monkeypatch):
    monkeypatch.setattr(
        resolver, "resolve_local_gguf", lambda wanted, **kwargs: (_SNAPSHOT, "UD-Q4_K_XL", "loader")
    )
    monkeypatch.setattr(inf, "_load_dir_offers_file", lambda load_dir, name, variant: False)
    original = _request(model_path = _MUSE_FILE_ID)
    assert inf._as_local_scan_folder_request(original, True) is original


def test_repo_file_id_managed_account_guard_applies(monkeypatch):
    def fail(wanted, **kwargs):
        pytest.fail("resolver consulted for a managed account session")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    request = _request(model_path = _MUSE_FILE_ID)
    assert inf._as_local_scan_folder_request(request, False) is request


def test_repo_file_id_scan_folder_copy_loads_from_disk(roots):
    _active, scan = roots
    snapshot = _cache_repo(scan, ["Tiny-Probe-UD-Q5_K_XL.gguf"])
    rewritten = inf._as_local_scan_folder_request(_request(model_path = _PROBE_FILE_ID), True)
    assert (rewritten.model_path, rewritten.gguf_variant) == (str(snapshot), "UD-Q5_K_XL")
    # A stale file id (the quant exists, the file name does not) must not load a different file.
    stale = f"{_REPO}/Tiny-Probe-Renamed-UD-Q5_K_XL.gguf"
    untouched = inf._as_local_scan_folder_request(_request(model_path = stale), True)
    assert (untouched.model_path, untouched.gguf_variant) == (stale, None)


def test_repo_file_id_companion_name_does_not_load_a_quant(roots):
    # A companion name can carry a real quant token (mmproj-…-F16 -> F16): it must not rewrite
    # to the weights file of that quant, which the picker never offered under this name.
    _active, scan = roots
    _cache_repo(scan, ["Tiny-Probe-F16.gguf", "mmproj-Tiny-Probe-F16.gguf"])
    companion = f"{_REPO}/mmproj-Tiny-Probe-F16.gguf"
    untouched = inf._as_local_scan_folder_request(_request(model_path = companion), True)
    assert (untouched.model_path, untouched.gguf_variant) == (companion, None)


def test_repo_file_id_uppercase_suffix_resolves(roots):
    _active, scan = roots
    snapshot = _cache_repo(scan, ["Tiny-Probe-UD-Q5_K_XL.gguf"])
    rewritten = inf._as_local_scan_folder_request(
        _request(model_path = f"{_REPO}/Tiny-Probe-UD-Q5_K_XL.GGUF"), True
    )
    assert (rewritten.model_path, rewritten.gguf_variant) == (str(snapshot), "UD-Q5_K_XL")


def test_repo_file_id_native_path_lease_keeps_the_requested_id(monkeypatch):
    def fail(wanted, **kwargs):
        pytest.fail("resolver consulted under a native lease")

    monkeypatch.setattr(resolver, "resolve_local_gguf", fail)
    request = _request(model_path = _MUSE_FILE_ID, native_path_lease = "lease-token")
    assert inf._as_local_scan_folder_request(request, True) is request


def test_repo_file_id_requires_a_gguf_suffix(monkeypatch):
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda wanted, **kwargs: None)
    original = _request(model_path = "unsloth/Muse-Glimmer-30B-GGUF/config.json")
    assert inf._as_local_scan_folder_request(original, True) is original


def test_the_rewrite_serves_api_key_sessions_too(monkeypatch):
    # An API key no longer disqualifies the session: the hub-cache path already serves
    # local copies to those clients, so the scan-folder copy is the same trust boundary.
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: False)
    assert inf._rewrite_local_session(object()) is True
    assert inf._rewrite_local_session(None) is False


def test_managed_account_sessions_keep_remote_semantics(monkeypatch):
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    assert inf._rewrite_local_session(object()) is False


def test_the_load_and_validate_sites_use_the_rewrite_predicate():
    src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    rewrite_calls = [
        line
        for line in src.splitlines()
        if "_as_local_scan_folder_request, request, _rewrite_local_session(fastapi_request)" in line
    ]
    assert len(rewrite_calls) == 2, "both rewrite call sites must gate on _rewrite_local_session"
    # Warning routing stays tied to the owner session, not to the rewrite predicate.
    assert "_caller_is_owner = _owner_session(fastapi_request)" in src
