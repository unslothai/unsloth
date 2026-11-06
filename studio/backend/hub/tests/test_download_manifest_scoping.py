# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.utils import download_manifest, state_dir


def _shared_setup_1(monkeypatch, spelled, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )


def _shared_setup_2(monkeypatch, tmp_path):
    spelled, _resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(tmp_path / "other")),
    )
    return spelled


def _write_manifest(path, payload):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


def _manifest_payload(
    repo_id,
    variant,
    hub_cache,
    *,
    size = 4,
):
    return {
        "version": 1,
        "repo_type": "model",
        "repo_id": repo_id,
        "variant": variant,
        "started_at": "2026-01-01T00:00:00+00:00",
        "expected_files": [{"path": "model.gguf", "size": size}],
        "transport": "http",
        "hub_cache": hub_cache,
    }


def _redirected_hub_cache(tmp_path):
    """Symlinked hub cache whose spelled and resolved paths differ, standing in for Windows redirects."""
    target = tmp_path / "resolved" / "hub"
    target.mkdir(parents = True)
    link = tmp_path / "redirected"
    try:
        link.symlink_to(tmp_path / "resolved", target_is_directory = True)
    except (NotImplementedError, OSError):  # pragma: no cover - unprivileged Windows
        pytest.skip("symlinks unavailable on this host")
    return link / "hub", target


def test_purge_state_preserves_active_legacy_when_deleting_inactive_cache(monkeypatch, tmp_path):
    """A scoped delete of an inactive cache must not erase the unscoped legacy
    state, which _legacy_state_applies attributes to the active cache."""
    active = tmp_path / "active" / "hub"
    previous = tmp_path / "previous" / "hub"
    for path in (active, previous):
        path.mkdir(parents = True)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(active)),
    )

    legacy = state_dir.manifest_path("model", "Org/Model")
    _write_manifest(legacy, {"version": 1})
    scoped = state_dir.manifest_path("model", "Org/Model", hub_cache = str(previous))
    _write_manifest(scoped, {"version": 1, "hub_cache": str(previous)})

    removed = download_manifest.purge_state("model", "Org/Model", hub_cache = str(previous))

    assert removed is True
    assert not scoped.is_file()
    assert legacy.is_file()


def test_purge_state_removes_legacy_owned_by_the_deleted_cache(monkeypatch, tmp_path):
    """A legacy file that recorded the deleted cache as its owner is purged."""
    active = tmp_path / "active" / "hub"
    previous = tmp_path / "previous" / "hub"
    for path in (active, previous):
        path.mkdir(parents = True)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(active)),
    )

    legacy = state_dir.manifest_path("model", "Org/Model")
    _write_manifest(legacy, {"version": 1, "hub_cache": str(previous)})

    removed = download_manifest.purge_state("model", "Org/Model", hub_cache = str(previous))

    assert removed is True
    assert not legacy.is_file()


def test_scope_digest_is_shared_with_the_ownership_canonicalization(monkeypatch, tmp_path):
    """cache_scope_name must normalize like _canonical_hub_cache, or redirected caches split their state."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    assert spelled != resolved
    assert state_dir.cache_scope_name(spelled) == state_dir.cache_scope_name(resolved)
    assert state_dir.cache_scope_name(spelled) == state_dir._cache_scope_digest(
        download_manifest._canonical_hub_cache(spelled)
    )
    assert state_dir.manifest_path(
        "model", "Org/Model", hub_cache = spelled
    ) == state_dir.manifest_path("model", "Org/Model", hub_cache = resolved)


def test_manifest_under_the_pre_resolve_digest_is_still_found(monkeypatch, tmp_path):
    """Manifests written under the pre-resolve digest stay readable through legacy_cache_scope_name."""
    spelled = _shared_setup_2(monkeypatch, tmp_path)

    legacy_scope = state_dir.legacy_cache_scope_name(spelled)
    assert legacy_scope != state_dir.cache_scope_name(spelled)
    orphan = state_dir.manifest_path(
        "model",
        "Org/Model",
        "Q4_K_M",
        hub_cache = spelled,
        cache_scope = legacy_scope,
    )
    _write_manifest(orphan, _manifest_payload("Org/Model", "Q4_K_M", str(spelled)))

    manifest = download_manifest.read_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        hub_cache = spelled,
    )

    assert manifest is not None
    assert manifest.expected_files[0].path == "model.gguf"


def _legacy_scoped_variant_manifest(
    tmp_path,
    spelled,
    variant = "Q4_K_M",
):
    """Plant a variant manifest under the pre-resolve digest, as an old build would."""
    path = state_dir.manifest_path(
        "model",
        "Org/Model",
        variant,
        hub_cache = spelled,
        cache_scope = state_dir.legacy_cache_scope_name(spelled),
    )
    assert path.parent.name != state_dir.cache_scope_name(spelled)
    _write_manifest(path, _manifest_payload("Org/Model", variant, str(spelled)))
    return path


def test_every_enumerator_agrees_about_the_pre_resolve_digest(monkeypatch, tmp_path):
    """Every state enumerator must see the pre-resolve digest, or reads and deletes disagree."""
    spelled = _shared_setup_2(monkeypatch, tmp_path)
    orphan = _legacy_scoped_variant_manifest(tmp_path, spelled)

    assert (
        download_manifest.read_manifest("model", "Org/Model", "Q4_K_M", hub_cache = spelled)
        is not None
    )
    assert [
        variant
        for variant, _path in download_manifest.iter_variant_manifests(
            "model", "Org/Model", hub_cache = spelled
        )
    ] == ["Q4_K_M"]
    index = download_manifest.build_variant_state_index(
        [("model", "Org/Model", spelled)],
        active_hub_cache = spelled,
    )
    state = index.for_repo("model", "Org/Model", hub_cache = spelled)
    assert state.manifest_for("Q4_K_M") is not None
    assert download_manifest.purge_all_state_for_repo("model", "Org/Model", hub_cache = spelled)
    assert not orphan.is_file()


def test_pre_resolve_digest_cancel_marker_is_cleared_by_a_new_attempt(monkeypatch, tmp_path):
    """A cancel marker found under the pre-resolve digest must also be clearable there, or it sticks."""
    spelled = _shared_setup_2(monkeypatch, tmp_path)
    marker = state_dir.marker_path(
        "model",
        "Org/Model",
        "Q4_K_M",
        hub_cache = spelled,
        cache_scope = state_dir.legacy_cache_scope_name(spelled),
    )
    _write_manifest(
        marker,
        {
            "version": 2,
            "repo_type": "model",
            "repo_id": "Org/Model",
            "variant": "Q4_K_M",
            "transport": "http",
            "cancelled_at": "2026-01-01T00:00:00+00:00",
            "hub_cache": str(spelled),
        },
    )

    assert download_manifest.has_cancel_marker("model", "Org/Model", "Q4_K_M", hub_cache = spelled)
    download_manifest.clear_cancel_marker("model", "Org/Model", "Q4_K_M", hub_cache = spelled)
    assert not marker.is_file()
    assert not download_manifest.has_cancel_marker(
        "model", "Org/Model", "Q4_K_M", hub_cache = spelled
    )


def test_repo_delete_clears_variant_state_under_a_redirected_cache(monkeypatch, tmp_path):
    """Repo delete must clear every variant manifest under a redirected cache, not only the spelled
    scope."""
    spelled = _shared_setup_2(monkeypatch, tmp_path)

    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model.gguf", size = 4)],
        "http",
        hub_cache = spelled,
    )
    written = state_dir.manifest_path("model", "Org/Model", "Q4_K_M", hub_cache = spelled)
    assert written.is_file()

    assert download_manifest.purge_all_state_for_repo("model", "Org/Model", hub_cache = spelled)
    assert not written.is_file()


def test_windows_shaped_copy_cache_scope_survives_a_restart(monkeypatch, tmp_path):
    """A Windows copy-layout cache with case-skewed spelling must still find its manifest after restart."""
    hub_cache = tmp_path / "Hub"
    snapshot = hub_cache / "models--Org--Model" / "snapshots" / "rev0"
    snapshot.mkdir(parents = True)
    (snapshot / "model.gguf").write_bytes(b"x" * 16)
    assert not (snapshot / "model.gguf").is_symlink()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(hub_cache)),
    )

    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model.gguf", size = 16)],
        "http",
        hub_cache = hub_cache,
    )

    entry = next(path for path in hub_cache.iterdir() if path.name.startswith("models--"))
    manifest = download_manifest.read_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        hub_cache = entry.parent,
    )
    assert manifest is not None
    assert download_manifest.verify_against_disk(manifest, snapshot).ok


def test_normalize_hub_cache_degrades_when_resolve_refuses(monkeypatch, tmp_path):
    """A path Windows can open but not resolve keeps a scope instead of losing one."""

    def _refuse(self, strict = False):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(Path, "resolve", _refuse)
    expected = os.path.normcase(str(tmp_path / "hub"))
    assert state_dir.normalize_hub_cache(tmp_path / "hub") == expected


def test_degraded_normalization_matches_its_own_recovery_probe(monkeypatch, tmp_path):
    """Degraded normalization must expanduser like the legacy recovery probe, or state is never found."""
    spellings = ["~/hf-hub", str(tmp_path / "hub") + "/", str(tmp_path / "hub" / "." / "x")]

    def _refuse(self, strict = False):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(Path, "resolve", _refuse)
    for spelling in spellings:
        assert state_dir.cache_scope_name(spelling) == state_dir.legacy_cache_scope_name(spelling)


def test_expanduser_failure_does_not_escape_a_plain_read(monkeypatch, tmp_path):
    """An undeterminable home directory must not make read_manifest raise RuntimeError on a raw ~ path."""

    def _refuse(self):
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(Path, "expanduser", _refuse)

    assert state_dir.cache_scope_names("~/hf-hub")
    assert download_manifest.read_manifest("model", "Org/Model", hub_cache = "~/hf-hub") is None


def _legacy_scoped_manifest(tmp_path, spelled, resolved, repo_id, variant):
    """Plant a manifest under the pre-resolve digest, where state lands when resolve fails at write time."""
    legacy = state_dir.manifest_path(
        "model",
        repo_id,
        variant,
        hub_cache = str(resolved),
        cache_scope = state_dir.legacy_cache_scope_name(str(spelled)),
    )
    _write_manifest(legacy, _manifest_payload(repo_id, variant, str(resolved)))
    assert state_dir.legacy_cache_scope_name(str(spelled)) != state_dir.cache_scope_name(
        str(spelled)
    ), "fixture is not exercising a split digest"
    return legacy


def test_repo_delete_clears_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """Production deletes pass an already-resolved root, so the legacy scope must still be cleared there."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    _shared_setup_1(monkeypatch, spelled, tmp_path)
    legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    removed = download_manifest.purge_all_state_for_repo(
        "model", "Org/Model", hub_cache = str(resolved)
    )

    assert removed > 0
    assert not legacy.is_file()
    assert download_manifest.read_manifest("model", "Org/Model", "Q4_K_M") is None


def test_variant_delete_clears_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """Variant delete with a resolved root must purge the legacy scope too, or the variant reappears."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    _shared_setup_1(monkeypatch, spelled, tmp_path)
    legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    removed = download_manifest.purge_state("model", "Org/Model", "Q4_K_M", hub_cache = str(resolved))

    assert removed is True
    assert not legacy.is_file()
    assert download_manifest.read_manifest("model", "Org/Model", "Q4_K_M") is None


def test_variant_index_sees_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """Variant index must probe the legacy scope for a resolved root, or cached views miss the variant."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    _shared_setup_1(monkeypatch, spelled, tmp_path)
    _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    index = download_manifest.build_variant_state_index(
        [("model", "Org/Model", str(resolved))],
        active_hub_cache = str(resolved),
    )
    state = index.for_repo("model", "Org/Model", hub_cache = str(resolved))

    assert state.manifest_for("Q4_K_M") is not None


def test_the_configured_spelling_is_only_borrowed_for_the_SAME_directory(monkeypatch, tmp_path):
    """Borrow the configured spelling only for the same directory, never sweeping another cache's state."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    other = tmp_path / "other" / "hub"
    other.mkdir(parents = True)
    _shared_setup_1(monkeypatch, spelled, tmp_path)
    active_legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")
    victim = state_dir.manifest_path("model", "Org/Model", "Q8_0", hub_cache = str(other))
    _write_manifest(victim, _manifest_payload("Org/Model", "Q8_0", str(other)))

    download_manifest.purge_all_state_for_repo("model", "Org/Model", hub_cache = str(other))

    assert not victim.is_file()
    assert active_legacy.is_file()


def test_disagreeing_manifests_across_caches_are_refused(monkeypatch, tmp_path):
    """Two caches with disagreeing manifests yield no manifest, so the name-based fallback takes over."""
    from hub.services.models import downloads
    from hub.utils import download_manifest

    old = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-01-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("old.gguf", 10, "aaa"),),
    )
    new = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-02-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("new.gguf", 20, "bbb"),),
    )
    first, second = tmp_path / "a" / "repo", tmp_path / "b" / "repo"
    served = {first.parent: old, second.parent: new}

    monkeypatch.setattr(downloads, "preferred_repo_cache_dirs", lambda *a, **k: [first, second])
    monkeypatch.setattr(download_manifest, "_canonical_hub_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        download_manifest,
        "read_manifest",
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            served.get(Path(hub_cache)) if hub_cache is not None else None
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None

    served[second.parent] = old
    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is old


def test_a_stale_active_manifest_is_compared_rather_than_returned(monkeypatch, tmp_path):
    """A stale active-cache manifest must be compared against remembered caches, not returned as-is."""
    from hub.services.models import downloads
    from hub.utils import download_manifest

    stale = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-01-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("old.gguf", 10, "aaa"),),
    )
    current = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-02-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("new.gguf", 20, "bbb"),),
    )
    remembered = tmp_path / "remembered" / "repo"

    monkeypatch.setattr(downloads, "preferred_repo_cache_dirs", lambda *a, **k: [remembered])
    monkeypatch.setattr(download_manifest, "_canonical_hub_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        download_manifest,
        "read_manifest",
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            stale if hub_cache is None else current
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None


def test_variant_enumeration_sees_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """local_path requests arrive resolved, so offline variant listing must probe the legacy scope too."""
    spelled, resolved = _redirected_hub_cache(tmp_path)
    _shared_setup_1(monkeypatch, spelled, tmp_path)
    _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    listed = dict(
        download_manifest.iter_variant_manifests("model", "Org/Model", hub_cache = str(resolved))
    )

    assert "Q4_K_M" in listed, (
        "the resolved spelling lost the legacy scope, so an offline listing cannot see the "
        "partial download it is meant to offer a resume for"
    )


def test_a_scanned_cache_with_no_manifest_refuses_the_others(monkeypatch, tmp_path):
    """A scanned cache with no manifest must refuse the others, not borrow their hashes for its blobs."""
    from pathlib import Path

    from hub.services.models import downloads
    from hub.utils import download_manifest

    only = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-01-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("old.gguf", 10, "aaa"),),
    )
    first, second = tmp_path / "a" / "repo", tmp_path / "b" / "repo"
    served: dict = {first.parent: only, second.parent: None}

    monkeypatch.setattr(downloads, "preferred_repo_cache_dirs", lambda *a, **k: [first, second])
    monkeypatch.setattr(download_manifest, "_canonical_hub_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        download_manifest,
        "read_manifest",
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            served.get(Path(hub_cache)) if hub_cache is not None else None
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None

    served[second.parent] = only
    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is only


def test_the_active_cache_must_have_a_manifest_when_it_is_scanned(monkeypatch, tmp_path):
    """Same rule for the active cache: snapshot_progress scans it like any other."""
    from pathlib import Path

    from hub.services.models import downloads
    from hub.utils import download_manifest

    other = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        variant = "Q4_K_M",
        started_at = "2026-01-01T00:00:00Z",
        expected_files = (download_manifest.ExpectedFile("old.gguf", 10, "aaa"),),
    )
    active_repo, remembered = tmp_path / "active" / "repo", tmp_path / "b" / "repo"

    monkeypatch.setattr(
        downloads, "preferred_repo_cache_dirs", lambda *a, **k: [active_repo, remembered]
    )
    monkeypatch.setattr(
        download_manifest,
        "_canonical_hub_cache",
        lambda path = None: str(active_repo.parent)
        if path in (None, active_repo.parent)
        else str(path),
    )
    monkeypatch.setattr(
        download_manifest,
        "read_manifest",
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            None if hub_cache is None else other
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None


def test_an_unreadable_cache_root_is_unknown_rather_than_absent(monkeypatch, tmp_path):
    """An unlistable cache root is unknown, not absent: an OSError must not read as a wiped cache."""
    from hub.services import snapshot_progress
    from hub.utils import hf_cache_state

    unreadable = tmp_path / "hub"
    unreadable.mkdir()

    def _explode(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [unreadable])
    monkeypatch.setattr(hf_cache_state, "hf_cache_root", lambda root = None, **kw: None)
    monkeypatch.setattr(type(unreadable), "iterdir", _explode)

    errors: list = []
    assert (
        hf_cache_state.preferred_repo_cache_dirs("model", "unsloth/Model-GGUF", scan_errors = errors)
        == []
    )
    assert errors and isinstance(errors[0], OSError)

    class _Registry:
        def get_job(self, key):
            return SimpleNamespace(state = "idle")

    reading = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "unsloth/Model-GGUF",
        job_key = "model:unsloth/Model-GGUF",
        expected_bytes = 33_000_000_000,
        hf_token = None,
        registry = _Registry(),
        metadata_resolver = lambda *a, **k: (33_000_000_000, frozenset()),
    )
    assert "cache_path" not in reading
    assert reading["cache_measured"] is False
    from hub.schemas.downloads import DownloadProgressResponse

    # Must be declared on DownloadProgressResponse or FastAPI drops it.
    assert "cache_measured" in DownloadProgressResponse.__annotations__
    assert reading["downloaded_bytes"] == 0


def test_a_scope_whose_payload_is_lost_reads_back_as_a_digest(monkeypatch, tmp_path):
    """A scope with a lost payload must still read as a recognisable digest, old tag form included."""
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(hub_cache)),
    )

    download_manifest.write_cancel_marker(
        "model", "Org/Model", "@diffusion", transport = "xet", hub_cache = str(hub_cache)
    )
    ((variant, path),) = download_manifest.iter_variant_markers(
        "model", "Org/Model", hub_cache = str(hub_cache)
    )
    assert variant == "@diffusion"
    assert "--variant--@sha256-" in path.name

    payload = json.loads(path.read_text(encoding = "utf-8"))
    payload.pop("variant")
    for name, expected_tag in (
        (path.name, "@sha256-"),
        (path.name.replace("@sha256-", "sha256-"), "sha256-"),
    ):
        target = path.with_name(name)
        _write_manifest(target, payload)
        ((recovered, _),) = download_manifest.iter_variant_markers(
            "model", "Org/Model", hub_cache = str(hub_cache)
        )
        assert recovered.startswith(expected_tag)
        assert state_dir.variant_is_hashed_fragment(recovered)
        target.unlink()

    for quant in ("Q4_K_M", "UD-Q4_K_XL", "sha256-short"):
        assert not state_dir.variant_is_hashed_fragment(quant)


def test_a_variant_with_nothing_of_its_own_says_so(monkeypatch, tmp_path):
    """Sibling quants share a repo dir, so a variant with no files of its own must not look resumable."""
    from hub.services import snapshot_progress

    entry = tmp_path / "hub" / "models--unsloth--Model-GGUF"
    (entry / "blobs").mkdir(parents = True)
    (entry / "blobs" / "sibling").write_bytes(b"x" * 32)

    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", lambda *a, **k: [entry])

    class _Registry:
        def get_job(self, key):
            return SimpleNamespace(state = "idle")

    def _reading(variant, expected_hashes):
        return snapshot_progress.compute_snapshot_progress(
            repo_type = "model",
            repo_id = "unsloth/Model-GGUF",
            job_key = "model:unsloth/Model-GGUF",
            expected_bytes = 33_000_000_000,
            hf_token = None,
            registry = _Registry(),
            metadata_resolver = lambda *a, **k: (33_000_000_000, expected_hashes),
            variant = variant,
        )

    ours = _reading("Q4_K_M", frozenset({"ours"}))
    assert ours["downloaded_bytes"] == 0
    assert ours["cache_path"] is not None, "the sibling keeps the directory alive"
    assert ours["target_present"] is False
    from hub.schemas.downloads import DownloadProgressResponse

    assert "target_present" in DownloadProgressResponse.__annotations__

    unknown = _reading("Q4_K_M", frozenset())
    assert unknown["target_present"] is None

    assert _reading(None, frozenset())["target_present"] is None


def test_a_root_that_cannot_even_be_stat_ed_is_unknown(monkeypatch, tmp_path):
    """An unstatable cache root must be unknown, not a measured absence, or hydration retires the job."""
    import os as _os

    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    real_stat = _os.stat

    def _explode(path, *args, **kwargs):
        if str(path) == str(root):
            raise OSError(errno.ELOOP, "Too many levels of symbolic links")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(_os, "stat", _explode)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [])

    errors: list = []
    assert (
        hf_cache_state.preferred_repo_cache_dirs(
            "model", "unsloth/Model-GGUF", active_root = root, scan_errors = errors
        )
        == []
    )
    assert errors and isinstance(
        errors[0], OSError
    ), "a root we could not stat is not evidence that the cache was deleted"
