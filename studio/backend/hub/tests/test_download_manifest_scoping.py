# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.utils import download_manifest, state_dir


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
    """A cache path whose resolved and unresolved spellings differ.

    A POSIX symlink stands in for the Windows shapes that make ``resolve``
    change a path: a directory junction, a OneDrive redirect, a mapped drive,
    an 8.3 short name. Returns (as the caller spells it, what it resolves to).
    """
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

    # Unowned legacy manifest -> belongs to the active cache.
    legacy = state_dir.manifest_path("model", "Org/Model")
    _write_manifest(legacy, {"version": 1})
    # The inactive cache's own scoped copy is the one being deleted.
    scoped = state_dir.manifest_path("model", "Org/Model", hub_cache = str(previous))
    _write_manifest(scoped, {"version": 1, "hub_cache": str(previous)})

    removed = download_manifest.purge_state("model", "Org/Model", hub_cache = str(previous))

    assert removed is True
    assert not scoped.is_file()  # the inactive cache's copy is gone
    assert legacy.is_file()  # the active cache's legacy state survives


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
    assert not legacy.is_file()  # owned by the deleted cache -> purged


def test_scope_digest_is_shared_with_the_ownership_canonicalization(monkeypatch, tmp_path):
    """cache_scope_name and _canonical_hub_cache must normalize identically.

    They did not: the digest skipped ``resolve``, so a caller that reached
    state_dir with its own spelling of a redirected cache filed state under one
    digest while every reader that had gone through _canonical_hub_cache looked
    under another. On Windows, where junctions, OneDrive redirects and 8.3 short
    names make the two spellings diverge routinely, that is a finished download
    whose manifest can never be found again.
    """
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
    """A digest change must not orphan state an earlier build already wrote.

    State written when the digest hashed the unresolved spelling -- or when
    ``resolve`` failed for a OneDrive placeholder and normalize_hub_cache
    degraded to it -- sits under legacy_cache_scope_name. Readers probe it after
    the canonical one, so the manifest survives the migration.
    """
    spelled, _resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(tmp_path / "other")),
    )

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
    """One reader finding state that another cannot is worse than neither finding it.

    Four code paths answer "what state does this repo have": the per-triple
    read, the per-repo variant enumeration, the one-pass index the inventory
    scan is built on, and the delete. They all derive their scope directories
    from cache_scope_names, so a manifest under the pre-resolve digest is either
    visible to all of them or to none. A reader that saw it while the delete did
    not would resurrect a purged variant on the next scan, and an index that
    missed it while the per-variant endpoint saw it would have the list view and
    the detail view disagree about the same quant.
    """
    spelled, _resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(tmp_path / "other")),
    )
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
    """A marker the read side can find has to be one the clear side can remove.

    has_cancel_marker suppresses a completed download, so a marker discoverable
    under the pre-resolve digest but not clearable there would pin a finished
    variant to partial for good.
    """
    spelled, _resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(tmp_path / "other")),
    )
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
    """purge_all_state_for_repo reaches state_dir with the caller's own spelling.

    The delete routes pass the scanned cache root straight through, so before
    the digest shared one canonicalization this globbed a scope directory that
    never existed and every variant manifest survived the delete.
    """
    spelled, _resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(tmp_path / "other")),
    )

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
    """Restart-after-completion on the Windows copy layout still reads its manifest.

    No symlinks anywhere (blobs are copied into the snapshot dir, as HF does
    when the filesystem denies symlink creation), a case-skewed spelling of the
    cache on the second run, and the state root rebuilt from scratch: the
    manifest written by the first run has to be the one the second run finds.
    """
    hub_cache = tmp_path / "Hub"
    snapshot = hub_cache / "models--Org--Model" / "snapshots" / "rev0"
    snapshot.mkdir(parents = True)
    (snapshot / "model.gguf").write_bytes(b"x" * 16)  # a copy, not a symlink
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

    # Second run: the same directory reached as the parent of a scanned entry.
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
    """The two halves of the resolve-failed pair have to agree on one spelling.

    normalize_hub_cache degrades to the expanded spelling when ``resolve``
    refuses, and legacy_cache_scope_name is what recovers state written in that
    state. If the degraded branch skipped expanduser while the probe applied it,
    a "~"-spelled cache would file state under a digest no reader could rebuild
    -- and its recorded ownership is not absolute, so nothing else could
    attribute it either.
    """
    spellings = ["~/hf-hub", str(tmp_path / "hub") + "/", str(tmp_path / "hub" / "." / "x")]

    def _refuse(self, strict = False):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(Path, "resolve", _refuse)
    for spelling in spellings:
        assert state_dir.cache_scope_name(spelling) == state_dir.legacy_cache_scope_name(spelling)


def test_expanduser_failure_does_not_escape_a_plain_read(monkeypatch, tmp_path):
    """A homeless "~" must not turn read_manifest into a RuntimeError.

    legacy_cache_scope_name is fed the caller's raw spelling now, so it sees
    values the canonical path had already expanded away.
    """

    def _refuse(self):
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(Path, "expanduser", _refuse)

    assert state_dir.cache_scope_names("~/hf-hub")
    assert download_manifest.read_manifest("model", "Org/Model", hub_cache = "~/hf-hub") is None
