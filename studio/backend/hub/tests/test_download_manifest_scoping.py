# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
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


def _legacy_scoped_manifest(tmp_path, spelled, resolved, repo_id, variant):
    """Plant a manifest under the PRE-resolve digest of ``spelled``.

    That is where state lands when ``resolve`` is unavailable at write time (a
    OneDrive placeholder, a locked junction), and where an 8.3 path's state
    lands until the directory it names exists. The reader recovers it; the
    point of these tests is that the delete and the index do too.
    """
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
    """The delete reaches purge_all_state_for_repo with an ALREADY-resolved root.

    Every production caller does: hub/services/models/deletion.py and
    hub/services/datasets/cache_inventory.py all pass
    resolve_delete_target_root(...), whose every branch calls .resolve(). So the
    raw-spelling probe inside cache_scope_names finds nothing here, while the
    read path -- fed the raw configured setting -- still has it. Left that way,
    a purged variant survives under the legacy digest and the next read brings
    it back, which is exactly the resurrection the scope fan-out exists to stop.
    """
    spelled, resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )
    legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    # The resolved spelling, as resolve_delete_target_root would hand it over.
    removed = download_manifest.purge_all_state_for_repo(
        "model", "Org/Model", hub_cache = str(resolved)
    )

    assert removed > 0
    assert not legacy.is_file()
    # ...and the read path agrees it is gone, rather than resurrecting it.
    assert download_manifest.read_manifest("model", "Org/Model", "Q4_K_M") is None


def test_variant_delete_clears_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """Same asymmetry, one variant at a time.

    The single-variant delete route reaches purge_state with the root
    resolve_delete_target_root already resolved, so the raw spelling reproduced the canonical
    digest and only that scope was probed. The read path still probed both, so the exact
    variant the user deleted came back as partial (or cancelled) on the next poll, its files
    gone but its state file sitting under the pre-resolve digest.
    """
    spelled, resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )
    legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    removed = download_manifest.purge_state("model", "Org/Model", "Q4_K_M", hub_cache = str(resolved))

    assert removed is True
    assert not legacy.is_file()
    assert download_manifest.read_manifest("model", "Org/Model", "Q4_K_M") is None


def test_variant_index_sees_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """Same asymmetry on the inventory side.

    cache_inventory feeds build_variant_state_index a directory derived from
    huggingface_hub.scan_cache_dir, which resolves. Without the configured
    spelling the cached-model views would report no state for a variant the
    progress endpoint can see.
    """
    spelled, resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )
    _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    index = download_manifest.build_variant_state_index(
        [("model", "Org/Model", str(resolved))],
        active_hub_cache = str(resolved),
    )
    state = index.for_repo("model", "Org/Model", hub_cache = str(resolved))

    assert state.manifest_for("Q4_K_M") is not None


def test_the_configured_spelling_is_only_borrowed_for_the_SAME_directory(monkeypatch, tmp_path):
    """The guard on the fan-out, which matters more than the fan-out.

    Borrowing the configured cache's spellings unconditionally would make a
    delete aimed at an INACTIVE cache sweep the active cache's state for the
    same repo -- silently losing state for a cache the user never touched,
    which is a far worse failure than the resurrection above.
    """
    spelled, resolved = _redirected_hub_cache(tmp_path)
    other = tmp_path / "other" / "hub"
    other.mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )
    active_legacy = _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")
    victim = state_dir.manifest_path("model", "Org/Model", "Q8_0", hub_cache = str(other))
    _write_manifest(victim, _manifest_payload("Org/Model", "Q8_0", str(other)))

    # Deleting the repo out of the OTHER cache must not touch the active one.
    download_manifest.purge_all_state_for_repo("model", "Org/Model", hub_cache = str(other))

    assert not victim.is_file()
    assert active_legacy.is_file()


def test_disagreeing_manifests_across_caches_are_refused(monkeypatch, tmp_path):
    """snapshot_progress picks its READING by bytes across every preferred cache dir, but the
    expected-file hashes come from one manifest lookup. Handing it the first cache's older
    revision filters out every blob of a later cache that holds the complete variant, so a
    finished download reports 0 or partial -- the exact failure this fallback exists to prevent,
    just sourced from the wrong cache. Two caches that disagree therefore yield no manifest, and
    the name-based fallback (which stays attributable per entry) takes over.
    """
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

    # Agreement is still answered: the point is the ambiguity, not the multiplicity.
    served[second.parent] = old
    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is old


def test_a_stale_active_manifest_is_compared_rather_than_returned(monkeypatch, tmp_path):
    """The configured cache's repo dir can be gone while its scoped state still holds an old
    manifest, and idle progress keeps scanning the remembered caches. Returning the active one
    unexamined applies a stale revision's hashes to a remembered cache that has the complete
    variant and filters every blob of it out -- the same wrong answer, one cache earlier."""
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
        # hub_cache omitted is the ACTIVE cache lookup.
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            stale if hub_cache is None else current
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None


def test_variant_enumeration_sees_legacy_scope_when_handed_a_RESOLVED_root(monkeypatch, tmp_path):
    """The third caller with the same asymmetry, and the one that was left out.

    A variant request carrying ``local_path`` is resolved by ``_repo_cache_dir_for_request``
    before the enumerator ever sees the cache root, so ``cache_scope_names`` -- which recovers
    the pre-resolve digest only from an unresolved path -- returned the canonical digest alone.
    On a cache reached through a symlink or junction the offline variant listing then lost the
    partial download entirely, along with its resume control, while the progress endpoint could
    still see it.
    """
    spelled, resolved = _redirected_hub_cache(tmp_path)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(spelled)),
    )
    _legacy_scoped_manifest(tmp_path, spelled, resolved, "Org/Model", "Q4_K_M")

    listed = dict(
        download_manifest.iter_variant_manifests("model", "Org/Model", hub_cache = str(resolved))
    )

    assert "Q4_K_M" in listed, (
        "the resolved spelling lost the legacy scope, so an offline listing cannot see the "
        "partial download it is meant to offer a resume for"
    )


def test_a_scanned_cache_with_no_manifest_refuses_the_others(monkeypatch, tmp_path):
    """A cache that contributes nothing is not a cache that agrees.

    Manifests get deleted, and older builds never wrote one, so a cache holding the COMPLETE
    snapshot can have no manifest at all. Returning another cache's manifest then applies its
    hashes to that snapshot's blobs -- filtering every one of them out -- and, worse, disables
    the per-entry name-based fallback that would still have counted them. The finished variant
    reports zero.
    """
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

    # ...and once that cache has its own agreeing manifest, the answer comes back.
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
        # The active cache (hub_cache=None) has none; the remembered one does.
        lambda repo_type, repo_id, variant = None, *, hub_cache = None: (
            None if hub_cache is None else other
        ),
    )

    assert downloads._variant_manifest_in_any_cache("unsloth/Model-GGUF", "Q4_K_M") is None


def test_an_unreadable_cache_root_is_unknown_rather_than_absent(monkeypatch, tmp_path):
    """A root that cannot be listed is not evidence that the cache was wiped.

    ``iter_repo_cache_dirs`` and ``iter_active_repo_cache_dirs`` swallow OSError per root, so
    an EACCES or EIO came back as "no cache dirs" -- and the all-zero reading that produces
    carried ``cache_path: null``, which hydration reads as gone and removes a persisted job
    whose partial cache is sitting on the disk behind that error. These errors never reached
    the exception fallback that already knew the difference, because nothing raised."""
    from hub.services import snapshot_progress
    from hub.utils import hf_cache_state

    unreadable = tmp_path / "hub"
    unreadable.mkdir()

    def _explode(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [unreadable])
    monkeypatch.setattr(hf_cache_state, "hf_cache_root", lambda root = None, **kw: None)
    monkeypatch.setattr(type(unreadable), "iterdir", _explode)

    # The enumeration reports the skip rather than only swallowing it...
    errors: list = []
    assert (
        hf_cache_state.preferred_repo_cache_dirs("model", "unsloth/Model-GGUF", scan_errors = errors)
        == []
    )
    assert errors and isinstance(errors[0], OSError)

    # ...and the reading built on it says unknown by omitting cache_path entirely.
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
    # And it survives DownloadProgressResponse, which defaults cache_path to None and would otherwise reinstate
    # the omission as an explicit "absent" before the frontend saw it.
    assert reading["cache_measured"] is False
    from hub.schemas.downloads import DownloadProgressResponse

    # DECLARED on the response model, or FastAPI drops it before the frontend sees it: these readings
    # serialize through DownloadProgressResponse, whose cache_path defaults to None, so the omission
    # alone was reinstated as an explicit "absent".
    assert "cache_measured" in DownloadProgressResponse.__annotations__
    assert reading["downloaded_bytes"] == 0


def test_a_scope_whose_payload_is_lost_reads_back_as_a_digest(monkeypatch, tmp_path):
    """A scope is unspellable in a filename, so it is stored hashed. Lose the payload
    and the reader falls back to that filename, handing back the digest instead of
    "@diffusion" -- and the older tag spells it without the "@". Both have to be
    recognisable as digests."""
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

    # A real quant label is never mistaken for one.
    for quant in ("Q4_K_M", "UD-Q4_K_XL", "sha256-short"):
        assert not state_dir.variant_is_hashed_fragment(quant)


def test_a_variant_with_nothing_of_its_own_says_so(monkeypatch, tmp_path):
    """Sibling quants share one repo cache directory, so "the directory exists" is the wrong
    granularity for a variant. With Q4_K_M's files deleted and Q8_0 keeping the dir alive, the
    reading was zero bytes with a non-null cache_path -- which hydration adopts as resumable,
    leaving a phantom card that blocks a fresh download of that same variant."""
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
    # Through the response model, or FastAPI drops the field and the frontend never sees it.
    from hub.schemas.downloads import DownloadProgressResponse

    # Declared on the response model too, else FastAPI drops it on the way out.
    assert "target_present" in DownloadProgressResponse.__annotations__

    # Unresolvable file set: nothing was established, so nothing is claimed either.
    unknown = _reading("Q4_K_M", frozenset())
    assert unknown["target_present"] is None

    # And a whole-repo job owns the directory, so the repo-level answer already covers it.
    assert _reading(None, frozenset())["target_present"] is None


def test_a_root_that_cannot_even_be_stat_ed_is_unknown(monkeypatch, tmp_path):
    """The failure can happen one step earlier than the listing.

    ``hf_cache_root`` calls ``_safe_is_dir``, which swallows the OSError from probing a
    restricted configured cache and answers None -- so an inaccessible active root produced a
    measured "no cache dir" answer with an empty scan_errors, and hydration retired the job as
    deleted. Statting is part of the scan.

    Driven through os.stat rather than Path.is_dir, because is_dir() suppresses the failure
    itself: it swallows several errnos on 3.13 and, as of 3.14, every OSError there is. A
    handler wrapped around it can never run."""
    import os as _os

    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    real_stat = _os.stat

    def _explode(path, *args, **kwargs):
        # ELOOP, which is one of the errnos Path.is_dir() answers False for rather than raising.
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
