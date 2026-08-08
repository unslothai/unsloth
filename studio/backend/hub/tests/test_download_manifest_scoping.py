# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from types import SimpleNamespace

from hub.utils import download_manifest, state_dir


def _write_manifest(path, payload):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


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
    assert variant == "@diffusion"  # intact while the payload is readable
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
