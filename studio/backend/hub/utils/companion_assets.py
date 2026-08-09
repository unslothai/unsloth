# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which cached repos are companion assets, and which installed models still need them.

An image or video GGUF ships only the denoiser. Everything else the pipeline needs -- text
encoders, VAE, tokenizer, scheduler and the pipeline manifest -- comes from a separate
*companion base repo*, cached under its own ``models--`` directory. It is routinely the larger
half: FLUX.2 klein 4B is 2.6 GB of Q4_K_M against 8.23 GB of companions, and one copy serves
every quant in the GGUF repo.

That sharing is what makes deletion delicate. Two questions have to be answerable without a
counter, because a counter that drifts is worse than none:

  "is this repo a companion base?"      -> :func:`is_companion_base`
  "what still needs it, right now?"     -> :func:`required_companion_bases`

Both are DERIVED at call time from what is installed, using the same resolvers the loader uses
to pick a base (``detect_family_for_pick`` + ``resolve_base_repo`` + ``mirror_repo``). If family
detection fails for a cached repo, the loader cannot load it either, so there is no dependent to
protect -- the derivation and the thing it protects fail together, by construction.

The one case derivation cannot see offline is a checkpoint whose base comes from its Hub card's
``base_model`` tag rather than the family default (``_resolve_base_repo``), because a GGUF pick
caches only the ``.gguf`` file, never the card. :func:`record_companion_link` closes that: the
resolver records the pair it chose, and the reader unions those links into the derived set. It
is a LINK, not a count -- an entry can only ever ADD a dependent, is filtered by "is that
checkpoint still installed", and its loss degrades to the family default rather than to a wrong
answer. Nothing here deletes; callers do, and only for a base with an empty dependent set.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Iterable, Optional

from loggers import get_logger

from hub.utils.state_dir import state_root

logger = get_logger(__name__)


_LINKS_FILENAME = "companion-assets.json"
_LINKS_VERSION = 1
# A runaway writer must not grow an unbounded state file; oldest links are dropped first.
_MAX_LINKS = 512
# Serialises the read-modify-write below. Losing a link to a lost update is the one direction
# that matters: an unrecorded base can look orphaned and be offered for removal.
_WRITE_LOCK = threading.Lock()


def _links_path(*, create: bool = False) -> Optional[Path]:
    root = state_root(create = create)
    if root is None:
        return None
    return root / _LINKS_FILENAME


def _normalise(repo_id: Optional[str]) -> str:
    return (repo_id or "").strip().lower()


def read_companion_links() -> dict[str, list[str]]:
    """``{checkpoint_repo_id_lower: [base_repo_id, ...]}`` recorded by past resolutions.

    Fails OPEN: a missing, unreadable or schema-mismatched file yields ``{}`` and every caller
    falls back to family-default derivation, which is what shipped before this file existed.
    """
    path = _links_path()
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _LINKS_VERSION:
        return {}
    raw = payload.get("links")
    if not isinstance(raw, dict):
        return {}
    links: dict[str, list[str]] = {}
    for checkpoint, bases in raw.items():
        if not isinstance(checkpoint, str) or not isinstance(bases, list):
            continue
        clean = [b.strip() for b in bases if isinstance(b, str) and b.strip()]
        if clean:
            links[_normalise(checkpoint)] = clean
    return links


def _write_companion_links(links: dict[str, list[str]]) -> bool:
    path = _links_path(create = True)
    if path is None:
        return False
    if len(links) > _MAX_LINKS:
        # dicts preserve insertion order, so this drops the oldest recorded checkpoints.
        links = dict(list(links.items())[-_MAX_LINKS:])
    payload = {"version": _LINKS_VERSION, "links": links}
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        # NOT sort_keys: json.loads keeps document order, so the file IS the recency record the
        # trim above reads. Sorting it made the next read alphabetical, and the trim then evicted
        # the lexicographically smallest link rather than the oldest -- so a link recorded minutes
        # ago could go, and its base become deletable while the checkpoint was still installed.
        tmp.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("Could not record companion links at %s: %s", path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def record_companion_link(checkpoint_repo_id: str, base_repo_id: str) -> bool:
    """Remember that *checkpoint_repo_id* resolves its companions from *base_repo_id*.

    Called from the base resolvers, so it sees every pick the product makes. Additive and
    idempotent; a self-reference (a full pipeline is its own base) is dropped, since it would
    make the pipeline look like a dependent of itself and block its own deletion forever.
    Best-effort: a write failure is logged at debug and never reaches the caller's load.
    """
    checkpoint = (checkpoint_repo_id or "").strip()
    base = (base_repo_id or "").strip()
    if not checkpoint or not base or _normalise(checkpoint) == _normalise(base):
        return False
    # A local filesystem path is not a Hub id and never appears in a cache scan.
    if (
        "/" not in checkpoint.replace("\\", "/").strip("/")
        or Path(checkpoint).expanduser().exists()
    ):
        return False
    if Path(base).expanduser().exists():
        return False
    with _WRITE_LOCK:
        # Re-read inside the lock: a concurrent resolver's link must not be dropped by this write.
        links = read_companion_links()
        key = _normalise(checkpoint)
        existing = links.get(key, [])
        known = any(_normalise(b) == _normalise(base) for b in existing)
        # Re-inserted at the end, not updated in place: recording is the only recency signal the
        # trim has, so a checkpoint that just resolved must not keep an old position and be
        # evicted ahead of one nothing has touched since. That applies to a REPEAT resolution too
        # -- a checkpoint reloaded every day but first recorded long ago is the most-used link
        # there is, and returning early on it let the cap throw it away.
        links.pop(key, None)
        links[key] = existing if known else [*existing, base]
        wrote = _write_companion_links(links)
        # False when nothing NEW was recorded; the refresh above still happened.
        return wrote and not known


def _cached_main_gguf_names(cache_scans) -> dict[str, str]:
    """``{repo_id_lower: one cached main GGUF filename}`` for the family probe below.

    Both diffusion loaders detect a family from the FILENAME as well as the repo id, so a
    perfectly runnable ``some-owner/custom`` holding ``flux-2-klein-4b-Q4_K_M.gguf`` is a
    dependent that a repo-id-only probe cannot see. That matters most where there is nothing else
    to fall back on: on an upgraded install no links have been recorded yet, so the family probe
    is the whole guard."""
    names: dict[str, str] = {}
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                key = _normalise(str(getattr(repo, "repo_id", "") or ""))
                if not key or key in names:
                    continue
                for revision in getattr(repo, "revisions", ()) or ():
                    for file in getattr(revision, "files", ()) or ():
                        name = str(getattr(file, "file_name", "") or "")
                        if name.lower().endswith(".gguf") and "mmproj" not in name.lower():
                            names[key] = name
                            break
                    if key in names:
                        break
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
    return names


def _family_bases(repo_id: str, gguf_filename: Optional[str] = None) -> set[str]:
    """Companion base ids *repo_id* could resolve to, from the family tables alone.

    Both identities: the upstream the tables key on and its ungated unsloth mirror, because
    ``prefer_ungated_mirror`` picks between them per fetch and either may be what landed on disk.

    ``gguf_filename`` is a cached checkpoint from this repo, passed for the same reason the
    loaders pass it: a repo whose family keyword appears only in the file name still loads, and a
    dependent this probe cannot see is a companion cleanup will happily delete underneath it.
    """
    try:
        from core.inference.diffusion_families import (
            detect_family_for_pick,
            resolve_base_repo,
        )
    except Exception as exc:  # noqa: BLE001 -- classification failure protects nothing new
        logger.debug("Companion base derivation unavailable for %s: %s", repo_id, exc)
        return set()
    try:
        fam = detect_family_for_pick(repo_id, gguf_filename)
    except Exception:  # noqa: BLE001
        return set()
    if fam is None:
        # The loader resolves a base with this same call. No family means no load, so there is
        # no dependent here to protect -- not a gap, the two fail together.
        return set()
    return _with_mirrors({resolve_base_repo(fam, None)})


def _with_mirrors(bases: Iterable[str]) -> set[str]:
    """Every id one companion base can be cached under: itself, the upstream it copies, its
    ungated mirror, and the community repack that mirror stands in for.

    The repack matters as much as the mirror. ``prefer_cached_legacy_source`` deliberately sends
    the native fetch back to a repack an upgraded install already holds, so on those machines the
    bytes protecting an installed GGUF sit under the OLD repo key and nothing else names it. The
    live loaded-repo guard in the sd.cpp backend already expands all four; this is the same set,
    for the recorded links."""
    try:
        from core.inference.diffusion_families import (
            canonical_base,
            legacy_source_repo,
            mirror_repo,
        )
    except Exception:  # noqa: BLE001
        return {b for b in bases if b}
    out: set[str] = set()
    for base in bases:
        if not base:
            continue
        out.add(base)
        upstream = canonical_base(base)
        if upstream:
            out.add(upstream)
        mirror = mirror_repo(upstream or base)
        if mirror:
            out.add(mirror)
        for candidate in (base, upstream, mirror):
            legacy = legacy_source_repo(candidate) if candidate else None
            if legacy:
                out.add(legacy)
    return out


def known_companion_base_ids() -> set[str]:
    """Every id (lowercased) that is a curated image-family companion base, or its mirror.

    Deliberately NOT "anything a link ever named": orphan cleanup offers only repos this set
    recognises, so a mis-recorded link can never turn an unrelated repo into a delete candidate.
    """
    try:
        from core.inference.diffusion_families import _FAMILIES
    except Exception as exc:  # noqa: BLE001
        logger.debug("Companion base table unavailable: %s", exc)
        return set()
    bases = {fam.base_repo for fam in _FAMILIES if getattr(fam, "base_repo", None)}
    # The native engine's component-only repos (a single-file VAE, a text encoder) are curated
    # table entries too, and for an sd.cpp pick they ARE the companions -- the largest half of the
    # footprint. Leaving them out made them link-only strangers: unlisted by Free up space and
    # dropped from the delete preview, so the assets this cleanup exists for stayed invisible.
    # Safe against the one hazard that table warns about, a chat model borrowed as a text encoder:
    # the orphan listing skips any repo holding a GGUF, so it is never offered as a leftover.
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        bases |= set(sd_cpp_companion_only_repo_ids())
    except Exception as exc:  # noqa: BLE001 -- one missing table never hides the rest
        logger.debug("sd.cpp companion table unavailable: %s", exc)
    # The gated/mirror pairs are companion bases by construction -- they exist only because a
    # GGUF pick of that family needs an ungated copy of them -- and they cover the variants one
    # family entry serves through a card tag (klein-9B under the klein-4B family, and so on).
    bases |= _mirror_pair_ids()
    return {_normalise(b) for b in _with_mirrors(bases)}


def _mirror_pair_ids() -> set[str]:
    try:
        from core.inference.diffusion_families import _GATED_MIRROR_PAIRS
    except Exception:  # noqa: BLE001
        return set()
    return {rid for pair in _GATED_MIRROR_PAIRS for rid in pair}


def is_companion_base(repo_id: str) -> bool:
    """Whether deleting *repo_id* could strand an installed model's companions.

    Wider than :func:`known_companion_base_ids` on purpose, and the two are conservative in
    opposite directions. This one gates a REFUSAL, so it also admits any base a recorded link
    names: a checkpoint whose base came from its card tag can point anywhere, and missing it
    would leave exactly the hole this guard exists to close. :func:`known_companion_base_ids`
    gates an OFFER TO DELETE, so it stays table-only and a bad link can never reach it.
    """
    key = _normalise(repo_id)
    if key in known_companion_base_ids():
        return True
    # Through the same identity expansion required_companion_bases uses, so the two agree on
    # WHICH copy is protected. A link recorded against the unsloth mirror is satisfied on an
    # upgraded install by the community repack the fetch fell back to, and comparing the literal
    # id alone left that copy deletable while the checkpoint holding it was still installed.
    return any(
        key in {_normalise(rid) for rid in _with_mirrors([base])}
        for bases in read_companion_links().values()
        for base in bases
    )


def _cached_model_repo_ids(cache_scans) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                repo_id = str(getattr(repo, "repo_id", "") or "")
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
            key = _normalise(repo_id)
            if not repo_id or key in seen:
                continue
            seen.add(key)
            ids.append(repo_id)
    return ids


def required_companion_bases(
    cache_scans, *, ignore_repo_ids: Iterable[str] = ()
) -> dict[str, set[str]]:
    """``{base_repo_id_lower: {checkpoint repo ids that still need it}}``.

    Derived from *cache_scans* -- what is installed at this instant -- unioned with the recorded
    links of those same installed checkpoints. *ignore_repo_ids* is the delete being previewed:
    dropping it answers "what would still need this base AFTER the delete" without mutating
    anything. A repo never counts as its own dependent.
    """
    ignored = {_normalise(r) for r in ignore_repo_ids}
    links = read_companion_links()
    gguf_names = _cached_main_gguf_names(cache_scans)
    required: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
        key = _normalise(repo_id)
        if key in ignored:
            continue
        bases = _family_bases(repo_id, gguf_names.get(key))
        recorded = links.get(key)
        if recorded:
            bases |= _with_mirrors(recorded)
        # Canonical, not literal: a cached MIRROR of a base resolves its own family back to the
        # UPSTREAM id, which would make each identity a dependent of the other and leave a cache
        # holding both unable to delete either. They are copies of one repo, not a pair that
        # needs each other.
        self_keys = {key, _normalise(_canonical(repo_id))}
        for base in bases:
            base_key = _normalise(base)
            if not base_key or base_key in self_keys or _normalise(_canonical(base)) in self_keys:
                continue
            required.setdefault(base_key, set()).add(repo_id)
    return required


def _canonical(repo_id: str) -> str:
    """``canonical_base``, degrading to the id itself when the family tables are unavailable."""
    try:
        from core.inference.diffusion_families import canonical_base
        return canonical_base(repo_id)
    except Exception:  # noqa: BLE001
        return repo_id
