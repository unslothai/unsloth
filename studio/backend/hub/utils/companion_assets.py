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
        tmp.write_text(json.dumps(payload, indent = 2, sort_keys = True), encoding = "utf-8")
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
    if "/" not in checkpoint.replace("\\", "/").strip("/") or Path(checkpoint).expanduser().exists():
        return False
    if Path(base).expanduser().exists():
        return False
    with _WRITE_LOCK:
        # Re-read inside the lock: a concurrent resolver's link must not be dropped by this write.
        links = read_companion_links()
        existing = links.get(_normalise(checkpoint), [])
        if any(_normalise(b) == _normalise(base) for b in existing):
            return False
        links[_normalise(checkpoint)] = [*existing, base]
        return _write_companion_links(links)


def _family_bases(repo_id: str) -> set[str]:
    """Companion base ids *repo_id* could resolve to, from the family tables alone.

    Both identities: the upstream the tables key on and its ungated unsloth mirror, because
    ``prefer_ungated_mirror`` picks between them per fetch and either may be what landed on disk.
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
        fam = detect_family_for_pick(repo_id)
    except Exception:  # noqa: BLE001
        return set()
    if fam is None:
        # The loader resolves a base with this same call. No family means no load, so there is
        # no dependent here to protect -- not a gap, the two fail together.
        return set()
    return _with_mirrors({resolve_base_repo(fam, None)})


def _with_mirrors(bases: Iterable[str]) -> set[str]:
    try:
        from core.inference.diffusion_families import canonical_base, mirror_repo
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
    return any(
        _normalise(base) == key
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
    cache_scans,
    *,
    ignore_repo_ids: Iterable[str] = (),
) -> dict[str, set[str]]:
    """``{base_repo_id_lower: {checkpoint repo ids that still need it}}``.

    Derived from *cache_scans* -- what is installed at this instant -- unioned with the recorded
    links of those same installed checkpoints. *ignore_repo_ids* is the delete being previewed:
    dropping it answers "what would still need this base AFTER the delete" without mutating
    anything. A repo never counts as its own dependent.
    """
    ignored = {_normalise(r) for r in ignore_repo_ids}
    links = read_companion_links()
    required: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
        key = _normalise(repo_id)
        if key in ignored:
            continue
        bases = _family_bases(repo_id)
        recorded = links.get(key)
        if recorded:
            bases |= _with_mirrors(recorded)
        for base in bases:
            base_key = _normalise(base)
            if not base_key or base_key == key:
                continue
            required.setdefault(base_key, set()).add(repo_id)
    return required
