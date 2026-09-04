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
# ... and per checkpoint, for the same reason: one key must not grow the file on its own.
_MAX_BASES_PER_CHECKPOINT = 16
# Serialises the read-modify-write below: losing a link to a lost update is the one direction that
# matters, since an unrecorded base can look orphaned and be offered for removal.
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
        # NOT sort_keys: json.loads keeps document order, so the file IS the recency record the trim reads.
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
        # Re-inserted at the end, not updated in place: recording is the only recency signal the trim has,
        # and returning early on a REPEAT resolution let the cap throw away the most-used link.
        links.pop(key, None)
        # Recency inside the list too, and capped: an explicit base_repo or a changing card tag makes one
        # checkpoint resolve a new base each load, and appending forever grew the file _MAX_LINKS bounds.
        fresh = [b for b in existing if _normalise(b) != _normalise(base)]
        links[key] = [*fresh, base][-_MAX_BASES_PER_CHECKPOINT:]
        wrote = _write_companion_links(links)
        # False when nothing NEW was recorded; the refresh above still happened.
        return wrote and not known


def _snapshot_relative_names(repo) -> Iterable[str]:
    """Every cached file of *repo* as a SNAPSHOT-RELATIVE posix name.

    ``CachedFileInfo.file_name`` is the basename alone; the path inside the snapshot only comes
    back from ``file_path`` relative to ``snapshot_path``. Family detection reads the whole
    relative name, so a GGUF filed under ``FLUX.2-klein/model-Q4_K_M.gguf`` is a dependent that a
    basename-only scan cannot see. The inventory and cleanup scanners already reconstruct it this
    way; this is the same reconstruction, so all three agree on what a cached file is called."""
    from hub.services.models.cache_inventory import cached_repo_files
    for revision in getattr(repo, "revisions", ()) or ():
        snapshot = getattr(revision, "snapshot_path", None)
        for file in cached_repo_files(revision):
            name = str(getattr(file, "file_name", "") or "")
            path = getattr(file, "file_path", None)
            if path and snapshot:
                try:
                    name = Path(path).relative_to(Path(snapshot)).as_posix()
                except ValueError:
                    pass
            if name:
                yield name


def holds_denoiser(name: str) -> bool:
    """A denoiser weight by NAME alone: a GGUF anywhere, or a shard under the denoiser folder.

    Its presence is what separates a checkpoint the user installed from a companion-only fetch,
    which takes everything BUT the denoiser folder. Use :func:`repo_holds_denoiser` where the repo
    is available: an SDXL-style whole-pipeline single file is a denoiser too, and only the repo's
    family says so."""
    lowered = name.lower()
    if lowered.endswith(".gguf"):
        return True
    return lowered.startswith(_DENOISER_DIRS) and lowered.endswith(_WEIGHT_SUFFIXES)


_DENOISER_DIRS = ("transformer/", "unet/")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf")


def repo_holds_denoiser(repo) -> bool:
    """Whether *repo* holds a runnable checkpoint, i.e. is a model the user installed.

    Beyond the name-only test: a single_file pick caches its checkpoint as ONE top-level file,
    ``sd_xl_base_1.0.safetensors`` for a whole-pipeline family and ``flux2-dev-fp8.safetensors``
    for a transformer-only one, with no denoiser folder to recognise. Those repos are curated
    companion bases whose self-dependency is dropped, so without this the orphan listing called an
    installed checkpoint an unused leftover and Free up space deleted it. A companion-only fetch
    never lands a weight at the snapshot root, and the two ways to be wrong are not symmetric: an
    orphan we decline to offer costs disk, a checkpoint we offer costs the model.

    The one root weight that is NOT a checkpoint is the curated component repos' own asset, which
    is why they are excluded by id: a single-file VAE or text encoder is the companion, and
    reading it as an installed model is what kept an orphaned pair pinned to disk."""
    names = list(_snapshot_relative_names(repo))
    if any(holds_denoiser(name) for name in names):
        return True
    repo_id = str(getattr(repo, "repo_id", "") or "")
    if _normalise(repo_id) in _component_only_repo_ids():
        return False
    for name in names:
        if "/" in name or not name.lower().endswith(_WEIGHT_SUFFIXES):
            continue
        if _detect_family(repo_id, name) is not None:
            return True
    return False


def _component_only_repo_ids() -> set[str]:
    """Curated sd.cpp component repo ids, under every identity they can be cached as."""
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        return {_normalise(r) for r in _with_mirrors(sd_cpp_companion_only_repo_ids())}
    except Exception as exc:  # noqa: BLE001 -- no table means no exclusions, as before
        logger.debug("sd.cpp companion table unavailable: %s", exc)
        return set()


def _is_checkpoint_pick_name(name: str) -> bool:
    """A cached file the loader would hand to ``detect_family_for_pick`` as the pick.

    Both kinds it accepts: a main GGUF anywhere in the repo, and a TOP-LEVEL single-file
    ``.safetensors``, which is how a transformer-only or whole-pipeline single_file pick is
    cached. Restricted to the snapshot root because that is where a single-file pick lands; a
    companion fetch's shards live under ``text_encoder/``, ``vae/`` and friends and say nothing
    about which checkpoint is installed."""
    lowered = name.lower()
    if lowered.endswith(".gguf"):
        return "mmproj" not in lowered
    return "/" not in name and lowered.endswith(".safetensors")


def _cached_checkpoint_pick_names(cache_scans) -> dict[str, list[str]]:
    """``{repo_id_lower: [every cached file the loader could pick]}`` for the family probe below.

    Both diffusion loaders detect a family from the FILENAME as well as the repo id, so a
    perfectly runnable ``some-owner/custom`` holding ``flux-2-klein-4b-Q4_K_M.gguf`` is a
    dependent that a repo-id-only probe cannot see. That matters most where there is nothing else
    to fall back on: on an upgraded install no links have been recorded yet, so the family probe
    is the whole guard.

    EVERY name, and across EVERY cache root: one generic repo can hold checkpoints of two
    different families, in separate subdirectories or in two remembered caches, and the loader
    will select any of them by file name. Probing one left the others' bases looking orphaned."""
    names: dict[str, list[str]] = {}
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                key = _normalise(str(getattr(repo, "repo_id", "") or ""))
                if not key:
                    continue
                found = names.setdefault(key, [])
                for name in _snapshot_relative_names(repo):
                    if _is_checkpoint_pick_name(name) and name not in found:
                        found.append(name)
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
    return {key: found for key, found in names.items() if found}


def _denoiser_holding_repo_ids(cache_scans) -> set[str]:
    """Cached repos (lowercased ids) that hold a runnable denoiser, i.e. real checkpoints."""
    held: set[str] = set()
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                key = _normalise(str(getattr(repo, "repo_id", "") or ""))
                if not key or key in held:
                    continue
                if repo_holds_denoiser(repo):
                    held.add(key)
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
    return held


def _family_bases_for_names(repo_id: str, gguf_filenames: Iterable[Optional[str]]) -> set[str]:
    """:func:`_family_bases` unioned over every cached GGUF name, plus the repo id on its own."""
    bases: set[str] = set()
    for name in dict.fromkeys([None, *(gguf_filenames or ())]):
        bases |= _family_bases(repo_id, name)
    return bases


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
        # The loader resolves a base with this same call, so no family means no load and there is no
        # dependent here to protect: the two fail together.
        return set()
    bases = {resolve_base_repo(fam, None)}
    bases |= _curated_variant_bases(fam, repo_id, gguf_filename)
    # The NATIVE engine never reads the diffusers base: it fetches a single-file VAE and text encoder
    # from their own repos, so a pre-existing native GGUF with no recorded link would have had its
    # encoder listed as unused and removed underneath it.
    bases |= {repo for repo, _file in _sd_cpp_component_specs(fam, repo_id, gguf_filename)}
    return _with_mirrors(bases)


def _curated_variant_bases(fam, repo_id: str, gguf_filename: Optional[str]) -> set[str]:
    """Curated bases of *fam* that the checkpoint's own identity names, beyond the family default.

    One family entry serves several sizes, and which one a pick uses comes from the Hub card's
    ``base_model`` tag. Offline that tag is gone, so ``resolve_base_repo(fam, None)`` answers the
    family DEFAULT: an installed ``unsloth/FLUX.2-klein-9B-GGUF`` derived as needing the 4B base,
    while its cached ``black-forest-labs/FLUX.2-klein-9B`` had no dependent at all. That base is
    in the curated offerable set, so before the checkpoint's first recorded load Free up space
    would list it and delete the companions of a model that is still installed.

    Conservative on both ends. Candidates come only from the curated tables, only from entries
    belonging to THIS family, and only when the checkpoint id or its GGUF file name names the
    base -- so a wrong guess can add a dependent, never invent a repo. Naming one base too many
    costs a delete that is refused; naming one too few costs an installed model.

    Matched with punctuation folded away, the way family detection accepts its own aliases:
    ``flux1-dev-Q4_K_M.gguf`` resolves to the flux.1 family, and a literal comparison against the
    curated name ``FLUX.1-dev`` missed it over one dot, leaving the dev base offerable while the
    dev checkpoint was installed."""
    family = _fold(getattr(fam, "name", "") or "")
    if not family:
        return set()
    identity = f"{_fold(repo_id)} {_fold(gguf_filename or '')}"
    out: set[str] = set()
    for candidate in _curated_base_ids():
        name = _fold(candidate.rsplit("/", 1)[-1])
        if name and name.startswith(family) and name in identity:
            out.add(candidate)
    return out


def _fold(text: str) -> str:
    """Lowercased with every separator dropped, so ``FLUX.1-dev`` and ``flux1-dev`` compare equal."""
    return "".join(c for c in (text or "").lower() if c.isalnum())


def _curated_base_ids() -> set[str]:
    """Curated family bases and their gated/mirror variants, as written in the tables."""
    ids: set[str] = set()
    try:
        from core.inference.diffusion_families import _FAMILIES
        ids |= {fam.base_repo for fam in _FAMILIES if getattr(fam, "base_repo", None)}
    except Exception as exc:  # noqa: BLE001 -- no table means no extra candidates, as before
        logger.debug("Companion base table unavailable: %s", exc)
    ids |= _mirror_pair_ids()
    return ids


def _sd_cpp_component_specs(
    fam, repo_id: str, gguf_filename: Optional[str]
) -> set[tuple[str, str]]:
    """``(repo, filename)`` for every single-file VAE / text encoder an sd.cpp load of *fam*
    opens. The FILENAME matters as much as the repo: the Qwen-Image encoder is one named quant
    inside a chat GGUF repo, so a sibling quant in the same repo is not a substitute for it.
    Never raises."""
    specs: set[tuple[str, str]] = set()
    try:
        from core.inference.diffusion_families import sd_cpp_text_encoder_candidates

        vae = getattr(fam, "sd_cpp_vae", None)
        if vae and vae[0]:
            specs.add((vae[0], vae[1]))
        # EVERY set the family could pick: there is no header to read and a renamed FLUX.2-klein 9B file
        # carries no size token, so guessing answers 4B and leaves the 9B encoder unprotected.
        for encoder in sd_cpp_text_encoder_candidates(fam) or ():
            if encoder and encoder[0]:
                specs.add((encoder[0], encoder[1]))
    except Exception as exc:  # noqa: BLE001 -- one missing table never hides the rest
        logger.debug("sd.cpp component derivation unavailable for %s: %s", repo_id, exc)
    return specs


def required_companion_asset_files(cache_scans) -> dict[str, set[str]]:
    """``{repo_id_lower: {filename, ...}}`` the installed checkpoints' native loads actually open.

    The repo-level view cannot answer a VARIANT delete. Native Qwen-Image opens exactly
    ``Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`` inside a chat GGUF repo, so removing that one quant
    strands the image checkpoint even though the repo still holds a Q8_0 nobody can substitute.
    """
    pick_names = _cached_checkpoint_pick_names(cache_scans)
    files: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
        # One repo can hold checkpoints of two families, and each opens its own components.
        for name in dict.fromkeys([None, *pick_names.get(_normalise(repo_id), [])]):
            fam = _detect_family(repo_id, name)
            if fam is None:
                continue
            for repo, filename in _sd_cpp_component_specs(fam, repo_id, name):
                if filename:
                    files.setdefault(_normalise(repo), set()).add(filename)
    return files


def _detect_family(repo_id: str, gguf_filename: Optional[str]):
    """``detect_family_for_pick`` that never raises and never imports at module scope."""
    try:
        from core.inference.diffusion_families import detect_family_for_pick
        return detect_family_for_pick(repo_id, gguf_filename)
    except Exception:  # noqa: BLE001
        return None


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
    # The pairs are companion bases by construction, existing only because a GGUF pick of that family
    # needs an ungated copy, and they cover the variants one family entry serves through a card tag.
    bases = _curated_base_ids()
    # The native engine's component-only repos ARE the companions for an sd.cpp pick, and the largest
    # half of the footprint; leaving them out made them link-only strangers. Safe against a chat model
    # borrowed as a text encoder, since the orphan listing skips any repo holding a GGUF.
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        bases |= set(sd_cpp_companion_only_repo_ids())
    except Exception as exc:  # noqa: BLE001 -- one missing table never hides the rest
        logger.debug("sd.cpp companion table unavailable: %s", exc)
    return {_normalise(b) for b in _with_mirrors(bases)}


def _mirror_pair_ids() -> set[str]:
    # The WHOLE table, gated and ungated: gating decides whether a fetch may override a user's cache,
    # not whether a base can strand companions, so reading only the gated half would offer an
    # installed checkpoint's companions for deletion.
    try:
        from core.inference.diffusion_families import _MIRROR_PAIRS
    except Exception:  # noqa: BLE001
        return set()
    return {rid for pair in _MIRROR_PAIRS for rid in pair}


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
    # Through the same identity expansion required_companion_bases uses, so the two agree WHICH copy
    # is protected: comparing the literal id left an upgraded install's repack copy deletable.
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
    pick_names = _cached_checkpoint_pick_names(cache_scans)
    component_only = _cached_component_only_repo_ids(cache_scans)
    required: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
        key = _normalise(repo_id)
        if key in ignored or key in component_only:
            continue
        bases = _family_bases_for_names(repo_id, pick_names.get(key, []))
        recorded = links.get(key)
        if recorded:
            bases |= _with_mirrors(recorded)
        # Canonical, not literal: a cached MIRROR resolves its own family back to the UPSTREAM id, which
        # would make each identity a dependent of the other and leave a cache holding both stuck.
        self_keys = {key, _normalise(_canonical(repo_id))}
        for base in bases:
            base_key = _normalise(base)
            if not base_key or base_key in self_keys or _normalise(_canonical(base)) in self_keys:
                continue
            required.setdefault(base_key, set()).add(repo_id)
    return required


def _cached_component_only_repo_ids(cache_scans) -> set[str]:
    """Cached curated component repos holding no denoiser: companions, never checkpoints.

    Several of those ids carry a family keyword of their own -- ``unsloth/FLUX.2-dev-ComfyUI``
    detects as flux.2-dev -- so the loop above read a bare text-encoder fetch as an installed
    checkpoint and recorded the sibling VAE as still required. The orphaned pair then held each
    other on disk: Free up space would not list them and a delete was refused, until the user
    happened to remove one component first. A repo that does hold a denoiser is a model the user
    installed and still counts, which is what keeps a borrowed chat GGUF a dependent of nothing
    and a checkpoint of itself.

    Through the same identity expansion the rest of this file uses, because an upgraded install
    holds the component under the legacy repack id the native fetch fell back to
    (``Comfy-Org/flux2-dev`` for ``unsloth/FLUX.2-dev-ComfyUI``). That id matches the flux.2-dev
    family just as literally, so leaving it out kept the very caches this exclusion is for."""
    curated = _component_only_repo_ids()
    if not curated:
        return set()
    return curated - _denoiser_holding_repo_ids(cache_scans)


def _canonical(repo_id: str) -> str:
    """``canonical_base``, degrading to the id itself when the family tables are unavailable."""
    try:
        from core.inference.diffusion_families import canonical_base
        return canonical_base(repo_id)
    except Exception:  # noqa: BLE001
        return repo_id
