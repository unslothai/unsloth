# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which cached repos are companion assets, and which installed models still need them. An image or video GGUF ships only the denoiser; everything else the pipeline needs (text encoders, VAE, tokenizer, scheduler and the pipeline manifest) comes from a separate *companion base repo* cached under its own ``models--`` directory, routinely the larger half: FLUX.2 klein 4B is 2.6 GB of Q4_K_M against 8.23 GB of companions, and one copy serves every quant in the GGUF repo. That sharing is what makes deletion delicate: "is this repo a companion base?" (:func:`is_companion_base`) and "what still needs it, right now?" (:func:`required_companion_bases`) both have to be answerable without a counter, because a counter that drifts is worse than none. Both are DERIVED at call time from what is installed, using the same resolvers the loader uses to pick a base (``detect_family_for_pick`` + ``resolve_base_repo`` + ``mirror_repo``), so if family detection fails for a cached repo the loader cannot load it either and there is no dependent to protect: the derivation and the thing it protects fail together. The one case derivation cannot see offline is a checkpoint whose base comes from its Hub card's ``base_model`` tag rather than the family default (``_resolve_base_repo``), because a GGUF pick caches only the ``.gguf`` file, never the card. :func:`record_companion_link` closes that: the resolver records the pair it chose and the reader unions those links into the derived set. It is a LINK, not a count: an entry can only ever ADD a dependent, is filtered by "is that checkpoint still installed", and its loss degrades to the family default rather than to a wrong answer. Nothing here deletes; callers do, and only for a base with an empty dependent set."""

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
# Serialises the read-modify-write below: losing a link to a lost update is the one direction that matters, since an unrecorded base can look orphaned and be offered for removal.
_WRITE_LOCK = threading.Lock()


def _links_path(*, create: bool = False) -> Optional[Path]:
    root = state_root(create = create)
    if root is None:
        return None
    return root / _LINKS_FILENAME


def _normalise(repo_id: Optional[str]) -> str:
    return (repo_id or "").strip().lower()


def read_companion_links() -> dict[str, list[str]]:
    """``{checkpoint_repo_id_lower: [base_repo_id, ...]}`` recorded by past resolutions. Fails OPEN: a missing, unreadable or schema-mismatched file yields ``{}`` and every caller falls back to family-default derivation, which is what shipped before this file existed."""
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
    retired = _retired_link_ids()
    links: dict[str, list[str]] = {}
    for checkpoint, bases in raw.items():
        if not isinstance(checkpoint, str) or not isinstance(bases, list):
            continue
        # A link to a repo nothing loads from any more is dropped on read (and so from the file on the next write):
        # it would otherwise pin a retired sd-cli repack for as long as the GGUF that once fetched it stays installed.
        clean = [
            b.strip()
            for b in bases
            if isinstance(b, str) and b.strip() and _normalise(b) not in retired
        ]
        if clean:
            links[_normalise(checkpoint)] = clean
    return links


def _retired_link_ids() -> frozenset[str]:
    """``retired_companion_link_ids``, empty when the family tables are unavailable (every link then counts)."""
    try:
        from core.inference.diffusion_families import retired_companion_link_ids
        return retired_companion_link_ids()
    except Exception:  # noqa: BLE001 -- no table means nothing is known to be retired
        return frozenset()


def _write_companion_links(links: dict[str, list[str]]) -> bool:
    path = _links_path(create = True)
    if path is None:
        return False
    if len(links) > _MAX_LINKS:
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
    """Remember that *checkpoint_repo_id* resolves its companions from *base_repo_id*. Called from the base resolvers, so it sees every pick the product makes. Additive and idempotent; a self-reference (a full pipeline is its own base) is dropped, since it would make the pipeline look like a dependent of itself and block its own deletion forever. Best-effort: a write failure is logged at debug and never reaches the caller's load."""
    checkpoint = (checkpoint_repo_id or "").strip()
    base = (base_repo_id or "").strip()
    if not checkpoint or not base or _normalise(checkpoint) == _normalise(base):
        return False
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
        # Re-inserted at the end, not updated in place: recording is the only recency signal the trim has, and returning early on a REPEAT resolution let the cap throw away the most-used link.
        links.pop(key, None)
        # Recency inside the list too, and capped: an explicit base_repo or a changing card tag makes one checkpoint resolve a new base each load, and appending forever grew the file _MAX_LINKS bounds.
        fresh = [b for b in existing if _normalise(b) != _normalise(base)]
        links[key] = [*fresh, base][-_MAX_BASES_PER_CHECKPOINT:]
        wrote = _write_companion_links(links)
        # False when nothing NEW was recorded; the refresh above still happened.
        return wrote and not known


def _snapshot_relative_names(repo) -> Iterable[str]:
    """Every cached file of *repo* as a SNAPSHOT-RELATIVE posix name. ``CachedFileInfo.file_name`` is the basename alone; the path inside the snapshot only comes back from ``file_path`` relative to ``snapshot_path``. Family detection reads the whole relative name, so a GGUF filed under ``FLUX.2-klein/model-Q4_K_M.gguf`` is a dependent that a basename-only scan cannot see. The inventory and cleanup scanners already reconstruct it this way; this is the same reconstruction, so all three agree on what a cached file is called."""
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
    """A denoiser weight by NAME alone: a GGUF anywhere, or a shard under the denoiser folder. Its presence is what separates a checkpoint the user installed from a companion-only fetch, which takes everything BUT the denoiser folder. Use :func:`repo_holds_denoiser` where the repo is available: an SDXL-style whole-pipeline single file is a denoiser too, and only the repo's family says so."""
    lowered = name.lower()
    if lowered.endswith(".gguf"):
        return True
    return lowered.startswith(_DENOISER_DIRS) and lowered.endswith(_WEIGHT_SUFFIXES)


_DENOISER_DIRS = ("transformer/", "unet/")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf")


_HOSTED_DENOISER = "denoiser"


def _hosted_component_table() -> dict[str, dict[str, set[tuple[str, str, str]]]]:
    """``{repo_id_lower: {filename_lower: {(registry, family, kind)}}}`` for every hosted quantized component the
    loaders fetch by name, read from the loaders' own tables. Keyed per FILE: one repo can serve several families
    (``unsloth/Qwen-Image-FP8``). Curated image bases/mirrors are excluded: the image derivation covers them."""
    table: dict[str, dict[str, set[tuple[str, str, str]]]] = {}

    def add(repo_id, names, owner, kind) -> None:
        key = _normalise(repo_id)
        if not key:
            return
        files = table.setdefault(key, {})
        for name in names:
            if name:
                files.setdefault(str(name).lower(), set()).add((*owner, kind))

    families = _family_tables()
    by_name: dict[str, list[tuple[str, str]]] = {}
    for registry, fam in families:
        by_name.setdefault(str(fam.name), []).append((registry, str(fam.name)))

    try:
        from core.inference.diffusion_prequant import derived_prequant_filenames, with_comfy_twins
        from core.inference.video_denoiser_prequant import denoiser_components
        for registry, fam in families:
            owner = (registry, str(fam.name))
            rows = [*getattr(fam, "prequant_repos", ())]
            rows += [
                (scheme, repo) for _b, scheme, repo in getattr(fam, "prequant_variant_repos", ())
            ]
            for scheme, repo in rows:
                names = [
                    n
                    for component in denoiser_components(fam)
                    for n in derived_prequant_filenames(repo, scheme, component = component)
                ]
                names += [
                    row[-1] for row in getattr(fam, "prequant_filenames", ()) if row[0] == scheme
                ]
                add(repo, with_comfy_twins(names), owner, _HOSTED_DENOISER)
    except Exception as exc:  # noqa: BLE001 -- no table: those repos keep the pre-table handling
        logger.debug("Hosted denoiser table unavailable: %s", exc)

    try:
        from core.inference.diffusion_te_prequant import (
            TE_INT8_CONVROT_FILES,
            TE_INT8_CONVROT_VARIANT_FILES,
            te_prequant_repo_filenames,
        )

        for registry, fam in families:
            owner = (registry, str(fam.name))
            rows = [
                (str(r[2]), str(r[3] if len(r) > 3 else r[1]), str(r[1]))
                for r in getattr(fam, "te_prequant_repos", ()) or ()
                if len(r) >= 3
            ]
            rows += [
                (str(r[3]), str(r[2]), str(r[2]))
                for r in getattr(fam, "te_prequant_variant_repos", ()) or ()
                if len(r) >= 4
            ]
            for repo, component, slot in rows:
                for scheme in ("fp8", "int8"):
                    add(repo, te_prequant_repo_filenames(repo, component, scheme), owner, slot)
            for row in getattr(fam, "hosted_bf16_vae", ()) or ():
                if len(row) == 3:
                    add(row[1], [row[2]], owner, "vae")
        entries = [
            (f, c, e) for f, comps in TE_INT8_CONVROT_FILES.items() for c, e in comps.items()
        ]
        entries += [
            (f, c, e)
            for f, bases in TE_INT8_CONVROT_VARIANT_FILES.items()
            for comps in bases.values()
            for c, e in comps.items()
        ]
        for fam_name, slot, entry in entries:
            for owner in by_name.get(str(fam_name), ()):
                if len(entry) >= 2:
                    add(entry[0], [entry[1]], owner, str(slot))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Hosted text-encoder table unavailable: %s", exc)

    try:
        from core.inference.video_minimax_h3_te import H3_TE_QUANT_FILES, H3_TE_QUANT_REPO

        # Old released builds kept the identical conditioner outside official/; both locations are this asset.
        names = [
            n
            for name in H3_TE_QUANT_FILES.values()
            for n in (name, name.replace("/official/", "/"))
        ]
        for owner in by_name.get("minimax-h3", ()):
            add(H3_TE_QUANT_REPO, names, owner, "text_encoder")
    except Exception as exc:  # noqa: BLE001
        logger.debug("H3 conditioner table unavailable: %s", exc)
    try:
        from core.inference.video_ltx25 import LTX25_TE_HOSTED_FILE, LTX25_TE_HOSTED_REPO
        for owner in by_name.get("ltx-2.5", ()):
            add(LTX25_TE_HOSTED_REPO, [LTX25_TE_HOSTED_FILE], owner, "text_encoder")
    except Exception as exc:  # noqa: BLE001
        logger.debug("LTX-2.5 encoder table unavailable: %s", exc)

    image_bases = {_normalise(b) for b in _with_mirrors(_curated_base_ids())}
    return {repo: files for repo, files in table.items() if repo not in image_bases}


def _hosted_names(table, repo_id: str) -> Optional[set[str]]:
    files = table.get(_normalise(repo_id))
    return None if files is None else set(files)


def _family_tables() -> list[tuple[str, object]]:
    """``[(registry, family)]`` for the image and video tables; one missing registry never hides the other."""
    families: list[tuple[str, object]] = []
    try:
        from core.inference.diffusion_families import _FAMILIES as image_families
        families += [("image", fam) for fam in image_families]
    except Exception as exc:  # noqa: BLE001
        logger.debug("image family table unavailable: %s", exc)
    try:
        from core.inference.video_families import _FAMILIES as video_families
        families += [("video", fam) for fam in video_families]
    except Exception as exc:  # noqa: BLE001
        logger.debug("video family table unavailable: %s", exc)
    return families


def _base_families() -> dict[str, set[tuple[str, str]]]:
    """``{base_id_lower: {(registry, family)}}``: every identity of a curated base a hosted component is loaded
    beside. Only these ids pin a hosted denoiser; an arbitrary VAE repo never becomes a dependent by filename."""
    out: dict[str, set[tuple[str, str]]] = {}
    for registry, fam in _family_tables():
        bases = {
            getattr(fam, "base_repo", None),
            *(getattr(fam, "named_variant_bases", ()) or ()),
        }
        bases |= {row[0] for row in getattr(fam, "prequant_variant_repos", ()) or ()}
        bases |= {row[0] for row in getattr(fam, "te_prequant_variant_repos", ()) or ()}
        for base in _with_mirrors(b for b in bases if b):
            out.setdefault(_normalise(base), set()).add((registry, str(fam.name)))
    return out


def _checkpoint_families(repo_id: str, names: Iterable[Optional[str]]) -> set[tuple[str, str]]:
    """``(registry, family)`` of every family the image or video loader would resolve for this checkpoint."""
    found: set[tuple[str, str]] = set()
    try:
        from core.inference.video_families import detect_video_family
    except Exception:  # noqa: BLE001
        detect_video_family = None
    for name in dict.fromkeys([None, *(names or ())]):
        fam = _detect_family(repo_id, name)
        if fam is not None:
            found.add(("image", str(fam.name)))
        if detect_video_family is not None:
            for probe in (repo_id, name):
                try:
                    vfam = detect_video_family(probe) if probe else None
                except Exception:  # noqa: BLE001
                    vfam = None
                if vfam is not None:
                    found.add(("video", str(vfam.name)))
    return found


def _hosted_repo_owners(table, repo) -> set[tuple[str, str, str]]:
    """``(registry, family, kind)`` of every curated file actually cached in *repo*."""
    files = table.get(_normalise(str(getattr(repo, "repo_id", "") or ""))) or {}
    owners: set[tuple[str, str, str]] = set()
    for name in _snapshot_relative_names(repo):
        owners |= files.get(name.lower(), set())
    return owners


def _hosted_holds_unknown(table, repo) -> bool:
    """Whether a hosted repo holds a weight the table does not name (or, if also a base mirror, any foldered
    file): that may be a user install, so it is preserved."""
    key = _normalise(str(getattr(repo, "repo_id", "") or ""))
    names = _hosted_names(table, key) or set()
    try:
        from core.inference.diffusion_families import canonical_base, mirror_repo
        dual_role = canonical_base(key).lower() != key or mirror_repo(key) is not None
    except Exception:  # noqa: BLE001
        dual_role = True
    for name in _snapshot_relative_names(repo):
        lowered = name.lower()
        if lowered in names:
            continue
        if lowered.endswith(_WEIGHT_SUFFIXES) or (
            dual_role and ("/" in lowered or lowered in _PIPELINE_INDEXES)
        ):
            return True
    return False


_PIPELINE_INDEXES = ("model_index.json", "modular_model_index.json")


def _staged_pipeline(repo) -> bool:
    """A speed-load base copy: a weight, or the pipeline index plus component configs (hosted files replace shards)."""
    names = [n.lower() for n in _snapshot_relative_names(repo)]
    if any(n.endswith(_WEIGHT_SUFFIXES) for n in names):
        return True
    return any(n in _PIPELINE_INDEXES for n in names) and any(
        n.count("/") == 1 and n.endswith("/config.json") for n in names
    )


def repo_holds_denoiser(repo, *, folders: bool = True) -> bool:
    """Whether *repo* holds a runnable checkpoint, i.e. is a model the user installed. Beyond the name-only test: a single_file pick caches its checkpoint as ONE top-level file, ``sd_xl_base_1.0.safetensors`` for a whole-pipeline family and ``flux2-dev-fp8.safetensors`` for a transformer-only one, with no denoiser folder to recognise. Those repos are curated companion bases whose self-dependency is dropped, so without this the orphan listing called an installed checkpoint an unused leftover and Free up space deleted it. A companion-only fetch never lands a weight at the snapshot root, and the two ways to be wrong are not symmetric: an orphan we decline to offer costs disk, a checkpoint we offer costs the model. The one root weight that is NOT a checkpoint is the curated component repos' own asset, which is why they are excluded by id: a single-file VAE or text encoder is the companion, and reading it as an installed model is what kept an orphaned pair pinned to disk."""
    names = list(_snapshot_relative_names(repo))
    # ``folders=False``: root checkpoints only, for an artifact mirror, whose transformer/ shards are no install.
    if folders and any(holds_denoiser(name) for name in names):
        return True
    repo_id = str(getattr(repo, "repo_id", "") or "")
    hosted = _hosted_component_table()
    if _normalise(repo_id) in hosted:
        # Never an install of its own: only files the table cannot name keep it.
        return _hosted_holds_unknown(hosted, repo)
    if _normalise(repo_id) in _component_only_repo_ids():
        return False
    encoder_casts = _te_prequant_names(repo_id)
    for name in names:
        if "/" in name or not name.lower().endswith(_WEIGHT_SUFFIXES):
            continue
        # A pre-cast text encoder (``Qwen-Image-2.1-text_encoder-FP8.safetensors``) also sits at the root of a prequant repo and is a companion, not a checkpoint.
        if name.lower() in encoder_casts:
            continue
        if _detect_family(repo_id, name) is not None:
            return True
    return bool(_video_pick_names(repo_id, [n for n in names if n.lower() not in encoder_casts]))


# Diffusers / split-release component folders: their weights are companions, never a video pick.
_VIDEO_COMPONENT_DIRS = frozenset(
    {
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "text_encoders",
        "image_encoder",
        "feature_extractor",
        "latent_upsampler",
        "latent_upscale_models",
        "transformer",
        "transformer_2",
    }
)


def _video_pick_names(repo_id: str, names: Iterable[str]) -> list[str]:
    """Cached ``.safetensors`` a video ``single_file`` pick can name, as ``video._resolve_load_family_with_source``."""
    try:
        from core.inference.video_families import detect_video_family
    except Exception:  # noqa: BLE001
        return []
    picks = []
    for name in names:
        lowered = name.lower()
        if (
            not lowered.endswith(".safetensors")
            or lowered.split("/", 1)[0] in _VIDEO_COMPONENT_DIRS
        ):
            continue
        try:
            if detect_video_family(f"{repo_id}/{name}") is not None:
                picks.append(name)
        except Exception:  # noqa: BLE001
            continue
    return picks


def _te_prequant_names(repo_id: str) -> set[str]:
    """Lowercased root file names the loader uses for a pre-cast text encoder hosted in *repo_id*."""
    try:
        from core.inference.diffusion_families import _FAMILIES
        from core.inference.diffusion_te_prequant import te_prequant_companion_names
        return te_prequant_companion_names(repo_id, _FAMILIES)
    except Exception as exc:  # noqa: BLE001 -- no table means no exclusions, as before
        logger.debug("te prequant names unavailable: %s", exc)
        return set()


def _component_only_repo_ids() -> set[str]:
    """Curated sd.cpp component repo ids, under every identity they can be cached as."""
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        return {_normalise(r) for r in _with_mirrors(sd_cpp_companion_only_repo_ids())}
    except Exception as exc:  # noqa: BLE001 -- no table means no exclusions, as before
        logger.debug("sd.cpp companion table unavailable: %s", exc)
        return set()


def _is_checkpoint_pick_name(name: str) -> bool:
    """A cached file the loader would hand to ``detect_family_for_pick`` as the pick. Both kinds it accepts: a main GGUF anywhere in the repo, and a TOP-LEVEL single-file ``.safetensors``, which is how a transformer-only or whole-pipeline single_file pick is cached. Restricted to the snapshot root because that is where a single-file pick lands; a companion fetch's shards live under ``text_encoder/``, ``vae/`` and friends and say nothing about which checkpoint is installed."""
    lowered = name.lower()
    if lowered.endswith(".gguf"):
        return "mmproj" not in lowered
    return "/" not in name and lowered.endswith(".safetensors")


def _cached_checkpoint_pick_names(cache_scans) -> dict[str, list[str]]:
    """``{repo_id_lower: [every cached file the loader could pick]}`` for the family probe below. Both diffusion loaders detect a family from the FILENAME as well as the repo id, so a perfectly runnable ``some-owner/custom`` holding ``flux-2-klein-4b-Q4_K_M.gguf`` is a dependent that a repo-id-only probe cannot see. That matters most where there is nothing else to fall back on: on an upgraded install no links have been recorded yet, so the family probe is the whole guard. EVERY name, and across EVERY cache root: one generic repo can hold checkpoints of two different families, in separate subdirectories or in two remembered caches, and the loader will select any of them by file name, so probing one left the others' bases looking orphaned."""
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
    """Companion base ids *repo_id* could resolve to, from the family tables alone. Both identities: the upstream the tables key on and its ungated unsloth mirror, because ``prefer_ungated_mirror`` picks between them per fetch and either may be what landed on disk. ``gguf_filename`` is a cached checkpoint from this repo, passed for the same reason the loaders pass it: a repo whose family keyword appears only in the file name still loads, and a dependent this probe cannot see is a companion cleanup will happily delete underneath it."""
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
        # The loader resolves a base with this same call, so no family means no load and there is no dependent here to protect: the two fail together.
        return set()
    bases = {resolve_base_repo(fam, None)}
    bases |= _curated_variant_bases(fam, repo_id, gguf_filename)
    # The NATIVE engine never reads the diffusers base: it fetches a single-file VAE and text encoder from their own repos, and those repos are offerable for deletion, so a pre-existing native GGUF with no recorded link would have had its encoder listed as unused and removed underneath it. The recorded links cover a load that has happened since; this covers the install that predates them.
    bases |= {repo for repo, _file in _sd_cpp_component_specs(fam, repo_id, gguf_filename)}
    return _with_mirrors(bases)


def _curated_variant_bases(fam, repo_id: str, gguf_filename: Optional[str]) -> set[str]:
    """Curated bases of *fam* that the checkpoint's own identity names, beyond the family default. One family entry serves several sizes, and which one a pick uses comes from the Hub card's ``base_model`` tag. Offline that tag is gone, so ``resolve_base_repo(fam, None)`` answers the family DEFAULT: an installed ``unsloth/FLUX.2-klein-9B-GGUF`` derived as needing the 4B base, while its cached ``black-forest-labs/FLUX.2-klein-9B`` had no dependent at all, and since that base is in the curated offerable set, Free up space would list it before the checkpoint's first recorded load and delete the companions of a model that is still installed. Conservative on both ends: candidates come only from the curated tables, only from entries belonging to THIS family, and only when the checkpoint id or its GGUF file name names the base, so a wrong guess can add a dependent, never invent a repo, and naming one base too many costs a delete that is refused while naming one too few costs an installed model. Matched with punctuation folded away, the way family detection accepts its own aliases: ``flux1-dev-Q4_K_M.gguf`` resolves to the flux.1 family, and a literal comparison against the curated name ``FLUX.1-dev`` missed it over one dot, leaving the dev base offerable while the dev checkpoint was installed."""
    family = _fold(getattr(fam, "name", "") or "")
    if not family:
        return set()
    identity = f"{_fold(repo_id)} {_fold(gguf_filename or '')}"
    hits: list[tuple[str, str]] = []
    for candidate in _curated_base_ids():
        name = _fold(candidate.rsplit("/", 1)[-1])
        if name and name.startswith(family) and name in identity:
            hits.append((candidate, name))
    # Most specific spelling only: "zimage" is inside "zimageturbo" and "qwenimageedit" inside "qwenimageedit2511", so
    # a plain substring test made a Turbo / 2511 checkpoint depend on the Z-Image / Edit base as well, and since those
    # bases' mirrors are now the Z-Image / Edit GGUF repos, the delete of an unrelated GGUF repo was refused.
    names = {name for _c, name in hits}
    return {c for c, name in hits if not any(name != other and name in other for other in names)}


def _fold(text: str) -> str:
    """Lowercased with every separator dropped, so ``FLUX.1-dev`` and ``flux1-dev`` compare equal."""
    return "".join(c for c in (text or "").lower() if c.isalnum())


def _curated_base_ids() -> set[str]:
    """Curated family bases and their gated/mirror variants, as written in the tables."""
    ids: set[str] = set()
    try:
        from core.inference.diffusion_families import _FAMILIES
        ids |= {fam.base_repo for fam in _FAMILIES if getattr(fam, "base_repo", None)}
        # Named variants (Qwen-Image-2.1-Turbo) are bases too: a Turbo pick's grid comes from that repo's index.
        ids |= {b for fam in _FAMILIES for b in (getattr(fam, "named_variant_bases", ()) or ())}
    except Exception as exc:  # noqa: BLE001 -- no table means no extra candidates, as before
        logger.debug("Companion base table unavailable: %s", exc)
    ids |= _mirror_pair_ids()
    return ids


def _sd_cpp_component_specs(
    fam, repo_id: str, gguf_filename: Optional[str]
) -> set[tuple[str, str]]:
    """``(repo, filename)`` for every single-file VAE / text encoder an sd.cpp load of *fam* opens. The FILENAME matters as much as the repo: the Qwen-Image encoder is one named quant inside a chat GGUF repo, so a sibling quant in the same repo is not a substitute for it. Never raises."""
    specs: set[tuple[str, str]] = set()
    try:
        from core.inference.diffusion_families import sd_cpp_text_encoder_candidates

        vae = getattr(fam, "sd_cpp_vae", None)
        if vae and vae[0]:
            specs.add((vae[0], vae[1]))
        tokenizer = getattr(fam, "sd_cpp_tokenizer", None)
        if tokenizer and tokenizer[0]:
            specs.add((tokenizer[0], tokenizer[1]))
        # EVERY set the family could pick: there is no header to read and a renamed FLUX.2-klein 9B file carries no size token, so guessing answers 4B and leaves the 9B encoder unprotected.
        for encoder in sd_cpp_text_encoder_candidates(fam) or ():
            if encoder and encoder[0]:
                specs.add((encoder[0], encoder[1]))
    except Exception as exc:  # noqa: BLE001 -- one missing table never hides the rest
        logger.debug("sd.cpp component derivation unavailable for %s: %s", repo_id, exc)
    return specs


def required_companion_asset_files(cache_scans) -> dict[str, set[str]]:
    """``{repo_id_lower: {filename, ...}}`` the installed checkpoints' native loads actually open. The repo-level view cannot answer a VARIANT delete: native Qwen-Image opens exactly ``Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`` inside a chat GGUF repo, so removing that one quant strands the image checkpoint even though the repo still holds a Q8_0 nobody can substitute."""
    pick_names = _cached_checkpoint_pick_names(cache_scans)
    files: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
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
    """Every id one companion base can be cached under: itself, the upstream it copies, its ungated mirror, and the community repack that mirror stands in for. The repack matters as much as the mirror: ``prefer_cached_legacy_source`` deliberately sends the native fetch back to a repack an upgraded install already holds, so on those machines the bytes protecting an installed GGUF sit under the OLD repo key and nothing else names it. The live loaded-repo guard in the sd.cpp backend already expands all four; this is the same set, for the recorded links."""
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
    """Every curated image-base id (lowercased, including mirrors) and every hosted quantized component repo id. Deliberately NOT "anything a link ever named": orphan cleanup offers only repos this set recognises, so a mis-recorded link can never turn an unrelated repo into a delete candidate."""
    bases = _curated_base_ids()
    # The native engine's component-only repos ARE the companions for an sd.cpp pick, and the largest half of the footprint; leaving them out made them link-only strangers. Safe against a chat model borrowed as a text encoder, since the orphan listing skips any repo holding a GGUF.
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        bases |= set(sd_cpp_companion_only_repo_ids())
    except Exception as exc:  # noqa: BLE001 -- one missing table never hides the rest
        logger.debug("sd.cpp companion table unavailable: %s", exc)
    # Own id only: expanding a video -FP8 repo's mirrors would make the upstream base offerable.
    return {_normalise(b) for b in _with_mirrors(bases)} | set(_hosted_component_table())


def _mirror_pair_ids() -> set[str]:
    # The WHOLE table, gated and ungated: gating decides whether a fetch may override a user's cache, not whether a base can strand companions, so reading only the gated half would offer an installed checkpoint's companions for deletion.
    try:
        from core.inference.diffusion_families import _MIRROR_PAIRS
    except Exception:  # noqa: BLE001
        return set()
    return {rid for pair in _MIRROR_PAIRS for rid in pair}


def is_companion_base(repo_id: str) -> bool:
    """Whether deleting *repo_id* could strand an installed model's companions. Wider than :func:`known_companion_base_ids` on purpose, and the two are conservative in opposite directions: this one gates a REFUSAL, so it also admits any base a recorded link names, since a checkpoint whose base came from its card tag can point anywhere and missing it would leave exactly the hole this guard exists to close, while :func:`known_companion_base_ids` gates an OFFER TO DELETE, so it stays table-only and a bad link can never reach it."""
    key = _normalise(repo_id)
    if key in known_companion_base_ids():
        return True
    # Through the same identity expansion required_companion_bases uses, so the two agree WHICH copy is protected: comparing the literal id left an upgraded install's repack copy deletable.
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
    """``{base_repo_id_lower: {checkpoint repo ids that still need it}}``. Derived from *cache_scans*, what is installed at this instant, unioned with the recorded links of those same installed checkpoints. *ignore_repo_ids* is the delete being previewed: dropping it answers "what would still need this base AFTER the delete" without mutating anything. A repo never counts as its own dependent."""
    ignored = {_normalise(r) for r in ignore_repo_ids}
    links = read_companion_links()
    pick_names = _cached_checkpoint_pick_names(cache_scans)
    # Denoiser-gated, not id-gated: a VAE-only fetch from a prequant repo (unsloth/Qwen-Image-2.1-FP8) cannot load, so it pins nothing (#11825).
    checkpoints = _denoiser_holding_repo_ids(cache_scans)
    required: dict[str, set[str]] = {}
    for repo_id in _cached_model_repo_ids(cache_scans):
        key = _normalise(repo_id)
        if key in ignored or key not in checkpoints:
            continue
        bases = _family_bases_for_names(repo_id, pick_names.get(key, []))
        recorded = links.get(key)
        if recorded:
            bases |= _with_mirrors(recorded)
        # Canonical, not literal: a cached MIRROR resolves its own family back to the UPSTREAM id, which would make each identity a dependent of the other and leave a cache holding both stuck.
        # Except an artifact-hosting mirror (unsloth/Z-Image-GGUF): it is a checkpoint in its own right, and folding it
        # onto its upstream would leave a cached upstream base with no dependent, offered for deletion while the GGUF
        # still loads from it.
        self_keys = {key} if _hosts_artifacts(repo_id) else {key, _normalise(_canonical(repo_id))}
        for base in bases:
            base_key = _normalise(base)
            if not base_key or base_key in self_keys or _normalise(_canonical(base)) in self_keys:
                continue
            required.setdefault(base_key, set()).add(repo_id)
    _add_hosted_dependents(required, cache_scans, checkpoints, pick_names, ignored)
    return required


def _cached_hosted_owners(table, cache_scans) -> dict[str, set[tuple[str, str, str]]]:
    cached: dict[str, set[tuple[str, str, str]]] = {}
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                key = _normalise(str(getattr(repo, "repo_id", "") or ""))
                if str(getattr(repo, "repo_type", "")) != "model" or key not in table:
                    continue
                cached.setdefault(key, set()).update(_hosted_repo_owners(table, repo))
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
    return cached


def repo_is_install(repo, cache_scans) -> bool:
    """:func:`repo_holds_denoiser`, or a curated base whose family's hosted denoiser is cached (a speed load)."""
    if repo_holds_denoiser(repo):
        return True
    try:
        table = _hosted_component_table()
        key = _normalise(str(getattr(repo, "repo_id", "") or ""))
        families = _base_families().get(key) if key not in table else None
        if not families:
            return False
        cached = _cached_hosted_owners(table, cache_scans)
        if not _staged_pipeline(repo):
            return False
        return any(
            (r, f) in families and k == _HOSTED_DENOISER
            for owners in cached.values()
            for r, f, k in owners
        )
    except Exception:  # noqa: BLE001 -- fail closed: an unreadable cache counts as an install
        return True


def _add_hosted_dependents(required, cache_scans, checkpoints, pick_names, ignored) -> None:
    """Pin hosted components per cached FILE to installs that read them; only a curated base pins a hosted denoiser."""
    table = _hosted_component_table()
    if not table:
        return
    cached = _cached_hosted_owners(table, cache_scans)
    listed = {repo_key: set().union(*files.values()) for repo_key, files in table.items()}
    in_view = {(r, f) for repo_key in cached for r, f, _k in listed[repo_key]}
    denoisers = {
        (r, f) for owners in cached.values() for r, f, k in owners if k == _HOSTED_DENOISER
    }
    bases = _base_families()
    # {key: (display id, families, reads hosted denoisers)}, merged across cache roots.
    dependents: dict[str, tuple[str, set[tuple[str, str]], bool]] = {}

    def depend(key: str, repo_id: str, families: set, reads_denoisers: bool) -> None:
        shown, known, reads = dependents.get(key, (repo_id, set(), False))
        dependents[key] = (shown, known | families, reads or reads_denoisers)

    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                repo_id = str(getattr(repo, "repo_id", "") or "")
                key = _normalise(repo_id)
                if not key or key in ignored:
                    continue
                families = bases.get(key) if key not in table else None
                if key in checkpoints:
                    names = [
                        *pick_names.get(key, []),
                        *_video_pick_names(repo_id, _snapshot_relative_names(repo)),
                    ]
                    found = _checkpoint_families(repo_id, names)
                    depend(key, repo_id, found | (families or set()), bool(families))
                    continue
                # A hosted repo that is also a base mirror is a companion here, never its own speed load.
                if not families:
                    continue
                if not _staged_pipeline(repo):
                    continue
                if families & in_view and not families & denoisers:
                    # This family's hosted repos are cached and none holds its denoiser: a companion-only fetch.
                    continue
                depend(key, repo_id, families, True)
                if families & denoisers:
                    required.setdefault(key, set()).add(dependents[key][0])
            except Exception:  # noqa: BLE001
                continue
    for own, (repo_id, families, reads_denoisers) in dependents.items():
        for repo_key, owners in listed.items():
            if repo_key == own:
                continue
            if any(
                (r, f) in families and (reads_denoisers or k != _HOSTED_DENOISER)
                for r, f, k in (cached[repo_key] if repo_key in cached else owners)
            ):
                required.setdefault(repo_key, set()).add(repo_id)


def _hosts_artifacts(repo_id: str) -> bool:
    """``mirror_hosts_artifacts``, False when the family tables are unavailable."""
    try:
        from core.inference.diffusion_families import mirror_hosts_artifacts
        return mirror_hosts_artifacts(repo_id)
    except Exception:  # noqa: BLE001 -- no table means no artifact mirrors
        return False


def _canonical(repo_id: str) -> str:
    """``canonical_base``, degrading to the id itself when the family tables are unavailable."""
    try:
        from core.inference.diffusion_families import canonical_base
        return canonical_base(repo_id)
    except Exception:  # noqa: BLE001
        return repo_id
