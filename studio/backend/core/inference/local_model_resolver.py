# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve an OpenAI-request ``model`` string to a downloaded local GGUF.

Used by the opt-in auto-switch path. The match is conservative: only names
that map to an already-downloaded local GGUF (and a quant that is actually on
disk) are eligible, so an arbitrary OpenAI model string still falls through to
the loaded model (drop-in compat) and no surprise multi-GB download is ever
triggered. The local-model scan is cached for a few seconds since auto-switch
consults it per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from core.inference.model_ids import public_model_id
from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class _LocalGgufEntry:
    loader_id: str  # advertised id (repo id / folder name), also the override key
    load_path: str  # concrete on-disk dir/file passed to /load so it never downloads
    variants: tuple[str, ...]  # local quant labels; () for a standalone .gguf


_CACHE_TTL_S = 5.0
_lock = threading.Lock()
_scan: tuple[float, dict[str, _LocalGgufEntry]] = (0.0, {})
# Not _lock: that is held for the whole scan, so the request path would wait on it.
_warm_lock = threading.Lock()
# Repos that finished downloading but are not in the published index yet: nothing
# else covers them until the next scan, and the request path must not call them absent.
_just_downloaded: set[str] = set()
_warming = False
_last_scan_s = 0.0
# Rescan at most a tenth of the time: on the TTL alone a slow scan would run continuously.
_WARM_DUTY = 10.0


def _is_abs_path_id(value: str) -> bool:
    """True when an id is an absolute filesystem path (the ./models and LM Studio
    scanners use the on-disk path as the id) rather than a repo id like org/name.

    Both spellings count on every host. Path() follows the running OS, so a
    Windows backend read "/home/me/x.gguf" as relative and a POSIX one read
    "C:\\models\\x.gguf" the same way, and either then reached /v1/models as a
    published id. Ids outlive the machine that wrote them: settings sync, a WSL
    session and a copied config all carry the other platform's spelling, and the
    model-override identity already folds both. Neither reading can misfire on a
    repo id, which has no leading separator, drive or UNC prefix."""
    from pathlib import PurePosixPath, PureWindowsPath
    try:
        return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    except Exception:
        return False


def _advertised_loader_id(info) -> Optional[str]:
    """The id to advertise for a scanned model: prefer a client-facing alias over
    an absolute filesystem path so /v1/models and the override key never expose a
    host path (the ./models and LM Studio scanners report the path as info.id)."""
    raw_id = getattr(info, "id", None)
    if not raw_id or not _is_abs_path_id(raw_id):
        return raw_id
    for alt in (getattr(info, "model_id", None), getattr(info, "display_name", None)):
        if alt and not _is_abs_path_id(alt):
            return alt
    # No clean alias: strip to a path-free public id so a host path is never advertised.
    return public_model_id(raw_id) or raw_id


def _resolve_load_dir(p):
    """The concrete dir holding the GGUFs. For an HF cache repo (``models--*``
    with ``snapshots/``) this is the latest snapshot dir, so /load takes the
    local branch instead of the download-capable repo-id branch."""
    from pathlib import Path

    try:
        if (p / "snapshots").is_dir():
            from routes.models import _resolve_hf_cache_realpath
            real = _resolve_hf_cache_realpath(p)
            if real:
                return Path(real)
    except Exception:
        pass
    return p


def _local_gguf_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Build an entry only when GGUF quants are on disk (not Transformers/
    safetensors), listing only on-disk quants. ``load_path`` is a concrete local
    path so /load resolves the variant locally and never fetches a remote one."""
    from pathlib import Path
    from utils.models.model_config import detect_gguf_model, list_local_gguf_variants

    path = getattr(info, "path", None)
    if not isinstance(path, str):
        return None
    p = Path(path)
    try:
        if p.is_file():
            # A standalone .gguf loads by its own path; no quant sub-selection. An
            # mmproj companion (vision/audio projector) is not a servable model on
            # its own: _scan_models_dir's standalone-file pass does not filter it
            # the way the directory scan does, so reject it here or /v1/models would
            # advertise a projector and a switch could load it instead of the weights,
            # evicting the loaded model. The directory branch below is already mmproj
            # free (list_local_gguf_variants drops mmproj quants).
            if p.suffix.lower() != ".gguf" or detect_gguf_model(str(p)) is None:
                return None
            return _LocalGgufEntry(loader_id, str(p), ())
        load_dir = _resolve_load_dir(p)
        # It only descends for an HF cache repo, so a changed path IS one repo's
        # whole snapshot listing and gets the drafter reprieve; a plain folder does not.
        variants, _ = list_local_gguf_variants(str(load_dir), whole_repo = load_dir != p)
        quants = tuple(v.quant for v in variants if getattr(v, "quant", None))
        if not quants:
            return None
        # That call orders by descending size, so the head is the biggest quant (often
        # F16). Downstream reads [0], and a bare id must mean whichever quant a plain
        # load would take: answering with the largest can evict a model and then OOM.
        from core.inference.openai_auto_download import preferred_quant

        best = preferred_quant(quants)
        if best and quants[0] != best:
            quants = (best, *(q for q in quants if q != best))
        return _LocalGgufEntry(loader_id, str(load_dir), quants)
    except Exception:
        return None


def local_gguf_quants(info) -> Optional[tuple[str, ...]]:
    """On-disk quant labels for *info*, or None when it is not a servable local
    GGUF. Read from the files, not ``info.model_format``: the HF-cache scanner
    leaves that unset for GGUF snapshots, so filtering on it drops every cached
    GGUF. One scan tells /v1/models what it can serve and which quant to name."""
    from pathlib import Path

    path = getattr(info, "path", None)
    # Ollama-link entries come from a scanner _build_index intentionally skips (it
    # creates symlinks on the request path), so their advertised ids never resolve.
    # Don't report them as servable, or /v1/models would list unswitchable models.
    if isinstance(path, str) and any(
        seg in (".studio_links", "ollama_links") for seg in Path(path).parts
    ):
        return None
    entry = _local_gguf_entry(getattr(info, "id", "") or "", info)
    return entry.variants if entry is not None else None


def info_has_local_gguf(info) -> bool:
    """True when *info* points to on-disk GGUF weights the auto-switch path can load."""
    return local_gguf_quants(info) is not None


def _build_index() -> dict[str, _LocalGgufEntry]:
    """Map normalized id/model_id/display_name -> local GGUF entry.

    Scans the same roots Unsloth's model picker lists (./models, the active plus
    legacy/default HF caches, LM Studio dirs, and user scan folders) so a named
    local model is never missed and silently served as the loaded one. Ollama's
    scanner is skipped: it creates symlinks as a side effect and this runs on the
    request path.
    """
    # Lazy import: routes.models imports core.inference, so import at call time.
    from pathlib import Path
    from routes.models import (
        _scan_models_dir,
        _scan_hf_cache,
        _scan_lmstudio_dir,
        _resolve_hf_cache_dir,
        _is_hidden_model,
    )
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir, lmstudio_model_dirs
    from utils.hf_cache_settings import known_hf_hub_caches
    from core.inference.model_ids import public_model_id

    index: dict[str, _LocalGgufEntry] = {}
    seen_hf: set[str] = set()

    try:
        active_root = str(Path(_resolve_hf_cache_dir()).resolve())
    except Exception:
        active_root = None

    def _scan_hf_once(directory) -> list:
        if directory is None:
            return []
        try:
            d = Path(directory)
            if not d.is_dir():
                return []
            rp = str(d.resolve())
            if rp in seen_hf:
                return []
            seen_hf.add(rp)
            # Only the active cache loads by repo id. Say so, or an inactive repo is
            # indexed under an id it cannot load by, and its snapshot basename (what
            # /v1/models advertises once loaded by path) is never a key at all.
            # No format classification here: nothing on this path reads model_format,
            # and its recursive walk would duplicate the one _local_gguf_entry already
            # does per snapshot, on the request path.
            return _scan_hf_cache(directory, active_cache = rp == active_root, classify_format = False)
        except Exception as exc:  # a missing/malformed root must skip, never crash the index
            logger.debug("auto-switch: skipping HF cache dir %r: %s", directory, exc)
            return []

    # Each source is guarded on its own so one bad root (a permission error, a
    # malformed cache) drops only that source, not the whole index.
    found: list = []
    try:
        found += _scan_models_dir(Path("./models").resolve())
    except Exception as exc:
        logger.debug("auto-switch: ./models scan failed: %s", exc)
    try:
        for hf_dir in (
            *known_hf_hub_caches(),
            _resolve_hf_cache_dir(),
            legacy_hf_cache_dir(),
            hf_default_cache_dir(),
        ):
            found += _scan_hf_once(hf_dir)
    except Exception as exc:
        logger.debug("auto-switch: HF cache scan failed: %s", exc)
    try:
        for lm_dir in lmstudio_model_dirs():
            found += _scan_lmstudio_dir(lm_dir)
    except Exception as exc:
        logger.debug("auto-switch: LM Studio scan failed: %s", exc)
    try:
        from storage.studio_db import list_scan_folders
        for folder in list_scan_folders():
            try:
                fp = Path(folder["path"])
                found += (
                    _scan_models_dir(fp, limit = 200) + _scan_hf_once(fp) + _scan_lmstudio_dir(fp)
                )
            except Exception as exc:
                logger.debug("auto-switch: scan folder %r failed: %s", folder, exc)
    except Exception as exc:
        logger.debug("auto-switch: scan folders enumerate failed: %s", exc)
    for info in found:
        raw_id = getattr(info, "id", None)
        if not raw_id:
            continue
        # Skip what Unsloth hides from its pickers (validation probe, RAG embed
        # weights): not chat models, so never an auto-switch target.
        if _is_hidden_model(
            raw_id,
            getattr(info, "model_id", None),
            getattr(info, "path", None),
        ):
            continue
        # Advertise a client-facing alias, not an absolute filesystem path.
        loader_id = _advertised_loader_id(info)
        entry = _local_gguf_entry(loader_id, info)
        if entry is None:
            continue
        # Index every alias (including the path) so a client can resolve by any of
        # them, even though only the non-path loader_id is advertised.
        for key in (
            raw_id,
            getattr(info, "model_id", None),
            getattr(info, "display_name", None),
            public_model_id(raw_id),
        ):
            if key:
                index.setdefault(key.strip().lower(), entry)
        # Other revisions of the same repo resolve to their own weights, so a pin on
        # one keeps working after Hugging Face writes a newer snapshot.
        for name, sibling_entry in _sibling_revision_entries(raw_id, loader_id):
            index.setdefault(name.strip().lower(), sibling_entry)
    return index


def _sibling_revision_entries(raw_id: str, loader_id: str):
    """Yield ``(revision_name, entry)`` for the repo's OTHER cached revisions.

    An inactive-cache repo carries its snapshot path as the id, and /v1/models
    advertises only that directory's basename once loaded, so anything durable
    pinned to it (a subagent config) holds one revision hash. Hugging Face writes a
    new snapshot dir on every update, and the scan emits a single entry per repo
    pointed at the newest one, so that pin would otherwise stop resolving and drop
    through to whatever model is loaded.

    Each revision gets an entry for its OWN directory rather than an alias onto the
    scanned one: aliasing would redirect a pin that names an older complete revision
    onto a newer half-downloaded snapshot and break a request that works today.
    Incomplete revisions are skipped for the same reason.

    Sibling names are only revisions inside a real cache repo
    (``<root>/models--org--name/snapshots/<rev>``). A scan folder that merely happens
    to be called ``snapshots`` holds unrelated models, and treating those as
    revisions would silently serve one model in place of another.
    """
    from pathlib import Path
    from types import SimpleNamespace

    snapshots = Path(raw_id).parent
    if snapshots.name != "snapshots" or not snapshots.parent.name.startswith("models--"):
        return
    from routes.models import snapshot_variants_all_complete

    try:
        siblings = [p for p in snapshots.iterdir() if p.is_dir() and p.name != Path(raw_id).name]
    except OSError:
        return
    for sibling in siblings:
        if not snapshot_variants_all_complete(str(sibling)):
            continue
        entry = _local_gguf_entry(loader_id, SimpleNamespace(path = str(sibling)))
        if entry is not None:
            yield sibling.name, entry


def note_downloaded(repo_id: Optional[str]) -> None:
    """Record a repo as present ahead of the scan that will index it."""
    if not repo_id:
        return
    with _lock:
        _just_downloaded.add(repo_id.strip().lower())


def recently_downloaded(repo_id: str) -> bool:
    """Whether *repo_id* finished downloading since the last completed scan."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        return False
    return repo_id.strip().lower() in _just_downloaded


def invalidate_index() -> None:
    """Mark the cached scan stale so the next resolve sees a just-finished download
    instead of waiting out the TTL.

    Keeps the entries: the request path reads this cache without scanning, so
    emptying it would leave it with no evidence about any local model until the
    rebuild lands, and a bare request for one would be answered by whatever is
    resident. Only a completed download invalidates, and that only adds, so the
    retained entries stay true.
    """
    global _scan
    with _lock:
        _scan = (0.0, _scan[1])


def _index() -> dict[str, _LocalGgufEntry]:
    global _scan
    # Build under the lock so concurrent callers with an expired cache don't all
    # run the (multi-dir) scan at once; the rest wait and reuse the fresh result.
    with _lock:
        now = time.monotonic()
        ts, cached = _scan
        if now - ts < _CACHE_TTL_S:
            return cached
        fresh = _build_index()
        # Stamp AFTER the scan, not with the pre-scan ``now``: a multi-root scan on
        # an install with many local models can itself exceed the TTL, which would
        # store the cache already expired and make every request rebuild the index.
        _scan = (time.monotonic(), fresh)
        # The scan supersedes the notes: whatever landed is in the index now.
        _just_downloaded.clear()
        return fresh


def index_is_built() -> bool:
    """Whether a scan has ever completed, freshness aside.

    Lock-free on purpose: ``_lock`` is held for the whole scan, so taking it would
    park the request path on the scan it is trying to stay off. Safe because
    ``_scan`` is only ever rebound, never mutated.
    """
    return bool(_scan[0])


def warm_index_soon() -> None:
    """(Re)build the index off the request path when it is missing or past its TTL.

    The only refresh for callers using ``allow_scan=False``. Covers a stale index,
    not just an absent one: a model downloaded through the Hub UI or dropped into a
    scan folder has no invalidation hook and would otherwise stay invisible to them
    for the life of the process. Never blocks, and never touches ``_lock``.
    """
    global _warming
    if time.monotonic() - _scan[0] < max(_CACHE_TTL_S, _last_scan_s * _WARM_DUTY):
        return
    with _warm_lock:
        if _warming:
            return
        _warming = True

    def _run() -> None:
        global _warming, _last_scan_s
        started = time.monotonic()
        try:
            _index()
        except Exception:
            pass
        finally:
            _last_scan_s = time.monotonic() - started
            with _warm_lock:
                _warming = False

    threading.Thread(target = _run, name = "local-model-index-warm", daemon = True).start()


def resolve_local_gguf(
    requested: str, *, allow_scan: bool = True
) -> Optional[tuple[str, Optional[str], str]]:
    """Return ``(load_path, gguf_variant, loader_id)`` for a local match, else None.

    ``load_path`` is the concrete on-disk path to hand /load (so it never fetches
    a remote), ``loader_id`` is the advertised id used as the launch-override key.
    ``requested`` is ``repo`` or ``repo:VARIANT``. An exact id match wins first
    (so ids containing a colon still resolve); else the last ``:VARIANT`` is split
    off and resolves only when that quant is on disk, unless it names no quant at
    all (an Ollama-style ":latest"), which means the repo.

    ``allow_scan=False`` answers from the last built index and never rebuilds, for
    the request path: the scan walks several model dirs and HF caches, takes seconds
    on a large install, and holds a lock everyone queues behind. Stale is fine there,
    since disk barely moves and a finished download calls :func:`invalidate_index`.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None
    requested = requested.strip()
    try:
        index = _index() if allow_scan else _scan[1]
        entry = index.get(requested.lower())
        if entry is not None:
            variant = entry.variants[0] if entry.variants else None
            return entry.load_path, variant, entry.loader_id

        base, sep, variant = requested.rpartition(":")
        if not sep:
            return None
        entry = index.get(base.strip().lower())
        if entry is None:
            return None
        wanted = variant.strip().lower()
        for v in entry.variants:
            if v.lower() == wanted:
                return entry.load_path, v, entry.loader_id
        from core.inference.openai_auto_download import looks_like_quant

        if looks_like_quant(variant):
            return None
        # ":latest" or ":8b" names no file, so it means the repo; a real quant that
        # is not on disk still misses, or a swap would serve the wrong weights.
        return entry.load_path, (entry.variants[0] if entry.variants else None), entry.loader_id
    except Exception:
        # Best-effort: any resolver failure falls through to the loaded model,
        # so a malformed name can never turn a servable request into a 500.
        return None


MISS_MODEL_NOT_FOUND = "model_not_found"
MISS_VARIANT_NOT_FOUND = "variant_not_found"


def describe_local_miss(requested: str) -> tuple[str, tuple[str, ...]]:
    """Why :func:`resolve_local_gguf` missed, so an error can say "wrong quant"
    instead of "no such model".

    ``(MISS_VARIANT_NOT_FOUND, <local quants>)`` when the repo is downloaded but the
    requested ``:VARIANT`` is not, else ``(MISS_MODEL_NOT_FOUND, ())``. Fail-safe: a
    scan failure reports the generic miss rather than raising into the handler.
    """
    if not isinstance(requested, str) or not requested.strip():
        return MISS_MODEL_NOT_FOUND, ()
    base, sep, variant = requested.strip().rpartition(":")
    from core.inference.openai_auto_download import looks_like_quant

    # Split like the resolver or the two disagree: a tag naming no quant means the
    # repo there, so reporting a missing quant for it would name one nobody asked for.
    if not sep or not looks_like_quant(variant):
        return MISS_MODEL_NOT_FOUND, ()
    try:
        entry = _index().get(base.strip().lower())
    except Exception:
        return MISS_MODEL_NOT_FOUND, ()
    if entry is None or not entry.variants:
        return MISS_MODEL_NOT_FOUND, ()
    return MISS_VARIANT_NOT_FOUND, entry.variants
