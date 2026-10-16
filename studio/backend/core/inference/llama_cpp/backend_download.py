# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: variant fallback and Hugging Face download."""

import os
import shutil
import threading
import time

from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Optional,
)
from hub.utils.hf_tokens import (
    call_hub_with_anonymous_retry,
    hf_token_rejected_hint,
    is_rejected_credential_error,
)
from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

from .core_types import (
    GgufDownloadCancelled,
    _PERMANENT_HUB_ERRORS,
    _UNRECOVERABLE_LOCAL_ERRNOS,
)
from .draft_args import _extra_args_device
from .drafters import (
    _is_big_endian_gguf_path,
    _is_companion_gguf_path,
    _is_dspark_drafter_path,
    _is_mtp_only_drafter_path,
    _is_root_dflash_drafter_path,
)
from .gguf_cache import (
    _SHARD_FULL_RE,
    _SHARD_RE,
    _cached_candidate_matches_revision_size,
    _cached_complete_candidate,
    _cached_hf_snapshot_file,
    _companion_snapshot_sibling,
    _drafter_set_bytes,
    _gguf_extra_shards,
    _gguf_snapshot_files,
    _hub_cache_dir_for_snapshot_path,
    _hub_download_in_flight,
    _is_published_mtp_drafter_name,
    _mtp_head_borrows,
    _pick_dspark,
    _pick_mmproj,
    _pick_mtp,
    _pick_mtp_root_only,
    _resolve_repo_id_casing,
    _resolve_variant_gguf_files,
    cached_gguf_for_load,
)
from .hf_offline import _hf_env_offline
from .log import logger
from .memory_policy import _extra_args_set_spec_type


class _DownloadMixin:
    # ── Variant fallback ────────────────────────────────────────────

    @staticmethod
    def _find_fitting_variant(
        hf_repo: str,
        free_bytes: int,
        requested_bytes: int,
        hf_token: Optional[str] = None,
        requested_file: Optional[str] = None,
    ) -> Optional[tuple[str, int, list[str]]]:
        """Find a GGUF variant (including all shards) smaller than the requested one.

        The largest that leaves ``_DISK_RESERVE_BYTES`` free, else the fewest new bytes.

        Returns (first_shard_filename, total_size_bytes, extra_shards) or None.
        """
        try:
            from huggingface_hub import get_paths_info, list_repo_files
            from core.inference.openai_auto_download import _DISK_RESERVE_BYTES

            files = call_hub_with_anonymous_retry(list_repo_files, hf_token, hf_repo)
            from hub.utils.gguf import drop_shadowed_appledouble_names, gguf_checkpoint_family

            gguf_files = [
                f
                for f in drop_shadowed_appledouble_names(list(files))
                if f.lower().endswith(".gguf")
                and not _is_companion_gguf_path(f)
                and not _is_big_endian_gguf_path(f)
            ]
            # A sibling checkpoint's smaller quant is different weights, not a fallback.
            if requested_file is not None:
                checkpoint = gguf_checkpoint_family(requested_file)
                gguf_files = [f for f in gguf_files if gguf_checkpoint_family(f) == checkpoint]
            if not gguf_files:
                return None

            # Sizes for all GGUF files
            path_infos = list(
                call_hub_with_anonymous_retry(get_paths_info, hf_token, hf_repo, gguf_files)
            )
            size_map = {p.path: (p.size or 0) for p in path_infos}

            # Group by variant: shards share a prefix before -NNNNN-of-NNNNN
            variants: dict[str, list[str]] = {}
            for f in gguf_files:
                m = _SHARD_RE.match(f)
                key = m.group(1) if m else f
                variants.setdefault(key, []).append(f)

            # Sum shard sizes per variant, track the first shard (for download)
            fitting: list[tuple[str, int, list[str], int]] = []
            for key, shard_files in variants.items():
                total = sum(size_map.get(f, 0) for f in shard_files)
                first = sorted(shard_files)[0]
                extra = [path for path in sorted(shard_files) if path != first]
                cached = _cached_complete_candidate(hf_repo, first, extra)
                needed = (
                    0
                    if cached and _cached_candidate_matches_revision_size(hf_repo, cached, hf_token)
                    else total
                )
                if 0 < total < requested_bytes and needed <= free_bytes:
                    fitting.append((first, total, extra, needed))
            if not fitting:
                return None

            roomy = [v for v in fitting if v[3] + _DISK_RESERVE_BYTES <= free_bytes]
            # Only cached variants can tie on new bytes, and the larger costs no more.
            first_file, total_size, extra_shards, _ = (
                max(roomy, key = lambda v: v[1])
                if roomy
                else min(fitting, key = lambda v: (v[3], -v[1]))
            )
            return first_file, total_size, extra_shards
        except Exception:
            return None

    # ── HF download (no lock held) ───────────────────────────────

    def _download_gguf(
        self,
        *,
        hf_repo: str,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        force: bool = False,
        allow_smaller_fallback: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Download GGUF file(s) from HuggingFace. Returns local path.

        Runs WITHOUT self._lock so unload_model() can set _cancel_event at
        any time; checks it between each shard download.

        ``force`` re-fetches even when a (possibly stale) blob is cached.
        ``allow_smaller_fallback=False`` raises on low disk instead of silently
        switching to a smaller quant. ``cancel_event`` overrides
        ``self._cancel_event`` so an update can use a private event without
        touching the shared one; defaults to the shared event.
        """
        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            raise GgufDownloadCancelled("Cancelled")
        from utils.hf_cache_settings import get_hf_cache_paths

        download_cache_dir = str(get_hf_cache_paths().hub_cache)
        try:
            import huggingface_hub  # noqa: F401 -- presence check only
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for HF model loading. "
                "Install it with: pip install huggingface_hub"
            )

        resolved_hf_repo = _resolve_repo_id_casing(hf_repo)
        if resolved_hf_repo != hf_repo:
            logger.info(
                "Using cached repo_id casing '%s' for requested '%s'",
                resolved_hf_repo,
                hf_repo,
            )
            hf_repo = resolved_hf_repo

        # Resolve the filename from the variant (shared with the pre-teardown header probe)
        gguf_filename, gguf_extra_shards = _resolve_variant_gguf_files(
            hf_repo, hf_variant, hf_token
        )

        # Prefer the existing model. Updates use force=True to fetch a new revision.
        if not force:
            if hf_variant:
                # Resolve by variant so a newer revision's filename does not hide
                # the complete older copy. Size-check against that older snapshot's
                # own revision when its metadata remains available.
                cached_main = cached_gguf_for_load(
                    hf_repo,
                    hf_variant,
                    verify_sizes = True,
                    hf_token = hf_token,
                )
            else:
                candidate = _cached_complete_candidate(hf_repo, gguf_filename, gguf_extra_shards)
                cached_main = (
                    candidate[0]
                    if candidate is not None
                    and _cached_candidate_matches_revision_size(hf_repo, candidate, hf_token)
                    else None
                )
            if cached_main is not None:
                logger.info(f"Reusing cached GGUF: {cached_main}")
                return cached_main

        # Check disk space; fall back to a smaller variant if needed
        all_gguf_files = [gguf_filename] + gguf_extra_shards
        try:
            from huggingface_hub import get_paths_info, try_to_load_from_cache

            path_infos = list(
                call_hub_with_anonymous_retry(get_paths_info, hf_token, hf_repo, all_gguf_files)
            )
            total_bytes = sum((p.size or 0) for p in path_infos)

            # Subtract bytes already in the HF cache so we only preflight
            # against what we must download. Without this, re-loading a
            # cached large model (e.g. MiniMax-M2.7-GGUF at 131 GB) fails
            # cold whenever free disk is below the full weight footprint,
            # even though nothing needs downloading.
            already_cached_bytes = 0
            # Count only files that can resume this download.
            offline = _hf_env_offline()
            # Offline split sets are reusable only when every shard shares a snapshot.
            split_needs_refetch = bool(offline and not force and gguf_extra_shards)
            if not force and not split_needs_refetch:
                for p in path_infos:
                    if not p.size:
                        continue
                    try:
                        cached_path = try_to_load_from_cache(
                            hf_repo,
                            p.path,
                            cache_dir = download_cache_dir,
                        )
                    except Exception:
                        cached_path = None
                    if (
                        not (isinstance(cached_path, str) and os.path.exists(cached_path))
                        and offline
                    ):
                        cached_path = _cached_hf_snapshot_file(
                            hf_repo,
                            p.path,
                            expected_size = p.size,
                            cache_dir = download_cache_dir,
                        )
                    if isinstance(cached_path, str) and os.path.exists(cached_path):
                        try:
                            on_disk = os.path.getsize(cached_path)
                        except OSError:
                            on_disk = 0
                        # Satisfied only when the full blob is present.
                        if on_disk >= p.size:
                            already_cached_bytes += p.size

            total_download_bytes = max(0, total_bytes - already_cached_bytes)

            if total_download_bytes > 0:
                Path(download_cache_dir).mkdir(parents = True, exist_ok = True)
                free_bytes = shutil.disk_usage(download_cache_dir).free

                total_gb = total_download_bytes / 1e9
                free_gb = free_bytes / 1e9
                cached_gb = already_cached_bytes / 1e9

                logger.info(
                    f"GGUF download: {total_gb:.1f} GB needed "
                    f"({cached_gb:.1f} GB already cached), "
                    f"{free_gb:.1f} GB free on disk"
                )

                if total_download_bytes > free_bytes:
                    if not allow_smaller_fallback:
                        # Update path: never silently switch to a smaller quant;
                        # surface the disk shortfall for the requested variant.
                        raise RuntimeError(
                            f"Not enough disk space to download {gguf_filename}. "
                            f"Only {free_gb:.1f} GB free in {download_cache_dir}"
                        )
                    smaller = self._find_fitting_variant(
                        hf_repo,
                        free_bytes,
                        total_bytes,
                        hf_token,
                        requested_file = gguf_filename,
                    )
                    if smaller:
                        fallback_file, fallback_size, fallback_shards = smaller
                        from utils.models.model_config import _gguf_variant_key

                        fallback_variant = _gguf_variant_key(fallback_file)
                        notice = (
                            f"Not enough disk space to download {hf_variant} "
                            f"({total_gb:.1f} GB needed, {free_gb:.1f} GB free), so "
                            f"{fallback_variant} ({fallback_size / 1e9:.1f} GB) "
                            "was loaded instead."
                        )
                        logger.warning(notice)
                        self._pending_variant_fallback = (fallback_variant, notice)
                        gguf_filename = fallback_file
                        gguf_extra_shards = fallback_shards

                        # The selected fallback is a new load target. Apply the
                        # same any-revision reuse policy before starting a fetch.
                        fallback_candidate = _cached_complete_candidate(
                            hf_repo, gguf_filename, gguf_extra_shards
                        )
                        if fallback_candidate is not None and (
                            _cached_candidate_matches_revision_size(
                                hf_repo, fallback_candidate, hf_token
                            )
                        ):
                            logger.info(f"Reusing cached fallback GGUF: {fallback_candidate[0]}")
                            return fallback_candidate[0]
                    else:
                        raise RuntimeError(
                            f"Not enough disk space to download any variant. "
                            f"Only {free_gb:.1f} GB free in {download_cache_dir}"
                        )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        gguf_label = f"{hf_repo}/{gguf_filename}" + (
            f" (+{len(gguf_extra_shards)} shards)" if gguf_extra_shards else ""
        )
        logger.info(f"Resolving GGUF: {gguf_label}")
        try:
            if cancel_event.is_set():
                raise GgufDownloadCancelled("Cancelled")
            dl_start = time.monotonic()
            # Xet primary, HTTP fallback on stall; per-file so finished shards stay cached.
            local_path = hf_hub_download_with_xet_fallback(
                hf_repo,
                gguf_filename,
                hf_token,
                cancel_event = cancel_event,
                on_status = lambda m: logger.info(m),
                force_download = force,
                cache_dir = download_cache_dir,
            )
            for shard in gguf_extra_shards:
                if cancel_event.is_set():
                    raise GgufDownloadCancelled("Cancelled")
                logger.info(f"Resolving GGUF shard: {shard}")
                hf_hub_download_with_xet_fallback(
                    hf_repo,
                    shard,
                    hf_token,
                    cancel_event = cancel_event,
                    force_download = force,
                    cache_dir = download_cache_dir,
                )
        except Exception as e:
            if isinstance(e, GgufDownloadCancelled):
                raise
            if isinstance(e, RuntimeError) and "Cancelled" in str(e):
                raise GgufDownloadCancelled(str(e)) from e
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
                # Only when a token went out: an anonymous 401 is a private or missing repo.
                f"{hf_token_rejected_hint(e) if is_rejected_credential_error(e, hf_token) else ''}"
            )

        dl_elapsed = time.monotonic() - dl_start
        if dl_elapsed < 2.0:
            logger.info(f"GGUF resolved from cache: {local_path}")
        else:
            logger.info(f"GGUF downloaded in {dl_elapsed:.1f}s: {local_path}")
        return local_path

    def _download_companion_gguf(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str],
        pick: Callable[[list[str]], Optional[str]],
        label: str,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
        reuse_snapshot_sibling: bool = True,
        outcome: Optional[dict] = None,
        on_transient_failure: Optional[Callable[[], None]] = None,
    ) -> Optional[str]:
        """Resolve and fetch a companion GGUF (mmproj / MTP drafter) by name.

        Prefers a companion co-located with ``near_path``'s cache snapshot,
        then tries the live repo file list, then the local HF cache snapshots
        (offline, same fallback as _download_gguf), then hf_hub_download.
        Runs WITHOUT self._lock (like _download_gguf); honors _cancel_event so
        an /unload between the main download and here skips the fetch.
        ``cancel_event`` overrides ``self._cancel_event`` (defaults to it).

        Split-aware, like the main-model download: a companion published as a
        split GGUF is only usable as a complete set, since llama-server resolves
        the sibling shards from the first one's directory. Fetching just the
        picked shard left a drafter whose header reads fine and which the server
        then cannot open, so the load fell back to no speculation with nothing to
        show for the download -- and it disagreed with the local scan, which
        accepts a split drafter only when every shard is present.

        ``reuse_snapshot_sibling=False`` is for a caller that already rejected the
        snapshot copy; without it this repeats the lookup and hands it back. ``near_path``
        is still passed: it also selects the cache directory the fetch writes into.

        ``on_transient_failure`` fires when the companion was lost to a listing that
        never completed or a download that dropped, the one None worth another attempt.
        A callback rather than an ``outcome`` key because the DFlash caller loops and
        needs the answer per attempt. Permanent errors, offline and cancellation do not
        fire it.
        """
        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            return None

        # The listing that produced the pick, kept so the shard siblings can be
        # resolved from the same names the pick chose among.
        available: list[str] = []

        def _pick_from(names: list[str]) -> Optional[str]:
            nonlocal available
            available = list(names)
            return pick(available)

        # Keep companion files in the main GGUF's snapshot.
        if near_path and reuse_snapshot_sibling:
            cached = _companion_snapshot_sibling(near_path, pick)
            if cached:
                logger.info("Reusing cached %s: %s", label, cached)
                return cached

        from utils.hf_cache_settings import get_hf_cache_paths

        companion_cache_dir = _hub_cache_dir_for_snapshot_path(near_path) or str(
            get_hf_cache_paths().hub_cache
        )

        if _hub_download_in_flight(hf_repo):
            logger.info("Skipping %s download while a hub download is active", label)
            return None

        target: Optional[str] = None
        # Whether the question was actually answered. A listing that never
        # completed (offline, transient Hub failure) leaves target None for a
        # reason that says nothing about the repo's contents.
        listing_answered = False
        # Whether the listing died of something retryable. Distinct from
        # listing_answered: a permanent error also leaves the listing unanswered.
        listing_failed = False
        from huggingface_hub import list_repo_files

        # Retry a transient listing blip; permanent repo/auth errors and offline
        # mode are not retried (offline raises at once -> fall through to cache).
        for attempt in range(3):
            if cancel_event.is_set():
                return None
            try:
                target = _pick_from(
                    call_hub_with_anonymous_retry(list_repo_files, hf_token, hf_repo)
                )
                listing_answered = True
                listing_failed = False
                break
            except Exception as e:
                if type(e).__name__ in _PERMANENT_HUB_ERRORS:
                    logger.debug(f"Could not list repo files for {label}: {e}")
                    break
                listing_failed = True
                logger.debug(
                    f"Could not list repo files for {label} (attempt {attempt + 1}/3): {e}"
                )
                if attempt < 2:
                    cancel_event.wait(2**attempt)

        # Captured before the cache fallback can name a file the live revision dropped.
        listed_live = (target is not None) if listing_answered else None

        if target is None:
            try:
                from utils.models.model_config import _iter_hf_cache_snapshots
                for snap in _iter_hf_cache_snapshots(hf_repo, companion_cache_dir):
                    rel_files = _gguf_snapshot_files(snap)
                    target = _pick_from(rel_files)
                    if target is not None:
                        logger.info("Resolved %s %s from local HF cache", label, target)
                        break
            except Exception as e:
                logger.debug(f"Offline cache lookup for {label} failed: {e}")

        # "The repo publishes none" and "the fetch failed" both return None, but
        # only the second is worth retrying on the next Apply. Left unset unless
        # the question was really answered -- by a listing that completed, or by
        # a cache hit that produced a target regardless -- so an unreachable Hub
        # is not recorded as a permanent absence.
        if (
            outcome is not None
            and not cancel_event.is_set()
            and (listing_answered or target is not None)
        ):
            outcome["listed"] = target is not None
        if outcome is not None and not cancel_event.is_set() and listed_live is not None:
            outcome["listed_live"] = listed_live
        if target is None or cancel_event.is_set():
            # The listing is the only step that can fail this far in.
            if target is None and listing_failed and not cancel_event.is_set():
                self._report_transient_companion_failure(on_transient_failure, label)
            return None

        # Offline, resolve the companion straight from the cache snapshot that
        # holds it. resolve_cached_repo_id_case can return a partial lower-case
        # spelling when any dir exists under the requested casing, so calling
        # hf_hub_download with hf_repo would miss the canonical file and silently
        # drop the companion. _cached_hf_snapshot_file scans every case variant.
        if _hf_env_offline():
            cached = _cached_hf_snapshot_file(
                hf_repo,
                target,
                cache_dir = companion_cache_dir,
            )
            if cached:
                from utils.models.model_config import _drafter_split_is_complete

                # Same whole-set rule as the snapshot reuse above: half a split
                # companion is not a companion, and offline there is no fetch to
                # complete it, so answering None leaves the load without a
                # drafter instead of with one llama-server cannot open.
                if _drafter_split_is_complete(Path(cached)):
                    logger.info("Resolved %s from local HF cache: %s", label, cached)
                    return cached

        # A split companion is one file to pick and N files to fetch. Same helper
        # the main-model download resolves its shards with, so the two cannot
        # disagree about what belongs to a set; empty for the single-file case,
        # which is every mmproj and every published sidecar so far.
        extra_shards = _gguf_extra_shards(available, target)
        # The filename carries the set size, so a short listing is answerable before
        # the download: half a split companion is not one, as the reuse paths above say.
        _shard_total = _SHARD_FULL_RE.match(target)
        if _shard_total and len(extra_shards) + 1 != int(_shard_total.group(3)):
            logger.info(
                "Skipping %s: %s lists %d of %s shards.",
                label,
                hf_repo,
                len(extra_shards) + 1,
                _shard_total.group(3),
            )
            # Settled, not retryable: the listing answered, so the caller records
            # absence rather than reloading on every Apply.
            if outcome is not None:
                outcome["listed"] = False
                outcome["listed_live"] = False
            return None
        try:
            logger.info(f"Downloading {label}: {hf_repo}/{target}")
            # Same policy; companions are best-effort (caller below swallows failures to None).
            local_path = hf_hub_download_with_xet_fallback(
                hf_repo,
                target,
                hf_token,
                cancel_event = cancel_event,
                cache_dir = companion_cache_dir,
            )
            for shard in extra_shards:
                if cancel_event.is_set():
                    return None
                logger.info(f"Downloading {label} shard: {hf_repo}/{shard}")
                hf_hub_download_with_xet_fallback(
                    hf_repo,
                    shard,
                    hf_token,
                    cancel_event = cancel_event,
                    cache_dir = companion_cache_dir,
                )
            return local_path
        except Exception as e:
            logger.warning(f"Could not download {label}: {e}")
            # The listing named the file, so this is the fetch dropping: retryable
            # unless the Hub or this machine answered for good, or we were cancelled.
            if (
                not cancel_event.is_set()
                and type(e).__name__ not in _PERMANENT_HUB_ERRORS
                and getattr(e, "errno", None) not in _UNRECOVERABLE_LOCAL_ERRNOS
                and not _hf_env_offline()
            ):
                self._report_transient_companion_failure(on_transient_failure, label)
            return None

    @staticmethod
    def _report_transient_companion_failure(
        on_transient_failure: Optional[Callable[[], None]], label: str
    ) -> None:
        """Tell the caller its companion was lost to something retryable.

        Best-effort both ways: the callback is optional, and one that raises must not
        lose the load over bookkeeping for an already best-effort companion.
        """
        if on_transient_failure is None:
            return
        try:
            on_transient_failure()
        except Exception as exc:
            logger.debug("Could not record the transient %s failure: %s", label, exc)

    def _download_mmproj(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
    ) -> Optional[str]:
        """Download the mmproj (vision projection) file from a GGUF repo.

        Prefers mmproj-F16.gguf, else any mmproj*.gguf. Returns the local
        path, or None if none exists. ``cancel_event`` overrides
        ``self._cancel_event`` (defaults to it). ``near_path`` prefers a
        copy co-located with the main GGUF's cache snapshot.

        Falls back to a projector hand-added beside the cached weight, only
        when the live listing says the repo publishes none (#9286).
        """

        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            return None

        outcome: dict = {}
        resolved = self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = _pick_mmproj,
            label = "mmproj",
            cancel_event = cancel_event,
            near_path = near_path,
            outcome = outcome,
        )
        if resolved is not None:
            # An interrupted zero-byte copy must not shadow a working projector.
            try:
                if Path(resolved).stat().st_size > 0:
                    return resolved
            except OSError:
                return resolved
            logger.info("Ignoring an empty mmproj resolved for %s: %s", hf_repo, resolved)
        elif outcome.get("listed_live") is True:
            # The repo publishes one and the fetch dropped: retry next Apply rather
            # than launch a hand-added file in its place. Offline still falls back.
            return None
        if not near_path or cancel_event.is_set():
            return None
        from utils.models.model_config import _hf_cached_local_mmproj

        cached = _hf_cached_local_mmproj(near_path)
        if cached is not None:
            logger.info("Reusing hand-added mmproj from the HF cache: %s", cached)
        return cached

    def _cached_repo_mtp_drafter(
        self,
        hf_repo: str,
        *,
        cache_dir: Optional[str] = None,
        allow_nested: bool = True,
    ) -> Optional[str]:
        """A drafter already in this repo's local HF cache, reused offline when a
        fresh copy can't be fetched. Prefers a repo-root ``mtp-*.gguf`` across all
        cached snapshots; else an existing ``MTP/`` copy. None if none is cached.

        Ranked with ``_pick_mtp``'s keys, or offline and online disagree about one
        cache: lexical order put ``mtp-Qwen3.8-Flash-Next-BF16.gguf`` first, so a
        cached user got the 7.77 GB slowest head while a fresh install downloaded
        the 2.79 GB shared Q8_0 one.

        ``allow_nested=False`` drops the ``MTP/`` half, so a non-qwen4exp target
        resolves the same way offline as online (``_pick_mtp_root_only``)."""
        try:
            from utils.models.drafters.preference import mtp_preference_key
            from utils.models.model_config import (
                _drafter_split_is_complete,
                _iter_hf_cache_snapshots,
            )

            roots: list[Path] = []
            subdirs: list[Path] = []
            snapshots = (
                _iter_hf_cache_snapshots(hf_repo)
                if cache_dir is None
                else _iter_hf_cache_snapshots(hf_repo, cache_dir)
            )
            for snap in snapshots:  # newest first
                snap_roots: list[str] = []
                snap_subdirs: list[str] = []
                for f in _gguf_snapshot_files(snap):
                    # MTP only: a DSpark drafter needs --spec-type draft-dspark.
                    # The nested tier needs a published drafter NAME too, since
                    # everything under MTP/ classifies as one and an mmproj or
                    # imatrix there would otherwise go to --model-draft.
                    if "/" not in f:
                        if _is_mtp_only_drafter_path(f):
                            snap_roots.append(f)
                    elif _is_mtp_only_drafter_path(f) and _is_published_mtp_drafter_name(f):
                        snap_subdirs.append(f)
                # Root lexical, nested by preference: the tiers as _pick_mtp orders them.
                roots.extend(snap / f for f in sorted(snap_roots))
                subdirs.extend(snap / f for f in sorted(snap_subdirs, key = mtp_preference_key))
            # Keep snapshot order (newest first), root before any MTP/ copy, so a
            # newer main GGUF pairs with the newest cached drafter, not a stale one.
            for cand in roots + subdirs if allow_nested else roots:
                # Half a split set is not a drafter, and offline there is no fetch
                # to complete it: the rule _download_companion_gguf applies too.
                if cand.is_file() and _drafter_split_is_complete(cand):
                    return str(cand)
        except Exception as e:
            logger.debug("Cached MTP drafter lookup failed for %s: %s", hf_repo, e)
        return None

    def _download_mtp(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
        allow_nested: bool = True,
    ) -> Optional[str]:
        """Download the separate MTP drafter (speculative head) from a GGUF repo.

        Prefers the repo-root ``mtp-*.gguf`` companion -- the Q8_0 drafter
        unsloth mirrors there for llama.cpp ``-hf`` auto-discovery (smallest,
        recommended for speculation) -- and falls back to the ``MTP/`` folder for
        a repo that publishes no root mirror, as Qwen3.8-Flash-Next and
        Qwen3.8-27B do. Repos that bake the MTP head into the main GGUF ship
        neither and this returns None. Returns the local path, or None.

        ``allow_nested=False`` skips the ``MTP/`` fallback, which is what every
        architecture but qwen4exp gets: elsewhere the sidecar would only displace a
        head the file already carries.
        """

        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            return None

        if near_path:
            from utils.models.gguf_metadata import read_gguf_nextn_predict_layers
            if (read_gguf_nextn_predict_layers(near_path) or 0) > 0:
                logger.info("Main GGUF contains an embedded MTP head; skipping separate drafter.")
                return None

        pick = _pick_mtp if allow_nested else _pick_mtp_root_only

        borrowed_cached: Optional[str] = None
        if near_path:
            cached = _companion_snapshot_sibling(near_path, pick)
            if cached and _mtp_head_borrows(cached) and not _hf_env_offline():
                # --fit cannot measure a borrowing head, so the MTP context OOMs
                # (unsloth#10322). hf_hub_download reuses this file if nothing is better.
                logger.info(
                    "Cached MTP drafter borrows the target's embeddings; checking the "
                    "repo for a self-contained head: %s",
                    cached,
                )
                borrowed_cached = cached
            elif cached:
                logger.info("Reusing cached MTP drafter: %s", cached)
                return cached

        # Offline, reuse any drafter already on disk (a fresh copy can't be
        # fetched). Online, _download_companion_gguf/hf_hub_download reuse the
        # current cached file and refetch a changed one, so skip the probe here
        # rather than pair new weights with a stale draft.
        if _hf_env_offline():
            cached = self._cached_repo_mtp_drafter(
                hf_repo,
                cache_dir = _hub_cache_dir_for_snapshot_path(near_path),
                allow_nested = allow_nested,
            )
            if cached:
                logger.info(f"Reusing cached MTP drafter (offline): {cached}")
                return cached

        fetched = self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = pick,
            label = "MTP drafter",
            cancel_event = cancel_event,
            near_path = near_path,
            reuse_snapshot_sibling = borrowed_cached is None,
        )
        if fetched is None and borrowed_cached is not None:
            # An unmeasurable drafter still drafts; no drafter at all is the bigger loss.
            logger.info("Keeping the cached borrowing MTP drafter: %s", borrowed_cached)
            return borrowed_cached
        return fetched

    def _cached_repo_dspark_drafter(
        self,
        hf_repo: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Optional[str]:
        """The preferred already-cached DSpark sidecar for a repo, Q8_0 first
        (dspark_preference_key), so an offline reuse picks the same file the
        online download would have fetched."""
        try:
            from utils.models.model_config import (
                _iter_hf_cache_snapshots,
                dspark_preference_key,
            )

            snapshots = (
                _iter_hf_cache_snapshots(hf_repo)
                if cache_dir is None
                else _iter_hf_cache_snapshots(hf_repo, cache_dir)
            )
            candidates: list[Path] = []
            for snap in snapshots:
                candidates.extend(
                    snap / name
                    for name in _gguf_snapshot_files(snap)
                    if _is_dspark_drafter_path(name)
                )
            for candidate in sorted(candidates, key = lambda p: dspark_preference_key(p.name)):
                if candidate.is_file():
                    return str(candidate)
        except Exception as exc:
            logger.debug("Cached DSpark drafter lookup failed for %s: %s", hf_repo, exc)
        return None

    def _download_dspark(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
        binary: Optional[str] = None,
        caps_probe: Optional[Callable[[Optional[str]], dict]] = None,
    ) -> Optional[str]:
        """Download the published DSpark sidecar, preferring its Q8_0 copy.

        The ~11 GB fetch is gated on ``supports_dspark``: a binary that cannot run
        ``draft-dspark`` (every prebuilt in the known-broken window included) falls
        back, so the download would never be opened. A raised probe still fetches,
        since launch re-probes and may yet engage.

        ``caps_probe`` is the caller's accumulator: this gate shapes the launch, and a
        guess here has to latch like any other, or a load that skipped the sidecar on an
        inconclusive probe looks conclusive once the retry window lapses mid-download.
        """
        cancel_event = (
            cancel_event
            if cancel_event is not None
            else getattr(self, "_cancel_event", threading.Event())
        )
        if cancel_event.is_set():
            return None
        if caps_probe is None:
            caps_probe = self.probe_server_capabilities

        cached = _companion_snapshot_sibling(near_path, _pick_dspark) if near_path else None
        if not cached and _hf_env_offline():
            cached = self._cached_repo_dspark_drafter(
                hf_repo,
                cache_dir = _hub_cache_dir_for_snapshot_path(near_path),
            )
        try:
            if not caps_probe(binary).get("supports_dspark"):
                logger.warning(
                    "Skipping the DSpark sidecar download: llama-server has no usable "
                    "--spec-type draft-dspark, so this load falls back to no speculative "
                    "decoding. Run `unsloth studio update`, then reload."
                )
                # A sidecar already on disk is still reported: the route rediscovers it
                # on every Apply, so answering None would make the reuse check compare
                # it against a launched None and reload the same server each time.
                # _build_speculative_flags re-checks the capability and still falls back.
                return cached
        except Exception as exc:
            logger.debug("DSpark capability probe failed before the sidecar fetch: %s", exc)

        if cached:
            logger.info("Reusing cached DSpark drafter: %s", cached)
            return cached
        outcome: dict = {}
        found = self._download_companion_gguf(
            hf_repo = hf_repo,
            hf_token = hf_token,
            pick = _pick_dspark,
            label = "DSpark drafter",
            cancel_event = cancel_event,
            near_path = near_path,
            outcome = outcome,
        )
        # Distinguishes a repo that ships no sidecar (the permanent state for
        # every repo but one) from a fetch that failed and could yet succeed.
        self._dspark_sidecar_absent = outcome.get("listed") is False
        return found

    def _cached_repo_dflash_drafter(
        self,
        hf_repo: str,
        *,
        cache_dir: Optional[str] = None,
        near_path: Optional[str] = None,
    ) -> Optional[str]:
        """The preferred already-cached DFlash sidecar for a repo, Q8_0 first
        (dflash_repo_preference_key), so an offline reuse picks the same file
        the online download would have fetched.

        ``near_path`` is the weight this sidecar would draft for. A repo hosting
        more than one family ships more than one sidecar, and precision alone
        would hand model B the sidecar named after model A, so the candidates
        are ranked against the weight actually being loaded (same rule as
        detect_dflash_file). Without it the ordering is precision only, as
        before.
        """
        try:
            from utils.models.drafters import (
                _drafter_split_is_complete,
                dflash_repo_preference_key,
                is_dflash_architecture,
            )
            from utils.models.model_config import _iter_hf_cache_snapshots

            snapshots = (
                _iter_hf_cache_snapshots(hf_repo)
                if cache_dir is None
                else _iter_hf_cache_snapshots(hf_repo, cache_dir)
            )
            weight_name = Path(near_path).name if near_path else None
            ranked: list[tuple[tuple[int, int, int, str], Path]] = []
            for snap in snapshots:
                names = _gguf_snapshot_files(snap)
                # Every non-sidecar GGUF in the snapshot is a weight some sidecar
                # could be naming; that is what tells a neighbour's sidecar apart
                # from one naming no family at all. A nested dflash-*.gguf counts
                # as one of those weights, since only a root-level file is a
                # sidecar here.
                others = [
                    Path(name).name for name in names if not _is_root_dflash_drafter_path(name)
                ]
                ranked.extend(
                    (dflash_repo_preference_key(name, weight_name, others), snap / name)
                    for name in names
                    if _is_root_dflash_drafter_path(name)
                )
            for _, candidate in sorted(ranked, key = lambda entry: entry[0]):
                if not candidate.is_file():
                    continue
                # The ranking above works off names, and dflash- is only a naming
                # convention: a cached weight whose basename happens to start with
                # it would be launched as --model-draft and refused at startup.
                # Same header rule, same helper, as the local scan, so an offline
                # reuse and a local scan cannot disagree about what a sidecar is.
                # Skipped rather than fatal, since the snapshot may hold a real one.
                if not is_dflash_architecture(str(candidate)):
                    logger.info(
                        "Ignoring cached DFlash candidate %s: general.architecture is not dflash",
                        candidate,
                    )
                    continue
                # Same whole-set rule _companion_snapshot_sibling applies, for the
                # same reason: llama-server resolves the sibling shards from this
                # one's directory, and what this lookup returns is handed straight
                # back as the drafter with no fetch left to complete it. Half a
                # set reads as a valid header and then cannot be opened, so the
                # load quietly drops speculation. Skipped rather than fatal, since
                # another snapshot may hold the complete copy.
                if not _drafter_split_is_complete(candidate):
                    logger.info(
                        "Ignoring cached DFlash candidate %s: split shard set is incomplete",
                        candidate,
                    )
                    continue
                return str(candidate)
        except Exception as exc:
            logger.debug("Cached DFlash drafter lookup failed for %s: %s", hf_repo, exc)
        return None

    def _download_dflash(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        near_path: Optional[str] = None,
        binary: Optional[str] = None,
    ) -> Optional[str]:
        """Download the published DFlash sidecar, preferring its Q8_0 copy.

        Unlike the ~11 GB DSpark sidecar, which is why that one is opt-in, the
        published DFlash sidecar is ~1.5 GiB and already ships in the model's own
        GGUF repo, so Auto fetches it the same way it fetches the MTP drafter.
        Still gated on ``supports_dflash``: a binary with no usable
        ``--spec-type draft-dflash`` falls back, so the download would never be
        opened. A raised probe still fetches, since launch re-probes and may yet
        engage.
        """

        cancel_event = cancel_event if cancel_event is not None else self._cancel_event
        if cancel_event.is_set():
            return None
        weight_name = Path(near_path).name if near_path else None
        # Whatever a previous load concluded, this one is asking again.
        self._dflash_retry_needed = False

        def _mark_retry_needed() -> None:
            self._dflash_retry_needed = True

        # Basenames whose header turned out not to say dflash. A name lands here
        # only once the file is readable on disk, and _pick_dflash then skips it,
        # so a repo whose best-ranked candidate is an impostor still reaches the
        # real sidecar behind it instead of the search giving up on the repo.
        rejected: set[str] = set()

        def _pick_dflash(candidates: list[str]) -> Optional[str]:
            # Ranked against the weight being loaded, not by precision alone: a
            # repo hosting two families ships a sidecar per family, and
            # dflash-model-A-Q8_0.gguf beats the generic dflash-kquant.gguf on
            # precision, so model B would download and launch model A's drafter.
            # The rest of the listing supplies the neighbouring weights that make
            # a foreign sidecar recognisable (a sidecar naming no family at all
            # stays eligible, which is what the published one does).
            from utils.models.drafters import (
                dflash_repo_preference_key,
                split_listing_is_complete,
            )

            # Root level only, as the local scan is: a nested dflash-*.gguf is an
            # ordinary weight, and offering it here spends its entire download
            # before the header check can turn it away.
            others = [
                Path(name).name
                for name in candidates
                if name.lower().endswith(".gguf") and not _is_root_dflash_drafter_path(name)
            ]
            # Same bound as dflash_plan_files: dflash- is a prefix real weights carry,
            # the header only reads once the bytes are here, and a drafter is a few
            # layers of its target. An unlisted size leaves the candidate in, as before.
            sizes = self._remote_root_gguf_sizes(hf_repo, hf_token)
            try:
                target_bytes = (self._get_gguf_size_bytes(near_path) or 0) if near_path else 0
            except OSError:
                target_bytes = 0
            files = sorted(
                (
                    name
                    for name in candidates
                    if _is_root_dflash_drafter_path(name)
                    and Path(name).name not in rejected
                    # Before the ranking, as dflash_plan_files is: the fetch refuses a
                    # half-listed set, and returning it ends the loop.
                    and split_listing_is_complete(candidates, name)
                    # Whole set, not the picked shard: each half of a split weight can
                    # sit under the target while the set does not.
                    and not (
                        target_bytes and _drafter_set_bytes(sizes, Path(name).name) >= target_bytes
                    )
                ),
                key = lambda name: dflash_repo_preference_key(name, weight_name, others),
            )
            return files[0] if files else None

        def _validated(path: Optional[str]) -> Optional[str]:
            """``path`` back iff its header really says ``dflash``.

            _is_dflash_drafter_path is a FILENAME test, deliberately (the prefix
            form is the only one accepted, so a sidecar cannot double as a
            selectable main model in the quant picker), which means a remote repo
            holding an ordinary weight called dflash-*.gguf passes it. The local
            scan settles that with the architecture in the header; the remote
            paths have to as well, or the load spends gigabytes of download on a
            file llama-server then refuses as --model-draft. Same helper as
            detect_dflash_file so the two rules cannot drift.
            """
            from utils.models.drafters import is_dflash_architecture

            if not path:
                return None
            if is_dflash_architecture(path):
                return path
            rejected.add(Path(path).name)
            logger.warning("Ignoring DFlash candidate %s: general.architecture is not dflash", path)
            return None

        # Retried rather than abandoned: every rejection removes one name from the
        # pool, so the next pick returns a different file and the scan ends at the
        # last candidate (or at the first real sidecar). Answers already seen end
        # it too, so a scan that ignores the pool cannot spin here.
        cached: Optional[str] = None
        seen_siblings: set[str] = set()
        while near_path and not cancel_event.is_set():
            sibling = _companion_snapshot_sibling(near_path, _pick_dflash)
            if sibling is None or sibling in seen_siblings:
                break
            seen_siblings.add(sibling)
            cached = _validated(sibling)
            if cached:
                break
        # _cached_repo_dflash_drafter applies the same header check to its own
        # candidates, so anything it hands back is already validated.
        if not cached and _hf_env_offline():
            cached = self._cached_repo_dflash_drafter(
                hf_repo,
                cache_dir = _hub_cache_dir_for_snapshot_path(near_path),
                near_path = near_path,
            )
        try:
            if not self.probe_server_capabilities(binary).get("supports_dflash"):
                logger.warning(
                    "Skipping the DFlash sidecar download: llama-server has no usable "
                    "--spec-type draft-dflash, so this load falls back to no speculative "
                    "decoding. Run `unsloth studio update`, then reload."
                )
                # A sidecar already on disk is still reported, for the same reason as
                # the DSpark path: the route rediscovers it on every Apply, so
                # answering None would make the reuse check compare it against a
                # launched None and reload the same server each time.
                # _build_speculative_flags re-checks the capability and still falls back.
                return cached
        except Exception as exc:
            logger.debug("DFlash capability probe failed before the sidecar fetch: %s", exc)

        if cached:
            logger.info("Reusing cached DFlash drafter: %s", cached)
            return cached
        outcome: dict = {}
        found: Optional[str] = None
        # The header can only be read once the bytes are here, so validation is a
        # post-fetch step and a rejection has to be able to fall through to the
        # next candidate rather than end the search. Bounded the same way as the
        # reuse scan above: a fetch that answers with a file already rejected has
        # nothing further to offer.
        fetched: set[str] = set()
        while True:
            candidate = self._download_companion_gguf(
                hf_repo = hf_repo,
                hf_token = hf_token,
                pick = _pick_dflash,
                label = "DFlash drafter",
                cancel_event = cancel_event,
                near_path = near_path,
                outcome = outcome,
                # Not a header rejection: that candidate is settled and the loop moves
                # on. Only the Hub dropping out leaves a real sidecar unfetched.
                on_transient_failure = _mark_retry_needed,
            )
            if candidate is None or candidate in fetched:
                break
            fetched.add(candidate)
            found = _validated(candidate)
            if found:
                break
        # Distinguishes a repo that ships no sidecar from a fetch that failed and
        # could yet succeed. A repo whose only dflash-*.gguf is an ordinary weight
        # exits the loop through a listing that picked nothing, which records the
        # same permanent absence: there is nothing usable here to fetch next time.
        self._dflash_sidecar_absent = outcome.get("listed") is False
        return found

    @staticmethod
    def _remote_root_gguf_sizes(hf_repo: str, hf_token: Optional[str] = None) -> dict[str, int]:
        """Root-level GGUF basenames to their listed byte size, {} when unavailable.

        list_repo_files answers names only, so the size bound needs the metadata
        listing. Best effort: an empty map leaves every candidate eligible, which is
        the behaviour before the bound existed.
        """
        try:
            from huggingface_hub import model_info
            info = call_hub_with_anonymous_retry(model_info, hf_token, hf_repo, files_metadata = True)
            return {
                name: int(getattr(sibling, "size", 0) or 0)
                for sibling in (getattr(info, "siblings", None) or [])
                if isinstance(name := getattr(sibling, "rfilename", None), str) and "/" not in name
            }
        except Exception as exc:
            logger.debug("Could not size the DFlash candidates for %s: %s", hf_repo, exc)
            return {}

    def _dspark_wins_auto(
        self,
        *,
        binary: Optional[str],
        dspark_draft_path: Optional[str],
        spec_canon: str,
        extra_args: Optional[list[str]],
        caps: Optional[Callable[[Optional[str]], dict]] = None,
    ) -> bool:
        """Whether Auto will actually resolve this load to DSpark.

        A DSpark path on its own does not answer that. _download_dspark reports
        an already-cached sidecar even when the binary has no usable
        ``--spec-type draft-dspark`` (deliberately: the route rediscovers it on
        every Apply, and answering None there would reload the same server each
        time), so the capability has to be applied by everything that reads the
        path as "DSpark is what Auto picks". Without it a binary that supports
        DFlash but not DSpark stands down on the DFlash fetch for a sidecar it
        can never launch and the load ends up with no drafter at all.

        A raised probe answers False, which keeps Auto on its other options
        exactly as the promotion does. probe_server_capabilities caches on the
        binary's revision, so asking here and at the promotion is one probe.
        """
        if spec_canon != "auto" or not dspark_draft_path:
            return False
        if _extra_args_set_spec_type(extra_args):
            return False
        try:
            # Through the caller's getter when it has one, so an inconclusive probe
            # still latches for this launch (#8317) even though the promotion moved
            # in here.
            probe = caps or self.probe_server_capabilities
            return bool(probe(binary).get("supports_dspark"))
        except Exception as exc:
            logger.debug("DSpark capability probe failed during Auto: %s", exc)
            return False

    def _resolve_launch_mmproj_path(
        self,
        *,
        model_path: str,
        mmproj_path: Optional[str],
        extra_args: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Resolve a projector, trusting an explicit path over filename heuristics."""
        override = _extra_args_device(extra_args, {"--mmproj", "-mm"})
        if override is not None:
            if not override or not Path(override).is_file():
                raise ValueError("The custom mmproj path must name an existing file.")
            return override
        if not mmproj_path:
            return None

        mmproj = Path(mmproj_path)
        if not mmproj.is_file():
            logger.warning(f"mmproj file not found: {mmproj_path}")
            return None

        from utils.models.model_config import mmproj_matches_model_family

        if not mmproj_matches_model_family(model_path, str(mmproj)):
            logger.warning(
                f"mmproj does not match model family: model={Path(model_path).name} "
                f"mmproj={mmproj.name}"
            )
            return None

        return str(mmproj)

    def _mmproj_vram_bytes(self, launch_mmproj_path: Optional[str]) -> int:
        """Return resolved mmproj VRAM bytes, or 0 when absent/unreadable."""
        if not launch_mmproj_path:
            return 0
        try:
            return self._get_gguf_size_bytes(launch_mmproj_path)
        except OSError as e:
            logger.debug(f"Could not size mmproj {launch_mmproj_path}: {e}")
            return 0

    def _inherited_mmproj_soft_overhead(self, mmproj_bytes: int, *, on_host: bool) -> int:
        """Runtime buffer allowance for a projector that arrived only through the env.

        The resolved projector gets this term in the fit's ``soft_overhead``, but that
        one is gated on ``effective_is_vision``, itself gated on ``launch_mmproj_path``,
        so a ``LLAMA_ARG_MMPROJ`` projector is charged its file size and nothing for the
        ~1.3x the encoder really allocates. Understating the footprint is the direction
        that claims a fit which is not there.

        VRAM arm only: on the host arm the resolved projector is priced raw too.
        """
        if mmproj_bytes <= 0 or on_host:
            return 0
        return int(mmproj_bytes * (self._MMPROJ_VRAM_SAFETY - 1.0))

    def _resolve_launch_mtp_path(
        self,
        *,
        mtp_draft_path: Optional[str],
        drafter_label: str = "MTP",
    ) -> Optional[str]:
        """Return mtp_draft_path iff it exists on disk, else None.

        No family check needed: the drafter is only ever auto-resolved from
        the same repo as the main GGUF (see _download_mtp).
        """
        if not mtp_draft_path:
            return None
        if not Path(mtp_draft_path).is_file():
            logger.warning(f"{drafter_label} drafter file not found: {mtp_draft_path}")
            return None
        return str(mtp_draft_path)
