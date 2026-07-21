# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Latest-transformers support check for brand-new model architectures.

When a model's ``model_type`` is absent from every installed transformers overlay
(base 4.57.x plus the .venv_t5_530/550/510 sidecars and, if provisioned, .venv_t5_latest),
Unsloth cannot load it today. This module answers, without authentication, code execution,
or trust_remote_code:

  1. Does the LATEST transformers release on PyPI ship this ``model_type``?
  2. Does transformers ``main`` on GitHub ship it (dev-only, not yet installable)?

Sources (all unauthenticated; raw.githubusercontent.com is not API rate-limited and
api.github.com is deliberately never used):
  - https://pypi.org/pypi/transformers/json                       -> latest release version
  - https://raw.githubusercontent.com/huggingface/transformers/{ref}/src/transformers/
        models/auto/configuration_auto.py + auto_mappings.py      -> CONFIG_MAPPING_NAMES

The fetched sources are parsed with the same AST extractor the static router uses
(:func:`utils.transformers_version._model_types_from_source`), so the remote answer is
computed exactly like the local overlay answer.

Results are cached in memory and in a small JSON snapshot under ``studio_root()/cache``
(ttl ~1 day) so repeated tier resolutions never re-fetch; failures are backed off in
memory. Every fetch is bounded (<=5s, one retry), so a hung network cannot block model
loading. Fully offline-safe: offline env vars or the kill switch
``UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS=1`` make every check return None (current
behavior preserved).

The consented install path (:func:`install_latest_transformers`) provisions the
persistent ``.venv_t5_latest`` sidecar via
:func:`utils.transformers_version.ensure_latest_transformers_venv`.
"""

import json
import os
import threading
import time
from pathlib import Path

from loggers import get_logger
from utils.paths.storage_roots import studio_root as _studio_root
from utils.transformers_version import (
    _env_offline,
    _load_config_json,
    _model_types_from_source,
    _tier_from_config_mapping,
    _config_model_types,
    _NESTED_CONFIG_KEYS,
    _TIER_RANK,
    _model_types_from_config,
    _TRANSFORMERS_510_MODEL_TYPES,
    _TRANSFORMERS_530_MODEL_TYPES,
    _TRANSFORMERS_550_MODEL_TYPES,
    ensure_latest_transformers_venv,
    latest_venv_pinned_version,
)

logger = get_logger(__name__)

_PYPI_JSON_URL = "https://pypi.org/pypi/transformers/json"
_RAW_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/{ref}"
    "/src/transformers/models/auto/{name}"
)
_AUTO_FILES = ("configuration_auto.py", "auto_mappings.py")

_FETCH_TIMEOUT_SECONDS = 5.0
_FETCH_RETRIES = 1
_CACHE_TTL_SECONDS = 24 * 60 * 60
_FAILURE_BACKOFF_SECONDS = 300

_CACHE_FILE_NAME = "transformers_latest_check.json"
_SNAPSHOT_SCHEMA = 1

# Snapshot: {"schema", "fetched_at", "pypi_version", "pypi_model_types", "main_model_types"}.
# Install-in-progress state lives in utils.transformers_version (the sidecar swap reservation).
_lock = threading.Lock()
_memory_snapshot: dict | None = None
_last_failure_at: float = 0.0
_is_fetching: bool = False

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _disabled() -> bool:
    """True if the operator disabled the latest-transformers check entirely."""
    return (
        os.environ.get("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "").strip().lower()
        in _TRUE_VALUES
    )


def _cache_file() -> Path:
    return _studio_root() / "cache" / _CACHE_FILE_NAME


# Sentinel for HTTP 404 (absent at ref), distinct from transient failures.
_FETCH_MISSING = "__unsloth_fetch_missing__"


def _fetch_text(url: str) -> str | None:
    """GET *url* with a bounded timeout and one retry; None on any failure.

    Returns ``_FETCH_MISSING`` (without retrying) on HTTP 404 so callers can tell
    "absent at this ref" apart from "network flaked".
    """
    import urllib.error
    import urllib.request

    for attempt in range(1 + _FETCH_RETRIES):
        try:
            req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
            with urllib.request.urlopen(req, timeout = _FETCH_TIMEOUT_SECONDS) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return _FETCH_MISSING
            logger.debug("Fetch failed (attempt %d) for %s: %s", attempt + 1, url, exc)
        except Exception as exc:
            logger.debug("Fetch failed (attempt %d) for %s: %s", attempt + 1, url, exc)
    return None


def _fetch_latest_pypi_version() -> str | None:
    """Latest transformers release version from PyPI's unauthenticated JSON API."""
    body = _fetch_text(_PYPI_JSON_URL)
    if body is None or body == _FETCH_MISSING:
        return None
    try:
        version = json.loads(body).get("info", {}).get("version")
    except Exception as exc:
        logger.debug("Could not parse PyPI JSON: %s", exc)
        return None
    return version if isinstance(version, str) and version else None


def _fetch_remote_model_types(ref: str) -> frozenset[str] | None:
    """CONFIG_MAPPING_NAMES keys at *ref* (a release tag like ``v5.12.0`` or ``main``).

    Fetches configuration_auto.py plus auto_mappings.py (the 5.10+ split) from
    raw.githubusercontent.com and parses them with the shared AST extractor. A file
    that 404s (auto_mappings.py on pre-5.10 tags) is skipped, but a transient fetch
    or parse failure of EITHER file fails the whole lookup: most model types live in
    auto_mappings.py on current releases, so a partial map cached for the TTL would
    make /validate skip the upgrade prompt for architectures the release does ship.
    An empty result is likewise a failure so it is never cached as "supports nothing".
    """
    keys: set[str] = set()
    fetched_any = False
    for name in _AUTO_FILES:
        source = _fetch_text(_RAW_URL.format(ref = ref, name = name))
        if source is None:
            return None
        if source == _FETCH_MISSING:
            continue
        fetched_any = True
        try:
            keys |= _model_types_from_source(source)
        except Exception as exc:
            logger.debug("Could not parse %s at %s: %s", name, ref, exc)
            return None
    if not fetched_any or not keys:
        return None
    return frozenset(keys)


def _load_snapshot_file() -> dict | None:
    """Persisted snapshot from disk, or None (missing/corrupt/old schema)."""
    try:
        with open(_cache_file(), encoding = "utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("schema") != _SNAPSHOT_SCHEMA:
        return None
    if not isinstance(data.get("fetched_at"), (int, float)):
        return None
    if not isinstance(data.get("pypi_version"), str):
        return None
    for key in ("pypi_model_types", "main_model_types"):
        value = data.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return None
    return data


def _save_snapshot_file(snapshot: dict) -> None:
    """Atomic best-effort write (tmp + os.replace, Windows-safe); failures only log."""
    path = _cache_file()
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp.write_text(json.dumps(snapshot), encoding = "utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("Could not persist %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok = True)
        except Exception:
            pass


def _snapshot_is_fresh(snapshot: dict | None) -> bool:
    return (
        snapshot is not None
        and (time.time() - float(snapshot.get("fetched_at", 0))) < _CACHE_TTL_SECONDS
    )


def _refresh_snapshot() -> dict | None:
    """Fetch a fresh snapshot from PyPI + raw.githubusercontent.com; None on failure.

    The PyPI version and its tagged mapping are required; the ``main`` mapping is
    best-effort (recorded as an empty list plus ``main_checked=False`` when unavailable,
    so a dev-only architecture is reported as "unknown" rather than "unsupported").
    """
    version = _fetch_latest_pypi_version()
    if version is None:
        return None
    pypi_types = _fetch_remote_model_types(f"v{version}")
    if pypi_types is None:
        return None
    main_types = _fetch_remote_model_types("main")
    return {
        "schema": _SNAPSHOT_SCHEMA,
        "fetched_at": time.time(),
        "pypi_version": version,
        "pypi_model_types": sorted(pypi_types),
        "main_model_types": sorted(main_types) if main_types is not None else [],
        "main_checked": main_types is not None,
    }


def _get_snapshot() -> dict | None:
    """Current support snapshot: memory -> disk -> network, with TTL and failure backoff.

    The network refresh runs outside the lock so a slow fetch cannot stall other
    threads in the ASGI pool; _is_fetching deduplicates concurrent refreshes
    (losers return None, the graceful fallthrough, rather than waiting).
    """
    global _memory_snapshot, _last_failure_at, _is_fetching
    with _lock:
        if _snapshot_is_fresh(_memory_snapshot):
            return _memory_snapshot
        disk = _load_snapshot_file()
        if _snapshot_is_fresh(disk):
            _memory_snapshot = disk
            return disk
        if _disabled() or _env_offline():
            return None
        if time.time() - _last_failure_at < _FAILURE_BACKOFF_SECONDS:
            return None
        if _is_fetching:
            return None
        _is_fetching = True
    fresh = None
    try:
        fresh = _refresh_snapshot()
    finally:
        with _lock:
            _is_fetching = False
            if fresh is None:
                _last_failure_at = time.time()
            else:
                _memory_snapshot = fresh
    if fresh is None:
        # A stale positive could offer a version PyPI no longer serves; be strict.
        return None
    _save_snapshot_file(fresh)
    return fresh


def clear_caches() -> None:
    """Test helper: drop the in-memory snapshot, failure backoff, and busy flags."""
    global _memory_snapshot, _last_failure_at, _is_fetching
    with _lock:
        _memory_snapshot = None
        _last_failure_at = 0.0
        _is_fetching = False
    from utils.transformers_version import end_sidecar_swap

    end_sidecar_swap()


def latest_transformers_supports(model_type: str) -> dict | None:
    """Whether the newest transformers (PyPI release and/or GitHub main) ships *model_type*.

    Returns ``{"pypi_version": str, "supported_in_pypi": bool, "supported_in_main": bool}``
    or None when the answer is unavailable (offline, kill switch, network failure) — the
    caller must then fall through to current behavior. Cached (memory + JSON snapshot on
    disk, ttl ~1 day) so repeated tier resolutions never re-fetch.
    """
    if not isinstance(model_type, str) or not model_type:
        return None
    if _disabled() or _env_offline():
        return None
    snapshot = _get_snapshot()
    if snapshot is None:
        return None
    return {
        "pypi_version": snapshot["pypi_version"],
        "supported_in_pypi": model_type in set(snapshot["pypi_model_types"]),
        "supported_in_main": model_type in set(snapshot["main_model_types"]),
    }


# model_types the hardcoded tier tables already route; never remote-check these.
def _hardcoded_model_types() -> frozenset[str]:
    return frozenset(
        _TRANSFORMERS_530_MODEL_TYPES
        | _TRANSFORMERS_550_MODEL_TYPES
        | _TRANSFORMERS_510_MODEL_TYPES
    )


def check_upgrade_for_model(
    model_name: str, hf_token: str | None = None
) -> dict | None:
    """Upgrade signal for *model_name*, or None when current routing already handles it.

    The tier hook for the pre-load ``/validate`` path: fires ONLY when the model's
    ``model_type`` is absent from every installed overlay (and from the hardcoded tier
    tables), i.e. exactly when today's load would fail with an unrecognized-architecture
    error. Returns ``{"model_type", "pypi_version", "supported_in_pypi",
    "supported_in_main"}`` when the newest transformers knows the type, else None.

    Never raises; every network touch is bounded and cached. Offline or with the
    ``UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS`` kill switch it returns None immediately.
    """
    try:
        if _disabled() or _env_offline():
            return None
        cfg = _load_config_json(model_name, hf_token)
        if not isinstance(cfg, dict):
            return None
        candidates = _model_types_from_config(cfg)
        if not candidates:
            return None
        # Without a readable base mapping every type looks brand new; bail out.
        if not _config_model_types("default"):
            return None
        hardcoded = _hardcoded_model_types()
        missing = [
            candidate
            for candidate in candidates
            if candidate not in hardcoded
            and not any(candidate in _config_model_types(tier) for tier in _TIER_RANK)
        ]
        if not missing:
            return None
        # Latest must load EVERY missing type (wrappers build nested sub-configs
        # through CONFIG_MAPPING) or the load still fails.
        supports = [latest_transformers_supports(candidate) for candidate in missing]
        if any(
            s is None or not (s["supported_in_pypi"] or s["supported_in_main"])
            for s in supports
        ):
            return None
        # Offer the PyPI install only if the release ships every missing type; a
        # main-only type in the mix surfaces as dev-only.
        model_type = missing[0]
        supported_in_pypi = all(s["supported_in_pypi"] for s in supports)
        supported_in_main = all(
            s["supported_in_pypi"] or s["supported_in_main"] for s in supports
        )
        logger.info(
            "Model %s has model_type=%s unknown to every installed transformers "
            "(latest PyPI %s: %s, main: %s)",
            model_name,
            model_type,
            supports[0]["pypi_version"],
            "supported" if supported_in_pypi else "unsupported",
            "supported" if supported_in_main else "unsupported",
        )
        return {
            "model_type": model_type,
            "pypi_version": supports[0]["pypi_version"],
            "supported_in_pypi": supported_in_pypi,
            "supported_in_main": supported_in_main,
        }
    except Exception as exc:
        logger.debug("Latest-transformers check failed for '%s': %s", model_name, exc)
        return None


# --- Dependency compatibility preflight ------------------------------------------------------
# Sidecars install transformers --no-deps atop the base env. Before installing, compare
# requires_dist: unsatisfied shadowable deps become exact --target pins, anything else blocks.

# Safe to shadow inside the sidecar dir (pure wheels, no torch coupling).
_SHADOWABLE_DEPS = frozenset({"tokenizers", "safetensors"})
# Provided by the sidecar recipe; checked against its pin, not the base env.
_SIDECAR_PROVIDED = {"huggingface-hub": "1.8.0", "hf-xet": "1.4.2"}
# CLI-only; never imported at runtime in Unsloth's workers.
_IGNORED_DEPS = frozenset({"typer"})


def _canonical_dep_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _fetch_requires_dist(version: str) -> list[str] | None:
    """Core (marker-free, non-extra) requires_dist of transformers *version* from PyPI."""
    body = _fetch_text(f"https://pypi.org/pypi/transformers/{version}/json")
    if body is None or body == _FETCH_MISSING:
        return None
    try:
        reqs = json.loads(body).get("info", {}).get("requires_dist")
    except Exception:
        return None
    if not isinstance(reqs, list):
        return None
    return [r for r in reqs if isinstance(r, str)]


def _resolve_exact_version(name: str, specifier) -> str | None:
    """Newest PyPI release of *name* satisfying *specifier* (exact pin for the shadow)."""
    body = _fetch_text(f"https://pypi.org/pypi/{name}/json")
    if body is None or body == _FETCH_MISSING:
        return None
    try:
        from packaging.version import InvalidVersion, Version

        releases = json.loads(body).get("releases", {})
        best = None
        for candidate in releases:
            try:
                parsed = Version(candidate)
            except InvalidVersion:
                continue
            if parsed.is_prerelease or not specifier.contains(candidate):
                continue
            if best is None or parsed > Version(best):
                best = candidate
        return best
    except Exception as exc:
        logger.debug("Could not resolve an exact %s version: %s", name, exc)
        return None


def compat_plan(version: str) -> tuple[tuple[str, ...], list[str]]:
    """(extra exact pins to shadow-install, blocking requirement strings) for *version*.

    Compares the release's core requires_dist against the running base env (the env the
    workers overlay the sidecar onto). A requirement the base env satisfies needs nothing;
    an unsatisfied shadowable dep becomes an exact pin inside the sidecar; any other
    unsatisfied requirement is a blocker. An unavailable requires_dist BLOCKS the
    install: proceeding unverified could pin a sidecar whose imports then crash the
    workers, and the caller just reached PyPI for the version check so a retry is cheap.
    """
    reqs = _fetch_requires_dist(version)
    if reqs is None:
        return (), [
            "dependency metadata for this release (could not be fetched from PyPI; retry)"
        ]
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _installed_version
        from packaging.requirements import InvalidRequirement, Requirement
    except Exception:
        return (), []
    extras: list[str] = []
    blockers: list[str] = []
    for raw in reqs:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.extras or (req.marker is not None and not req.marker.evaluate()):
            continue
        name = _canonical_dep_name(req.name)
        if name in _IGNORED_DEPS:
            continue
        if name in _SIDECAR_PROVIDED:
            if not req.specifier.contains(_SIDECAR_PROVIDED[name], prereleases = True):
                blockers.append(raw)
            continue
        try:
            installed = _installed_version(req.name)
        except PackageNotFoundError:
            installed = None
        if installed is not None and req.specifier.contains(
            installed, prereleases = True
        ):
            continue
        if name in _SHADOWABLE_DEPS:
            exact = _resolve_exact_version(name, req.specifier)
            if exact is None:
                blockers.append(raw)
            else:
                extras.append(f"{name}=={exact}")
        else:
            blockers.append(raw)
    return tuple(extras), blockers


def is_install_in_progress() -> bool:
    """True while a latest-transformers install or lazy repair holds the sidecar swap
    reservation. Training and export starts check this so a fresh worker never
    activates the sidecar mid-swap."""
    from utils.transformers_version import sidecar_swap_in_progress
    return sidecar_swap_in_progress()


def install_latest_transformers(
    version: str,
    before_swap = None,
    reserved: bool = False,
) -> dict:
    """Consented install of the latest transformers sidecar; returns a structured result.

    Guards: the requested *version* must match the current PyPI latest from the (cached)
    snapshot, so a client cannot pin an arbitrary package version through this endpoint.
    On success ``.venv_t5_latest`` is provisioned and pinned; routing then resolves the
    new tier automatically on this and every future start. *before_swap* is forwarded
    to the stage-and-swap: it runs only after the staged install succeeded, right
    before the live sidecar is replaced. *reserved* means the caller already holds the
    sidecar swap reservation (the install route takes it before waiting on the
    inference lifecycle gate, so worker starts see it for the whole window).
    """
    from utils.transformers_version import end_sidecar_swap, try_begin_sidecar_swap

    if not reserved and not try_begin_sidecar_swap():
        return {
            "success": False,
            "version": version,
            "message": "A transformers installation is already in progress.",
        }
    try:
        return _install_latest_transformers_locked(version, before_swap = before_swap)
    finally:
        if not reserved:
            end_sidecar_swap()


def _install_latest_transformers_locked(version: str, before_swap = None) -> dict:
    """Body of install_latest_transformers; runs with the in-progress flag held."""
    if _disabled():
        return {
            "success": False,
            "version": version,
            "message": "Latest-transformers installs are disabled "
            "(UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS).",
        }
    if _env_offline():
        return {
            "success": False,
            "version": version,
            "message": "Cannot install: Unsloth is in offline mode.",
        }
    # Re-verify against a LIVE snapshot (a release may land inside the cache TTL);
    # fall back to the cached one on fetch failure.
    global _memory_snapshot
    snapshot = _refresh_snapshot()
    if snapshot is not None:
        with _lock:
            _memory_snapshot = snapshot
        _save_snapshot_file(snapshot)
    else:
        snapshot = _get_snapshot()
    if snapshot is None:
        return {
            "success": False,
            "version": version,
            "message": "Could not verify the latest transformers release on PyPI.",
        }
    if version != snapshot["pypi_version"]:
        return {
            "success": False,
            "version": version,
            "message": f"Requested version {version!r} is not the latest transformers "
            f"release ({snapshot['pypi_version']}).",
            # Lets the consent dialog retry with the release that superseded the
            # one /validate saw, instead of re-sending the stale version forever.
            "latest_version": snapshot["pypi_version"],
        }
    extra_packages, blockers = compat_plan(version)
    if blockers:
        return {
            "success": False,
            "version": version,
            "message": "Cannot install transformers "
            f"{version}: this environment does not satisfy {', '.join(blockers)}. "
            "An Unsloth update is required first.",
        }
    if not ensure_latest_transformers_venv(
        version, extra_packages, before_swap = before_swap
    ):
        return {
            "success": False,
            "version": version,
            "message": f"Installing transformers {version} failed; see the Unsloth logs.",
        }
    _invalidate_capability_caches()
    return {
        "success": True,
        "version": version,
        "message": f"Installed transformers {version} into the latest sidecar "
        f"(pinned: {latest_venv_pinned_version()}).",
    }


def _invalidate_capability_caches():
    """Drop caches computed before the new sidecar existed: tier probes and the
    latest tier's model_type mapping (stale on upgrade) plus vision detection
    (a raw-heuristic False may now defer to the sidecar AutoConfig probe)."""
    try:
        from utils import transformers_version as tv
        tv._probe_tier_cache.clear()
        tv._config_mapping_cache.pop("latest", None)
    except Exception:
        pass
    try:
        from utils.models import model_config as mc
        mc._vision_detection_cache.clear()
    except Exception:
        pass
