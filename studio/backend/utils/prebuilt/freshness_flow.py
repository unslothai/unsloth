# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared mechanics of the llama.cpp / whisper.cpp prebuilt freshness checks. The component modules (utils.llama_cpp_freshness / utils.whisper_cpp_freshness) keep their public names, per-module caches and version-comparison policy; everything mechanical (marker walk-up, GitHub release fetch, memo + disk cache, the freshness report skeleton) lives here, parameterized by call-time callables so the modules' monkeypatch seams keep working."""

from __future__ import annotations

import http.client
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from utils.update_status import update_checks_disabled

logger = structlog.get_logger(__name__)

# 24h TTL keeps the GitHub call off the hot path and within rate limits.
RELEASE_CACHE_TTL_SECONDS = 24 * 60 * 60
# Briefly memoize failed lookups so recurring status reads do not retry an unreachable GitHub endpoint on every request.
RELEASE_FAILURE_CACHE_TTL_SECONDS = 60
GITHUB_RATE_LIMITED_DEFAULT_SECONDS = 15 * 60
# The primary window is an hour; a skewed reset header is held to that ceiling.
GITHUB_RATE_LIMIT_MAX_SECONDS = 60 * 60
GITHUB_RATE_LIMIT_STATUS = (403, 429)

# One lockout for the whole process: the quota is per token or per IP, not per repo.
_api_rate_limited_lock = threading.Lock()
_api_rate_limited_until: float = 0.0


def header_value(headers: Any, name: str) -> str:
    if headers is None:
        return ""
    try:
        return str(headers.get(name) or "").strip()
    except (AttributeError, TypeError):
        return ""


def rate_limit_wait_seconds(headers: Any, *, now: Optional[float] = None) -> Optional[float]:
    now = time.time() if now is None else now

    def _number(value: str) -> Optional[float]:
        try:
            return float(value)
        except ValueError:
            return None

    retry_after = header_value(headers, "Retry-After")
    after = _number(retry_after)
    if after is None and retry_after:
        try:
            import email.utils
            after = email.utils.parsedate_to_datetime(retry_after).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            after = None
    if after is not None:
        return max(after, 0.0)
    if header_value(headers, "X-RateLimit-Remaining") == "0":
        reset = _number(header_value(headers, "X-RateLimit-Reset"))
        if reset is not None:
            return max(reset - now, 0.0)
    return None


# A secondary limit can answer 403 with the quota untouched and no Retry-After, and only the body names it. Same markers as gh_client.
_RATE_LIMIT_BODY_MARKERS = (
    "api rate limit exceeded",
    "rate limit exceeded",
    "secondary rate limit",
    "secondary limit",
    "abuse detection mechanism",
    "abuse detection",
)


def names_a_rate_limit(body: object) -> bool:
    if not body:
        return False
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", errors = "replace")
    text = str(body).lower()
    return any(marker in text for marker in _RATE_LIMIT_BODY_MARKERS)


def error_body(exc: BaseException, *, limit: int = 2048) -> str:
    """The refusal's body, cached on the exception since reading an HTTPError consumes it."""
    cached = getattr(exc, "_unsloth_body", None)
    if cached is not None:
        return cached
    try:
        raw = exc.read(limit)  # type: ignore[attr-defined]
        text = raw.decode("utf-8", errors = "replace") if isinstance(raw, bytes) else str(raw)
    except Exception:  # noqa: BLE001 - a body we cannot read simply names nothing
        text = ""
    try:
        exc._unsloth_body = text  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - exotic exception types
        pass
    return text


def is_rate_limited(
    headers: Any = None,
    *,
    status: Optional[int] = None,
    body: object = None,
) -> bool:
    """Whether throttling explains this refusal, by gh_client._is_rate_limit_response's rule. A secondary limit leaves X-RateLimit-* untouched, so a 429 or a naming body counts; a 403 without those headers is a permission or policy refusal."""
    if status == 429:
        return True
    if header_value(headers, "Retry-After"):
        return True
    if header_value(headers, "X-RateLimit-Remaining") == "0":
        return True
    return names_a_rate_limit(body)


def rate_limit_verdict(
    headers: Any = None,
    *,
    status: Optional[int] = None,
    body: object = None,
) -> Optional[float]:
    """How long a refusal says to wait, bounded to one window, or None if it was not a rate limit."""
    if not is_rate_limited(headers, status = status, body = body):
        return None
    wait = rate_limit_wait_seconds(headers)
    if wait is None:
        wait = GITHUB_RATE_LIMITED_DEFAULT_SECONDS
    return min(max(wait, 0.0), GITHUB_RATE_LIMIT_MAX_SECONDS)


def note_github_rate_limited(
    headers: Any = None,
    *,
    status: Optional[int] = None,
    body: object = None,
) -> float:
    """Record the lockout and return its length; 0 if this was not a rate limit. Never shortens one already in place, or a secondary limit's brief Retry-After would release every caller early."""
    global _api_rate_limited_until
    wait = rate_limit_verdict(headers, status = status, body = body)
    if wait is None:
        return 0.0
    with _api_rate_limited_lock:
        _api_rate_limited_until = max(_api_rate_limited_until, time.monotonic() + wait)
    return wait


def github_rate_limit_remaining() -> float:
    return max(_api_rate_limited_until - time.monotonic(), 0.0)


def read_install_marker(
    binary_path: Optional[str],
    *,
    marker_name: str,
    cache: dict[str, Optional[dict]],
    log_message: str,
) -> Optional[dict]:
    """Walk up from binary_path to find the install marker JSON. None = no marker (source build / custom path) or unusable JSON. "Unusable" includes JSON that parses but is not an object: a marker holding ``[]`` or ``123`` reaches every caller as something without ``.get``, and the update planner, the backend picker and crash recovery then raise AttributeError on what is only a corrupt file."""
    if not binary_path:
        return None
    cached = cache.get(binary_path)
    if cached is not None or binary_path in cache:
        return cached
    p = Path(binary_path)
    marker: Optional[dict] = None
    # Cover all managed binary layouts (binary is 1-4 dirs deep).
    for parent in p.parents[:5]:
        candidate = parent / marker_name
        if candidate.is_file():
            try:
                marker = json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(log_message, path = str(candidate), error = str(exc))
                marker = None
            else:
                if not isinstance(marker, dict):
                    logger.debug(
                        log_message,
                        path = str(candidate),
                        error = f"marker is {type(marker).__name__}, not an object",
                    )
                    marker = None
            break
    cache[binary_path] = marker
    return marker


def cache_path_for(repo: str, cache_dir: Path) -> Path:
    safe = repo.replace("/", "__")
    return cache_dir / f"{safe}.json"


def load_disk_cache(repo: str, cache_dir: Path) -> Optional[tuple[float, Optional[str]]]:
    path = cache_path_for(repo, cache_dir)
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = payload.get("fetched_at")
    tag = payload.get("latest_tag")
    if not isinstance(ts, (int, float)):
        return None
    return float(ts), tag if isinstance(tag, str) else None


def save_disk_cache(
    repo: str, latest_tag: Optional[str], cache_dir: Path, *, log_message: str
) -> None:
    path = cache_path_for(repo, cache_dir)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "latest_tag": latest_tag}),
            encoding = "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.debug(log_message, repo = repo, error = str(exc))


def _fetch_newest_published_release(
    repo: str, timeout: float, *, log_message: str
) -> Optional[dict]:
    """Newest published release object for `repo`, bounded by a wall-clock deadline. Not redundant with `timeout`: urllib applies that per address, so a host whose leading addresses blackhole pays it once for each, and /api/inference/status reads this, so that multiplication becomes the route's response time."""
    from utils.utils import call_with_deadline
    try:
        return call_with_deadline(
            lambda: _fetch_newest_published_release_blocking(
                repo, timeout, log_message = log_message
            ),
            timeout + 1,
            name = "prebuilt-freshness-fetch",
        )
    except TimeoutError as exc:
        logger.debug(log_message, repo = repo, error = str(exc))
        return None


def _fetch_newest_published_release_blocking(
    repo: str, timeout: float, *, log_message: str
) -> Optional[dict]:
    """Newest published (non-draft, non-prerelease) release for `repo`, by ``published_at``, the way the installers resolve "latest": ``/releases/latest`` sorts by commit date and can lag the installed build. None on failure or while the lockout holds."""
    import os
    import urllib.error
    import urllib.request

    remaining = github_rate_limit_remaining()
    if remaining > 0:
        logger.debug(
            log_message, repo = repo, error = f"GitHub API rate limited for {int(remaining)}s more"
        )
        return None
    url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "unsloth-studio-freshness-check",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers = headers)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        wait = 0.0
        if exc.code in GITHUB_RATE_LIMIT_STATUS:
            wait = note_github_rate_limited(exc.headers, status = exc.code, body = error_body(exc))
        if wait:
            logger.debug(
                log_message,
                repo = repo,
                error = f"HTTP {exc.code}: rate limited, backing off {int(wait)}s",
            )
        else:
            logger.debug(log_message, repo = repo, error = str(exc))
        return None
    except (
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ) as exc:
        logger.debug(log_message, repo = repo, error = str(exc))
        return None
    if not isinstance(data, list):
        return None
    published = [
        r
        for r in data
        if isinstance(r, dict)
        and not r.get("draft")
        and not r.get("prerelease")
        and isinstance(r.get("tag_name"), str)
        and r.get("tag_name")
    ]
    if not published:
        return None
    return max(published, key = lambda r: r.get("published_at") or "")


def fetch_latest_release_tag(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time. None on failure."""
    newest = _fetch_newest_published_release(repo, timeout, log_message = log_message)
    return newest["tag_name"] if newest else None


def fetch_latest_release_assets(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[dict[str, int]]:
    """Asset name -> size (bytes) for the newest published release of `repo`, selected exactly like fetch_latest_release_tag. None on any failure."""
    newest = _fetch_newest_published_release(repo, timeout, log_message = log_message)
    if newest is None:
        return None
    assets: dict[str, int] = {}
    for a in newest.get("assets") or []:
        name, size = a.get("name"), a.get("size")
        if isinstance(name, str) and isinstance(size, int):
            assets[name] = size
    return assets


def latest_published_release(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, Optional[str]]],
    cache_dir: Callable[[], Path],
    fetch: Callable[[str], Optional[str]],
    save: Callable[[str, Optional[str]], None],
    failed_at: Optional[dict[str, float]] = None,
) -> Optional[str]:
    """Latest release tag with optional short-lived failure caching. Successes use the 24h memory and disk cache; supplying ``failed_at`` also caches failures for ``RELEASE_FAILURE_CACHE_TTL_SECONDS``, while omitting it keeps retry-on-every-call behavior."""
    if not repo:
        return None
    # Success timestamps persist to disk and need wall time. Failure timestamps are process-local and use monotonic time so clock changes cannot extend them.
    wall_now = time.time()
    if not force_refresh:
        last_failure = failed_at.get(repo) if failed_at is not None else None
        if (
            last_failure is not None
            and time.monotonic() - last_failure < RELEASE_FAILURE_CACHE_TTL_SECONDS
        ):
            cached = memo.get(repo)
            if cached:
                return cached[1]
            disk = load_disk_cache(repo, cache_dir())
            return disk[1] if disk else None
        cached = memo.get(repo)
        if cached and wall_now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
        disk = load_disk_cache(repo, cache_dir())
        if disk and wall_now - disk[0] < RELEASE_CACHE_TTL_SECONDS:
            memo[repo] = disk
            return disk[1]
    latest = fetch(repo)
    if latest is None:
        if failed_at is not None:
            failed_at[repo] = time.monotonic()
        # Keep the last-good disk value rather than poison it with None.
        disk = load_disk_cache(repo, cache_dir())
        if disk:
            memo[repo] = disk
            return disk[1]
        return None
    if failed_at is not None:
        failed_at.pop(repo, None)
    memo[repo] = (wall_now, latest)
    save(repo, latest)
    return latest


def latest_release_assets(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, dict[str, int]]],
    fetch: Callable[[str], Optional[dict[str, int]]],
) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL). None when offline and never fetched. In-memory only, so a restart re-fetches."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        cached = memo.get(repo)
        if cached and now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
    assets = fetch(repo)
    if assets is None:
        cached = memo.get(repo)
        return cached[1] if cached else None
    memo[repo] = (now, assets)
    return assets


def parse_installed_at(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = timezone.utc)
    return dt


def check_freshness(
    binary_path: Optional[str],
    *,
    threshold_days: int,
    now: Optional[datetime],
    read_marker: Callable[[Optional[str]], Optional[dict]],
    latest_release: Callable[[str], Optional[str]],
    behind: Callable[[Optional[str], Optional[str]], bool],
    display_tag: Callable[[dict], Any],
    compare_tag: Callable[[dict], Any],
) -> dict:
    """Freshness report skeleton shared by both components; the component's marker-tag choice and is_behind policy come in as callables. Fails open on missing data (behind/stale stay False)."""
    out: dict = {
        "has_marker": False,
        "stale": False,
        "behind": False,
        "installed_tag": None,
        "latest_tag": None,
        "installed_at_utc": None,
        "age_days": None,
        "published_repo": None,
        "threshold_days": int(threshold_days),
    }
    marker = read_marker(binary_path)
    if not marker:
        return out
    out["has_marker"] = True
    out["installed_tag"] = display_tag(marker)
    out["installed_at_utc"] = marker.get("installed_at_utc")
    out["published_repo"] = marker.get("published_repo")

    installed_full = compare_tag(marker)
    repo = out["published_repo"]
    if not repo or not installed_full or update_checks_disabled():
        return out
    latest = latest_release(repo)
    out["latest_tag"] = latest
    out["behind"] = behind(installed_full, latest)
    if not out["behind"]:
        return out

    installed_at = parse_installed_at(out["installed_at_utc"])
    if installed_at is None:
        return out
    now = now or datetime.now(tz = timezone.utc)
    age_seconds = (now - installed_at).total_seconds()
    out["age_days"] = max(0, int(age_seconds // 86400))
    if age_seconds >= threshold_days * 86400:
        out["stale"] = True
    return out


def format_stale_warning(info: dict, *, component: str) -> str:
    """Human-readable one-liner for stale prebuilt info."""
    age = info.get("age_days")
    installed = info.get("installed_tag") or "unknown"
    latest = info.get("latest_tag") or "unknown"
    age_str = f"{age} day{'s' if age != 1 else ''}" if age is not None else "some time"
    return (
        f"{component} prebuilt is {age_str} behind: installed "
        f"{installed}, latest {latest}. Run `unsloth studio update` "
        f"to refresh."
    )


def reset_caches(
    caches: tuple[dict, ...], *, drop_disk: bool, cache_dir: Callable[[], Path]
) -> None:
    """Drop the in-memory freshness caches; with drop_disk also the on-disk 24h release cache (see the component modules for why)."""
    for cache in caches:
        cache.clear()
    if drop_disk:
        import shutil

        # cache_dir() is a freshness-only subdir re-created on the next save_disk_cache, and ignore_errors so a missing or locked dir cannot break an otherwise successful install.
        shutil.rmtree(cache_dir(), ignore_errors = True)
