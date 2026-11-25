# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-install scan of the PyPI packages the latest-transformers sidecar installs.

Runs before Studio installs a PyPI transformers release without asking. It checks, for every
pinned package of the install plan:

* the name is one the sidecar recipe installs, and not a near miss of one (typosquat guard);
* PyPI's JSON API knows that exact release under that name and none of its files is yanked;
* OSV (osv.dev) lists no advisory for that version; a MAL-* (malicious package) entry flags it;

and for transformers itself it downloads the wheel, checks its sha256 against PyPI's published
digest, and refuses a wheel holding a ``.pth`` file or anything outside ``transformers/`` and its
dist-info. The digests it collected are returned so the install pins them (``--require-hashes``).

Only ``clean`` lets the install skip the user's confirmation. ``review`` (offline, a network or
parse failure, a known advisory, a new dependency) keeps the confirmation; ``flagged`` refuses
the install outright. Nothing here raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from typing import Sequence

from loggers import get_logger

logger = get_logger(__name__)

CLEAN = "clean"
REVIEW = "review"
FLAGGED = "flagged"

_PYPI_RELEASE_URL = "https://pypi.org/pypi/{name}/{version}/json"
_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
# Wheels come only from PyPI's file host, never a URL a JSON body could redirect us to.
_TRUSTED_FILE_HOST = "https://files.pythonhosted.org/"

_JSON_TIMEOUT_SECONDS = 10.0
_JSON_MAX_BYTES = 8 << 20
_WHEEL_TIMEOUT_SECONDS = 30.0
_WHEEL_DEADLINE_SECONDS = 180.0
_WHEEL_MAX_BYTES = 256 << 20
_READ_CHUNK_BYTES = 1 << 16

# A clean verdict is reused for a while; anything else is recomputed on the next ask.
_CLEAN_TTL_SECONDS = 6 * 60 * 60

# Every package the latest sidecar recipe may install (utils.transformers_version).
_EXPECTED_PACKAGES = frozenset(
    {
        "transformers",
        "huggingface-hub",
        "hf-xet",
        "tiktoken",
        "tokenizers",
        "safetensors",
    }
)
# transformers' core runtime dependencies, past and present. A dependency outside this set is new:
# it keeps the confirmation, and one a typo away from a known name is flagged.
_KNOWN_TRANSFORMERS_DEPS = frozenset(
    {
        "filelock",
        "hf-xet",
        "huggingface-hub",
        "numpy",
        "packaging",
        "pyyaml",
        "regex",
        "requests",
        "safetensors",
        "tokenizers",
        "tqdm",
        "typer",
        "typer-slim",
    }
)
# Our own recipe names are fixed, so anything within two edits of one is a squat. transformers'
# dependencies can legitimately change, so only a one-edit miss of a known one is flagged there.
_PLAN_TYPO_DISTANCE = 2
_DEP_TYPO_DISTANCE = 1

_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen = True)
class PackageScanResult:
    status: str
    reason: str
    # spec -> PyPI sha256 digests of that release, for --require-hashes.
    hashes: dict = field(default_factory = dict)

    @property
    def clean(self) -> bool:
        return self.status == CLEAN


def canonical_name(name: str) -> str:
    """PEP 503 normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_spec(spec: str) -> tuple[str, str | None]:
    name, _, version = spec.partition("==")
    return canonical_name(name.strip()), (version.strip() or None)


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _near_miss(name: str, known: frozenset, distance: int) -> str | None:
    """The known name *name* is a typo of, or None (an exact match is not a near miss)."""
    if name in known:
        return None
    for candidate in sorted(known):
        if _edit_distance(name, candidate) <= distance:
            return candidate
    return None


def _offline() -> bool:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UNSLOTH_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in _TRUE_VALUES:
            return True
    return False


def _read_capped(resp, max_bytes: int, deadline: float) -> bytes | None:
    read1 = getattr(resp, "read1", None) or resp.read
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            return None
        chunk = read1(_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)


def _http(url: str, payload: dict | None = None) -> bytes | None:
    """GET (or POST *payload* as JSON) *url*; None on any failure, a 404 included."""
    import urllib.request

    data = None
    headers = {"User-Agent": "unsloth-studio", "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(url, data = data, headers = headers)
        with urllib.request.urlopen(req, timeout = _JSON_TIMEOUT_SECONDS) as resp:
            return _read_capped(resp, _JSON_MAX_BYTES, time.monotonic() + 2 * _JSON_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.debug("Package scan fetch failed for %s: %s", url, exc)
        return None


def _http_json(url: str, payload: dict | None = None):
    body = _http(url, payload)
    if body is None:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _download(url: str, dest: str) -> bool:
    """Stream *url* into *dest* within the wheel budget."""
    import urllib.request

    deadline = time.monotonic() + _WHEEL_DEADLINE_SECONDS
    total = 0
    try:
        req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
        with (
            urllib.request.urlopen(req, timeout = _WHEEL_TIMEOUT_SECONDS) as resp,
            open(dest, "wb") as out,
        ):
            read1 = getattr(resp, "read1", None) or resp.read
            while True:
                if time.monotonic() >= deadline:
                    return False
                chunk = read1(_READ_CHUNK_BYTES)
                if not chunk:
                    return True
                total += len(chunk)
                if total > _WHEEL_MAX_BYTES:
                    return False
                out.write(chunk)
    except Exception as exc:
        logger.debug("Package scan download failed for %s: %s", url, exc)
        return False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_wheel(path: str, name: str, version: str) -> str | None:
    """Why the wheel is malicious-looking, or None. Every member must sit under the package or
    its dist-info, and no .pth (run at every interpreter start) may ride along."""
    module = name.replace("-", "_")
    # No .data root: pip relocates its contents (scripts, purelib, ...) outside the package.
    allowed_roots = {module, f"{module}-{version}.dist-info"}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                normalized = member.replace("\\", "/")
                if normalized.startswith("/") or ".." in normalized.split("/"):
                    return f"the {name} wheel has an unsafe path {member!r}"
                if normalized.lower().endswith(".pth"):
                    return f"the {name} wheel ships a .pth file ({member})"
                root = normalized.split("/", 1)[0]
                if root not in allowed_roots:
                    return f"the {name} wheel installs an unexpected top-level entry {root!r}"
    except zipfile.BadZipFile:
        return f"the {name} wheel is not a valid archive"
    return None


def _core_requirements(info: dict) -> list[str]:
    """Names of the release's unconditional runtime dependencies."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except Exception:
        return []
    names = []
    for raw in info.get("requires_dist") or []:
        if not isinstance(raw, str):
            continue
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.marker is not None and "extra" in str(req.marker):
            continue
        names.append(canonical_name(req.name))
    return names


def _check_release(spec: str, name: str, version: str, primary: bool) -> PackageScanResult:
    meta = _http_json(_PYPI_RELEASE_URL.format(name = name, version = version))
    if not isinstance(meta, dict) or not isinstance(meta.get("info"), dict):
        return PackageScanResult(REVIEW, f"PyPI did not describe {name} {version}")
    info = meta["info"]
    if canonical_name(str(info.get("name") or "")) != name or str(info.get("version")) != version:
        return PackageScanResult(
            FLAGGED, f"PyPI answered for {info.get('name')} {info.get('version')}, not {spec}"
        )
    files = meta.get("urls")
    if not isinstance(files, list) or not files:
        return PackageScanResult(REVIEW, f"PyPI lists no files for {spec}")
    # A review reason never ends the checks: only flagged returns early, so a later red flag
    # still refuses a release an earlier oddity would only have sent to the user.
    reviews: list[str] = []
    digests = []
    for entry in files:
        sha = (entry.get("digests") or {}).get("sha256") if isinstance(entry, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            reviews.append(f"PyPI publishes no sha256 for a {spec} file")
            continue
        if entry.get("yanked"):
            reviews.append(f"{spec} has a yanked file on PyPI")
        digests.append(sha)

    def _verdict(pinned: tuple) -> PackageScanResult:
        hashes = {spec: pinned} if pinned else {}
        return PackageScanResult(
            REVIEW if reviews else CLEAN, reviews[0] if reviews else "", hashes
        )

    if not primary:
        return _verdict(tuple(digests))

    for dep in _core_requirements(info):
        typo_of = _near_miss(dep, _KNOWN_TRANSFORMERS_DEPS, _DEP_TYPO_DISTANCE)
        if typo_of is not None:
            return PackageScanResult(
                FLAGGED, f"{spec} depends on {dep!r}, a near miss of {typo_of!r}"
            )
        if dep not in _KNOWN_TRANSFORMERS_DEPS:
            reviews.append(f"{spec} adds a new dependency, {dep}")

    wheel = next(
        (
            entry
            for entry in files
            if isinstance(entry, dict)
            and entry.get("packagetype") == "bdist_wheel"
            and str(entry.get("filename", "")).endswith("-py3-none-any.whl")
        ),
        None,
    )
    expected = (wheel.get("digests") or {}).get("sha256") if wheel else None
    if wheel is None or expected not in digests:
        reviews.append(f"PyPI has no pure-Python wheel with a sha256 for {spec}")
        return _verdict(())
    url = str(wheel.get("url") or "")
    if not url.startswith(_TRUSTED_FILE_HOST):
        return PackageScanResult(FLAGGED, f"the {spec} wheel is not hosted on PyPI ({url})")
    workdir = tempfile.mkdtemp(prefix = "unsloth_pkg_scan_")
    try:
        path = os.path.join(workdir, "package.whl")
        if not _download(url, path):
            reviews.append(f"could not download the {spec} wheel to scan it")
            return _verdict(())
        if _sha256_file(path) != expected:
            return PackageScanResult(
                FLAGGED,
                f"the downloaded {spec} wheel does not match PyPI's published sha256",
            )
        problem = _inspect_wheel(path, name, version)
        if problem is not None:
            return PackageScanResult(FLAGGED, problem)
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors = True)
    # Only the inspected wheel: an sdist (PIP_NO_BINARY) must not inherit its clean verdict.
    return _verdict((expected,))


def _check_advisories(pinned: list[tuple[str, str, str]]) -> PackageScanResult:
    payload = {
        "queries": [
            {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
            for _, name, version in pinned
        ]
    }
    answer = _http_json(_OSV_BATCH_URL, payload)
    results = answer.get("results") if isinstance(answer, dict) else None
    if not isinstance(results, list) or len(results) != len(pinned):
        return PackageScanResult(REVIEW, "the OSV advisory database could not be reached")
    # Every package is read before answering: a MAL-* entry anywhere outranks an advisory.
    review = None
    for (spec, _, _), result in zip(pinned, results):
        if not isinstance(result, dict):
            review = review or "the OSV advisory database gave a malformed answer"
            continue
        ids = [str(v.get("id")) for v in result.get("vulns") or [] if isinstance(v, dict)]
        malicious = [i for i in ids if i.startswith("MAL-")]
        if malicious:
            return PackageScanResult(
                FLAGGED, f"{spec} is reported as malicious ({', '.join(malicious[:3])})"
            )
        if result.get("next_page_token"):
            review = review or f"OSV lists more advisories for {spec} than one page holds"
        elif ids:
            shown = ", ".join(ids[:3]) + (" and more" if len(ids) > 3 else "")
            review = review or f"{spec} has known advisories ({shown})"
    return PackageScanResult(REVIEW, review) if review else PackageScanResult(CLEAN, "")


CUSTOM_INDEX_REASON = "custom package index, not verified against PyPI"

# Replace the default index: anything but PyPI here is a mirror.
_INDEX_URL_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")
# Add sources beside (or ranked above) PyPI: any value at all can serve the package.
_EXTRA_SOURCE_ENVS = (
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_FIND_LINKS",
)
_PIP_INDEX_KEYS = ("index-url", "extra-index-url", "find-links")
_UV_INDEX_KEYS = ("index-url", "default-index", "extra-index-url", "find-links", "index")


def _is_pypi_url(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() in ("pypi.org", "pypi.python.org")
        and parsed.path.rstrip("/") in ("/simple", "")
    )


def _only_pypi(value) -> bool:
    """True when every source in *value* (a string, a list, or uv's [[index]] tables) is PyPI."""
    if isinstance(value, dict):
        return _only_pypi(value.get("url") or "")
    if isinstance(value, (list, tuple)):
        return all(_only_pypi(item) for item in value)
    entries = str(value or "").split()
    return bool(entries) and all(_is_pypi_url(e) for e in entries)


def _uv_config_files() -> list[str]:
    """The uv.toml / pyproject.toml files ``uv pip install`` would read, highest priority first."""
    explicit = (os.environ.get("UV_CONFIG_FILE") or "").strip()
    if explicit:
        return [explicit]
    if (os.environ.get("UV_NO_CONFIG") or "").strip().lower() in _TRUE_VALUES:
        return []
    files: list[str] = []
    try:
        directory = os.getcwd()
    except OSError:
        directory = ""
    while directory:
        for name in ("uv.toml", "pyproject.toml"):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                files.append(candidate)
        parent = os.path.dirname(directory)
        directory = "" if parent == directory else parent
    if os.name == "nt":
        for root in (os.environ.get("APPDATA"), os.environ.get("PROGRAMDATA")):
            if root:
                files.append(os.path.join(root, "uv", "uv.toml"))
    else:
        home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        files.append(os.path.join(home, "uv", "uv.toml"))
        for root in (os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":"):
            if root:
                files.append(os.path.join(root, "uv", "uv.toml"))
        files.append("/etc/uv/uv.toml")
    return [f for f in files if os.path.isfile(f)]


def _uv_config_names_index(path: str) -> bool:
    try:
        import tomllib  # novermin
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Unreadable here: it may name a mirror, so assume it does.
            return True
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return True
    if os.path.basename(path) == "pyproject.toml":
        data = (data.get("tool") or {}).get("uv") or {}
    if not isinstance(data, dict):
        return False
    pip = data.get("pip") if isinstance(data.get("pip"), dict) else {}
    for table in (data, pip):
        for key in _UV_INDEX_KEYS:
            if table.get(key) and not _only_pypi(table[key]):
                return True
    return False


def custom_package_index() -> str | None:
    """Where a non-PyPI package source is configured for pip or uv (env, pip config, uv.toml), or
    None. The sidecar install tries uv then pip, so either one's mirror can serve the package, and
    its artifacts need not match the digests PyPI publishes."""
    for var in _INDEX_URL_ENVS + _EXTRA_SOURCE_ENVS:
        value = (os.environ.get(var) or "").strip()
        if value and not _only_pypi(value):
            return var
    try:
        from utils.transformers_version import _pip_effective_settings
        settings = _pip_effective_settings() or {}
    except Exception:
        settings = {}
    for key, value in settings.items():
        if key.rpartition(".")[2] in _PIP_INDEX_KEYS and value and not _only_pypi(value):
            return f"pip config {key}"
    for path in _uv_config_files():
        if _uv_config_names_index(path):
            return path
    return None


_cache_lock = threading.Lock()
_clean_cache: dict = {}


def clear_cache() -> None:
    with _cache_lock:
        _clean_cache.clear()


def scan_pypi_install(packages: Sequence[str], primary: str = "transformers") -> PackageScanResult:
    """Scan the install plan *packages* (``name==version`` specs, or bare names for unpinned
    optional packages). Never raises; see the module docstring for the verdicts."""
    key = (tuple(packages), primary)
    with _cache_lock:
        hit = _clean_cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < _CLEAN_TTL_SECONDS:
            return hit[1]
    try:
        result = _scan(packages, canonical_name(primary))
    except Exception as exc:
        logger.warning("Package scan failed: %s", exc)
        result = PackageScanResult(REVIEW, "the package scan did not complete")
    if result.clean:
        with _cache_lock:
            _clean_cache[key] = (time.monotonic(), result)
    else:
        logger.info("Package scan verdict %s: %s", result.status, result.reason)
    return result


def _scan(packages: Sequence[str], primary: str) -> PackageScanResult:
    if not packages:
        return PackageScanResult(REVIEW, "there is nothing to scan")
    pinned: list[tuple[str, str, str]] = []
    for spec in packages:
        if not isinstance(spec, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+(==[A-Za-z0-9_.+!-]+)?", spec
        ):
            return PackageScanResult(FLAGGED, f"unexpected install spec {spec!r}")
        name, version = _split_spec(spec)
        if name not in _EXPECTED_PACKAGES:
            typo_of = _near_miss(name, _EXPECTED_PACKAGES, _PLAN_TYPO_DISTANCE)
            detail = f", a near miss of {typo_of!r}" if typo_of else ""
            return PackageScanResult(FLAGGED, f"unexpected package {name!r}{detail}")
        if version is not None:
            pinned.append((spec, name, version))
    if not any(name == primary for _, name, _ in pinned):
        return PackageScanResult(REVIEW, f"{primary} is not pinned to an exact release")
    if _offline():
        return PackageScanResult(REVIEW, "Unsloth is offline, so the release could not be scanned")
    # Run every check before answering: flagged anywhere wins over a review elsewhere, since a
    # confirmed install still proceeds past review but never past flagged.
    hashes: dict = {}
    review = None
    for spec, name, version in pinned:
        verdict = _check_release(spec, name, version, primary = name == primary)
        if verdict.status == FLAGGED:
            return verdict
        # A review still carries the digests it verified; a confirmed install keeps them pinned.
        hashes.update(verdict.hashes)
        if not verdict.clean:
            review = review or verdict.reason
    advisories = _check_advisories(pinned)
    if advisories.status == FLAGGED:
        return advisories
    review = review or (advisories.reason if not advisories.clean else None)
    # A confirmed install past a review still pins whatever digests were verified.
    return PackageScanResult(REVIEW if review else CLEAN, review or "", hashes)
