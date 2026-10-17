# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp models Unsloth Studio can offer and the Audio pages each runs on: every umbrella
folder, plus the Community repos the runtime's ``model_specs/`` download from. Public data only.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, RepoFile, recommended_gguf_files
from core.inference.audio_workflows import AUDIO_WORKFLOW_IDS
from hub.utils.state_dir import state_root
from loggers import get_logger
from models.inference import AudioCatalogModel, AudioCatalogResponse

logger = get_logger(__name__)

_SAVED_NAME = "audio_cpp_catalog.json"
_MAX_SAVED_BYTES = 2 << 20
_MAX_SPEC_FILES = 512
_MAX_SPEC_BYTES = 1 << 20
_MAX_PROBES = 24
_WORKERS = 4
_DEADLINE_SECONDS = 10.0
_LIVE_TTL_SECONDS = 300.0
# A saved or cache-only answer is retried sooner, but not on every call of an offline host.
_DEGRADED_TTL_SECONDS = 30.0
_FAMILY_RE = re.compile(r"^[a-z0-9_]+$")
_STATUSES = frozenset(("supported", "community", "experimental", "testing", "wip"))
_TASKS = frozenset(("tts", "music", "asr", "sep"))
_NO_FAMILY = "Unsloth Studio could not tell which model family this is."
_NO_PAGE = "Unsloth Studio has no Audio page for this model yet."

_cache_lock = threading.Lock()
_cache: dict[tuple, tuple[float, float, AudioCatalogResponse]] = {}


def forget_audio_catalog() -> None:
    with _cache_lock:
        _cache.clear()


def audio_catalog_payload(*, network: bool = True) -> AudioCatalogResponse:
    """Every audio.cpp row, unfiltered: the route filters per account. Never raises for Hub/IO."""
    specs_dir = _model_specs_dir()
    try:
        key = (bool(network), str(specs_dir), specs_dir.stat().st_mtime if specs_dir else None)
    except OSError:
        key = (bool(network), None, None)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < hit[1]:
        payload = hit[2]
    else:
        payload = _build(network, specs_dir)
        ttl = _LIVE_TTL_SECONDS if payload.source == "hub" else _DEGRADED_TTL_SECONDS
        with _cache_lock:
            _cache[key] = (time.monotonic(), ttl, payload)
    # Downloads change faster than the listing; downloaded_models() keeps its own short cache.
    try:
        ids = {m.id.lower() for m in acm.downloaded_models()}
    except Exception as exc:  # noqa: BLE001 - an unreadable cache marks nothing downloaded
        logger.debug("audio.cpp catalog: downloaded models unreadable: %s", exc)
        ids = set()
    models = [m.model_copy(update = {"downloaded": m.id.lower() in ids}) for m in payload.models]
    return payload.model_copy(update = {"models": models})


def _model_specs_dir() -> Optional[Path]:
    """``model_specs/`` beside the installed runtime, found the way ``runtime_spec`` looks."""
    try:
        from core.inference.audio_cpp_server import find_audio_cpp_server_binary

        binary = find_audio_cpp_server_binary()
        parents = list(Path(binary).resolve().parents)[:3] if binary else []
        return next((p / "model_specs" for p in parents if (p / "model_specs").is_dir()), None)
    except Exception:  # noqa: BLE001 - no runtime, no specs
        return None


def _spec_index(specs_dir: Optional[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if specs_dir is None:
        return out
    try:
        paths = sorted(specs_dir.glob("*.json"))[:_MAX_SPEC_FILES]
    except OSError:
        return out
    for path in paths:
        try:
            if path.stat().st_size > _MAX_SPEC_BYTES:
                continue
            data = json.loads(path.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            continue
        family = data.get("family", path.stem) if isinstance(data, dict) else None
        if isinstance(family, str) and _FAMILY_RE.match(family):
            out.setdefault(family, data)
    return out


def _spec_targets(index: dict[str, dict], folders: Sequence[str]) -> tuple[dict, dict]:
    """``({umbrella folder: family}, {community repo: family})`` from the specs' gguf packages."""
    by_lower = {f.lower(): f for f in folders}
    umbrella: dict[str, str] = {}
    community: dict[str, str] = {}
    for family, spec in index.items():
        packages = spec.get("packages")
        base = _dict(_dict(spec.get("package_defaults")).get("download"))
        for package in packages if isinstance(packages, list) else ():
            if not isinstance(package, dict) or package.get("format") != "gguf":
                continue  # other formats are conversion sources
            repo = str({**base, **_dict(package.get("download"))}.get("repo") or "").strip()
            if not repo or repo.lower() == AUDIO_CPP_REPO.lower():
                files = package.get("files") if isinstance(package.get("files"), list) else [""]
                where = str(package.get("target_directory") or files[0] or "")
                folder = by_lower.get(where.replace("\\", "/").strip("/").split("/")[0].lower())
                if folder:
                    umbrella.setdefault(folder, family)
            elif acm._REPO_ID_RE.match(repo):
                community.setdefault(repo, family)
    return umbrella, community


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _saved_path(create: bool = False) -> Optional[Path]:
    root = state_root(create = create)
    return root / _SAVED_NAME if root is not None else None


def _read_saved() -> dict:
    """The persisted listing; anything unreadable or malformed reads as none."""
    try:
        path = _saved_path()
        if path is None or path.stat().st_size > _MAX_SAVED_BYTES:
            return {}
        data = json.loads(path.read_text(encoding = "utf-8"))
        folders = {}
        for name, entry in data["folders"].items():
            files = [RepoFile(str(rel), int(size)) for rel, size in entry["files"]]
            # Only repo paths under their own folder: a damaged file must not point at the host.
            if not re.fullmatch(r"[\w.\-]+", name) or name in (".", ".."):
                continue
            files = [f for f in files if f.path.startswith(f"{name}/") and ".." not in f.path]
            family = entry.get("family")
            folders[name] = (files, family if isinstance(family, str) else None, entry.get("spec"))
        if data.get("schema") != 1:
            return {}
        return {"fetched_at": float(data["fetched_at"]), "folders": folders}
    except Exception:  # noqa: BLE001 - missing, damaged or foreign: no saved listing
        return {}


def _write_saved(fetched_at: float, folders: dict, found: dict) -> None:
    def dump(with_specs: bool) -> bytes:
        rows = {
            name: {
                "files": [[f.path, f.size] for f in files],
                "family": found.get(name, (None, None))[0],
                "spec": found.get(name, (None, None))[1] if with_specs else None,
            }
            for name, files in folders.items()
        }
        data = {"schema": 1, "fetched_at": fetched_at, "folders": rows}
        return json.dumps(data, separators = (",", ":")).encode("utf-8")

    text = dump(True)
    if len(text) > _MAX_SAVED_BYTES:
        text = dump(False)
    path = _saved_path(create = True)
    if path is None or len(text) > _MAX_SAVED_BYTES:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(text)
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("audio.cpp catalog: could not save the listing: %s", exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


def _group(files: Sequence[RepoFile]) -> dict[str, list[RepoFile]]:
    out: dict[str, list[RepoFile]] = {}
    for f in files:
        head, sep, _rest = f.path.partition("/")
        if sep and not acm._SKIP_FILE_RE.search(f.path):
            out.setdefault(head, []).append(f)
    return {k: v for k, v in out.items() if any(f.path.lower().endswith(".gguf") for f in v)}


def _bounded(jobs: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Run ``jobs`` on a few workers under one deadline; a late or failed job is left out."""
    out: dict[str, Any] = {}
    if not jobs:
        return out
    pool = ThreadPoolExecutor(max_workers = _WORKERS, thread_name_prefix = "audio-catalog")
    try:
        futures = {pool.submit(job): name for name, job in jobs.items()}
        done, _late = wait(futures, timeout = _DEADLINE_SECONDS)
        for future in done:
            try:
                out[futures[future]] = future.result()
            except Exception as exc:  # noqa: BLE001 - one unreadable repo never hides the rest
                logger.debug("audio.cpp catalog: %s: %s", futures[future], exc)
    finally:
        pool.shutdown(wait = False, cancel_futures = True)
    return out


def _ranked(
    files: Sequence[RepoFile],
    folder: str,
    policy = None,
) -> Optional[acm.AudioCppVariant]:
    if policy is not None and policy.package:
        variants = acm._package_variants(policy, files, folder)
    else:
        variants = sorted(acm._single_file_variants(files, folder), key = acm._variant_rank)
    return variants[0] if variants else None


def _text(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _row(
    model_id: str,
    repo_id: str,
    folder: str,
    family: Optional[str],
    spec: Optional[dict],
    files: Sequence[RepoFile],
) -> AudioCatalogModel:
    spec = spec if isinstance(spec, dict) else {}
    display = _text(spec.get("display_name"), 120) or folder or repo_id.split("/")[-1]
    names = (model_id, folder, *(Path(f.path).name for f in files if f.path.endswith(".gguf")))
    workflows: list[str] = []
    task, size, reason = "", None, _NO_FAMILY
    if family:
        try:
            policy = acm.family_policy(family, spec or None, names, display)
            bindings = policy.workflows
            workflows = [w for w in AUDIO_WORKFLOW_IDS if w in bindings]
            task = policy.task if policy.task in _TASKS else ""
            reason = policy.unsupported or (None if workflows else _NO_PAGE)
            wanted = set(recommended_gguf_files(spec or None))
            picked = [f for f in files if Path(f.path).name in wanted]
            variant = None if picked else _ranked(files, folder, policy)
            size = sum(f.size for f in picked) if picked else (variant and variant.size_bytes)
        except Exception as exc:  # noqa: BLE001 - a malformed spec refuses its row, not the list
            logger.debug("audio.cpp catalog: policy of %s failed: %s", family, exc)
            reason = "Unsloth Studio could not read this model's spec."
    languages = spec.get("languages")
    if isinstance(languages, list):
        languages = [x for x in languages if isinstance(x, str) and x][:32] or None
    else:
        languages = None
    status = spec.get("status")
    return AudioCatalogModel(
        id = model_id,
        repo_id = repo_id,
        folder = folder,
        source = "umbrella" if folder else "community",
        display_name = display,
        family = family,
        task = task,
        workflows = [] if reason else workflows,
        description = _text(spec.get("description"), 280),
        languages = None if languages == ["auto"] else languages,
        status = status if status in _STATUSES else None,
        size_bytes = size or None,
        unsupported_reason = reason or None,
    )


def _build(network: bool, specs_dir: Optional[Path]) -> AudioCatalogResponse:
    index = _spec_index(specs_dir)
    files: list[RepoFile] = []
    from_hub = False
    try:
        # Anonymous (False): the repo is public, so neither cache nor saved file holds a token.
        files, from_hub = acm.list_files(AUDIO_CPP_REPO, "", False, network = network)
    except Exception as exc:  # noqa: BLE001 - no listing at all degrades to the saved one
        logger.debug("audio.cpp catalog: umbrella listing failed: %s", exc)
    saved = _read_saved()
    saved_folders = saved.get("folders", {})
    if from_hub:
        source, fetched_at, folders = "hub", time.time(), _group(files)
    elif saved_folders:
        source, fetched_at = "saved", saved["fetched_at"]
        folders = {name: entry[0] for name, entry in saved_folders.items()}
    else:
        source, fetched_at, folders = "cache", time.time(), _group(files)
    targets, community = _spec_targets(index, list(folders))
    try:
        snapshots = acm.snapshot_dirs(AUDIO_CPP_REPO)
    except Exception:  # noqa: BLE001 - no cache: nothing downloaded to read
        snapshots = []
    found: dict[str, tuple[Optional[str], Optional[dict]]] = {}
    probes: dict[str, Callable[[], Any]] = {}
    online = network and not acm.hub_offline()
    for name, folder_files in folders.items():
        if name in targets:
            found[name] = (targets[name], None)
            continue
        local = [s / f.path for f in folder_files if f.path.endswith(".gguf") for s in snapshots]
        local = next((p for p in local if p.is_file()), None)
        header = acm.read_local_header(local) if local is not None else None
        if header is not None and header.is_audio_cpp and header.family:
            found[name] = (header.family, header.spec)
            continue
        family = acm.family_from_names((name, *(Path(f.path).name for f in folder_files)))
        _files, saved_family, saved_spec = saved_folders.get(name) or ((), None, None)
        if family not in acm.FAMILIES and family not in index and saved_family:
            found[name] = (saved_family, _dict(saved_spec) or None)
            continue
        found[name] = (family, None)
        if family in acm.FAMILIES or family in index or not online or len(probes) >= _MAX_PROBES:
            continue
        main = _ranked(folder_files, name)
        if main is not None:
            size = next((f.size for f in main.files if f.path == main.main_file), 0)
            probes[name] = lambda path = main.main_file, size = size: acm.read_remote_header(
                AUDIO_CPP_REPO, path, False, size = size or None
            )
    for name, header in _bounded(probes).items():
        if header is not None and header.is_audio_cpp and header.family:
            found[name] = (header.family, header.spec if isinstance(header.spec, dict) else None)
    if source == "hub":
        _write_saved(fetched_at, folders, found)
    rows = []
    for name in sorted(folders, key = str.lower):
        family, spec = found.get(name, (None, None))
        if family and not _FAMILY_RE.match(family):
            family = None
        spec = index.get(family or "") or spec
        model_id = f"{AUDIO_CPP_REPO}/{name}"
        rows.append(_row(model_id, AUDIO_CPP_REPO, name, family, spec, folders[name]))
    listings = _bounded(
        {
            repo: (lambda repo = repo: acm.list_files(repo, "", False, network = network)[0])
            for repo in community
        }
    )
    for repo in sorted(community, key = str.lower):
        family = community[repo]
        rows.append(_row(repo, repo, "", family, index.get(family), listings.get(repo) or []))
    return AudioCatalogResponse(
        source = source,
        fetched_at = fetched_at,
        runtime_specs = specs_dir is not None,
        models = rows,
    )
