# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ModelScope's public API, answered in Hugging Face Hub shapes."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import weakref
from typing import Any, Awaitable, Callable, Optional

import httpx

MODELSCOPE = "https://www.modelscope.cn"
PAGE_SIZE = 50

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_REVISIONS = {"", "main", "master", "HEAD", "refs/heads/main", "refs/heads/master"}
# ModelScope's own dataset metadata format, which ``datasets`` fails to parse.
_HIDDEN_DATASET_FILES = {"dataset_infos.json"}
_TREE_PAGE = 500
_TREE_MAX_PAGES = 200


class UpstreamError(Exception):
    """ModelScope could not answer. Never read as absence."""


class NotFound(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def repo_missing(repo: str) -> NotFound:
    return NotFound("RepoNotFound", f"Repository {repo} not found on ModelScope.")


class _TTLCache:
    def __init__(self, max_entries: int = 512):
        self._data: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            return hit[1] if hit is not None and hit[0] > time.monotonic() else None

    def put(self, key, value, ttl: float) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                self._data.clear()
            self._data[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = _TTLCache()
# A client and a limiter per event loop: the loopback listener runs its own.
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[httpx.AsyncClient, asyncio.Semaphore]]" = weakref.WeakKeyDictionary()
_clients_lock = threading.Lock()
transport: Optional[httpx.AsyncBaseTransport] = None


def client() -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
    loop = asyncio.get_running_loop()
    with _clients_lock:
        entry = _clients.get(loop)
        if entry is None:
            entry = _clients[loop] = (
                httpx.AsyncClient(
                    base_url = MODELSCOPE,
                    timeout = httpx.Timeout(20.0, connect = 10.0),
                    transport = transport,
                    headers = {"User-Agent": "unsloth-studio"},
                ),
                asyncio.Semaphore(8),
            )
        return entry


async def _get(
    path: str,
    *,
    params = None,
    headers = None,
) -> httpx.Response:
    http, limiter = client()
    async with limiter:
        try:
            return await http.get(path, params = params, headers = headers)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"ModelScope request failed: {type(exc).__name__}") from exc


async def _json(path: str, params = None) -> Any:
    response = await _get(path, params = params)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise UpstreamError(f"ModelScope answered {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError("ModelScope answered with invalid JSON") from exc


def _cached(ttl: float):
    def wrap(fn: Callable[..., Awaitable[Any]]):
        async def inner(*args):
            key = (fn.__name__, *args)
            hit = _cache.get(key)
            if hit is not None:
                return hit
            value = await fn(*args)
            _cache.put(key, value, ttl)
            return value

        return inner

    return wrap


def parse_ref_advertisement(body: bytes) -> dict[str, str]:
    refs: dict[str, str] = {}
    pos = 0
    while pos + 4 <= len(body):
        try:
            size = int(body[pos : pos + 4], 16)
        except ValueError:
            break
        if size < 4:
            pos += 4
            continue
        line = body[pos + 4 : pos + size].split(b"\0", 1)[0].decode("ascii", "replace").strip()
        pos += size
        sha, _, ref = line.partition(" ")
        if _SHA.fullmatch(sha) and ref:
            refs[ref] = sha
    return refs


@_cached(60)
async def _refs(kind: str, repo: str) -> dict[str, str]:
    # The REST API names no commit for a branch; git does.
    prefix = "/datasets" if kind == "dataset" else ""
    response = await _get(
        f"{prefix}/{repo}.git/info/refs",
        params = {"service": "git-upload-pack"},
        # Anything that does not present as git is answered 421.
        headers = {"User-Agent": "git/2.45.0"},
    )
    if response.status_code in (401, 403, 404):
        raise repo_missing(repo)
    if response.status_code != 200:
        raise UpstreamError(f"ModelScope refs answered {response.status_code}")
    refs = parse_ref_advertisement(response.content)
    if not refs:
        raise UpstreamError("ModelScope advertised no refs")
    return refs


async def repo_exists(kind: str, repo: str) -> None:
    await _refs(kind, repo)


async def resolve_revision(kind: str, repo: str, revision: str) -> str:
    if _SHA.fullmatch(revision.lower()):
        return revision.lower()
    refs = await _refs(kind, repo)
    if revision in _DEFAULT_REVISIONS:
        sha = refs.get("HEAD") or refs.get("refs/heads/master")
    else:
        name = revision.removeprefix("refs/heads/").removeprefix("refs/tags/")
        sha = refs.get(f"refs/heads/{name}") or refs.get(f"refs/tags/{name}")
    if not sha:
        raise NotFound("RevisionNotFound", f"Revision {revision} not found on ModelScope.")
    return sha


async def _file_entries(kind: str, repo: str, sha: str) -> list[dict]:
    if kind == "model":
        data = await _json(
            f"/api/v1/models/{repo}/repo/files", {"Revision": sha, "Recursive": "true"}
        )
        if data is None:
            raise repo_missing(repo)
        return (data.get("Data") or {}).get("Files") or []
    entries: list[dict] = []
    for page in range(1, _TREE_MAX_PAGES + 1):
        data = await _json(
            f"/api/v1/datasets/{repo}/repo/tree",
            {
                "Revision": sha,
                "Root": "/",
                "Recursive": "True",
                "PageNumber": page,
                "PageSize": _TREE_PAGE,
            },
        )
        if data is None:
            raise repo_missing(repo)
        batch = (data.get("Data") or {}).get("Files") or []
        entries.extend(batch)
        if len(batch) < _TREE_PAGE:
            break
    return entries


@_cached(300)
async def files(kind: str, repo: str, sha: str) -> dict[str, dict]:
    """Every file at ``sha`` by path. The one place existence is decided: ModelScope
    answers a missing file with a 500, not a 404."""
    found = {}
    for entry in await _file_entries(kind, repo, sha):
        path = entry.get("Path")
        if entry.get("Type") == "tree" or not isinstance(path, str):
            continue
        if kind == "dataset" and path in _HIDDEN_DATASET_FILES:
            continue
        found[path] = {
            "size": int(entry.get("Size") or 0),
            "sha256": str(entry.get("Sha256") or ""),
            "lfs": bool(entry.get("IsLFS")),
        }
    return found


@_cached(300)
async def detail(kind: str, repo: str) -> Optional[dict]:
    data = await _json(f"/openapi/v1/{kind}s/{repo}")
    return (data or {}).get("data") or None


@_cached(60)
async def search(kind: str, query: str, author: str, sort: str, page: int) -> tuple[list, int]:
    params: dict[str, Any] = {"page_number": page, "page_size": PAGE_SIZE}
    if query:
        params["search"] = query
    if author:
        params["author"] = author
    if sort != "default":
        params["sort"] = sort
    body = ((await _json(f"/openapi/v1/{kind}s", params)) or {}).get("data") or {}
    return body.get(f"{kind}s") or [], int(body.get("total_count") or 0)


@_cached(30)
async def reachable() -> bool:
    try:
        response = await _get("/openapi/v1/models", params = {"page_size": 1})
    except UpstreamError:
        return False
    return response.status_code < 500


def file_url(kind: str, repo: str, sha: str, path: str) -> str:
    from urllib.parse import quote
    return f"{MODELSCOPE}/{kind}s/{repo}/resolve/{sha}/{quote(path)}"


def branch_url(kind: str, repo: str, revision: str, path: str) -> str:
    """``path`` on a named revision, without asking ModelScope; its default branch is ``master``."""
    from urllib.parse import quote

    name = revision.removeprefix("refs/heads/").removeprefix("refs/tags/")
    return file_url(
        kind, repo, "master" if revision in _DEFAULT_REVISIONS else quote(name, safe = ""), path
    )


SORTS = {"downloads": "downloads", "likes": "likes", "lastModified": "last_modified"}
_EPOCH = "1970-01-01T00:00:00Z"


def _tag_parts(tags) -> tuple[list[str], list[str], Optional[str], list[str]]:
    libraries, hub_tags, model_type, tasks = [], [], None, []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        prefix, _, value = tag.partition(":")
        if not value:
            continue
        if prefix == "library":
            libraries.append("transformers" if value == "transformer" else value)
        elif prefix == "model_type":
            model_type = value
        elif prefix == "task":
            tasks.append(value)
        elif prefix == "license":
            hub_tags.append(tag)
        elif prefix == "custom_tag":
            hub_tags.append(value)
    return libraries, hub_tags, model_type, tasks


def hub_entry(kind: str, item: dict) -> dict:
    repo = str(item.get("id") or "")
    libraries, tags, model_type, tag_tasks = _tag_parts(item.get("tags"))
    tasks = [task for task in (item.get("tasks") or []) if isinstance(task, str)] or tag_tasks
    downloads = int(item.get("downloads") or 0)
    entry = {
        "_id": repo,
        "id": repo,
        "author": repo.split("/", 1)[0],
        "private": bool(item.get("private")),
        "gated": "manual" if item.get("gated") else False,
        "downloads": downloads,
        "downloadsAllTime": downloads,
        "likes": int(item.get("likes") or 0),
        # The browser's Hub client rejects an entry without it.
        "lastModified": item.get("last_modified") or item.get("created_at") or _EPOCH,
        "createdAt": item.get("created_at") or _EPOCH,
        "tags": list(dict.fromkeys(libraries + tags + tasks)),
    }
    if kind == "dataset":
        entry["cardData"] = {"pretty_name": item.get("display_name")}
        return entry
    params = int(item.get("params") or 0)
    entry["modelId"] = repo
    entry["pipeline_tag"] = tasks[0] if tasks else None
    entry["library_name"] = (
        "transformers" if "transformers" in libraries else next(iter(libraries), None)
    )
    if model_type:
        entry["config"] = {"model_type": model_type}
    if params and "gguf" in libraries:
        entry["gguf"] = {"total": params, **({"architecture": model_type} if model_type else {})}
    elif params and "safetensors" in libraries:
        entry["safetensors"] = {"total": params, "parameters": {"BF16": params}}
    return entry


def matches(entry: dict, filters: list[str], pipeline_tag: str) -> bool:
    tags = {tag.lower() for tag in entry.get("tags", [])}
    pipeline_tag = pipeline_tag.lower()
    if pipeline_tag and pipeline_tag not in tags and entry.get("pipeline_tag") != pipeline_tag:
        return False
    library = str(entry.get("library_name")).lower()
    return all(f.lower() in tags or f.lower() == library for f in filters)


def sibling(path: str, meta: dict) -> dict:
    entry = {"rfilename": path, "size": meta["size"], "blobId": meta["sha256"]}
    if meta["lfs"]:
        entry["lfs"] = {"sha256": meta["sha256"], "size": meta["size"], "pointerSize": 134}
    return entry


def tree_entry(path: str, meta: Optional[dict]) -> dict:
    if meta is None:
        return {"type": "directory", "oid": "", "size": 0, "path": path}
    entry = {"type": "file", "oid": meta["sha256"], "size": meta["size"], "path": path}
    if meta["lfs"]:
        entry["lfs"] = {"oid": meta["sha256"], "size": meta["size"], "pointerSize": 134}
    return entry


def directories(paths) -> set[str]:
    found = set()
    for path in paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            found.add("/".join(parts[:index]))
    return found
