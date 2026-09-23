# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Hugging Face Hub API, read-only, over ModelScope: ``HF_ENDPOINT`` points here.

A loopback server on its own thread serves this process and its workers, since a
synchronous Hub call made on the main event loop could never be answered by it;
the browser uses a mount in the main app, whose catalog routes need Studio auth.
"""

from __future__ import annotations

import asyncio
import re
import socket
import threading
import time
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from hub.modelscope import upstream as ms
from loggers import get_logger

logger = get_logger(__name__)

BROWSER_PREFIX = "/api/hub/modelscope"

_KINDS = {"models": "model", "datasets": "dataset"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FILTER_SCAN_PAGES = 4
# The browser cannot follow a redirect to ModelScope (no CORS), so small files are relayed.
_RELAY_LIMIT = 16 * 1024 * 1024
_UPLOAD_REFUSAL = (
    "Uploading is unavailable while ModelScope is the model source. "
    "Switch to Hugging Face in Settings to upload."
)


def _error(status: int, code: Optional[str], message: str) -> JSONResponse:
    headers = {"X-Error-Message": message}
    if code:
        headers["X-Error-Code"] = code
    return JSONResponse({"error": message}, status_code = status, headers = headers)


async def _answer(call: Callable[[], Awaitable[Response]]) -> Response:
    try:
        return await call()
    except ms.NotFound as exc:
        return _error(404, exc.code, str(exc))
    except ms.UpstreamError as exc:
        logger.info("ModelScope adapter: %s", exc)
        return _error(502, None, str(exc))


def _repo(owner: str, name: str) -> str:
    if not (_NAME.fullmatch(owner) and _NAME.fullmatch(name)) or ".." in name:
        raise HTTPException(status_code = 400, detail = "Invalid repository id.")
    return f"{owner}/{name}"


def _file_path(path: str) -> str:
    parts = path.split("/")
    if (
        len(path) > 1024
        or "\\" in path
        or any(part in ("", ".", "..") for part in parts)
        or any(ord(char) < 0x20 for char in path)
    ):
        raise HTTPException(status_code = 400, detail = "Invalid file path.")
    return path


def _kind(kinds: str) -> str:
    kind = _KINDS.get(kinds)
    if kind is None:
        raise HTTPException(status_code = 404, detail = "Not found.")
    return kind


async def _listing(kind: str, repo: str, revision: str) -> tuple[str, dict[str, dict]]:
    sha = await ms.resolve_revision(kind, repo, revision)
    return sha, await ms.files(kind, repo, sha)


def build_router(*, browser: bool) -> APIRouter:
    """``browser`` puts the catalog behind Studio auth and relays small files to signed-in callers."""
    dependencies = []
    if browser:
        from auth.authentication import get_current_subject
        dependencies = [Depends(get_current_subject)]
    router = APIRouter()
    api = APIRouter(dependencies = dependencies)

    @api.get("/api/{kinds}")
    async def search(kinds: str, request: Request) -> Response:
        kind = _kind(kinds)
        query = request.query_params
        filters = [value for value in query.getlist("filter") if value]
        pipeline_tag = query.get("pipeline_tag") or ""
        sort = ms.SORTS.get(query.get("sort") or "", "default")
        try:
            page = max(1, int(query.get("p") or 1))
        except ValueError:
            page = 1

        async def run() -> Response:
            found, current, more = [], page, False
            for _ in range(_FILTER_SCAN_PAGES if filters or pipeline_tag else 1):
                items, total = await ms.search(
                    kind, query.get("search") or "", query.get("author") or "", sort, current
                )
                entries = [ms.hub_entry(kind, item) for item in items]
                found.extend(entry for entry in entries if ms.matches(entry, filters, pipeline_tag))
                more = bool(items) and current * ms.PAGE_SIZE < total
                current += 1
                if not more or found:
                    break
            response = JSONResponse(found)
            if more:
                pairs = [(k, v) for k, v in query.multi_items() if k != "p"] + [("p", str(current))]
                # Absolute: the browser's Hub client resolves it without a base.
                response.headers["Link"] = (
                    f'<{request.url.replace(query = urlencode(pairs))}>; rel="next"'
                )
            return response

        return await _answer(run)

    async def info(kind: str, repo: str, revision: str, request: Request) -> Response:
        expand = set(request.query_params.getlist("expand"))
        # huggingface_hub wants the commit and files; the browser's expanded read does not.
        full = not expand or bool(expand & {"sha", "siblings"})

        async def run() -> Response:
            found = await ms.detail(kind, repo)
            if found is None:
                raise ms.repo_missing(repo)
            body = ms.hub_entry(kind, {**found, "id": repo})
            if full:
                sha, files = await _listing(kind, repo, revision)
                body["sha"] = sha
                body["siblings"] = [ms.sibling(path, meta) for path, meta in files.items()]
            return JSONResponse(body)

        return await _answer(run)

    @api.get("/api/{kinds}/{owner}/{name}")
    async def repo_info(kinds: str, owner: str, name: str, request: Request) -> Response:
        return await info(_kind(kinds), _repo(owner, name), "main", request)

    @api.get("/api/{kinds}/{owner}/{name}/revision/{revision:path}")
    async def repo_info_at(
        kinds: str, owner: str, name: str, revision: str, request: Request
    ) -> Response:
        return await info(_kind(kinds), _repo(owner, name), revision, request)

    @api.get("/api/{kinds}/{owner}/{name}/auth-check")
    async def auth_check(kinds: str, owner: str, name: str) -> Response:
        kind, repo = _kind(kinds), _repo(owner, name)

        async def run() -> Response:
            await ms.repo_exists(kind, repo)
            return JSONResponse({})

        return await _answer(run)

    @api.get("/api/{kinds}/{owner}/{name}/tree/{revision}")
    @api.get("/api/{kinds}/{owner}/{name}/tree/{revision}/{path:path}")
    async def tree(
        kinds: str,
        owner: str,
        name: str,
        revision: str,
        request: Request,
        path: str = "",
    ) -> Response:
        kind, repo = _kind(kinds), _repo(owner, name)
        prefix = path.strip("/")
        recursive = request.query_params.get("recursive", "").lower() in ("1", "true")

        async def run() -> Response:
            _, files = await _listing(kind, repo, revision)
            dirs = ms.directories(files)
            if prefix and prefix not in dirs:
                raise ms.NotFound("EntryNotFound", f"{prefix} not found in {repo}.")
            base = f"{prefix}/" if prefix else ""
            entries = [
                ms.tree_entry(candidate, files.get(candidate))
                for candidate in sorted(dirs | set(files))
                if candidate.startswith(base)
                and candidate != prefix
                and (recursive or "/" not in candidate[len(base) :])
            ]
            return JSONResponse(entries)

        return await _answer(run)

    @api.post("/api/{kinds}/{owner}/{name}/paths-info/{revision}")
    async def paths_info(
        kinds: str, owner: str, name: str, revision: str, request: Request
    ) -> Response:
        kind, repo = _kind(kinds), _repo(owner, name)
        wanted = [path for path in (await request.form()).getlist("paths") if isinstance(path, str)]

        async def run() -> Response:
            _, files = await _listing(kind, repo, revision)
            dirs = ms.directories(files)
            return JSONResponse(
                [
                    ms.tree_entry(path, files.get(path))
                    for path in dict.fromkeys(item.strip("/") for item in wanted)
                    if path in files or path in dirs
                ]
            )

        return await _answer(run)

    router.include_router(api)

    async def resolve(kind: str, repo: str, revision: str, path: str, request: Request) -> Response:
        path = _file_path(path)
        session = browser and await _studio_session(request)
        if browser and not session:
            if "authorization" in request.headers:
                # A stale session: the page refreshes it on a 401, but cannot read a cross-origin redirect.
                return _error(401, None, "Sign in again to browse ModelScope.")
            # Anonymous loads (README images) must not cost upstream lookups.
            return RedirectResponse(ms.branch_url(kind, repo, revision, path), status_code = 302)

        async def run() -> Response:
            sha, files = await _listing(kind, repo, revision)
            meta = files.get(path)
            if meta is None:
                raise ms.NotFound("EntryNotFound", f"{path} not found in {repo}.")
            etag = f'"{meta["sha256"]}"'
            headers = {
                "X-Repo-Commit": sha,
                "ETag": etag,
                "X-Linked-Etag": etag,
                "X-Linked-Size": str(meta["size"]),
                "Accept-Ranges": "bytes",
            }
            upstream = ms.file_url(kind, repo, sha, path)
            if request.method == "GET" and session and meta["size"] <= _RELAY_LIMIT:
                return await _relay(upstream, headers)
            # huggingface_hub downloads from a HEAD's Location, so the bytes match the commit.
            return RedirectResponse(upstream, status_code = 302, headers = headers)

        return await _answer(run)

    @router.api_route("/{owner}/{name}/resolve/{revision}/{path:path}", methods = ["GET", "HEAD"])
    async def resolve_model(
        owner: str, name: str, revision: str, path: str, request: Request
    ) -> Response:
        return await resolve("model", _repo(owner, name), revision, path, request)

    @router.api_route(
        "/datasets/{owner}/{name}/resolve/{revision}/{path:path}", methods = ["GET", "HEAD"]
    )
    async def resolve_dataset(
        owner: str, name: str, revision: str, path: str, request: Request
    ) -> Response:
        return await resolve("dataset", _repo(owner, name), revision, path, request)

    @router.api_route("/", methods = ["GET", "HEAD"])
    async def root() -> Response:
        return Response(status_code = 200 if await ms.reachable() else 503)

    @router.get("/datasets/{owner}/{name}")
    async def dataset_page(owner: str, name: str) -> Response:
        return RedirectResponse(f"{ms.MODELSCOPE}/datasets/{_repo(owner, name)}")

    @router.get("/{owner}/{name}")
    async def model_page(owner: str, name: str) -> Response:
        return RedirectResponse(f"{ms.MODELSCOPE}/models/{_repo(owner, name)}")

    @router.api_route("/{rest:path}", methods = ["POST", "PUT", "PATCH", "DELETE"])
    async def refuse_writes(rest: str) -> Response:
        return _error(403, None, _UPLOAD_REFUSAL)

    return router


async def _studio_session(request: Request) -> bool:
    from fastapi.security import HTTPAuthorizationCredentials

    from auth.authentication import get_current_subject

    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    try:
        await get_current_subject(HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token))
    except HTTPException:
        return False
    return True


async def _relay(url: str, headers: dict) -> Response:
    http, limiter = ms.client()
    async with limiter:
        try:
            response = await http.get(url, follow_redirects = True)
        except Exception as exc:  # noqa: BLE001 - any transport failure is an upstream one
            raise ms.UpstreamError(f"ModelScope request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise ms.UpstreamError(f"ModelScope answered {response.status_code}")
    return Response(
        content = response.content,
        media_type = response.headers.get("content-type", "application/octet-stream"),
        headers = headers,
    )


_listener_lock = threading.Lock()
_listener_url: Optional[str] = None


def internal_endpoint() -> str:
    """The loopback listener's base URL. Started on first use and kept for the process, so
    workers spawned under it keep working after the source changes."""
    global _listener_url
    with _listener_lock:
        if _listener_url is not None:
            return _listener_url
        import uvicorn

        app = FastAPI(docs_url = None, redoc_url = None, openapi_url = None)
        app.include_router(build_router(browser = False))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(app, log_level = "warning", access_log = False, lifespan = "off")
        )
        # Off the main thread, serve() installs no signal handlers.
        threading.Thread(
            target = lambda: asyncio.run(server.serve(sockets = [sock])),
            name = "modelscope-hub",
            daemon = True,
        ).start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("The ModelScope adapter did not start.")
            time.sleep(0.02)
        _listener_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        return _listener_url
