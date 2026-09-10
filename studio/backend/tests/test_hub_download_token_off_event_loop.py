# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Slow saved-token refresh must not stall requests or orphan claimed downloads."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import huggingface_hub.utils

from hub.schemas.downloads import DownloadModelRequest, DownloadDatasetRequest
from hub.services import download_lifecycle
from hub.services.models import downloads as models
from hub.services.datasets import downloads as datasets
from hub.utils import download_manifest, download_registry
from utils import hf_cache_settings


@pytest.fixture(params = ["model", "dataset"])
def download(monkeypatch, tmp_path, request):
    model = request.param == "model"
    service = models if model else datasets
    request_cls = DownloadModelRequest if model else DownloadDatasetRequest
    handler = service.download_model_response if model else service.download_dataset_response
    registry = download_registry.DownloadRegistry()
    monkeypatch.setattr(service, "_registry", registry)
    monkeypatch.setattr(service, "resolve_cached_repo_id_case", lambda repo, **kw: repo)
    monkeypatch.setattr(models, "_reject_if_load_in_flight", lambda repo: None)
    monkeypatch.setattr(models, "_load_in_flight", lambda repo: False)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *a, **kw: None)
    paths = SimpleNamespace(
        hub_cache = tmp_path / "hub",
        xet_cache = tmp_path / "xet",
        child_env = lambda *a: {},
    )
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    spawned = []
    registered = threading.Event()
    proc = SimpleNamespace(pid = 4242, poll = lambda: None)

    def popen(*args, **kwargs):
        spawned.append(kwargs["env"]["HF_TOKEN"])
        return proc

    def register(registry, key, proc, **kwargs):
        accepted = registry.register_process(key, proc)
        registered.set()
        return accepted

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", popen)
    monkeypatch.setattr(download_lifecycle, "register_worker", register)
    monkeypatch.setattr(huggingface_hub.utils, "get_token_to_send", lambda token: "hf_fixture")
    return SimpleNamespace(
        handler = handler,
        body = request_cls(repo_id = "fixture/public", use_xet = False),
        registry = registry,
        key = models._download_job_key("fixture/public", None)
        if model
        else datasets._download_job_key("fixture/public"),
        repo_type = request.param,
        proc = proc,
        spawned = spawned,
        registered = registered,
    )


@pytest.mark.parametrize("cancel_request", [False, True])
def test_slow_token_resolution_keeps_loop_responsive(monkeypatch, download, cancel_request):
    entered = threading.Event()
    release = threading.Event()
    responsive = threading.Event()
    observations = []

    def resolve(token):
        entered.set()
        assert release.wait(15), "observer failed to release token resolution"
        return "hf_fixture"

    monkeypatch.setattr(huggingface_hub.utils, "get_token_to_send", resolve)

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(download.handler(download.body, None))

        def probe():
            if cancel_request:
                task.cancel()
            responsive.set()

        def observer():
            try:
                if not entered.wait(10):
                    observations.append(False)
                    return
                loop.call_soon_threadsafe(probe)
                observations.append(responsive.wait(5))
            finally:
                release.set()

        thread = threading.Thread(target = observer, daemon = True)
        thread.start()
        try:
            if cancel_request:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task)["accepted"] is True
            assert await asyncio.to_thread(download.registered.wait, 10)
        finally:
            release.set()
            await asyncio.to_thread(thread.join, 15)

    asyncio.run(run())
    assert observations == [True], "token resolution blocked the event loop"
    assert download.spawned == ["hf_fixture"]
    assert download.registry.get_process(download.key) is download.proc


def test_cancellation_while_executor_busy_leaves_no_claim(monkeypatch, download):
    release = threading.Event()
    original_transport = download_lifecycle.resolve_transport

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
        task = asyncio.current_task()
        blocker = None

        def transport(use_xet):
            nonlocal blocker
            # Earlier preparation awaits have finished. Occupy the only thread so
            # the next offloaded operation is queued when the request is cancelled.
            blocker = loop.run_in_executor(None, release.wait, 15)
            loop.call_soon(task.cancel)
            return original_transport(use_xet)

        monkeypatch.setattr(download_lifecycle, "resolve_transport", transport)
        try:
            with pytest.raises(asyncio.CancelledError):
                await download.handler(download.body, None)
        finally:
            release.set()
            if blocker is not None:
                await blocker
            await asyncio.to_thread(lambda: None)

    asyncio.run(run())
    assert download.registry.get_job(download.key).state == "idle"
    assert download.registry.get_process(download.key) is None
    assert download.spawned == []


def test_cancel_stops_worker_registered_after_initial_lookup(monkeypatch, download):
    import logging

    registry = download.registry
    assert registry.claim(
        download.key,
        download_registry.TRANSPORT_HTTP,
        repo_type = download.repo_type,
        repo_id = "fixture/public",
    )[0]
    killed = []
    download.proc.kill = lambda: killed.append(True)
    get_process = registry.get_process
    first = True

    def lookup(key):
        nonlocal first
        proc = get_process(key)
        if first:
            first = False
            assert proc is None
            # The launch thread registers after cancel reads None but before it
            # arms pending cancellation. Registration cannot see the future flag.
            assert registry.register_process(key, download.proc)
        return proc

    monkeypatch.setattr(registry, "get_process", lookup)
    assert (
        download_lifecycle.cancel_worker(
            registry,
            download.key,
            generation = registry.current_generation(download.key),
            label = "fixture/public",
            logger = logging.getLogger(__name__),
        )
        == "cancelling"
    )
    assert killed == [True]


@pytest.mark.parametrize("cancel_download", [False, True])
def test_token_failure_preserves_download_cancellation(monkeypatch, download, cancel_download):
    from fastapi import HTTPException
    from hub.schemas.downloads import CancelDownloadRequest, CancelDatasetDownloadRequest

    entered = threading.Event()
    release = threading.Event()

    def resolve(token):
        entered.set()
        assert release.wait(10), "test did not release token resolution"
        raise RuntimeError("fixture token exchange failed")

    monkeypatch.setattr(huggingface_hub.utils, "get_token_to_send", resolve)

    async def run():
        task = asyncio.create_task(download.handler(download.body, None))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if cancel_download:
                model = download.repo_type == "model"
                handler = (
                    models.cancel_download_model_response
                    if model
                    else datasets.cancel_dataset_download_response
                )
                request_cls = CancelDownloadRequest if model else CancelDatasetDownloadRequest
                result = await handler(
                    request_cls(
                        repo_id = "fixture/public",
                        generation = download.registry.current_generation(download.key),
                    )
                )
                assert result["state"] == "cancelling"
            release.set()
            if cancel_download:
                assert (await task)["state"] == "cancelled"
            else:
                with pytest.raises(HTTPException) as exc:
                    await task
                assert exc.value.status_code == 500
                assert "fixture token exchange failed" in exc.value.detail
        finally:
            release.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions = True)

    asyncio.run(run())
    state = download.registry.get_job(download.key)
    assert state.state == ("cancelled" if cancel_download else "error")
    assert state.error == (None if cancel_download else "fixture token exchange failed")
    assert download.spawned == []
    assert download.registry.get_process(download.key) is None


@pytest.mark.parametrize("failure_at", ["token", "manifest", "popen", None])
def test_scoped_manifest_ownership_on_spawn_failure(monkeypatch, tmp_path, failure_at):
    import json
    from pathlib import Path
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(child_env = lambda: {})
    )
    error = (
        RuntimeError("fixture token exchange failed")
        if failure_at == "token"
        else OSError("fixture spawn failure")
    )
    created = []
    write = download_lifecycle.write_files_manifest
    proc = object()
    files = ["model_index.json", "weights.safetensors"]
    commands = []

    def resolve(token):
        if failure_at == "token":
            raise error
        return None

    def manifest(names):
        path = write(names)
        created.append(Path(path))
        return path

    def dump(*args, **kwargs):
        raise error

    def popen(args, **kwargs):
        commands.append(args)
        path = Path(args[args.index("--files-json") + 1])
        assert json.loads(path.read_text(encoding = "utf-8")) == files
        if failure_at == "popen":
            raise error
        return proc

    monkeypatch.setattr(huggingface_hub.utils, "get_token_to_send", resolve)
    monkeypatch.setattr(download_lifecycle, "write_files_manifest", manifest)
    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", popen)
    if failure_at == "manifest":
        monkeypatch.setattr(json, "dump", dump)

    def spawn():
        return models._spawn_download_worker(
            "fixture/public", "@diffusion", None, use_xet = False, files = files
        )

    if failure_at is None:
        assert spawn() is proc
        assert len(created) == 1
        assert created[0].exists(), "a started worker must retain its manifest"
    else:
        with pytest.raises(type(error)) as exc:
            spawn()
        assert exc.value is error
        assert list(tmp_path.glob("unsloth-dl-files-*.json")) == []
    if failure_at == "token":
        assert created == []
    assert len(commands) == (1 if failure_at in (None, "popen") else 0)


@pytest.mark.parametrize(
    "phase", ["registration_rejected", "cancelled_after_registration", "kill_failed"]
)
def test_exited_worker_releases_unread_scoped_manifest(monkeypatch, tmp_path, phase):
    import io
    import logging
    import subprocess
    from pathlib import Path
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = Path(download_lifecycle.write_files_manifest(["weights.safetensors"]))
    registry = download_registry.DownloadRegistry()
    key = models._download_job_key("fixture/public", "@diffusion")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "fixture/public",
        variant = "@diffusion",
        hub_cache = str(tmp_path / "hub"),
    )[0]
    exited = False

    def kill():
        nonlocal exited
        if phase == "kill_failed":
            raise PermissionError("fixture kill failed")
        exited = True

    def wait(timeout = None):
        if not exited:
            raise subprocess.TimeoutExpired("fixture-worker", timeout or 0)
        return -9

    proc = SimpleNamespace(
        pid = 4242,
        args = ["python", "worker", "--files-json", str(path)],
        stderr = io.BytesIO(),
        kill = kill,
        wait = wait,
        poll = lambda: -9 if exited else None,
    )
    kwargs = dict(
        hf_token = None,
        label = "fixture/public",
        log_prefix = "Download",
        logger = logging.getLogger(__name__),
        repo_type = "model",
        repo_id = "fixture/public",
        transport = download_registry.TRANSPORT_HTTP,
    )
    if phase == "cancelled_after_registration":
        assert registry.register_process(key, proc)
        assert registry.request_cancel(key, proc, registry.current_generation(key))
        proc.kill()
        assert download_lifecycle.finalize_worker_exit(registry, key, proc, **kwargs) == "cancelled"
    else:
        assert registry.mark_pending_cancel(key, registry.current_generation(key))
        assert (
            download_lifecycle.register_worker(
                registry, key, proc, watch_name = "fixture-watch", **kwargs
            )
            is False
        )
    assert path.exists() is (phase == "kill_failed")
