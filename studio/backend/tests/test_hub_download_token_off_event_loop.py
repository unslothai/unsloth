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
