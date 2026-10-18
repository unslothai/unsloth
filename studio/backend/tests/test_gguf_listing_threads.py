# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF variant listings stay off the default executor (#14038).

A stalled Hub holds a remote listing's thread until it answers. On the shared default executor
enough of them blocked every ``asyncio.to_thread`` caller, and the On Device picker's disk-only
listing of a downloaded repo timed out behind them.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from hub.services.models import gguf_variants


def _cached_repo(root, repo_id, quants):
    # Plain files, not blob symlinks: the layout huggingface_hub writes where symlinks are unavailable (Windows).
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    sha = "a" * 40
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(sha)
    leaf = repo_id.split("/")[1].removesuffix("-GGUF")
    for i, quant in enumerate(quants):
        (snapshot / f"{leaf}-{quant}.gguf").write_bytes(b"GGUF" + bytes([i]) * 4096)


def _pin_cache_roots(monkeypatch, active, home):
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active, hf_home = home, source = "studio", cache_home = home),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None, **kw: root if root is not None else active,
    )


async def _with_saturated_default_executor(body):
    """Run *body* while the loop's only default-executor worker is held, as stalled Hub calls do."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
    gate = threading.Event()
    holder = asyncio.ensure_future(asyncio.to_thread(gate.wait, 30))
    await asyncio.sleep(0.05)
    try:
        return await body()
    finally:
        gate.set()
        await holder


def test_disk_only_listing_answers_while_the_default_executor_is_saturated(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    _cached_repo(hub, "unsloth/Qwen3.8-27B-GGUF", ["UD-Q4_K_XL", "Q8_0"])
    _pin_cache_roots(monkeypatch, hub, tmp_path)

    async def listing():
        answer = await asyncio.wait_for(
            gguf_variants.get_gguf_variants_answer(
                "unsloth/Qwen3.8-27B-GGUF",
                prefer_local_cache = True,
                offline = True,
                include_cache_locations = True,
            ),
            10,
        )
        return answer.response

    response = asyncio.run(_with_saturated_default_executor(listing))
    assert sorted(v.quant for v in response.variants) == ["Q8_0", "UD-Q4_K_XL"]
    assert all(v.downloaded for v in response.variants)


def test_a_stalled_hub_listing_leaves_the_default_executor_free():
    async def body():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
        gate = threading.Event()
        stalled = asyncio.ensure_future(
            gguf_variants._shared_variants_scan(("stalled-hub-listing",), lambda: gate.wait(30))
        )
        await asyncio.sleep(0.05)
        try:
            # What model load, chat and every other to_thread caller need.
            return await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 10)
        finally:
            gate.set()
            await stalled

    assert asyncio.run(body()) == "free"


def test_a_disk_only_listing_does_not_queue_behind_stalled_hub_listings(monkeypatch):
    monkeypatch.setattr(gguf_variants, "_HUB_LISTING_POOL", ThreadPoolExecutor(max_workers = 1))

    async def body():
        gate = threading.Event()
        stalled = asyncio.ensure_future(
            gguf_variants._shared_variants_scan(("hub",), lambda: gate.wait(30))
        )
        await asyncio.sleep(0.05)
        try:
            return await asyncio.wait_for(
                gguf_variants._shared_variants_scan(("disk",), lambda: "disk", offline = True), 10
            )
        finally:
            gate.set()
            await stalled

    assert asyncio.run(body()) == "disk"


def test_listing_threads_carry_context_variables():
    import contextvars

    marker = contextvars.ContextVar("marker", default = None)

    async def body():
        marker.set("request")
        return await gguf_variants.listing_thread(True, marker.get)

    assert asyncio.run(body()) == "request"


def test_a_stalled_managed_account_preflight_leaves_the_default_executor_free(monkeypatch):
    # For an ungranted repo the preflight asks the Hub, so it stalls with it.
    gate = threading.Event()
    monkeypatch.setattr(gguf_variants.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(
        gguf_variants.account_access, "require_model_access", lambda *a, **k: gate.wait(30)
    )

    async def body():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
        listing = asyncio.ensure_future(
            gguf_variants.get_gguf_variants_answer("unsloth/Stalled-GGUF", local_path = "/x")
        )
        await asyncio.sleep(0.05)
        try:
            return await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 10)
        finally:
            gate.set()
            listing.cancel()

    assert asyncio.run(body()) == "free"
