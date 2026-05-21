"""Event-loop regression test for Unsloth Studio issue #5642.

The bug: ``studio/backend/routes/inference.py:863`` calls
``llama_backend.detect_audio_type()`` synchronously inside an async
``/api/inference/load`` handler. ``detect_audio_type`` issues up to
eight sequential sync ``httpx.Client.post()`` requests (10 s timeout
each), which blocks the FastAPI event loop. While blocked, the
``/api/inference/load-progress`` poll and any other in-flight HTTP
request stalls, so Studio's UI never receives the load completion or a
progress update and appears frozen even though ``llama-server`` is
healthy.

This test stands up a real uvicorn server on a free port (matching
Studio's actual deployment shape) and fires a slow ``/probe`` request
from one worker thread while a second thread polls ``/health`` and
records latencies. The buggy route shape stalls every concurrent
``/health`` request behind the sync ``detect_audio_type`` call; the
fixed shape keeps them under 200 ms. A pure-asyncio repro is not
possible because asyncio timers cannot fire while the event loop
itself is blocked, so the test wouldn't be able to schedule the
concurrent ``/health`` while ``/probe`` is in its sync window.

The fake ``llama-server`` (``llama_server_shim.py``) runs in-process
(stdlib ``http.server``) and serves ``/health``, ``/tokenize``,
``/detokenize``, ``/props``, and ``/completion`` so
``detect_audio_type`` walks the real network code path (no patching
of ``httpx`` itself).
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bootstrap: locate the repo root by walking up from this file looking for
# a ``studio/backend`` sibling, then put it on sys.path so we can import
# ``core.inference.llama_cpp``.
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Walk up from this file until ``studio/backend`` exists. Falls
    back to ``<repo>/unsloth/studio/backend`` so the same test file
    works whether it lives in unslothai/unsloth (sibling
    ``studio/backend``) or under a workspace that clones the repo
    into ``./unsloth`` (i.e. ``./unsloth/studio/backend``)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
        if (parent / "unsloth" / "studio" / "backend").is_dir():
            return parent / "unsloth"
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend relative to this file. "
        "Run from inside the unslothai/unsloth repo (or a clone of it).",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
sys.path.insert(0, str(_STUDIO_BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Stub ``loggers`` and ``structlog`` -- same pattern as
# studio/backend/tests/test_llama_cpp_wait_for_health.py:27-30.
import logging as _logging  # noqa: E402

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: _logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", types.ModuleType("structlog"))

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

from llama_server_shim import FakeLlamaServer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shim_slow():
    """Fake llama-server that responds to /health instantly but stalls
    on /tokenize and /detokenize. Models the real-world bug surface:
    server is healthy but the post-health vocabulary probes take seconds."""
    srv = FakeLlamaServer(tok_delay = 0.8, detok_delay = 0.8)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def shim_fast():
    """Fake llama-server that responds to everything instantly."""
    srv = FakeLlamaServer(tok_delay = 0.0, detok_delay = 0.0)
    srv.start()
    yield srv
    srv.stop()


def _make_backend(port: int) -> LlamaCppBackend:
    """Build a barebones backend pointing at the shim. Uses the same
    ``__new__`` bypass pattern as studio/backend/tests/test_llama_cpp_*.py
    so we don't pull in subprocess / atexit / logging side effects."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._port = port
    b._api_key = None
    # is_loaded requires _process is not None AND _healthy is True.
    b._process = object()  # non-None sentinel; detect_audio_type never touches it
    b._healthy = True
    return b


# ---------------------------------------------------------------------------
# Uvicorn-in-thread harness
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class _UvicornServerThread:
    """Run a uvicorn server on a daemon thread. Cleanly stop on close()."""

    def __init__(self, app, *, host: str = "127.0.0.1", port: int) -> None:
        import uvicorn

        self.host = host
        self.port = port
        cfg = uvicorn.Config(
            app, host = host, port = port, log_level = "warning", access_log = False
        )
        self._server = uvicorn.Server(cfg)
        # Uvicorn's install_signal_handlers() must NOT run on a thread.
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._thread: threading.Thread | None = None

    def start(self) -> "_UvicornServerThread":
        self._thread = threading.Thread(
            target = self._server.run, daemon = True, name = f"uvicorn-{self.port}"
        )
        self._thread.start()
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"http://{self.host}:{self.port}/health", timeout = 0.5)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                pass
            time.sleep(0.05)
        raise RuntimeError(f"uvicorn did not become ready within {timeout}s")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout = 5.0)
            self._thread = None

    def __enter__(self) -> "_UvicornServerThread":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _build_app(backend: LlamaCppBackend, *, wrap_in_thread: bool):
    """Build a FastAPI app with the buggy or fixed route shape.

    ``/health`` is intentionally trivial. ``/probe`` mirrors the exact
    semantics of ``studio/backend/routes/inference.py:863-866`` -- the
    only differing line is whether ``detect_audio_type`` runs on the
    event loop (buggy) or in the threadpool (fixed)."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if wrap_in_thread:
        @app.get("/probe")
        async def probe():
            audio_type = await asyncio.to_thread(backend.detect_audio_type)
            return {"audio_type": audio_type}
    else:
        @app.get("/probe")
        async def probe():
            audio_type = backend.detect_audio_type()
            return {"audio_type": audio_type}

    return app


# ---------------------------------------------------------------------------
# Concurrent probe + health driver (sync, runs from worker threads)
# ---------------------------------------------------------------------------


def _drive_concurrent_probe_and_health(
    base_url: str, *, n_health: int = 12, health_gap: float = 0.05
) -> tuple[float, float, list[float]]:
    """Fire one /probe in worker A; concurrently issue ``n_health``
    /health requests from worker B with ``health_gap`` seconds between
    requests. Returns (max_health_latency, probe_elapsed, all_latencies).

    Crucially each request goes over a real TCP socket to uvicorn, so
    /health requests submitted while /probe is mid-sync-call queue on
    uvicorn's event loop -- proving (or disproving) event-loop block."""
    probe_elapsed = -1.0
    health_latencies: list[float] = []

    def fire_probe() -> None:
        nonlocal probe_elapsed
        t0 = time.perf_counter()
        with httpx.Client(timeout = 30.0) as client:
            r = client.get(f"{base_url}/probe")
            assert r.status_code == 200, r.text
        probe_elapsed = time.perf_counter() - t0

    def fire_health_loop() -> None:
        # Give the probe a moment to enter detect_audio_type before we
        # start hammering /health -- otherwise the first health request
        # may complete before the probe handler runs.
        time.sleep(0.1)
        with httpx.Client(timeout = 10.0) as client:
            for _ in range(n_health):
                t0 = time.perf_counter()
                r = client.get(f"{base_url}/health")
                health_latencies.append(time.perf_counter() - t0)
                assert r.status_code == 200, r.text
                time.sleep(health_gap)

    with ThreadPoolExecutor(max_workers = 2) as pool:
        f_probe = pool.submit(fire_probe)
        f_health = pool.submit(fire_health_loop)
        f_probe.result(timeout = 60.0)
        f_health.result(timeout = 60.0)

    assert probe_elapsed >= 0, "probe never set its elapsed time (worker crashed?)"
    return max(health_latencies), probe_elapsed, health_latencies


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_buggy_route_blocks_event_loop(shim_slow):
    """With a sync ``detect_audio_type`` call inside an async route,
    /health requests issued while /probe is in flight stall behind
    the blocking httpx.Client.post calls.

    /probe makes up to 8 vocab probes x ~0.8 s. The event loop is
    blocked for that entire window, so at least one /health request
    must observe a latency >= one tokenize delay. This is the
    behavioural canary that proves the bug class -- it is NOT the
    test that asserts the production code is fixed (see
    ``test_routes_inference_wraps_detect_audio_type_in_to_thread``).
    """
    backend = _make_backend(shim_slow.port)
    app = _build_app(backend, wrap_in_thread = False)
    port = _free_port()
    with _UvicornServerThread(app, port = port) as uv:
        max_latency, probe_elapsed, _ = _drive_concurrent_probe_and_health(
            f"http://127.0.0.1:{uv.port}"
        )
    assert probe_elapsed >= 1.0, (
        f"buggy /probe was suspiciously fast ({probe_elapsed:.2f}s); "
        "shim may not be exercising the slow path."
    )
    # The canary: at least one /health request was blocked for >= one
    # tokenize delay. That's the bug -- the sync call held the event
    # loop and uvicorn could not dispatch the inbound /health request
    # to its handler.
    assert max_latency >= 0.5, (
        f"buggy: max /health latency was only {max_latency:.3f}s; the "
        "current code shape is supposed to stall /health behind the sync "
        "detect_audio_type call. If this is failing, the bug may have "
        "been fixed elsewhere or the shim isn't being hit."
    )


def test_fixed_route_keeps_event_loop_responsive(shim_slow):
    """With the proposed fix (await asyncio.to_thread), /health
    requests stay responsive throughout the entire /probe lifetime.
    The slow tokenize / detokenize work runs in the threadpool and
    does not block uvicorn's asyncio event loop, so inbound /health
    requests are dispatched immediately."""
    backend = _make_backend(shim_slow.port)
    app = _build_app(backend, wrap_in_thread = True)
    port = _free_port()
    with _UvicornServerThread(app, port = port) as uv:
        max_latency, probe_elapsed, latencies = _drive_concurrent_probe_and_health(
            f"http://127.0.0.1:{uv.port}"
        )
    assert probe_elapsed >= 1.0, "fixed-route /probe must still exercise the slow path"
    # 250 ms upper bound absorbs CI jitter while staying well below the
    # 800 ms-per-tokenize-call threshold the bug produces. If the wrap
    # regresses this jumps to >= 500 ms (see the buggy test).
    assert max_latency < 0.25, (
        f"fixed: max /health latency was {max_latency:.3f}s "
        f"(all latencies: {[round(x, 3) for x in latencies]}); "
        "the to_thread wrap is supposed to keep the event loop responsive."
    )


def test_routes_inference_wraps_detect_audio_type_in_to_thread():
    """Static guard: studio/backend/routes/inference.py must call
    ``detect_audio_type`` via ``await asyncio.to_thread(...)`` so the
    fix for issue #5642 cannot regress silently. The behavioural tests
    above prove the pattern works; this test ensures the production
    route file actually uses it."""
    routes_inference = _REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"
    if not routes_inference.is_file():
        pytest.skip(f"routes/inference.py not present at {routes_inference}")
    text = routes_inference.read_text()
    # The buggy shape is exactly the line `_gguf_audio = llama_backend.detect_audio_type()`.
    # The fixed shape is `_gguf_audio = await asyncio.to_thread(llama_backend.detect_audio_type)`.
    assert "await asyncio.to_thread(llama_backend.detect_audio_type)" in text, (
        "routes/inference.py is missing the asyncio.to_thread wrap around "
        "llama_backend.detect_audio_type. See issue unslothai/unsloth#5642."
    )
    assert "llama_backend.detect_audio_type()" not in text, (
        "routes/inference.py still contains a bare ``llama_backend.detect_audio_type()`` "
        "call. The fix for #5642 requires every call site to go through "
        "asyncio.to_thread."
    )


def test_fast_path_load_completes_quickly(shim_fast):
    """Sanity / regression guard: with a zero-delay shim, the entire
    detect_audio_type sequence finishes in well under a second. Locks
    in an upper bound for the post-_wait_for_health work so a future
    regression that adds new sync probes is caught early."""
    backend = _make_backend(shim_fast.port)
    app = _build_app(backend, wrap_in_thread = True)
    port = _free_port()
    with _UvicornServerThread(app, port = port) as uv:
        t0 = time.perf_counter()
        with httpx.Client(timeout = 5.0) as client:
            r = client.get(f"http://127.0.0.1:{uv.port}/probe")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 2.0, (
        f"/probe took {elapsed:.2f}s against a zero-delay shim; expected <2s. "
        "Either the shim is slow or detect_audio_type grew a new blocking call."
    )
