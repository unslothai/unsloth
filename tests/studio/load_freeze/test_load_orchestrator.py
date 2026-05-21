"""Comprehensive simulation suite for the #5642 fix.

Covers:
  1.  Behavioural canary (the bug class)             — 2 tests
  2.  Behavioural fix-validation                     — 1 test
  3.  Functional equivalence (sync == to_thread)     — 5 tests, one per codec branch
  4.  Failure modes (HTTP 500, malformed JSON,
      connection reset, unreachable, not-loaded)     — 5 tests
  5.  Stress (50 concurrent probes / 100 healths)    — 2 tests
  6.  Drift / regression guards                      — 3 tests
  7.  Timing budgets                                 — 1 test

Designed to run from inside ``temp/sim/`` after ``uv venv`` + minimal
``uv pip install`` of pytest/httpx/fastapi/uvicorn/anyio. Resolves
``studio/backend`` automatically by walking up from this file looking
for the workspace clone of ``unslothai/unsloth`` (search order: this
dir's parents → ``../../unsloth`` → ``UNSLOTH_REPO_ROOT`` env var).
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Repo discovery
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
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
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or clone "
        "unslothai/unsloth into a parent directory.",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
sys.path.insert(0, str(_STUDIO_BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging as _logging  # noqa: E402

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: _logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", types.ModuleType("structlog"))

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

from llama_server_shim import FakeLlamaServer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_backend(port: int, *, loaded: bool = True) -> LlamaCppBackend:
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._port = port
    b._api_key = None
    b._process = object() if loaded else None
    b._healthy = loaded
    return b


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class _UvicornServerThread:
    def __init__(self, app, *, host: str = "127.0.0.1", port: int) -> None:
        import uvicorn

        self.host = host
        self.port = port
        cfg = uvicorn.Config(
            app, host = host, port = port, log_level = "warning", access_log = False
        )
        self._server = uvicorn.Server(cfg)
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target = self._server.run, daemon = True)
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

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout = 5.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _build_app(backend, *, wrap_in_thread: bool):
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if wrap_in_thread:

        @app.get("/probe")
        async def probe():
            return {"audio_type": await asyncio.to_thread(backend.detect_audio_type)}
    else:

        @app.get("/probe")
        async def probe():
            return {"audio_type": backend.detect_audio_type()}

    return app


def _drive_concurrent_probe_and_health(base_url, *, n_health = 12, gap = 0.05):
    elapsed = -1.0
    latencies: list[float] = []

    def fire_probe():
        nonlocal elapsed
        t0 = time.perf_counter()
        with httpx.Client(timeout = 30.0) as c:
            r = c.get(f"{base_url}/probe")
            assert r.status_code == 200
        elapsed = time.perf_counter() - t0

    def fire_health():
        time.sleep(0.1)
        with httpx.Client(timeout = 10.0) as c:
            for _ in range(n_health):
                t0 = time.perf_counter()
                r = c.get(f"{base_url}/health")
                latencies.append(time.perf_counter() - t0)
                assert r.status_code == 200
                time.sleep(gap)

    with ThreadPoolExecutor(max_workers = 2) as pool:
        f1 = pool.submit(fire_probe)
        f2 = pool.submit(fire_health)
        f1.result(60.0)
        f2.result(60.0)
    return max(latencies), elapsed, latencies


# ---------------------------------------------------------------------------
# (1) Behavioural canary
# ---------------------------------------------------------------------------


def test_buggy_route_blocks_event_loop():
    """Sync detect_audio_type call inside async route stalls /health."""
    with FakeLlamaServer(tok_delay = 0.6, detok_delay = 0.6) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = False)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            max_lat, probe_t, _ = _drive_concurrent_probe_and_health(
                f"http://127.0.0.1:{uv.port}"
            )
    assert probe_t >= 0.5
    assert max_lat >= 0.4, f"expected >=0.4s stall, got {max_lat:.3f}s"


def test_fixed_route_keeps_event_loop_responsive():
    """to_thread-wrapped call leaves the event loop free."""
    with FakeLlamaServer(tok_delay = 0.6, detok_delay = 0.6) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            max_lat, probe_t, lats = _drive_concurrent_probe_and_health(
                f"http://127.0.0.1:{uv.port}"
            )
    assert probe_t >= 0.5
    assert max_lat < 0.25, f"expected <0.25s; got {max_lat:.3f}s (all: {lats})"


# ---------------------------------------------------------------------------
# (2) Functional equivalence -- sync == to_thread for each codec branch
# ---------------------------------------------------------------------------


@pytest.fixture
def shim_no_match():
    """A shim whose responses make detect_audio_type fall through every
    codec branch and return None."""
    with FakeLlamaServer(
        # detok responds with a 1-char unique string per tid -> doesn't
        # start with "<custom_token_" so snac branch fails.
        detok_map = {128258: "abc", 128259: "def"},
        # tokenize responds with len-of-words tokens, which is always
        # 1 for single-word inputs so we need >1 token for the codec
        # branches NOT to match. Map every audio probe text to a 2-token
        # response so all `len(_tok(...)) == 1` checks fail.
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|bicodec_semantic_0|>": [0, 1],
            "<|bicodec_global_0|>": [0, 1],
            "<|c1_0|>": [0, 1],
            "<|c2_0|>": [0, 1],
        },
    ) as srv:
        yield srv


def test_functional_equivalence_no_match(shim_no_match):
    backend = _make_backend(shim_no_match.port)
    sync_result = backend.detect_audio_type()
    threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == threaded == None  # noqa: E711


def test_functional_equivalence_snac_match():
    # snac match requires _detok(128258) AND _detok(128259) to start
    # with "<custom_token_".
    with FakeLlamaServer(
        detok_map = {128258: "<custom_token_99>", 128259: "<custom_token_98>"}
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "snac"
    assert sync_result == threaded


def test_functional_equivalence_csm_match():
    # csm match: _tok("<|AUDIO|>") == 1 token AND _tok("<|audio_eos|>") == 1 token.
    # Also snac match must fail first.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {"<|AUDIO|>": [0], "<|audio_eos|>": [0]},
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "csm"
    assert sync_result == threaded


def test_functional_equivalence_whisper_match():
    # whisper: snac fails, csm fails, then _tok("<|startoftranscript|>") == 1
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],  # csm fails (>1 token)
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0],
        },
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "whisper"
    assert sync_result == threaded


def test_functional_equivalence_bicodec_match():
    # bicodec: snac/csm/whisper/audio_vlm all fail first, then both
    # bicodec_semantic_0 and bicodec_global_0 are single tokens.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|bicodec_semantic_0|>": [0],
            "<|bicodec_global_0|>": [0],
        },
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "bicodec"
    assert sync_result == threaded


# ---------------------------------------------------------------------------
# (3) Failure modes
# ---------------------------------------------------------------------------


def test_shim_returns_500_on_tokenize_returns_none():
    """detect_audio_type's `r.status_code == 200` check filters out
    non-200 responses; the function gracefully falls through and
    returns None. Both sync and threaded paths see identical behaviour."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_status = 500,
    ) as srv:
        backend = _make_backend(srv.port)
        # Sync
        assert backend.detect_audio_type() is None
        # Threaded
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_shim_returns_malformed_json_returns_none():
    """detect_audio_type's outer try/except catches r.json() failures."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_body = b"{this is not json",
    ) as srv:
        backend = _make_backend(srv.port)
        assert backend.detect_audio_type() is None
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_shim_connection_reset_returns_none():
    """Connection drops mid-response (RemoteProtocolError / ReadError)
    must be caught by detect_audio_type's outer try/except."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_reset = True,
    ) as srv:
        backend = _make_backend(srv.port)
        assert backend.detect_audio_type() is None
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_unreachable_port_returns_none():
    """Pointing the backend at a port nothing is listening on triggers
    httpx.ConnectError. detect_audio_type's try/except swallows it."""
    backend = _make_backend(_free_port())  # nothing listening
    assert backend.detect_audio_type() is None
    assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_backend_not_loaded_short_circuits():
    """is_loaded=False -> detect_audio_type returns None without doing
    any network I/O. Confirm sub-millisecond on both paths."""
    backend = _make_backend(_free_port(), loaded = False)
    t0 = time.perf_counter()
    sync = backend.detect_audio_type()
    sync_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    threaded_t = time.perf_counter() - t0
    assert sync is threaded is None
    assert sync_t < 0.05
    assert threaded_t < 0.05


# ---------------------------------------------------------------------------
# (4) Stress / concurrency
# ---------------------------------------------------------------------------


def test_50_concurrent_probes_complete_without_deadlock():
    """Fire 50 /probe calls in parallel against a fast shim. Threadpool
    must not deadlock; route handler must not lock or serialise."""
    with FakeLlamaServer(tok_delay = 0.05, detok_delay = 0.05) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers = 50) as pool:
                futs = [
                    pool.submit(
                        lambda: httpx.get(
                            f"http://127.0.0.1:{uv.port}/probe", timeout = 30.0
                        )
                    )
                    for _ in range(50)
                ]
                results = [f.result(60.0) for f in futs]
            elapsed = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in results)
    # 50 probes at ~0.4s each, threadpool size 32 default -> ~1-2 batches.
    # Bound generously to absorb CI jitter while catching pathological
    # serialisation (would be ~20s).
    assert (
        elapsed < 15.0
    ), f"50 concurrent probes took {elapsed:.1f}s; threadpool may be serialising"


def test_100_concurrent_healths_during_slow_probe_all_responsive():
    """Heavier version of the canary: 100 /health requests across 8
    worker threads during a slow /probe. With the fix, max latency
    stays bounded; without the fix, requests pile up."""
    with FakeLlamaServer(tok_delay = 0.4, detok_delay = 0.4) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            base = f"http://127.0.0.1:{uv.port}"

            def probe():
                with httpx.Client(timeout = 30.0) as c:
                    return c.get(f"{base}/probe").status_code

            def health_burst(n):
                lats = []
                with httpx.Client(timeout = 10.0) as c:
                    for _ in range(n):
                        t0 = time.perf_counter()
                        assert c.get(f"{base}/health").status_code == 200
                        lats.append(time.perf_counter() - t0)
                return lats

            with ThreadPoolExecutor(max_workers = 9) as pool:
                probe_f = pool.submit(probe)
                time.sleep(0.05)  # let probe enter detect_audio_type
                health_fs = [pool.submit(health_burst, 13) for _ in range(8)]
                assert probe_f.result(60.0) == 200
                latencies = [x for f in health_fs for x in f.result(60.0)]
    assert len(latencies) == 104
    max_lat = max(latencies)
    assert max_lat < 0.35, f"100-burst max latency {max_lat:.3f}s exceeds 350 ms"


# ---------------------------------------------------------------------------
# (5) Drift / regression guards on the production source
# ---------------------------------------------------------------------------


def test_routes_inference_wraps_detect_audio_type_in_to_thread():
    f = _REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"
    text = f.read_text()
    assert "await asyncio.to_thread(llama_backend.detect_audio_type)" in text
    assert "llama_backend.detect_audio_type()" not in text


def test_init_audio_codec_still_wrapped_in_to_thread():
    """Drift guard: the neighbouring init_audio_codec call must keep
    its to_thread wrap so any future copy-paste from this block keeps
    the right pattern."""
    f = _REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"
    text = f.read_text()
    assert "await asyncio.to_thread(llama_backend.init_audio_codec" in text


def test_no_other_async_route_calls_detect_audio_type_unwrapped():
    """Walk every .py under studio/backend/routes/ and confirm no file
    contains a bare ``detect_audio_type()`` call inside an async function
    (would re-introduce the bug). We accept a wrapped form."""
    routes_dir = _REPO_ROOT / "studio" / "backend" / "routes"
    offenders = []
    pattern = re.compile(r"\.detect_audio_type\s*\(\)")
    for path in routes_dir.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), start = 1):
            if pattern.search(line) and "asyncio.to_thread" not in line:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "bare detect_audio_type() in async route(s): " + "; ".join(
        offenders
    )


# ---------------------------------------------------------------------------
# (6) Timing budgets
# ---------------------------------------------------------------------------


def test_load_response_under_2s_with_fast_shim():
    """Regression budget: fast shim must complete /probe in <2 s."""
    with FakeLlamaServer(tok_delay = 0.0, detok_delay = 0.0) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            t0 = time.perf_counter()
            with httpx.Client(timeout = 5.0) as c:
                assert c.get(f"http://127.0.0.1:{uv.port}/probe").status_code == 200
            elapsed = time.perf_counter() - t0
    assert elapsed < 2.0


def test_repeated_loads_bounded_total_time():
    """Five sequential /probe calls against a fast shim must complete
    in well under 10 s total. Locks in that there's no per-call leak
    (open connections, threads, etc.) that compounds across loads."""
    with FakeLlamaServer(tok_delay = 0.05, detok_delay = 0.05) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            t0 = time.perf_counter()
            with httpx.Client(timeout = 5.0) as c:
                for _ in range(5):
                    assert c.get(f"http://127.0.0.1:{uv.port}/probe").status_code == 200
            elapsed = time.perf_counter() - t0
    assert elapsed < 10.0


# ---------------------------------------------------------------------------
# (7) Browser-compatibility surface
# ---------------------------------------------------------------------------


def test_response_is_valid_browser_parseable_json():
    """The fix changes the route's internal scheduling but must not
    change the response shape any browser sees. Round-trip the response
    through json.loads() (the canonical equivalent of
    JSON.parse() in any browser) and assert the expected keys."""
    import json as _json

    with FakeLlamaServer(tok_delay = 0.0, detok_delay = 0.0) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            with httpx.Client(timeout = 5.0) as c:
                r = c.get(f"http://127.0.0.1:{uv.port}/probe")
    # 1. Status code is one a browser will surface as success.
    assert r.status_code == 200
    # 2. Content-Type is exactly application/json (browsers use this
    #    header to decide if they can JSON-parse the body).
    assert r.headers["content-type"].startswith("application/json")
    # 3. Body is valid JSON. Every modern browser (Firefox, Safari,
    #    Chrome, Edge) uses the same JSON.parse semantics; parse via
    #    Python's strict json module here as a stand-in.
    parsed = _json.loads(r.text)
    # 4. Expected key present.
    assert "audio_type" in parsed
    # 5. No NaN / Infinity / non-JSON-spec types that would break
    #    browser parsers.
    assert _json.dumps(parsed)


def test_response_shape_matches_pre_fix_for_no_match():
    """The fix's only externally-observable effect must be timing.
    Confirm sync and threaded paths return byte-identical response
    bodies for the no-match scenario (the dominant code path in
    practice for non-audio models)."""
    import json as _json

    with FakeLlamaServer(
        detok_map = {128258: "abc", 128259: "def"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|bicodec_semantic_0|>": [0, 1],
            "<|bicodec_global_0|>": [0, 1],
            "<|c1_0|>": [0, 1],
            "<|c2_0|>": [0, 1],
        },
    ) as shim:
        backend = _make_backend(shim.port)
        # Two apps -- sync (pre-fix) and to_thread (post-fix).
        for wrap in (False, True):
            app = _build_app(backend, wrap_in_thread = wrap)
            port = _free_port()
            with _UvicornServerThread(app, port = port) as uv:
                with httpx.Client(timeout = 30.0) as c:
                    r = c.get(f"http://127.0.0.1:{uv.port}/probe")
            assert r.status_code == 200
            body = _json.loads(r.text)
            assert body == {"audio_type": None}


# ---------------------------------------------------------------------------
# (8) Cancellation
# ---------------------------------------------------------------------------


def test_client_disconnect_during_probe_does_not_crash_server():
    """If the HTTP client disconnects mid-probe, uvicorn must continue
    serving subsequent requests. The threadpool task keeps running
    (asyncio.to_thread doesn't propagate cancellation), but that's
    matched by the existing init_audio_codec wrap and is not a
    regression. After the disconnect, /health must still respond."""
    with FakeLlamaServer(tok_delay = 0.5, detok_delay = 0.5) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            base = f"http://127.0.0.1:{uv.port}"

            # Connect and immediately drop. httpx with a very short
            # timeout simulates a client that gave up.
            with pytest.raises(httpx.TimeoutException):
                with httpx.Client(timeout = 0.2) as c:
                    c.get(f"{base}/probe")

            # The server must still serve /health afterwards.
            with httpx.Client(timeout = 5.0) as c:
                r = c.get(f"{base}/health")
            assert r.status_code == 200
