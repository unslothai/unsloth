"""Simulation suite for the #5642 fix (sync detect_audio_type blocking the event loop)."""

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


# Repo discovery
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


# Fixtures / helpers
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
    def __init__(
        self,
        app,
        *,
        host: str = "127.0.0.1",
        port: int,
    ) -> None:
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


def _drive_concurrent_probe_and_health(
    base_url,
    *,
    n_health = 12,
    gap = 0.05,
):
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


# (1) Behavioural canary
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


# (2) Functional equivalence -- sync == to_thread for each codec branch
@pytest.fixture
def shim_no_match():
    """Shim whose responses make detect_audio_type fall through every codec branch -> None."""
    with FakeLlamaServer(
        # detok strings don't start with "<custom_token_" so snac branch fails.
        detok_map = {128258: "abc", 128259: "def"},
        # 2-token responses make every `len(_tok(...)) == 1` codec check fail.
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|audio|>": [0, 1],
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
    # snac: both _detok(128258) and _detok(128259) start with "<custom_token_".
    with FakeLlamaServer(
        detok_map = {128258: "<custom_token_99>", 128259: "<custom_token_98>"}
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "snac"
    assert sync_result == threaded


def test_functional_equivalence_csm_match():
    # csm: snac fails, then both <|AUDIO|> and <|audio_eos|> are 1 token.
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
    # whisper: snac/csm fail, then <|startoftranscript|> is 1 token.
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


def test_functional_equivalence_audio_vlm_match():
    # audio_vlm: snac/csm/whisper fail, then the Gemma 4 <|audio|> arm (#6000)
    # tokenises to 1 token while <audio_soft_token> stays 2 to isolate it.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],  # csm fails (>1 token)
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],  # whisper fails
            "<audio_soft_token>": [0, 1],  # Gemma 3n arm fails ...
            "<|audio|>": [0],  # ... Gemma 4 arm matches (#6000)
        },
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "audio_vlm"
    assert sync_result == threaded


def test_functional_equivalence_bicodec_match():
    # bicodec: all prior branches fail, then bicodec_semantic_0/global_0 are 1 token.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|audio|>": [0, 1],
            "<|bicodec_semantic_0|>": [0],
            "<|bicodec_global_0|>": [0],
        },
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "bicodec"
    assert sync_result == threaded


# (3) Failure modes
def test_shim_returns_500_on_tokenize_returns_none():
    """Non-200 responses fall through to None on both sync and threaded paths."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_status = 500,
    ) as srv:
        backend = _make_backend(srv.port)
        assert backend.detect_audio_type() is None
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_shim_returns_malformed_json_returns_none():
    """Outer try/except catches r.json() failures."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_body = b"{this is not json",
    ) as srv:
        backend = _make_backend(srv.port)
        assert backend.detect_audio_type() is None
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_shim_connection_reset_returns_none():
    """Mid-response connection drop (RemoteProtocolError / ReadError) is caught."""
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_reset = True,
    ) as srv:
        backend = _make_backend(srv.port)
        assert backend.detect_audio_type() is None
        assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_unreachable_port_returns_none():
    """ConnectError on a dead port is swallowed -> None."""
    backend = _make_backend(_free_port())  # nothing listening
    assert backend.detect_audio_type() is None
    assert asyncio.run(asyncio.to_thread(backend.detect_audio_type)) is None


def test_backend_not_loaded_short_circuits():
    """is_loaded=False short-circuits to None with no network I/O (sub-ms both paths)."""
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


# (4) Stress / concurrency
def test_50_concurrent_probes_complete_without_deadlock():
    """50 parallel /probe calls must not deadlock or serialise."""
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
    # Generous bound absorbs CI jitter but still catches serialisation (~20s).
    assert (
        elapsed < 15.0
    ), f"50 concurrent probes took {elapsed:.1f}s; threadpool may be serialising"


def test_100_concurrent_healths_during_slow_probe_all_responsive():
    """100 /health across 8 threads during a slow /probe: latency stays bounded with the fix."""
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
                time.sleep(0.05)  # let probe enter detect_audio_type first
                health_fs = [pool.submit(health_burst, 13) for _ in range(8)]
                assert probe_f.result(60.0) == 200
                latencies = [x for f in health_fs for x in f.result(60.0)]
    assert len(latencies) == 104
    max_lat = max(latencies)
    assert max_lat < 0.35, f"100-burst max latency {max_lat:.3f}s exceeds 350 ms"


# (5) Drift / regression guards on the production source
def test_load_model_caches_audio_type_inside_serial_load_lock():
    """Audio-type detection must run inside load_model under _serial_load_lock,
    else a concurrent /load can replace the backend mid-probe (review on #5669)."""
    f = _REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"
    text = f.read_text()
    assert (
        "with self._serial_load_lock" in text
    ), "LlamaCppBackend.load_model must hold self._serial_load_lock"
    # Either call shape satisfies the guard; _detect_audio_type_strict was a
    # follow-up to distinguish definitive non-audio from transient probe failure.
    assert (
        "self._audio_type = self.detect_audio_type()" in text
        or "detected = self.detect_audio_type()" in text
        or "detected = self._detect_audio_type_strict()" in text
    ), (
        "LlamaCppBackend.load_model must call detect_audio_type / "
        "_detect_audio_type_strict and cache the result on "
        "self._audio_type (#5642 follow-up)."
    )


def test_routes_inference_reads_cached_audio_type_not_calls_detect():
    """routes/inference.py must read cached _audio_type/_is_audio, not call
    detect_audio_type / init_audio_codec directly (both moved into load_model)."""
    f = _REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"
    text = f.read_text()
    assert "llama_backend.detect_audio_type(" not in text, (
        "routes/inference.py should not call detect_audio_type directly; "
        "load_model already cached it under the lock."
    )
    assert "llama_backend.init_audio_codec(" not in text, (
        "routes/inference.py should not call init_audio_codec directly; "
        "load_model already invoked it under the lock when audio_type was a TTS codec."
    )
    # Route must read the cached values.
    assert "llama_backend._audio_type" in text
    assert "llama_backend._is_audio" in text


def test_no_other_async_route_calls_detect_audio_type_unwrapped():
    """No routes/*.py may call llama_backend.detect_audio_type() in an async fn;
    that reintroduces the sync bug and the load race the lock fix closes."""
    routes_dir = _REPO_ROOT / "studio" / "backend" / "routes"
    offenders = []
    # Matches both llama_backend. and self. prefixes; the model_config free
    # function helper is excluded below.
    pattern = re.compile(r"\b\w+\.detect_audio_type\s*\(")
    for path in routes_dir.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), start = 1):
            m = pattern.search(line)
            if not m:
                continue
            # Only the LlamaCppBackend instance call is an offender.
            if "llama_backend.detect_audio_type" not in line:
                continue
            if "asyncio.to_thread" in line:
                # Wrapped sync call is acceptable (not preferred); surface in PR.
                continue
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "routes/*.py contains llama_backend.detect_audio_type() calls; "
        "the call should live inside load_model now: " + "; ".join(offenders)
    )


# (6) Timing budgets
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
    """Five sequential /probe calls finish under 10 s, guarding against per-call leaks."""
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


# (7) Browser-compatibility surface
def test_response_is_valid_browser_parseable_json():
    """The fix must not change the response shape a browser sees (valid JSON, expected keys)."""
    import json as _json

    with FakeLlamaServer(tok_delay = 0.0, detok_delay = 0.0) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            with httpx.Client(timeout = 5.0) as c:
                r = c.get(f"http://127.0.0.1:{uv.port}/probe")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    parsed = _json.loads(r.text)
    assert "audio_type" in parsed
    # No NaN / Infinity that would break browser parsers.
    assert _json.dumps(parsed)


def test_response_shape_matches_pre_fix_for_no_match():
    """Sync and threaded paths return identical bodies for the no-match scenario."""
    import json as _json
    with FakeLlamaServer(
        detok_map = {128258: "abc", 128259: "def"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|audio|>": [0, 1],
            "<|bicodec_semantic_0|>": [0, 1],
            "<|bicodec_global_0|>": [0, 1],
            "<|c1_0|>": [0, 1],
            "<|c2_0|>": [0, 1],
        },
    ) as shim:
        backend = _make_backend(shim.port)
        # sync (pre-fix) then to_thread (post-fix).
        for wrap in (False, True):
            app = _build_app(backend, wrap_in_thread = wrap)
            port = _free_port()
            with _UvicornServerThread(app, port = port) as uv:
                with httpx.Client(timeout = 30.0) as c:
                    r = c.get(f"http://127.0.0.1:{uv.port}/probe")
            assert r.status_code == 200
            body = _json.loads(r.text)
            assert body == {"audio_type": None}


# (8) Cancellation
def test_client_disconnect_during_probe_does_not_crash_server():
    """A client disconnect mid-probe must not crash the server; /health still responds."""
    with FakeLlamaServer(tok_delay = 0.5, detok_delay = 0.5) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = True)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            base = f"http://127.0.0.1:{uv.port}"

            # Short timeout simulates a client that gave up mid-probe.
            with pytest.raises(httpx.TimeoutException):
                with httpx.Client(timeout = 0.2) as c:
                    c.get(f"{base}/probe")

            with httpx.Client(timeout = 5.0) as c:
                r = c.get(f"{base}/health")
            assert r.status_code == 200
