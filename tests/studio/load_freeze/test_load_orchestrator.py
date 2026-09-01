"""Simulation suite for the #5642 fix (sync detect_audio_type blocking the event loop)."""

from __future__ import annotations

import asyncio
import importlib.util
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
# structlog is a hard studio.txt requirement imported only lazily, so a bare setdefault would shadow the real package.
# Stub only a genuinely absent one (check sys.modules first: find_spec() raises ValueError on another module's bare
# stub), then backfill get_logger.
_structlog = sys.modules.get("structlog")
if _structlog is None and importlib.util.find_spec("structlog") is None:
    _structlog = sys.modules.setdefault("structlog", types.ModuleType("structlog"))
if _structlog is not None and not hasattr(_structlog, "get_logger"):
    _structlog.get_logger = lambda *args, **kwargs: _logging.getLogger(
        args[0] if args else "structlog"
    )

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

from llama_server_shim import FakeLlamaServer  # noqa: E402


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
        cfg = uvicorn.Config(app, host = host, port = port, log_level = "warning", access_log = False)
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

    @app.get("/loop-thread")
    async def loop_thread():
        # An async route body runs on the event loop, so this is the loop's thread id.
        return {"ident": threading.get_ident()}

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


# Used only by the canary below, which asserts a LOWER bound: that the pre-#5642 route does hold /health.
# Measured with the shim's 0.6 + 0.6 second delays, the blocking route holds it for 1.72s and does so every time (1.719,
# 1.729, 1.721, 1.719, 1.727 over five runs), so contention can only push that number further above the bound, never
_MAX_HEALTH_LATENCY_SEC = 0.25

# satisfied in milliseconds or never satisfied at all, so all it decides is how long a
# Neither a latency budget nor a performance claim.
_DEADLOCK_GUARD_SEC = 30.0


class _GatedProbe:
    """A slow synchronous detect_audio_type whose duration the test controls.

    It announces that it has started (``entered``) and then parks on ``released``
    rather than sleeping. While it is parked the route is in exactly the state the
    old wall-clock bound was trying to sample -- a blocking call in flight -- except
    that the state now persists until the test ends it, so nothing has to be timed.

    It also records the thread it ran on, which is the property itself: the call
    belongs on a worker thread, not on the event loop's.
    """

    def __init__(self, result = "snac") -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.result = result
        self.calls = 0
        self.thread_ident: int | None = None

    def __call__(self):
        self.calls += 1
        self.thread_ident = threading.get_ident()
        self.entered.set()
        # Deliberately the LONGEST wait in the file.
        if not self.released.wait(_DEADLOCK_GUARD_SEC * 3):
            raise AssertionError(
                f"the gated probe was never released within {_DEADLOCK_GUARD_SEC * 3}s; "
                "the test body failed before it got that far"
            )
        return self.result

    def release(self) -> None:
        self.released.set()


def _gated_app(gate: _GatedProbe):
    """The fixed route shape, with the blocking call replaced by the gate."""
    backend = _make_backend(_free_port())
    backend.detect_audio_type = gate
    return _build_app(backend, wrap_in_thread = True)


def _loop_thread_ident(base_url: str) -> int:
    with httpx.Client(timeout = _DEADLOCK_GUARD_SEC) as c:
        return int(c.get(f"{base_url}/loop-thread").json()["ident"])


def _fire_probe(base_url: str) -> dict:
    with httpx.Client(timeout = _DEADLOCK_GUARD_SEC) as c:
        r = c.get(f"{base_url}/probe")
    assert r.status_code == 200, f"/probe returned {r.status_code}"
    return r.json()


def _health_burst(
    base_url: str,
    n_per_worker: int,
    *,
    workers: int = 1,
) -> list[int]:
    """Fire ``workers * n_per_worker`` /health requests and return their status codes.

    A blocked event loop does not show up here as a slow-but-finished request, it shows
    up as one that never answers, so a read timeout is reported with the request index
    and the measured wait rather than being allowed to surface as a bare httpx error.
    """
    codes: list[int] = []
    lock = threading.Lock()

    def burst() -> None:
        with httpx.Client(timeout = _DEADLOCK_GUARD_SEC) as c:
            for i in range(n_per_worker):
                t0 = time.perf_counter()
                try:
                    r = c.get(f"{base_url}/health")
                except httpx.TimeoutException as exc:
                    raise AssertionError(
                        f"/health #{i + 1} of {n_per_worker} went unanswered for "
                        f"{time.perf_counter() - t0:.1f}s while a synchronous "
                        f"detect_audio_type was in flight, so the event loop is blocked "
                        f"on that call ({exc!r})"
                    ) from exc
                with lock:
                    codes.append(r.status_code)

    with ThreadPoolExecutor(max_workers = workers) as pool:
        for f in [pool.submit(burst) for _ in range(workers)]:
            f.result()
    return codes


def test_buggy_route_blocks_event_loop():
    """Sync detect_audio_type call inside async route stalls /health."""
    with FakeLlamaServer(tok_delay = 0.6, detok_delay = 0.6) as shim:
        backend = _make_backend(shim.port)
        app = _build_app(backend, wrap_in_thread = False)
        port = _free_port()
        with _UvicornServerThread(app, port = port) as uv:
            max_lat, probe_t, _ = _drive_concurrent_probe_and_health(f"http://127.0.0.1:{uv.port}")
    assert probe_t >= 0.5
    assert max_lat >= _MAX_HEALTH_LATENCY_SEC, f"expected a stalled loop, got {max_lat:.3f}s"


def test_fixed_route_keeps_event_loop_responsive():
    """The to_thread-wrapped call leaves the event loop free to serve other routes.

    This used to be twelve /health latencies compared against 250 ms, retried up to
    three times. The 250 ms was a proxy: what the fix buys is that the blocking call
    runs off the event loop thread, and a descheduled runner thread can falsify the
    proxy on its own. It did, on a single 0.261s sample among eleven 0.0013s ones,
    with the code entirely correct.

    Nothing here is timed. The blocking call is held open, twelve /health requests are
    answered while it is held, and only then is it released, so the ordering is the
    evidence. On the pre-#5642 shape the blocked loop cannot answer /health, the
    release never comes, and the run deadlocks rather than merely running late -- a
    faster or slower machine does not change that, so this cannot flake into a pass
    either.
    """
    gate = _GatedProbe()
    with _UvicornServerThread(_gated_app(gate), port = _free_port()) as uv:
        base = f"http://127.0.0.1:{uv.port}"
        loop_ident = _loop_thread_ident(base)
        with ThreadPoolExecutor(max_workers = 1) as pool:
            probe_f = pool.submit(_fire_probe, base)
            try:
                assert gate.entered.wait(_DEADLOCK_GUARD_SEC), (
                    f"/probe did not reach detect_audio_type within "
                    f"{_DEADLOCK_GUARD_SEC}s, so the request never got that far"
                )
                codes = _health_burst(base, 12)
                assert codes == [200] * 12, f"/health returned {codes}"
                # Nothing has released the call, so all twelve were answered with a synchronous detect_audio_type still
                assert not gate.released.is_set()
                assert not probe_f.done(), "/probe returned before the gate was released"
            finally:
                gate.release()
            assert probe_f.result(_DEADLOCK_GUARD_SEC) == {"audio_type": "snac"}
    assert gate.calls == 1, f"detect_audio_type ran {gate.calls} times, expected 1"
    assert gate.thread_ident is not None
    assert gate.thread_ident != loop_ident, (
        "detect_audio_type ran on the event loop thread itself, so the asyncio.to_thread "
        "wrapper is gone and #5642 is back"
    )


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
    with FakeLlamaServer(
        detok_map = {128258: "<custom_token_99>", 128259: "<custom_token_98>"}
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "snac"
    assert sync_result == threaded


def test_functional_equivalence_csm_match():
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
    # snac: both _detok(128258) and _detok(128259) start with "<custom_token_".
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
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
    # audio_vlm:
    # csm: snac fails, then both <|AUDIO|> and <|audio_eos|> are 1 token.
    # whisper: snac/csm fail, then <|startoftranscript|> is 1 token.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],
            "<audio_soft_token>": [0, 1],
            "<|audio|>": [0],  # ... Gemma 4 arm matches (#6000)
        },
    ) as srv:
        backend = _make_backend(srv.port)
        sync_result = backend.detect_audio_type()
        threaded = asyncio.run(asyncio.to_thread(backend.detect_audio_type))
    assert sync_result == "audio_vlm"
    assert sync_result == threaded


def test_functional_equivalence_bicodec_match():
    # audio_vlm: snac/csm/whisper fail, then the Gemma 4 <|audio|> arm (#6000) tokenises to 1 token while
    # <audio_soft_token> stays 2 to isolate it.
    with FakeLlamaServer(
        detok_map = {128258: "non-snac", 128259: "non-snac"},
        tok_response_map = {
            "<|AUDIO|>": [0, 1],
            "<|audio_eos|>": [0, 1],
            "<|startoftranscript|>": [0, 1],  # whisper fails Gemma 3n arm fails ...
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


def test_shim_returns_500_on_tokenize_returns_none():
    """Non-200 responses fall through to None on both sync and threaded paths."""
    # bicodec: all prior branches fail, then bicodec_semantic_0/global_0 are 1 token.
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
    backend = _make_backend(_free_port())
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
                        lambda: httpx.get(f"http://127.0.0.1:{uv.port}/probe", timeout = 30.0)
                    )
                    for _ in range(50)
                ]
                results = [f.result(60.0) for f in futs]
            elapsed = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in results)
    assert (
        elapsed < 15.0
    ), f"50 concurrent probes took {elapsed:.1f}s; threadpool may be serialising"


def test_100_concurrent_healths_during_slow_probe_all_responsive():
    """104 /health across 8 connections are all answered while a slow /probe is in flight.

    The old assertion was that the worst of those 104 latencies stayed under 350 ms.
    That is a claim about the machine as much as about the route, and it is not the
    thing the fix guarantees: what matters is that none of the 104 SERIALISE behind
    the probe. Reaching for it through a threshold also made the test worse the more
    concurrency it added, since a wider burst is a bigger sample of the worst case.

    So the burst is now required to finish IN FULL before the probe is allowed to
    return at all. It also no longer waits 0.05s hoping the probe got into
    detect_audio_type first; it waits on the call announcing that it did.
    """
    gate = _GatedProbe()
    with _UvicornServerThread(_gated_app(gate), port = _free_port()) as uv:
        base = f"http://127.0.0.1:{uv.port}"
        loop_ident = _loop_thread_ident(base)
        with ThreadPoolExecutor(max_workers = 1) as pool:
            probe_f = pool.submit(_fire_probe, base)
            try:
                assert gate.entered.wait(_DEADLOCK_GUARD_SEC), (
                    f"/probe did not reach detect_audio_type within "
                    f"{_DEADLOCK_GUARD_SEC}s, so the request never got that far"
                )
                codes = _health_burst(base, 13, workers = 8)
                assert len(codes) == 104, f"collected {len(codes)} responses, expected 104"
                assert set(codes) == {200}, f"/health returned {sorted(set(codes))}"
                assert not gate.released.is_set()
                assert not probe_f.done(), "/probe returned before the gate was released"
            finally:
                gate.release()
            assert probe_f.result(_DEADLOCK_GUARD_SEC) == {"audio_type": "snac"}
    assert gate.calls == 1, f"detect_audio_type ran {gate.calls} times, expected 1"
    assert gate.thread_ident != loop_ident, (
        "detect_audio_type ran on the event loop thread itself, so the asyncio.to_thread "
        "wrapper is gone and #5642 is back"
    )


# (5) Drift / regression guards on the production source
def test_load_model_caches_audio_type_inside_serial_load_lock():
    """Audio-type detection must run inside load_model under _serial_load_lock,
    else a concurrent /load can replace the backend mid-probe (review on #5669)."""
    f = _REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"
    text = f.read_text(encoding = "utf-8")
    # Generous bound absorbs CI jitter but still catches serialisation (~20s).
    assert (
        "with self._serial_load_lock" in text
    ), "LlamaCppBackend.load_model must hold self._serial_load_lock"
    # Either call shape satisfies the guard;
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
    text = f.read_text(encoding = "utf-8")
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
    # Matches both llama_backend.
    # the model_config free function helper is excluded below.
    pattern = re.compile(r"\b\w+\.detect_audio_type\s*\(")
    for path in routes_dir.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding = "utf-8").splitlines(), start = 1):
            m = pattern.search(line)
            if not m:
                continue
            # Only the LlamaCppBackend instance call is an offender.
            if "llama_backend.detect_audio_type" not in line:
                continue
            if "asyncio.to_thread" in line:
                # Wrapped sync call is acceptable (not preferred);
                continue
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "routes/*.py contains llama_backend.detect_audio_type() calls; "
        "the call should live inside load_model now: " + "; ".join(offenders)
    )


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
