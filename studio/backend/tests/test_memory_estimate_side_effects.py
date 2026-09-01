# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Side-effect contracts for POST /api/inference/estimate-memory.

``test_memory_estimate.py`` guards the arithmetic; this file guards the promises the
route makes about what it does NOT do -- "no model is loaded, no device is touched,
nothing is downloaded" -- behaviourally rather than by reading the source, because the
panel fires this endpoint on every settings change and a regression here is a Hub round
trip, a disk write, or a reaped llama-server behind a slider drag.

Four properties, one section each:

* the on-disk gate (``_estimate_target_is_on_this_disk``) is fail-OPEN on every error
  path, so this pins which hosts fall through it into the network;
* ``_probe_backend``'s ``except TypeError`` must not turn a fault raised INSIDE a
  constructor into a silently process-reaping backend;
* the blocking capability probe belongs off the event loop, where
  ``_effective_default_slots`` already puts it;
* both TTL caches are shared mutable module state reached from real threads.

No GPU, no network, no model load: every GGUF here is a synthetic header on tmp_path.
"""

import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

# Stub heavy / unavailable deps before importing the module under test.
# Copied verbatim from tests/test_memory_estimate.py -- same reasons.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog. Carries get_logger because this stub is process-wide: whichever test
# module is imported first wins the setdefault, and utils/prebuilt/freshness_flow
# calls structlog.get_logger at import time. A bare module here fails that import
# for every later module on a runner without the real package.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx -- only stub when the real library is missing. Unconditional stubbing
# shadows HTTPError/Response that huggingface_hub.errors imports at load time,
# silently breaking the transformers introspection tier.
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    _httpx_stub.Timeout = _FakeTimeout
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import routes.inference as ri  # noqa: E402
from models.inference import EstimateMemoryRequest  # noqa: E402

# Reuse the GGUF blob builder rather than copying it: these tests are only worth
# anything if the bytes they parse are the bytes the real parser was written for.
# Loaded by path because `tests` is not importable as a package name from every
# runner layout.
import importlib.util as _ilu  # noqa: E402

_kv_spec = _ilu.spec_from_file_location(
    "_kv_cache_estimation_for_estimate_side_effects",
    Path(__file__).resolve().parent / "test_kv_cache_estimation.py",
)
_kv_mod = _ilu.module_from_spec(_kv_spec)
_kv_spec.loader.exec_module(_kv_mod)
_make_gguf_bytes = _kv_mod._make_gguf_bytes


@pytest.fixture(autouse = True)
def _side_effect_caches_are_clean():
    """Both module caches are TTL'd, not per-request, so they leak across tests."""
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()
    yield
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()


_SIDE_EFFECT_GQA_FIELDS = {
    "context_length": 8192,
    "block_count": 12,
    "attention.head_count": 8,
    "attention.head_count_kv": 4,
    "attention.key_length": 64,
    "attention.value_length": 64,
    "embedding_length": 512,
}


@pytest.fixture
def side_effect_gguf(tmp_path) -> str:
    """A pure-GQA header: standard branch, and the SWA resolver never fires."""
    arch = "qwen3"
    kv = {"general.architecture": arch}
    kv.update({f"{arch}.{k}": v for k, v in _SIDE_EFFECT_GQA_FIELDS.items()})
    path = tmp_path / "side_effects.gguf"
    path.write_bytes(_make_gguf_bytes(arch, kv))
    return str(path)


def _run_estimate(fastapi_request = None, **kwargs):
    """Call the route directly, bypassing the auth dependency."""
    return asyncio.run(
        ri.estimate_memory(
            EstimateMemoryRequest(**kwargs),
            fastapi_request = fastapi_request,
            current_subject = "test",
        )
    )


def _request_carrying_slots(slots: int):
    """A stand-in for the FastAPI request carrying the server's slot default."""
    return SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = slots)))


def _priced_locally(monkeypatch, gguf_path: str):
    """Pin the route onto a local header so only the property under test moves."""
    config = SimpleNamespace(
        identifier = "local/model",
        gguf_file = gguf_path,
        is_gguf = True,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
    )
    monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)
    monkeypatch.setattr(ri, "_gguf_resident_file_gb", lambda cfg, **kw: 2.0)
    return config


def _pin_on_disk(monkeypatch):
    """Answer the on-disk gate positively, however this revision asks it.

    The gate is consulted through a bool wrapper and, for the narrower question of
    whether a resolution may go online, through a tri-state. Both are pinned so these
    tests measure the property they name rather than the gate, and so the same file
    runs against a revision that has only the first.
    """
    monkeypatch.setattr(ri, "_estimate_target_is_on_this_disk", lambda _id: True)
    monkeypatch.setattr(
        ri,
        "_estimate_disk_residency",
        lambda _id: getattr(ri, "_ESTIMATE_DISK_PRESENT", "present"),
        raising = False,
    )


def _hub_offline_now() -> bool:
    """Whether the in-process HF offline switch is currently thrown.

    ``force_hf_offline`` flips ``huggingface_hub.constants.HF_HUB_OFFLINE`` rather than
    only the env var, because the env var is read once at import. So the constant is
    the honest observable for "would this call have gone to the network".
    """
    import huggingface_hub.constants as _hf_constants
    return bool(_hf_constants.HF_HUB_OFFLINE)


# D6. The on-disk gate and the network


class TestTheOnDiskGateAndTheNetwork:
    """The gate is fail-OPEN, and what it opens onto is a Hub round trip."""

    def test_a_host_with_no_establishable_cache_root_yields_no_roots(self, monkeypatch):
        # The premise the rest of this class rests on: `roots == []` is a real host
        # state, not a hypothetical. Both tiers are driven to fail the way they fail in
        # the field -- tier 1's import reaches through hub.utils.paths into the logging
        # stack (a lean interpreter without structlog raises exactly here), and tier 2
        # keeps only roots that `is_dir()`, which drops a path that does not exist.
        def _tier_one_import_fails():
            raise ModuleNotFoundError("No module named 'structlog'")

        monkeypatch.setattr(
            ri, "_estimate_hf_cache_roots", ri._estimate_hf_cache_roots, raising = False
        )
        import hub.utils.hf_cache_state as _state
        import utils.hf_cache_settings as _settings

        monkeypatch.setattr(_state, "hf_cache_roots", _tier_one_import_fails)
        monkeypatch.setattr(_settings, "known_hf_hub_caches", lambda: [])
        import huggingface_hub.constants as _hf_constants

        monkeypatch.setattr(_hf_constants, "HF_HUB_CACHE", "/nonexistent/hf/hub")

        assert ri._estimate_hf_cache_roots() == []

    def test_no_cache_root_means_the_gate_cannot_refuse_anything(self, monkeypatch):
        # Fail-open, stated as behaviour. Every error path in the gate returns True,
        # and `roots == []` is the one that is live in the field.
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", list)
        assert ri._estimate_target_is_on_this_disk("org/definitely-not-cached") is True

    def test_a_gate_that_could_not_be_answered_never_resolves_online(self, monkeypatch):
        # THE claim under test: "nothing is downloaded", which the gate keeps true by
        # stopping from_identifier before it walks to the Hub.
        #
        # But the gate fails OPEN, and it falls through to more than a local resolution:
        # _cached_estimate_config tries offline first and then RETRIES ONLINE. On a host
        # where no cache root can be established, an uncached remote id therefore gets
        # the full identification -- model_info attempts plus an hf_hub_download of
        # config.json writing blob, ref, snapshot and lock -- for a request whose answer
        # is "not on this disk".
        #
        # The observable is the offline switch, not a socket: force_hf_offline flips
        # huggingface_hub's own constant, so a resolution seeing it False was allowed to
        # dial out.
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", list)
        offline_at_each_call = []

        def _fake_from_identifier(**kwargs):
            offline_at_each_call.append(_hub_offline_now())
            return None

        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(_fake_from_identifier))

        resp = _run_estimate(model_path = "org/definitely-not-cached")
        assert resp.available is False
        assert offline_at_each_call, "the resolution never ran at all"
        assert False not in offline_at_each_call, (
            "the on-disk gate could not be answered and the resolution was allowed "
            "online anyway: that is a Hub round trip and an HF-cache write behind a "
            f"slider, offline flag per attempt was {offline_at_each_call}"
        )

    def test_the_fail_open_path_dials_no_socket_and_writes_no_cache_file(self, monkeypatch):
        # The end-to-end form of the same claim, driven through the real
        # ModelConfig.from_identifier with the session socket guard counting. The guard
        # is reused rather than HF_HUB_OFFLINE being set, so the ONLINE branch is the
        # one under test -- setting the env var would select the branch that trivially
        # passes.
        import socket

        attempts = []
        guarded_connect = socket.socket.connect
        guarded_connect_ex = socket.socket.connect_ex
        guarded_resolve = socket.getaddrinfo

        def _record_connect(self, address, *a, **kw):
            attempts.append(("connect", address))
            return guarded_connect(self, address, *a, **kw)

        def _record_connect_ex(self, address, *a, **kw):
            attempts.append(("connect_ex", address))
            return guarded_connect_ex(self, address, *a, **kw)

        def _record_resolve(host, port, *a, **kw):
            attempts.append(("getaddrinfo", host))
            return guarded_resolve(host, port, *a, **kw)

        monkeypatch.setattr(socket.socket, "connect", _record_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", _record_connect_ex)
        monkeypatch.setattr(socket, "getaddrinfo", _record_resolve)

        # A host where the gate cannot answer.
        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", list)

        from utils.hf_cache_settings import get_hf_cache_paths

        hub_cache = Path(get_hf_cache_paths().hub_cache)

        def _snapshot():
            if not hub_cache.is_dir():
                return []
            return sorted(str(p.relative_to(hub_cache)) for p in hub_cache.rglob("*"))

        before = _snapshot()
        resp = _run_estimate(model_path = "org/definitely-not-cached-side-effects")
        assert resp.available is False

        after = _snapshot()
        assert after == before, (
            "the estimate wrote into the HF cache: " f"{sorted(set(after) - set(before))[:12]}"
        )
        _LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        outbound = [
            a
            for a in attempts
            if not (isinstance(a[1], str) and a[1] in _LOCAL)
            and not (isinstance(a[1], tuple) and a[1] and str(a[1][0]) in _LOCAL)
        ]
        assert (
            outbound == []
        ), f"the estimate tried to leave the machine {len(outbound)}x: {outbound[:5]}"

    def test_an_unreadable_cache_root_answers_not_downloaded(self, tmp_path, monkeypatch):
        # Recorded because the gate's docstring promises the opposite: "an unreadable
        # cache root must not turn into a blanket not_downloaded". It does.
        # _iter_hf_cache_snapshots swallows every OSError, PermissionError included,
        # and reports the repo as absent -- so the gate's own except never fires and
        # the answer is a refusal, not a fall-through. Safer than the docstring says,
        # but the docstring is what a reader will believe.
        if sys.platform == "win32":
            pytest.skip("POSIX permission bits")
        import os

        if os.geteuid() == 0:
            pytest.skip("root reads through any mode")

        root = tmp_path / "locked"
        (root / "models--org--Model-GGUF" / "snapshots" / "rev1").mkdir(parents = True)
        root.chmod(0o000)
        try:
            monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [root])
            answered = ri._estimate_target_is_on_this_disk("org/Model-GGUF")
        finally:
            root.chmod(0o755)
        assert (
            answered is False
        ), "behaviour changed to match the docstring; update the docstring too"

    def test_passing_the_gate_does_not_mean_the_resolution_can_stay_local(
        self, tmp_path, monkeypatch
    ):
        # The asymmetry that survives every fix above. The gate scans EVERY cache root
        # Studio knows; from_identifier's offline readers take only the CONFIGURED one.
        # So a repo sitting in the legacy or a historical root passes the gate, the
        # offline resolve then finds nothing where it looks, and the online retry runs
        # with the gate having said yes. "On this disk" does not imply "resolvable
        # without the network", and only the first of those is what the gate checks.
        from utils.models.model_config import _iter_hf_cache_snapshots

        other_root = tmp_path / "legacy"
        snap = other_root / "models--org--Model-GGUF" / "snapshots" / "rev1"
        snap.mkdir(parents = True)
        (snap / "model.gguf").write_bytes(b"\0" * 8)

        monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [other_root])
        assert ri._estimate_target_is_on_this_disk("org/Model-GGUF") is True
        # ...while the reader the resolution actually uses, on its default root, does
        # not see it.
        assert list(_iter_hf_cache_snapshots("org/Model-GGUF")) == []

        offline_at_each_call = []

        def _fake_from_identifier(**kwargs):
            offline_at_each_call.append(_hub_offline_now())
            return None

        monkeypatch.setattr(ri.ModelConfig, "from_identifier", staticmethod(_fake_from_identifier))
        _run_estimate(model_path = "org/Model-GGUF")
        assert offline_at_each_call, "the resolution never ran"
        # Deliberately NOT asserted as a fix: a repo that is genuinely present but
        # whose metadata lives elsewhere is the one case the online retry exists for.
        # This pins the shape so the trade-off is visible if it is ever revisited.
        assert offline_at_each_call[0] is True, "the first attempt must still be offline"


# D4. The inert-probe fallback


class TestInertProbeFallback:
    """``_probe_backend`` must not answer a fault with a process-reaping backend."""

    def test_a_fault_inside_the_constructor_is_not_answered_by_reaping(self, monkeypatch):
        # The fallback is for stand-ins that PREDATE the keyword -- two in
        # test_chat_load_during_training.py are plain classes taking no arguments, so
        # the keyword is a signature TypeError and the bare constructor is correct.
        #
        # A TypeError raised from INSIDE __init__ lands in the same except, and the bare
        # constructor it falls back to runs _kill_orphaned_servers, which walks /proc and
        # SIGNALS the llama-servers it recognises, then registers an atexit handler
        # holding the instance forever. A fault would silently become "reap the user's
        # running server", on a route the panel fires on every change.
        constructions = []
        reaps = []

        class _FaultyBackend:
            def __init__(self, *, manages_processes: bool = True):
                constructions.append(manages_processes)
                if not manages_processes:
                    # NOT a signature rejection: the keyword bound fine. This stands
                    # for any TypeError from the constructor body.
                    raise TypeError("'NoneType' object is not subscriptable")
                reaps.append("_kill_orphaned_servers")

        monkeypatch.setattr(ri, "LlamaCppBackend", _FaultyBackend)

        raised = None
        try:
            ri._probe_backend()
        except TypeError as exc:
            raised = exc

        assert reaps == [], (
            "a TypeError from inside __init__ was answered by constructing the "
            "process-reaping backend; an estimate must never be able to kill a server"
        )
        assert constructions == [False], (
            "the constructor ran twice for one probe: the fallback must only fire when "
            f"the keyword itself is unsupported, saw {constructions}"
        )
        assert raised is not None, "a fault inside the constructor must surface, not be masked"

    def test_a_stand_in_that_predates_the_keyword_still_gets_the_bare_constructor(
        self, monkeypatch
    ):
        # The compatibility the narrowed except must not cost. Modelled on the real
        # stand-ins: plain classes, no __init__ at all.
        class _PlainStandIn:
            is_diffusion = False
            _architecture = None

        monkeypatch.setattr(ri, "LlamaCppBackend", _PlainStandIn)
        assert isinstance(ri._probe_backend(), _PlainStandIn)

        class _NoKeywordStandIn:
            def __init__(self):
                self.built = True

        monkeypatch.setattr(ri, "LlamaCppBackend", _NoKeywordStandIn)
        assert ri._probe_backend().built is True

    def test_the_real_constructor_cannot_raise_the_typeerror_the_except_catches(self):
        # The honest scope of the fix. Under `manages_processes = False` the real
        # __init__ body is attribute assignment and threading primitives only: every
        # call it makes -- _kill_orphaned_servers, atexit.register, time.monotonic --
        # sits inside the `if manages_processes:` branch that is not taken. So the
        # broad except was unreachable for the real class, and the narrowing matters
        # for substituted and subclassed backends, which is the population it serves.
        import ast
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        tree = ast.parse(inspect.getsource(inspect.getmodule(LlamaCppBackend)))
        init = next(
            f
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "LlamaCppBackend"
            for f in node.body
            if isinstance(f, ast.FunctionDef) and f.name == "__init__"
        )
        guarded = {
            getattr(n, "lineno", None)
            for st in init.body
            if isinstance(st, ast.If)
            for n in ast.walk(st)
        }
        unguarded_calls = [
            ast.unparse(n)
            for n in ast.walk(init)
            if isinstance(n, ast.Call) and n.lineno not in guarded
        ]
        assert set(unguarded_calls) <= {
            "threading.Lock()",
            "threading.RLock()",
            "threading.Event()",
        }, (
            "the inert branch of LlamaCppBackend.__init__ grew a call that could "
            f"raise: {sorted(set(unguarded_calls))}"
        )


# D5. The blocking capability probe


class TestSlotResolutionStaysOffTheEventLoop:
    """``_effective_parallel_slots`` shells out to ``llama-server --help``."""

    def test_the_capability_probe_never_runs_on_the_event_loop_thread(
        self, monkeypatch, side_effect_gguf
    ):
        # _effective_parallel_slots asks the binary about --kv-unified, and
        # _find_llama_server_binary walks nine layouts per call before the capability
        # cache is consulted, retrying a denied one five times at 0.2s; on a cold cache
        # the probe is `llama-server --help` with a ten second timeout.
        #
        # The sibling already knows: _effective_default_slots wraps the same helper in
        # asyncio.to_thread because inline it "stalled every other request on the first
        # open of the panel after an update". This route's loop streams chat tokens.
        #
        # Asserted by thread identity, not a stopwatch: same property, cannot flake
        # under a loaded xdist runner, and it names the thread the work belongs on.
        _priced_locally(monkeypatch, side_effect_gguf)

        probe_threads = []

        def _caps(*a, **kw):
            probe_threads.append(threading.get_ident())
            return {"found": True, "supports_kv_unified": True}

        monkeypatch.setattr(ri.LlamaCppBackend, "probe_server_capabilities", staticmethod(_caps))

        async def _drive():
            loop_thread = threading.get_ident()
            resp = await ri.estimate_memory(
                EstimateMemoryRequest(model_path = side_effect_gguf, n_ctx = 4096),
                fastapi_request = _request_carrying_slots(4),
                current_subject = "test",
            )
            return loop_thread, resp

        loop_thread, resp = asyncio.run(_drive())
        assert resp.available is True
        assert probe_threads, "the slot clamp never asked the binary; the probe moved"
        assert loop_thread not in probe_threads, (
            "the capability probe ran on the event loop thread; on a cold cache that is "
            "`llama-server --help` with a ten second timeout, in front of chat streaming"
        )

    def test_the_slot_default_is_still_read_from_the_app_state(self, monkeypatch, side_effect_gguf):
        # Moving the probe off the loop must not take the settings read with it: the
        # published default lives on the FastAPI app state, which belongs to the loop
        # side, and pricing 1 there underestimates the KV cache and the slot-scaled
        # compute buffers for the default configuration.
        _priced_locally(monkeypatch, side_effect_gguf)
        monkeypatch.setattr(
            ri.LlamaCppBackend,
            "probe_server_capabilities",
            staticmethod(lambda *a, **kw: {"found": True, "supports_kv_unified": True}),
        )
        resp = _run_estimate(
            fastapi_request = _request_carrying_slots(4),
            model_path = side_effect_gguf,
            n_ctx = 4096,
        )
        assert resp.n_parallel == 4, (
            "the server's published slot default was lost; blank Parallel Slots must "
            "price the 4 slots a standard launch serves, not 1"
        )

    def test_a_single_slot_never_reaches_the_probe(self, monkeypatch):
        # One slot cannot be clamped below one, so the binary is not asked at all.
        # Pins the cheap path so the fix cannot start paying for a probe nobody needs.
        def boom(*a, **kw):
            raise AssertionError("a single slot must not probe the binary")

        monkeypatch.setattr(ri.LlamaCppBackend, "probe_server_capabilities", staticmethod(boom))
        assert ri._effective_parallel_slots(1) == 1
        assert ri._effective_parallel_slots(0) == 1


# D7. The FastAPI request parameter


class TestTheRequestParameterAnnotation:
    """``fastapi_request: Request = None`` is the form that works, not a slip."""

    def test_the_route_registers_and_receives_a_real_request(self):
        # Through the app the default is never used: FastAPI special-cases a bare
        # `Request` annotation and always injects the live object. The None default is
        # what lets a unit test await the coroutine directly.
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.post("/probe")
        async def _probe(request: EstimateMemoryRequest, fastapi_request: Request = None):
            return {"injected": type(fastapi_request).__name__}

        with TestClient(app) as client:
            body = client.post("/probe", json = {"model_path": "x"}).json()
        assert body["injected"] == "Request"

    def test_optional_request_would_break_route_registration(self):
        # The obvious "correction" is not one. FastAPI recognises the Request type by
        # the bare annotation; Optional[Request] falls through to pydantic field
        # creation and raises at decoration time, so the module would not import.
        from typing import Optional

        import fastapi
        from fastapi import FastAPI, Request

        app = FastAPI()
        with pytest.raises(fastapi.exceptions.FastAPIError):

            @app.post("/broken")
            async def _broken(
                request: EstimateMemoryRequest, fastapi_request: Optional[Request] = None
            ):
                return {}

    def test_the_route_signature_matches_the_convention_in_this_module(self):
        import inspect

        from fastapi import Request

        param = inspect.signature(ri.estimate_memory).parameters["fastapi_request"]
        assert param.annotation is Request
        assert param.default is None


# Contract: cache eviction under concurrency


class TestEstimateCachesUnderConcurrency:
    """Both caches are module state reached from ``asyncio.to_thread`` workers."""

    def test_evicting_and_inserting_from_many_threads_never_raises(self, monkeypatch):
        # min() walks the dict through a Python key function the interpreter is free to
        # switch out of: a concurrent pop makes that walk raise KeyError, a concurrent
        # insert RuntimeError ("dictionary changed size during iteration"). Nothing
        # between here and the to_thread body catches either, so it surfaces as a 500
        # on a slider drag. This is the behavioural form of that guarantee -- the
        # source-reading assertion in test_memory_estimate.py cannot see a torn walk.
        threads = 64
        keys = ri._ESTIMATE_CONFIG_CACHE_MAX * 4  # well past the eviction threshold
        _pin_on_disk(monkeypatch)

        class _FakeModelConfig:
            @staticmethod
            def from_identifier(*, model_id, **kw):
                return SimpleNamespace(identifier = model_id, is_gguf = True)

        monkeypatch.setattr(ri, "ModelConfig", _FakeModelConfig)

        start = threading.Barrier(threads)
        errors = []
        sizes = []

        def _worker(n: int):
            start.wait()
            try:
                for i in range(40):
                    ri._cached_estimate_config(f"org/model-{(n + i) % keys}", None, None, False)
                    sizes.append(len(ri._estimate_config_cache))
            except Exception as exc:  # noqa: BLE001 - the point is that none escape
                errors.append(exc)

        workers = [threading.Thread(target = _worker, args = (n,)) for n in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert errors == [], f"concurrent eviction raised {errors[:3]}"
        # The lock covers evict-and-insert, not check-then-insert across threads, so
        # the ceiling is MAX + (concurrent inserters). Bounded is the contract; exact
        # is not, and asserting exact would be asserting a race.
        assert max(sizes) <= ri._ESTIMATE_CONFIG_CACHE_MAX + threads
        assert len(ri._estimate_config_cache) <= ri._ESTIMATE_CONFIG_CACHE_MAX + threads

    def test_the_files_cache_evicts_under_the_same_pressure(self, monkeypatch):
        threads = 64
        keys = ri._ESTIMATE_FILES_CACHE_MAX * 4
        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", lambda *a, **kw: 3.0)
        monkeypatch.setattr(ri, "_remote_gguf_compute_reserve_gb", lambda **kw: 0.5)

        start = threading.Barrier(threads)
        errors = []
        sizes = []

        def _worker(n: int):
            start.wait()
            try:
                for i in range(40):
                    ri._gguf_resident_file_gb(
                        SimpleNamespace(
                            identifier = f"org/files-{(n + i) % keys}",
                            gguf_file = None,
                            gguf_variant = None,
                        ),
                    )
                    sizes.append(len(ri._estimate_files_cache))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        workers = [threading.Thread(target = _worker, args = (n,)) for n in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert errors == [], f"concurrent files-cache eviction raised {errors[:3]}"
        assert max(sizes) <= ri._ESTIMATE_FILES_CACHE_MAX + threads

    def test_a_cold_key_is_resolved_once_per_caller_not_once(self, monkeypatch):
        # Documented, not celebrated. There is no in-flight dedup, so a thundering herd
        # on one cold key runs the expensive resolution once per caller. Harmless for a
        # panel that prices one model at a time; the assertion exists so that if a
        # single-flight is ever added, this test is the thing that has to change.
        threads = 32
        _pin_on_disk(monkeypatch)
        resolutions = []
        lock = threading.Lock()
        start = threading.Barrier(threads)
        # Inside the resolution, so the count is decided by the code and not by the
        # scheduler: every caller that gets here must arrive before any of them may
        # leave. If a single-flight is ever added, fewer than `threads` arrive and this
        # breaks rather than passing by luck.
        inside = threading.Barrier(threads, timeout = 30)

        class _CountingModelConfig:
            @staticmethod
            def from_identifier(*, model_id, **kw):
                with lock:
                    resolutions.append(model_id)
                inside.wait()
                return SimpleNamespace(identifier = model_id, is_gguf = True)

        monkeypatch.setattr(ri, "ModelConfig", _CountingModelConfig)

        errors = []

        def _worker():
            start.wait()
            try:
                ri._cached_estimate_config("org/one-cold-key", None, None, False)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        workers = [threading.Thread(target = _worker) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert errors == []
        assert (
            len(resolutions) == threads
        ), "the stampede shape changed; if a single-flight was added, say so here"
        assert len(ri._estimate_config_cache) == 1

    def test_an_entry_is_not_born_already_expired(self, monkeypatch):
        # The TTL stamp is the observable. Taken BEFORE the resolution, an entry that
        # took longer than the TTL to produce is inserted already stale, so the very
        # next request re-resolves and the cache stops existing on exactly the slow
        # models it was added for.
        clock = {"t": 1000.0}
        _pin_on_disk(monkeypatch)
        monkeypatch.setattr(
            ri,
            "time",
            SimpleNamespace(monotonic = lambda: clock["t"], time = time.time),
        )

        resolutions = []
        slow = ri._ESTIMATE_CONFIG_TTL_SECONDS + 10.0

        class _SlowModelConfig:
            @staticmethod
            def from_identifier(*, model_id, **kw):
                resolutions.append(model_id)
                clock["t"] += slow  # the resolution really did take this long
                return SimpleNamespace(identifier = model_id, is_gguf = True)

        monkeypatch.setattr(ri, "ModelConfig", _SlowModelConfig)

        ri._cached_estimate_config("org/slow", None, None, False)
        assert len(resolutions) == 1
        # No clock advance at all between the two calls: the second request is
        # immediate, and must be a hit.
        ri._cached_estimate_config("org/slow", None, None, False)
        assert len(resolutions) == 1, (
            "the entry was born expired: its TTL stamp was taken before a "
            f"{slow:.0f}s resolution, so an immediate repeat re-resolved"
        )

    def test_the_files_entry_is_not_born_already_expired_either(self, monkeypatch):
        # Same stamp, same cache, other half. A multi-shard repo whose header walk and
        # file stats outlast the TTL must not be inserted pre-expired, or the slider
        # re-walks it on every tick.
        clock = {"t": 5000.0}
        monkeypatch.setattr(
            ri, "time", SimpleNamespace(monotonic = lambda: clock["t"], time = time.time)
        )
        walks = []
        slow = ri._ESTIMATE_FILES_TTL_SECONDS + 10.0

        def _slow_required(cfg, **kw):
            walks.append(1)
            clock["t"] += slow
            return 3.0

        monkeypatch.setattr(ri, "_estimate_gguf_required_gb", _slow_required)
        monkeypatch.setattr(ri, "_remote_gguf_compute_reserve_gb", lambda **kw: 0.5)

        config = SimpleNamespace(identifier = "org/many-shards", gguf_file = None, gguf_variant = None)
        ri._gguf_resident_file_gb(config)
        assert len(walks) == 1
        ri._gguf_resident_file_gb(config)
        assert len(walks) == 1, (
            "the files entry was born expired: its TTL stamp was taken before a "
            f"{slow:.0f}s walk"
        )


# Contract: token and subject isolation


class TestTokenAndSubjectIsolation:
    """A gated repo resolves per credential; entries must not cross."""

    def test_two_tokens_never_receive_each_others_config(self, monkeypatch):
        _pin_on_disk(monkeypatch)

        class _PerTokenModelConfig:
            @staticmethod
            def from_identifier(
                *,
                model_id,
                hf_token = None,
                **kw,
            ):
                if hf_token is None:
                    raise PermissionError("gated repo needs a token")
                return SimpleNamespace(identifier = model_id, is_gguf = True, resolved_with = hf_token)

        monkeypatch.setattr(ri, "ModelConfig", _PerTokenModelConfig)

        for _round in range(3):  # interleaved, not one subject then the other
            alice = ri._cached_estimate_config("org/gated", None, "token-alice", False)
            bob = ri._cached_estimate_config("org/gated", None, "token-bob", False)
            assert alice.resolved_with == "token-alice"
            assert bob.resolved_with == "token-bob"

        # And the tokenless caller gets neither -- it gets the failure it earned.
        assert ri._cached_estimate_config("org/gated", None, None, False) is None

    def test_the_token_is_fingerprinted_not_stored_in_the_key(self):
        # A token is a credential; it does not belong in a dict key verbatim, because
        # module state is dumped by every debugger and every heap snapshot.
        secret = "hf_thisIsASecretToken"
        fingerprint = ri._estimate_token_fingerprint(secret)
        assert secret not in fingerprint
        assert len(fingerprint) == 16
        assert ri._estimate_token_fingerprint(secret) == fingerprint
        assert ri._estimate_token_fingerprint("hf_other") != fingerprint
        assert ri._estimate_token_fingerprint(None) == ""
        assert ri._estimate_token_fingerprint("") == ""

    def test_the_subject_is_absent_from_the_key_and_the_token_is_what_isolates(self, monkeypatch):
        # Recorded deliberately. current_subject is NOT part of the cache key: the
        # credential is, and two subjects presenting the same token are entitled to the
        # same answer. What this pins is that nothing else about a subject leaks in --
        # so if per-subject state is ever added to the resolution, the key must grow
        # with it and this test is where that shows up.
        _pin_on_disk(monkeypatch)
        monkeypatch.setattr(
            ri,
            "ModelConfig",
            SimpleNamespace(
                from_identifier = staticmethod(
                    lambda *, model_id, **kw: SimpleNamespace(identifier = model_id, is_gguf = True)
                )
            ),
        )
        ri._cached_estimate_config("org/model", "Q4_K_M", "tok", True)
        (key,) = list(ri._estimate_config_cache)
        assert key == ("org/model", "Q4_K_M", True, ri._estimate_token_fingerprint("tok"))

    def test_the_native_grant_is_keyed_as_a_flag_not_as_an_identity(self, monkeypatch):
        # The grant changes the resolution: it passes _native_drafter_accept into
        # from_identifier. So it has to be in the key, and it is -- but as a boolean.
        # Two DIFFERENT subjects each holding a valid lease share one entry. That is
        # sound only because the lease is verified before the key is built, never
        # after; this test is the record of that reasoning.
        _pin_on_disk(monkeypatch)
        seen = []

        class _GrantAwareModelConfig:
            @staticmethod
            def from_identifier(
                *,
                model_id,
                drafter_accept = None,
                **kw,
            ):
                seen.append(drafter_accept is not None)
                return SimpleNamespace(identifier = model_id, is_gguf = True)

        monkeypatch.setattr(ri, "ModelConfig", _GrantAwareModelConfig)

        ri._cached_estimate_config("org/model", None, "tok", False)
        ri._cached_estimate_config("org/model", None, "tok", True)
        assert seen == [False, True], "the grant flag must reach the resolution"
        assert len(ri._estimate_config_cache) == 2, "a grant-backed resolve must not "
        "overwrite the ungranted one"


# Contract: no GPU is touched


class TestNoDeviceIsTouched:
    """ "no device is touched" -- the third clause of the route's docstring."""

    def test_a_full_request_allocates_nothing_and_launches_nothing(
        self, monkeypatch, side_effect_gguf
    ):
        # The route reaches _guard_device_count -> _tensor_split_possible ->
        # LlamaCppBackend._effective_gpu_count, which imports torch and asks CUDA how
        # many devices it can see. That is driver ENUMERATION, and it is as far as this
        # is allowed to go: no context creation, no property read, no free-memory
        # probe, no nvidia-smi, no llama-server. Each of those is a real cost on a box
        # where a training run owns the cards, and this endpoint fires on every
        # settings change.
        import subprocess

        import torch

        touched = []

        def _forbid(name):
            def _hit(*a, **kw):
                touched.append(name)
                raise AssertionError(f"the estimate touched the GPU via {name}")

            return _hit

        # Enumeration: permitted, but recorded so the boundary is visible.
        enumerated = []
        real_is_available = torch.cuda.is_available
        monkeypatch.setattr(
            torch.cuda,
            "is_available",
            lambda: (enumerated.append("is_available"), real_is_available())[1],
        )
        real_device_count = torch.cuda.device_count
        monkeypatch.setattr(
            torch.cuda,
            "device_count",
            lambda: (enumerated.append("device_count"), real_device_count())[1],
        )

        # Everything past enumeration: forbidden.
        for attr in (
            "init",
            "set_device",
            "mem_get_info",
            "get_device_properties",
            "synchronize",
            "empty_cache",
        ):
            if hasattr(torch.cuda, attr):
                monkeypatch.setattr(torch.cuda, attr, _forbid(f"torch.cuda.{attr}"))
        monkeypatch.setattr(subprocess, "run", _forbid("subprocess.run"))
        monkeypatch.setattr(subprocess, "Popen", _forbid("subprocess.Popen"))
        monkeypatch.setattr(subprocess, "check_output", _forbid("subprocess.check_output"))

        _priced_locally(monkeypatch, side_effect_gguf)
        resp = _run_estimate(
            fastapi_request = _request_carrying_slots(1),
            model_path = side_effect_gguf,
            n_ctx = 8192,
        )
        assert resp.available is True
        assert touched == [], f"the estimate touched the GPU: {touched}"
        # Recorded, not forbidden, and this is the honest edge of the claim: the
        # device-count path really does ask the CUDA driver how many cards are
        # visible: measured on this box, is_available() and device_count() both fire.
        # No context is created and no property is read, but "no device is touched" is
        # a slight overstatement of that, and this is where it shows. Subset, not
        # equality: a CPU-only runner answers is_available() False and never reaches
        # device_count, and a Vulkan build short-circuits before either.
        assert set(enumerated) <= {"is_available", "device_count"}, enumerated
