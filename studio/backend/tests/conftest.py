# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared pytest configuration for the backend test suite.

Puts the backend root on sys.path (mirrors app launch) and provides a hybrid
``studio_server`` session fixture for end-to-end tests with two modes:
external server (``UNSLOTH_E2E_BASE_URL``/``UNSLOTH_E2E_API_KEY``) for fast
iteration, or a fixture-managed server started/torn down per session for CI.
Model/variant for the managed mode resolve from ``--unsloth-model`` /
``--unsloth-gguf-variant``, then env vars, then ``test_studio_api.py`` defaults.
"""

import errno
import itertools
import os
import sys
from pathlib import Path

import pytest

# Add backend root to sys.path (mirrors app launch)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# Let the diffusion patch backend lazily import unsloth_zoo on a CPU-only test host: unsloth_zoo runs accelerator
# detection at import and raises without a GPU unless this is set. setdefault so an explicit override wins.
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
# The other half of the same guard: unsloth_zoo.__init__ refuses to import unless this is present, normally set by `import unsloth`. Without it
# the patch backend's only route to the helpers is that ~940 MB import, which a CPU-only host cannot complete. run.py and main.py do the same.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


@pytest.fixture(scope = "session")
def _studio_home_root(tmp_path_factory):
    """One parent directory for every per-test studio home.

    ``tmp_path_factory.mktemp`` scans the whole basetemp on every call to pick
    the next number, so calling it once per test is quadratic in the number of
    tests. Paid once per session here, the per-test cost below is a bare mkdir.
    """
    return tmp_path_factory.mktemp("studio_homes")


_studio_home_counter = itertools.count()


@pytest.fixture(autouse = True)
def _isolate_studio_home(_studio_home_root, monkeypatch):
    home = _studio_home_root / f"home-{next(_studio_home_counter)}"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("storage.", "hub.storage.")) and hasattr(module, "_schema_ready"):
            monkeypatch.setattr(module, "_schema_ready", False)


# Pytest CLI options


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: let this test make non-loopback connections (see _no_outbound_network)",
    )


def pytest_addoption(parser):
    group = parser.getgroup(
        "unsloth-e2e",
        "Unsloth Studio end-to-end test options",
    )
    group.addoption(
        "--unsloth-model",
        action = "store",
        default = None,
        help = (
            "GGUF model id used when starting a server for e2e tests. "
            "Ignored if UNSLOTH_E2E_BASE_URL is set. Overrides "
            "UNSLOTH_E2E_MODEL env var. Defaults to test_studio_api.py's "
            "DEFAULT_MODEL."
        ),
    )
    group.addoption(
        "--unsloth-gguf-variant",
        action = "store",
        default = None,
        help = (
            "GGUF variant used when starting a server for e2e tests. "
            "Ignored if UNSLOTH_E2E_BASE_URL is set. Overrides "
            "UNSLOTH_E2E_VARIANT env var. Defaults to test_studio_api.py's "
            "DEFAULT_VARIANT."
        ),
    )


# E2E server fixtures


@pytest.fixture(scope = "session", autouse = True)
def _isolate_xet_health_home(tmp_path_factory):
    """Point HF_HOME at a temp dir for the whole session, before any server is spawned.

    Session scope is load-bearing: pytest builds higher-scoped fixtures first, so setting HF_HOME
    function-scoped landed AFTER ``studio_server`` snapshotted os.environ, leaving that server
    rewriting the developer's real unsloth_xet_health.json.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from huggingface_hub import constants as hf_constants

    mp = MonkeyPatch()
    mp.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path_factory.mktemp("studio_home_session")))
    # Pin these to what the hub resolved from the REAL environment before moving HF_HOME, which
    # also defaults HF_HUB_CACHE, HF_XET_CACHE and HF_TOKEN_PATH: moving it alone would send the
    # E2E server to an empty cache and token store, so a ~1.1GB GGUF redownload inside the 120s
    # startup deadline and no credentials for a private --unsloth-model.
    mp.setenv("HF_HUB_CACHE", hf_constants.HF_HUB_CACHE)
    mp.setenv("HF_TOKEN_PATH", hf_constants.HF_TOKEN_PATH)
    xet_cache = getattr(hf_constants, "HF_XET_CACHE", None)
    if xet_cache:
        mp.setenv("HF_XET_CACHE", xet_cache)
    mp.setenv("HF_HOME", str(tmp_path_factory.mktemp("xet_health_home")))
    yield
    mp.undo()


@pytest.fixture(autouse = True)
def _isolate_xet_health_state():
    """Keep the persisted Xet health verdict out of the developer's real HF home.

    The verdict is sticky across sessions (two consecutive failures pin a machine to HTTP for 24h),
    so a machine that has genuinely stalled starts the ladder on HTTP inside a test expecting Xet.
    That reproduced: `test_shim_injects_studio_prepare_on_http_retry` saw the fallback without the
    Xet attempt it asserts. Clean CI runners hide it, developer machines do not.
    """
    # Load it the way the shim does: a bare `from unsloth_zoo import ...` raises NotImplementedError
    # on a CPU-only host (its __init__ runs accelerator detection), so a plain import would skip
    # this isolation on exactly the hosts the shim's GPU-init retry exists to support.
    from utils.hf_xet_fallback import _load_optional

    hf_xet_health = _load_optional("unsloth_zoo.hf_xet_health")
    if hf_xet_health is None:
        yield
        return
    hf_xet_health.clear_xet_health()
    yield
    hf_xet_health.clear_xet_health()


@pytest.fixture(autouse = True)
def _no_background_model_scan(monkeypatch):
    """Keep the /v1 admission hook from scanning the real HF cache during tests.

    The hook warms the local-model index on a background thread: right in a server,
    wrong here, since it walks the developer's actual caches and the I/O starves the
    loop under timing-sensitive streaming tests. Warm tests patch it back.
    """
    import time

    from core.inference import local_model_resolver

    monkeypatch.setattr(local_model_resolver, "warm_index_soon", lambda: None)
    # Start from a built, empty index: stubbing only the warm left the cold path
    # walking those caches inside the admission wait, so on a large install the
    # assertion became a 503 "still indexing". Cold-path tests reset _scan themselves;
    # _build_index is untouched so tests calling it directly still walk for real.
    monkeypatch.setattr(local_model_resolver, "_scan", (time.monotonic(), {}))


@pytest.fixture(autouse = True)
def _assume_bare_metal(monkeypatch):
    """Pin the virtualised-Metal detector off so the suite is host independent.

    The dedupe comparators consult real hardware, so on a Mac (and on GitHub's macos
    runners, which are paravirtual) every test of them would normalize the incoming
    request and mismatch fixture state that was never normalized. Tests that want the
    fallback patch it back on.
    """
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    # The route rebinds the detector as a module global (its import sits in a module-level
    # try), so patching llama_cpp alone leaves it on real hardware.
    try:
        from routes import inference as routes_inference
    except Exception:  # optional deps absent on some CI legs
        return
    monkeypatch.setattr(
        routes_inference, "_metal_device_is_paravirtual", lambda: False, raising = False
    )


_LOOPBACK_HOSTS = frozenset({"::1", "localhost", "localhost.localdomain", "0.0.0.0", "::", ""})

# Server URLs the suite is explicitly configured to talk to. Both documented external-server
# modes may name a remote host, and neither suite carries the allow_network marker, so
# blocking them would make a deliberately configured integration run unusable.
_EXTERNAL_SERVER_ENV_VARS = ("UNSLOTH_E2E_BASE_URL", "STUDIO_TEST_URL")


def _configured_server_hosts() -> frozenset:
    """Hostnames from the external-server env vars, so a configured endpoint stays dialable."""
    from urllib.parse import urlsplit

    hosts = set()
    for name in _EXTERNAL_SERVER_ENV_VARS:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            host = urlsplit(raw).hostname
        except ValueError:
            continue
        if host:
            hosts.add(host.lower())
    return frozenset(hosts)


# What those hostnames resolved to during this test. socket.create_connection looks the
# name up and then dials the numeric result, so allowing the name alone still refuses the
# connect that follows: by then the destination is an address that matches no name rule.
# Filled in at resolution time, and only for names the rules already allowed, so the
# address-literal exemption below cannot widen anything through it.
_RESOLVED_SERVER_ADDRESSES: set = set()


def _host_is_allowed(host) -> bool:
    """True for loopback and for any host the suite was explicitly pointed at."""
    if not isinstance(host, str):
        return True
    lowered = host.lower()
    return (
        lowered.startswith("127.")
        or lowered in _LOOPBACK_HOSTS
        or lowered in _configured_server_hosts()
        or lowered in _RESOLVED_SERVER_ADDRESSES
    )


def _is_ip_literal(host) -> bool:
    """True when *host* is already an address, so resolving it consults no resolver."""
    import ipaddress

    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def _is_local_endpoint(sock, address) -> bool:
    """True for anything the suite may dial: local IPC and loopback addresses.

    Family first, because AF_UNIX carries a filesystem path rather than a host and
    multiprocessing's pools connect over exactly that -- reading ``address[0]`` there
    yields a directory name that matches no host rule and blocks process pools.
    """
    import socket as _socket

    if getattr(sock, "family", None) not in (_socket.AF_INET, _socket.AF_INET6):
        return True
    try:
        host = address[0]
    except Exception:
        return True
    return _host_is_allowed(host)


@pytest.fixture(autouse = True)
def _no_outbound_network(request, monkeypatch):
    """Refuse every non-loopback connect, so no test depends on a live Hub.

    Several routes probe huggingface.co on paths these tests exercise, and every one
    of them fails open, so the traffic never showed up as a failure -- it only showed
    up as time. When the Hub answers slowly rather than refusing (a rate-limited CI
    egress IP is the usual way), huggingface_hub retries with backoff and a file that
    normally runs in six seconds sits there for minutes, until the job hits its cap
    and is killed having reported nothing.

    Blocking the socket keeps the online code path intact -- unlike HF_HUB_OFFLINE,
    which flips the branch and changes what is under test -- while making the call
    fail immediately instead of hanging. Mark a test ``allow_network`` if it genuinely
    needs to dial out; the e2e ``studio_server`` tests reach their server on loopback
    and are unaffected.
    """
    # Per test, so a name a test pointed the env vars at itself does not stay dialable
    # for the rest of the run.
    _RESOLVED_SERVER_ADDRESSES.clear()

    if request.node.get_closest_marker("allow_network") is not None:
        return

    import socket

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guard(wrapped):
        def blocked(self, address, *args, **kwargs):
            if _is_local_endpoint(self, address):
                return wrapped(self, address, *args, **kwargs)
            raise OSError(
                errno.ENETUNREACH,
                f"outbound network blocked in tests (tried {address!r}); "
                f"stub the call, or mark the test with @pytest.mark.allow_network",
            )

        return blocked

    monkeypatch.setattr(socket.socket, "connect", _guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(real_connect_ex))

    # Resolution runs before either of those: socket.create_connection, which is what the
    # Hub's HTTP stack ends up in, calls getaddrinfo first. Left live, an uncached request
    # still hits the host resolver and can stall there, so the dependency is not actually
    # gone. Refuse the lookup too, which is also where a real resolver would fail.
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        # An address literal consults no resolver, so it cannot stall and is left alone;
        # the SSRF guards resolve private literals on purpose to prove they reject them,
        # and connect() still refuses anything non-loopback afterwards.
        allowed_by_name = _host_is_allowed(host)
        if not (allowed_by_name or _is_ip_literal(host)):
            raise socket.gaierror(
                socket.EAI_NONAME,
                f"name resolution blocked in tests ({host!r}); "
                f"stub the call, or mark the test with @pytest.mark.allow_network",
            )
        infos = real_getaddrinfo(host, port, *args, **kwargs)
        if allowed_by_name:
            # Carry the permission across the lookup, so the numeric address this hands
            # back is still dialable. Deliberately not done on the literal branch: that
            # one exists so a private literal can be resolved and then refused.
            for info in infos:
                try:
                    _RESOLVED_SERVER_ADDRESSES.add(str(info[4][0]).lower())
                except Exception:
                    continue
        return infos

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    # A refused connection still looks retryable to huggingface_hub, which backs off
    # 1+2+4+8+8s over five attempts before giving up -- so blocking the socket without
    # this turns a fast failure back into a ~23s one per call. Swap the clock only
    # inside that module, since `time` is shared and real sleeps elsewhere matter.
    try:
        from huggingface_hub.utils import _http as hf_http
    except Exception:
        return
    import time as _time

    class _NoBackoffClock:
        def __getattr__(self, name):
            return getattr(_time, name)

        @staticmethod
        def sleep(_seconds):
            return None

    if getattr(hf_http, "time", None) is not None:
        monkeypatch.setattr(hf_http, "time", _NoBackoffClock())


@pytest.fixture(autouse = True)
def _hub_reachable_without_probing(monkeypatch):
    """Report the Hub as reachable without dialling it.

    The reachability probes are themselves network calls, so with outbound traffic
    blocked they would report the Hub down and send every caller into its offline
    branch -- which is a different code path from the one these tests mean to cover.
    Pinning them to "reachable" keeps the online path selected; the individual
    request that follows is still blocked, and the callers already fail open on it.
    Tests about offline behaviour patch these back.

    Seeded through the memo rather than by patching ``hf_dns_dead`` /
    ``hf_unreachable``: callers ``from utils.utils import`` those names, so patching
    the source module leaves already-imported bindings pointing at the real probes.
    Every caller reaches the memo through a function that reads the module global at
    call time, so seeding it covers them all regardless of import style.

    The verdict is also pinned fresh for the whole test. The memo expires after
    ``_HF_REACHABILITY_TTL_S`` (five seconds), so a seed stamped now would lapse in any
    test that reaches a Hub-guarded operation later than that, and the real probe would
    run into the blocked socket and select the offline branch this fixture exists to
    avoid. The stamp is dated far ahead instead, which keeps it fresh under the real
    freshness rule rather than disabling that rule for every test.
    """
    import time

    from utils import utils as utils_utils

    # Dated far ahead rather than by overriding _reachability_fresh: the freshness rule is
    # itself under test (test_verdict_expires_so_a_disconnect_is_noticed shortens the TTL and
    # asserts the verdict lapses), so it has to keep working. A future stamp keeps this seed
    # fresh under the real rule, and a test that writes its own verdict replaces it.
    monkeypatch.setattr(
        utils_utils, "_hf_reachability", (time.monotonic() + 10**6, False), raising = False
    )


@pytest.fixture(scope = "session")
def studio_server(request):
    """Yield ``(base_url, api_key)`` for e2e tests.

    Uses ``UNSLOTH_E2E_BASE_URL`` (requires ``UNSLOTH_E2E_API_KEY``) if set,
    else starts/tears down a fresh server via ``_start_server``. Session-scoped
    and lazy so the GGUF load happens at most once and only when requested.
    """
    external_url = os.environ.get("UNSLOTH_E2E_BASE_URL")
    if external_url:
        api_key = os.environ.get("UNSLOTH_E2E_API_KEY")
        if not api_key:
            pytest.skip(
                "UNSLOTH_E2E_BASE_URL is set but UNSLOTH_E2E_API_KEY is "
                "missing — tests that require auth cannot run against an "
                "external server without it.",
            )
        yield external_url, api_key
        return

    # Lazy import; pytest has already loaded test_studio_api, so this is a cache hit.
    import test_studio_api as _e2e

    model = (
        request.config.getoption("--unsloth-model")
        or os.environ.get("UNSLOTH_E2E_MODEL")
        or _e2e.DEFAULT_MODEL
    )
    variant = (
        request.config.getoption("--unsloth-gguf-variant")
        or os.environ.get("UNSLOTH_E2E_VARIANT")
        or _e2e.DEFAULT_VARIANT
    )

    proc, api_key = _e2e._start_server(model, variant)
    try:
        yield f"http://{_e2e.HOST}:{_e2e.PORT}", api_key
    finally:
        _e2e._kill_server(proc)


@pytest.fixture
def base_url(studio_server):
    """Base URL for the e2e Unsloth server (from ``studio_server``)."""
    return studio_server[0]


@pytest.fixture
def api_key(studio_server):
    """API key for the e2e Unsloth server (from ``studio_server``)."""
    return studio_server[1]


# ── RAG fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def rag_home(tmp_path, monkeypatch):
    """Isolate the RAG database under a fresh UNSLOTH_STUDIO_HOME per test.

    Points the storage root at ``tmp_path`` and resets the lazy schema flag so
    each test starts from an empty rag.db. Yields the temp home path.
    """
    from storage import rag_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(rag_db, "_schema_ready", False)
    return tmp_path


@pytest.fixture
def rag_conn(rag_home):
    """A fresh RAG connection bound to the isolated ``rag_home`` database."""
    from storage import rag_db

    conn = rag_db.get_connection()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Stub ``core.rag.embeddings`` with deterministic hash-based vectors.

    Lets store / retrieval / ingestion tests run fast without downloading a
    sentence-transformers model. Returns the fixed embedding dimension.
    """
    import hashlib
    import math

    from core.rag import embeddings

    dim = 32

    def _vec(text: str):
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [seed[i % len(seed)] / 255.0 for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def fake_encode(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        return [_vec(t) for t in texts]

    monkeypatch.setattr(embeddings, "encode", fake_encode)
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: dim)
    monkeypatch.setattr(
        embeddings,
        "token_counter",
        lambda model_name = None: lambda t: len(t.split()),
    )
    monkeypatch.setattr(embeddings, "warm", lambda model_name = None: None)
    return dim


@pytest.fixture
def dit_train_host(monkeypatch):
    """Pretend this host can train the DiT families.

    ``family_train_infos()`` and the start preflight both gate on the accelerator / bf16 probes, so
    on a GPU-less runner every DiT family reports no precision modes, ``supports_compile`` False,
    and a "needs a GPU" note that replaces any other preflight message. Tests about family metadata
    or about a different preflight pin the probes here so they assert the same thing on every host;
    the gate itself is covered by its own tests in test_diffusion_base_precision.py.
    """
    import core.training.diffusion_train_common as dtc

    monkeypatch.setattr(dtc, "dit_accelerator_missing_reason", lambda *_a, **_k: None)
    monkeypatch.setattr(dtc, "bf16_unsupported_reason", lambda *_a, **_k: None)
    return dtc


@pytest.fixture(autouse = True)
def _reset_optional_module_memo():
    """Forget the shim's memoised optional-module results between tests.

    ``_load_optional`` caches per module name including failures, so without this one test's fake
    module would answer the next test's question.
    """
    import utils.hf_xet_fallback as _shim

    _shim._reset_optional_module_cache()
    yield
    _shim._reset_optional_module_cache()
