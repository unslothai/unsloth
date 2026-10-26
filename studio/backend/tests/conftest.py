# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Puts the backend root on sys.path; studio_server uses UNSLOTH_E2E_BASE_URL or a managed server."""

# Must run before torch is imported; see tests/_shared/compile_cache_isolation.py.
import importlib.util as _ilu  # noqa: E402
import pathlib as _pathlib  # noqa: E402

_iso = _pathlib.Path(__file__).resolve()
for _up in _iso.parents:
    _candidate = _up / "tests" / "_shared" / "compile_cache_isolation.py"
    if _candidate.is_file():
        _spec = _ilu.spec_from_file_location("_unsloth_compile_cache_isolation", _candidate)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        break

import contextlib
import errno
import itertools
import os
import shutil
import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# Import the real loggers package first: test stubs without __path__ break
# 'from loggers.media_progress import' when pytest runs from the repo root.
try:
    import loggers  # noqa: E402
except ImportError:
    pass
else:
    # __path__ is what separates the real package from a stub; the use also feeds verify_import_hoist.py.
    assert hasattr(loggers, "__path__"), (
        f"the 'loggers' slot holds a non-package ({loggers!r}); a ModuleType stub from some "
        "test module got there first, so `from loggers.media_progress import ...` will fail"
    )

# Module scope: test modules import from tests/_shared at collection.
for _up in Path(__file__).resolve().parents:
    _repo_shared = _up / "tests" / "_shared"
    if (_repo_shared / "growth.py").is_file():
        if str(_repo_shared) not in sys.path:
            sys.path.insert(0, str(_repo_shared))
        break

# Let the diffusion patch backend lazily import unsloth_zoo on a CPU-only test host: unsloth_zoo runs accelerator
# detection at import and raises without a GPU unless this is set. setdefault so an explicit override wins.
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
# unsloth_zoo refuses to import without this; mirrors run.py and main.py.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
# Avoid real pip installs for attention backends; setdefault so an override wins.
os.environ.setdefault("UNSLOTH_DIFFUSION_ATTENTION_INSTALL", "0")
os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", "1")
# Tests that hide nvidia-smi must not find real GPUs via NVML.
os.environ.setdefault("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
# A fake CPU host must not take the Vulkan route through this machine's real GPUs.
os.environ.setdefault("UNSLOTH_VULKAN_DEVICE_PROBE", "0")
# settled_snapshot_device_memory spaces its retried VRAM reads a real second apart so a
# transient tenant on a live card has time to clear. Under test the snapshots are stubs
# whose answers do not change with time, so the wait buys nothing and the max() over the
# reads -- which is what the retry is actually for -- is unaffected. Measured on the
# backend suite: 142s of test_diffusion_backend.py's 328s went here, in tests reaching it
# through _plan_memory, which has no way to pass delay_s. setdefault, so a test that wants
# the production spacing can still set it.
os.environ.setdefault("UNSLOTH_SETTLE_DELAY_S", "0")


@pytest.fixture(scope = "session")
def _studio_home_root(tmp_path_factory):
    """Created once per session: mktemp rescans basetemp on every call, so a per-test call is quadratic."""
    return tmp_path_factory.mktemp("studio_homes")


_studio_home_counter = itertools.count()


@pytest.fixture(scope = "session")
def _skills_home_root(tmp_path_factory):
    # One mktemp per session; per-test mktemp is quadratic (see _studio_home_root).
    return tmp_path_factory.mktemp("skills_homes")


_skills_home_counter = itertools.count()


@pytest.fixture(autouse = True)
def _no_vae_structural_verdict_cache(monkeypatch):
    """Structural VAE fusion verdicts persist under Studio's cache root; tests must not read or write the real one
    (the cache tests turn it on against a temp dir)."""
    monkeypatch.setenv("UNSLOTH_VAE_FUSED_STRUCTURAL_CACHE", "0")


@pytest.fixture(autouse = True)
def _zoo_without_gdn_decode_kernel(monkeypatch):
    # Whether Qwen3.5 / Qwen3-Next ask for causal-conv1d depends on the installed unsloth_zoo; pin the
    # older-zoo answer so results do not follow the host. Tests of the newer zoo patch it back.
    ssm_runtime = sys.modules.get("utils.ssm_runtime")
    if ssm_runtime is None:
        try:
            from utils import ssm_runtime
        except Exception:
            return
    monkeypatch.setattr(ssm_runtime, "_zoo_gdn_decode_kernel_present", lambda: False, raising = False)


@pytest.fixture(autouse = True)
def _sd_cpp_windows_rocm_ignores_the_host_dlls(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SD_CPP_WINDOWS_ROCM", "1")


@pytest.fixture(autouse = True)
def _no_real_mxc_drive_aliases(monkeypatch):
    # Windows hosts would map real drive letters; MXC tests opt back in.
    monkeypatch.setenv("UNSLOTH_MXC_DRIVE_ALIAS", "0")


@pytest.fixture(autouse = True)
def _forget_hub_unreachable_verdict():
    # Per-process Hub verdict must not leak between tests.
    def _forget():
        reach = sys.modules.get("core.inference.hub_reachability")
        if reach is not None:
            reach._UNREACHABLE.clear()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _forget_mcp_discovery_state():
    # Discovered tools and failed-probe cool-offs are keyed by server id, and most tests reuse "s1": a cool-off
    # one test records made a later test on the worker skip discovery for its own "s1".
    def _forget():
        client = sys.modules.get("core.inference.mcp_client")
        if client is not None:
            client._tool_cache.clear()
            client._probe_cooloff_until.clear()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _forget_load_budget_snapshots():
    def _forget():
        for name, module in list(sys.modules.items()):
            if name.endswith("core.inference.diffusion_memory"):
                getattr(module, "_LOAD_BUDGET_FRACTIONS", {}).clear()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _forget_mxc_isolation_settings():
    # The setting is cached for a second across tests that each get their own Studio home.
    def _forget():
        settings = sys.modules.get("utils.mxc_isolation_settings")
        if settings is not None:
            settings.forget_cached_setting()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _forget_embedding_device_setting():
    # Same one-store-per-test reason as above.
    def _forget():
        settings = sys.modules.get("utils.embedding_model_settings")
        if settings is not None:
            settings.forget_cached_embedding_device()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _no_background_sandbox_probes(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_SANDBOX_WARMUP", "1")

    def _forget():
        os_sandbox = sys.modules.get("core.inference.os_sandbox")
        if os_sandbox is not None:
            os_sandbox.forget_tool_isolation()

    _forget()
    yield
    _forget()


@pytest.fixture(autouse = True)
def _no_restricted_region_defaults(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MIRROR_FALLBACK", "0")


@pytest.fixture(autouse = True)
def _isolate_agent_skills(_skills_home_root, monkeypatch):
    # The developer's own skills must not leak into tool-selection tests.
    from core.inference import skills as _skills

    home = _skills_home_root / f"h{next(_skills_home_counter)}"
    home.mkdir()
    # Owner home under tmp, bundled root empty; managed-account roots stay for the account matrix.
    monkeypatch.delenv("UNSLOTH_STUDIO_SKILLS_HOME", raising = False)
    monkeypatch.setattr(_skills, "_owner_home", lambda: home)
    monkeypatch.setattr(_skills, "_BUNDLED_ROOT", ("bundled", home / "bundled-absent"))
    try:
        from routes import inference as _inference_routes
    except Exception:
        return
    monkeypatch.setattr(_inference_routes, "_AGENT_SKILLS_CACHE", {})


@pytest.fixture(autouse = True)
def _contain_installer_venv_root(tmp_path_factory, monkeypatch):
    """test_torchao_select runs install_python_stack in process, so it would rewrite this venv's
    manifest."""
    for _up in _iso.parents:
        _shared = _up / "tests" / "_shared"
        if (_shared / "installer_venv_root.py").is_file():
            if str(_shared) not in sys.path:
                sys.path.insert(0, str(_shared))
            break
    else:
        return
    from installer_venv_root import contain_installer_venv_root

    contain_installer_venv_root(monkeypatch, tmp_path_factory)


# apply_hub_settings() writes these straight into os.environ, and importing the app runs it. A test
# that left ModelScope selected put every later test on its xdist worker on that source, so the
# auth-check probe in test_hub_token_caller_identity.py built huggingface.co URLs for a mirror.
_HUB_SETTINGS_ENV = (
    "UNSLOTH_STUDIO_HUB_SOURCE",
    "HF_ENDPOINT",
    "HF_DATASETS_SERVER",
    "NO_PROXY",
    "no_proxy",
)


@pytest.fixture(autouse = True)
def _restore_hub_settings_env():
    saved = {name: os.environ.get(name) for name in _HUB_SETTINGS_ENV}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse = True)
def _reset_gpu_query_cache():
    # Only if already imported: importing utils.hardware would change import-order tests.
    def _reset():
        gpu_query = sys.modules.get("utils.hardware.gpu_query")
        # A background probe may still be importing it.
        reset = getattr(gpu_query, "reset", None)
        if reset is not None:
            reset()
        hw = sys.modules.get("utils.hardware.hardware")
        if hw is not None and hasattr(hw, "_last_good_visible_info"):
            with hw._last_good_visible_lock:
                hw._last_good_visible_info.clear()
        amd = sys.modules.get("utils.hardware.amd")
        if amd is not None and hasattr(amd, "_hip_id_map_lock"):
            with amd._hip_id_map_lock:
                amd._hip_id_map_cache = None
        # The last NVML inventory is class state the launch pins the child from: a MIG
        # probe test leaving its slices there pinned later launches to MIG-dddd4444-0.
        llama = sys.modules.get("core.inference.llama_cpp")
        backend = getattr(llama, "LlamaCppBackend", None)
        if backend is not None and "_NVML_ROWS" in vars(backend):
            backend._NVML_ROWS = []

    _reset()
    yield
    _reset()


@pytest.fixture(autouse = True)
def _restore_fp32_matmul_precision():
    # torchao's default config handler sets float32 matmul precision process-wide.
    def _get():
        getter = getattr(sys.modules.get("torch"), "get_float32_matmul_precision", None)
        return getter() if getter is not None else None

    before = _get() or "highest"
    yield
    after = _get()
    if after is not None and after != before:
        sys.modules["torch"].set_float32_matmul_precision(before)


@pytest.fixture(autouse = True)
def _reset_media_import_window(monkeypatch):
    # A load path claims the window for the process; later prewarm tests would skip.
    warm = sys.modules.get("utils.torch_warmup")
    if warm is not None and hasattr(warm, "_media_import_claimed"):
        monkeypatch.setattr(warm, "_media_import_claimed", False)
        monkeypatch.setattr(warm, "_media_import_owner", None)


@pytest.fixture(autouse = True)
def _isolate_studio_home(_studio_home_root, monkeypatch):
    home = _studio_home_root / f"home-{next(_studio_home_counter)}"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("storage.", "hub.storage.")) and hasattr(module, "_schema_ready"):
            monkeypatch.setattr(module, "_schema_ready", set())


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: let this test make non-loopback connections (see _no_outbound_network)",
    )
    config.addinivalue_line(
        "markers",
        "stages_switch_waiter: this test leaves routes.inference._auto_switch_waiters populated "
        "on purpose (see the autouse fixture in test_openai_auto_switch.py)",
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


@pytest.fixture(scope = "session", autouse = True)
def _isolate_xet_health_home(tmp_path_factory):
    """Session scope: a function-scoped HF_HOME would land after studio_server snapshots os.environ."""
    from _pytest.monkeypatch import MonkeyPatch

    from huggingface_hub import constants as hf_constants

    mp = MonkeyPatch()
    mp.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path_factory.mktemp("studio_home_session")))
    # Pin hub paths before moving HF_HOME, or the E2E server gets an empty cache and token store.
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
    """Keep the persisted Xet health verdict out of the real HF home, or a stalled machine starts on
    HTTP."""
    # Load like the shim: a bare unsloth_zoo import raises on CPU-only hosts.
    from utils.hf_xet_fallback import _load_optional

    hf_xet_health = _load_optional("unsloth_zoo.hf_xet_health")
    if hf_xet_health is None:
        yield
        return
    hf_xet_health.clear_xet_health()
    yield
    hf_xet_health.clear_xet_health()


@pytest.fixture(autouse = True)
def _confine_prequant_registration_memo():
    """Restore diffusion_prequant's memo after each test: a fake torch can cache a False answer."""
    from core.inference import diffusion_prequant

    registered = diffusion_prequant._SAFE_GLOBALS_REGISTERED
    resolved = set(diffusion_prequant._RESOLVED_SAFE_GLOBALS)
    yield
    diffusion_prequant._SAFE_GLOBALS_REGISTERED = registered
    diffusion_prequant._RESOLVED_SAFE_GLOBALS.clear()
    diffusion_prequant._RESOLVED_SAFE_GLOBALS.update(resolved)


@pytest.fixture(autouse = True)
def _isolate_generation_state():
    """Reset active_generations per test: fenced accounts from one test otherwise leak into the next."""
    from state import active_generations

    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


@pytest.fixture(autouse = True)
def _forget_the_managed_provider_url_setting():
    """Forget the cached managed-provider URL setting per test, or a stale True answers the next test."""
    settings = sys.modules.get("utils.managed_provider_url_settings")
    if settings is not None:
        settings.forget_cached_setting()
    yield
    settings = sys.modules.get("utils.managed_provider_url_settings")
    if settings is not None:
        settings.forget_cached_setting()


@pytest.fixture(autouse = True)
def _forget_the_cached_owner_identity():
    """Clear process_lifetime's cached owner identity per test, or a fake pid identity leaks onward."""
    yield
    lifetime = sys.modules.get("utils.process_lifetime")
    if lifetime is not None:
        lifetime._owner_identity = None


@pytest.fixture(autouse = True)
def _isolate_wal_keepers():
    """Close the WAL keepers a test opened, so a later test's assertion on _wal_keepers sees only
    its own."""
    from storage import studio_db

    before = set(studio_db._wal_keepers)
    unsupported = set(studio_db._wal_unsupported)
    yield
    for path in set(studio_db._wal_keepers) - before:
        studio_db.close_wal_keeper_for(path)
    studio_db._wal_unsupported.intersection_update(unsupported)


@pytest.fixture(autouse = True)
def _isolate_auth_wal_keeper():
    """Close an auth.db keeper a test opened, so it never outlives that test's DB_PATH."""
    auth_storage = sys.modules.get("auth.storage")
    before = auth_storage._wal_keeper if auth_storage is not None else None
    yield
    auth_storage = sys.modules.get("auth.storage")
    if auth_storage is not None and auth_storage._wal_keeper is not before:
        auth_storage.close_wal_keeper()


@pytest.fixture(autouse = True)
def _isolate_audio_gallery(monkeypatch, tmp_path):
    """Point audio_gallery's studio_root at tmp_path so generated clips never reach the real gallery."""
    from core.inference import audio_gallery

    monkeypatch.setattr(audio_gallery, "studio_root", lambda: tmp_path)
    yield


@pytest.fixture(autouse = True)
def _no_background_model_scan(monkeypatch):
    """Patch out the /v1 admission hook's background index warm so tests never walk real HF caches."""
    import time

    from core.inference import local_model_resolver

    monkeypatch.setattr(local_model_resolver, "warm_index_soon", lambda: None)
    # Start from a built empty index, or the cold path walks caches inside the admission wait.
    monkeypatch.setattr(local_model_resolver, "_scan", (time.monotonic(), {}))
    monkeypatch.setattr(local_model_resolver, "_misses", {})


@pytest.fixture(scope = "session")
def _empty_hf_hub_cache(tmp_path_factory):
    """One empty hub-cache root for the whole session; per-test mktemp is quadratic."""
    return str(tmp_path_factory.mktemp("hf_hub_cache_empty"))


@pytest.fixture(autouse = True)
def _hf_cache_is_empty(_empty_hf_hub_cache, monkeypatch):
    """Pin HF_HUB_CACHE at the root so every cache read sees an empty hub, not just one probe."""
    from utils import hf_cache_settings

    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", _empty_hf_hub_cache)
    monkeypatch.setenv("HF_HUB_CACHE", _empty_hf_hub_cache)
    try:
        from huggingface_hub import constants
    except Exception:  # optional deps absent on some CI legs
        return
    monkeypatch.setattr(constants, "HF_HUB_CACHE", _empty_hf_hub_cache)


@pytest.fixture(autouse = True)
def _no_live_metal_wired_ceiling(monkeypatch):
    """Keep Metal context verdicts off the host's live GPU memory."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_wired_ceiling_bytes", staticmethod(lambda: 0)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_wired_headroom_bytes", staticmethod(lambda: None)
    )


@pytest.fixture(autouse = True)
def _no_leftover_generation_account(monkeypatch):
    """Reset routes.video's _generation_account per test, since no route ever clears it."""
    routes_video = sys.modules.get("routes.video")
    if routes_video is None:
        return
    monkeypatch.setattr(routes_video, "_generation_account", None, raising = False)


@pytest.fixture(autouse = True)
def _assume_bare_metal(monkeypatch):
    """Force the Metal paravirtual detector False: on a Mac or macOS runner, fixture state would
    mismatch."""
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    # The route rebinds the detector as a module global, so patching llama_cpp alone is not enough.
    try:
        from routes import inference as routes_inference
    except Exception:  # optional deps absent on some CI legs
        return
    monkeypatch.setattr(
        routes_inference, "_metal_device_is_paravirtual", lambda: False, raising = False
    )


_LOOPBACK_HOSTS = frozenset({"::1", "localhost", "localhost.localdomain", "0.0.0.0", "::", ""})

# Loopback spellings for NO_PROXY (no wildcards or empty string).
_LOOPBACK_PROXY_BYPASS = ("localhost", "localhost.localdomain", "127.0.0.1", "::1")

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def no_proxy_with_test_servers(*existing) -> str:
    """Merge every existing NO_PROXY spelling, so a configured bypass is not lost to lowercase."""
    bypass = list(_LOOPBACK_PROXY_BYPASS) + sorted(_configured_server_hosts())
    parts = [
        part.strip() for value in existing for part in (value or "").split(",") if part.strip()
    ]
    return ",".join(dict.fromkeys(parts + bypass))


# Explicitly configured external servers must stay reachable.
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


# Addresses allowed names resolved to: create_connection dials the numeric result.
_RESOLVED_SERVER_ADDRESSES: set = set()

# Lifted only by allow_outbound(); a module global since callers run before any per-test fixture.
_outbound_permitted = False


@contextlib.contextmanager
def allow_outbound():
    """Session or module fixtures must call this; an allow_network marker is applied too late for them."""
    global _outbound_permitted

    previous = _outbound_permitted
    _outbound_permitted = True
    try:
        yield
    finally:
        _outbound_permitted = previous


def _decoded_host(host):
    """Decode bytes hostnames to str first; comparing bytes to strings matches no rule."""
    if isinstance(host, (bytes, bytearray)):
        try:
            return bytes(host).decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(host, str):
        return host
    return None


def _host_is_allowed(host) -> bool:
    """Unreadable hosts are refused, since a permissive default let byte-string hosts bypass the guard."""
    if host is None:
        return True
    decoded = _decoded_host(host)
    if decoded is None:
        return False
    lowered = decoded.strip().lower()
    return (
        lowered.startswith("127.")
        or lowered in _LOOPBACK_HOSTS
        or lowered in _configured_server_hosts()
        or lowered in _RESOLVED_SERVER_ADDRESSES
    )


def _is_ip_literal(host) -> bool:
    """True when *host* is already an address, so resolving it consults no resolver."""
    import ipaddress

    decoded = _decoded_host(host)
    if decoded is None:
        return False
    try:
        ipaddress.ip_address(decoded.strip().strip("[]"))
    except ValueError:
        return False
    return True


def _is_local_endpoint(sock, address) -> bool:
    """Check the family first: AF_UNIX addresses are paths, not hosts, so address[0] matches no rule."""
    import socket as _socket

    if getattr(sock, "family", None) not in (_socket.AF_INET, _socket.AF_INET6):
        return True
    try:
        host = address[0]
    except Exception:
        return True
    return _host_is_allowed(host)


class _RealSocketCalls:
    """The unpatched socket entry points, kept so a test can hand them back out."""

    def __init__(self, connect, connect_ex, getaddrinfo):
        self.connect = connect
        self.connect_ex = connect_ex
        self.getaddrinfo = getaddrinfo


@pytest.fixture(scope = "session", autouse = True)
def _outbound_network_guard():
    """Block non-loopback sockets so a slow Hub cannot stall a test; the allow_network marker lifts it."""
    import socket

    real = _RealSocketCalls(socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
    patch = pytest.MonkeyPatch()

    def blocked_connect(self, address, *args, **kwargs):
        if _outbound_permitted or _is_local_endpoint(self, address):
            return real.connect(self, address, *args, **kwargs)
        raise OSError(
            errno.ENETUNREACH,
            f"outbound network blocked in tests (tried {address!r}); "
            f"stub the call, or mark the test with @pytest.mark.allow_network",
        )

    def blocked_connect_ex(self, address, *args, **kwargs):
        if _outbound_permitted or _is_local_endpoint(self, address):
            return real.connect_ex(self, address, *args, **kwargs)
        # Returned, not raised: connect_ex callers branch on the errno.
        return errno.ENETUNREACH

    # Refuse lookups too, or an uncached request can still stall in the host resolver.
    def guarded_getaddrinfo(host, port, *args, **kwargs):
        # Address literals use no resolver; SSRF tests resolve private literals on purpose.
        allowed_by_name = _outbound_permitted or _host_is_allowed(host)
        if not (allowed_by_name or _is_ip_literal(host)):
            raise socket.gaierror(
                socket.EAI_NONAME,
                f"name resolution blocked in tests ({host!r}); "
                f"stub the call, or mark the test with @pytest.mark.allow_network",
            )
        infos = real.getaddrinfo(host, port, *args, **kwargs)
        if allowed_by_name and not _outbound_permitted:
            # Allow the resolved address so it stays dialable; not on the literal branch.
            for info in infos:
                try:
                    _RESOLVED_SERVER_ADDRESSES.add(str(info[4][0]).lower())
                except Exception:
                    continue
        return infos

    patch.setattr(socket.socket, "connect", blocked_connect)
    patch.setattr(socket.socket, "connect_ex", blocked_connect_ex)
    patch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    # huggingface_hub backs off ~23s on refused connects; patch its clock only.
    try:
        from huggingface_hub.utils import _http as hf_http
    except Exception:
        hf_http = None
    if hf_http is not None and getattr(hf_http, "time", None) is not None:
        import time as _time
        class _NoBackoffClock:
            def __getattr__(self, name):
                return getattr(_time, name)

            @staticmethod
            def sleep(_seconds):
                return None

        patch.setattr(hf_http, "time", _NoBackoffClock())

    # A configured proxy would be dialled instead of the test server, so bypass it via NO_PROXY.
    # Bypass, not allow, so the proxy cannot carry Hub traffic. Session scope for session fixtures.
    if any(os.environ.get(name) for name in _PROXY_ENV_VARS):
        combined = no_proxy_with_test_servers(
            os.environ.get("NO_PROXY"), os.environ.get("no_proxy")
        )
        for name in ("NO_PROXY", "no_proxy"):
            patch.setenv(name, combined)

    try:
        yield real
    finally:
        patch.undo()


@pytest.fixture
def forget_resolved_servers():
    """Drop the addresses resolved so far, so a test can check something is refused."""
    return _RESOLVED_SERVER_ADDRESSES.clear


@pytest.fixture(scope = "session")
def no_proxy_bypass_value():
    """Hand out the NO_PROXY builder, which is otherwise only reachable as a fixture."""
    return no_proxy_with_test_servers


@pytest.fixture(scope = "session")
def allow_outbound_network(_outbound_network_guard):
    """Hand a fixture the context manager that lifts the guard around a real fetch."""
    return allow_outbound


@pytest.fixture(autouse = True)
def _no_outbound_network(request, monkeypatch, _outbound_network_guard):
    """Per-test: clear names the last test pointed at, and lift the guard for allow_network tests."""
    # Per test, so names a test pointed env vars at do not stay dialable.
    _RESOLVED_SERVER_ADDRESSES.clear()

    if request.node.get_closest_marker("allow_network") is not None:
        import socket

        real = _outbound_network_guard
        monkeypatch.setattr(socket.socket, "connect", real.connect)
        monkeypatch.setattr(socket.socket, "connect_ex", real.connect_ex)
        monkeypatch.setattr(socket, "getaddrinfo", real.getaddrinfo)


@pytest.fixture(autouse = True)
def _hub_reachable_without_probing(monkeypatch):
    """Seed the reachability memo as reachable: patching hf_dns_dead misses bindings already imported."""
    import time

    from utils import utils as utils_utils

    # Seed a future stamp instead of overriding _reachability_fresh, which is itself under test.
    monkeypatch.setattr(
        utils_utils, "_hf_reachability", (time.monotonic() + 10**6, False), raising = False
    )


@pytest.fixture(autouse = True)
def _no_hub_mirror_probe(monkeypatch):
    """Saving a mirror endpoint would start a background probe; test_hub_mirror_probe.py turns it back on."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HUB_PROBE", "0")


@pytest.fixture(scope = "session")
def studio_server(request):
    """Yields (base_url, api_key): UNSLOTH_E2E_BASE_URL if set, else a managed server started lazily."""
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


@pytest.fixture(scope = "session")
def linkable_temp_base(tmp_path_factory):
    """Scratch under the home dir: macOS temp paths under /private/var are denied as system dirs."""
    basetemp = tmp_path_factory.getbasetemp()
    root = Path.home() / ".unsloth-test-tmp"
    base = root / basetemp.name
    base.mkdir(parents = True, exist_ok = True)
    for stale in root.iterdir():
        if stale.name != basetemp.name and not (basetemp.parent / stale.name).exists():
            shutil.rmtree(stale, ignore_errors = True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors = True)


@pytest.fixture
def rag_home(tmp_path, monkeypatch, linkable_temp_base):
    """Give each test a fresh UNSLOTH_STUDIO_HOME and reset the lazy rag.db schema flag so it starts
    empty."""
    from hub.storage.scan_folders import is_denied_system_path
    from storage import rag_db

    root = tmp_path
    if is_denied_system_path(os.path.realpath(str(tmp_path))):
        root = linkable_temp_base / tmp_path.name
        root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(root))
    monkeypatch.setattr(rag_db, "_schema_ready", set())
    return root


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
    """Stub core.rag.embeddings with hash vectors, so tests never download a sentence-transformers model."""
    import hashlib
    import math

    from core.rag import config, embeddings

    # Pin the backend: 'auto' probes nvidia-smi for each backend.
    monkeypatch.setattr(config, "EMBED_BACKEND", "sentence-transformers")
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
    monkeypatch.setattr(embeddings, "max_tokens", lambda model_name = None: None)
    monkeypatch.setattr(
        embeddings,
        "token_counter",
        lambda model_name = None: lambda t: len(t.split()),
    )
    monkeypatch.setattr(embeddings, "warm", lambda model_name = None: None)
    return dim


@pytest.fixture
def dit_train_host(monkeypatch):
    """Pin the accelerator and bf16 probes so DiT family metadata does not vary by runner's GPU."""
    import core.training.diffusion_train_common as dtc

    monkeypatch.setattr(dtc, "dit_accelerator_missing_reason", lambda *_a, **_k: None)
    monkeypatch.setattr(dtc, "bf16_unsupported_reason", lambda *_a, **_k: None)
    return dtc


@pytest.fixture(autouse = True)
def _fresh_archive_vector_cache():
    """The archive's pre-embedded vector cache is process-wide; one test's vectors must not answer another's."""
    archive = sys.modules.get("core.rag.conversation_archive")
    if archive is not None:
        archive._vector_cache.clear()
        archive._noted_branches.clear()
    yield
    archive = sys.modules.get("core.rag.conversation_archive")
    if archive is not None:
        archive._prewarm_stop.set()
        if archive._prewarm_worker is not None:
            archive._prewarm_worker.join()
        archive._vector_cache.clear()
        archive._noted_branches.clear()


@pytest.fixture(autouse = True)
def _reset_optional_module_memo():
    """_load_optional caches failures too, so one test's fake module would answer the next question."""
    import utils.hf_xet_fallback as _shim

    _shim._reset_optional_module_cache()
    yield
    _shim._reset_optional_module_cache()


@pytest.fixture
def healthy_diffusers(monkeypatch):
    """Proxy diffusers to answer any pipeline class, since a pin-skewed import raises RuntimeError."""
    import types

    try:
        import diffusers as _real
    except Exception:  # noqa: BLE001 -- an absent diffusers is exactly what this stands in for
        _real = None

    class _AnyPipeline(types.ModuleType):
        def __getattr__(self, name):
            if _real is not None:
                try:
                    return getattr(_real, name)
                except Exception:  # noqa: BLE001 -- the lazy submodule is what may be broken
                    pass
            # Video families name a transformer Model, not a Pipeline.
            if name.endswith("Pipeline") or name.endswith("Model"):
                return object
            raise AttributeError(name)

    proxy = _AnyPipeline("diffusers")
    proxy.__version__ = str(getattr(_real, "__version__", "0.39.0"))
    for attr in ("__path__", "__file__", "__spec__", "__loader__"):
        if _real is not None and hasattr(_real, attr):
            setattr(proxy, attr, getattr(_real, attr))
    monkeypatch.setitem(sys.modules, "diffusers", proxy)


@pytest.fixture
def real_prequant_safe_globals(monkeypatch):
    """Stand in only for allowlist names that do not resolve, so torchao hosts still test real classes."""
    import core.inference.diffusion_prequant as pq

    resolver = pq._prequant_safe_globals
    resolved = {name: obj for obj, name in resolver()}
    pairs = [
        (resolved.get(f"{module}.{name}") or type(name, (), {}), f"{module}.{name}")
        for module, name in pq._PREQUANT_SAFE_GLOBALS
    ]
    monkeypatch.setattr(pq, "_prequant_safe_globals", lambda: pairs)
    # Per test: the memo is a module global.
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(pq, "_RESOLVED_SAFE_GLOBALS", set())
    return resolver


@pytest.fixture(autouse = True)
def _no_carried_over_hardware_measurements():
    """Clear the torch build and GPU inventory caches per test, since their 60 second TTL outlives a
    test."""
    from utils.hardware import hardware as _hw

    def _clear():
        # Under the locks (torch lock first, matching the refresh) so a prior test's host cannot land.
        with _hw._torch_build_snapshot_lock, _hw._physical_gpu_inventory_lock:
            _hw._torch_build_snapshot_cache = None
            _hw._physical_gpu_inventory_cache = None

    _clear()
    yield
    _clear()


@pytest.fixture(autouse = True)
def _process_shutdown_latch_is_clear():
    """Reopen the shutdown latch per test: it is sticky in production, so one test would leak it."""
    from utils import process_lifetime

    def _reopen():
        process_lifetime.begin_process_lifecycle()
        # Reset the route latch set by cancel_pending_loads, or later loads get cancelled.
        # Only if already imported, to avoid perturbing import-order tests.
        mod = sys.modules.get("routes.inference")
        if mod is not None:
            try:
                mod.begin_load_lifecycle()
            except Exception:
                pass

    _reopen()
    try:
        yield
    finally:
        _reopen()


@pytest.fixture(autouse = True)
def _no_leaked_inventory_handles():
    """Empty the per-request handle table per test: its ContextVar is never reset, so handles leak
    onward."""
    try:
        from hub.utils import host_paths
    except Exception:  # optional deps absent on some CI legs
        yield
        return
    token = host_paths._request_handles.set(None)
    try:
        yield
    finally:
        host_paths._request_handles.reset(token)


@pytest.fixture(autouse = True)
def _drop_the_settings_memo_between_tests():
    """Clear the openai_auto_switch_settings memo per test; its 2 second TTL lets values leak onward."""
    from utils import openai_auto_switch_settings as _settings

    _settings._cache.clear()
    try:
        yield
    finally:
        _settings._cache.clear()


@pytest.fixture(autouse = True)
def _drop_the_idle_reload_stash_between_tests():
    """Clear keepwarm's stash via _set_last_unloaded per test, so no later test reloads another's model."""
    from core.inference import llama_keepwarm as _keepwarm

    _keepwarm._set_last_unloaded(None)
    try:
        yield
    finally:
        _keepwarm._set_last_unloaded(None)


# Run with the NVFP4 switch on; test_nvfp4_diffusion_flag and test_build_prequant_checkpoint must stay off.
_NVFP4_ENABLED_TEST_MODULES = frozenset(
    {
        "test_dense_quant_rocm_gate_9396",
        "test_diffusion_auto_policy",
        "test_diffusion_backend",
        "test_diffusion_inference_info",
        "test_diffusion_lora",
        "test_diffusion_more_families",
        "test_diffusion_native_quant",
        "test_diffusion_pipeline_prequant",
        "test_diffusion_precision",
        "test_diffusion_prequant",
        "test_diffusion_quant_pad",
        "test_diffusion_routes",
        "test_diffusion_te_prequant",
        "test_diffusion_transformer_quant",
        "test_train_precision_scheme_contract",
        "test_video_backend",
        "test_video_families",
        "test_video_h3_te_quant",
        "test_video_prequant",
        "test_video_routes",
        "test_xformers_stub_diffusion_parity",
    }
)
_NVFP4_ENABLED_TEST_PREFIX = "test_diffusion_nvfp4_"


@pytest.fixture(autouse = True)
def _nvfp4_diffusion_enabled_for_nvfp4_tests(request, monkeypatch):
    """Switch NVFP4 on for the modules above; every other module sees the default (off)."""
    module = getattr(request, "module", None)
    name = getattr(module, "__name__", "").rsplit(".", 1)[-1]
    if name in _NVFP4_ENABLED_TEST_MODULES or name.startswith(_NVFP4_ENABLED_TEST_PREFIX):
        monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    else:
        monkeypatch.delenv("UNSLOTH_NVFP4_DIFFUSION", raising = False)
    yield


@pytest.fixture(autouse = True)
def pin_installer_torch_vendor(monkeypatch):
    """Pin the installer's torch-vendor probe so a ROCm-torch dev box answers like CI."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    for module in list(sys.modules.values()):
        # __dict__: hasattr would trip a lazy __getattr__.
        if "_rocm_torch_preferred" in (getattr(module, "__dict__", None) or {}):
            monkeypatch.setattr(module, "_installed_torch_is_rocm", lambda: None)


@pytest.fixture(autouse = True)
def _clear_github_rate_limit_lockout():
    from utils.prebuilt import freshness_flow

    freshness_flow._api_rate_limited_until = 0.0
    yield
    freshness_flow._api_rate_limited_until = 0.0


@pytest.fixture
def traced_offload_hooks(monkeypatch):
    """diffusers' own group-offload hook methods for one test (install_group_offload_hooks_eager is process-wide)."""
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    from core.inference.diffusion_memory import install_group_offload_hooks_eager

    install_group_offload_hooks_eager()
    for cls in (
        go.GroupOffloadingHook,
        go.LayerExecutionTrackerHook,
        go.LazyPrefetchGroupOffloadingHook,
    ):
        for name, fn in list(vars(cls).items()):
            orig = getattr(fn, "_unsloth_orig", None)
            if orig is not None:
                monkeypatch.setattr(cls, name, orig)


@pytest.fixture(scope = "session")
def _kernel_lock_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("kernel-locks"))


@pytest.fixture(autouse = True)
def _isolated_kernel_install_locks(monkeypatch, _kernel_lock_dir):
    # Tests never take (or create) the environment's real kernel install locks.
    monkeypatch.setenv("UNSLOTH_KERNEL_LOCK_DIR", _kernel_lock_dir)
