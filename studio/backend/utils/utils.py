# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared backend utilities."""

import os
import structlog
import threading
import time
from loggers import get_logger
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import shutil
import tempfile
from utils.paths.path_utils import is_appledouble_metadata


logger = get_logger(__name__)


# An offline load must never touch the network (a DNS-dead session hangs on hub retries), so
# these read the local HF cache.

# ── Offline / HF-cache helpers ──────────────────────────────────
# An offline load must never touch the network (a DNS-dead session hangs on hub retries); these read the local HF cache.
_HF_OFFLINE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def hf_env_offline() -> bool:
    """True when HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE asks for offline mode.

    TRANSFORMERS_OFFLINE counts too (the hub reads only HF_HUB_OFFLINE), as does an open
    force_hf_offline window: hf_environment_restored_for_spawn briefly puts the user's
    values back, and an env-only check on another thread would then read "online".
    """
    if force_hf_offline_active():
        return True
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in _HF_OFFLINE_TRUE_VALUES:
            return True
    return False


def anonymous_and_offline(hf_token) -> bool:
    """The one condition under which a Hub-reaching request can only be answered by disk.

    ``token=False`` denies authentication, not the cache: offline, huggingface_hub and
    datasets both resolve a previously downloaded private repo without ever authorizing.
    A caller holding the anonymous sentinel has no network to establish access over, so
    every downstream read is a disk read it never earned.

    Guarding this at the route entry rather than at each call site is deliberate. The
    per-site version was fixed six times -- the snapshot walk, the config probes, the
    embedding marker, the GGUF listing, the preview slices, AutoConfig -- and each fix
    only moved the boundary to the next reader. This states the rule once, before any of
    them run, so a path nobody has enumerated is covered too.
    """
    from hub.utils.hf_tokens import is_anonymous
    return is_anonymous(hf_token) and hf_env_offline()


def canonical_model_repo_id(model_name: str) -> str:
    """Normalize a Hugging Face model repository ID selected in Unsloth."""
    return model_name.strip()


def hf_endpoint_url() -> str:
    """Configured hub endpoint, scheme-normalised. Mirror users point this elsewhere."""
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip() or "https://huggingface.co"
    return endpoint if "://" in endpoint else "https://" + endpoint


def hf_endpoint_host() -> str:
    """Host of the configured endpoint; probing huggingface.co would misjudge a mirror."""
    try:
        from urllib.parse import urlparse
        return urlparse(hf_endpoint_url()).hostname or "huggingface.co"
    except Exception:
        return "huggingface.co"


def _stdlib_proxy_for_url(url: str) -> Optional[str]:
    """requests' proxy selection rebuilt on the stdlib, for installs without requests.

    huggingface_hub 1.x dropped requests, so importing requests.utils raises there and we
    would report "no proxy" on a machine that has one, forcing a working proxy-only setup
    offline. getproxies covers the same sources, incl. macOS sysconf and the Windows registry.
    """
    from urllib.parse import urlparse
    from urllib.request import getproxies, proxy_bypass

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None
    try:
        if proxy_bypass(host):
            return None
    except Exception:
        pass
    proxies = {k.lower(): v for k, v in getproxies().items()}
    scheme = (parsed.scheme or "https").lower()
    # select_proxy order: scheme://host, then scheme, then the all catch-all.
    for key in (f"{scheme}://{host}", scheme, "all"):
        value = proxies.get(key)
        if value:
            return value
    return None


def hf_proxy_for_endpoint(endpoint: Optional[str] = None) -> Optional[str]:
    """Return the Hub client's proxy choice, including ALL_PROXY and NO_PROXY rules."""
    url = endpoint or hf_endpoint_url()
    try:
        from requests.utils import get_environ_proxies, select_proxy
        return select_proxy(url, get_environ_proxies(url))
    except ImportError:
        # No requests (huggingface_hub 1.x); fall back rather than go blind.
        pass
    except Exception:
        return None
    try:
        return _stdlib_proxy_for_url(url)
    except Exception:
        return None


def hf_proxy_usable_by_urllib(proxy: Optional[str]) -> bool:
    """True when urllib can route through this proxy.

    urllib speaks only http/https, so a socks5:// proxy makes urlopen raise "unknown url
    type", which reads as no egress even though the Hub client reaches the hub through it.
    """
    if not proxy:
        return True
    from urllib.parse import urlparse

    scheme = urlparse(proxy if "://" in proxy else "http://" + proxy).scheme.lower()
    return scheme in ("http", "https")


def hf_proxy_configured() -> bool:
    """True when egress goes through a proxy: it resolves the hub host, so local DNS
    proves nothing about reachability and must not declare the hub offline."""
    return hf_proxy_for_endpoint() is not None


def call_with_deadline(
    fn,
    timeout_s: float,
    *,
    name: str = "deadline-call",
):
    """Run `fn()` on a daemon thread; raise TimeoutError if it outlives `timeout_s`.

    For network work that is bounded on paper but not in practice: a connect timeout applies
    per address, so a host whose leading addresses blackhole pays it once for each. A
    timed-out worker is abandoned, not stopped, and holds the callable until the kernel gives
    up, so keep this to short work. The callable's own exception is re-raised rather than
    swallowed, which stops a deadline turning a bug into an apparent dead network.
    """
    import contextvars

    outcome: dict = {}
    # Log context is per-thread: without the copy, fn()'s own logging loses the request
    # fields it carries when the same call runs inline.
    context = contextvars.copy_context()

    def _run() -> None:
        try:
            outcome["value"] = context.run(fn)
        except BaseException as exc:  # noqa: BLE001 - re-raised below, in the caller
            outcome["error"] = exc

    t = threading.Thread(target = _run, daemon = True, name = name)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"call did not finish within {timeout_s}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def dns_host_dead(host: str, timeout: float = 2.0) -> bool:
    """True only when host definitively does not resolve. Daemon thread, so a wedged
    resolver cannot block past the deadline and socket.setdefaulttimeout is left alone.

    getaddrinfo, not gethostbyname: the latter is IPv4-only and would call an AAAA-only
    mirror or an IPv6 literal dead.

    A missed deadline is inconclusive, not dead. Slow-but-working DNS (cold cache, DNSSEC,
    a fresh VPN) resolves past 2s, and this shortcut skips the fail-open probe, so calling
    it dead would strand a working machine for a whole job. A truly wedged resolver is
    still caught: the caller's HEAD probe hangs on the same lookup and times out.
    """
    result: list = [None]

    def _probe() -> None:
        import socket as _socket
        try:
            _socket.getaddrinfo(host, None)
            result[0] = False
        except Exception:
            result[0] = True

    t = threading.Thread(target = _probe, daemon = True)
    t.start()
    t.join(timeout)
    return False if result[0] is None else result[0]


def hf_connect_target(endpoint: Optional[str] = None):
    """(host, port) egress actually has to reach: the proxy when one applies, else the endpoint."""
    from urllib.parse import urlparse

    url = endpoint or hf_endpoint_url()
    parsed = urlparse(url)
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        proxy = hf_proxy_for_endpoint(url)
        if proxy:
            p = urlparse(proxy if "://" in proxy else "http://" + proxy)
            # An https:// proxy with no explicit port listens on 443, not 80.
            return p.hostname, p.port or (443 if p.scheme == "https" else 80)
    except Exception:
        pass
    return parsed.hostname, parsed.port or default_port


def hf_tcp_reachable(timeout: float = 3.0, endpoint: Optional[str] = None) -> bool:
    """True when a TCP connection to the hub (or its proxy) can be established.

    Separates "no egress" from "slow to answer": a loaded server still handshakes promptly,
    a blackholed route times out. A refusal counts as reachable, since something answered.
    """
    import socket as _socket

    host, port = hf_connect_target(endpoint)
    if not host:
        return True
    try:
        with _socket.create_connection((host, port), timeout = timeout):
            return True
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    except Exception:
        return True


def hf_dns_dead(timeout: float = 2.0) -> bool:
    """Fast offline shortcut: the endpoint's host does not resolve and no proxy applies.

    False whenever a proxy is configured, so proxy-only setups fall through to the real
    reachability probe instead of being wrongly declared offline."""
    if hf_proxy_configured():
        return False
    return dns_host_dead(hf_endpoint_host(), timeout)


# One load makes many hub calls, so the verdict is shared briefly. Kept short in BOTH directions:
# a stale "reachable" misses the plug being pulled, and a stale "unreachable" sends a load to
# the cache after the user reconnected.
_HF_REACHABILITY_TTL_S = 5.0
_hf_reachability: Optional[tuple] = None
_hf_reachability_lock = threading.Lock()


def _reachability_fresh(entry) -> bool:
    """True while a cached (timestamp, unreachable) verdict may still be reused."""
    return entry is not None and (time.monotonic() - entry[0]) < _HF_REACHABILITY_TTL_S


def hf_probe_disabled() -> bool:
    """True when UNSLOTH_OFFLINE_PROBE opts out of the reachability probe."""
    return os.environ.get("UNSLOTH_OFFLINE_PROBE", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def hf_reachability_memo() -> Optional[bool]:
    """The memoised verdict while still fresh, else None.

    Lets a caller skip a cheaper-but-still-slow shortcut it has already effectively run:
    one request opens several guards, and repeating a 2s DNS lookup per guard adds up.
    Lock-free like force_hf_offline_active: the tuple read is atomic.
    """
    cached = _hf_reachability
    return cached[1] if _reachability_fresh(cached) else None


def reset_hf_reachability_cache() -> None:
    """Drop the memoised verdict so the next call re-probes (tests, network changes)."""
    global _hf_reachability
    with _hf_reachability_lock:
        _hf_reachability = None


def hf_unreachable(timeout: int = 3) -> bool:
    """True when the HF endpoint is unreachable, memoised for _HF_REACHABILITY_TTL_S.

    DNS resolving does not mean the Hub is reachable: a live router with the WAN down, a
    captive portal or a stale DNS cache all answer lookups while every request then burns
    huggingface_hub's retry backoff. Bounded and proxy-aware, as the export path already
    does; UNSLOTH_OFFLINE_PROBE=0 disables it. Fails open, so an unavailable probe reports
    reachable and the load decides as it does today.
    """
    if hf_probe_disabled():
        return False

    global _hf_reachability
    cached = _hf_reachability
    if _reachability_fresh(cached):
        return cached[1]

    with _hf_reachability_lock:
        cached = _hf_reachability
        if _reachability_fresh(cached):
            return cached[1]
        try:
            from utils.transformers_version import hf_endpoint_unreachable

            # Both flags off for the same reason: an ambiguous answer must not force offline.
            unreachable = hf_endpoint_unreachable(
                timeout,
                gateway_errors_offline = False,
                proxy_timeouts_offline = False,
            )
        except Exception:
            unreachable = False
        _hf_reachability = (time.monotonic(), unreachable)
        return unreachable


def _reset_hf_sessions() -> None:
    """Drop cached hub sessions so they remount with the current offline adapter."""
    try:
        from huggingface_hub.utils import _http

        for name in ("_get_session_from_cache", "get_session"):
            cache_clear = getattr(getattr(_http, name, None), "cache_clear", None)
            if cache_clear is not None:
                cache_clear()
        reset = getattr(_http, "reset_sessions", None)
        if reset is not None:
            reset()
    except Exception:
        pass


# Process-global, so nested/concurrent loads refcount rather than restore out from under each other.
_force_offline_depth = 0
_force_offline_saved: list = []
_force_offline_saved_env: dict = {}
# Spawn contexts can nest while holding this lock through Process.start().
_force_offline_lock = threading.RLock()

_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_OFFLINE_CONSTANTS = (
    ("huggingface_hub.constants", ("HF_HUB_OFFLINE",)),
    ("transformers.utils.hub", ("_is_offline_mode", "OFFLINE")),
)


def force_hf_offline_active() -> bool:
    """True while a force_hf_offline window is open anywhere in this process.

    Lets a concurrent caller tell our forced offline apart from one the user set, so it
    takes its own reference instead of no-opping and losing offline when the first exits.

    Lock-free: hf_environment_restored_for_spawn holds the lock across Process.start(), and
    blocking for that window would stall the operation the guard protects. The int read is
    atomic and the depth rises only after env and constants are already offline.
    """
    return _force_offline_depth > 0


def force_hf_offline_state() -> tuple[bool, bool]:
    """Return guard ownership and env presence under one lock."""
    with _force_offline_lock:
        return _force_offline_depth > 0, "HF_HUB_OFFLINE" in os.environ


def _restore_saved_offline_env(environment) -> None:
    """Apply the user's pre-guard offline intent to a child environment mapping."""
    for key in _OFFLINE_ENV_KEYS:
        value = _force_offline_saved_env.get(key)
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    # Hub ignores TRANSFORMERS_OFFLINE, so preserve that user intent in children.
    if (
        "HF_HUB_OFFLINE" not in environment
        and str(environment.get("TRANSFORMERS_OFFLINE", "")).strip().lower()
        in _HF_OFFLINE_TRUE_VALUES
    ):
        environment["HF_HUB_OFFLINE"] = "1"


def hf_environment_for_spawn() -> dict[str, str]:
    """Copy the environment without scoped offline values."""
    return hf_environment_scrubbed(os.environ)


def hf_environment_scrubbed(base) -> dict[str, str]:
    """Copy an env mapping with our scoped offline values replaced by the user's intent.

    A caller that captured os.environ itself would otherwise hand a child the
    HF_HUB_OFFLINE=1 we set for one operation, and the child would stay cache-only for life.
    """
    with _force_offline_lock:
        environment = dict(base)
        if _force_offline_depth > 0:
            _restore_saved_offline_env(environment)
        return environment


@contextmanager
def hf_environment_restored_for_spawn():
    """Restore user offline values while multiprocessing snapshots ``os.environ``."""
    with _force_offline_lock:
        if _force_offline_depth == 0:
            yield
            return

        missing = object()
        forced_environment = {key: os.environ.get(key, missing) for key in _OFFLINE_ENV_KEYS}
        _restore_saved_offline_env(os.environ)
        try:
            yield
        finally:
            for key, value in forced_environment.items():
                if value is missing:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@contextmanager
def force_hf_offline():
    """Force HF offline for this block, in-process.

    Env vars alone are too late once running: huggingface_hub and transformers read their
    offline constants at import and sessions cache a non-offline adapter. Flip the constants
    and rebuild the sessions so hub calls fail fast. All restored on exit."""
    global _force_offline_depth, _force_offline_saved, _force_offline_saved_env
    import importlib

    with _force_offline_lock:
        if _force_offline_depth == 0:
            saved: list = []
            saved_env: dict = {}
            # Snapshot constants BEFORE forcing the env, else a module imported inside the window reads the "1".
            for mod_name, attrs in _OFFLINE_CONSTANTS:
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                for attr in attrs:
                    if hasattr(mod, attr):
                        saved.append((mod, attr, getattr(mod, attr)))
            for key in _OFFLINE_ENV_KEYS:
                saved_env[key] = os.environ.get(key)
                os.environ[key] = "1"
            for mod, attr, _ in saved:
                try:
                    setattr(mod, attr, True)
                except Exception:
                    pass
            _force_offline_saved = saved
            _force_offline_saved_env = saved_env
            _reset_hf_sessions()
        _force_offline_depth += 1
    try:
        yield
    finally:
        with _force_offline_lock:
            _force_offline_depth -= 1
            if _force_offline_depth == 0:
                for mod, attr, val in _force_offline_saved:
                    try:
                        setattr(mod, attr, val)
                    except Exception:
                        pass
                _force_offline_saved = []
                for key, val in _force_offline_saved_env.items():
                    if val is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = val
                _force_offline_saved_env = {}
                _reset_hf_sessions()


def st_repo_id_candidates(model_name: str) -> list:
    """Repo ids a Sentence-Transformers load may resolve model_name to; a slashless name
    also resolves under the sentence-transformers/ namespace, so both are candidates."""
    name = (model_name or "").strip().strip("/")
    if not name:
        return []
    candidates = [name]
    if "/" not in name:
        candidates.append(f"sentence-transformers/{name}")
    return candidates


def _expand_path(raw: str) -> Path:
    """Expand ~ and $VARS as huggingface_hub does, so the gate resolves the loader's dir."""
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def _hf_cache_roots() -> list:
    """Cache roots to search for a model's local snapshot, most-authoritative first.

    The app's selected hub cache (set via /settings) is searched first: after a
    no-restart cache switch the process env is stale, yet the loader reads the
    selected cache via ``cache_folder=active_hf_hub_cache()``, so the snapshot
    and offline security lookups must match where it actually loads. The env
    precedence (SENTENCE_TRANSFORMERS_HOME, HF_HUB_CACHE, HF_HOME/hub,
    ~/.cache/huggingface/hub) follows so a copy still in a previous cache resolves."""
    roots: list = []
    seen: set = set()

    def _add(path) -> None:
        if path is None:
            return
        expanded = _expand_path(str(path))
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            roots.append(expanded)

    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        _add(get_hf_cache_paths().hub_cache)
    except Exception:
        pass

    if st_home := os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        _add(st_home)
    if hub := (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        _add(hub)
    if hf_home := os.environ.get("HF_HOME"):
        _add(_expand_path(hf_home) / "hub")
    if not roots:
        _add(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


ST_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def is_st_weight_name(basename: str) -> bool:
    """Whether a filename is a checkpoint SentenceTransformer can load.

    ``.bin`` is the loose one: ``tokenizer.bin`` shares the extension with real
    weights. Shared so the resolver's plan and the loader's cache check cannot
    disagree about what counts as a checkpoint."""
    name = basename.lower()
    for suffix in ST_WEIGHT_SUFFIXES:
        if not name.endswith(suffix):
            continue
        if suffix == ".bin":
            return name.startswith(("pytorch_model", "model", "adapter_model", "consolidated"))
        return True
    return False


def cached_st_source(model_name: str) -> Optional[tuple]:
    """``(repo id, snapshot dir)`` whose cache holds ST-loadable weights, complete.

    Alias-aware, and it reports WHICH candidate matched: a slashless name caches
    under ``sentence-transformers/``, so the literal id names a repo that usually
    does not exist, and a stale literal cache entry is not the directory that
    supplied the weights. Completeness comes from
    ``hf_cache_snapshot_is_loadable`` on that same candidate: ST weights alone are
    satisfied by the first finalized shard of a transfer still in flight.
    """
    for candidate in st_repo_id_candidates(model_name):
        # Exactly this candidate: the alias-expanding lookup answers a literal
        # slashless name with the namespaced snapshot, pairing a directory with a
        # repo id that supplied nothing.
        snapshot = hf_cache_snapshot_dir_for_repo(candidate)
        if snapshot is None:
            continue
        try:
            if not any(is_st_weight_name(p.name) and p.is_file() for p in snapshot.rglob("*")):
                continue
        except OSError:
            continue
        # This snapshot, not whatever the alias-expanding lookup would find: with
        # several cache roots those differ, and a complete namespaced copy in one
        # would vouch for the partial literal copy in another that gets loaded.
        if snapshot_is_loadable(snapshot, candidate):
            return (candidate, snapshot)
    return None


def cached_st_repo(model_name: str) -> Optional[str]:
    """Repo id whose cached snapshot holds complete ST-loadable weights."""
    source = cached_st_source(model_name)
    return source[0] if source else None


def snapshot_has_st_weights(model_name: str) -> bool:
    """Whether ``model_name`` has a complete cached checkpoint ST can open.

    ``hf_cache_snapshot_is_loadable`` counts ``.gguf``, which is right for the
    llama backend and wrong wherever SentenceTransformer is the loader; this pairs
    it with the ST-specific file family so both hold."""
    return cached_st_source(model_name) is not None


def _snapshot_in_root(cache_root: Path, repo_id: str) -> Optional[Path]:
    """``repo_id``'s main-revision snapshot under exactly ``cache_root``, or None."""
    try:
        from huggingface_hub.file_download import repo_folder_name
    except Exception:
        repo_folder_name = None
    try:
        if repo_folder_name is not None:
            folder = repo_folder_name(repo_id = repo_id, repo_type = "model")
        else:
            folder = "models--" + repo_id.replace("/", "--")
        repo_dir = cache_root / folder
        ref = repo_dir / "refs" / "main"
        if not ref.is_file():
            return None
        commit = ref.read_text(encoding = "utf-8").strip()
        if not commit:
            return None
        snapshot = repo_dir / "snapshots" / commit
        return snapshot if snapshot.is_dir() else None
    # UnicodeDecodeError is a ValueError, not an OSError: a torn refs file must keep meaning "not cached here".
    except (OSError, UnicodeDecodeError):
        return None


def hf_cache_snapshot_dir_for_repo(repo_id: str) -> Optional[Path]:
    """Snapshot dir for exactly ``repo_id``, with no alias expansion.

    ``hf_cache_snapshot_dir`` answers "is this model cached anywhere", trying the
    ST alias, so asking it about a literal slashless name can return the
    namespaced snapshot. A caller that has to report WHICH repo supplied the
    weights needs this one instead, or it pairs the alias's directory with the
    literal id and sends verification at a repo that does not exist."""
    for cache_root in _hf_cache_roots():
        snapshot = _snapshot_in_root(cache_root, repo_id)
        if snapshot is not None:
            return snapshot
    return None


def hf_cache_snapshot_dir(model_name: str) -> Optional[Path]:
    """Active local snapshot dir for model_name's main revision, or None if not cached.
    Reads refs/main then snapshots/<commit>; no network. Tries the ST alias for slashless names."""
    for cache_root in _hf_cache_roots():
        for repo_id in st_repo_id_candidates(model_name):
            snapshot = _snapshot_in_root(cache_root, repo_id)
            if snapshot is not None:
                return snapshot
    return None


# A weight file plus a config distinguishes a real cached model from a metadata-only partial cache.
_LOADABLE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt"})


def checkpoint_directory_is_complete(root: Path, weights = None) -> bool:
    """Whether ``root`` holds a whole checkpoint, shards and declared modules alike.

    Shared by the Hub-cache check and the local-path one so a directory is judged
    the same way however it got there: a single shard of a two-shard family, or a
    module ``modules.json`` declares and the directory does not have, is a torn
    checkpoint that SentenceTransformer fails to open at the first index.

    ``weights`` is the already-scanned weight list when the caller has one.
    """
    from hub.utils.inventory_scan import snapshot_holds_a_complete_payload

    if weights is None:
        weights = [
            path
            for path in root.rglob("*")
            if path.suffix.lower() in _LOADABLE_WEIGHT_SUFFIXES
            and path.is_file()
            and not is_appledouble_metadata(path)
        ]
    # SentenceTransformer modules may keep their own transformer checkpoint
    # below 0_Transformer/. Validate every module subtree that carries weights;
    # config-only modules such as Pooling need no weight family of their own.
    if (root / "modules.json").is_file():
        import json
        from pathlib import PurePosixPath

        try:
            modules = json.loads((root / "modules.json").read_text(encoding = "utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        roots = []
        for module in modules if isinstance(modules, list) else []:
            value = module.get("path") if isinstance(module, dict) else None
            if not isinstance(value, str) or "\\" in value:
                continue
            relative = PurePosixPath(value or ".")
            if relative.is_absolute() or ".." in relative.parts:
                continue
            module_root = root.joinpath(*relative.parts)
            # A declared module the directory lacks entirely is a torn checkpoint whatever the others hold; existence is
            # the whole test, since config-only modules have no weight family.
            if module_root != root and not module_root.is_dir():
                return False
            if any(path == module_root or module_root in path.parents for path in weights):
                roots.append(module_root)
        if roots:
            return all(snapshot_holds_a_complete_payload(r, quants = False) for r in roots)
    return snapshot_holds_a_complete_payload(root, quants = False)


def hf_cache_snapshot_is_loadable(model_name: str) -> bool:
    """True when the cached snapshot can satisfy a cache-only transformer load.

    App-managed downloads are checked against their exact manifest. Imported or
    legacy caches without one fall back to the same weight-family/index scanner
    used by Hub inventory, so one shard of a cancelled checkpoint is not enough.
    No network.
    """
    snapshot = hf_cache_snapshot_dir(model_name)
    if snapshot is None:
        return False
    return snapshot_is_loadable(snapshot, model_name)


def snapshot_is_loadable(snapshot, model_name: str) -> bool:
    """``hf_cache_snapshot_is_loadable`` for a snapshot the caller already has.

    A caller that picked a specific directory has to have THAT one judged: the
    lookup above expands the ST alias within each cache root while an exact
    per-repo lookup walks the roots for one id, so with several roots configured
    the two can land on different snapshots, and the verdict would then belong to
    a directory nobody is going to load.
    """
    try:
        has_config = (snapshot / "config.json").is_file() or (snapshot / "modules.json").is_file()
        if not has_config:
            return False

        weights = []
        for path in snapshot.rglob("*"):
            if path.suffix.lower() not in _LOADABLE_WEIGHT_SUFFIXES or not path.is_file():
                continue
            if not is_appledouble_metadata(path):
                weights.append(path)
        if not weights:
            return False

        # A managed full-snapshot transfer records its exact expected files
        # before downloading. A cancel marker or unfinished blob is conclusive
        # even when config.json and the first finalized shard already exist.
        repo_dir = snapshot.parent.parent
        hub_cache = repo_dir.parent
        repo_id = model_name
        try:
            from huggingface_hub.file_download import repo_folder_name
            for candidate in st_repo_id_candidates(model_name):
                if repo_folder_name(repo_id = candidate, repo_type = "model") == repo_dir.name:
                    repo_id = candidate
                    break
        except Exception:
            pass
        from hub.utils import download_manifest
        from hub.utils.hf_cache_state import snapshot_has_broken_symlinks

        if download_manifest.has_cancel_marker("model", repo_id, None, hub_cache = hub_cache):
            return False
        manifest = download_manifest.read_manifest("model", repo_id, None, hub_cache = hub_cache)
        if manifest is not None:
            # This exact full-snapshot plan is stronger evidence than an
            # unrelated .incomplete blob left under the repository by another
            # revision or scoped GGUF job.
            return download_manifest.verify_against_disk(manifest, snapshot).ok
        # Judge THIS snapshot's own links, not every blob in the shared cache directory, or a stray .incomplete from
        # another revision condemns a model that is fully present.
        if snapshot_has_broken_symlinks(snapshot):
            return False

        return checkpoint_directory_is_complete(snapshot, weights)
    except OSError:
        return False
    except Exception:
        # Completeness is a safety property here: an unprovable partial must keep the pending marker so
        # the loader cannot silently reach the network.
        return False


# Never return raw exception text to clients: log server-side, return generic.


# ── Client-safe error helpers ───────────────────────────────────
# Never return raw exception text to clients; log server-side, return generic.
def safe_error_detail(error: Exception, fallback: str = "An internal error occurred") -> str:
    """Map an exception to a generic, client-safe message (never raw
    ``str(error)``, which can leak paths). Log the real exception server-side.
    """
    # A mid-stream llama-server failure carries a message that was written to be shown
    # Without this the non-streaming paths reduced it to the fallback while streaming clients got the cause. Imported
    # lazily: utils is low level and must not depend on core.inference at import time.
    try:
        from core.inference.stream_errors import LlamaStreamError  # noqa: PLC0415
        if isinstance(error, LlamaStreamError) and error.friendly:
            return error.friendly
    except Exception:  # noqa: BLE001 -- fall through to the generic mapping below
        pass
    text = str(error).lower()
    if (
        isinstance(error, (ConnectionError, TimeoutError))
        or "connection" in text
        or "timed out" in text
        or "timeout" in text
    ):
        return "Could not reach an upstream service. Please try again."
    if "out of memory" in text or "cuda error" in text:
        return "Ran out of memory. Try a smaller model or shorter input."
    return fallback


def safe_curated_detail(error: Exception, fallback: str = "An internal error occurred") -> str:
    """Client-safe text for curated domain/validation exceptions.

    Keeps the message (paths stripped) instead of a generic fallback; for known
    exception types only (use ``safe_error_detail`` for generic ``Exception``).
    """
    from utils.native_path_leases import redact_native_paths

    msg = redact_native_paths(str(error)).strip()
    return msg or fallback


def log_and_http_error(
    error: Exception,
    status_code: int,
    public_message: str,
    *,
    event: str = "request_failed",
    log = None,
    headers: Optional[dict] = None,
):
    """Log ``error`` in full server-side and return an ``HTTPException`` whose
    ``detail`` is only ``public_message`` -- never the raw exception text.

    Usage:  raise log_and_http_error(e, 500, "Failed to start training")
    """
    from fastapi import HTTPException

    # A 4xx is a normal outcome the caller handles.
    # One warning line and no traceback: at error with exc_info, one generation buried the log under 54 rejected saves.
    # 5xx keeps the traceback, and exc_info works for structlog too.
    emitter = log or logger
    if 400 <= status_code < 500:
        emitter.warning(f"{event}: {error}")
    else:
        emitter.error(f"{event}: {error}", exc_info = error)
    return HTTPException(status_code = status_code, detail = public_message, headers = headers)


@contextmanager
def without_hf_auth():
    """
    Temporarily disable HuggingFace authentication.

    Usage:
        with without_hf_auth():
            # Code that should run without cached tokens
            model_info(model_name, token=None)
    """
    saved_env = {}
    env_vars = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME"]
    for var in env_vars:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    saved_disable = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    token_files = []
    token_locations = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]

    for token_loc in token_locations:
        if token_loc.exists():
            temp = tempfile.NamedTemporaryFile(delete = False)
            temp.close()
            shutil.move(str(token_loc), temp.name)
            token_files.append((token_loc, temp.name))

    try:
        yield
    finally:
        for original, temp in token_files:
            try:
                original.parent.mkdir(parents = True, exist_ok = True)
                shutil.move(temp, str(original))
            except Exception as e:
                logger.error(f"Failed to restore token {original}: {e}")

        for var, value in saved_env.items():
            os.environ[var] = value

        if saved_disable is not None:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = saved_disable
        else:
            os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)


def is_hf_authentication_error(error: Exception) -> bool:
    """Return whether an exception chain contains a definitive HF auth failure."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        try:
            if status is not None and int(status) == 401:
                return True
        except (TypeError, ValueError):
            pass
        message = str(current).lower()
        if "invalid user token" in message or "invalid hf token" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def format_error_message(error: Exception, model_name: str) -> str:
    """
    Format a user-friendly error message for common load issues.

    Args:
        error: The exception that occurred
        model_name: Name of the model being loaded
    """
    error_str = str(error).lower()
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    if "repository not found" in error_str or "404" in error_str:
        return f"Model '{model_short}' not found. Check the model name."

    if "401" in error_str or "unauthorized" in error_str:
        return f"Authentication failed for '{model_short}'. Please provide a valid HF token."

    if "gated" in error_str or "access to model" in error_str:
        return f"Model '{model_short}' requires authentication. Please provide a valid HF token."

    if "invalid user token" in error_str:
        return "Invalid HF token. Please check your token and try again."

    if (
        "out of memory" in error_str
        or "out of device memory" in error_str
        or "out_of_device_memory" in error_str
        or "out_of_host_memory" in error_str
        or "not enough memory" in error_str
        or "cannot allocate memory" in error_str
        or "memory allocation failed" in error_str
        or "cublas_status_alloc_failed" in error_str
        or ("cuda error" in error_str and "alloc" in error_str)
        or ("xpu" in error_str and ("alloc" in error_str or "memory" in error_str))
        or isinstance(error, MemoryError)
        or ("mlx" in error_str and ("memory" in error_str or "allocate" in error_str))
    ):
        # Resolve get_device() at call time so tests that monkey-patch it after import see the patch.
        from utils.hardware import get_device

        device = get_device()
        device_label = {
            "cuda": "GPU",
            "xpu": "Intel GPU",
            "mlx": "Apple Silicon GPU",
            "cpu": "system",
        }.get(device.value, "GPU")
        return f"Not enough {device_label} memory to load '{model_short}'. Try a smaller model or free memory."

    return str(error)
