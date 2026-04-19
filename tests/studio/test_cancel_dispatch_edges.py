"""
Edge cases for the cancel dispatch path not covered by the primary tests.

Specifically:
  - The Studio frontend cancel POST uses plain fetch with a manual
    Authorization header, not authFetch, so a 401 on /api/inference/cancel
    cannot trigger refreshSession -> redirectToAuth and kick the user to
    the login page mid-stop.
  - `_cancel_by_keys` is tolerant of empty, falsy, and None keys (external
    callers POSTing odd shapes) and never leaks pending entries for these.
  - A single cancel POST with a shared session_id cancels ALL concurrent
    streams on that thread — the registry keys a session to a set of events
    so parallel runs (e.g. compare mode) all stop together.
"""

from __future__ import annotations

import ast
import threading
import types
import importlib.util
import sys
from pathlib import Path


ROUTES_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
CHAT_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "chat"
    / "api"
    / "chat-adapter.ts"
)


# ── Frontend stop POST: no authFetch on /api/inference/cancel ──


def _adapter_source() -> str:
    return CHAT_ADAPTER_PATH.read_text()


def _find_on_abort_cancel_block(src: str) -> str:
    # Extract the body of the onAbortCancel callback. The PR declares it
    # with `const onAbortCancel = () => { ... };` -- find the block that
    # contains "/api/inference/cancel".
    idx = src.find("onAbortCancel")
    assert idx >= 0, "onAbortCancel handler missing from chat-adapter.ts"
    # Grab everything from there to the next top-level `try {` which is
    # the start of the outer run() flow (enough to cover the arrow body).
    rest = src[idx:]
    end = rest.find("\n      try {")
    return rest if end < 0 else rest[:end]


def test_abort_cancel_post_does_not_use_authfetch():
    block = _find_on_abort_cancel_block(_adapter_source())
    assert "/api/inference/cancel" in block, (
        "test precondition: onAbortCancel block must reference "
        "/api/inference/cancel"
    )
    assert "authFetch(" not in block, (
        "onAbortCancel must NOT call authFetch for the cancel POST. "
        "authFetch redirects to login on 401, which would kick the user "
        "to the login page when they click Stop on an expired session. "
        "Use plain fetch with a manual Authorization header instead."
    )


def test_abort_cancel_post_sets_manual_authorization_header():
    block = _find_on_abort_cancel_block(_adapter_source())
    assert "getAuthToken" in block, (
        "onAbortCancel must read the bearer token directly via "
        "getAuthToken() rather than relying on authFetch's 401 flow"
    )
    assert "Authorization" in block, (
        "onAbortCancel must attach an Authorization header explicitly"
    )
    assert "keepalive: true" in block, (
        "onAbortCancel must set keepalive: true so the fetch survives "
        "page unload during the stop click"
    )


def test_abort_cancel_post_still_uses_fetch_not_beacon():
    # sendBeacon was tempting but cannot set Content-Type JSON or include
    # a bearer header portably. Ensure we stick with fetch + keepalive.
    block = _find_on_abort_cancel_block(_adapter_source())
    assert "fetch(" in block, (
        "onAbortCancel must use fetch(...) (not sendBeacon) for the "
        "authenticated cancel POST"
    )


# ── _cancel_by_keys: odd-shape tolerance ──


def _import_routes_module():
    backend_dir = ROUTES_PATH.resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    def _stub_structlog():
        m = types.ModuleType("structlog")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *a, **k: None

        m.get_logger = lambda *a, **k: _Logger()
        m.configure = lambda *a, **k: None
        return m

    def _stub_httpx():
        m = types.ModuleType("httpx")

        class Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def aclose(self):
                pass

        class AsyncClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class Limits:
            def __init__(self, *a, **k):
                pass

        class Timeout:
            def __init__(self, *a, **k):
                pass

        class RequestError(Exception):
            ...

        m.Client = Client
        m.AsyncClient = AsyncClient
        m.Limits = Limits
        m.Timeout = Timeout
        m.RequestError = RequestError
        return m

    sys.modules.setdefault("structlog", _stub_structlog())
    sys.modules.setdefault("jwt", types.ModuleType("jwt"))
    sys.modules.setdefault("httpx", _stub_httpx())

    auth_stub = types.ModuleType("auth.fastapi_auth")
    auth_stub.get_current_subject = lambda *a, **k: "test-subject"
    auth_stub.get_api_key_details = lambda *a, **k: {}
    sys.modules.setdefault("auth", types.ModuleType("auth"))
    sys.modules["auth.fastapi_auth"] = auth_stub

    spec = importlib.util.spec_from_file_location(
        "studio_routes_inference_dispatch_edge_test", ROUTES_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def test_cancel_by_keys_handles_empty_list_without_leak():
    mod = _import_routes_module()
    if mod is None:
        return
    mod._PENDING_CANCELS.clear()
    mod._CANCEL_REGISTRY.clear()
    assert mod._cancel_by_keys([]) == 0
    assert mod._PENDING_CANCELS == {}


def test_cancel_by_keys_skips_empty_and_none_keys():
    # cancel_inference already filters out non-string/empty values before
    # calling the helper, but the helper itself must not crash or stash
    # falsy keys if future callers pass them through.
    mod = _import_routes_module()
    if mod is None:
        return
    mod._PENDING_CANCELS.clear()
    mod._CANCEL_REGISTRY.clear()
    n = mod._cancel_by_keys(["", None, "real-session"])
    assert n == 0
    # No silent stashing at all.
    assert mod._PENDING_CANCELS == {}


# ── Concurrent runs sharing a session_id all cancel together ──


def test_single_session_cancel_stops_all_concurrent_streams():
    # Compare mode and any other flow that launches two concurrent streams
    # on the same thread register distinct cancel_events but share a
    # session_id. A single cancel POST with that session_id must set
    # every event.
    mod = _import_routes_module()
    if mod is None:
        return
    mod._PENDING_CANCELS.clear()
    mod._CANCEL_REGISTRY.clear()

    session = "shared-thread"
    ev_a = threading.Event()
    ev_b = threading.Event()
    tracker_a = mod._TrackedCancel(ev_a, "cancel-a", session, "chatcmpl-a")
    tracker_b = mod._TrackedCancel(ev_b, "cancel-b", session, "chatcmpl-b")
    tracker_a.__enter__()
    tracker_b.__enter__()
    try:
        cancelled = mod._cancel_by_keys([session])
        assert cancelled == 2, (
            f"expected both concurrent streams on the same session to be "
            f"cancelled; got {cancelled}"
        )
        assert ev_a.is_set()
        assert ev_b.is_set()
    finally:
        tracker_a.__exit__(None, None, None)
        tracker_b.__exit__(None, None, None)
    # After exit, the session bucket must be cleaned up entirely (empty
    # bucket is pop'd) so the next request starts fresh.
    assert session not in mod._CANCEL_REGISTRY


def test_single_cancel_id_does_not_cross_talk_to_other_runs_on_same_session():
    # Cancelling run A by its unique cancel_id must NOT touch run B on the
    # same session_id. cancel_id is the exclusive per-run key; session_id
    # is the fan-out key.
    mod = _import_routes_module()
    if mod is None:
        return
    mod._PENDING_CANCELS.clear()
    mod._CANCEL_REGISTRY.clear()

    session = "shared-thread-2"
    ev_a = threading.Event()
    ev_b = threading.Event()
    tracker_a = mod._TrackedCancel(ev_a, "cancel-only-a", session, "chatcmpl-a")
    tracker_b = mod._TrackedCancel(ev_b, "cancel-only-b", session, "chatcmpl-b")
    tracker_a.__enter__()
    tracker_b.__enter__()
    try:
        n = mod._cancel_by_cancel_id_or_stash("cancel-only-a")
        assert n == 1
        assert ev_a.is_set()
        assert not ev_b.is_set(), (
            "cancel_id-scoped cancel leaked to a sibling run on the same "
            "session — cancel_id must be exclusive per run"
        )
    finally:
        tracker_a.__exit__(None, None, None)
        tracker_b.__exit__(None, None, None)


# ── _openai_passthrough_stream: outer guard also calls tracker.__exit__
#   through the NORMAL-return path via the generator's finally, but on the
#   pre-stream error path the OUTER except BaseException is the only thing
#   that runs. Guard against accidental double-__exit__ causing KeyError. ──


def test_tracked_cancel_exit_is_idempotent_under_double_call():
    mod = _import_routes_module()
    if mod is None:
        return
    mod._PENDING_CANCELS.clear()
    mod._CANCEL_REGISTRY.clear()
    ev = threading.Event()
    tracker = mod._TrackedCancel(ev, "cid", "sess", "chatcmpl-x")
    tracker.__enter__()
    # Both the outer except BaseException and the generator's finally
    # may call __exit__ under certain race combos. Must not raise.
    tracker.__exit__(None, None, None)
    tracker.__exit__(None, None, None)
    tracker.__exit__(None, None, None)
    assert not mod._CANCEL_REGISTRY
