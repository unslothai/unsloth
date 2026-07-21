"""Unit tests for the offline-loading helpers in unsloth/models/loader_utils.py:
error classification, _force_hf_offline flip/restore, and the retry orchestrator.
Pure CPU, no network, no GPU."""

import os
import socket

import pytest

from unsloth.models import loader_utils as L


# ---------------------------------------------------------------------------
# _env_says_offline / _get_effective_local_files_only
# ---------------------------------------------------------------------------

_OFFLINE_TRUE = ("1", "true", "yes", "on", "ON", " 1 ", "\tyes\n")
_OFFLINE_FALSE = ("0", "no", "false", "off", "", "  ", "maybe")


@pytest.mark.parametrize("value", _OFFLINE_TRUE)
def test_env_says_offline_truthy(monkeypatch, value):
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert L._env_says_offline() is True


@pytest.mark.parametrize("value", _OFFLINE_FALSE)
def test_env_says_offline_falsy(monkeypatch, value):
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert L._env_says_offline() is False


def test_env_says_offline_absent(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    assert L._env_says_offline() is False


def test_env_says_offline_transformers_var(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert L._env_says_offline() is True


def test_effective_lfo_kwarg_wins(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    assert L._get_effective_local_files_only({"local_files_only": True}) is True


def test_effective_lfo_env_only(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert L._get_effective_local_files_only({}) is True


def test_effective_lfo_neither(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    assert L._get_effective_local_files_only({"local_files_only": False}) is False


def test_effective_lfo_is_read_only():
    # Must not pop local_files_only: the weight load reuses the same kwarg.
    kwargs = {"local_files_only": True}
    L._get_effective_local_files_only(kwargs)
    assert kwargs == {"local_files_only": True}


# ---------------------------------------------------------------------------
# _is_offline_related_error
# ---------------------------------------------------------------------------


def _http_error(status):
    import requests

    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError("http %s" % status, response = resp)


def test_none_is_not_offline():
    assert L._is_offline_related_error(None) is False


def test_plain_connection_error_is_offline():
    assert L._is_offline_related_error(ConnectionError("down")) is True


def test_timeout_error_is_offline():
    assert L._is_offline_related_error(TimeoutError("slow")) is True


def test_plain_file_not_found_propagates():
    assert L._is_offline_related_error(FileNotFoundError("config.json")) is False


def test_unrelated_error_is_not_offline():
    assert L._is_offline_related_error(ValueError("bad arg")) is False


def test_requests_connection_error_is_offline():
    import requests
    assert L._is_offline_related_error(requests.exceptions.ConnectionError("x")) is True


@pytest.mark.parametrize("status", (500, 502, 503, 504))
def test_http_5xx_is_offline(status):
    assert L._is_offline_related_error(_http_error(status)) is True


@pytest.mark.parametrize("status", (400, 401, 403, 404))
def test_http_4xx_propagates(status):
    assert L._is_offline_related_error(_http_error(status)) is False


def test_status_less_http_with_network_wording_is_offline():
    import requests
    err = requests.exceptions.HTTPError("Couldn't connect to the server")
    assert L._is_offline_related_error(err) is True


def test_status_less_http_without_network_wording_propagates():
    import requests
    err = requests.exceptions.HTTPError("I'm a teapot")
    assert L._is_offline_related_error(err) is False


def test_gaierror_dns_failure_is_offline():
    assert (
        L._is_offline_related_error(socket.gaierror(-2, "Name or service not known"))
        is True
    )


def test_gaierror_without_wording_is_offline_by_type():
    # Matched by type, so a locale-specific / empty message still classifies offline.
    assert L._is_offline_related_error(socket.gaierror(-2, "")) is True


def test_urllib_urlerror_is_offline():
    import urllib.error
    assert (
        L._is_offline_related_error(urllib.error.URLError("connection failed")) is True
    )


def test_urllib_httperror_404_propagates():
    import urllib.error
    err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
    assert L._is_offline_related_error(err) is False


def test_urllib_httperror_503_is_offline():
    import urllib.error
    err = urllib.error.HTTPError("http://x", 503, "Service Unavailable", {}, None)
    assert L._is_offline_related_error(err) is True


def test_ssl_error_is_not_offline():
    # TLS/cert failure must surface, not silently fall back to cached files.
    import ssl
    assert (
        L._is_offline_related_error(ssl.SSLError("certificate verify failed")) is False
    )


def test_requests_ssl_error_is_not_offline():
    # requests.SSLError subclasses ConnectionError, but is still a TLS failure -> not offline.
    requests = pytest.importorskip("requests")
    assert (
        L._is_offline_related_error(requests.exceptions.SSLError("bad cert")) is False
    )


def test_urlerror_wrapping_ssl_is_not_offline():
    import ssl
    import urllib.error

    err = urllib.error.URLError(ssl.SSLCertVerificationError("self-signed certificate"))
    assert L._is_offline_related_error(err) is False


def test_ssl_node_does_not_hide_deeper_connection_cause():
    # Skipping a TLS node must not abort the walk: a genuine outage deeper still counts.
    import ssl

    outer = RuntimeError("load failed")
    mid = ssl.SSLError("cert")
    mid.__context__ = ConnectionError("down")
    outer.__cause__ = mid
    assert L._is_offline_related_error(outer) is True


def test_oserror_network_unreachable_is_offline():
    assert L._is_offline_related_error(OSError("Network is unreachable")) is True


def test_offline_mode_is_enabled_is_offline():
    errors = pytest.importorskip("huggingface_hub.errors")
    assert L._is_offline_related_error(errors.OfflineModeIsEnabled("offline")) is True


def test_local_entry_not_found_is_offline():
    # Both a FileNotFoundError and an HfHubHTTPError, but means "not cached + Hub down" -> offline.
    errors = pytest.importorskip("huggingface_hub.errors")
    assert (
        L._is_offline_related_error(errors.LocalEntryNotFoundError("missing")) is True
    )


def test_chained_cause_connection_error_is_offline():
    err = RuntimeError("combined load failure")
    err.__cause__ = ConnectionError("down")
    assert L._is_offline_related_error(err) is True


def test_chained_context_connection_error_is_offline():
    try:
        try:
            raise ConnectionError("down")
        except ConnectionError:
            raise RuntimeError("wrap")
    except RuntimeError as e:
        err = e
    assert L._is_offline_related_error(err) is True


def test_chained_cause_404_still_propagates():
    err = RuntimeError("combined load failure")
    err.__cause__ = _http_error(404)
    assert L._is_offline_related_error(err) is False


def test_cause_context_cycle_terminates():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__context__ = b
    b.__context__ = a
    # Must not hang; neither is network-related.
    assert L._is_offline_related_error(a) is False


# ---------------------------------------------------------------------------
# _force_hf_offline
# ---------------------------------------------------------------------------


def _inprocess_offline_flags():
    flags = []
    try:
        import huggingface_hub.constants as hfc
        if hasattr(hfc, "HF_HUB_OFFLINE"):
            flags.append(hfc.HF_HUB_OFFLINE)
    except Exception:
        pass
    try:
        import transformers.utils.hub as tuh
        for attr in ("_is_offline_mode", "OFFLINE"):
            if hasattr(tuh, attr):
                flags.append(getattr(tuh, attr))
    except Exception:
        pass
    return flags


def test_force_offline_sets_and_restores_absent_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    with L._force_hf_offline():
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    # Absent before -> absent after (not left as "1").
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_force_offline_preserves_prior_env_value(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with L._force_hf_offline():
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "0"


def test_force_offline_flips_inprocess_constants():
    before = _inprocess_offline_flags()
    with L._force_hf_offline():
        during = _inprocess_offline_flags()
        assert during, "expected at least one in-process offline flag to inspect"
        assert all(flag is True for flag in during)
    assert _inprocess_offline_flags() == before


def test_force_offline_nesting_shares_one_flip(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    with L._force_hf_offline():
        with L._force_hf_offline():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # Inner exit must NOT restore while the outer window is still open.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") is None


def test_force_offline_restores_on_exception(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    with pytest.raises(RuntimeError):
        with L._force_hf_offline():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            raise RuntimeError("boom")
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_force_offline_depth_returns_to_zero():
    assert L._force_offline_depth == 0
    with L._force_hf_offline():
        assert L._force_offline_depth == 1
    assert L._force_offline_depth == 0


def test_reset_hf_sessions_is_safe():
    # Best-effort no-op when the hub helper is missing; must never raise.
    L._reset_hf_sessions()


# ---------------------------------------------------------------------------
# _has_local_tokenizer_files / _resolve_checkpoint_tokenizer_name
# ---------------------------------------------------------------------------


def _touch(path, name):
    open(os.path.join(path, name), "w").close()


def test_has_local_tokenizer_json(tmp_path):
    _touch(tmp_path, "tokenizer.json")
    assert L._has_local_tokenizer_files(str(tmp_path)) is True


def test_has_local_tokenizer_model(tmp_path):
    _touch(tmp_path, "tokenizer.model")
    assert L._has_local_tokenizer_files(str(tmp_path)) is True


def test_has_local_tokenizer_bpe_needs_merges(tmp_path):
    # vocab.json alone is not loadable BPE; it needs merges.txt.
    _touch(tmp_path, "vocab.json")
    assert L._has_local_tokenizer_files(str(tmp_path)) is False
    _touch(tmp_path, "merges.txt")
    assert L._has_local_tokenizer_files(str(tmp_path)) is True


def test_has_local_tokenizer_empty_dir(tmp_path):
    assert L._has_local_tokenizer_files(str(tmp_path)) is False


def test_resolve_tokenizer_explicit_override_wins(tmp_path):
    kwargs = {"tokenizer_name": "base/repo"}
    assert L._resolve_checkpoint_tokenizer_name(str(tmp_path), kwargs) == "base/repo"
    # tokenizer_name is always popped (it is passed explicitly downstream too).
    assert "tokenizer_name" not in kwargs


def test_resolve_tokenizer_self_sufficient_dir(tmp_path):
    _touch(tmp_path, "tokenizer_config.json")
    _touch(tmp_path, "tokenizer.json")
    kwargs = {}
    assert L._resolve_checkpoint_tokenizer_name(str(tmp_path), kwargs) == str(tmp_path)


def test_resolve_tokenizer_config_without_files_falls_back(tmp_path):
    # Has tokenizer_config.json but no loadable tokenizer file -> base repo.
    _touch(tmp_path, "tokenizer_config.json")
    assert L._resolve_checkpoint_tokenizer_name(str(tmp_path), {}) is None


def test_resolve_tokenizer_nonexistent_dir_falls_back():
    assert L._resolve_checkpoint_tokenizer_name("/no/such/dir", {}) is None


# ---------------------------------------------------------------------------
# _offline_aware_load (the retry orchestrator)
# ---------------------------------------------------------------------------


def test_retry_once_on_offline_error_then_succeed(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ConnectionError("network down")
        return "ok"

    assert fake("model") == "ok"
    assert len(calls) == 2
    assert not calls[0].get("local_files_only")
    assert calls[1].get("local_files_only") is True
    assert L._force_offline_depth == 0


def test_no_retry_on_non_offline_error(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        raise ValueError("genuine bug, not a network issue")

    with pytest.raises(ValueError):
        fake("model")
    assert len(calls) == 1


def test_no_retry_when_already_offline_via_kwarg(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(dict(kwargs))
        # Offline window is active for the single attempt.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        return "ok"

    assert fake("model", local_files_only = True) == "ok"
    assert len(calls) == 1
    assert L._force_offline_depth == 0


def test_offline_error_when_already_offline_propagates(monkeypatch):
    # Already offline -> no online attempt to retry, so the error propagates once.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        fake("model")
    assert len(calls) == 1
    assert L._force_offline_depth == 0


def test_kwargs_preserved_across_retry(monkeypatch):
    # Callee popping config/tokenizer_name must not change what the retry sees:
    # fn(*args, **kwargs) re-packs a fresh **kwargs per call.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    seen = []

    @L._offline_aware_load
    def fake(model_name, **kwargs):
        cfg = kwargs.pop("config", None)
        tok = kwargs.pop("tokenizer_name", None)
        seen.append((cfg, tok))
        if len(seen) == 1:
            raise ConnectionError("down")
        return cfg, tok

    assert fake("m", config = "CFG", tokenizer_name = "TOK") == ("CFG", "TOK")
    assert seen == [("CFG", "TOK"), ("CFG", "TOK")]


def test_retry_runs_gc_collect_between_attempts(monkeypatch):
    # The retry lives OUTSIDE the except so the failed attempt's traceback (a
    # partial model) is freed by gc.collect() before the second load reallocates.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    gc_calls = []
    monkeypatch.setattr(L.gc, "collect", lambda *a, **k: gc_calls.append(1))
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("down")
        # By the retry attempt, gc.collect() must already have fired.
        assert gc_calls, "gc.collect must run before the offline retry"
        return "ok"

    gc_calls.clear()
    assert fake("model") == "ok"
    assert len(calls) == 2
    assert len(gc_calls) == 1


# ---------------------------------------------------------------------------
# _force_hf_offline — constant restore (no stale offline pin)
# ---------------------------------------------------------------------------


def test_force_offline_restores_freshly_imported_constant(monkeypatch):
    # If huggingface_hub.constants is first imported inside the window, the saved value must
    # be the pre-window state, not the just-forced "1"; otherwise the process pins offline.
    import sys

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    saved_mod = sys.modules.get("huggingface_hub.constants")
    saved_val = getattr(saved_mod, "HF_HUB_OFFLINE", None) if saved_mod else None
    try:
        sys.modules.pop(
            "huggingface_hub.constants", None
        )  # simulate "not imported yet"
        with L._force_hf_offline():
            import huggingface_hub.constants as hfc_in
            assert hfc_in.HF_HUB_OFFLINE is True  # forced offline inside the window
        import huggingface_hub.constants as hfc_after

        assert hfc_after.HF_HUB_OFFLINE is False  # restored, not pinned True
        assert os.environ.get("HF_HUB_OFFLINE") is None
    finally:
        if saved_mod is not None:
            sys.modules["huggingface_hub.constants"] = saved_mod
            if saved_val is not None:
                saved_mod.HF_HUB_OFFLINE = saved_val


# ---------------------------------------------------------------------------
# _resolve_checkpoint_tokenizer_name — VLM needs local processor files
# ---------------------------------------------------------------------------


def test_resolve_tokenizer_vlm_without_processor_falls_back(tmp_path):
    # VLM checkpoint with tokenizer files but no processor config -> base repo (None), so its
    # cached processor still loads instead of AutoProcessor failing on the local dir.
    _touch(tmp_path, "tokenizer_config.json")
    _touch(tmp_path, "tokenizer.json")
    assert (
        L._resolve_checkpoint_tokenizer_name(str(tmp_path), {}, require_processor = True)
        is None
    )


def test_resolve_tokenizer_vlm_with_processor_uses_local_dir(tmp_path):
    _touch(tmp_path, "tokenizer_config.json")
    _touch(tmp_path, "tokenizer.json")
    _touch(tmp_path, "preprocessor_config.json")
    assert L._resolve_checkpoint_tokenizer_name(
        str(tmp_path), {}, require_processor = True
    ) == str(tmp_path)
