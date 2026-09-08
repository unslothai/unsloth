"""Unit tests for the offline-loading helpers in unsloth/models/loader_utils.py:
error classification, _force_hf_offline flip/restore, and the retry orchestrator.
Pure CPU, no network, no GPU."""

import os
import socket
import traceback

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
    assert L._is_offline_related_error(socket.gaierror(-2, "Name or service not known")) is True


def test_gaierror_without_wording_is_offline_by_type():
    # Matched by type, so a locale-specific / empty message still classifies offline.
    assert L._is_offline_related_error(socket.gaierror(-2, "")) is True


def test_urllib_urlerror_is_offline():
    import urllib.error
    assert L._is_offline_related_error(urllib.error.URLError("connection failed")) is True


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
    assert L._is_offline_related_error(ssl.SSLError("certificate verify failed")) is False


def test_requests_ssl_error_is_not_offline():
    # requests.SSLError subclasses ConnectionError, but is still a TLS failure -> not offline.
    requests = pytest.importorskip("requests")
    assert L._is_offline_related_error(requests.exceptions.SSLError("bad cert")) is False


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
    assert L._is_offline_related_error(errors.LocalEntryNotFoundError("missing")) is True


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
    # vocab.json alone is not loadable BPE;
    # it needs merges.txt.
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
    # Callee popping config/tokenizer_name must not change what the retry sees: fn(*args, **kwargs) re-packs a fresh
    # **kwargs per call.
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
    # The retry lives OUTSIDE the except so the failed attempt's traceback (a partial model) is freed by gc.collect()
    # before the second load reallocates.
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
    # If huggingface_hub.constants is first imported inside the window, the saved value must be the pre-window state,
    # not the just-forced "1"; otherwise the process pins offline.
    import sys

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    saved_mod = sys.modules.get("huggingface_hub.constants")
    saved_val = getattr(saved_mod, "HF_HUB_OFFLINE", None) if saved_mod else None
    try:
        sys.modules.pop("huggingface_hub.constants", None)  # simulate "not imported yet"
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
    # VLM checkpoint with tokenizer files but no processor config -> base repo (None), so its cached processor still
    # loads instead of AutoProcessor failing on the local dir.
    _touch(tmp_path, "tokenizer_config.json")
    _touch(tmp_path, "tokenizer.json")
    assert L._resolve_checkpoint_tokenizer_name(str(tmp_path), {}, require_processor = True) is None


def test_resolve_tokenizer_vlm_with_processor_uses_local_dir(tmp_path):
    _touch(tmp_path, "tokenizer_config.json")
    _touch(tmp_path, "tokenizer.json")
    _touch(tmp_path, "preprocessor_config.json")
    assert L._resolve_checkpoint_tokenizer_name(str(tmp_path), {}, require_processor = True) == str(
        tmp_path
    )


# ---------------------------------------------------------------------------
# what the retry reports when it fails too
# ---------------------------------------------------------------------------


def test_the_online_error_is_what_surfaces_when_the_cache_is_empty(monkeypatch):
    """The retry only succeeds on what is cached, so its own failure names an empty
    cache badly: offline mode skips Transformers' "does not appear to have a file
    named" raise, so the user saw `AttributeError: 'NoneType' ... 'endswith'`."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ConnectionError("connection reset while downloading model.safetensors")
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(ConnectionError) as caught:
        fake("model")
    assert len(calls) == 2
    assert "connection reset" in str(caught.value)
    # The offline attempt is kept as the cause rather than thrown away.
    assert isinstance(caught.value.__cause__, AttributeError)


def test_the_surfaced_error_is_tagged_so_an_outer_wrapper_does_not_retry(monkeypatch):
    """Stacked loaders must not reload twice more, so the tag has to travel on
    whichever exception actually leaves."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def inner(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("network down")

    @L._offline_aware_load
    def outer(*args, **kwargs):
        return inner(*args, **kwargs)

    with pytest.raises(ConnectionError) as caught:
        outer("model")
    assert getattr(caught.value, "_unsloth_offline_retried", False)
    assert len(calls) == 2, "the outer wrapper retried an already-retried load"


def test_an_out_of_memory_retry_reports_itself(monkeypatch):
    """A large VLM can exhaust memory on the retry's second load. That says nothing
    about the network, so it must not be replaced by the network error."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("network down")
        raise MemoryError("CUDA out of memory")

    with pytest.raises(MemoryError):
        fake("model")


def test_a_wrapped_out_of_memory_retry_is_recognised(monkeypatch):
    """Loaders re-raise through their own error types, so the OOM is usually a
    cause rather than the exception itself."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("network down")
        try:
            raise MemoryError("CUDA out of memory")
        except MemoryError as oom:
            raise RuntimeError("failed to place weights") from oom

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert isinstance(caught.value.__cause__, MemoryError)


def test_a_successful_retry_is_unchanged(monkeypatch):
    """The point of the retry: a cached model still loads after the network drops."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("network down")
        return "loaded from cache"

    assert fake("model") == "loaded from cache"
    assert L._force_offline_depth == 0


# ---------------------------------------------------------------------------
# what the retry must not hold, hide, or overwrite
# ---------------------------------------------------------------------------


def test_the_failed_attempt_is_not_pinned_by_the_error_it_raised(monkeypatch):
    """Holding the online error holds its frames, and its frames hold the partial
    model. The retry then reallocates alongside it and a large VLM runs out of
    memory doing exactly what the retry exists to do."""
    import gc
    import weakref

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    class _PartialModel:
        pass

    witness = {}
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            partial = _PartialModel()
            witness["ref"] = weakref.ref(partial)
            raise ConnectionError("connection reset while downloading model.safetensors")
        witness["alive_during_retry"] = witness["ref"]() is not None
        return "loaded"

    monkeypatch.setattr(gc, "collect", lambda *a, **k: 0)
    assert fake("model") == "loaded"
    assert (
        witness["alive_during_retry"] is False
    ), "the first attempt's model was still reachable while the retry reloaded"


def test_a_real_retry_failure_is_not_replaced_by_the_network_error(monkeypatch):
    """A corrupt checkpoint found offline is actionable and must reach the user.
    Reporting the earlier connection error instead sends them to fix their wifi."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("network down")
        raise ValueError("checkpoint header is corrupt")

    with pytest.raises(ValueError) as caught:
        fake("model")
    assert "corrupt" in str(caught.value)


def test_an_oom_spelled_as_a_bare_runtimeerror_still_surfaces(monkeypatch):
    """accelerate re-raises an accelerator OOM as a plain RuntimeError, and XPU has
    its own class. Selecting on the empty-cache artifact rather than on a list of
    OOM spellings covers both without naming either."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("network down")
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert "out of memory" in str(caught.value)


def test_a_network_error_wrapped_in_a_runtimeerror_keeps_its_cause(monkeypatch):
    """`FastModel.from_pretrained` raises a RuntimeError FROM the connection error,
    so the chain is the only thing that makes it classifiable. Overwriting the cause
    with the retry's failure loses that."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            try:
                raise ConnectionError("connection reset")
            except ConnectionError as connection_error:
                raise RuntimeError("could not load model") from connection_error
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert isinstance(
        caught.value.__cause__, ConnectionError
    ), "the retry replaced the connection error that made this classifiable"
    assert L._is_offline_related_error(
        caught.value
    ), "the surfaced error is no longer recognisable as network-related"


def test_a_wrapped_online_error_does_not_pin_the_failed_attempt(monkeypatch):
    """Same pinning as above, but through the wrapper path the retry already supports:
    the loader re-raises the connection error as its own type, so the partial model is
    held by the CAUSE's traceback, not the top-level one."""
    import gc
    import weakref

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    class _PartialModel:
        pass

    def _download():
        raise ConnectionError("connection reset while downloading model.safetensors")

    witness = {}
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            partial = _PartialModel()
            witness["ref"] = weakref.ref(partial)
            try:
                _download()
            except ConnectionError as connection_error:
                raise RuntimeError("could not load model") from connection_error
        witness["alive_during_retry"] = witness["ref"]() is not None
        return "loaded"

    monkeypatch.setattr(gc, "collect", lambda *a, **k: 0)
    assert fake("model") == "loaded"
    assert (
        witness["alive_during_retry"] is False
    ), "the wrapper's cause still held the first attempt's model while the retry reloaded"


def test_an_implicitly_chained_network_error_stays_recognisable(monkeypatch):
    """Loaders also chain implicitly (`raise RuntimeError(...)` inside an except, no
    `from`), so the network error is in `__context__`. A raise inside the retry's
    except block overwrites `__context__`, so the surfacing raise has to happen
    outside it."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            try:
                raise ConnectionError("connection reset")
            except ConnectionError:
                raise RuntimeError("could not load model")  # implicit chaining
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert L._is_offline_related_error(
        caught.value
    ), "the retry replaced the implicitly chained connection error that made this classifiable"
    # The connection error keeps the slot the traceback prints, and the retry is reported alongside it rather than in
    # place of it.
    assert isinstance(caught.value.__context__, ConnectionError)
    assert isinstance(caught.value._unsloth_offline_retry_error, AttributeError)


def test_the_retry_is_reported_even_when_the_online_error_has_an_explicit_cause(monkeypatch):
    """`FastModel.from_pretrained` raises `RuntimeError(...) from _cause` for a failed
    config probe (loader.py:756). Python prints a cause INSTEAD of a context, so hanging
    the retry off `__context__` there shows the user nothing about the cache attempt."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("could not load model") from ConnectionError("connection reset")
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert isinstance(caught.value.__cause__, ConnectionError)
    assert isinstance(caught.value._unsloth_offline_retry_error, AttributeError)
    if hasattr(caught.value, "add_note"):  # notes are 3.11+
        rendered = "".join(
            traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
        )
        assert "retrying from the local cache also failed" in rendered
        assert "'NoneType' object has no attribute 'endswith'" in rendered


def test_a_context_the_loader_suppressed_is_not_promoted_to_a_cause(monkeypatch):
    """`raise ... from None` keeps `__context__` but sets `__suppress_context__` to keep
    it out of the traceback. Copying it into `__cause__` publishes what the loader chose
    to hide."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            try:
                raise ConnectionError("connection reset from 10.0.0.1 with token hf_xxx")
            except ConnectionError:
                raise RuntimeError("could not load model") from None
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(RuntimeError) as caught:
        fake("model")
    assert caught.value.__cause__ is None, "the suppressed context was promoted to a cause"
    assert caught.value.__suppress_context__ is True
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert "10.0.0.1" not in rendered


@pytest.mark.parametrize(
    "artifact",
    [
        # A missing vocabulary resolves to None and is then dereferenced, opened or stat'd, so the family spans four
        # exception types (#7845).
        AttributeError("'NoneType' object has no attribute 'readlines'"),
        TypeError(
            "argument should be a str or an os.PathLike object where __fspath__ "
            "returns a str, not 'NoneType'"
        ),
        TypeError("expected str, bytes or os.PathLike object, not NoneType"),
        TypeError("stat: path should be string, bytes, os.PathLike or integer, not NoneType"),
        ValueError("Can't find a vocabulary file at path 'None'."),
    ],
)
def test_a_tokenizer_cache_miss_also_surfaces_the_network_error(monkeypatch, artifact):
    """The retry loads the tokenizer and processor too, and an empty cache there is just
    as opaque as it is for the weights."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("connection reset while downloading tokenizer.json")
        raise artifact

    with pytest.raises(ConnectionError) as caught:
        fake("model")
    assert "connection reset" in str(caught.value)
    assert caught.value.__cause__ is artifact


def test_the_same_wording_about_a_real_path_is_not_a_cache_miss():
    """The TypeError spellings are generic, so they must stay gated on the None: a real
    path in that message is a caller bug and has to reach the user unchanged."""
    assert (
        L._empty_cache_artifact(
            TypeError("stat: path should be string, bytes, os.PathLike or integer, not int")
        )
        is False
    )


def test_the_vlm_tokenizer_fallback_does_not_pin_the_built_model(monkeypatch):
    """The real path this matters on (vision.py:1683-1708): the model is already built,
    `patch_tokenizer` fails, the AutoTokenizer fallback then hits the network, and the
    network error is re-raised with the patch failure implicitly chained onto it. The
    patch failure's traceback is the frame holding the model, so clearing only the
    network error's own traceback leaves the whole model allocated for the retry."""
    import gc
    import weakref

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    class _BuiltModel:
        pass

    def _patch_tokenizer(model, tok):
        raise TypeError("Unsloth: this VLM processor cannot be patched")

    def _fallback_auto_tokenizer():
        raise ConnectionError("Max retries exceeded with url: /api/models")

    witness = {}
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            model = _BuiltModel()  # already allocated by the weight load
            witness["ref"] = weakref.ref(model)
            try:
                model, tokenizer = _patch_tokenizer(model, object())
            except Exception as patch_error:
                try:
                    _fallback_auto_tokenizer()
                except Exception as fallback_error:
                    if L._is_offline_related_error(fallback_error):
                        raise
                    raise patch_error
        witness["alive_during_retry"] = witness["ref"]() is not None
        return "loaded"

    monkeypatch.setattr(gc, "collect", lambda *a, **k: 0)
    assert fake("model") == "loaded"
    assert (
        witness["alive_during_retry"] is False
    ), "the implicitly chained patch failure still held the built model during the retry"


def test_the_surfaced_online_error_still_names_where_it_failed(monkeypatch):
    """Freeing the failed attempt's memory must not cost the user the origin of the
    network failure: with the traceback detached, the report says only that the
    decorator re-raised something, which is useless for a failure raised deep inside
    `trust_remote_code`."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    calls = []

    def _resolve_config_from_the_hub():
        raise ConnectionError("Max retries exceeded with url: /api/models")

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            _resolve_config_from_the_hub()
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    with pytest.raises(ConnectionError) as caught:
        fake("model")
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert "_resolve_config_from_the_hub" in rendered, rendered


def test_the_retrys_own_frames_do_not_pin_the_cached_model(monkeypatch):
    """The retry can load the whole model from the cache and only then trip over a
    missing tokenizer file. That error is kept on the surfaced one, so its frames hold
    the cached model for as long as the caller holds the error, and nothing collects
    after this point."""
    import gc
    import weakref

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    class _CachedModel:
        pass

    witness = {}
    calls = []

    @L._offline_aware_load
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("connection reset while downloading tokenizer.json")
        model = _CachedModel()  # the retry got the weights, then found no tokenizer
        witness["ref"] = weakref.ref(model)
        raise AttributeError("'NoneType' object has no attribute 'endswith'")

    monkeypatch.setattr(gc, "collect", lambda *a, **k: 0)
    with pytest.raises(ConnectionError) as caught:
        fake("model")
    assert caught.value.__cause__ is not None
    assert witness["ref"]() is None, "the retry error's traceback still held the cached model"
