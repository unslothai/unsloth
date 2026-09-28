# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A token Hugging Face refuses must not make public repos unreadable (#11551).

An expired or revoked credential (an ``hf_oauth_`` token in the report) is answered 401 on
every read, public repos included, and huggingface_hub reports that as
RepositoryNotFoundError. A token the Hub accepts gets 404 for a repo it cannot see, so a 401
on a read that carried one is the credential being refused: retry that read once
anonymously, and say so.
"""

import re
from types import SimpleNamespace

import huggingface_hub
import pytest

import core.inference.llama_cpp as llama_cpp
import hub.utils.hf_tokens as hf_tokens
import utils.hf_cache_settings as hf_cache_settings
import utils.models.model_config as mc
from hub.utils.hf_tokens import (
    HUB_TOKEN_REJECTED_WARNING,
    call_with_anonymous_retry,
    collecting_hub_token_rejections,
    is_rejected_credential_error,
)
from utils.models.model_config import (
    GgufRepoUnreadableError,
    GgufVariantInfo,
    ModelConfig,
    _hub_model_info,
    shared_hub_model_info,
)
from utils.utils import is_hf_authentication_error

REPO = "unsloth/Qwen3-0.6B-GGUF"
GGUF = "Qwen3-0.6B-UD-Q4_K_XL.gguf"
OAUTH = "hf_oauth_" + "a" * 40


class RepositoryNotFoundError(Exception):
    """Matched by type name, as the helpers do."""

    def __init__(
        self,
        message = "401 Client Error",
        status_code = 401,
        reason = None,
    ):
        super().__init__(message)
        headers = {"X-Error-Message": reason} if reason else {}
        self.response = SimpleNamespace(status_code = status_code, headers = headers)


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"{status_code} Client Error")
        self.response = SimpleNamespace(status_code = status_code, headers = {})


def _refused_with_a_token(calls):
    """The Hub's answer to a rejected OAuth token: 401 for any read that carries it."""

    def read(token):
        calls.append(token)
        if token is False:
            return "public answer"
        raise RepositoryNotFoundError(reason = "OAuth token verification failed")

    return read


@pytest.fixture(autouse = True)
def _no_ambient_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: None)


def test_a_refused_token_is_retried_once_anonymously():
    calls = []
    with collecting_hub_token_rejections() as rejections:
        assert call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH) == "public answer"
    assert calls == [OAUTH, False]
    assert rejections.recovered and not rejections.refused


def test_the_next_read_in_the_same_request_skips_the_refused_token():
    calls = []
    with collecting_hub_token_rejections():
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
    assert calls == [OAUTH, False, False]


def test_a_new_request_asks_with_the_token_again():
    calls = []
    with collecting_hub_token_rejections():
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
    with collecting_hub_token_rejections():
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
    assert calls == [OAUTH, False, OAUTH, False]


def test_another_credential_in_the_request_is_not_skipped():
    calls = []
    other = "hf_" + "b" * 34
    with collecting_hub_token_rejections():
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
        call_with_anonymous_retry(_refused_with_a_token(calls), other)
    assert calls == [OAUTH, False, other, False]


def test_an_accepted_token_reads_once():
    calls = []

    def read(token):
        calls.append(token)
        return "private answer"

    with collecting_hub_token_rejections() as rejections:
        assert call_with_anonymous_retry(read, OAUTH) == "private answer"
    assert calls == [OAUTH]
    assert not rejections.rejected


def test_an_anonymous_401_is_the_repos_answer_and_is_not_retried():
    calls = []

    def read(token):
        calls.append(token)
        raise RepositoryNotFoundError()

    with pytest.raises(RepositoryNotFoundError):
        call_with_anonymous_retry(read, False)
    assert calls == [False]


def test_no_credential_at_all_is_not_retried():
    calls = []

    def read(token):
        calls.append(token)
        raise RepositoryNotFoundError()

    with pytest.raises(RepositoryNotFoundError):
        call_with_anonymous_retry(read, None)
    assert calls == [None]


def test_an_ambient_credential_is_retried(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OAUTH))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: OAUTH)
    calls = []

    def read(token):
        calls.append(token)
        if token is None:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return "public answer"

    assert call_with_anonymous_retry(read, None) == "public answer"
    assert calls == [None, False]


@pytest.mark.parametrize(
    "failure",
    [HTTPError(403), HTTPError(404), HTTPError(429), TimeoutError("read timed out")],
    ids = ["403", "404", "429", "timeout"],
)
def test_other_failures_are_not_retried(failure):
    calls = []

    def read(token):
        calls.append(token)
        raise failure

    with pytest.raises(type(failure)):
        call_with_anonymous_retry(read, OAUTH)
    assert calls == [OAUTH]
    assert not is_rejected_credential_error(failure, OAUTH)


def test_a_private_repo_raises_the_original_error_and_notes_the_refusal():
    calls = []
    original = RepositoryNotFoundError(reason = "OAuth token verification failed")

    def read(token):
        calls.append(token)
        if token is False:
            raise RepositoryNotFoundError(reason = "Invalid username or password.")
        raise original

    with collecting_hub_token_rejections() as rejections:
        with pytest.raises(RepositoryNotFoundError) as exc_info:
            call_with_anonymous_retry(read, OAUTH)
    assert exc_info.value is original
    assert calls == [OAUTH, False]
    assert rejections.refused and not rejections.recovered


def test_the_model_info_scope_keeps_each_credential_its_own(monkeypatch):
    calls = []

    def model_info(repo_id, **kwargs):
        token = kwargs.get("token")
        calls.append(token)
        if token == OAUTH:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return SimpleNamespace(siblings = [], answered_for = token)

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    other = "hf_" + "b" * 34
    with shared_hub_model_info(), collecting_hub_token_rejections():
        assert _hub_model_info(REPO, OAUTH).answered_for is False
        assert _hub_model_info(REPO, OAUTH).answered_for is False
        assert _hub_model_info(REPO, other).answered_for == other
    assert calls == [OAUTH, False, other]


# End to end through ModelConfig.from_identifier, the resolver /validate and /load call.


@pytest.fixture
def _gguf_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path)
    )
    monkeypatch.setattr(mc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: "/fake/llama-server"),
    )
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kwargs: None)
    monkeypatch.setattr(
        mc,
        "list_gguf_variants",
        lambda identifier, hf_token = None: (
            [GgufVariantInfo(filename = GGUF, quant = "UD-Q4_K_XL", size_bytes = 1)],
            False,
        ),
    )
    monkeypatch.setattr(mc, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mc, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mc, "is_model_cached", lambda *a, **k: False)
    return tmp_path


def _hub_refusing_the_token(calls, *, public = True):
    def model_info(repo_id, **kwargs):
        token = kwargs.get("token")
        calls.append(token)
        if token is False and public:
            return SimpleNamespace(siblings = [SimpleNamespace(rfilename = GGUF)])
        reason = "OAuth token verification failed" if token else "Invalid username or password."
        raise RepositoryNotFoundError(reason = reason)

    return model_info


def test_a_public_gguf_repo_loads_despite_a_refused_token(monkeypatch, _gguf_env):
    calls = []
    monkeypatch.setattr(huggingface_hub, "model_info", _hub_refusing_the_token(calls))
    with collecting_hub_token_rejections() as rejections:
        config = ModelConfig.from_identifier(REPO, hf_token = OAUTH)
    assert config.is_gguf and config.gguf_hf_repo == REPO
    assert calls[:2] == [OAUTH, False]
    assert OAUTH not in calls[2:]
    assert rejections.recovered


def test_a_repo_no_one_can_read_names_the_refused_token(monkeypatch, _gguf_env):
    snapshot = _gguf_env / f"models--{REPO.replace('/', '--')}" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / GGUF).write_bytes(b"cached")
    calls = []
    monkeypatch.setattr(huggingface_hub, "model_info", _hub_refusing_the_token(calls, public = False))
    with pytest.raises(GgufRepoUnreadableError) as exc_info:
        ModelConfig.from_identifier(REPO, hf_token = OAUTH)
    msg = str(exc_info.value)
    assert "rejected the saved token" in msg and "Settings" in msg
    assert "HTTP 401" in msg
    assert calls == [OAUTH, False]


def test_hub_gguf_listing_retries_anonymously(monkeypatch):
    from hub.utils import gguf as hub_gguf

    calls = []

    class FakeApi:
        def __init__(self, token = None):
            self.token = token

        def model_info(self, repo_id, **kwargs):
            calls.append(self.token)
            if self.token is not False:
                raise RepositoryNotFoundError(reason = "OAuth token verification failed")
            return SimpleNamespace(siblings = [SimpleNamespace(rfilename = GGUF, size = 1)])

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub_gguf, "_env_offline", lambda: False)
    variants, _has_vision, _siblings = hub_gguf.list_gguf_variants(REPO, hf_token = OAUTH)
    assert [v.filename for v in variants] == [GGUF]
    assert calls == [OAUTH, False]


# Error classification and redaction.


def test_a_repo_level_401_is_not_reported_as_a_bad_token():
    assert not is_hf_authentication_error(
        RepositoryNotFoundError(reason = "Invalid username or password.")
    )


def test_a_refused_token_is_reported_as_one():
    assert is_hf_authentication_error(
        RepositoryNotFoundError(reason = "OAuth token verification failed")
    )
    with collecting_hub_token_rejections() as rejections:
        rejections.refused = True
        assert is_hf_authentication_error(RepositoryNotFoundError())


def test_a_plain_401_still_reads_as_authentication():
    assert is_hf_authentication_error(HTTPError(401))


def test_the_load_response_carries_the_warning():
    from models.inference import LoadResponse
    from routes.inference import _with_token_rejected_warning

    response = LoadResponse(status = "loaded", model = REPO, display_name = REPO, inference = {})
    with collecting_hub_token_rejections() as rejections:
        assert _with_token_rejected_warning(response, rejections) is response
        rejections.recovered = True
        warned = _with_token_rejected_warning(response, rejections)
    assert warned.memory_warning == HUB_TOKEN_REJECTED_WARNING


JWS = "hf_oauth_eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl"


def test_oauth_tokens_are_redacted_whole():
    from core.inference.llama_cpp import LlamaCppBackend
    from core.research import redaction as research_redaction
    from hub.utils.download_registry import scrub_secrets
    from utils.log_redaction import redact_log_text

    line = f"token={JWS} failed"
    assert "eyJ" not in redact_log_text(f"sent {JWS} to the Hub")
    assert "eyJ" not in scrub_secrets(line)
    assert any(p.fullmatch(JWS) for p in LlamaCppBackend._SECRET_VALUE_RES)
    assert research_redaction._QUERY_OPAQUE_TOKEN.search(JWS).group(0).startswith("hf_oauth_")
    assert re.search(r"hf_oauth_\S*eyJ", redact_log_text(line)) is None


def test_an_ambient_token_the_hub_never_sees_is_not_blamed(monkeypatch):
    # HF_HUB_DISABLE_IMPLICIT_TOKEN=1: huggingface_hub keeps the ambient token off the wire,
    # so a 401 is about the repo and the saved token must not be called rejected.
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OAUTH))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: OAUTH)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    calls = []
    with collecting_hub_token_rejections() as rejections:
        with pytest.raises(RepositoryNotFoundError):
            call_with_anonymous_retry(_refused_with_a_token(calls), None)
    assert calls == [None]
    assert not rejections.rejected


def test_the_remote_code_scan_reads_public_files_without_a_refused_token(monkeypatch):
    from hub.utils.hf_tokens import anonymous_retrying

    calls = []

    def hf_hub_download(
        repo_id,
        filename,
        *,
        token = None,
        cache_dir = None,
    ):
        calls.append(token)
        if token is False:
            return f"/cache/{repo_id}/{filename}"
        raise RepositoryNotFoundError(reason = "OAuth token verification failed")

    download = anonymous_retrying(hf_hub_download)
    assert download(REPO, "config.json", token = OAUTH, cache_dir = "x") == f"/cache/{REPO}/config.json"
    assert calls == [OAUTH, False]


def test_an_anonymous_download_401_does_not_blame_a_saved_token():
    exc = RepositoryNotFoundError()
    assert not is_rejected_credential_error(exc, False)
    assert is_rejected_credential_error(exc, OAUTH)


def test_an_alias_huggingface_hub_never_sends_is_not_blamed(monkeypatch):
    # HF_HUB_TOKEN and friends count for cache bookkeeping, but huggingface_hub's own
    # get_token() does not read them, so a token=None read went out anonymously.
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OAUTH))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: None)
    calls = []
    with pytest.raises(RepositoryNotFoundError):
        call_with_anonymous_retry(_refused_with_a_token(calls), None)
    assert calls == [None]


def test_a_cancelled_anonymous_retry_stays_a_cancellation():
    def read(token):
        if token is False:
            raise RuntimeError("Cancelled")
        raise RepositoryNotFoundError(reason = "OAuth token verification failed")

    with collecting_hub_token_rejections():
        with pytest.raises(RuntimeError, match = "Cancelled"):
            call_with_anonymous_retry(read, OAUTH)


def test_the_pre_import_config_reads_retry_without_a_refused_token(monkeypatch):
    # The worker picks its transformers tier from raw config/tokenizer reads before any
    # huggingface_hub import; a refused token there must not hide a public repo's config.
    import io
    import json
    import urllib.error

    import utils.transformers_version as tv

    sent = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout):
        auth = req.get_header("Authorization")
        sent.append(auth)
        if auth:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return _Resp(json.dumps({"model_type": "qwen3"}).encode())

    monkeypatch.setattr(tv, "_hf_urlopen", urlopen)
    assert tv._hf_json("https://huggingface.co/x/resolve/main/config.json", OAUTH) == {
        "model_type": "qwen3"
    }
    assert sent == [f"Bearer {OAUTH}", None]
    sent.clear()
    tv._hf_json("https://huggingface.co/x/resolve/main/config.json", None)
    assert sent == [None]


def test_an_anonymous_first_success_after_a_transient_failure_is_recorded():
    from hub.utils.hf_tokens import saved_token_rejected

    attempts = {"anonymous": 0}

    def read(token):
        if token is False:
            attempts["anonymous"] += 1
            if attempts["anonymous"] == 1:
                raise TimeoutError("read timed out")
            return "public answer"
        raise RepositoryNotFoundError(reason = "OAuth token verification failed")

    with collecting_hub_token_rejections() as rejections:
        with pytest.raises(RepositoryNotFoundError):
            call_with_anonymous_retry(read, OAUTH)
        assert not saved_token_rejected(OAUTH)
        # The caller's own retry: the anonymous-first path now answers.
        assert call_with_anonymous_retry(read, OAUTH) == "public answer"
        assert saved_token_rejected(OAUTH)
        assert rejections.recovered


def test_admission_sizing_counts_companions_when_the_token_is_refused(monkeypatch):
    # The training guard sizes a GGUF with its mmproj: a refused token must not zero it.
    import routes.inference as inference_routes

    def model_info(
        repo,
        token = None,
        files_metadata = False,
    ):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return SimpleNamespace(
            siblings = [
                SimpleNamespace(rfilename = "Qwen3-VL-2B-Q4_K_M.gguf", size = 1_000_000_000),
                SimpleNamespace(rfilename = "mmproj-F16.gguf", size = 800_000_000),
            ]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    sized = inference_routes._remote_gguf_companion_bytes(
        "unsloth/Qwen3-VL-2B-Instruct-GGUF", hf_token = OAUTH, include_mmproj = True
    )
    assert sized == 800_000_000


def test_training_admission_reads_safetensors_totals_without_a_refused_token(monkeypatch):
    # Multimodal sizing needs the safetensors total; the config estimate is the text tower only.
    from utils.hardware import hardware

    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: False)

    def model_info(repo, token = None):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return SimpleNamespace(safetensors = {"total": 4_400_000_000})

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    assert hardware._get_hf_safetensors_total_params("org/public-vlm", OAUTH) == 4_400_000_000


def test_the_malware_status_is_read_without_a_refused_token(monkeypatch):
    from utils.security import file_security

    status = {"filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}]}

    def model_info(
        repo,
        revision = None,
        token = None,
        securityStatus = False,
        timeout = None,
    ):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return SimpleNamespace(security_repo_status = status)

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    assert file_security._fetch_security_status("org/public-model", OAUTH) == status


def test_an_explicit_remote_drafter_is_sized_without_a_refused_token(monkeypatch):
    # The flat reserve undercharges a large public drafter beside a training run.
    import routes.inference as inference_routes

    def model_info(
        repo,
        token = None,
        files_metadata = False,
    ):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "drafter-Q8_0.gguf", size = 30 * 1024**3)]
        )

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    sized = inference_routes._remote_drafter_repo_bytes("org/big-drafter-GGUF", hf_token = OAUTH)
    assert sized == 30 * 1024**3


def test_security_weight_indexes_are_read_without_a_refused_token(monkeypatch, tmp_path):
    # An unread index is inconclusive, which blocks every flagged nested pickle.
    import json

    import utils.hf_probe
    from utils.security import file_security

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors"}}))

    def hf_hub_download(
        repo,
        filename,
        revision = None,
        token = None,
        cache_dir = None,
    ):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return str(index)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(
        utils.hf_probe,
        "hf_file_definitely_absent",
        lambda repo, filename, **k: filename != "model.safetensors.index.json",
    )
    paths = file_security._indexed_shard_paths("org/public-model", OAUTH)
    assert paths == {"model-00001-of-00002.safetensors"}


def test_the_gguf_sliding_window_config_is_read_without_a_refused_token(monkeypatch, tmp_path):
    # Without it, context sizing falls back to the one-global-in-four guess.
    import json

    import utils.hf_probe
    from core.inference import llama_cpp

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"sliding_window_pattern": 6}))

    def hf_hub_download(
        repo,
        filename,
        repo_type = None,
        token = None,
        cache_dir = None,
    ):
        if token is not False:
            raise RepositoryNotFoundError(reason = "OAuth token verification failed")
        return str(cfg)

    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OAUTH))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: OAUTH)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(utils.hf_probe, "hf_file_definitely_absent", lambda *a, **k: False)
    assert llama_cpp._fetch_swa_entry_from_hf("org/public-model") == 6


def test_a_private_repo_after_a_public_recovery_is_reported_as_refused():
    # A public adapter recovered anonymously, then a private base no one lets in: /load must
    # see the refusal and name the token, not answer 500.
    calls = []

    def private(token):
        calls.append(token)
        raise RepositoryNotFoundError(reason = "OAuth token verification failed")

    with collecting_hub_token_rejections() as rejections:
        call_with_anonymous_retry(_refused_with_a_token(calls), OAUTH)
        with pytest.raises(RepositoryNotFoundError):
            call_with_anonymous_retry(private, OAUTH)
    assert calls == [OAUTH, False, False, OAUTH]
    assert rejections.refused
