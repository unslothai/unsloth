# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A gated model must say how to get in, not paste the 403 back at the user.

The load error is surfaced verbatim in a toast, where a raw GatedRepoError is a
request id and a resolve URL wrapped around one useful sentence.
"""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.diffusion import _hf_token_in_play, hub_access_message

_GATED = (
    "403 Client Error. (Request ID: Root=1-6a73b83b) Cannot access gated repo for url "
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/model_index.json. "
    "Access to model black-forest-labs/FLUX.2-klein-9B is restricted and you are not in "
    "the authorized list."
)


def _gated(text = _GATED):
    """A real GatedRepoError by type, without hub's constructor.

    HfHubHTTPError.__init__ takes a required response on hub 1.x and not on 0.x,
    and the pin spans both. The helper screens on isinstance and str(), so a
    subclass that only carries the message pins the contract on either version.
    """
    from huggingface_hub.errors import GatedRepoError

    class _Gated(GatedRepoError):
        def __init__(self, message):
            Exception.__init__(self, message)

    return _Gated(text)


def test_no_token_asks_for_access_and_a_token():
    message = hub_access_message(_gated(), had_token = False)

    assert message is not None
    assert "black-forest-labs/FLUX.2-klein-9B is gated" in message
    assert "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" in message
    assert "token" in message
    # The resolve URL and request id are the noise this replaces.
    assert "model_index.json" not in message
    assert "Request ID" not in message


def test_a_token_that_still_bounces_names_the_account():
    message = hub_access_message(_gated(), had_token = True)

    assert message is not None
    assert "not on its access list" in message
    # Telling someone with a token to add a token sends them in a circle.
    assert "add a Hugging Face token" not in message


def test_a_metadata_api_url_names_the_model_not_the_endpoint():
    """HfApi.model_info is the first Hub call a load makes, and its 403 carries
    /api/models/<owner>/<repo>, the shape huggingface_hub's own GatedRepoError docstring
    shows. A two-segment match on that names "api/models" as the gated repo."""
    message = hub_access_message(
        _gated(
            "403 Client Error. (Request ID: ViT1Bf7O) Cannot access gated repo for url "
            "https://huggingface.co/api/models/ardent-figment/gated-model."
        ),
        had_token = False,
    )

    assert message is not None
    assert "ardent-figment/gated-model is gated" in message
    assert "https://huggingface.co/ardent-figment/gated-model" in message
    assert "api/models" not in message


def test_a_non_repo_api_url_falls_back_rather_than_inventing_a_repo():
    message = hub_access_message(
        _gated(
            "403 Client Error. Cannot access gated repo for url https://huggingface.co/api/whoami-v2."
        ),
        had_token = False,
    )

    assert message is not None
    assert "its Hugging Face page" in message
    assert "api/" not in message


def test_an_unparseable_repo_still_gives_the_instruction():
    message = hub_access_message(
        _gated("403 Client Error. Cannot access gated repo."), had_token = False
    )

    assert message is not None
    assert "its Hugging Face page" in message


@pytest.mark.parametrize(
    "exc",
    [
        OSError("No space left on device"),
        ValueError("Could not decode image"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_other_failures_keep_their_own_text(exc):
    # None is the signal to fall back to str(exc); rewriting these would bury the cause.
    assert hub_access_message(exc, had_token = False) is None


def test_a_wrapped_gated_error_is_still_rewritten():
    """Transformers config/tokenizer loads re-raise the 403 inside an OSError, so matching only
    the outermost exception misses the shape this rewrite exists for."""
    try:
        try:
            raise _gated()
        except Exception as inner:
            raise OSError("We couldn't connect to huggingface.co to load this model.") from inner
    except OSError as outer:
        message = hub_access_message(outer, had_token = False)

    assert message is not None
    assert "black-forest-labs/FLUX.2-klein-9B is gated" in message


def test_a_self_referential_chain_terminates():
    exc = ValueError("boom")
    exc.__context__ = exc

    assert hub_access_message(exc, had_token = False) is None


def test_an_ambient_token_counts_as_a_token(monkeypatch):
    """With token=None the Hub still uses HF_TOKEN or the cached login, so keying off Studio's
    own token alone tells an already-authenticated user to add a token they have."""
    import huggingface_hub.utils as hub_utils

    monkeypatch.setattr(hub_utils, "get_token_to_send", lambda _t: "hf_ambient")
    assert _hf_token_in_play(None) is True

    monkeypatch.setattr(hub_utils, "get_token_to_send", lambda _t: None)
    assert _hf_token_in_play(None) is False
    assert _hf_token_in_play("hf_explicit") is True


def test_a_disabled_implicit_token_is_not_a_token(monkeypatch):
    """HF_HUB_DISABLE_IMPLICIT_TOKEN leaves get_token() answering with the cached login while
    build_hf_headers sends no authorization header, so the refusal was anonymous. Asking the
    same helper the Hub asks is what keeps the two in step."""
    from huggingface_hub import constants
    from huggingface_hub.utils import _headers

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(_headers, "get_token", lambda: "hf_cached_login", raising = False)

    # Real get_token_to_send, so this pins hub's actual policy rather than a stand-in.
    assert _hf_token_in_play(None) is False
    assert _hf_token_in_play("hf_explicit") is True

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    assert _hf_token_in_play(None) is True


def test_an_unreadable_ambient_token_is_not_a_token(monkeypatch):
    import huggingface_hub.utils as hub_utils

    def _raise(_t):
        raise OSError("token file unreadable")

    monkeypatch.setattr(hub_utils, "get_token_to_send", _raise)
    assert _hf_token_in_play(None) is False
