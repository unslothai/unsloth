"""HF auth on the llama.cpp prebuilt installer: auth_headers sends HF_TOKEN to huggingface.co only, and a redirect handler strips Authorization on cross-host redirects. Offline."""

import importlib.util
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt_hf_auth", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_TOKEN_VARS = ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HF_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
GH_URL = "https://api.github.com/repos/unslothai/llama.cpp/releases"


def _headers(url, env):
    """Call auth_headers under a fully controlled token environment."""
    with patch.dict(mod.os.environ, env, clear = False):
        for var in _TOKEN_VARS:
            if var not in env:
                mod.os.environ.pop(var, None)
        return mod.auth_headers(url)


class TestAuthHeaderRouting:
    def test_hf_token_sent_to_huggingface(self):
        headers = _headers(HF_URL, {"HF_TOKEN": "hf_x"})
        assert headers.get("Authorization") == "Bearer hf_x"

    def test_hub_token_fallback(self):
        headers = _headers(HF_URL, {"HUGGING_FACE_HUB_TOKEN": "hf_y"})
        assert headers.get("Authorization") == "Bearer hf_y"

    def test_hf_token_not_sent_to_github(self):
        headers = _headers(GH_URL, {"HF_TOKEN": "hf_x"})
        assert "Authorization" not in headers

    def test_hf_token_not_sent_to_other_hosts(self):
        headers = _headers("https://cdn-lfs.huggingface.co/x", {"HF_TOKEN": "hf_x"})
        assert "Authorization" not in headers

    def test_gh_token_not_sent_to_huggingface(self):
        headers = _headers(HF_URL, {"GH_TOKEN": "gh_x"})
        assert "Authorization" not in headers

    def test_gh_token_still_wins_on_github(self):
        headers = _headers(GH_URL, {"GH_TOKEN": "gh_x", "HF_TOKEN": "hf_x"})
        assert headers.get("Authorization") == "Bearer gh_x"

    def test_no_tokens_no_auth(self):
        assert "Authorization" not in _headers(HF_URL, {})

    def test_validation_model_url_is_hf(self):
        assert mod.should_send_hf_auth(mod.TEST_MODEL_URL) is True


class TestCrossHostRedirectStripsAuth:
    def _redirect(self, newurl):
        req = urllib.request.Request(HF_URL, headers = {"Authorization": "Bearer hf_x"})
        handler = mod._CrossHostAuthStrippingRedirectHandler()
        return handler.redirect_request(req, None, 302, "Found", {}, newurl)

    def test_cross_host_redirect_drops_authorization(self):
        new_request = self._redirect("https://cdn-lfs.huggingface.co/signed/blob")
        assert new_request is not None
        assert "Authorization" not in new_request.headers
        assert "Authorization" not in new_request.unredirected_hdrs

    def test_same_host_redirect_keeps_authorization(self):
        new_request = self._redirect("https://huggingface.co/elsewhere/blob")
        assert new_request is not None
        assert new_request.headers.get("Authorization") == "Bearer hf_x"


class TestDownloadBytesWiring:
    def test_download_bytes_sends_hf_auth(self):
        response = MagicMock()
        response.__enter__ = lambda s: s
        response.__exit__ = lambda s, *a: False
        response.headers.get.return_value = None
        response.read.side_effect = [b"data", b""]
        with (
            patch.object(mod._URL_OPENER, "open", return_value = response) as opened,
            patch.dict(mod.os.environ, {"HF_TOKEN": "hf_x"}, clear = False),
        ):
            for var in ("GH_TOKEN", "GITHUB_TOKEN"):
                mod.os.environ.pop(var, None)
            data = mod.download_bytes(HF_URL)
        assert data == b"data"
        request = opened.call_args.args[0]
        assert request.headers.get("Authorization") == "Bearer hf_x"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
