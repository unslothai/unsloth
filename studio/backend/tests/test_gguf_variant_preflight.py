"""An explicit --gguf-variant that exists nowhere must be rejected by
ModelConfig.from_identifier, before the load path unloads the resident model."""

import huggingface_hub
import pytest

import core.inference.llama_cpp as llama_cpp
import utils.models.model_config as mc
from utils.models.model_config import GgufVariantInfo, ModelConfig

REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"

REPO_FILES = [
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
]

VARIANTS = [
    GgufVariantInfo(filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 100),
    GgufVariantInfo(filename = "Llama-3.2-1B-Instruct-Q8_0.gguf", quant = "Q8_0", size_bytes = 200),
]


@pytest.fixture
def remote_gguf_repo(monkeypatch):
    monkeypatch.setattr(
        mc, "detect_gguf_model_remote", lambda identifier, hf_token = None: VARIANTS[0].filename
    )
    monkeypatch.setattr(
        mc, "list_gguf_variants", lambda identifier, hf_token = None: (list(VARIANTS), False)
    )
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo_id, token = None: list(REPO_FILES)
    )
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: "/fake/llama-server"),
    )
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda repo, variant, **kwargs: None)


def test_unknown_variant_rejected_before_load(remote_gguf_repo):
    with pytest.raises(ValueError) as exc_info:
        ModelConfig.from_identifier(REPO, gguf_variant = "NOPE_Q9")
    msg = str(exc_info.value)
    assert "NOPE_Q9" in msg
    assert "Q4_K_M" in msg and "Q8_0" in msg


def test_known_variant_accepted(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO, gguf_variant = "Q8_0")
    assert config is not None
    assert config.is_gguf
    assert config.gguf_variant == "Q8_0"


def test_variant_match_is_case_insensitive(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO, gguf_variant = "q4_k_m")
    assert config is not None
    assert config.gguf_variant == "q4_k_m"


def test_variant_only_in_verified_local_cache_accepted(remote_gguf_repo, monkeypatch):
    monkeypatch.setattr(
        llama_cpp,
        "cached_gguf_for_load",
        lambda repo, variant, **kwargs: "/cache/Llama-3.2-1B-Instruct-OLD_Q2.gguf",
    )
    config = ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert config is not None
    assert config.gguf_variant == "OLD_Q2"


def test_cache_escape_uses_size_verified_predicate(remote_gguf_repo, monkeypatch):
    calls = {}

    def fake_cached_gguf_for_load(repo, variant, **kwargs):
        calls["repo"] = repo
        calls["kwargs"] = kwargs
        return None

    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", fake_cached_gguf_for_load)
    with pytest.raises(ValueError):
        ModelConfig.from_identifier(REPO, gguf_variant = "OLD_Q2")
    assert calls["repo"] == REPO
    assert calls["kwargs"].get("verify_sizes") is True


def test_variant_matching_uncollapsed_repo_file_accepted(remote_gguf_repo, monkeypatch):
    # Same quant label across shards collapses to one GgufVariantInfo; the
    # preflight must still match against every repo file, like the load path.
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, token = None: [
            "Llama-3.2-1B-Instruct-Q4_K_M-00001-of-00002.gguf",
            "Llama-3.2-1B-Instruct-Q4_K_M-00002-of-00002.gguf",
        ],
    )
    config = ModelConfig.from_identifier(REPO, gguf_variant = "00002-of-00002")
    assert config is not None
    assert config.gguf_variant == "00002-of-00002"


def test_listing_unavailable_skips_check(remote_gguf_repo, monkeypatch):
    def boom(repo_id, token = None):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", boom)
    config = ModelConfig.from_identifier(REPO, gguf_variant = "NOPE_Q9")
    assert config is not None
    assert config.gguf_variant == "NOPE_Q9"


def test_no_variant_still_autoselects(remote_gguf_repo):
    config = ModelConfig.from_identifier(REPO)
    assert config is not None
    assert config.gguf_variant in {"Q4_K_M", "Q8_0"}
