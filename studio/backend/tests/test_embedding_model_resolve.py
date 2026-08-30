# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The picker asks what saving a model would need fetched before it saves, so the
download can be offered up front instead of happening invisibly at the first index.

The endpoint must agree with the loader about which repo and which file that is,
and with the PUT about what is unusable here."""

from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "_local_gguf_backend_error", lambda m: None)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def _resolve(c, model = "unsloth/bge-small-en-v1.5"):
    return c.get("/embedding-model/resolve", params = {"model": model})


def test_sentence_transformers_backend_points_at_the_model_repo(client, monkeypatch):
    """No GGUF involved: ST loads the model repo itself, and the cache check is the
    ordinary snapshot probe."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: True)
    # Cached now also means "holds a checkpoint ST can open", not just "loadable",
    # and names the repo it was filed under: for an exact id, that id.
    monkeypatch.setattr(
        settings, "_cached_st_source", lambda m: ("unsloth/bge-small-en-v1.5", Path("/snap"))
    )

    body = _resolve(client).json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] == "unsloth/bge-small-en-v1.5"
    assert body["cached"] is True
    assert body["files"] is None
    assert body["error"] is None


def test_uncached_sentence_transformers_plan_reports_snapshot_size(client, monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_hf_snapshot_size", lambda repo, token: 987_654)
    # The repo has to publish something loadable before a size means anything.
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: ["model.safetensors"])
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    body = _resolve(client, "org/uncached-st").json()
    assert body["backend"] == "sentence-transformers"
    assert body["cached"] is False
    assert body["size_bytes"] == 987_654


def test_a_repo_with_no_loadable_checkpoint_is_refused_not_offered(client, monkeypatch):
    """A feature-extraction repo publishing only GGUF passes the tag gate and
    reaches the direct ST branch, where it would be offered as a download ST
    cannot open."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    body = _resolve(client, "org/gguf-only").json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] is None
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]


def test_resolution_selects_the_backend_for_the_new_model_not_the_old_one(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        settings,
        "_llama_backend_active",
        lambda model = None: seen.append(model) or False,
    )
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda model: True)
    body = _resolve(client, "org/new-model").json()

    assert seen == ["org/new-model"]
    assert body["backend"] == "sentence-transformers"


def test_runtime_st_failure_is_planned_as_a_managed_gguf_download(client, monkeypatch):
    from core.rag import embeddings

    monkeypatch.setattr(embeddings.config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_forced_backends", {})
    monkeypatch.setattr(
        embeddings,
        "_resolve_auto_for_model",
        lambda model = None: "sentence-transformers",
    )
    monkeypatch.setattr(embeddings, "sentence_transformers_runtime_available", lambda: False)
    monkeypatch.setattr(embeddings, "_llama_server_runtime_available", lambda: True)
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    monkeypatch.setattr(
        settings,
        "_remote_embedding_gguf_plan",
        lambda candidates, token: (candidates[0], ["embed-F16.gguf"]),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: 1234)

    body = _resolve(client, "org/embed").json()

    assert body["backend"] == "llama"
    assert body["download_repo"] == "org/embed-GGUF"
    assert body["files"] == ["embed-F16.gguf"]
    assert body["size_bytes"] == 1234


def test_sentence_transformers_local_path_is_already_present(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    local = tmp_path / "embedder"
    local.mkdir()
    # "Present" means a checkpoint is there. An empty directory used to pass on
    # existence alone; see test_a_local_dir_without_weights_is_not_already_present.
    (local / "modules.json").write_text("[]")
    (local / "model.safetensors").write_bytes(b"ST")

    body = _resolve(client, str(local)).json()
    assert body["backend"] == "sentence-transformers"
    assert body["cached"] is True
    assert body["download_repo"] is None


def test_uncached_gguf_names_the_one_file_the_loader_would_open(client, monkeypatch):
    """The companion repo carries every quant; only the variant the embedder opens
    should be fetched, so the picker gets a file list rather than a whole repo."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    monkeypatch.setattr(
        settings,
        "_remote_embedding_gguf_plan",
        lambda candidates, token: (candidates[0], ["bge-small-en-v1.5-F16.gguf"]),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: 133_000_000)

    body = _resolve(client).json()
    assert body["backend"] == "llama"
    assert body["download_repo"] == "unsloth/bge-small-en-v1.5-GGUF"
    assert body["files"] == ["bge-small-en-v1.5-F16.gguf"]
    assert body["cached"] is False
    assert body["size_bytes"] == 133_000_000
    assert body["error"] is None


def test_relaxed_cache_does_not_hide_the_preferred_online_variant(client, monkeypatch):
    """A fallback quant/candidate is only the loader's offline last resort; it
    must not suppress the configured variant's download while Hub listing works."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    seen = []

    def _cached(candidates, require_variant):
        seen.append((tuple(candidates), require_variant))
        return candidates[0] if not require_variant else None

    monkeypatch.setattr(settings, "_cached_embedding_gguf", _cached)
    monkeypatch.setattr(
        settings,
        "_remote_embedding_gguf_plan",
        lambda candidates, token: (candidates[0], ["embed-F16.gguf"]),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda *a: 10)

    body = _resolve(client, "acme/embed").json()
    assert seen == [(("acme/embed-GGUF",), True)]
    assert body["cached"] is False
    assert body["files"] == ["embed-F16.gguf"]


def test_cached_gguf_asks_for_no_download(client, monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings,
        "_cached_embedding_gguf",
        lambda candidates, require_variant: candidates[0],
    )

    def _unreachable(*args, **kwargs):  # pragma: no cover - the point of the test
        raise AssertionError("a cached model must not be listed against the hub")

    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", _unreachable)

    body = _resolve(client).json()
    assert body["cached"] is True
    assert body["files"] is None


def test_the_search_fallback_finds_an_off_convention_name(client, monkeypatch):
    """The companion may not be named "<model>-GGUF" at all, so the owner's repos
    are searched before giving up."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda candidates, token: None)
    monkeypatch.setattr(
        settings,
        "_search_hub_for_gguf",
        lambda m, token: (
            "unsloth/embeddinggemma-300m-GGUF",
            ["embeddinggemma-300M-F16.gguf"],
        ),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: None)

    body = _resolve(client, "unsloth/embeddinggemma-300m-qat-q8_0-unquantized").json()
    assert body["download_repo"] == "unsloth/embeddinggemma-300m-GGUF"
    assert body["files"] == ["embeddinggemma-300M-F16.gguf"]
    assert body["error"] is None


def test_the_search_never_leaves_the_model_owner(monkeypatch):
    """Picking unsloth/X must download unsloth's own weights. A repo name is not
    proof of provenance, so a third party's "X-GGUF" is not an acceptable source."""
    seen: dict = {}

    class _Hit:
        def __init__(self, repo_id):
            self.id = repo_id

    class _Api:
        def list_models(self, **kwargs):
            seen.update(kwargs)
            # The Hub can return neighbours; only the owner's own may be taken.
            return [_Hit("someone-else/Qwen3-Embedding-8B-GGUF"), _Hit("unsloth/unrelated")]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: ["x.gguf"])

    assert settings._search_hub_for_gguf("unsloth/Qwen3-Embedding-8B", None) is None
    # And the query itself is scoped to the owner, not filtered only afterwards.
    assert seen["author"] == "unsloth"


def test_the_search_requires_an_exact_conversion_name(monkeypatch):
    class _Hit:
        def __init__(self, repo_id):
            self.id = repo_id

    class _Api:
        def list_models(self, **kwargs):
            return [_Hit("acme/foo-bar-GGUF"), _Hit("acme/foo-GGUF")]

    import huggingface_hub
    import utils.utils as utils

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: ["foo-F16.gguf"])
    monkeypatch.setattr(
        utils,
        "call_with_deadline",
        lambda fn, timeout, name: fn(),
    )

    assert settings._search_hub_for_gguf("acme/foo", None) == ("acme/foo-GGUF", ["foo-F16.gguf"])


def test_split_gguf_plan_contains_every_shard():
    names = [
        "F16/embed-00002-of-00002.gguf",
        "F16/embed-00001-of-00002.gguf",
        "Q8/embed-Q8_0.gguf",
    ]
    assert settings._pick_downloadable_gguf(names) == [
        "F16/embed-00001-of-00002.gguf",
        "F16/embed-00002-of-00002.gguf",
    ]


def test_repo_file_listing_uses_a_deadline(monkeypatch):
    import huggingface_hub
    import utils.utils as utils

    seen = {}
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo, token = None: [repo])

    def _bounded(fn, timeout, name):
        seen.update(timeout = timeout, name = name)
        return fn()

    monkeypatch.setattr(utils, "call_with_deadline", _bounded)
    assert settings._list_repo_files_bounded("acme/embed", None) == ["acme/embed"]
    assert seen == {
        "timeout": settings._GGUF_LIST_DEADLINE_S,
        "name": "embed-settings-repo-listing",
    }


def test_resolution_shares_one_deadline_across_every_remote_fallback(monkeypatch):
    import huggingface_hub
    import utils.utils as utils

    now = [100.0]
    timeouts = []

    class _Api:
        def list_models(self, **kwargs):
            return []

    def _bounded(fn, timeout, name):
        timeouts.append(timeout)
        result = fn()
        now[0] += 6.0
        return result

    monkeypatch.setattr(settings.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(utils, "call_with_deadline", _bounded)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: [])
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    # An install that embeds with llama-server has the binary; the test host does not.
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: True)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda model: False)
    monkeypatch.setattr(settings, "_local_gguf_backend_error", lambda model: None)
    monkeypatch.setattr(
        settings,
        "_embedding_gguf_candidates",
        lambda model: ["acme/one", "acme/two", "acme/three"],
    )
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    monkeypatch.setattr(settings, "_sentence_transformers_fallback_allowed", lambda model: False)

    plan = settings._resolve_embedding_model_plan("acme/embed", None)

    assert plan.error is not None
    assert timeouts == pytest.approx([20.0, 14.0, 8.0, 2.0])


def _no_gguf_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda candidates, token: None)
    monkeypatch.setattr(settings, "_search_hub_for_gguf", lambda m, token: None)


def test_no_gguf_falls_back_to_the_models_own_safetensors(client, monkeypatch):
    """Safetensors cost about 1 GB more memory but they load, so they beat both
    refusing the model and pulling a stranger's conversion."""
    _no_gguf_anywhere(monkeypatch)
    monkeypatch.setattr(settings, "_safetensors_plan", lambda m, token: (m, ["model.safetensors"]))
    monkeypatch.setattr(settings, "_hf_snapshot_size", lambda repo, token: 4096)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    body = _resolve(client, "unsloth/Qwen3-Embedding-8B").json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] == "unsloth/Qwen3-Embedding-8B"
    # No file list: ST needs the config and tokenizer too, not just the weights.
    assert body["files"] is None
    assert body["size_bytes"] == 4096
    assert body["error"] is None


def test_sentence_transformers_weight_probe_ignores_companion_bins(monkeypatch):
    monkeypatch.setattr(
        settings,
        "_list_repo_files_bounded",
        lambda model, token: ["tokenizer.bin", "assets/vocab.bin"],
    )
    assert settings._st_weight_files("acme/not-a-checkpoint", None) is None

    monkeypatch.setattr(
        settings,
        "_list_repo_files_bounded",
        lambda model, token: ["tokenizer.bin", "weights/pytorch_model-00001-of-00001.bin"],
    )
    assert settings._st_weight_files("acme/checkpoint", None) == [
        "weights/pytorch_model-00001-of-00001.bin"
    ]


def test_sentence_transformers_size_matches_the_full_snapshot_download(monkeypatch):
    import huggingface_hub

    siblings = [
        _types.SimpleNamespace(rfilename = "config.json", size = 10),
        _types.SimpleNamespace(rfilename = "tokenizer.json", size = 20),
        _types.SimpleNamespace(rfilename = "tokenizer.bin", size = 30),
        _types.SimpleNamespace(rfilename = "model.safetensors", size = 100),
        _types.SimpleNamespace(rfilename = "pytorch_model.bin", size = 110),
        _types.SimpleNamespace(rfilename = "old.gguf", size = 1_000),
        _types.SimpleNamespace(rfilename = "consolidated.00.pth", size = 2_000),
    ]
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda repo, files_metadata, token: _types.SimpleNamespace(siblings = siblings),
    )

    # The full-snapshot worker keeps configs/tokenizers and both transformer
    # weight formats, while applying its normal GGUF/consolidated exclusions.
    assert settings._hf_snapshot_size("acme/embedder", None) == 270


def test_explicit_llama_policy_does_not_offer_safetensors(client, monkeypatch):
    _no_gguf_anywhere(monkeypatch)
    # The policy is what makes this llama-only; the install still has the binary,
    # or the plan is refused for that reason before the GGUF search runs.
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: True)
    monkeypatch.setattr(settings, "_sentence_transformers_fallback_allowed", lambda model: False)
    monkeypatch.setattr(
        settings,
        "_safetensors_plan",
        lambda *a: (_ for _ in ()).throw(AssertionError("ST fallback must not be probed")),
    )
    body = _resolve(client, "acme/safetensors-only").json()
    assert body["backend"] == "llama"
    assert body["error"].startswith("No GGUF weights found")


def test_the_fallback_stays_in_the_models_own_repo(monkeypatch):
    """Same rule as the GGUF search: only the publisher is a source."""
    monkeypatch.setattr(settings, "_st_backend_available", lambda: True)
    monkeypatch.setattr(
        settings,
        "_st_weight_files",
        lambda m, token: ["model.safetensors"] if m == "acme/embedder" else None,
    )
    assert settings._safetensors_plan("acme/embedder", None) == (
        "acme/embedder",
        ["model.safetensors"],
    )
    assert settings._safetensors_plan("acme/no-weights", None) is None


def test_a_gguf_only_install_is_not_offered_safetensors(monkeypatch):
    """No torch here, so ST is not a working answer and the save reason stands."""
    monkeypatch.setattr(settings, "_st_backend_available", lambda: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, token: ["model.safetensors"])
    assert settings._safetensors_plan("acme/embedder", None) is None


def test_nothing_anywhere_still_reports_the_save_reason(client, monkeypatch):
    """Only when the name candidates, the Hub search AND safetensors come up empty."""
    _no_gguf_anywhere(monkeypatch)
    monkeypatch.setattr(settings, "_safetensors_plan", lambda m, token: None)
    body = _resolve(client, "unsloth/nothing-like-this").json()
    assert body["download_repo"] is None
    assert body["files"] is None
    assert body["cached"] is False
    assert body["error"].startswith("No GGUF weights found")


def test_a_local_gguf_is_already_the_artifact(client, monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    # An install that embeds with llama-server has the binary; the test host does not.
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: True)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: True)

    body = _resolve(client, "/models/my-embedder.gguf").json()
    assert body["cached"] is True
    assert body["download_repo"] is None
    assert body["error"] is None


def test_candidates_follow_the_loader_order(monkeypatch):
    """One helper builds them, so the picker cannot fetch from a repo the loader
    would not have opened."""
    assert settings._embedding_gguf_candidates("acme/embedder") == [
        "acme/embedder-GGUF",
        "acme/embedder",
    ]


def test_stored_off_convention_repo_is_the_preferred_candidate(monkeypatch):
    import utils.embedding_model_settings as ems
    monkeypatch.setattr(
        ems,
        "get_stored_gguf_repo",
        lambda model: "acme/special-conversion" if model == "acme/embedder" else None,
    )
    assert settings._embedding_gguf_candidates("acme/embedder") == [
        "acme/special-conversion",
        "acme/embedder-GGUF",
        "acme/embedder",
    ]


def test_explicit_gguf_repo_disables_stored_and_discovered_mirrors(monkeypatch):
    from core.rag import config as rag_config
    import utils.embedding_model_settings as ems

    monkeypatch.setenv("RAG_EMBED_GGUF_REPO", "deploy/pinned-embedder")
    monkeypatch.setattr(rag_config, "EMBED_GGUF_REPO", "deploy/pinned-embedder")
    monkeypatch.setattr(ems, "get_stored_gguf_repo", lambda model: "acme/old-mirror")

    assert settings._embedding_gguf_candidates("acme/embedder") == ["deploy/pinned-embedder"]
    assert settings._search_hub_for_gguf("acme/embedder", None) is None


def test_resolved_mirror_reports_its_exact_downloaded_files_as_cached(client, monkeypatch):
    _no_gguf_anywhere(monkeypatch)
    monkeypatch.setattr(
        settings,
        "_search_hub_for_gguf",
        lambda model, token: ("acme/embedder_gguf", ["embed-F16.gguf"]),
    )
    monkeypatch.setattr(
        settings,
        "_cached_embedding_gguf_files",
        lambda repo, files: repo == "acme/embedder_gguf" and files == ["embed-F16.gguf"],
    )

    body = _resolve(client, "acme/embedder").json()
    assert body["download_repo"] == "acme/embedder_gguf"
    assert body["cached"] is True
    assert body["files"] == ["embed-F16.gguf"]
    # A repo that already names GGUF is its own candidate, not "...-GGUF-GGUF".
    assert settings._embedding_gguf_candidates("acme/embedder-GGUF") == ["acme/embedder-GGUF"]
    # unsloth's unquantized re-uploads keep their GGUF on the base name.
    assert settings._embedding_gguf_candidates(
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized"
    ) == [
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized-GGUF",
        "unsloth/embeddinggemma-300m-GGUF",
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized",
    ]


def test_the_resolved_repo_is_what_the_loader_opens(monkeypatch):
    """A conversion under another owner follows no naming rule, so it is stored and
    read back rather than re-derived."""
    import storage.studio_db as db

    store: dict = {}
    monkeypatch.setattr(
        db, "get_app_settings", lambda keys: {k: store[k] for k in keys if k in store}
    )
    monkeypatch.setattr(db, "upsert_app_settings", lambda s: store.update(s) or store)

    import utils.embedding_model_settings as ems
    from core.rag import config as rag_config

    ems._invalidate_cache()
    ems.set_rag_embedding_model(
        "unsloth/Qwen3-Embedding-4B", gguf_repo = "Qwen/Qwen3-Embedding-4B-GGUF"
    )
    assert rag_config.effective_gguf_repo() == "Qwen/Qwen3-Embedding-4B-GGUF"

    # A pair recorded for another model must never be served for this one.
    store[ems.EMBEDDING_GGUF_SETTING_KEY] = "Qwen/Qwen3-Embedding-4B-GGUF"
    ems._invalidate_cache()
    assert ems.get_stored_gguf_repo("unsloth/bge-m3") is None
    ems._invalidate_cache()


def test_the_chosen_backend_is_read_back_by_the_loader(monkeypatch):
    """A safetensors-only model must not be handed to llama-server, which would have
    nothing to open. The picker records the backend; ``auto`` honours it."""
    import storage.studio_db as db

    store: dict = {}
    monkeypatch.setattr(
        db, "get_app_settings", lambda keys: {k: store[k] for k in keys if k in store}
    )
    monkeypatch.setattr(db, "upsert_app_settings", lambda s: store.update(s) or store)

    import utils.embedding_model_settings as ems
    from core.rag import embeddings as rag_embeddings

    ems._invalidate_cache()
    ems.set_rag_embedding_model("unsloth/Qwen3-Embedding-8B", backend = "sentence-transformers")
    assert ems.get_stored_backend("unsloth/Qwen3-Embedding-8B") == "sentence-transformers"
    assert rag_embeddings._resolve_auto_for_model() == "sentence-transformers"

    # A backend recorded for another model must not be served for this one.
    store[ems.EMBEDDING_MODEL_SETTING_KEY] = "unsloth/bge-m3"
    store[ems.EMBEDDING_RESOLUTION_SETTING_KEY] = {
        "model": "unsloth/bge-m3",
        "gguf_repo": None,
        "backend": "llama-server",
    }
    ems._invalidate_cache()
    # A model this process never resolved gets nothing, which is the property that
    # matters: the record belongs to bge-m3 and is not lent to anyone else.
    assert ems.get_stored_backend("unsloth/never-resolved") is None
    # Qwen keeps ITS OWN earlier resolution, not bge-m3's. Forgetting it dropped a
    # still-running job for Qwen back onto the hardware default; the memo is keyed
    # per model, so this is Qwen's record, not a stale pair.
    assert ems.get_stored_backend("unsloth/Qwen3-Embedding-8B") == "sentence-transformers"
    ems._resolved_gguf_memo.clear()
    assert ems.get_stored_backend("unsloth/Qwen3-Embedding-8B") is None
    ems._invalidate_cache()


def test_the_token_is_a_header_not_a_query_parameter(client, monkeypatch):
    """A gated repo's credential must stay out of URLs and access logs."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings, "_cached_embedding_gguf", lambda candidates, require_variant: None
    )
    seen: dict = {}

    def _plan(candidates, token):
        seen["token"] = token
        return candidates[0], ["model-F16.gguf"]

    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", _plan)
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: None)

    client.get(
        "/embedding-model/resolve",
        params = {"model": "acme/gated-embedder"},
        headers = {"X-Unsloth-HF-Token": "hf_secret"},
    )
    assert seen["token"] == "hf_secret"


def test_a_local_gguf_is_not_reported_as_a_ready_sentence_transformers_model(
    client, monkeypatch, tmp_path
):
    """The presence probe accepted any existing path. `auto` now routes a local
    .gguf to llama-server; this covers an explicit sentence-transformers setting,
    where the no-loadable-weights error is the honest answer."""
    gguf = tmp_path / "embed.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    assert settings._local_sentence_transformer_is_present(str(gguf)) is False
    # A real ST folder on the same filesystem is still recognised.
    folder = tmp_path / "st-model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"ST")
    assert settings._local_sentence_transformer_is_present(str(folder)) is True

    body = _resolve(client, str(gguf)).json()
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]


def test_a_cached_gguf_only_repo_is_not_reported_as_a_ready_st_model(client, monkeypatch, tmp_path):
    """hf_cache_snapshot_is_loadable counts .gguf, right for llama and wrong here:
    a cached GGUF-only repo came back ready, skipping the Hub weight check."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model-Q8_0.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: True)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", lambda m: snapshot)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir_for_repo", lambda m: snapshot)

    assert settings._cached_snapshot_has_st_weights("org/gguf-only") is False

    body = _resolve(client, "org/gguf-only").json()
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]

    # A real cached ST snapshot on the same path still reports ready.
    (snapshot / "model.safetensors").write_bytes(b"ST")
    assert settings._cached_snapshot_has_st_weights("org/gguf-only") is True
    body = _resolve(client, "org/real-st").json()
    assert body["cached"] is True
    assert body["error"] is None


def test_a_cached_safetensors_model_is_selectable_offline(client, monkeypatch, tmp_path):
    """The safetensors fallback proved weights only through the remote listing, so
    offline a fully downloaded model could not be selected at all."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ST")
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(settings, "_st_backend_available", lambda: True)
    monkeypatch.setattr(settings, "_sentence_transformers_fallback_allowed", lambda m: True)
    # Offline: every remote probe fails.
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda c, t: None)
    monkeypatch.setattr(settings, "_search_hub_for_gguf", lambda m, t: None)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda c, require_variant = True: None)
    monkeypatch.setattr(settings, "_hf_snapshot_size", lambda repo, token: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: True)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", lambda m: snapshot)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir_for_repo", lambda m: snapshot)

    assert settings._safetensors_plan("org/st-only", None) == ("org/st-only", ["model.safetensors"])

    body = _resolve(client, "org/st-only").json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] == "org/st-only"
    assert body["cached"] is True
    assert body["error"] is None


def test_a_local_dir_without_weights_is_not_already_present(client, monkeypatch, tmp_path):
    """A folder holding modules.json and no checkpoint also passes
    is_embedding_model's local-path check, so it was reported cached and accepted
    without force, then failed at the first index when SentenceTransformer went
    looking for the weights."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    empty = tmp_path / "metadata-only"
    empty.mkdir()
    (empty / "modules.json").write_text("[]")
    (empty / "config.json").write_text("{}")
    assert settings._local_sentence_transformer_is_present(str(empty)) is False

    body = _resolve(client, str(empty)).json()
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]

    # A complete module subtree is enough; the weights need not sit at the root.
    # modules.json has to declare it, as a real one does: the directory is judged
    # by the layout it announces, the same way a Hub snapshot is.
    module = empty / "0_Transformer"
    module.mkdir()
    (module / "model.safetensors").write_bytes(b"ST")
    (empty / "modules.json").write_text('[{"idx": 0, "name": "0", "path": "0_Transformer"}]')
    assert settings._local_sentence_transformer_is_present(str(empty)) is True


def test_a_slashless_alias_resolves_under_the_sentence_transformers_namespace(client, monkeypatch):
    """A slashless name resolves under sentence-transformers/, as the loader's own
    st_repo_id_candidates encodes. Probing only the literal id refused a download
    that worked before this picker existed."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    listed = []

    def _files(repo, token):
        listed.append(repo)
        return ["model.safetensors"] if repo == "sentence-transformers/all-MiniLM-L6-v2" else None

    monkeypatch.setattr(settings, "_st_weight_files", _files)
    monkeypatch.setattr(settings, "_hf_snapshot_size", lambda repo, token: 90_000_000)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    body = _resolve(client, "all-MiniLM-L6-v2").json()
    assert body["error"] is None
    # The setting keeps the alias the user typed; the download names the repo that
    # actually publishes the weights.
    assert body["embedding_model"] == "all-MiniLM-L6-v2"
    assert body["download_repo"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert listed == ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"]


def test_a_bare_checkpoint_file_is_not_a_local_sentence_transformer(client, monkeypatch, tmp_path):
    """SentenceTransformer takes a directory or a repo id, never a bare file."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    bare = tmp_path / "model.safetensors"
    bare.write_bytes(b"ST")
    assert settings._local_sentence_transformer_is_present(str(bare)) is False

    body = _resolve(client, str(bare)).json()
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]


def test_a_tokenizer_bin_is_not_mistaken_for_a_checkpoint(client, monkeypatch, tmp_path):
    """.bin is the loose suffix: tokenizer.bin shares it with real weights, so a
    directory holding only that beside modules.json resolved as cached."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    local = tmp_path / "partial"
    local.mkdir()
    (local / "modules.json").write_text("[]")
    (local / "tokenizer.bin").write_bytes(b"NOT WEIGHTS")
    assert settings._local_sentence_transformer_is_present(str(local)) is False

    body = _resolve(client, str(local)).json()
    assert body["cached"] is False
    assert "no checkpoint this backend can load" in body["error"]

    # A real checkpoint under any of the accepted families still counts.
    (local / "pytorch_model.bin").write_bytes(b"ST")
    assert settings._local_sentence_transformer_is_present(str(local)) is True


def test_the_cached_snapshot_probe_uses_the_same_filename_family(monkeypatch, tmp_path):
    """The cached-snapshot probe had the same suffix-only test as the local one."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.bin").write_bytes(b"NOT WEIGHTS")
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", lambda m: snapshot)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir_for_repo", lambda m: snapshot)
    assert settings._cached_snapshot_has_st_weights("org/partial") is False
    assert settings._cached_st_weight_names("org/partial") == []

    (snapshot / "model.safetensors").write_bytes(b"ST")
    assert settings._cached_snapshot_has_st_weights("org/partial") is True
    assert settings._cached_st_weight_names("org/partial") == ["model.safetensors"]


def test_a_cached_alias_names_the_namespace_it_is_filed_under(client, monkeypatch, tmp_path):
    """hf_cache_snapshot_dir is alias-aware, so a slashless name finds its snapshot
    under sentence-transformers/. Returning the literal id as the download repo
    sent the manager at a top-level repo that usually does not exist."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"ST")
    import utils.utils as utils

    _snapshot_of = (
        lambda repo: snapshot if repo == "sentence-transformers/all-MiniLM-L6-v2" else None
    )
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", _snapshot_of)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir_for_repo", _snapshot_of)
    monkeypatch.setattr(settings, "_st_backend_available", lambda: True)
    # Offline: the remote listing cannot answer, so only the cache can.
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)

    assert utils.cached_st_repo("all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings._safetensors_plan("all-MiniLM-L6-v2", None) == (
        "sentence-transformers/all-MiniLM-L6-v2",
        ["model.safetensors"],
    )
    # A name with no cached snapshot anywhere still falls through to the listing.
    assert settings._safetensors_plan("org/uncached", None) is None


def test_a_stale_literal_cache_does_not_hide_a_complete_alias_snapshot(
    client, monkeypatch, tmp_path
):
    """hf_cache_snapshot_is_loadable stops at the first matching cache directory
    while the ST predicate walks candidates for a complete one. Conjoining them
    let a stale models--all-MiniLM-L6-v2 entry report the model uncached even with
    a complete sentence-transformers/ snapshot beside it; offline the Hub probe
    then failed and /resolve returned an error the loader would have disagreed
    with."""
    literal = tmp_path / "literal"
    literal.mkdir()
    (literal / "config.json").write_text("{}")
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "config.json").write_text("{}")
    (alias / "model.safetensors").write_bytes(b"ST")
    import utils.utils as utils

    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    _snapshot_of = (
        lambda repo: alias if repo == "sentence-transformers/all-MiniLM-L6-v2" else literal
    )
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", _snapshot_of)
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir_for_repo", _snapshot_of)
    # The generic check answers about the stale literal entry.
    monkeypatch.setattr(
        utils,
        "hf_cache_snapshot_is_loadable",
        lambda repo: repo == "sentence-transformers/all-MiniLM-L6-v2",
    )
    # Offline: the Hub probe cannot rescue it.
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, t: None)

    body = _resolve(client, "all-MiniLM-L6-v2").json()
    assert body["cached"] is True
    assert body["error"] is None


def test_a_remote_gguf_repo_is_served_by_llama_server_on_an_st_host(monkeypatch):
    """A GPU host resolves ``auto`` to sentence-transformers, which then searches a
    typed or API-selected ``owner/model-GGUF`` for safetensors, finds none and errors.
    Saving over that error sets the pending marker, and the ST loader answers a
    pending model with "not downloaded" before the runtime fallback can reach
    llama-server, so the repo could never load however it was cached."""
    from core.rag import embeddings as rag_embeddings
    import utils.embedding_model_settings as ems

    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    # A GPU is present, so the hardware default is sentence-transformers.
    monkeypatch.setattr(rag_embeddings, "_resolve_auto", lambda: "sentence-transformers")
    # And the forced save recorded no backend, exactly as a failed plan does.
    monkeypatch.setattr(ems, "get_stored_backend", lambda model: None)

    assert rag_embeddings._resolve_auto_for_model("unsloth/bge-m3-GGUF") == "llama-server"
    assert rag_embeddings._resolve_auto_for_model("unsloth/bge-m3") == "sentence-transformers"


def test_a_local_directory_named_gguf_is_still_a_sentence_transformers_model(monkeypatch, tmp_path):
    """The remote rule is a name test, and a directory may be named anything. Only
    the filesystem can say what ``~/models/my-gguf`` holds, and here it holds
    safetensors."""
    from core.rag import embeddings as rag_embeddings
    import utils.embedding_model_settings as ems

    local = tmp_path / "my-gguf"
    local.mkdir()
    (local / "model.safetensors").write_bytes(b"\x00")
    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    monkeypatch.setattr(rag_embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(ems, "get_stored_backend", lambda model: None)

    assert rag_embeddings._resolve_auto_for_model(str(local)) == "sentence-transformers"


def test_a_gguf_only_model_is_refused_when_llama_server_is_missing(client, monkeypatch):
    """A GGUF-named model routes to llama-server because nothing else can open it,
    which says nothing about whether this install can run that backend. Without the
    binary the plan was advertised as valid and the transfer persisted, and the
    first warm failed in _resolve_binary with the download already done."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: False)

    body = _resolve(client, "unsloth/bge-m3-GGUF").json()
    assert body["download_repo"] is None
    assert "no llama-server binary was found" in body["error"]


def test_a_safetensors_model_still_falls_back_when_llama_server_is_missing(client, monkeypatch):
    """The refusal is scoped to models only llama can serve. One that publishes
    safetensors has a usable plan on the other backend, and turning that into an
    error would strand it too."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: False)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda *a, **k: None)
    monkeypatch.setattr(settings, "_search_hub_for_gguf", lambda *a, **k: None)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda *a, **k: None)
    monkeypatch.setattr(settings, "_sentence_transformers_fallback_allowed", lambda m: True)
    monkeypatch.setattr(
        settings, "_safetensors_plan", lambda m, t: ("org/st-only", ["model.safetensors"])
    )
    monkeypatch.setattr(settings, "_cached_snapshot_has_st_weights", lambda m: True)

    body = _resolve(client, "org/st-only").json()
    assert body["error"] is None
    assert body["backend"] == "sentence-transformers"


def test_a_cached_alias_is_verified_under_the_namespace_it_is_filed_under(client, monkeypatch):
    """A slashless model cached only as sentence-transformers/<name> is on disk and
    loadable, but reporting the literal alias as the download repo sent the PUT's
    verification and security scan at a top-level repo that usually does not exist,
    rejecting a valid cached model with a 409 the loader would have ignored."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(
        settings,
        "_cached_st_source",
        lambda m: ("sentence-transformers/all-MiniLM-L6-v2", Path("/snap")),
    )

    body = _resolve(client, "all-MiniLM-L6-v2").json()
    assert body["cached"] is True
    assert body["download_repo"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["error"] is None


def test_a_validated_backend_outranks_the_gguf_name_heuristic(monkeypatch):
    """A repo called <name>-GGUF whose published family is torn or absent falls
    back to its own safetensors, and the resolver persists that plan. Reading the
    name as llama-server anyway sent the first index down the GGUF pending path,
    to fail as "not downloaded" on a model whose weights were validated."""
    from core.rag import embeddings as rag_embeddings
    import utils.embedding_model_settings as ems

    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    monkeypatch.setattr(rag_embeddings, "_resolve_auto", lambda: "llama-server")
    monkeypatch.setattr(ems, "get_stored_backend", lambda model: "sentence-transformers")
    # The name is a guess; a local .gguf is not, and keeps its precedence (see
    # test_a_local_gguf_beats_a_stored_sentence_transformers_record).

    assert rag_embeddings._resolve_auto_for_model("org/torn-GGUF") == "sentence-transformers"
    # With nothing validated, the name still decides, which is what stops a forced
    # save over a failed plan from stranding the model on the wrong backend.
    monkeypatch.setattr(ems, "get_stored_backend", lambda model: None)
    assert rag_embeddings._resolve_auto_for_model("org/torn-GGUF") == "llama-server"


def test_a_torn_local_checkpoint_is_not_reported_as_present(client, monkeypatch, tmp_path):
    """Any one matching filename used to be enough, so half a shard family read as
    ready and the first index failed when SentenceTransformer opened it. Same
    completeness test a Hub snapshot gets."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    torn = tmp_path / "half-copied"
    torn.mkdir()
    (torn / "config.json").write_text("{}")
    # The index is what says how many shards the family has, as a real sharded
    # checkpoint ships it.
    (torn / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    (torn / "model-00001-of-00002.safetensors").write_bytes(b"ST")

    assert settings._local_sentence_transformer_is_present(str(torn)) is False

    # The missing shard completes it.
    (torn / "model-00002-of-00002.safetensors").write_bytes(b"ST")
    assert settings._local_sentence_transformer_is_present(str(torn)) is True


def test_a_declared_module_the_directory_lacks_is_not_present(client, monkeypatch, tmp_path):
    """modules.json announces the layout ST will load. A module it names and the
    directory does not have is a torn copy however complete the rest looks."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    partial = tmp_path / "missing-module"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    (partial / "model.safetensors").write_bytes(b"ST")
    (partial / "modules.json").write_text(
        '[{"idx": 0, "name": "0", "path": ""}, {"idx": 1, "name": "1", "path": "1_Pooling"}]'
    )

    assert settings._local_sentence_transformer_is_present(str(partial)) is False

    (partial / "1_Pooling").mkdir()
    assert settings._local_sentence_transformer_is_present(str(partial)) is True


def test_an_explicit_llama_policy_refuses_any_model_without_the_binary(client, monkeypatch):
    """An explicit RAG_EMBED_BACKEND=llama-server refuses the safetensors fallback
    for every model, not only GGUF-named ones, so an ordinary repo id is just as
    unservable without a binary. Scoping the check to the name offered and
    persisted a managed download _resolve_binary would reject at first use."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(settings, "_llama_runtime_available", lambda: False)
    monkeypatch.setattr(settings, "_sentence_transformers_fallback_allowed", lambda model: False)
    monkeypatch.setattr(
        settings,
        "_remote_embedding_gguf_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no plan may be offered")),
    )

    body = _resolve(client, "org/embedder").json()
    assert body["download_repo"] is None
    assert "no llama-server binary was found" in body["error"]


def test_a_whole_segment_gguf_name_routes_to_llama_server(monkeypatch):
    """config.gguf_repo_candidates treats "gguf" as a whole name segment, so
    owner/GGUF-model is a direct GGUF repo there. Deciding it by suffix here sent
    those to sentence-transformers, which has nothing to open, and the rejection
    stuck even under a forced selection."""
    from core.rag import embeddings as rag_embeddings
    import utils.embedding_model_settings as ems

    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    monkeypatch.setattr(rag_embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(ems, "get_stored_backend", lambda model: None)

    for model in ("owner/GGUF-model", "owner/model-GGUF-Q8", "owner/model-GGUF"):
        assert rag_embeddings._resolve_auto_for_model(model) == "llama-server", model
    # A plain substring is still not a GGUF repo, as config's predicate says.
    assert rag_embeddings._resolve_auto_for_model("owner/bigguf") == "sentence-transformers"
