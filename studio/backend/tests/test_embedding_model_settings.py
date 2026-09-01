# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test for the customizable RAG embedding model: a saved override becomes the
effective model and derives its GGUF companion for the llama-server backend."""

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

import utils.embedding_model_settings as ems
from core.rag import config as rag_config


@pytest.fixture
def settings_store(monkeypatch):
    """In-memory app_settings store patched under the module's lazy imports."""
    import storage.studio_db as studio_db

    store: dict = {}
    monkeypatch.setattr(
        studio_db,
        "get_app_settings",
        lambda keys: {key: store[key] for key in keys if key in store},
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda settings: store.update(settings) or store
    )

    def _cas(key, expected, value):
        if store.get(key) != expected:
            return False
        store[key] = value
        return True

    monkeypatch.setattr(studio_db, "compare_and_set_app_setting", _cas)
    # Process-wide and outliving the patched store, so a resolution recorded here
    # would answer for the same model in every later test file.
    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    yield store
    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()


def test_custom_model_overrides_default_and_derives_gguf(settings_store, monkeypatch):
    """The core contract: with nothing stored the default is in effect; a saved
    custom model becomes the effective embedding model and derives its -GGUF
    companion (what the llama-server backend loads); reset clears the override."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL
    assert rag_config.effective_gguf_repo() == rag_config.EMBED_GGUF_REPO

    assert ems.set_rag_embedding_model("  org/my-embedder  ") == "org/my-embedder"
    assert rag_config.effective_embedding_model() == "org/my-embedder"
    assert rag_config.effective_gguf_repo() == "org/my-embedder-GGUF"

    assert ems.reset_rag_embedding_model() == rag_config.EMBEDDING_MODEL
    assert ems.get_stored_embedding_model() is None


def test_env_default_derives_its_gguf_companion(monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-default-embedder")

    assert rag_config.default_gguf_repo() == "org/env-default-embedder-GGUF"


def test_env_default_keeps_its_resolved_gguf_without_becoming_custom(settings_store, monkeypatch):
    """An env default can resolve to an off-convention repo even though selecting
    it should not turn the default itself into a persisted override."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-default-embedder")

    ems.set_rag_embedding_model(
        "org/env-default-embedder",
        gguf_repo = "org/published-conversion",
        backend = "llama-server",
    )

    assert ems.get_stored_embedding_model() is None
    assert ems.get_stored_gguf_repo("org/env-default-embedder") == "org/published-conversion"
    assert rag_config.effective_gguf_repo() == "org/published-conversion"


def test_resolution_record_keeps_model_repo_and_backend_atomic(settings_store):
    ems.set_rag_embedding_model(
        "org/embedder",
        gguf_repo = "org/embedder-conversion",
        backend = "llama-server",
    )
    assert settings_store[ems.EMBEDDING_RESOLUTION_SETTING_KEY] == {
        "model": "org/embedder",
        "gguf_repo": "org/embedder-conversion",
        "backend": "llama-server",
        "download_pending": False,
        "gguf_files": None,
    }
    assert settings_store[ems.EMBEDDING_GGUF_SETTING_KEY] is None
    assert settings_store[ems.EMBEDDING_BACKEND_SETTING_KEY] is None


def test_pending_download_is_stored_with_the_same_atomic_resolution(settings_store):
    ems.set_rag_embedding_model(
        "org/embedder",
        gguf_repo = "org/embedder-conversion",
        backend = "llama-server",
        download_pending = True,
    )
    assert ems.get_stored_download_pending("org/embedder") is True
    assert ems.get_stored_download_pending("org/another") is False
    assert settings_store[ems.EMBEDDING_RESOLUTION_SETTING_KEY]["download_pending"] is True


def test_a_completed_transfer_retires_the_pending_marker(settings_store):
    """Nothing else clears it: the picker re-resolves after a download but does not
    save again, so a marker left behind pins the model cache-only for good and a
    later cache eviction reads as "never downloaded"."""
    ems.set_rag_embedding_model(
        "org/embedder",
        gguf_repo = "org/embedder-conversion",
        backend = "llama-server",
        download_pending = True,
    )

    assert ems.clear_stored_download_pending("org/embedder") is True
    assert ems.get_stored_download_pending("org/embedder") is False
    # The rest of the resolution survives: the loader still opens what was fetched.
    assert ems.get_stored_gguf_repo("org/embedder") == "org/embedder-conversion"
    assert ems.get_stored_backend("org/embedder") == "llama-server"
    # Idempotent, and never touches another model's record.
    assert ems.clear_stored_download_pending("org/embedder") is False
    assert ems.clear_stored_download_pending("org/another") is False
    assert ems.get_stored_gguf_repo("org/embedder") == "org/embedder-conversion"


def test_a_concurrent_save_is_not_reverted_by_a_late_pending_clear(settings_store):
    """The loader reads model A's pending resolution, the user saves model B, and
    only then does the clear land. A plain upsert would put A's record back beside
    B's override, leaving B to re-derive a backend and a companion it never
    resolved. The write is conditional on the record it read."""
    ems.set_rag_embedding_model(
        "org/a", gguf_repo = "org/a-GGUF", backend = "llama-server", download_pending = True
    )
    stale = ems._get_stored_state()  # A's loader has read it
    assert stale[1] == "org/a" and stale[4] is True
    ems.set_rag_embedding_model("org/b", gguf_repo = "org/b-GGUF", backend = "sentence-transformers")
    from utils.workspace_context import current_workspace_subject

    ems._cached[current_workspace_subject()] = (0.0, stale)  # its 2s snapshot still says A

    assert ems.clear_stored_download_pending("org/a") is False
    ems._invalidate_cache()
    assert ems.get_stored_gguf_repo("org/b") == "org/b-GGUF"
    assert ems.get_stored_backend("org/b") == "sentence-transformers"
    assert ems.get_stored_gguf_repo("org/a") is None


def test_a_pinned_jobs_resolved_repo_survives_a_save_for_another_model(settings_store, monkeypatch):
    """One stored record, so saving B takes A's repo away while a job pinned to A
    is still ingesting, moving its identity to the derived A-GGUF mid-run."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()
    from core.rag import config

    ems.set_rag_embedding_model(
        "org/embedder-a", gguf_repo = "mirror/off-convention-GGUF", backend = "llama-server"
    )
    assert config.effective_gguf_repo_for_embedding_model("org/embedder-a") == (
        "mirror/off-convention-GGUF"
    )

    ems.set_rag_embedding_model("org/embedder-b", gguf_repo = None, backend = None)

    # The stored record is B's now, and the staleness rule still holds.
    assert ems.get_stored_gguf_repo("org/embedder-a") is None
    # But the pinned job keeps embedding through the same mirror.
    assert config.effective_gguf_repo_for_embedding_model("org/embedder-a") == (
        "mirror/off-convention-GGUF"
    )
    # A model this process never resolved is still derived, not invented.
    assert config.effective_gguf_repo_for_embedding_model("org/never-seen") == (
        config.gguf_repo_for_embedding_model("org/never-seen")
    )


def test_a_reset_keeps_the_repo_a_running_job_still_needs(settings_store, monkeypatch):
    """Reset clears the selection, but a job pinned to the old model is still
    ingesting through the repo that was resolved for it. The memo is per model and
    consulted only when the store has nothing, so dropping it here moved that job
    onto the derived <model>-GGUF mid-run."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()
    from core.rag import config

    ems.set_rag_embedding_model(
        "org/embedder-a", gguf_repo = "mirror/off-convention-GGUF", backend = "llama-server"
    )
    assert ems.get_stored_gguf_repo("org/embedder-a") == "mirror/off-convention-GGUF"
    ems.reset_rag_embedding_model()

    # The stored selection is gone...
    assert ems.get_stored_embedding_model() is None
    # ...but the pinned job keeps embedding through the same mirror.
    assert ems.remembered_gguf_repo("org/embedder-a") == "mirror/off-convention-GGUF"
    assert config.effective_gguf_repo_for_embedding_model("org/embedder-a") == (
        "mirror/off-convention-GGUF"
    )
    # A model this process never resolved is still derived, not invented.
    assert config.effective_gguf_repo_for_embedding_model("org/never-seen") == (
        config.gguf_repo_for_embedding_model("org/never-seen")
    )


def test_a_pinned_jobs_backend_and_pending_survive_a_save_for_another_model(
    settings_store, monkeypatch
):
    """On an auto CPU install a model with no GGUF resolves to
    sentence-transformers; losing that record drops a still-running job onto the
    hardware default, and losing the marker re-enables the implicit download."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()

    ems.set_rag_embedding_model(
        "org/st-only",
        gguf_repo = None,
        backend = "sentence-transformers",
        download_pending = True,
    )
    assert ems.get_stored_backend("org/st-only") == "sentence-transformers"
    assert ems.get_stored_download_pending("org/st-only") is True

    ems.set_rag_embedding_model("org/other", gguf_repo = None, backend = "llama-server")

    assert ems.get_stored_backend("org/st-only") == "sentence-transformers"
    assert ems.get_stored_download_pending("org/st-only") is True
    # The newly saved model answers from the record, not the memo.
    assert ems.get_stored_backend("org/other") == "llama-server"
    # A model this process never resolved still has no opinion.
    assert ems.get_stored_backend("org/never-seen") is None
    assert ems.get_stored_download_pending("org/never-seen") is False


def test_retiring_the_pending_marker_retires_it_in_the_memo_too(settings_store, monkeypatch):
    """Or a pinned job keeps reading pending=True and stays cache-only after the
    download landed."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()

    ems.set_rag_embedding_model(
        "org/embedder",
        gguf_repo = None,
        backend = "sentence-transformers",
        download_pending = True,
    )
    assert ems.get_stored_download_pending("org/embedder") is True
    assert ems.clear_stored_download_pending("org/embedder") is True

    ems.set_rag_embedding_model("org/other", gguf_repo = None, backend = None)
    assert ems.get_stored_download_pending("org/embedder") is False
    # The backend it was resolved with is still remembered.
    assert ems.get_stored_backend("org/embedder") == "sentence-transformers"


def test_a_reset_makes_the_restored_defaults_resolution_durable(settings_store, monkeypatch):
    """The memo survives a reset for jobs still pinned to a model, but the restored
    default is not a running job: new work resolves through it too, and a
    process-only answer would change identity on the next restart, stranding
    whatever was indexed in between."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()
    default = ems.default_embedding_model()

    # The default itself was resolved to an off-convention mirror...
    ems.set_rag_embedding_model(
        default, gguf_repo = "mirror/off-convention-GGUF", backend = "llama-server"
    )
    assert ems.get_stored_gguf_repo(default) == "mirror/off-convention-GGUF"
    # ...then another model is selected, taking the single record with it...
    ems.set_rag_embedding_model("org/other", gguf_repo = None, backend = None)
    # ...and the selection is reset.
    assert ems.reset_rag_embedding_model() == default

    # The override is gone, but the default's resolution is durable again, not
    # living only in this process.
    assert ems.get_stored_embedding_model() is None
    ems._resolved_gguf_memo.clear()
    assert ems.get_stored_gguf_repo(default) == "mirror/off-convention-GGUF"
    assert ems.get_stored_backend(default) == "llama-server"


def test_a_reset_with_nothing_remembered_stays_a_plain_reset(settings_store, monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems._resolved_gguf_memo.clear()

    ems.set_rag_embedding_model("org/other", gguf_repo = None, backend = "sentence-transformers")
    assert ems.reset_rag_embedding_model() == ems.default_embedding_model()

    assert ems.get_stored_embedding_model() is None
    assert settings_store[ems.EMBEDDING_RESOLUTION_SETTING_KEY] is None


def test_the_planned_gguf_family_is_stored_and_survives_the_clear(settings_store):
    """The loader needs the family to tell the quant the picker actually delivered
    from an unrelated one in the same repo. The clear is conditional on the record
    as read, so carrying an extra field must not make it a permanent no-op."""
    ems._resolved_gguf_memo.clear()
    ems.set_rag_embedding_model(
        "org/a",
        gguf_repo = "org/a-GGUF",
        backend = "llama-server",
        download_pending = True,
        gguf_files = ["a-Q8_0-00001-of-00002.gguf", "a-Q8_0-00002-of-00002.gguf"],
    )

    assert ems.get_stored_gguf_files("org/a") == [
        "a-Q8_0-00001-of-00002.gguf",
        "a-Q8_0-00002-of-00002.gguf",
    ]
    assert ems.clear_stored_download_pending("org/a") is True
    assert ems.get_stored_download_pending("org/a") is False
    # The family outlives the flag: it describes the artifact, not the transfer.
    assert ems.get_stored_gguf_files("org/a") == [
        "a-Q8_0-00001-of-00002.gguf",
        "a-Q8_0-00002-of-00002.gguf",
    ]


def test_a_record_written_before_the_family_existed_still_clears(settings_store):
    """Upgrade path: an install that saved under the previous build has a record
    with no gguf_files key. Comparing against a rebuilt record would never match
    it, pinning every such model cache-only for good."""
    from storage.studio_db import upsert_app_settings

    ems._resolved_gguf_memo.clear()
    upsert_app_settings(
        {
            ems.EMBEDDING_MODEL_SETTING_KEY: "org/legacy",
            ems.EMBEDDING_RESOLUTION_SETTING_KEY: {
                "model": "org/legacy",
                "gguf_repo": "org/legacy-GGUF",
                "backend": "llama-server",
                "download_pending": True,
            },
        }
    )
    ems._invalidate_cache()

    assert ems.get_stored_gguf_files("org/legacy") is None
    assert ems.clear_stored_download_pending("org/legacy") is True
    assert ems.get_stored_download_pending("org/legacy") is False


def test_a_reset_keeps_a_pending_only_resolution_for_the_default(settings_store):
    """A default saved over a failed resolution remembers no repo and no backend,
    only the pending flag, and that flag is the one thing keeping the first index
    from starting the implicit download this picker replaces."""
    ems._resolved_gguf_memo.clear()
    default = ems.default_embedding_model()
    ems.set_rag_embedding_model(default, download_pending = True)
    assert ems.get_stored_download_pending(default) is True

    ems.set_rag_embedding_model("org/other", gguf_repo = "org/other-GGUF", backend = "llama-server")
    assert ems.reset_rag_embedding_model() == default

    # Read the store, not the memo: the memo answers for this process either way,
    # and what the reset has to preserve is the record a restart will find.
    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    assert ems.get_stored_download_pending(default) is True


def test_pinning_a_model_memoizes_what_was_resolved_for_it(settings_store, monkeypatch):
    """A worker pins its model by reading the effective one, then scans for a while
    and embeds afterwards. Only the repo/backend getters used to populate the memo,
    so a save for another model in that gap left the pinned job with nothing to
    fall back to and moved it onto the derived <model>-GGUF mid-run."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    from core.rag import config

    ems.set_rag_embedding_model(
        "org/embedder-a", gguf_repo = "mirror/off-convention-GGUF", backend = "llama-server"
    )
    # Nothing has read the resolution yet; the pin is the only thing that happens.
    ems._resolved_gguf_memo.clear()
    assert config.effective_embedding_model() == "org/embedder-a"

    ems.set_rag_embedding_model("org/embedder-b", gguf_repo = None, backend = None)

    assert config.effective_gguf_repo_for_embedding_model("org/embedder-a") == (
        "mirror/off-convention-GGUF"
    )
    assert ems.get_stored_backend("org/embedder-a") == "llama-server"
