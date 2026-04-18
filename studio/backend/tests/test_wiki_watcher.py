# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

from core.wiki.manager import WikiManager
from core.wiki.watcher import WikiFileEventHandler


class _FakeEngine:
    def __init__(self, wiki_dir: Path | None = None):
        self.wiki_dir = wiki_dir

    def _slug(self, _title: str) -> str:
        return "source-slug"


class _FakeWikiManager:
    def __init__(self, probe_sequence=None, wiki_dir: Path | None = None):
        self.engine = _FakeEngine(wiki_dir=wiki_dir)
        self.calls = []
        self.health_calls = 0
        self.retry_fallback_calls = 0
        self.enrich_calls = 0
        self._probe_sequence = list(probe_sequence or [{"used_extractive_fallback": False, "fallback_reason": None}])
        self._probe_index = 0

    def query_rag(
        self,
        question: str,
        query_context_max_chars_override=None,
        save_answer: bool = True,
        preferred_context_page: str | None = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ):
        self.calls.append(
            {
                "question": question,
                "query_context_max_chars_override": query_context_max_chars_override,
                "save_answer": save_answer,
                "preferred_context_page": preferred_context_page,
                "keep_preferred_context_full": keep_preferred_context_full,
                "preferred_context_only": preferred_context_only,
            }
        )

        if not save_answer:
            probe = self._probe_sequence[min(self._probe_index, len(self._probe_sequence) - 1)]
            self._probe_index += 1
            return {
                "status": "ok",
                "answer_page": None,
                "used_extractive_fallback": bool(probe.get("used_extractive_fallback", False)),
                "fallback_reason": probe.get("fallback_reason"),
            }

        return {
            "status": "ok",
            "answer_page": "analysis/test",
            "used_extractive_fallback": False,
            "fallback_reason": None,
        }

    def get_health(self):
        self.health_calls += 1
        return {
            "orphans": [],
            "stale_pages": [],
            "broken_links": [],
        }

    def enrich_analysis_pages(self, dry_run: bool = False, max_analysis_pages: int = 64):
        self.enrich_calls += 1
        return {
            "status": "ok",
            "dry_run": dry_run,
            "scanned_pages": max_analysis_pages,
            "updated_pages": 0,
            "changes": [],
        }

    def retry_fallback_analysis_pages(self, dry_run: bool = False, max_analysis_pages: int = 24):
        self.retry_fallback_calls += 1
        return {
            "status": "ok",
            "dry_run": dry_run,
            "scanned_pages": max_analysis_pages,
            "fallback_pages_found": 0,
            "retried_pages": 0,
            "regenerated_pages": 0,
            "fallback_still": 0,
            "skipped_no_question": 0,
            "errors": [],
            "results": [],
        }


class _FakeIngestor:
    def __init__(self, wiki_manager):
        self.wiki_manager = wiki_manager

    def should_skip_local_file(self, _file_path: Path) -> bool:
        return False

    def ingest_file(self, _file_path: Path, contributor=None):
        return "Sample Source"


def test_manager_query_rag_passes_context_override(tmp_path: Path, monkeypatch):
    manager = WikiManager.create(vault_root=tmp_path, llm_fn=lambda _: "ok")

    captured = {}

    def _fake_query(
        question: str,
        save_answer: bool = True,
        query_context_max_chars_override=None,
        preferred_context_page: str | None = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ):
        captured["question"] = question
        captured["save_answer"] = save_answer
        captured["query_context_max_chars_override"] = query_context_max_chars_override
        captured["preferred_context_page"] = preferred_context_page
        captured["keep_preferred_context_full"] = keep_preferred_context_full
        captured["preferred_context_only"] = preferred_context_only
        return {"status": "ok", "answer": "done", "context_pages": []}

    monkeypatch.setattr(manager.engine, "query", _fake_query)

    result = manager.query_rag("hello", query_context_max_chars_override=4321)

    assert result["status"] == "ok"
    assert captured["question"] == "hello"
    assert captured["save_answer"] is True
    assert captured["query_context_max_chars_override"] == 4321
    assert captured["preferred_context_page"] is None
    assert captured["keep_preferred_context_full"] is False
    assert captured["preferred_context_only"] is False


def test_watcher_applies_fraction_of_model_context(tmp_path: Path, monkeypatch):
    sources_dir = tmp_path / "wiki" / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "source-slug.md").write_text("X" * 1000, encoding="utf-8")

    wiki_manager = _FakeWikiManager(wiki_dir=tmp_path / "wiki")
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: 8192,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert len(wiki_manager.calls) == 2
    assert wiki_manager.calls[0]["save_answer"] is False
    assert wiki_manager.calls[1]["save_answer"] is True
    assert wiki_manager.calls[0]["query_context_max_chars_override"] == 22937
    assert wiki_manager.calls[1]["query_context_max_chars_override"] == 22937
    assert wiki_manager.calls[0]["preferred_context_page"] == "sources/source-slug.md"
    assert wiki_manager.calls[1]["preferred_context_page"] == "sources/source-slug.md"
    assert wiki_manager.calls[0]["keep_preferred_context_full"] is True
    assert wiki_manager.calls[1]["keep_preferred_context_full"] is True
    assert wiki_manager.calls[0]["preferred_context_only"] is False
    assert wiki_manager.calls[1]["preferred_context_only"] is False


def test_watcher_uses_default_context_when_window_unknown(tmp_path: Path, monkeypatch):
    wiki_manager = _FakeWikiManager(wiki_dir=tmp_path / "wiki")
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: None,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper-2.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert len(wiki_manager.calls) == 2
    assert wiki_manager.calls[0]["save_answer"] is False
    assert wiki_manager.calls[1]["save_answer"] is True
    assert wiki_manager.calls[0]["query_context_max_chars_override"] is None
    assert wiki_manager.calls[1]["query_context_max_chars_override"] is None
    assert wiki_manager.calls[0]["preferred_context_only"] is False
    assert wiki_manager.calls[1]["preferred_context_only"] is False


def test_watcher_retries_on_fallback_with_reduced_context(tmp_path: Path, monkeypatch):
    sources_dir = tmp_path / "wiki" / "sources"
    sources_dir.mkdir(parents=True)
    # Keep source page small so retries can still reduce below initial context.
    (sources_dir / "source-slug.md").write_text("X" * 4000, encoding="utf-8")

    wiki_manager = _FakeWikiManager(
        probe_sequence=[
            {"used_extractive_fallback": True, "fallback_reason": "repetition"},
            {"used_extractive_fallback": False, "fallback_reason": None},
        ],
        wiki_dir=tmp_path / "wiki",
    )
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: 8192,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
        analysis_retry_on_fallback=True,
        analysis_max_retries=3,
        analysis_retry_reduction=0.5,
        analysis_min_context_chars=8000,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper-3.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert len(wiki_manager.calls) == 3
    assert wiki_manager.calls[0]["save_answer"] is False
    assert wiki_manager.calls[1]["save_answer"] is False
    assert wiki_manager.calls[2]["save_answer"] is True
    assert wiki_manager.calls[0]["query_context_max_chars_override"] == 22937
    assert wiki_manager.calls[1]["query_context_max_chars_override"] == 11468
    assert wiki_manager.calls[2]["query_context_max_chars_override"] == 11468


def test_watcher_final_retry_can_switch_to_source_only(tmp_path: Path, monkeypatch):
    sources_dir = tmp_path / "wiki" / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "source-slug.md").write_text("X" * 20000, encoding="utf-8")

    wiki_manager = _FakeWikiManager(
        probe_sequence=[
            {"used_extractive_fallback": True, "fallback_reason": "repetition"},
            {"used_extractive_fallback": True, "fallback_reason": "repetition"},
        ],
        wiki_dir=tmp_path / "wiki",
    )
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: 8192,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
        analysis_retry_on_fallback=True,
        analysis_max_retries=0,
        analysis_source_only_final_retry=True,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper-4.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert len(wiki_manager.calls) == 3
    assert wiki_manager.calls[0]["save_answer"] is False
    assert wiki_manager.calls[0]["preferred_context_only"] is False
    assert wiki_manager.calls[1]["save_answer"] is False
    assert wiki_manager.calls[1]["preferred_context_only"] is True
    assert wiki_manager.calls[2]["save_answer"] is True
    assert wiki_manager.calls[2]["preferred_context_only"] is True


def test_watcher_can_force_source_only_mode(tmp_path: Path, monkeypatch):
    sources_dir = tmp_path / "wiki" / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "source-slug.md").write_text("X" * 4000, encoding="utf-8")

    wiki_manager = _FakeWikiManager(wiki_dir=tmp_path / "wiki")
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: 8192,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
        analysis_source_only=True,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper-source-only.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert len(wiki_manager.calls) == 2
    assert wiki_manager.calls[0]["save_answer"] is False
    assert wiki_manager.calls[1]["save_answer"] is True
    assert wiki_manager.calls[0]["preferred_context_only"] is True
    assert wiki_manager.calls[1]["preferred_context_only"] is True


def test_watcher_runs_enrichment_on_same_schedule_as_lint(tmp_path: Path, monkeypatch):
    sources_dir = tmp_path / "wiki" / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "source-slug.md").write_text("X" * 1000, encoding="utf-8")

    wiki_manager = _FakeWikiManager(wiki_dir=tmp_path / "wiki")
    ingestor = _FakeIngestor(wiki_manager)

    handler = WikiFileEventHandler(
        ingestor=ingestor,
        contributor="tester",
        auto_analyze=True,
        lint_every=1,
        llm_available_fn=lambda: True,
        llm_context_window_tokens_fn=lambda: 8192,
        analysis_context_fraction=0.70,
        analysis_chars_per_token=4,
    )

    monkeypatch.setattr("core.wiki.watcher.time.sleep", lambda _seconds: None)

    raw_file = tmp_path / "paper-maintenance.txt"
    raw_file.write_text("content", encoding="utf-8")

    handler._process_file(raw_file)

    assert wiki_manager.health_calls == 1
    assert wiki_manager.retry_fallback_calls == 1
    assert wiki_manager.enrich_calls == 1
