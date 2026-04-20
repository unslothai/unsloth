# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import json
from pathlib import Path

from core.wiki.engine import LLMWikiEngine, WikiConfig
from core.wiki.ingestor import WikiIngestor
from core.wiki.manager import WikiManager


class _RecordingWikiManager:
    def __init__(self):
        self.calls = []

    def ingest_content(self, title: str, content: str, reference: str | None = None):
        self.calls.append(
            {
                "title": title,
                "content": content,
                "reference": reference,
            }
        )
        return {"status": "ok"}


def test_ingest_file_pdf_extracts_local_text(tmp_path: Path, monkeypatch):
    manager = _RecordingWikiManager()
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    pdf_path = tmp_path / "raw" / "Resume.pdf"
    pdf_path.parent.mkdir(parents = True, exist_ok = True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(
        ingestor, "_extract_pdf_text", lambda _: "John Doe\nSenior ML Engineer"
    )

    title = ingestor.ingest_file(pdf_path)

    assert title == "Resume"
    assert len(manager.calls) == 1
    assert manager.calls[0]["title"] == "Resume"
    assert "Senior ML Engineer" in manager.calls[0]["content"]
    assert manager.calls[0]["reference"] == str(pdf_path.resolve())


def test_ingest_file_skips_ds_store(tmp_path: Path):
    manager = _RecordingWikiManager()
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    metadata = tmp_path / ".DS_Store"
    metadata.write_bytes(b"metadata")

    title = ingestor.ingest_file(metadata)

    assert title is None
    assert manager.calls == []


def test_ingest_pending_raw_files_uses_collision_resistant_titles(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    file_a = tmp_path / "raw" / "repoA" / "README.md"
    file_b = tmp_path / "raw" / "repoB" / "README.md"
    file_a.parent.mkdir(parents = True, exist_ok = True)
    file_b.parent.mkdir(parents = True, exist_ok = True)
    file_a.write_text("repoA content marker", encoding = "utf-8")
    file_b.write_text("repoB content marker", encoding = "utf-8")

    results = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")

    assert len(results) == 2

    source_pages = sorted((tmp_path / "wiki" / "sources").glob("*.md"))
    assert len(source_pages) == 2
    slugs = [page.stem for page in source_pages]
    assert any("repoa-readme" in slug for slug in slugs)
    assert any("repob-readme" in slug for slug in slugs)

    combined_text = "\n".join(page.read_text(encoding = "utf-8") for page in source_pages)
    assert "repoA content marker" in combined_text
    assert "repoB content marker" in combined_text


def test_ingest_pending_raw_files_uses_hash_state_and_reingests_on_change(
    tmp_path: Path,
):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    raw_file = tmp_path / "raw" / "notes.txt"
    raw_file.write_text("initial content", encoding = "utf-8")

    first = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    assert len(first) == 1

    state_path = tmp_path / "raw" / ".ingest_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding = "utf-8"))
    assert str(raw_file.resolve()) in state

    second = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    assert second == []

    raw_file.write_text("updated content", encoding = "utf-8")
    third = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    assert len(third) == 1


def test_ingest_pending_raw_files_skips_sensitive_candidates(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    nested = tmp_path / "raw" / "nested"
    nested.mkdir(parents = True, exist_ok = True)

    valid = nested / "design-notes.md"
    valid.write_text("Retrieval design notes", encoding = "utf-8")

    sensitive = tmp_path / "raw" / "api_token_dump.txt"
    sensitive.write_text("secret token contents", encoding = "utf-8")

    results = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    ingested_paths = {item["source_path"] for item in results}

    assert str(valid.resolve()) in ingested_paths
    assert str(sensitive.resolve()) not in ingested_paths


def test_ingest_pending_raw_files_skips_hidden_subdirectories(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    visible = tmp_path / "raw" / "visible-notes.md"
    visible.write_text("Visible notes", encoding = "utf-8")

    archived = tmp_path / "raw" / ".archive" / "archived-notes.md"
    archived.parent.mkdir(parents = True, exist_ok = True)
    archived.write_text("Archived notes", encoding = "utf-8")

    results = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    ingested_paths = {item["source_path"] for item in results}

    assert str(visible.resolve()) in ingested_paths
    assert str(archived.resolve()) not in ingested_paths


def test_engine_surfaces_extraction_diagnostics(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda prompt: prompt,
    )

    result = engine.ingest_source(
        source_title = "Resume",
        source_text = "John Doe is a machine learning engineer with Python and retrieval experience.",
        source_ref = "file:///tmp/Resume.pdf",
    )

    source_page = tmp_path / "wiki" / f"{result['source_page']}.md"
    text = source_page.read_text(encoding = "utf-8")

    assert "## Extraction Diagnostics" in text
    assert (
        "- reason: llm_json_repaired" in text
        or "- reason: llm_json_parse_failed" in text
        or "- reason: llm_prompt_echo" in text
        or "- reason: llm_garbled_output" in text
    )
    assert "## Source Excerpt" in text
    assert "John Doe" in text


def test_manager_retrieve_context_returns_snippets(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")

    manager.ingest_content(
        title = "Resume",
        content = "John Doe builds Python retrieval pipelines and RAG systems.",
        reference = "resume.txt",
    )
    manager.ingest_content(
        title = "Cooking Notes",
        content = "Sourdough bread recipe and fermentation timings.",
        reference = "cooking.txt",
    )

    result = manager.retrieve_context(
        question = "How does the resume describe Python retrieval work?",
        max_pages = 3,
        max_chars_per_page = 500,
    )

    assert result["status"] == "ok"
    assert result["context_blocks"]
    assert any(
        "python retrieval" in block["content"].lower()
        for block in result["context_blocks"]
    )


def test_manager_retrieve_context_zero_limits_return_full_content(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")

    full_marker = "FULL_CONTENT_MARKER_98765"
    manager.ingest_content(
        title = "Long Source",
        content = "alpha topic\n" + ("x" * 5000) + "\n" + full_marker,
        reference = "long.txt",
    )

    result = manager.retrieve_context(
        question = "alpha topic",
        max_pages = 0,
        max_chars_per_page = 0,
    )

    assert result["status"] == "ok"
    assert result["context_blocks"]
    assert any(full_marker in block["content"] for block in result["context_blocks"])


def test_manager_retrieve_context_can_exclude_source_pages(tmp_path: Path):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha Source\n\nalpha retrieval source content\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "alpha.md").write_text(
        "# Alpha Entity\n\nalpha retrieval entity details\n",
        encoding = "utf-8",
    )

    result = manager.retrieve_context(
        question = "alpha retrieval",
        max_pages = 8,
        max_chars_per_page = 400,
        include_source_pages = False,
    )

    assert result["status"] == "ok"
    assert result["context_pages"]
    assert all(not page.startswith("sources/") for page in result["context_pages"])


def test_rank_pages_can_expand_via_links(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    # Seed one lexically-matching page and one linked page that does not share query terms.
    source_page = tmp_path / "wiki" / "sources" / "alpha.md"
    concept_page = tmp_path / "wiki" / "concepts" / "beta.md"
    source_page.write_text(
        "# Alpha\n\nalpha unique query token\n\nSee [[concepts/beta]].\n",
        encoding = "utf-8",
    )
    concept_page.write_text(
        "# Beta\n\nThis page has deeper details but no lexical overlap.\n",
        encoding = "utf-8",
    )

    engine.cfg.ranking_link_depth = 1
    engine.cfg.ranking_link_fanout = 4
    ranked = engine._rank_pages("alpha unique query token")

    ranked_paths = [rel for rel, _ in ranked]
    assert "sources/alpha.md" in ranked_paths
    assert "concepts/beta.md" in ranked_paths


def test_rank_pages_boosts_entity_targets_for_who_is_queries(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    entity_page = tmp_path / "wiki" / "entities" / "zohair.md"
    source_page = tmp_path / "wiki" / "sources" / "notes.md"
    entity_page.write_text(
        "# Zohair\n\nProfile summary and background details.\n",
        encoding = "utf-8",
    )
    source_page.write_text(
        "# Notes\n\nzohair zohair zohair zohair repeated mention in generic notes.\n",
        encoding = "utf-8",
    )

    ranked = engine._rank_pages("Who is Zohair?")

    assert ranked
    assert ranked[0][0] == "entities/zohair.md"


def test_rank_pages_falls_back_to_recency_when_no_match(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    older_page = tmp_path / "wiki" / "sources" / "older.md"
    newer_page = tmp_path / "wiki" / "sources" / "newer.md"
    older_page.write_text("# Older\n\nGeneral notes.\n", encoding = "utf-8")
    newer_page.write_text("# Newer\n\nMore general notes.\n", encoding = "utf-8")

    os.utime(older_page, (1_700_000_000, 1_700_000_000))
    os.utime(newer_page, (1_900_000_000, 1_900_000_000))

    ranked = engine._rank_pages("qzvjkplm")
    ranked_paths = [rel for rel, _ in ranked]

    assert ranked_paths
    assert ranked_paths[0] == "sources/newer.md"
    assert "sources/older.md" in ranked_paths


def test_rank_pages_llm_rerank_reorders_candidates(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "ordered_pages" in prompt and "INDEX_FILE:" in prompt:
            return '{"ordered_pages": ["sources/beta.md", "sources/alpha.md"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )
    engine.cfg.ranking_llm_rerank_enabled = True
    engine.cfg.ranking_llm_rerank_candidates = 8
    engine.cfg.ranking_llm_rerank_top_n = 2
    engine.cfg.ranking_link_depth = 0

    alpha_page = tmp_path / "wiki" / "sources" / "alpha.md"
    beta_page = tmp_path / "wiki" / "sources" / "beta.md"
    alpha_page.write_text(
        "# Alpha\n\npython retrieval answer notes and ranking clues.\n",
        encoding = "utf-8",
    )
    beta_page.write_text(
        "# Beta\n\npython retrieval answer notes and ranking clues.\n",
        encoding = "utf-8",
    )

    ranked = engine._rank_pages("python retrieval answer")
    ranked_paths = [rel for rel, _ in ranked]

    assert ranked_paths[:2] == ["sources/beta.md", "sources/alpha.md"]


def test_rank_pages_llm_rerank_invalid_output_falls_back_to_deterministic_ranking(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if "ordered_pages" in prompt and "INDEX_FILE:" in prompt:
            return "not valid json and no usable page ids"
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )
    engine.cfg.ranking_llm_rerank_enabled = True
    engine.cfg.ranking_llm_rerank_candidates = 8
    engine.cfg.ranking_llm_rerank_top_n = 2
    engine.cfg.ranking_link_depth = 0

    alpha_page = tmp_path / "wiki" / "sources" / "alpha.md"
    beta_page = tmp_path / "wiki" / "sources" / "beta.md"
    alpha_page.write_text(
        "# Alpha\n\nalpha token token token plus retrieval context.\n",
        encoding = "utf-8",
    )
    beta_page.write_text(
        "# Beta\n\nbeta token token token plus retrieval context.\n",
        encoding = "utf-8",
    )

    ranked = engine._rank_pages("alpha retrieval")
    ranked_paths = [rel for rel, _ in ranked]

    assert ranked_paths
    assert ranked_paths[0] == "sources/alpha.md"
    assert "sources/beta.md" in ranked_paths


def test_rank_pages_does_not_promote_unrelated_entity_on_person_query(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    unrelated_entity = tmp_path / "wiki" / "entities" / "ryoma-sato.md"
    recent_source = tmp_path / "wiki" / "sources" / "recent-notes.md"

    unrelated_entity.write_text(
        "# Ryoma Sato\n\nAuthor profile unrelated to target person.\n",
        encoding = "utf-8",
    )
    recent_source.write_text(
        "# Recent Notes\n\nGeneral context page with no person match.\n",
        encoding = "utf-8",
    )

    # Ensure deterministic recency ordering for zero-match fallback.
    os.utime(unrelated_entity, (1_700_000_000, 1_700_000_000))
    os.utime(recent_source, (1_900_000_000, 1_900_000_000))

    ranked = engine._rank_pages("Who is Zohair?")
    ranked_paths = [rel for rel, _ in ranked]

    assert ranked_paths
    assert ranked_paths[0] == "sources/recent-notes.md"
    assert "entities/ryoma-sato.md" in ranked_paths


def test_query_zero_limits_use_full_context(tmp_path: Path):
    captured_prompt = {"text": ""}

    def _llm(prompt: str) -> str:
        captured_prompt["text"] = prompt
        return "This is a grounded wiki answer with enough detail to pass quality checks. [[sources/alpha]]"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    long_tail = "TAIL_MARKER_UNTRUNCATED_12345"
    alpha_page = tmp_path / "wiki" / "sources" / "alpha.md"
    beta_page = tmp_path / "wiki" / "sources" / "beta.md"
    alpha_page.write_text(
        "# Alpha\n\nalpha topic\n\n" + ("A" * 5000) + "\n" + long_tail + "\n",
        encoding = "utf-8",
    )
    beta_page.write_text(
        "# Beta\n\nalpha topic companion details\n",
        encoding = "utf-8",
    )

    engine.cfg.max_context_pages = 0
    engine.cfg.max_chars_per_page = 0
    engine.cfg.query_context_max_chars = 0
    engine.cfg.ranking_max_chars = 0

    result = engine.query("alpha topic", save_answer = False)

    assert result["status"] == "ok"
    assert "sources/alpha.md" in result["context_pages"]
    assert long_tail in captured_prompt["text"]


def test_query_respects_analysis_exclusion_on_empty_fallback(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "Fallback answer without analysis context.",
    )
    engine.cfg.include_analysis_pages_in_query = False

    (tmp_path / "wiki" / "analysis" / "alpha.md").write_text(
        "# Alpha Analysis\n\nalpha retrieval details\n",
        encoding = "utf-8",
    )

    result = engine.query("alpha retrieval", save_answer = False)

    assert result["status"] == "ok"
    assert result["context_pages"] == []


def test_query_saved_analysis_page_slug_uses_question_topic_terms(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: (
            "FORGE uses graph embeddings for foundational optimization representation "
            "with structured constraints and objective features. [[entities/forge]]"
        ),
    )

    (tmp_path / "wiki" / "entities" / "forge.md").write_text(
        "# Forge\n\nFoundational optimization representations from graph embeddings.\n",
        encoding = "utf-8",
    )

    result = engine.query(
        "What does FORGE describe for optimization embeddings?",
        save_answer = True,
    )

    assert result["status"] == "ok"
    assert result["answer_page"]
    assert "forge" in str(result["answer_page"])
    saved = tmp_path / "wiki" / f"{result['answer_page']}.md"
    assert saved.exists()


def test_query_saved_analysis_page_slug_prefers_llm_title_output(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: (
            "Title: Accelerating Set Cover With Graph Neural Networks\n"
            "Section A: This source explains graph-based optimization signals, "
            "decision constraints, and representation-learning tradeoffs for set cover. "
            "[[sources/12345]]"
        ),
    )

    (tmp_path / "wiki" / "sources" / "12345.md").write_text(
        "# Raw Export 12345\n\nArbitrary raw filename content.\n",
        encoding = "utf-8",
    )

    result = engine.query("Please summarize this source", save_answer = True)

    assert result["answer_page"]
    assert "accelerating-set-cover" in str(result["answer_page"])


def test_query_saved_analysis_page_slug_anchors_to_context_and_is_unique(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: (
            "Grounded answer with adequate lexical diversity and source citation. "
            "[[sources/forge-model-overview]]"
        ),
    )

    (tmp_path / "wiki" / "sources" / "forge-model-overview.md").write_text(
        "# Forge Model Overview\n\nDetails about forge optimization modeling.\n",
        encoding = "utf-8",
    )

    first = engine.query("Please tell me", save_answer = True)
    second = engine.query("Please tell me", save_answer = True)

    assert first["answer_page"]
    assert second["answer_page"]
    assert "forge" in str(first["answer_page"])
    assert first["answer_page"] != second["answer_page"]
    assert str(second["answer_page"]).endswith("-2")


def test_query_saved_analysis_page_compacts_prompt_question_block(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "Source-first summary with grounded details. [[sources/accelerating-set-cover-problems-with-graph-neural-networks]]",
    )

    source_slug = "accelerating-set-cover-problems-with-graph-neural-networks"
    (tmp_path / "wiki" / "sources" / f"{source_slug}.md").write_text(
        "# Accelerating Set Cover Problems with Graph Neural Networks\n\nTechnical source details.\n",
        encoding = "utf-8",
    )

    watcher_style_prompt = (
        "Summarize source 'Accelerating Set Cover Problems with Graph Neural Networks' with a source-first lens.\n"
        f"Primary page to ground on: [[sources/{source_slug}]].\n\n"
        "Focus on:\n"
        "1. What this source is about (2-3 sentences)\n"
        "2. 4-7 concrete key takeaways\n"
        "3. What changed in the wiki after ingest (new or updated entities/concepts)\n"
        "4. Any caveats, uncertainty, or possible extraction gaps\n\n"
        "Output format:\n"
        "- Section A: Brief summary paragraph\n"
        "- Section B: Key takeaways (bullets)\n\n"
        "Requirements:\n"
        "- Cite claims inline with wiki links like [[sources/...]] [[entities/...]] [[concepts/...]]\n"
        f"- Prioritize [[sources/{source_slug}]] over unrelated pages"
    )

    result = engine.query(watcher_style_prompt, save_answer = True)
    assert result["answer_page"]

    analysis_text = (tmp_path / "wiki" / f"{result['answer_page']}.md").read_text(
        encoding = "utf-8"
    )
    assert "## Question\n" in analysis_text
    assert "Focus on:" not in analysis_text
    assert "Output format:" not in analysis_text
    assert "Requirements:" not in analysis_text
    assert (
        "Primary page: [[sources/accelerating-set-cover-problems-with-graph-neural-networks]]."
        in analysis_text
    )


def test_low_unique_ratio_gate_is_less_aggressive_for_valid_repetitive_answers(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    # 10 unique lexical tokens repeated 4x -> ratio = 0.25.
    # Old threshold (0.35) would flag this; current threshold keeps it.
    answer = (
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "[[sources/paper-a]] [[sources/paper-b]]"
    )

    assert engine._low_quality_reason(answer) is None


def test_low_unique_ratio_still_flags_degenerate_output(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    # 4 unique tokens over 48 lexical tokens -> ratio ~= 0.083, should still flag.
    answer = (
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
    )

    assert engine._low_quality_reason(answer) == "low_unique_ratio"


def test_low_unique_ratio_threshold_is_configurable(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    engine.cfg.low_unique_ratio_threshold = 0.30

    answer = (
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
        "method improves accuracy retrieval analysis source context evidence results limitations "
    )

    assert engine._low_quality_reason(answer) == "low_unique_ratio"


def test_low_unique_ratio_min_tokens_is_configurable(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    engine.cfg.low_unique_ratio_min_tokens = 100

    answer = (
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
        "model context model context model context model context "
    )

    assert engine._low_quality_reason(answer) is None


def test_enrich_analysis_pages_prepends_enrichment_section(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "retrieval-pipeline.md").write_text(
        "# Retrieval Pipeline\n\nEntity page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "vector-search.md").write_text(
        "# Vector Search\n\nConcept page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "paper-a.md").write_text(
        "# Paper A\n\nSource page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "index.md").write_text(
        "# Index\n\n"
        "## Sources\n"
        "- [[sources/paper-a]]\n\n"
        "## Entities\n"
        "- [[entities/retrieval-pipeline]]\n\n"
        "## Concepts\n"
        "- [[concepts/vector-search]]\n\n"
        "## Analysis\n"
        "- [[analysis/sample]]\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\nHow does a retrieval pipeline improve vector search quality?\n\n"
        "## Answer\n"
        "A retrieval pipeline can improve vector search relevance through better filtering and ranking.\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(dry_run = False, max_analysis_pages = 10)

    assert report["status"] == "ok"
    assert report["updated_pages"] == 1

    updated = analysis_path.read_text(encoding = "utf-8")
    assert updated.startswith("# Query Result\n\n## Enrichment\n")
    assert "[[entities/retrieval-pipeline]]" in updated
    assert "[[concepts/vector-search]]" in updated

    second_report = engine.enrich_analysis_pages(dry_run = False, max_analysis_pages = 10)
    assert second_report["updated_pages"] == 0


def test_lint_includes_graphify_insights_payload(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nSee [[concepts/beta]].\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "beta.md").write_text(
        "# Beta\n\nBeta details.\n",
        encoding = "utf-8",
    )

    report = engine.lint()
    insights = report["graphify_insights"]

    assert isinstance(insights, dict)
    assert "available" in insights
    assert "god_nodes" in insights
    assert "surprising_connections" in insights


def test_export_graphify_wiki_returns_report_and_writes_index_when_available(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nSee [[concepts/beta]].\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "beta.md").write_text(
        "# Beta\n\nBeta details.\n",
        encoding = "utf-8",
    )

    report = engine.export_graphify_wiki(output_subdir = "graphify-wiki-test")

    assert report["status"] in {"ok", "unavailable"}
    if report["status"] == "ok":
        index_file = Path(report["index_file"])
        assert index_file.exists()
        assert report["articles_written"] >= 1
        assert report["communities"] >= 1
    else:
        assert report.get("reason")


def test_enrich_analysis_pages_dry_run_does_not_edit_file(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "data-ingestion.md").write_text(
        "# Data Ingestion\n\nEntity page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "index.md").write_text(
        "# Index\n\n"
        "## Sources\n- (none)\n\n"
        "## Entities\n- [[entities/data-ingestion]]\n\n"
        "## Concepts\n- (none)\n\n"
        "## Analysis\n- [[analysis/sample]]\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    original = (
        "# Query Result\n\n"
        "## Question\nWhat changed?\n\n"
        "## Answer\nThe data ingestion path now skips hidden files.\n"
    )
    analysis_path.write_text(original, encoding = "utf-8")

    report = engine.enrich_analysis_pages(dry_run = True, max_analysis_pages = 10)

    assert report["updated_pages"] == 1
    assert analysis_path.read_text(encoding = "utf-8") == original


def test_enrich_analysis_pages_can_fill_lint_gaps_from_web(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    engine.lint = lambda: {"missing_concepts": ["retrieval-benchmarking"]}  # type: ignore[assignment]
    engine._web_search_results = (  # type: ignore[assignment]
        lambda _query, _max_results: [
            {
                "title": "Benchmarking retrieval systems",
                "url": "https://example.com/retrieval-benchmarking",
                "snippet": "Retrieval benchmarking evaluates ranking quality with relevance metrics.",
            }
        ]
    )

    report = engine.enrich_analysis_pages(
        dry_run = False,
        max_analysis_pages = 10,
        fill_gaps_from_web = True,
        max_web_gap_queries = 2,
    )

    concept_page = tmp_path / "wiki" / "concepts" / "retrieval-benchmarking.md"
    assert concept_page.exists()

    concept_text = concept_page.read_text(encoding = "utf-8")
    assert "# Retrieval Benchmarking" in concept_text
    assert "## External Sources" in concept_text
    assert "https://example.com/retrieval-benchmarking" in concept_text

    web_gap_fill = report["web_gap_fill"]
    assert web_gap_fill["enabled"] is True
    assert web_gap_fill["lint_missing_concepts"] == 1
    assert web_gap_fill["concepts_created"] == 1
    assert "concepts/retrieval-benchmarking.md" in web_gap_fill["created_pages"]


def test_enrich_analysis_pages_web_gap_fill_dry_run_does_not_write_pages(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    engine.lint = lambda: {"missing_concepts": ["semantic-layering"]}  # type: ignore[assignment]
    engine._web_search_results = (  # type: ignore[assignment]
        lambda _query, _max_results: [
            {
                "title": "Semantic layering overview",
                "url": "https://example.com/semantic-layering",
                "snippet": "Semantic layering organizes abstractions into structured conceptual levels.",
            }
        ]
    )

    report = engine.enrich_analysis_pages(
        dry_run = True,
        max_analysis_pages = 10,
        fill_gaps_from_web = True,
        max_web_gap_queries = 1,
    )

    concept_page = tmp_path / "wiki" / "concepts" / "semantic-layering.md"
    assert not concept_page.exists()

    web_gap_fill = report["web_gap_fill"]
    assert web_gap_fill["enabled"] is True
    assert web_gap_fill["queries_used"] == 1
    assert web_gap_fill["concepts_created"] == 1
    assert "concepts/semantic-layering.md" in web_gap_fill["created_pages"]


def test_retry_fallback_analysis_pages_retries_only_fallback_pages(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "A grounded answer with enough detail and citations. [[sources/alpha]]",
    )

    # Seed at least one retrievable source page so query context is non-empty.
    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nalpha context details for retry\n",
        encoding = "utf-8",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "fallback.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is alpha?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer\n\n"
        "## Fallback Reason\nrepetition\n",
        encoding = "utf-8",
    )

    non_fallback_page = tmp_path / "wiki" / "analysis" / "non-fallback.md"
    non_fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is beta?\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nRegular answer\n",
        encoding = "utf-8",
    )

    report = engine.retry_fallback_analysis_pages(dry_run = False, max_analysis_pages = 20)

    assert report["status"] == "ok"
    assert report["fallback_pages_found"] == 1
    assert report["regenerated_pages"] == 1
    assert report["fallback_still"] == 0
    assert report["retried_pages"] == 1
    assert any(
        item.get("source_page") == "analysis/fallback.md" for item in report["results"]
    )


def test_index_flags_fallback_analysis_pages(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "fallback.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is alpha?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer\n\n"
        "## Fallback Reason\nrepetition\n",
        encoding = "utf-8",
    )

    normal_page = tmp_path / "wiki" / "analysis" / "normal.md"
    normal_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is beta?\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nNormal answer\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    fallback_line = next(
        line for line in index_text.splitlines() if "[[analysis/fallback]]" in line
    )
    normal_line = next(
        line for line in index_text.splitlines() if "[[analysis/normal]]" in line
    )

    assert "[fallback: repetition]" in fallback_line
    assert "[fallback" not in normal_line


def test_retry_fallback_updates_index_when_resolved(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "A grounded answer with sufficient detail and evidence. [[sources/alpha]]",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nalpha context details\n",
        encoding = "utf-8",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "legacy-fallback.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is alpha?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer\n\n"
        "## Fallback Reason\nrepetition\n",
        encoding = "utf-8",
    )

    report = engine.retry_fallback_analysis_pages(dry_run = False, max_analysis_pages = 20)

    assert report["regenerated_pages"] == 1

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    fallback_line = next(
        line
        for line in index_text.splitlines()
        if "[[analysis/legacy-fallback]]" in line
    )
    assert "[fallback-resolved:" in fallback_line


def test_index_analysis_summary_uses_title_and_full_primary_source_link(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "set-cover-summary.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Accelerating Set Cover Problems with Graph Neural Networks' with a source-first lens.\n"
        "Primary page: [[sources/accelerating-set-cover-problems-with-graph-neural-networks]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Accelerating Set Cover via Graph Neural Networks\n"
        "Section A: Technical summary. [[sources/accelerating-set-cover-problems-with-graph-neural-networks]]\n\n"
        "## Context Pages\n"
        "- [[sources/accelerating-set-cover-problems-with-graph-neural-networks]]\n"
        "- [[entities/set-cover]]\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/set-cover-summary]]" in item
    )

    assert "Summarize source" not in line
    assert "Accelerating Set Cover via Graph Neural Networks" in line
    assert (
        "[[sources/accelerating-set-cover-problems-with-graph-neural-networks]]" in line
    )


def test_rebuild_index_omits_sources_when_source_index_disabled(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("UNSLOTH_WIKI_INDEX_INCLUDE_SOURCE_PAGES", "false")

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha Source\n\nsource detail\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "alpha-entity.md").write_text(
        "# Alpha Entity\n\nentity detail\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    assert "## Sources" in index_text
    assert "(omitted by source-exclusion policy)" in index_text
    assert "[[sources/alpha]]" not in index_text
    assert "[[entities/alpha-entity]]" in index_text


def test_llm_rerank_uses_compact_index_without_sources_when_disabled(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("UNSLOTH_WIKI_INDEX_INCLUDE_SOURCE_PAGES", "false")

    captured = {"prompt": ""}

    def _llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"ordered_pages": ["entities/alpha.md"]}'

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha Source\n\nsource detail\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "alpha.md").write_text(
        "# Alpha Entity\n\nentity detail\n",
        encoding = "utf-8",
    )
    engine._rebuild_index()

    reranked = engine._llm_rerank_candidates(
        "alpha",
        [("sources/alpha.md", 1.0), ("entities/alpha.md", 0.8)],
    )

    assert reranked
    prompt = captured["prompt"]
    assert "entities/alpha.md" in prompt
    assert "sources/alpha.md" not in prompt
    assert "(omitted by source-exclusion policy)" in prompt


def test_lint_reports_entity_and_concept_merge_candidates(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "retrieval-pipeline.md").write_text(
        "---\n"
        "title: Retrieval Pipeline\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "retrieval-pipeline-system.md").write_text(
        "---\n"
        "title: Retrieval Pipeline System\n"
        "updated_at: 2026-04-21T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline System\n",
        encoding = "utf-8",
    )

    (tmp_path / "wiki" / "concepts" / "vector-search.md").write_text(
        "---\n"
        "title: Vector Search\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Vector Search\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "vector-search-optimization.md").write_text(
        "---\n"
        "title: Vector Search Optimization\n"
        "updated_at: 2026-04-21T00:00:00+00:00\n"
        "---\n\n"
        "# Vector Search Optimization\n",
        encoding = "utf-8",
    )

    report = engine.lint()

    entity_candidates = report.get("entity_merge_candidates", [])
    concept_candidates = report.get("concept_merge_candidates", [])

    assert entity_candidates
    assert concept_candidates

    assert any(
        {
            item.get("canonical_title"),
            item.get("duplicate_title"),
        }
        == {"Retrieval Pipeline", "Retrieval Pipeline System"}
        for item in entity_candidates
    )
    assert any(
        {
            item.get("canonical_title"),
            item.get("duplicate_title"),
        }
        == {"Vector Search", "Vector Search Optimization"}
        for item in concept_candidates
    )


def test_merge_duplicate_knowledge_pages_dry_run_reports_without_writing(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "retrieval-pipeline.md").write_text(
        "---\n"
        "title: Retrieval Pipeline\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline\n\n"
        "## Summary\n"
        "Legacy retrieval details.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "retrieval-pipeline-system.md").write_text(
        "---\n"
        "title: Retrieval Pipeline System\n"
        "updated_at: 2026-04-21T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline System\n\n"
        "## Summary\n"
        "Canonical retrieval details.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    original_analysis = (
        "# Query Result\n\n"
        "## Question\n"
        "What changed in retrieval?\n\n"
        "## Answer\n"
        "See [[entities/retrieval-pipeline]].\n"
    )
    analysis_path.write_text(original_analysis, encoding = "utf-8")

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = True,
        include_entities = True,
        include_concepts = False,
        similarity_threshold = 0.75,
        max_merges = 8,
    )

    assert report["status"] == "ok"
    assert report["planned_merges"] == 1
    assert report["applied_merges"] == 0
    assert report["rewritten_links"] >= 1
    assert (tmp_path / "wiki" / "entities" / "retrieval-pipeline.md").exists()
    assert analysis_path.read_text(encoding = "utf-8") == original_analysis


def test_merge_duplicate_knowledge_pages_apply_archives_and_rewrites_links(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "retrieval-pipeline.md").write_text(
        "---\n"
        "title: Retrieval Pipeline\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline\n\n"
        "## Summary\n"
        "Legacy retrieval details.\n\n"
        "## Facts\n"
        "- Older architecture note\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "retrieval-pipeline-system.md").write_text(
        "---\n"
        "title: Retrieval Pipeline System\n"
        "updated_at: 2026-04-21T00:00:00+00:00\n"
        "---\n\n"
        "# Retrieval Pipeline System\n\n"
        "## Summary\n"
        "Canonical retrieval details.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "What changed in retrieval?\n\n"
        "## Answer\n"
        "See [[entities/retrieval-pipeline]].\n",
        encoding = "utf-8",
    )

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = False,
        include_entities = True,
        include_concepts = False,
        similarity_threshold = 0.75,
        max_merges = 8,
    )

    assert report["status"] == "ok"
    assert report["applied_merges"] == 1
    assert report["rewritten_links"] >= 1
    assert report["archived_pages"]

    merge = report["merges"][0]
    canonical_rel = str(merge["canonical"])
    duplicate_rel = str(merge["duplicate"])
    archived_rel = str(merge["archived_to"])

    canonical_link = (
        canonical_rel[:-3] if canonical_rel.endswith(".md") else canonical_rel
    )
    duplicate_link = (
        duplicate_rel[:-3] if duplicate_rel.endswith(".md") else duplicate_rel
    )

    assert not (tmp_path / "wiki" / duplicate_rel).exists()
    assert (tmp_path / "wiki" / archived_rel).exists()

    analysis_text = analysis_path.read_text(encoding = "utf-8")
    assert f"[[{canonical_link}]]" in analysis_text
    assert f"[[{duplicate_link}]]" not in analysis_text

    canonical_text = (tmp_path / "wiki" / canonical_rel).read_text(encoding = "utf-8")
    assert "## Merge History" in canonical_text
    assert duplicate_rel in canonical_text

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    assert f"[[{duplicate_link}]]" not in index_text
    assert f"[[{canonical_link}]]" in index_text
