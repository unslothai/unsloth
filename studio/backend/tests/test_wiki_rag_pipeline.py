# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import json
import ast
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


def test_ingest_pending_raw_files_backfills_hash_for_existing_source_page(
    tmp_path: Path,
):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    raw_file = tmp_path / "raw" / "report.md"
    raw_file.write_text("first ingest content", encoding = "utf-8")

    initial_title = ingestor.ingest_file(raw_file, contributor = "tester")
    assert initial_title

    calls = {"count": 0}

    def _should_not_run(*_args, **_kwargs):
        calls["count"] += 1
        return None

    ingestor.ingest_file = _should_not_run  # type: ignore[method-assign]

    results = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")

    assert results == []
    assert calls["count"] == 0

    state_path = tmp_path / "raw" / ".ingest_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding = "utf-8"))
    assert str(raw_file.resolve()) in state


def test_ingest_pending_raw_files_reingests_changed_content_when_source_mtime_is_newer(
    tmp_path: Path,
):
    manager = WikiManager.create(vault_root = tmp_path, llm_fn = lambda _: "{}")
    ingestor = WikiIngestor(manager, tmp_path / "raw")

    raw_file = tmp_path / "raw" / "notes.txt"
    raw_file.write_text("initial content", encoding = "utf-8")

    first = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    assert len(first) == 1

    state_path = tmp_path / "raw" / ".ingest_state.json"
    first_state = json.loads(state_path.read_text(encoding = "utf-8"))
    previous_hash = first_state[str(raw_file.resolve())]

    source_slug = manager.engine._slug(ingestor._local_source_title(raw_file))
    source_page = tmp_path / "wiki" / "sources" / f"{source_slug}.md"
    assert source_page.exists()
    source_mtime = source_page.stat().st_mtime

    raw_file.write_text("updated content", encoding = "utf-8")
    # Simulate coarse timestamp resolution where source appears as new or newer.
    os.utime(raw_file, (source_mtime, source_mtime))

    second = ingestor.ingest_pending_raw_files(max_files = 8, contributor = "tester")
    assert len(second) == 1

    second_state = json.loads(state_path.read_text(encoding = "utf-8"))
    assert second_state[str(raw_file.resolve())] != previous_hash
    assert "updated content" in source_page.read_text(encoding = "utf-8")


def _load_route_stream_merge_helper() -> object:
    route_file = Path(__file__).resolve().parents[1] / "routes" / "inference.py"
    src = route_file.read_text(encoding = "utf-8")
    module = ast.parse(src)

    helper_def = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_merge_streamed_text_chunks"
    )

    helper_module = ast.Module(body = [helper_def], type_ignores = [])
    namespace: dict[str, object] = {"Any": object}
    exec(compile(helper_module, str(route_file), "exec"), namespace)
    return namespace["_merge_streamed_text_chunks"]


def test_route_stream_merge_helper_handles_cumulative_snapshots():
    merge = _load_route_stream_merge_helper()
    assert callable(merge)

    chunks = ["a", "ab", "abc"]
    assert merge(chunks) == "abc"


def test_route_stream_merge_helper_handles_delta_chunks():
    merge = _load_route_stream_merge_helper()
    assert callable(merge)

    chunks = ["a", "b", "c"]
    assert merge(chunks) == "abc"


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


def test_upsert_knowledge_page_skips_duplicate_incremental_writes(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    page = tmp_path / "wiki" / "entities" / "forge.md"
    initial = (
        "---\n"
        "title: Forge\n"
        "type: entity\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Forge\n\n"
        "## Summary\n"
        "Foundational optimization representation model.\n\n"
        "## Facts\n"
        "- Supports transfer learning\n\n"
        "## Contradictions\n\n"
        "## Sources\n"
        "- [[sources/forge-paper]] (Forge Paper)\n"
    )
    page.write_text(initial, encoding = "utf-8")

    engine._upsert_knowledge_page(
        folder = engine.entities_dir,
        page_name = "Forge",
        page_type = "entity",
        summary = "Foundational optimization representation model.",
        facts = ["Supports transfer learning"],
        contradictions = [],
        source_title = "Forge Paper",
        source_slug = "forge-paper",
        updated_at = "2026-04-20T12:00:00+00:00",
    )

    assert page.read_text(encoding = "utf-8") == initial


def test_upsert_knowledge_page_keeps_single_incremental_section(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    page = tmp_path / "wiki" / "concepts" / "forge-sat.md"
    page.write_text(
        "---\n"
        "title: Forge-Sat\n"
        "type: concept\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Forge-Sat\n\n"
        "## Summary\n"
        "SAT-native variant.\n\n"
        "## Facts\n"
        "- Uses SAT-specific features\n\n"
        "## Contradictions\n\n"
        "## Sources\n"
        "- [[sources/forge-paper]] (Forge Paper)\n",
        encoding = "utf-8",
    )

    engine._upsert_knowledge_page(
        folder = engine.concepts_dir,
        page_name = "Forge-Sat",
        page_type = "concept",
        summary = "SAT-native variant.",
        facts = ["Achieved highest clustering NMI"],
        contradictions = [],
        source_title = "Forge Paper",
        source_slug = "forge-paper",
        updated_at = "2026-04-20T13:00:00+00:00",
    )
    first = page.read_text(encoding = "utf-8")

    engine._upsert_knowledge_page(
        folder = engine.concepts_dir,
        page_name = "Forge-Sat",
        page_type = "concept",
        summary = "SAT-native variant.",
        facts = ["Achieved highest clustering NMI"],
        contradictions = [],
        source_title = "Forge Paper",
        source_slug = "forge-paper",
        updated_at = "2026-04-20T14:00:00+00:00",
    )
    second = page.read_text(encoding = "utf-8")

    assert first == second
    assert second.count("## Incremental Updates") == 1
    assert second.count("- Achieved highest clustering NMI") == 1


def test_upsert_knowledge_page_caps_incremental_updates(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )
    engine.cfg.knowledge_max_incremental_updates = 2

    page = tmp_path / "wiki" / "entities" / "forge.md"
    page.write_text(
        "---\n"
        "title: Forge\n"
        "type: entity\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Forge\n\n"
        "## Summary\n"
        "Foundational optimization representation model.\n\n"
        "## Facts\n"
        "- Supports transfer learning\n\n"
        "## Contradictions\n\n"
        "## Sources\n"
        "- [[sources/forge-paper]] (Forge Paper)\n",
        encoding = "utf-8",
    )

    for idx, fact in enumerate(["Fact One", "Fact Two", "Fact Three"], start = 1):
        engine._upsert_knowledge_page(
            folder = engine.entities_dir,
            page_name = "Forge",
            page_type = "entity",
            summary = "Foundational optimization representation model.",
            facts = [fact],
            contradictions = [],
            source_title = "Forge Paper",
            source_slug = "forge-paper",
            updated_at = f"2026-04-20T1{idx}:00:00+00:00",
        )

    text = page.read_text(encoding = "utf-8")
    updates = engine._extract_markdown_section(text, "Incremental Updates")

    assert updates.count("### New facts") == 2
    assert "- Fact Three" in updates
    assert "- Fact Two" in updates
    assert "- Fact One" not in updates


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


def test_rank_pages_link_expansion_can_use_llm_selector(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "link expansion selector" in prompt:
            return '{"ordered_links": ["L002"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    source_page = tmp_path / "wiki" / "sources" / "alpha.md"
    preferred_page = tmp_path / "wiki" / "concepts" / "beta.md"
    lexical_page = tmp_path / "wiki" / "concepts" / "gamma.md"

    source_page.write_text(
        "# Alpha\n\n"
        "alpha unique query token appears in this source.\n\n"
        "See [[concepts/gamma]] and [[concepts/beta]].\n",
        encoding = "utf-8",
    )
    preferred_page.write_text(
        "# Beta\n\nBeta details without lexical overlap.\n",
        encoding = "utf-8",
    )
    lexical_page.write_text(
        "# Gamma\n\nContains alpha unique query token for lexical fallback preference.\n",
        encoding = "utf-8",
    )

    engine.cfg.ranking_llm_rerank_enabled = False
    engine.cfg.ranking_link_depth = 1
    engine.cfg.ranking_link_fanout = 1

    ranked = engine._rank_pages("alpha unique query token")
    ranked_paths = [rel for rel, _ in ranked]

    assert "concepts/beta.md" in ranked_paths


def test_rank_pages_link_expansion_llm_selector_invalid_does_not_expand(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "link expansion selector" in prompt:
            return "not-json"
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    source_page = tmp_path / "wiki" / "sources" / "alpha.md"
    preferred_page = tmp_path / "wiki" / "concepts" / "beta.md"
    lexical_page = tmp_path / "wiki" / "concepts" / "gamma.md"

    source_page.write_text(
        "# Alpha\n\n"
        "alpha unique query token appears in this source.\n\n"
        "See [[concepts/gamma]] and [[concepts/beta]].\n",
        encoding = "utf-8",
    )
    preferred_page.write_text(
        "# Beta\n\nBeta details without lexical overlap.\n",
        encoding = "utf-8",
    )
    lexical_page.write_text(
        "# Gamma\n\nUnrelated topic details with no relevant overlap.\n",
        encoding = "utf-8",
    )

    engine.cfg.ranking_llm_rerank_enabled = False
    engine.cfg.ranking_link_depth = 1
    engine.cfg.ranking_link_fanout = 1

    ranked = engine._rank_pages("alpha unique query token")
    ranked_paths = [rel for rel, _ in ranked]

    assert "concepts/gamma.md" not in ranked_paths
    assert "concepts/beta.md" not in ranked_paths


def test_rank_pages_boosts_entity_targets_for_who_is_queries(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "entity intent parser" in prompt:
            return '{"is_entity_lookup": true, "target": "Zohair"}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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


def test_entity_query_focus_uses_llm_parser_when_available(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "entity intent parser" in prompt:
            return '{"is_entity_lookup": true, "target": "Jane Doe"}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    terms, slug = engine._entity_query_focus("Could you profile this person for me?")

    assert slug == "jane-doe"
    assert "jane" in terms
    assert "doe" in terms


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


def test_rank_pages_llm_rerank_does_not_append_deterministic_tail(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "ordered_pages" in prompt and "INDEX_FILE:" in prompt:
            return '{"ordered_pages": ["sources/beta.md"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )
    engine.cfg.ranking_llm_rerank_enabled = True
    engine.cfg.ranking_llm_rerank_candidates = 8
    engine.cfg.ranking_llm_rerank_top_n = 3
    engine.cfg.max_context_pages = 3
    engine.cfg.ranking_link_depth = 0

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\npython retrieval answer notes and ranking clues.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "beta.md").write_text(
        "# Beta\n\npython retrieval answer notes and ranking clues.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "gamma.md").write_text(
        "# Gamma\n\npython retrieval answer notes and ranking clues.\n",
        encoding = "utf-8",
    )

    ranked = engine._rank_pages("python retrieval answer")
    ranked_paths = [rel for rel, _ in ranked]

    assert ranked_paths == ["sources/beta.md"]


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
    def _llm(prompt: str) -> str:
        if "enrichment link selector for wiki analysis maintenance" in prompt:
            return '{"selected_ids":["E001"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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


def test_enrich_link_selection_can_use_llm_selector(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "enrichment link selector for wiki analysis maintenance" in prompt:
            return '{"selected_links":["concepts/currency-debasement"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    selected = engine._select_enrichment_links(
        analysis_text = "Retrieval pipeline details for ranking and context assembly.",
        candidates = [
            "concepts/retrieval-pipeline",
            "concepts/currency-debasement",
        ],
        existing_links = set(),
        limit = 1,
        group_name = "concepts",
    )

    assert selected == ["concepts/currency-debasement"]


def test_enrich_link_selection_llm_invalid_returns_empty_in_strict_mode(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "enrichment link selector for wiki analysis maintenance" in prompt:
            return "not-json"
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    selected = engine._select_enrichment_links(
        analysis_text = "Retrieval pipeline details for ranking and context assembly.",
        candidates = [
            "concepts/retrieval-pipeline",
            "concepts/currency-debasement",
        ],
        existing_links = set(),
        limit = 1,
        group_name = "concepts",
    )

    assert selected == []


def test_enrich_link_selection_llm_empty_does_not_fallback_to_lexical(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if "enrichment link selector for wiki analysis maintenance" in prompt:
            return '{"selected_ids":[],"selected_links":[]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    selected = engine._select_enrichment_links(
        analysis_text = "Retrieval pipeline details for ranking and context assembly.",
        candidates = [
            "concepts/retrieval-pipeline",
            "concepts/currency-debasement",
        ],
        existing_links = set(),
        limit = 1,
        group_name = "concepts",
    )

    assert selected == []


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
    def _llm(prompt: str) -> str:
        if "enrichment link selector for wiki analysis maintenance" in prompt:
            return '{"selected_ids":["E001"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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


def test_enrich_repairs_broken_links_only_in_maintenance_sections(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "valid-entity.md").write_text(
        "# Valid Entity\n\nEntity page.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Answer\n"
        "Answer body keeps unresolved references unchanged: [[concepts/ghost-answer-link]].\n\n"
        "## Context Pages\n"
        "- [[entities/valid-entity]]\n"
        "- [[concepts/missing-context]]\n\n"
        "## Enrichment\n"
        "- generated_at: 2026-04-26T00:00:00+00:00\n"
        "- strategy: index-driven enrichment from index.md link inventory\n"
        "### Related Entities\n"
        "- [[entities/missing-enrichment]]\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(dry_run = False, max_analysis_pages = 10)

    repair = report.get("analysis_link_repair", {})
    assert repair.get("repair_answer_links_enabled") is False
    assert repair.get("repaired_pages") == 1
    assert repair.get("removed_links") == 2

    updated = analysis_path.read_text(encoding = "utf-8")
    assert "[[entities/valid-entity]]" in updated
    assert "[[concepts/missing-context]]" not in updated
    assert "[[entities/missing-enrichment]]" not in updated
    # Safe mode: answer prose is not rewritten by link-repair maintenance.
    assert "[[concepts/ghost-answer-link]]" in updated

    broken_targets = {
        item.get("target") for item in engine.lint().get("broken_links", [])
    }
    assert "concepts/ghost-answer-link.md" in broken_targets
    assert "concepts/missing-context.md" not in broken_targets
    assert "entities/missing-enrichment.md" not in broken_targets


def test_enrich_can_repair_broken_links_in_answer_when_enabled(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "valid-entity.md").write_text(
        "# Valid Entity\n\nEntity page.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Answer\n"
        "Aggressive mode can repair unresolved references in prose: [[concepts/ghost-answer-link]].\n"
        "A valid link should remain: [[entities/valid-entity]].\n\n"
        "## Context Pages\n"
        "- [[entities/valid-entity]]\n"
        "- [[concepts/missing-context]]\n\n"
        "## Enrichment\n"
        "### Related Entities\n"
        "- [[entities/missing-enrichment]]\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(
        dry_run = False,
        max_analysis_pages = 10,
        repair_answer_links = True,
    )

    repair = report.get("analysis_link_repair", {})
    assert repair.get("repair_answer_links_enabled") is True
    assert repair.get("repaired_pages") == 1
    assert repair.get("removed_links") == 3

    updated = analysis_path.read_text(encoding = "utf-8")
    assert "[[entities/valid-entity]]" in updated
    assert "[[concepts/ghost-answer-link]]" not in updated
    assert "[[concepts/missing-context]]" not in updated
    assert "[[entities/missing-enrichment]]" not in updated

    broken_targets = {
        item.get("target") for item in engine.lint().get("broken_links", [])
    }
    assert "concepts/ghost-answer-link.md" not in broken_targets
    assert "concepts/missing-context.md" not in broken_targets
    assert "entities/missing-enrichment.md" not in broken_targets


def test_enrich_repair_answer_links_handles_backslash_literals(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "valid-entity.md").write_text(
        "# Valid Entity\n\nEntity page.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "backslash-answer.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Answer\n"
        "Math notation should remain literal: x \\in S.\n"
        "Broken link should be removed: [[concepts/ghost-answer-link]].\n"
        "Valid link should remain: [[entities/valid-entity]].\n\n"
        "## Context Pages\n"
        "- [[entities/valid-entity]]\n\n"
        "## Enrichment\n"
        "### Related Entities\n"
        "- [[entities/valid-entity]]\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(
        dry_run = False,
        max_analysis_pages = 10,
        repair_answer_links = True,
    )

    repair = report.get("analysis_link_repair", {})
    assert repair.get("repair_answer_links_enabled") is True
    assert repair.get("repaired_pages") == 1
    assert repair.get("removed_links") == 1

    updated = analysis_path.read_text(encoding = "utf-8")
    assert "x \\in S" in updated
    assert "[[concepts/ghost-answer-link]]" not in updated
    assert "[[entities/valid-entity]]" in updated


def test_enrich_link_repair_dry_run_reports_without_writing(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    original = (
        "# Query Result\n\n"
        "## Context Pages\n"
        "- [[concepts/missing-context]]\n\n"
        "## Enrichment\n"
        "### Related Concepts\n"
        "- [[concepts/missing-enrichment]]\n"
    )
    analysis_path.write_text(original, encoding = "utf-8")

    report = engine.enrich_analysis_pages(dry_run = True, max_analysis_pages = 10)

    repair = report.get("analysis_link_repair", {})
    assert repair.get("repaired_pages") == 1
    assert repair.get("removed_links") == 2
    assert analysis_path.read_text(encoding = "utf-8") == original


def test_enrich_analysis_pages_can_fill_lint_gaps_from_web(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "web research planner" in prompt:
            return '{"queries":["retrieval benchmarking ranking metrics"]}'
        if "selecting the best external sources" in prompt:
            return '{"selected_ids":["R001"]}'
        return (
            "Source summary: retrieval benchmarking compares ranking quality, "
            "recall tradeoffs, and evaluation setup details across datasets."
        )

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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
    engine._fetch_external_page_text = (  # type: ignore[assignment]
        lambda _url, max_chars: (
            "Retrieval benchmarking measures ranking quality, precision, recall, and nDCG across curated relevance datasets."
        )
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
    assert "## External Source Summaries" in concept_text
    assert "https://example.com/retrieval-benchmarking" in concept_text

    analysis_pages = sorted((tmp_path / "wiki" / "analysis").glob("*.md"))
    assert analysis_pages
    analysis_text = analysis_pages[0].read_text(encoding = "utf-8")
    assert "Summarize source" in analysis_text
    assert "Primary page: [[sources/" in analysis_text

    web_gap_fill = report["web_gap_fill"]
    assert web_gap_fill["enabled"] is True
    assert web_gap_fill["lint_missing_concepts"] == 1
    assert web_gap_fill["concepts_created"] == 1
    assert web_gap_fill["external_sources_ingested"] == 1
    assert web_gap_fill["external_summary_pages_created"] == 1
    assert web_gap_fill["created_summary_pages"]
    assert "concepts/retrieval-benchmarking.md" in web_gap_fill["created_pages"]


def test_enrich_analysis_pages_web_gap_fill_dry_run_does_not_write_pages(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if "web research planner" in prompt:
            return '{"queries":["semantic layering systems overview"]}'
        if "selecting the best external sources" in prompt:
            return '{"selected_ids":["R001"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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
    assert web_gap_fill["external_sources_ingested"] == 0
    assert web_gap_fill["external_summary_pages_created"] == 0
    assert web_gap_fill["created_summary_pages"] == []
    assert "concepts/semantic-layering.md" in web_gap_fill["created_pages"]


def test_llm_web_gap_discovery_uses_planner_and_selector(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if "web research planner" in prompt:
            return '{"queries":["retrieval benchmarking ranking metrics"]}'
        if "selecting the best external sources" in prompt:
            return '{"selected_ids":["R002"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    seen_queries: list[str] = []

    def _fake_search(query: str, _max_results: int):
        seen_queries.append(query)
        return [
            {
                "title": "Result A",
                "url": "https://example.com/a",
                "snippet": "General overview of retrieval.",
            },
            {
                "title": "Result B",
                "url": "https://example.com/b",
                "snippet": "Detailed benchmarking metrics, datasets, and evaluation setup.",
            },
        ]

    engine._web_search_results = _fake_search  # type: ignore[assignment]

    selected, meta = engine._llm_web_discover_results_for_concept(
        concept_slug = "retrieval-benchmarking",
        query_budget = 2,
        max_results = 1,
    )

    assert seen_queries == ["retrieval benchmarking ranking metrics"]
    assert selected
    assert selected[0]["url"] == "https://example.com/b"
    assert meta["plan_status"] == "ok"
    assert meta["selector_status"] == "ok"


def test_llm_web_gap_discovery_ignores_planner_direct_results(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if "web research planner" in prompt:
            return (
                '{"queries":["liquidity squeeze mechanics"],'
                '"direct_results":[{"title":"LLM Injected",'
                '"url":"https://example.com/llm-injected",'
                '"snippet":"should be ignored"}]}'
            )
        if "selecting the best external sources" in prompt:
            return '{"selected_ids":["R001"]}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    seen_queries: list[str] = []

    def _fake_search(query: str, _max_results: int):
        seen_queries.append(query)
        return [
            {
                "title": "DDGS Result",
                "url": "https://example.com/ddgs-result",
                "snippet": "Market liquidity and short squeeze mechanics overview.",
            }
        ]

    engine._web_search_results = _fake_search  # type: ignore[assignment]

    selected, meta = engine._llm_web_discover_results_for_concept(
        concept_slug = "liquidity-squeeze",
        query_budget = 1,
        max_results = 1,
    )

    assert seen_queries == ["liquidity squeeze mechanics"]
    assert selected
    assert selected[0]["url"] == "https://example.com/ddgs-result"
    assert meta["queries_consumed"] == 1
    assert meta["direct_results"] == 0


def test_external_source_summary_uses_watcher_style_source_first_prompt(
    tmp_path: Path,
    monkeypatch,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    monkeypatch.setattr(
        engine,
        "_fetch_external_page_text",
        lambda _url, max_chars: "alpha source content"[:max_chars],
    )

    monkeypatch.setattr(
        engine,
        "ingest_source",
        lambda source_title, source_text, source_ref = None: {
            "status": "ok",
            "source_page": "sources/alpha-source",
        },
    )

    captured: dict[str, str] = {}

    def _fake_query(
        question: str,
        save_answer: bool = True,
        query_context_max_chars_override = None,
        preferred_context_page = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ):
        captured["question"] = question
        return {
            "status": "ok",
            "answer_page": "analysis/alpha-source-summary",
            "used_extractive_fallback": False,
        }

    monkeypatch.setattr(engine, "query", _fake_query)

    report = engine._ingest_and_summarize_external_source(
        concept_title = "Alpha Concept",
        search_result = {
            "title": "Alpha Source",
            "url": "https://example.com/alpha",
            "snippet": "Alpha overview",
        },
    )

    assert report["status"] == "ok"
    question = captured["question"]
    assert "Primary page to ground on: [[sources/alpha-source]]." in question
    assert (
        "- Section I: Is this information date/time sensitive? If yes, print timestamp."
        in question
    )


def test_external_source_summary_reuses_existing_source_and_summary(
    tmp_path: Path,
    monkeypatch,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_slug = "external-source-inflation-wikipedia-en-wikipedia-org-6e41185e"
    source_page = tmp_path / "wiki" / "sources" / f"{source_slug}.md"
    source_page.write_text(
        "---\n"
        "title: External Source: Inflation - Wikipedia [en.wikipedia.org#6e41185e]\n"
        "type: source\n"
        "source_ref: https://en.wikipedia.org/wiki/Inflation\n"
        "ingested_at: 2026-04-28T00:00:00+00:00\n"
        "---\n\n"
        "# Inflation\n\n"
        "## Summary\n"
        "Existing source page content.\n",
        encoding = "utf-8",
    )

    analysis_slug = "2026-04-28-inflation-wikipedia-source-first-len-primary"
    analysis_page = tmp_path / "wiki" / "analysis" / f"{analysis_slug}.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Inflation - Wikipedia' with a source-first lens.\n"
        f"Primary page to ground on: [[sources/{source_slug}]].\n\n"
        "## Answer Mode\n"
        "llm\n\n"
        "## Answer\n"
        "Existing source-first summary.\n\n"
        "## Context Pages\n"
        f"- [[sources/{source_slug}]]\n",
        encoding = "utf-8",
    )

    def _unexpected_fetch(_url: str, _max_chars: int) -> str:
        raise AssertionError("Should not fetch external content for reused source")

    def _unexpected_ingest(*_args, **_kwargs):
        raise AssertionError("Should not ingest duplicate external source")

    def _unexpected_query(*_args, **_kwargs):
        raise AssertionError("Should not regenerate duplicate source-first summary")

    monkeypatch.setattr(engine, "_fetch_external_page_text", _unexpected_fetch)
    monkeypatch.setattr(engine, "ingest_source", _unexpected_ingest)
    monkeypatch.setattr(engine, "query", _unexpected_query)

    report = engine._ingest_and_summarize_external_source(
        concept_title = "Inflation",
        search_result = {
            "title": "Inflation - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Inflation#history",
            "snippet": "Inflation overview from encyclopedia.",
        },
    )

    assert report["status"] == "ok"
    assert report["source_page"] == f"sources/{source_slug}"
    assert report["summary_page"] == f"analysis/{analysis_slug}"
    assert report["reused_source"] is True
    assert report["reused_summary"] is True

    assert len(list((tmp_path / "wiki" / "analysis").glob("*.md"))) == 1


def test_enrich_analysis_pages_can_compact_knowledge_updates(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    entity_page = tmp_path / "wiki" / "entities" / "forge.md"
    entity_page.write_text(
        "---\n"
        "title: Forge\n"
        "type: entity\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Forge\n\n"
        "## Summary\nStable summary.\n\n"
        "## Facts\n- Initial fact\n\n"
        "## Contradictions\n\n"
        "## Sources\n- [[sources/forge-paper]]\n\n"
        "## Incremental Updates\n\n"
        "### Update 1\n- one\n\n"
        "### Update 2\n- two\n\n"
        "### Update 3\n- three\n\n"
        "### Update 4\n- four\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(
        dry_run = False,
        max_analysis_pages = 10,
        compact_knowledge_pages = True,
        max_incremental_updates = 2,
    )

    compaction = report.get("knowledge_compaction", {})
    assert compaction.get("enabled") is True
    assert compaction.get("compacted_pages") == 1

    updated = entity_page.read_text(encoding = "utf-8")
    updates = engine._extract_markdown_section(updated, "Incremental Updates")
    assert updates.count("### ") == 2
    assert "### Update 4" in updates
    assert "### Update 3" in updates
    assert "### Update 1" not in updates


def test_refresh_oldest_non_fallback_analysis_pages_skips_fallback_and_refreshes_oldest(
    tmp_path: Path,
):
    refreshed_answer = (
        "Refreshed analysis from current wiki context with grounded evidence, "
        "trend synthesis, and updated implications tied to sources. [[sources/alpha]]"
    )
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: refreshed_answer,
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nAlpha source context used for refresh decisions.\n",
        encoding = "utf-8",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "fallback.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat was fallback output?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer text.\n\n"
        "## Fallback Reason\ntoo_short\n",
        encoding = "utf-8",
    )

    oldest_non_fallback = tmp_path / "wiki" / "analysis" / "oldest-non-fallback.md"
    oldest_non_fallback.write_text(
        "# Query Result\n\n"
        "## Question\nSummarize the latest alpha source implications.\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nStale answer that should be refreshed.\n\n"
        "## Context Pages\n- [[sources/alpha]]\n",
        encoding = "utf-8",
    )

    newer_non_fallback = tmp_path / "wiki" / "analysis" / "newer-non-fallback.md"
    newer_non_fallback.write_text(
        "# Query Result\n\n"
        "## Question\nWhat changed recently in alpha?\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nRecent answer text.\n\n"
        "## Context Pages\n- [[sources/alpha]]\n",
        encoding = "utf-8",
    )

    # fallback is oldest overall, but non-fallback refresh must ignore it.
    os.utime(fallback_page, (100.0, 100.0))
    os.utime(oldest_non_fallback, (200.0, 200.0))
    os.utime(newer_non_fallback, (300.0, 300.0))

    report = engine.refresh_oldest_non_fallback_analysis_pages(
        dry_run = False,
        max_analysis_pages = 1,
    )

    assert report["status"] == "ok"
    assert report["enabled"] is True
    assert report["requested_pages"] == 1
    assert report["candidate_pages"] == 1
    assert report["refreshed_pages"] == 1
    assert report["skipped_refresh_fallback"] == 0
    assert report["results"][0]["page"] == "analysis/oldest-non-fallback.md"

    refreshed_text = oldest_non_fallback.read_text(encoding = "utf-8")
    assert "## Refresh Status" in refreshed_text
    assert "oldest non-fallback analysis refresh" in refreshed_text
    assert refreshed_answer in refreshed_text

    fallback_text = fallback_page.read_text(encoding = "utf-8")
    assert "extractive-fallback" in fallback_text


def test_enrich_analysis_pages_reports_non_fallback_refresh(tmp_path: Path):
    refreshed_answer = (
        "Detailed refreshed synthesis with concrete evidence, timeline framing, "
        "and source-grounded implications for downstream planning. [[sources/alpha]]"
    )
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: refreshed_answer,
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nAlpha source context with evidence details.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "alpha-team.md").write_text(
        "# Alpha Team\n\nEntity page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "alpha-planning.md").write_text(
        "# Alpha Planning\n\nConcept page.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "index.md").write_text(
        "# Index\n\n"
        "## Sources\n- [[sources/alpha]]\n\n"
        "## Entities\n- [[entities/alpha-team]]\n\n"
        "## Concepts\n- [[concepts/alpha-planning]]\n\n"
        "## Analysis\n- [[analysis/sample]]\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "sample.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\nHow should alpha planning adapt to recent source evidence?\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nInitial answer.\n",
        encoding = "utf-8",
    )

    report = engine.enrich_analysis_pages(
        dry_run = False,
        max_analysis_pages = 10,
        refresh_non_fallback_oldest_pages = 1,
    )

    refresh_report = report.get("non_fallback_refresh", {})
    assert refresh_report.get("enabled") is True
    assert refresh_report.get("requested_pages") == 1
    assert refresh_report.get("refreshed_pages") == 1


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
    assert report["results"][0]["new_answer_page"] == "analysis/fallback"
    assert any(
        item.get("source_page") == "analysis/fallback.md" for item in report["results"]
    )

    fallback_text = fallback_page.read_text(encoding = "utf-8")
    assert "## Answer Mode\nllm" in fallback_text
    assert "## Fallback Reason" not in fallback_text
    assert "## Retry Status" in fallback_text
    assert "- status: resolved_in_place" in fallback_text
    assert "- resolved_by: [[analysis/fallback]]" in fallback_text
    assert not (tmp_path / "wiki" / "analysis" / "fallback-2.md").exists()


def test_retry_fallback_analysis_pages_reduces_context_before_regeneration(
    tmp_path: Path,
    monkeypatch,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(
            vault_root = tmp_path,
            query_context_max_chars = 20000,
            analysis_max_retries = 3,
            analysis_retry_reduction = 0.5,
            analysis_min_context_chars = 4000,
            analysis_source_only = False,
            analysis_source_only_final_retry = False,
        ),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nalpha context details for retry\n",
        encoding = "utf-8",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "fallback-adaptive.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is alpha?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer\n\n"
        "## Fallback Reason\nrepetition\n\n"
        "## Context Pages\n- [[sources/alpha]]\n",
        encoding = "utf-8",
    )

    call_history = []

    def _fake_query(
        question: str,
        save_answer: bool = True,
        query_context_max_chars_override = None,
        preferred_context_page = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ):
        call_history.append(
            {
                "question": question,
                "save_answer": save_answer,
                "chars": query_context_max_chars_override,
                "preferred_context_page": preferred_context_page,
                "keep_preferred_context_full": keep_preferred_context_full,
                "preferred_context_only": preferred_context_only,
            }
        )
        if (
            query_context_max_chars_override is None
            or query_context_max_chars_override > 6000
        ):
            return {
                "used_extractive_fallback": True,
                "fallback_reason": "repetition",
            }
        return {
            "used_extractive_fallback": False,
            "answer": "Detailed regenerated answer with grounded evidence. [[sources/alpha]]",
            "context_pages": ["sources/alpha.md"],
            "query_context_max_chars": query_context_max_chars_override,
        }

    monkeypatch.setattr(engine, "query", _fake_query)

    report = engine.retry_fallback_analysis_pages(dry_run = False, max_analysis_pages = 20)

    assert report["fallback_pages_found"] == 1
    assert report["regenerated_pages"] == 1
    assert report["fallback_still"] == 0

    result = report["results"][0]
    assert result["status"] == "regenerated"
    assert result["retries_attempted"] == 2
    assert result["context_chars_override"] == 5000
    assert result["source_only"] is False
    assert result["new_answer_page"] == "analysis/fallback-adaptive"

    assert [entry["chars"] for entry in call_history] == [20000, 10000, 5000]
    assert all(entry["question"] == "What is alpha?" for entry in call_history)
    assert all(
        entry["preferred_context_page"] == "sources/alpha" for entry in call_history
    )
    assert all(entry["keep_preferred_context_full"] is True for entry in call_history)
    assert all(entry["preferred_context_only"] is False for entry in call_history)


def test_query_source_first_uses_primary_source_only_context(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(
            vault_root = tmp_path,
            max_context_pages = 0,
            max_chars_per_page = 0,
            query_context_max_chars = 12000,
        ),
        llm_fn = lambda _: "Grounded source summary. [[sources/alpha]]",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nAlpha source details for source-first summary.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "analysis" / "alpha-noise.md").write_text(
        "# Query Result\n\n## Question\nSummarize source 'Alpha'\n\n## Answer\nNoise.\n",
        encoding = "utf-8",
    )

    result = engine.query(
        "Summarize source 'Alpha' with a source-first lens. Primary page: [[sources/alpha]].",
        save_answer = False,
    )

    assert result["context_pages"] == ["sources/alpha.md"]


def test_retry_fallback_prefers_primary_source_link_from_question(
    tmp_path: Path,
    monkeypatch,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n\nalpha source body\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "beta.md").write_text(
        "# Beta\n\nbeta source body\n",
        encoding = "utf-8",
    )

    fallback_page = tmp_path / "wiki" / "analysis" / "fallback-source-link.md"
    fallback_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Alpha' with a source-first lens. Primary page: [[sources/alpha]].\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer with unrelated source [[sources/beta]].\n\n"
        "## Fallback Reason\nrepetition\n\n"
        "## Context Pages\n- [[sources/beta]]\n",
        encoding = "utf-8",
    )

    call_history = []

    def _fake_query(
        question: str,
        save_answer: bool = True,
        query_context_max_chars_override = None,
        preferred_context_page = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ):
        call_history.append(
            {
                "question": question,
                "preferred_context_page": preferred_context_page,
                "keep_preferred_context_full": keep_preferred_context_full,
                "preferred_context_only": preferred_context_only,
            }
        )
        return {"used_extractive_fallback": False}

    monkeypatch.setattr(engine, "query", _fake_query)

    report = engine.retry_fallback_analysis_pages(dry_run = True, max_analysis_pages = 20)

    assert report["fallback_pages_found"] == 1
    assert report["regenerated_pages"] == 1
    assert call_history
    assert call_history[0]["preferred_context_page"] == "sources/alpha"


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


def test_lint_normalizes_md_suffixed_links_and_ignores_log_page_links(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "entities" / "graph-model.md").write_text(
        "# Graph Model\n\nEntity page.\n",
        encoding = "utf-8",
    )

    (tmp_path / "wiki" / "analysis" / "md-suffix-links.md").write_text(
        "# Query Result\n\n"
        "## Question\nWhat is graph model?\n\n"
        "## Answer\n"
        "See [[entities/graph-model.md]] and [[entities/graph-model.md.md]].\n",
        encoding = "utf-8",
    )

    # log.md can contain free-form user text snippets, which should not affect
    # wiki broken-link linting.
    (tmp_path / "wiki" / "log.md").write_text(
        "# Log\n\n"
        "## [2026-04-27] query | [[sources/2406-\n"
        "- Result page: [[analysis/some-result]]\n",
        encoding = "utf-8",
    )

    report = engine.lint()
    broken_links = report.get("broken_links", [])

    assert not any(
        item.get("source") == "analysis/md-suffix-links.md" for item in broken_links
    )
    assert not any(item.get("source") == "log.md" for item in broken_links)


def test_retry_fallback_skips_pages_marked_resolved_by(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    resolved_page = tmp_path / "wiki" / "analysis" / "resolved-fallback.md"
    resolved_page.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is alpha?\n\n"
        "## Answer Mode\nextractive-fallback\n\n"
        "## Answer\nFallback answer\n\n"
        "## Fallback Reason\nrepetition\n\n"
        "## Retry Status\n"
        "- status: superseded\n"
        "- resolved_by: [[analysis/alpha-v2]]\n"
        "- resolved_at: 2026-04-27T00:00:00+00:00\n",
        encoding = "utf-8",
    )

    report = engine.retry_fallback_analysis_pages(dry_run = True, max_analysis_pages = 10)

    assert report["fallback_pages_found"] == 0
    assert report["retried_pages"] == 0


def test_index_flags_source_first_llm_with_missing_sections_as_fallback(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_slug = "alpha-source"
    (tmp_path / "wiki" / "sources" / f"{source_slug}.md").write_text(
        "# Alpha Source\n\nDetails.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "source-first-llm.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Alpha Source' with a source-first lens. "
        "Primary page: [[sources/alpha-source]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Alpha Source Summary\n"
        "Section A: A brief summary tied to [[sources/alpha-source]].\n"
        "Section B: - Takeaway one.\n\n"
        "## Context Pages\n"
        "- [[sources/alpha-source]]\n",
        encoding = "utf-8",
    )

    page_text = analysis_path.read_text(encoding = "utf-8")
    assert engine._analysis_page_uses_fallback(page_text) is True

    engine._rebuild_index()
    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    flagged_line = next(
        line
        for line in index_text.splitlines()
        if "[[analysis/source-first-llm]]" in line
    )
    assert "[fallback: missing_watcher_sections:" in flagged_line

    report = engine.retry_fallback_analysis_pages(dry_run = True, max_analysis_pages = 10)
    assert report["fallback_pages_found"] == 1


def test_source_first_llm_heading_sections_with_sectino_typo_not_flagged_fallback(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_slug = "alpha-source"
    (tmp_path / "wiki" / "sources" / f"{source_slug}.md").write_text(
        "# Alpha Source\n\nDetails.\n",
        encoding = "utf-8",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "source-first-heading-typo.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Alpha Source' with a source-first lens. "
        "Primary page: [[sources/alpha-source]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Alpha Source Summary\n\n"
        "## Sectino A\n"
        "A brief summary tied to [[sources/alpha-source]].\n\n"
        "## Section B\n- Key findings.\n\n"
        "## Section C\n- Supporting evidence.\n\n"
        "## Section D\n- Method notes.\n\n"
        "## Section E\n- Limitations.\n\n"
        "## Section F\n- Comparison context.\n\n"
        "## Section G\n- Relevance to the graph.\n\n"
        "## Section H\n- Open questions.\n\n"
        "## Section I\n- Confidence and caveats.\n\n"
        "## Retrieval Diagnostics\n"
        "- llm_rerank_enabled: true\n\n"
        "## Context Pages\n"
        "- [[sources/alpha-source]]\n",
        encoding = "utf-8",
    )

    page_text = analysis_path.read_text(encoding = "utf-8")
    assert engine._analysis_missing_watcher_sections_reason(page_text) is None
    assert engine._analysis_page_uses_fallback(page_text) is False

    engine._rebuild_index()
    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/source-first-heading-typo]]" in item
    )
    assert "[fallback:" not in line


def test_non_source_first_llm_page_without_sections_is_not_fallback(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    analysis_path = tmp_path / "wiki" / "analysis" / "normal-llm.md"
    analysis_path.write_text(
        "# Query Result\n\n"
        "## Question\nWhat is beta?\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\nNormal free-form answer without section labels.\n",
        encoding = "utf-8",
    )

    page_text = analysis_path.read_text(encoding = "utf-8")
    assert engine._analysis_page_uses_fallback(page_text) is False

    report = engine.retry_fallback_analysis_pages(dry_run = True, max_analysis_pages = 10)
    assert report["fallback_pages_found"] == 0


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


def test_index_analysis_summary_ignores_template_placeholder_title(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "source-first-placeholder.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'FORGE Framework' with a source-first lens. "
        "Primary page: [[sources/2508-20330v4]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Section A: Brief summary paragraph\n"
        "Section A: Placeholder-like output.\n\n"
        "## Context Pages\n"
        "- [[sources/2508-20330v4]]\n\n"
        "## Retry Status\n"
        "- status: resolved_in_place\n"
        "- resolved_by: [[analysis/source-first-placeholder]]\n"
        "- resolved_at: 2026-04-28T22:00:00+00:00\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/source-first-placeholder]]" in item
    )

    assert "Section A: Brief summary paragraph" not in line
    assert "FORGE Framework" in line
    assert "[[sources/2508-20330v4]]" in line
    assert "[fallback-resolved:" in line


def test_index_analysis_summary_replaces_identifier_source_title_with_source_page_title(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_page = tmp_path / "wiki" / "sources" / "2508-20330v4.md"
    source_page.write_text(
        "---\n"
        "title: FORGE: Foundational Optimization with Representation Graph Engineering\n"
        "type: source\n"
        "source_ref: https://arxiv.org/abs/2508.20330v4\n"
        "ingested_at: 2026-04-28T22:00:00+00:00\n"
        "---\n\n"
        "# FORGE: Foundational Optimization with Representation Graph Engineering\n\n"
        "## Summary\n"
        "FORGE introduces a graph-based optimization framework for robust reasoning.\n",
        encoding = "utf-8",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "forge-summary.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source '2508.20330v4' with a source-first lens. "
        "Primary page: [[sources/2508-20330v4]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: 2508.20330v4\n"
        "Section A: A concise summary of FORGE.\n\n"
        "## Context Pages\n"
        "- [[sources/2508-20330v4]]\n\n"
        "## Retry Status\n"
        "- status: resolved_in_place\n"
        "- resolved_by: [[analysis/forge-summary]]\n"
        "- resolved_at: 2026-04-28T22:30:00+00:00\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item for item in index_text.splitlines() if "[[analysis/forge-summary]]" in item
    )

    assert "2508.20330v4 | primary" not in line
    assert "FORGE: Foundational Optimization with Representation Graph Engineering" in line
    assert "[[sources/2508-20330v4]]" in line
    assert "[fallback-resolved:" in line


def test_index_analysis_summary_replaces_generic_chat_history_title_with_source_summary(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_page = tmp_path / "wiki" / "sources" / "chat-history-20260420-184704-930406.md"
    source_page.write_text(
        "---\n"
        "title: Chat History Log from April 20 2026 Session 184704\n"
        "type: source\n"
        "source_ref: chat-history://2026-04-20/184704\n"
        "ingested_at: 2026-04-29T00:00:00+00:00\n"
        "---\n\n"
        "# Chat History Log from April 20 2026 Session 184704\n\n"
        "## Summary\n"
        "QLoRA memory bottlenecks and adapter merge trade-offs for 24GB GPUs.\n",
        encoding = "utf-8",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "chat-history-summary.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Chat History Log from April 20 2026 Session 184704' with a source-first lens. "
        "Primary page: [[sources/chat-history-20260420-184704-930406]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Chat History Log from April 20 2026 Session 184704\n"
        "Section A: Discusses practical fine-tuning constraints and merge strategy.\n\n"
        "## Context Pages\n"
        "- [[sources/chat-history-20260420-184704-930406]]\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/chat-history-summary]]" in item
    )

    assert "Chat History Log from April 20 2026 Session 184704" not in line
    assert "QLoRA memory bottlenecks and adapter merge trade-offs for 24GB GPUs" in line
    assert "[[sources/chat-history-20260420-184704-930406]]" in line


def test_index_analysis_summary_uses_answer_excerpt_when_chat_title_is_generic(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    source_page = tmp_path / "wiki" / "sources" / "chat-history-20260422-153443-261090.md"
    source_page.write_text(
        "---\n"
        "title: Chat History Log from April 22 2026 Session 153443\n"
        "type: source\n"
        "source_ref: chat-history://2026-04-22/153443\n"
        "ingested_at: 2026-04-29T00:00:00+00:00\n"
        "---\n\n"
        "# Chat History Log from April 22 2026 Session 153443\n",
        encoding = "utf-8",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "chat-history-fallback-answer.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source 'Chat History Log from April 22 2026 Session 153443' with a source-first lens. "
        "Primary page: [[sources/chat-history-20260422-153443-261090]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: Chat History Log from April 22 2026 Session 153443\n"
        "Section A: Diagnosing duplicate wiki source ingestion during watcher bursts.\n"
        "Section B: Key takeaways.\n\n"
        "## Context Pages\n"
        "- [[sources/chat-history-20260422-153443-261090]]\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/chat-history-fallback-answer]]" in item
    )

    assert "Chat History Log from April 22 2026 Session 153443" not in line
    assert "Diagnosing duplicate wiki source ingestion during watcher bursts" in line
    assert "[[sources/chat-history-20260422-153443-261090]]" in line


def test_index_analysis_summary_can_generate_llm_title_when_flag_enabled(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if "concise index title for a wiki analysis page" in prompt:
            return '{"title":"FORGE Graph Optimization Pretraining Overview"}'
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )
    engine.cfg.index_llm_title_on_rebuild = True

    analysis_page = tmp_path / "wiki" / "analysis" / "forge-llm-index-title.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source '2508.20330v4' with a source-first lens. "
        "Primary page: [[sources/2508-20330v4]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "The paper introduces FORGE (Foundational Optimization Representations from Graph Embeddings)...\n\n"
        "## Context Pages\n"
        "- [[sources/2508-20330v4]]\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/forge-llm-index-title]]" in item
    )

    assert "FORGE Graph Optimization Pretraining Overview" in line
    assert "[[sources/2508-20330v4]]" in line


def test_index_analysis_summary_llm_title_invalid_falls_back_when_flag_enabled(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if "concise index title for a wiki analysis page" in prompt:
            return "not-json"
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )
    engine.cfg.index_llm_title_on_rebuild = True

    source_page = tmp_path / "wiki" / "sources" / "2508-20330v4.md"
    source_page.write_text(
        "---\n"
        "title: FORGE: Foundational Optimization with Representation Graph Engineering\n"
        "type: source\n"
        "source_ref: https://arxiv.org/abs/2508.20330v4\n"
        "ingested_at: 2026-04-28T22:00:00+00:00\n"
        "---\n\n"
        "# FORGE: Foundational Optimization with Representation Graph Engineering\n",
        encoding = "utf-8",
    )

    analysis_page = tmp_path / "wiki" / "analysis" / "forge-llm-index-fallback.md"
    analysis_page.write_text(
        "# Query Result\n\n"
        "## Question\n"
        "Summarize source '2508.20330v4' with a source-first lens. "
        "Primary page: [[sources/2508-20330v4]].\n\n"
        "## Answer Mode\nllm\n\n"
        "## Answer\n"
        "Title: 2508.20330v4\n"
        "Section A: A concise summary of FORGE.\n\n"
        "## Context Pages\n"
        "- [[sources/2508-20330v4]]\n",
        encoding = "utf-8",
    )

    engine._rebuild_index()

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    line = next(
        item
        for item in index_text.splitlines()
        if "[[analysis/forge-llm-index-fallback]]" in item
    )

    assert "FORGE: Foundational Optimization with Representation Graph Engineering" in line
    assert "[[sources/2508-20330v4]]" in line


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
    def _llm(prompt: str) -> str:
        if (
            "semantic duplicate merge planner" in prompt
            and "Page kind: entities" in prompt
        ):
            return (
                '{"merges":['
                '{"canonical_id":"M001","duplicate_id":"M002","confidence":0.88,'
                '"reason":"Same entity naming variant"}'
                "]}"
            )
        if "semantic concept merge planner" in prompt:
            return (
                '{"merges":['
                '{"canonical_id":"C001","duplicate_id":"C002","confidence":0.9,'
                '"reason":"Same concept naming variant"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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


def test_lint_semantic_filters_missing_concepts(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if "semantic filter for wiki concept maintenance" in prompt:
            return (
                "{"
                '"keep_missing":["retrieval-benchmarking"],'
                '"related_to_existing":[{"slug":"instances","existing":"instance","reason":"plural variant of existing concept"}],'
                '"reject":[{"slug":"business","reason":"generic non-concept term"}]'
                "}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    (tmp_path / "wiki" / "concepts" / "instance.md").write_text(
        "# Instance\n",
        encoding = "utf-8",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "beta.md").write_text(
        "# Beta\n",
        encoding = "utf-8",
    )

    # Force lexical candidates with one true missing concept, one related
    # pluralization, and one generic noisy term.
    engine._top_concepts = (  # type: ignore[assignment]
        lambda _text, limit: ["instances", "business", "retrieval benchmarking"]
    )

    report = engine.lint()

    assert report.get("missing_concepts") == ["retrieval-benchmarking"]
    semantic = report.get("semantic_missing_concepts", {})
    assert semantic.get("status") == "ok"
    assert semantic.get("related_to_existing") == 1
    assert semantic.get("rejected_candidates") == 1


def test_lint_missing_concepts_semantic_filter_strict_on_invalid_output(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    (tmp_path / "wiki" / "concepts" / "instance.md").write_text(
        "# Instance\n",
        encoding = "utf-8",
    )

    (tmp_path / "wiki" / "sources" / "alpha.md").write_text(
        "# Alpha\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "sources" / "beta.md").write_text(
        "# Beta\n",
        encoding = "utf-8",
    )

    engine._top_concepts = (  # type: ignore[assignment]
        lambda _text, limit: ["instances", "business", "retrieval benchmarking"]
    )

    report = engine.lint()

    assert report.get("missing_concepts", []) == []
    semantic = report.get("semantic_missing_concepts", {})
    assert semantic.get("status") == "strict_semantic_only"
    assert semantic.get("reason") == "semantic_missing_schema_invalid"


def test_merge_duplicate_knowledge_pages_dry_run_reports_without_writing(
    tmp_path: Path,
):
    def _llm(prompt: str) -> str:
        if (
            "semantic duplicate merge planner" in prompt
            and "Page kind: entities" in prompt
        ):
            return (
                '{"merges":['
                '{"canonical_id":"M001","duplicate_id":"M002","confidence":0.9,'
                '"reason":"Entity alias"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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
    def _llm(prompt: str) -> str:
        if (
            "semantic duplicate merge planner" in prompt
            and "Page kind: entities" in prompt
        ):
            return (
                '{"merges":['
                '{"canonical_id":"M001","duplicate_id":"M002","confidence":0.9,'
                '"reason":"Entity alias"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
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
    assert canonical_text.startswith("---\n")
    assert canonical_text.count("\n---\n") == 1
    assert canonical_text.count("## Merge History") == 1
    assert canonical_text.index("## Merge History") > canonical_text.index(
        "# Retrieval Pipeline System"
    )
    assert "## Merge History" in canonical_text
    assert duplicate_rel in canonical_text

    index_text = (tmp_path / "wiki" / "index.md").read_text(encoding = "utf-8")
    assert f"[[{duplicate_link}]]" not in index_text
    assert f"[[{canonical_link}]]" in index_text


def test_replace_wikilinks_with_map_preserves_alias_and_heading_fragments(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    original = (
        "See [[entities/retrieval-pipeline]].\n"
        "See [[entities/retrieval-pipeline#timeline]].\n"
        "See [[entities/retrieval-pipeline#timeline|legacy profile]].\n"
        "See [[entities/retrieval-pipeline.md#details|legacy details]].\n"
    )
    updated, rewritten = engine._replace_wikilinks_with_map(
        original,
        {
            "entities/retrieval-pipeline": "entities/retrieval-pipeline-system",
        },
    )

    assert rewritten == 4
    assert "[[entities/retrieval-pipeline-system]]" in updated
    assert "[[entities/retrieval-pipeline-system#timeline]]" in updated
    assert "[[entities/retrieval-pipeline-system#timeline|legacy profile]]" in updated
    assert "[[entities/retrieval-pipeline-system#details|legacy details]]" in updated
    assert "[[entities/retrieval-pipeline#timeline|legacy profile]]" not in updated


def test_merge_candidates_handle_instance_pluralization(tmp_path: Path):
    def _llm(prompt: str) -> str:
        if "semantic concept merge planner" in prompt:
            return (
                '{"merges":['
                '{"canonical_id":"C001","duplicate_id":"C002","confidence":0.92,'
                '"reason":"Plural variant of same concept"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = _llm,
    )

    singular = tmp_path / "wiki" / "concepts" / "instance.md"
    plural = tmp_path / "wiki" / "concepts" / "instances.md"

    singular.write_text(
        "---\n"
        "title: Instance\n"
        "updated_at: 2026-04-27T20:00:00+00:00\n"
        "---\n\n"
        "# Instance\n",
        encoding = "utf-8",
    )
    plural.write_text(
        "---\n"
        "title: Instances\n"
        "updated_at: 2026-04-27T21:00:00+00:00\n"
        "---\n\n"
        "# Instances\n",
        encoding = "utf-8",
    )

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = True,
        include_entities = False,
        include_concepts = True,
        similarity_threshold = 0.75,
        max_merges = 8,
    )

    assert report["concept_candidates"] >= 1
    assert report["planned_merges"] >= 1

    concept_merges = [
        merge
        for merge in report.get("merges", [])
        if merge.get("canonical", "").startswith("concepts/")
        and merge.get("duplicate", "").startswith("concepts/")
    ]
    assert concept_merges

    merged_pair_pages = {
        concept_merges[0]["canonical"],
        concept_merges[0]["duplicate"],
    }
    assert merged_pair_pages == {"concepts/instance.md", "concepts/instances.md"}


def test_semantic_concept_merge_candidates_use_llm_pass(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if "semantic concept merge planner" in prompt:
            return (
                '{"merges":['
                '{"canonical_id":"C001","duplicate_id":"C002","confidence":0.91,'
                '"reason":"Same concept, alias naming"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    (tmp_path / "wiki" / "concepts" / "lattice-routing.md").write_text(
        "---\n"
        "title: Lattice Routing\n"
        "updated_at: 2026-04-27T10:00:00+00:00\n"
        "---\n\n"
        "# Lattice Routing\n\n"
        "## Summary\nA routing method that uses lattice state transitions.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "concepts" / "policy-distillation.md").write_text(
        "---\n"
        "title: Policy Distillation\n"
        "updated_at: 2026-04-27T09:00:00+00:00\n"
        "---\n\n"
        "# Policy Distillation\n\n"
        "## Summary\nA synthetic alias page intentionally used for semantic-merge regression.\n",
        encoding = "utf-8",
    )

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = True,
        include_entities = False,
        include_concepts = True,
        similarity_threshold = 0.8,
        max_merges = 8,
        semantic_concept_merge = True,
        semantic_merge_writeback = False,
    )

    assert report["semantic_concept_merge_enabled"] is True
    assert report["semantic_concept_candidates"] >= 1
    assert report["planned_merges"] >= 1

    merge = report["merges"][0]
    assert merge["canonical"].startswith("concepts/")
    assert merge["duplicate"].startswith("concepts/")
    assert merge["reason"].startswith("semantic-llm")


def test_semantic_entity_merge_candidates_use_llm_pass(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if (
            "semantic duplicate merge planner" in prompt
            and "Page kind: entities" in prompt
        ):
            return (
                '{"merges":['
                '{"canonical_id":"M001","duplicate_id":"M002","confidence":0.89,'
                '"reason":"Alias pages for the same entity"}'
                "]}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    (tmp_path / "wiki" / "entities" / "alpha-expert.md").write_text(
        "---\n"
        "title: Alpha Expert\n"
        "updated_at: 2026-04-27T10:00:00+00:00\n"
        "---\n\n"
        "# Alpha Expert\n\n"
        "## Summary\nProfiles a specialist in alpha methods.\n",
        encoding = "utf-8",
    )
    (tmp_path / "wiki" / "entities" / "a-expert-profile.md").write_text(
        "---\n"
        "title: A Expert Profile\n"
        "updated_at: 2026-04-27T09:00:00+00:00\n"
        "---\n\n"
        "# A Expert Profile\n\n"
        "## Summary\nAlternative naming for the same person profile.\n",
        encoding = "utf-8",
    )

    candidates = engine._merge_candidates_for_folder(
        tmp_path / "wiki" / "entities",
        "entities",
        similarity_threshold = 0.8,
    )

    assert candidates
    assert candidates[0]["canonical"].startswith("entities/")
    assert candidates[0]["duplicate"].startswith("entities/")
    assert str(candidates[0]["reason"]).startswith("semantic-llm")


def test_semantic_merge_writeback_updates_canonical_content(tmp_path: Path):
    def llm_fn(prompt: str) -> str:
        if "semantic concept merge planner" in prompt:
            return (
                '{"merges":['
                '{"canonical_id":"C001","duplicate_id":"C002","confidence":0.95,'
                '"reason":"Equivalent concept pages"}'
                "]}"
            )
        if "semantic concept merge writer" in prompt:
            return (
                "{"
                '"merged_summary":"Unified semantic summary for alpha graph.",'
                '"merged_facts":["Supports weighted path traversal","Used in scheduler internals"],'
                '"merged_contradictions":["No contradictions currently confirmed"],'
                '"merged_sources":["[[sources/alpha-paper]]","[[sources/alpha-notes]]"],'
                '"confidence":0.88,'
                '"rationale":"Both pages describe the same artifact with naming variation"'
                "}"
            )
        return "{}"

    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = llm_fn,
    )

    canonical_path = tmp_path / "wiki" / "concepts" / "alpha-graph.md"
    duplicate_path = tmp_path / "wiki" / "concepts" / "alpha-graph-alias.md"

    canonical_path.write_text(
        "---\n"
        "title: Alpha Graph\n"
        "updated_at: 2026-04-27T10:00:00+00:00\n"
        "---\n\n"
        "# Alpha Graph\n\n"
        "## Summary\nOriginal summary.\n\n"
        "## Facts\n- Existing fact\n\n"
        "## Contradictions\n- none\n\n"
        "## Sources\n- [[sources/alpha-paper]]\n",
        encoding = "utf-8",
    )
    duplicate_path.write_text(
        "---\n"
        "title: Alpha Graph Alias\n"
        "updated_at: 2026-04-27T09:00:00+00:00\n"
        "---\n\n"
        "# Alpha Graph Alias\n\n"
        "## Summary\nAlias page summary.\n\n"
        "## Facts\n- Alias fact\n\n"
        "## Contradictions\n- none\n\n"
        "## Sources\n- [[sources/alpha-notes]]\n",
        encoding = "utf-8",
    )

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = False,
        include_entities = False,
        include_concepts = True,
        similarity_threshold = 0.8,
        max_merges = 8,
        semantic_concept_merge = True,
        semantic_merge_writeback = True,
    )

    assert report["applied_merges"] == 1
    assert report["semantic_merge_writeback_enabled"] is True

    updated = canonical_path.read_text(encoding = "utf-8")
    assert "Unified semantic summary for alpha graph." in updated
    assert "Supports weighted path traversal" in updated
    assert "Used in scheduler internals" in updated
    assert "[[sources/alpha-notes]]" in updated
    assert "## Merge History" in updated
    assert "semantic_summary_applied: true" in updated


def test_merge_duplicate_normalizes_misplaced_frontmatter_blocks(tmp_path: Path):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    malformed_canonical = (
        "## Merge History\n"
        "### 2026-04-20T19:59:25.086376+00:00 merged concepts/forge-mip.md\n"
        "- similarity: 1.0\n"
        "- archived_to: .archive/concepts/forge-mip.md\n\n"
        "---\n"
        "title: Forge-Mip-Sat\n"
        "type: concept\n"
        "updated_at: 2026-04-20T13:58:27.476368+00:00\n"
        "---\n\n"
        "# Forge-Mip-Sat\n\n"
        "## Summary\n"
        "A feature-adapted transfer variant of the Forge model.\n"
    )
    duplicate_text = (
        "---\n"
        "title: Forge-Sat\n"
        "updated_at: 2026-04-20T20:18:11.964246+00:00\n"
        "---\n\n"
        "# Forge-Sat\n\n"
        "## Summary\n"
        "A SAT-native foundational model variant.\n\n"
        "## Facts\n"
        "- Retains the Forge architecture\n\n"
        "## Contradictions\n"
        "- none\n\n"
        "## Sources\n"
        "- [[sources/transfer-learning]]\n"
    )

    merged = engine._merge_canonical_with_duplicate(
        malformed_canonical,
        duplicate_text,
        duplicate_rel = "concepts/forge-sat.md",
        archived_rel = ".archive/concepts/forge-sat.md",
        similarity = 1.0,
    )

    assert merged.startswith("---\n")
    assert merged.count("\n---\n") == 1
    assert merged.count("## Merge History") == 1
    assert "title: Forge-Mip-Sat" in merged
    assert "concepts/forge-sat.md" in merged


def test_merge_maintenance_can_compact_knowledge_updates_without_merges(
    tmp_path: Path,
):
    engine = LLMWikiEngine(
        cfg = WikiConfig(vault_root = tmp_path),
        llm_fn = lambda _: "{}",
    )

    entity_page = tmp_path / "wiki" / "entities" / "forge.md"
    entity_page.write_text(
        "---\n"
        "title: Forge\n"
        "type: entity\n"
        "updated_at: 2026-04-20T00:00:00+00:00\n"
        "---\n\n"
        "# Forge\n\n"
        "## Summary\nStable summary.\n\n"
        "## Facts\n- Initial fact\n\n"
        "## Contradictions\n\n"
        "## Sources\n- [[sources/forge-paper]]\n\n"
        "## Incremental Updates\n\n"
        "### Update 1\n- one\n\n"
        "### Update 2\n- two\n\n"
        "### Update 3\n- three\n",
        encoding = "utf-8",
    )

    report = engine.merge_duplicate_knowledge_pages(
        dry_run = False,
        include_entities = True,
        include_concepts = False,
        similarity_threshold = 0.75,
        max_merges = 8,
        compact_knowledge_pages = True,
        max_incremental_updates = 2,
    )

    assert report["status"] == "ok"
    assert report["planned_merges"] == 0
    compaction = report.get("knowledge_compaction", {})
    assert compaction.get("enabled") is True
    assert compaction.get("compacted_pages") == 1

    updated = entity_page.read_text(encoding = "utf-8")
    updates = engine._extract_markdown_section(updated, "Incremental Updates")
    assert updates.count("### ") == 2
    assert "### Update 3" in updates
    assert "### Update 1" not in updates
