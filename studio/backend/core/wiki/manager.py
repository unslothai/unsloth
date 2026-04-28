from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Import types from engine
from .engine import (
    WikiConfig,
    LLMWikiEngine,
    SessionMemoryConfig,
    SessionMemoryState,
    SessionMemoryCompactConfig,
    Message,
    auto_compact_if_needed,
    estimate_message_tokens,
    has_text_blocks,
    count_tool_calls_since,
    has_tool_calls_in_last_assistant_turn,
    should_extract_session_memory,
    adjust_index_to_preserve_api_invariants,
    calculate_messages_to_keep_index,
    try_session_memory_compaction,
)

# llm_fn receives a prompt and returns model text.
LLMFn = Callable[[str], str]


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


_MERGE_MAINTENANCE_DEFAULT_MAX_MERGES = _env_int(
    "UNSLOTH_WIKI_MERGE_MAINTENANCE_MAX_MERGES",
    512,
    minimum = 1,
    maximum = 512,
)


@dataclass
class WikiManager:
    engine: LLMWikiEngine
    # We'll use this to keep track of per-session memory state if needed
    # For now, we'll assume a global or per-user state is managed by the caller

    @classmethod
    def create(cls, vault_root: Path, llm_fn: LLMFn) -> "WikiManager":
        cfg = WikiConfig(vault_root = vault_root)
        return cls(engine = LLMWikiEngine(cfg = cfg, llm_fn = llm_fn))

    def query_rag(
        self,
        question: str,
        query_context_max_chars_override: Optional[int] = None,
        save_answer: bool = True,
        preferred_context_page: Optional[str] = None,
        keep_preferred_context_full: bool = False,
        preferred_context_only: bool = False,
    ) -> Dict:
        """Perform RAG using the wiki engine."""
        return self.engine.query(
            question = question,
            save_answer = save_answer,
            query_context_max_chars_override = query_context_max_chars_override,
            preferred_context_page = preferred_context_page,
            keep_preferred_context_full = keep_preferred_context_full,
            preferred_context_only = preferred_context_only,
        )

    def retrieve_context(
        self,
        question: str,
        max_pages: Optional[int] = None,
        max_chars_per_page: Optional[int] = None,
        include_source_pages: bool = True,
    ) -> Dict:
        """Retrieve top wiki pages and text snippets for prompt injection."""
        ranked = self.engine._rank_pages(
            question,
            include_source_pages = include_source_pages,
        )
        pages_limit = (
            self.engine.cfg.max_context_pages if max_pages is None else max_pages
        )
        chars_limit = (
            self.engine.cfg.max_chars_per_page
            if max_chars_per_page is None
            else max_chars_per_page
        )

        top_pages = ranked if pages_limit <= 0 else ranked[:pages_limit]
        blocks: List[Dict] = []
        for rel_path, score in top_pages:
            page_text = (self.engine.wiki_dir / rel_path).read_text(
                encoding = "utf-8", errors = "ignore"
            )
            blocks.append(
                {
                    "page": rel_path,
                    "score": score,
                    "content": page_text
                    if chars_limit <= 0
                    else page_text[:chars_limit],
                }
            )

        return {
            "status": "ok",
            "question": question,
            "context_pages": [page for page, _ in top_pages],
            "context_blocks": blocks,
        }

    def ingest_content(
        self, title: str, content: str, reference: Optional[str] = None
    ) -> Dict:
        """Ingest new content into the wiki."""
        return self.engine.ingest_source(
            source_title = title, source_text = content, source_ref = reference
        )

    def get_health(self) -> Dict:
        """Check wiki health."""
        return self.engine.lint()

    def enrich_analysis_pages(
        self,
        dry_run: bool = False,
        max_analysis_pages: int = 64,
        fill_gaps_from_web: Optional[bool] = None,
        max_web_gap_queries: Optional[int] = None,
        refresh_non_fallback_oldest_pages: Optional[int] = None,
        repair_answer_links: Optional[bool] = None,
        compact_knowledge_pages: bool = False,
        max_incremental_updates: Optional[int] = None,
    ) -> Dict:
        """Enrich analysis pages using index-driven link suggestions."""
        return self.engine.enrich_analysis_pages(
            dry_run = dry_run,
            max_analysis_pages = max_analysis_pages,
            fill_gaps_from_web = fill_gaps_from_web,
            max_web_gap_queries = max_web_gap_queries,
            refresh_non_fallback_oldest_pages = refresh_non_fallback_oldest_pages,
            repair_answer_links = repair_answer_links,
            compact_knowledge_pages = compact_knowledge_pages,
            max_incremental_updates = max_incremental_updates,
        )

    def retry_fallback_analysis_pages(
        self,
        dry_run: bool = False,
        max_analysis_pages: int = 24,
    ) -> Dict:
        """Retry analysis questions for pages that were previously fallback-generated."""
        return self.engine.retry_fallback_analysis_pages(
            dry_run = dry_run,
            max_analysis_pages = max_analysis_pages,
        )

    def merge_duplicate_knowledge_pages(
        self,
        dry_run: bool = True,
        include_entities: bool = True,
        include_concepts: bool = True,
        similarity_threshold: float = 0.75,
        max_merges: int = _MERGE_MAINTENANCE_DEFAULT_MAX_MERGES,
        semantic_concept_merge: bool = True,
        semantic_merge_writeback: bool = True,
        compact_knowledge_pages: bool = False,
        max_incremental_updates: Optional[int] = None,
    ) -> Dict:
        """Merge duplicate entity/concept pages and rewrite wiki links."""
        return self.engine.merge_duplicate_knowledge_pages(
            dry_run = dry_run,
            include_entities = include_entities,
            include_concepts = include_concepts,
            similarity_threshold = similarity_threshold,
            max_merges = max_merges,
            semantic_concept_merge = semantic_concept_merge,
            semantic_merge_writeback = semantic_merge_writeback,
            compact_knowledge_pages = compact_knowledge_pages,
            max_incremental_updates = max_incremental_updates,
        )
