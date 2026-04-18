from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import hashlib
import logging
import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from .ingestor import WikiIngestor
except ImportError:
    from core.wiki.ingestor import WikiIngestor

logger = logging.getLogger(__name__)


class WikiFileEventHandler(FileSystemEventHandler):
    """
    Handles file system events in the wiki raw directory.
    """

    def __init__(
        self,
        ingestor: WikiIngestor,
        contributor: Optional[str] = None,
        auto_analyze: bool = True,
        lint_every: int = 10,
        llm_available_fn: Optional[Callable[[], bool]] = None,
        llm_context_window_tokens_fn: Optional[Callable[[], Optional[int]]] = None,
        analyze_chat_history: bool = False,
        analysis_context_fraction: float = 0.70,
        analysis_chars_per_token: int = 4,
        analysis_retry_on_fallback: bool = True,
        analysis_max_retries: int = 3,
        analysis_retry_reduction: float = 0.5,
        analysis_min_context_chars: int = 8000,
        maintenance_retry_fallback_max_pages: int = 24,
        analysis_source_only: bool = False,
        analysis_source_only_final_retry: bool = True,
    ):
        self.ingestor = ingestor
        self.contributor = contributor
        self.auto_analyze = auto_analyze
        self.lint_every = max(0, int(lint_every))
        self.llm_available_fn = llm_available_fn
        self.llm_context_window_tokens_fn = llm_context_window_tokens_fn
        self.analyze_chat_history = analyze_chat_history
        self.analysis_context_fraction = min(
            max(float(analysis_context_fraction), 0.0), 1.0
        )
        self.analysis_chars_per_token = max(1, int(analysis_chars_per_token))
        self.analysis_retry_on_fallback = bool(analysis_retry_on_fallback)
        self.analysis_max_retries = max(0, int(analysis_max_retries))
        self.analysis_retry_reduction = min(
            max(float(analysis_retry_reduction), 0.1), 0.95
        )
        self.analysis_min_context_chars = max(500, int(analysis_min_context_chars))
        self.maintenance_retry_fallback_max_pages = max(
            0, int(maintenance_retry_fallback_max_pages)
        )
        self.analysis_source_only = bool(analysis_source_only)
        self.analysis_source_only_final_retry = bool(analysis_source_only_final_retry)
        self._analysis_runs = 0
        self._lock = threading.Lock()
        self._processed_mtime_ns: dict[str, int] = {}
        self._processed_hash: dict[str, str] = {}

    def _analysis_context_override_chars(self) -> Optional[int]:
        if (
            self.llm_context_window_tokens_fn is None
            or self.analysis_context_fraction <= 0.0
        ):
            return None

        try:
            context_tokens = self.llm_context_window_tokens_fn()
        except Exception:
            return None

        if context_tokens is None:
            return None

        try:
            tokens = int(context_tokens)
        except (TypeError, ValueError):
            return None

        if tokens <= 0:
            return None

        return max(
            500,
            int(
                tokens * self.analysis_context_fraction * self.analysis_chars_per_token
            ),
        )

    def _reduced_context_override(self, current_chars: Optional[int]) -> Optional[int]:
        if current_chars is None:
            return None
        if current_chars <= self.analysis_min_context_chars:
            return None

        reduced = max(
            self.analysis_min_context_chars,
            int(current_chars * self.analysis_retry_reduction),
        )
        return reduced if reduced < current_chars else None

    def _source_page_chars(self, source_slug: str) -> Optional[int]:
        try:
            wiki_dir = getattr(self.ingestor.wiki_manager.engine, "wiki_dir", None)
            if wiki_dir is None:
                return None
            source_page = Path(wiki_dir) / "sources" / f"{source_slug}.md"
            if not source_page.exists():
                return None
            return len(source_page.read_text(encoding = "utf-8", errors = "ignore"))
        except Exception:
            return None

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha1()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def on_created(self, event):
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self._process_file(Path(event.dest_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def _process_file(self, file_path: Path):
        # Debounce slightly to allow file writing to complete
        time.sleep(0.5)
        if self.ingestor.should_skip_local_file(file_path):
            return
        if file_path.exists():
            try:
                resolved = str(file_path.resolve())
                mtime_ns = file_path.stat().st_mtime_ns
                file_hash = self._compute_file_hash(file_path)
            except OSError:
                return

            with self._lock:
                last_mtime = self._processed_mtime_ns.get(resolved)
                last_hash = self._processed_hash.get(resolved)
                if last_hash == file_hash:
                    return
                if last_mtime == mtime_ns:
                    return
                self._processed_mtime_ns[resolved] = mtime_ns
                self._processed_hash[resolved] = file_hash

            logger.info(f"New file detected in wiki raw directory: {file_path}")
            title = self.ingestor.ingest_file(file_path, contributor = self.contributor)
            if not title:
                return

            if not self.auto_analyze:
                return

            if not self.analyze_chat_history and file_path.stem.lower().startswith(
                "chat_history_"
            ):
                return

            if self.llm_available_fn is not None and not self.llm_available_fn():
                logger.info(
                    "Skipping auto wiki analysis for %s (no active model loaded)",
                    file_path.name,
                )
                return

            source_slug = self.ingestor.wiki_manager.engine._slug(title)
            source_page_rel = f"sources/{source_slug}.md"
            source_chars = self._source_page_chars(source_slug)
            question = (
                f"Summarize source '{title}' with a source-first lens.\n"
                f"Primary page to ground on: [[sources/{source_slug}]].\n\n"
                "Focus on:\n"
                "1. What this source is about (2-3 sentences)\n"
                "2. 4-7 concrete key takeaways\n"
                "3. What changed in the wiki after ingest (new or updated entities/concepts)\n"
                "4. Any caveats, uncertainty, or possible extraction gaps\n\n"
                "Output format:\n"
                "- Title: Summary title (either from the document or rephrased for brevity)\n"
                "- Section A: Brief summary paragraph\n"
                "- Section B: Key takeaways (bullets)\n"
                "- Section C: Wiki updates (bullets)\n"
                "- Section D: Any important equations or formulas (bullets)\n"
                "- Section E: Caveats (bullets)\n"
                "- Section F: Any assumptions (bullets)\n"
                "- Section G: Is this a source or a conversation?\n"
                "- Section H: Any potential disputable claims?\n\n"
                "Requirements:\n"
                "- Cite claims inline with wiki links like [[sources/...]] [[entities/...]] [[concepts/...]]\n"
                "- Keep the response specific and avoid generic filler\n"
                "- Make sure you populate caveats and limitations by looking at the content critically, especially if it's technical. If the source is very clean and straightforward, say so but still include a caveats section with a note to that effect.\n"
                f"- Prioritize [[sources/{source_slug}]] over unrelated pages"
            )
            context_override_chars = self._analysis_context_override_chars()
            if source_chars is not None and context_override_chars is not None:
                context_override_chars = max(context_override_chars, source_chars)
            try:
                attempt_override = context_override_chars
                probe_result = None
                reductions_done = 0
                source_only_mode = self.analysis_source_only
                if source_only_mode and source_chars is not None:
                    attempt_override = (
                        source_chars
                        if attempt_override is None
                        else max(attempt_override, source_chars)
                    )

                while True:
                    probe_result = self.ingestor.wiki_manager.query_rag(
                        question,
                        query_context_max_chars_override = attempt_override,
                        save_answer = False,
                        preferred_context_page = source_page_rel,
                        keep_preferred_context_full = True,
                        preferred_context_only = source_only_mode,
                    )

                    if not probe_result.get("used_extractive_fallback"):
                        break

                    can_reduce = (
                        self.analysis_retry_on_fallback
                        and not source_only_mode
                        and reductions_done < self.analysis_max_retries
                    )
                    if can_reduce:
                        next_override = self._reduced_context_override(attempt_override)
                        if source_chars is not None and next_override is not None:
                            next_override = max(next_override, source_chars)
                        if next_override is not None:
                            logger.info(
                                "Auto wiki analysis fallback for %s (reason=%s). "
                                "Retrying with smaller context (%s -> %s chars).",
                                file_path.name,
                                probe_result.get("fallback_reason"),
                                attempt_override,
                                next_override,
                            )
                            attempt_override = next_override
                            reductions_done += 1
                            continue

                    if (
                        self.analysis_source_only_final_retry
                        and source_chars is not None
                        and not self.analysis_source_only
                        and not source_only_mode
                    ):
                        source_only_mode = True
                        attempt_override = (
                            source_chars
                            if attempt_override is None
                            else max(attempt_override, source_chars)
                        )
                        logger.info(
                            "Auto wiki analysis fallback for %s (reason=%s). "
                            "Final retry with source-only context.",
                            file_path.name,
                            probe_result.get("fallback_reason"),
                        )
                        continue

                    break

                result = self.ingestor.wiki_manager.query_rag(
                    question,
                    query_context_max_chars_override = attempt_override,
                    save_answer = True,
                    preferred_context_page = source_page_rel,
                    keep_preferred_context_full = True,
                    preferred_context_only = source_only_mode,
                )
                answer_page = result.get("answer_page")

                with self._lock:
                    self._analysis_runs += 1
                    run_count = self._analysis_runs

                logger.info(
                    "Auto wiki analysis complete for %s (run=%d, answer_page=%s, "
                    "context_chars_override=%s, source_only=%s, fallback=%s, reason=%s)",
                    file_path.name,
                    run_count,
                    answer_page,
                    attempt_override,
                    source_only_mode,
                    result.get("used_extractive_fallback"),
                    result.get("fallback_reason"),
                )

                if self.lint_every > 0 and run_count % self.lint_every == 0:
                    try:
                        lint_report = self.ingestor.wiki_manager.get_health()
                        logger.info(
                            "Auto wiki lint complete after %d analyses: orphans=%d stale=%d broken=%d",
                            run_count,
                            len(lint_report.get("orphans", [])),
                            len(lint_report.get("stale_pages", [])),
                            len(lint_report.get("broken_links", [])),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Auto wiki lint failed after %d analyses: %s",
                            run_count,
                            exc,
                        )

                    if self.maintenance_retry_fallback_max_pages > 0:
                        try:
                            retry_report = self.ingestor.wiki_manager.retry_fallback_analysis_pages(
                                dry_run = False,
                                max_analysis_pages = self.maintenance_retry_fallback_max_pages,
                            )
                            logger.info(
                                "Auto wiki fallback-retry complete after %d analyses: scanned=%d fallback_found=%d regenerated=%d still_fallback=%d",
                                run_count,
                                int(retry_report.get("scanned_pages", 0)),
                                int(retry_report.get("fallback_pages_found", 0)),
                                int(retry_report.get("regenerated_pages", 0)),
                                int(retry_report.get("fallback_still", 0)),
                            )
                        except Exception as exc:
                            logger.warning(
                                "Auto wiki fallback-retry failed after %d analyses: %s",
                                run_count,
                                exc,
                            )

                    try:
                        enrich_report = (
                            self.ingestor.wiki_manager.enrich_analysis_pages(
                                dry_run = False
                            )
                        )
                        logger.info(
                            "Auto wiki enrichment complete after %d analyses: scanned=%d updated=%d",
                            run_count,
                            int(enrich_report.get("scanned_pages", 0)),
                            int(enrich_report.get("updated_pages", 0)),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Auto wiki enrichment failed after %d analyses: %s",
                            run_count,
                            exc,
                        )
            except Exception as exc:
                logger.warning("Auto wiki analysis failed for %s: %s", file_path, exc)


class WikiIngestionWatcher:
    """
    Monitors the wiki raw directory for new files and triggers ingestion.
    """

    def __init__(
        self,
        ingestor: WikiIngestor,
        raw_dir: Path,
        contributor: Optional[str] = None,
        auto_analyze: bool = True,
        lint_every: int = 10,
        llm_available_fn: Optional[Callable[[], bool]] = None,
        llm_context_window_tokens_fn: Optional[Callable[[], Optional[int]]] = None,
        analyze_chat_history: bool = False,
    ):
        self.ingestor = ingestor
        self.raw_dir = raw_dir
        self.contributor = contributor
        self.observer = Observer()
        try:
            analysis_context_fraction = float(
                os.getenv("UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION", "0.70")
            )
        except ValueError:
            analysis_context_fraction = 0.70

        try:
            analysis_chars_per_token = int(
                os.getenv("UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN", "4")
            )
        except ValueError:
            analysis_chars_per_token = 4

        analysis_retry_on_fallback = os.getenv(
            "UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_ON_FALLBACK", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}

        try:
            analysis_max_retries = int(
                os.getenv("UNSLOTH_WIKI_AUTO_ANALYSIS_MAX_RETRIES", "3")
            )
        except ValueError:
            analysis_max_retries = 3

        try:
            analysis_retry_reduction = float(
                os.getenv("UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_REDUCTION", "0.5")
            )
        except ValueError:
            analysis_retry_reduction = 0.5

        try:
            analysis_min_context_chars = int(
                os.getenv("UNSLOTH_WIKI_AUTO_ANALYSIS_MIN_CONTEXT_CHARS", "8000")
            )
        except ValueError:
            analysis_min_context_chars = 8000

        try:
            maintenance_retry_fallback_max_pages = int(
                os.getenv("UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES", "24")
            )
        except ValueError:
            maintenance_retry_fallback_max_pages = 24

        analysis_source_only = os.getenv(
            "UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY", "false"
        ).strip().lower() not in {"0", "false", "no", "off"}

        analysis_source_only_final_retry = os.getenv(
            "UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}

        self.event_handler = WikiFileEventHandler(
            ingestor,
            contributor,
            auto_analyze = auto_analyze,
            lint_every = lint_every,
            llm_available_fn = llm_available_fn,
            llm_context_window_tokens_fn = llm_context_window_tokens_fn,
            analyze_chat_history = analyze_chat_history,
            analysis_context_fraction = analysis_context_fraction,
            analysis_chars_per_token = analysis_chars_per_token,
            analysis_retry_on_fallback = analysis_retry_on_fallback,
            analysis_max_retries = analysis_max_retries,
            analysis_retry_reduction = analysis_retry_reduction,
            analysis_min_context_chars = analysis_min_context_chars,
            maintenance_retry_fallback_max_pages = maintenance_retry_fallback_max_pages,
            analysis_source_only = analysis_source_only,
            analysis_source_only_final_retry = analysis_source_only_final_retry,
        )

    def start(self):
        """Starts the background observer."""
        self.observer.schedule(self.event_handler, str(self.raw_dir), recursive = False)
        self.observer.start()
        logger.info(f"Started WikiIngestionWatcher monitoring: {self.raw_dir}")

    def stop(self):
        """Stops the background observer."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped WikiIngestionWatcher")
