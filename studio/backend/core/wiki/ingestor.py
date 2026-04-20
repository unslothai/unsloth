from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging
import sys
import importlib
import urllib.parse
import json
import hashlib


def _import_graphify_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Monorepo fallback: allow running backend without installing graphify as a wheel.
        graphify_root = Path(__file__).resolve().parents[4] / "graphify"
        if str(graphify_root) not in sys.path:
            sys.path.insert(0, str(graphify_root))
        if "graphify" in sys.modules:
            del sys.modules["graphify"]
        importlib.invalidate_caches()
        return importlib.import_module(module_name)


_GRAPHIFY_INGEST = _import_graphify_module("graphify.ingest")
ingest = getattr(_GRAPHIFY_INGEST, "ingest")

try:
    _GRAPHIFY_DETECT = _import_graphify_module("graphify.detect")
except Exception:
    _GRAPHIFY_DETECT = None

try:
    _GRAPHIFY_CACHE = _import_graphify_module("graphify.cache")
except Exception:
    _GRAPHIFY_CACHE = None

from .manager import WikiManager

logger = logging.getLogger(__name__)


class WikiIngestor:
    """
    Service to ingest various file types into the LLM Wiki.
    Uses graphify for robust multi-format ingestion and structural code extraction.
    """

    def __init__(self, wiki_manager: WikiManager, raw_dir: Path):
        self.wiki_manager = wiki_manager
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents = True, exist_ok = True)

    _SKIPPED_LOCAL_FILENAMES = {".ds_store", "thumbs.db"}
    _FALLBACK_SUPPORTED_SUFFIXES = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".pdf",
        ".txt",
        ".md",
        ".rst",
    }
    _INGEST_STATE_FILENAME = ".ingest_state.json"

    def _ingest_state_path(self) -> Path:
        return self.raw_dir / self._INGEST_STATE_FILENAME

    def _load_ingest_state(self) -> Dict[str, str]:
        state_path = self._ingest_state_path()
        if not state_path.exists():
            return {}

        try:
            raw = json.loads(state_path.read_text(encoding = "utf-8"))
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        out: Dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            out[key] = value
        return out

    def _save_ingest_state(self, state: Dict[str, str]) -> None:
        state_path = self._ingest_state_path()
        state_path.write_text(
            json.dumps(state, indent = 2, sort_keys = True), encoding = "utf-8"
        )

    def _compute_file_hash(self, file_path: Path) -> Optional[str]:
        try:
            if _GRAPHIFY_CACHE is not None and hasattr(_GRAPHIFY_CACHE, "file_hash"):
                return str(_GRAPHIFY_CACHE.file_hash(file_path))

            hasher = hashlib.sha256()
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None

    def _is_supported_local_file(self, file_path: Path) -> bool:
        if _GRAPHIFY_DETECT is not None and hasattr(_GRAPHIFY_DETECT, "classify_file"):
            try:
                kind = _GRAPHIFY_DETECT.classify_file(file_path)
                return bool(kind in {"code", "document", "paper"})
            except Exception:
                pass

        return file_path.suffix.lower() in self._FALLBACK_SUPPORTED_SUFFIXES

    def _graphify_detect_candidates(self) -> Tuple[List[Path], int]:
        if _GRAPHIFY_DETECT is None:
            return [], 0

        classify_file = getattr(_GRAPHIFY_DETECT, "classify_file", None)
        if classify_file is None:
            return [], 0

        is_sensitive = getattr(_GRAPHIFY_DETECT, "_is_sensitive", None)

        try:
            candidates: List[Path] = []
            skipped_sensitive = 0

            for file_path in self.raw_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if self.should_skip_local_file(file_path):
                    continue

                resolved = file_path.expanduser().resolve()
                if callable(is_sensitive):
                    try:
                        if bool(is_sensitive(resolved)):
                            skipped_sensitive += 1
                            continue
                    except Exception:
                        pass

                kind = classify_file(resolved)
                kind_value = str(getattr(kind, "value", kind or "")).strip().lower()
                if kind_value not in {"code", "document", "paper"}:
                    continue
                candidates.append(resolved)

            candidates.sort(key = lambda p: p.stat().st_mtime, reverse = True)
            return candidates, skipped_sensitive
        except Exception as exc:
            logger.warning(
                "Graphify detect unavailable for wiki ingest candidates: %s", exc
            )
            return [], 0

    def _fallback_detect_candidates(self) -> List[Path]:
        candidates: List[Path] = []
        for file_path in self.raw_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if self.should_skip_local_file(file_path):
                continue
            if not self._is_supported_local_file(file_path):
                continue
            candidates.append(file_path.expanduser().resolve())

        candidates.sort(key = lambda p: p.stat().st_mtime, reverse = True)
        return candidates

    def _local_ingest_candidates(self) -> Tuple[List[Path], int]:
        graphify_candidates, skipped_sensitive = self._graphify_detect_candidates()
        if graphify_candidates:
            return graphify_candidates, skipped_sensitive
        return self._fallback_detect_candidates(), 0

    def should_skip_local_file(self, file_path: Path) -> bool:
        """Skip hidden/system metadata files that should never enter wiki ingestion."""
        name = file_path.name
        lowered = name.lower()
        return (
            lowered in self._SKIPPED_LOCAL_FILENAMES
            or name.startswith("._")
            or name.startswith(".")
        )

    def _is_remote_source(self, value: str) -> bool:
        scheme = urllib.parse.urlparse(value).scheme.lower()
        return scheme in {"http", "https"}

    def _extract_pdf_text(self, file_path: Path) -> str:
        extraction_errors: List[str] = []

        try:
            import pymupdf4llm

            text = pymupdf4llm.to_markdown(
                str(file_path),
                write_images = False,
                show_progress = False,
                use_ocr = False,
            )
            if text and text.strip():
                return text
            extraction_errors.append("pymupdf4llm returned empty content")
        except Exception as exc:
            extraction_errors.append(f"pymupdf4llm failed: {exc}")

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            chunks: List[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    chunks.append(text)
            extracted = "\n".join(chunks).strip()
            if extracted:
                return extracted
            extraction_errors.append("pypdf returned empty content")
        except Exception as exc:
            extraction_errors.append(f"pypdf failed: {exc}")

        raise RuntimeError(
            f"No extractable text found in PDF {file_path}. "
            f"Extraction attempts: {'; '.join(extraction_errors)}"
        )

    def _read_text_file(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding = "utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding = "latin-1")

    def _read_local_content(self, file_path: Path) -> Tuple[str, str]:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            content = self._extract_pdf_text(file_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
            raise ValueError(
                f"Unsupported wiki ingestion type for local file {file_path.name}: {suffix}. "
                "Image ingestion requires a vision extraction stage that is not configured here."
            )
        else:
            content = self._read_text_file(file_path)

        cleaned = content.strip()
        if not cleaned:
            raise ValueError(f"Ingestion produced empty content for {file_path}")
        return file_path.stem, cleaned

    def _ingest_remote_source(
        self, source: str, contributor: Optional[str]
    ) -> Tuple[str, str]:
        ingested_path = ingest(source, self.raw_dir, contributor = contributor)
        if not ingested_path.exists():
            raise FileNotFoundError(f"Ingestion failed: {ingested_path} does not exist")
        title, content = self._read_local_content(ingested_path)
        return title, content

    def ingest_file(
        self, file_path: Path, contributor: Optional[str] = None
    ) -> Optional[str]:
        """
        Ingest a single file into the wiki.

        1. Uses graphify.ingest to convert the file to a structured markdown format in the raw directory.
        2. If it's code, uses graphify.extract to get structural metadata.
        3. Uses WikiManager to ingest the content into the knowledge graph.
        """
        try:
            source = str(file_path)

            # 1. Local files are parsed directly so we can reliably extract PDFs.
            # Remote URLs still go through graphify.ingest fetch logic.
            if self._is_remote_source(source):
                title, content = self._ingest_remote_source(
                    source, contributor = contributor
                )
                reference = source
            else:
                resolved = Path(file_path).expanduser().resolve()
                if not resolved.exists():
                    logger.error(f"Ingestion failed: {resolved} does not exist.")
                    return None
                if self.should_skip_local_file(resolved):
                    logger.info(
                        "Skipping hidden/system metadata file during wiki ingestion: %s",
                        resolved.name,
                    )
                    return None
                title, content = self._read_local_content(resolved)
                reference = str(resolved)

            # 2. If it's code, we can enrich the ingestion with structural extraction
            # For now, we'll treat it as a text source, but we could use graphify.extract
            # to add more metadata to the wiki nodes.

            # 3. Ingest into the wiki engine
            result = self.wiki_manager.ingest_content(
                title = title,
                content = content,
                reference = reference,
            )

            logger.info(
                f"Successfully ingested {file_path.name} into wiki. Result: {result}"
            )
            return title

        except Exception as e:
            logger.exception(f"Failed to ingest {file_path}: {e}")
            return None

    def ingest_pending_raw_files(
        self,
        max_files: int = 8,
        contributor: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Ingest the newest pending files from raw/ using graphify-backed detection.

        Uses graphify detection/classification when available and keeps a persistent
        hash state so unchanged files are skipped across backend restarts.
        """
        max_files = max(1, int(max_files))

        sources_dir = self.wiki_manager.engine.wiki_dir / "sources"
        sources_dir.mkdir(parents = True, exist_ok = True)

        candidates, skipped_sensitive = self._local_ingest_candidates()
        if skipped_sensitive > 0:
            logger.info(
                "Graphify detection skipped %d sensitive file(s) during wiki ingest",
                skipped_sensitive,
            )

        state = self._load_ingest_state()
        state_changed = False

        existing_paths = {str(p.expanduser().resolve()) for p in candidates}
        stale_paths = [path for path in state.keys() if path not in existing_paths]
        for stale in stale_paths:
            state.pop(stale, None)
            state_changed = True

        ingested = 0
        results: List[Dict[str, str]] = []
        for path in candidates:
            if ingested >= max_files:
                break

            resolved = path.expanduser().resolve()
            if self.should_skip_local_file(resolved):
                continue
            if not self._is_supported_local_file(resolved):
                continue

            key = str(resolved)
            current_hash = self._compute_file_hash(resolved)
            previous_hash = state.get(key)

            slug = self.wiki_manager.engine._slug(resolved.stem)
            source_page = sources_dir / f"{slug}.md"

            if source_page.exists() and (
                (current_hash is not None and previous_hash == current_hash)
                or current_hash is None
            ):
                continue

            title = self.ingest_file(resolved, contributor = contributor)
            if not title:
                continue

            ingested += 1
            results.append({"source_path": str(resolved), "result": title})

            if current_hash is not None and state.get(key) != current_hash:
                state[key] = current_hash
                state_changed = True

        if state_changed:
            self._save_ingest_state(state)

        return results

    def ingest_directory(
        self, directory_path: Path, contributor: Optional[str] = None
    ) -> List[str]:
        """Ingest all supported files in a directory."""
        successes = []
        for file in directory_path.iterdir():
            if file.is_file():
                if self.should_skip_local_file(file):
                    continue
                if self._is_supported_local_file(file):
                    res = self.ingest_file(file, contributor = contributor)
                    if res:
                        successes.append(res)
        return successes
