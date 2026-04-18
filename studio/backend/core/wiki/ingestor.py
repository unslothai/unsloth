from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import sys
import importlib
import urllib.parse

try:
    from graphify.ingest import ingest
except ModuleNotFoundError:
    # Monorepo fallback: allow running backend without installing graphify as a wheel.
    graphify_root = Path(__file__).resolve().parents[4] / "graphify"
    if str(graphify_root) not in sys.path:
        sys.path.insert(0, str(graphify_root))
    if "graphify" in sys.modules:
        del sys.modules["graphify"]
    importlib.invalidate_caches()
    from graphify.ingest import ingest

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

    def ingest_directory(
        self, directory_path: Path, contributor: Optional[str] = None
    ) -> List[str]:
        """Ingest all supported files in a directory."""
        successes = []
        for file in directory_path.iterdir():
            if file.is_file():
                if self.should_skip_local_file(file):
                    continue
                # Basic filter for supported types (can be expanded)
                if file.suffix.lower() in (
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".pdf",
                    ".txt",
                    ".md",
                    ".png",
                    ".jpg",
                    ".jpeg",
                ):
                    res = self.ingest_file(file, contributor = contributor)
                    if res:
                        successes.append(res)
        return successes
