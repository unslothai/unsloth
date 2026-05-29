# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG ingestion pipeline.

Spawn-subprocess per job (parse/chunk/embed); parent persists chunks,
vectors, and rebuilds BM25 on completion. Only the parent opens rag.db.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_module
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from loggers import get_logger
from storage.studio_db import get_connection
from utils.rag.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBEDDING_MODEL,
)

from . import bm25, embeddings, vector_store
from .vector_store import kb_scope, thread_scope

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")
_QUEUE_TIMEOUT_SECONDS = 300


# --- Subprocess worker ---

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}


def _subprocess_worker(
    stored_path: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    out_queue: Any,
    chunking_strategy: str = "standard",
    mode: str = "text",
    document_id: str = "",
    vlm_url: str | None = None,
    vlm_model: str | None = None,
) -> None:
    # Spawned subprocess: structlog isn't configured here (the parent's
    # setup runs in the FastAPI process only), so configure it the same
    # way so captioner / parser logs render as JSON like the rest, not
    # structlog's default dev ConsoleRenderer.
    try:
        import os as _os

        from loggers.config import LogConfig

        LogConfig.setup_logging(
            env = _os.getenv("ENVIRONMENT_TYPE", "production"),
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.rag.captioner import caption_images
        from core.rag.chunking import chunk_pages
        from core.rag.parsers import inline_image_captions, parse

        out_queue.put({"type": "progress", "stage": "parse", "progress": 0.05})
        # Always extract images so we can caption + splice for both
        # modes. Text mode uses the captions inline in markdown; multimodal
        # additionally embeds the raw images as image-kind chunks.
        parsed = parse(Path(stored_path), want_images = True)
        pages = parsed.pages
        if not pages and not parsed.images:
            out_queue.put(
                {"type": "error", "error": "no extractable content in document"}
            )
            return

        # Caption figures once (chat VLM if available, else helper VLM
        # fallback), then splice captions into the page markdown so the
        # chunker indexes them like any other text. Multimodal mode also
        # passes these same captions through to _stream_image_chunks
        # below — no duplicate VLM calls per image.
        captions: list[str] = []
        if parsed.images:
            out_queue.put(
                {"type": "progress", "stage": "caption_images", "progress": 0.08}
            )
            captions = caption_images(
                [img.image_bytes for img in parsed.images],
                vlm_url = vlm_url,
                vlm_model = vlm_model,
            )
            pages = inline_image_captions(pages, parsed.images, captions)

        out_queue.put(
            {
                "type": "document_pages",
                "pages": [
                    {
                        "page_index": index,
                        "page_number": page.page_number,
                        "text": page.text,
                        "char_count": len(page.text),
                        "line_count": len(page.text.splitlines()),
                    }
                    for index, page in enumerate(pages)
                ],
            }
        )

        out_queue.put({"type": "progress", "stage": "load_model", "progress": 0.1})
        from core.rag.embeddings import (
            get_embedder,
            late_chunk_encode,
            token_counter,
        )

        model = get_embedder(model_name)
        counter = token_counter(model_name)
        dim = int(model.get_sentence_embedding_dimension())
        out_queue.put({"type": "dim", "dim": dim})

        if chunking_strategy == "late":
            _run_late_chunking(
                pages = pages,
                stored_path = Path(stored_path),
                chunk_size = chunk_size,
                overlap = overlap,
                counter = counter,
                model_name = model_name,
                late_chunk_encode = late_chunk_encode,
                out_queue = out_queue,
            )
            return

        text_count = _run_standard_chunking(
            pages = pages,
            stored_path = Path(stored_path),
            chunk_size = chunk_size,
            overlap = overlap,
            counter = counter,
            batch_size = batch_size,
            model = model,
            chunk_pages = chunk_pages,
            out_queue = out_queue,
            send_complete = False,
        )
        image_count = 0
        if mode == "multimodal" and parsed.images and document_id:
            image_count = _stream_image_chunks(
                images = parsed.images,
                document_id = document_id,
                model_name = model_name,
                out_queue = out_queue,
                first_index = text_count,
                precomputed_captions = captions,
            )
        out_queue.put({"type": "complete", "num_chunks": text_count + image_count})
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion subprocess failed")
        out_queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})


def _run_standard_chunking(
    *,
    pages,
    stored_path,
    chunk_size,
    overlap,
    counter,
    batch_size,
    model,
    chunk_pages,
    out_queue,
    send_complete: bool = True,
) -> int:
    """Stream text chunks; returns count. send_complete=False when images follow."""
    out_queue.put({"type": "progress", "stage": "chunk", "progress": 0.2})
    chunks = chunk_pages(
        pages,
        max_tokens = chunk_size,
        overlap_tokens = overlap,
        token_counter = counter,
    )
    if not chunks:
        if send_complete:
            out_queue.put({"type": "error", "error": "chunker produced no chunks"})
        return 0
    from core.rag.locators import pdf_regions_for_chunks

    pdf_regions = pdf_regions_for_chunks(stored_path, pages, chunks)

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectors = model.encode(
            [c.text for c in batch],
            batch_size = batch_size,
            normalize_embeddings = True,
            convert_to_numpy = True,
            show_progress_bar = False,
        )
        out_queue.put(
            {
                "type": "chunks_batch",
                "first_index": i,
                "chunks": [
                    {
                        "text": c.text,
                        "token_count": c.token_count,
                        "page_number": c.page_number,
                        "source_page_index": c.source_page_index,
                        "page_char_start": c.page_char_start,
                        "page_char_end": c.page_char_end,
                        "line_start": c.line_start,
                        "line_end": c.line_end,
                        "pdf_regions": pdf_regions[i + offset],
                        "kind": "text",
                    }
                    for offset, c in enumerate(batch)
                ],
                "vectors": vectors.tolist(),
            }
        )
        progress = 0.3 + 0.65 * min(1.0, (i + len(batch)) / total)
        out_queue.put({"type": "progress", "stage": "embed", "progress": progress})

    if send_complete:
        out_queue.put({"type": "complete", "num_chunks": total})
    return total


def _stream_image_chunks(
    *,
    images,
    document_id: str,
    model_name: str,
    out_queue,
    first_index: int,
    precomputed_captions: list[str] | None = None,
) -> int:
    """Persist images, emit image+caption chunks; pairs share pair_group.

    ``precomputed_captions`` come from the parent's earlier
    caption_images call (used so we don't VLM-caption the same images
    twice — once for markdown splicing, once for the caption-kind chunk).
    If absent we fall back to each image's nearest_caption (page text).
    """
    from core.rag.embeddings import encode, encode_images
    from utils.paths.storage_roots import ensure_dir, rag_uploads_root

    if not images:
        return 0

    out_queue.put({"type": "progress", "stage": "extract_images", "progress": 0.85})

    img_dir = ensure_dir(rag_uploads_root() / "images" / document_id)

    paths: list[str] = []
    bytes_for_encoding: list[bytes] = []
    captions: list[str] = []
    pages: list[int | None] = []
    pre_caps = precomputed_captions or []
    for idx, img in enumerate(images):
        ext = _MIME_TO_EXT.get(img.mime_type, ".bin")
        path = img_dir / f"img-{idx:04d}{ext}"
        try:
            path.write_bytes(img.image_bytes)
        except OSError:
            logger.warning("failed to save image; skipping", path = str(path))
            continue
        paths.append(str(path))
        bytes_for_encoding.append(img.image_bytes)
        vlm_cap = pre_caps[idx].strip() if idx < len(pre_caps) and pre_caps[idx] else ""
        captions.append(vlm_cap or (img.nearest_caption or ""))
        pages.append(img.page_number)

    if not paths:
        return 0

    image_vectors = encode_images(bytes_for_encoding, model_name = model_name)

    caption_to_image: list[int] = [i for i, cap in enumerate(captions) if cap.strip()]
    if caption_to_image:
        caption_vectors_arr = encode(
            [captions[i] for i in caption_to_image],
            model_name = model_name,
        )
        caption_vectors = caption_vectors_arr.tolist()
    else:
        caption_vectors = []

    out_chunks: list[dict] = []
    out_vectors: list[list[float]] = []
    cap_iter = iter(zip(caption_to_image, caption_vectors))
    next_cap = next(cap_iter, None)

    for idx, (path, page, caption) in enumerate(zip(paths, pages, captions)):
        group_id = f"img-{idx:04d}"
        out_chunks.append(
            {
                "text": caption[:1000] if caption else "",
                "token_count": 0,
                "page_number": page,
                "kind": "image",
                "image_path": path,
                "pair_group": group_id,
            }
        )
        out_vectors.append(image_vectors[idx].tolist())
        if next_cap is not None and next_cap[0] == idx:
            _cap_index, cap_vec = next_cap
            out_chunks.append(
                {
                    "text": caption,
                    "token_count": max(1, len(caption.split())),
                    "page_number": page,
                    "kind": "caption",
                    "image_path": None,
                    "pair_group": group_id,
                }
            )
            out_vectors.append(cap_vec)
            next_cap = next(cap_iter, None)

    out_queue.put(
        {
            "type": "chunks_batch",
            "first_index": first_index,
            "chunks": out_chunks,
            "vectors": out_vectors,
        }
    )
    out_queue.put({"type": "progress", "stage": "extract_images", "progress": 0.95})
    return len(out_chunks)


def _run_late_chunking(
    *,
    pages,
    stored_path,
    chunk_size,
    overlap,
    counter,
    model_name,
    late_chunk_encode,
    out_queue,
) -> None:
    """Chunk once, embed in one pass, ship all chunks in one chunks_batch."""
    from core.rag.chunking import chunk_pages_with_spans
    from core.rag.locators import pdf_regions_for_chunks

    out_queue.put({"type": "progress", "stage": "chunk", "progress": 0.2})
    full_doc, chunks, char_spans = chunk_pages_with_spans(
        pages,
        max_tokens = chunk_size,
        overlap_tokens = overlap,
        token_counter = counter,
    )
    if not chunks:
        out_queue.put({"type": "error", "error": "chunker produced no chunks"})
        return

    out_queue.put({"type": "progress", "stage": "embed", "progress": 0.4})
    vectors = late_chunk_encode(
        full_doc,
        char_spans,
        model_name = model_name,
        normalize = True,
    )
    pdf_regions = pdf_regions_for_chunks(stored_path, pages, chunks)

    out_queue.put({"type": "progress", "stage": "embed", "progress": 0.9})
    out_queue.put(
        {
            "type": "chunks_batch",
            "first_index": 0,
            "chunks": [
                {
                    "text": c.text,
                    "token_count": c.token_count,
                    "page_number": c.page_number,
                    "source_page_index": c.source_page_index,
                    "page_char_start": c.page_char_start,
                    "page_char_end": c.page_char_end,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "pdf_regions": pdf_regions[index],
                    "kind": "text",
                }
                for index, c in enumerate(chunks)
            ],
            "vectors": [v.tolist() for v in vectors],
        }
    )
    out_queue.put({"type": "complete", "num_chunks": len(chunks)})


# --- Job manager (parent side) ---


class _JobState:
    def __init__(self, job_id: str, document_id: str, scope: str) -> None:
        self.job_id = job_id
        self.document_id = document_id
        self.scope = scope
        self.status = "pending"
        self.stage: str | None = None
        self.progress: float = 0.0
        self.error: str | None = None
        self.cancelled = False
        self.proc: Any = None
        self.out_queue: Any = None
        self.subscribers: list[queue_module.Queue[dict]] = []
        self.lock = threading.Lock()

    def push_event(self, event: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue_module.Full:
                pass

    def subscribe(self) -> queue_module.Queue[dict]:
        q: queue_module.Queue[dict] = queue_module.Queue(maxsize = 256)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue_module.Queue[dict]) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


_jobs: dict[str, _JobState] = {}
_jobs_lock = threading.Lock()


def get_job_state(job_id: str) -> _JobState | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _scope_for(kb_id: str | None, thread_id: str | None) -> str:
    if kb_id:
        return kb_scope(kb_id)
    if thread_id:
        return thread_scope(thread_id)
    raise ValueError("must supply kb_id or thread_id")


def _update_job_row(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    values = list(fields.values()) + [job_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE rag_ingestion_jobs SET {set_clause} WHERE id = ?", values)
        conn.commit()


def _update_document_row(document_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    values = list(fields.values()) + [document_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE rag_documents SET {set_clause} WHERE id = ?", values)
        conn.commit()


def _insert_chunks_and_collect_for_bm25(
    document_id: str,
    scope: str,
    first_index: int,
    chunks_meta: list[dict],
    vectors: list[list[float]],
) -> list[dict]:
    """Insert chunks into sqlite + vector_store; return [{id, text}] for BM25."""
    rows: list[tuple] = []
    points: list[dict] = []
    bm25_rows: list[dict] = []
    pair_groups: dict[str, list[str]] = {}

    for offset, (meta, vec) in enumerate(zip(chunks_meta, vectors)):
        chunk_index = first_index + offset
        chunk_id = str(uuid4())
        kind = meta.get("kind", "text")
        image_path = meta.get("image_path")
        pair_group = meta.get("pair_group")
        if pair_group:
            pair_groups.setdefault(pair_group, []).append(chunk_id)
        rows.append(
            (
                chunk_id,
                document_id,
                chunk_index,
                meta["text"],
                meta["token_count"],
                meta["page_number"],
                kind,
                image_path,
                meta.get("source_page_index"),
                meta.get("page_char_start"),
                meta.get("page_char_end"),
                meta.get("line_start"),
                meta.get("line_end"),
                json.dumps(meta.get("pdf_regions") or [], separators = (",", ":"))
                if meta.get("pdf_regions")
                else None,
            )
        )
        points.append(
            {
                "id": chunk_id,
                "vector": vec,
                "payload": {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": meta["text"],
                    "page_number": meta["page_number"],
                    "kind": kind,
                    "image_path": image_path,
                    "source_page_index": meta.get("source_page_index"),
                    "page_char_start": meta.get("page_char_start"),
                    "page_char_end": meta.get("page_char_end"),
                    "line_start": meta.get("line_start"),
                    "line_end": meta.get("line_end"),
                    "pdf_regions": meta.get("pdf_regions") or [],
                },
            }
        )
        if kind in ("text", "caption") and meta["text"]:
            bm25_rows.append({"id": chunk_id, "text": meta["text"]})
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO rag_chunks
            (id, document_id, chunk_index, text, token_count, page_number,
             kind, image_path, source_page_index, page_char_start,
             page_char_end, line_start, line_end, pdf_regions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        # Link only when exactly two members in a pair_group.
        for ids in pair_groups.values():
            if len(ids) != 2:
                continue
            id_a, id_b = ids
            conn.execute(
                "UPDATE rag_chunks SET linked_chunk_id = ? WHERE id = ?",
                (id_b, id_a),
            )
            conn.execute(
                "UPDATE rag_chunks SET linked_chunk_id = ? WHERE id = ?",
                (id_a, id_b),
            )
        conn.commit()
    vector_store.upsert_chunks(scope, points)
    return bm25_rows


def _replace_document_pages(document_id: str, pages: list[dict]) -> None:
    now = int(time.time())
    rows = [
        (
            document_id,
            int(page["page_index"]),
            page.get("page_number"),
            page.get("text") or "",
            int(page.get("char_count", len(page.get("text") or ""))),
            int(page.get("line_count", len((page.get("text") or "").splitlines()))),
            now,
        )
        for page in pages
    ]
    with get_connection() as conn:
        doc_row = conn.execute(
            "SELECT 1 FROM rag_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if doc_row is None:
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
        conn.execute(
            "DELETE FROM rag_document_pages WHERE document_id = ?", (document_id,)
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO rag_document_pages
                (document_id, page_index, page_number, text, char_count,
                 line_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()


def _all_scope_chunks(scope: str) -> list[dict]:
    if scope.startswith("kb_"):
        kb_id = scope[len("kb_") :]
        sql = (
            "SELECT c.id, c.text FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.kb_id = ?"
        )
        bind = (kb_id,)
    elif scope.startswith("thread_"):
        thread_id = scope[len("thread_") :]
        sql = (
            "SELECT c.id, c.text FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.thread_id = ?"
        )
        bind = (thread_id,)
    else:
        return []
    with get_connection() as conn:
        rows = conn.execute(sql, bind).fetchall()
    return [{"id": r["id"], "text": r["text"]} for r in rows]


def _pump(
    state: _JobState,
    proc: Any,
    out_queue: Any,
) -> None:
    """Drain queue until subprocess completes/errors/dies."""
    bm25_buffer: list[dict] = []
    embedding_dim: int | None = None
    final_status = "failed"
    final_error: str | None = None
    final_num_chunks = 0

    started_at = int(time.time())
    state.status = "running"
    _update_job_row(state.job_id, status = "running", started_at = started_at)
    _update_document_row(state.document_id, status = "running")
    state.push_event({"type": "status", "status": "running"})

    try:
        while True:
            if state.cancelled:
                break
            try:
                msg = out_queue.get(timeout = _QUEUE_TIMEOUT_SECONDS)
            except queue_module.Empty:
                if not proc.is_alive():
                    final_error = "subprocess exited without completion message"
                    break
                continue
            mtype = msg.get("type")
            if mtype == "__cancel__":
                state.cancelled = True
                break
            if mtype == "progress":
                state.stage = msg.get("stage")
                state.progress = float(msg.get("progress", 0.0))
                _update_job_row(
                    state.job_id,
                    stage = state.stage,
                    progress = state.progress,
                )
                state.push_event(msg)
            elif mtype == "dim":
                embedding_dim = int(msg["dim"])
                vector_store.ensure_collection(state.scope, embedding_dim)
            elif mtype == "document_pages":
                try:
                    _replace_document_pages(
                        state.document_id,
                        list(msg.get("pages") or []),
                    )
                except sqlite3.IntegrityError as exc:
                    final_error = (
                        f"document was removed before ingestion finished ({exc})"
                    )
                    break
            elif mtype == "chunks_batch":
                if embedding_dim is None:
                    embedding_dim = len(msg["vectors"][0]) if msg["vectors"] else None
                    if embedding_dim is not None:
                        vector_store.ensure_collection(state.scope, embedding_dim)
                try:
                    bm25_rows = _insert_chunks_and_collect_for_bm25(
                        state.document_id,
                        state.scope,
                        int(msg["first_index"]),
                        msg["chunks"],
                        msg["vectors"],
                    )
                except sqlite3.IntegrityError as exc:
                    # rag_documents row was deleted mid-ingest (user removed
                    # the chip / cleared the index). Fail the job cleanly
                    # rather than crashing the pump thread.
                    final_error = (
                        f"document was removed before ingestion finished ({exc})"
                    )
                    break
                bm25_buffer.extend(bm25_rows)
            elif mtype == "complete":
                final_status = "completed"
                final_num_chunks = int(msg.get("num_chunks", len(bm25_buffer)))
                break
            elif mtype == "error":
                final_error = str(msg.get("error", "unknown error"))
                break
            else:
                logger.warning("ingestion: unknown message type", mtype = repr(mtype))
    finally:
        proc.join(timeout = 30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout = 5)

    finished_at = int(time.time())
    if state.cancelled:
        # User cancelled mid-flight. The route-side deleteDocument removes the
        # row, file, and chunk artifacts; here we just mark terminal and notify
        # subscribers so the SSE stream closes cleanly.
        _update_document_row(state.document_id, status = "cancelled")
        _update_job_row(
            state.job_id,
            status = "cancelled",
            stage = "cancelled",
            finished_at = finished_at,
        )
        state.status = "cancelled"
        state.push_event({"type": "cancelled"})
        return
    if final_status == "completed":
        full_scope_chunks = _all_scope_chunks(state.scope)
        bm25.rebuild_index(state.scope, full_scope_chunks)
        _update_document_row(
            state.document_id,
            status = "completed",
            num_chunks = final_num_chunks,
        )
        _update_job_row(
            state.job_id,
            status = "completed",
            progress = 1.0,
            stage = "done",
            finished_at = finished_at,
        )
        state.status = "completed"
        state.progress = 1.0
        state.push_event(
            {
                "type": "complete",
                "num_chunks": final_num_chunks,
            }
        )
    else:
        _update_document_row(
            state.document_id,
            status = "failed",
            error = final_error,
        )
        _update_job_row(
            state.job_id,
            status = "failed",
            error = final_error,
            finished_at = finished_at,
        )
        state.status = "failed"
        state.error = final_error
        state.push_event({"type": "error", "error": final_error})


def _probe_loaded_vlm() -> tuple[str | None, str | None]:
    """Best-effort: return (base_url, model_name) when a vision-capable
    chat model is currently loaded via llama-server; (None, None) otherwise.

    Used to caption figures with the user's chat VLM instead of loading
    a dedicated captioning model — no extra VRAM, no extra download.
    Only the llama-server backend is supported today; transformers /
    unsloth in-process VLMs would need a different bridge.
    """
    try:
        # The singleton getter lives in routes.inference, not the
        # llama_cpp module. Importing from the wrong place silently
        # returned None for every probe — captioner always fell back
        # to the helper VLM even when the chat model was vision-capable.
        from routes.inference import get_llama_cpp_backend
    except Exception as exc:
        logger.warning("RAG probe: get_llama_cpp_backend import failed", error = str(exc))
        return None, None
    try:
        backend = get_llama_cpp_backend()
    except Exception as exc:
        logger.warning("RAG probe: get_llama_cpp_backend() raised", error = str(exc))
        return None, None
    if not getattr(backend, "is_loaded", False):
        return None, None
    if not getattr(backend, "is_vision", False):
        return None, None
    base_url = getattr(backend, "base_url", None)
    model_id = getattr(backend, "model_identifier", None)
    if not base_url or not model_id:
        return None, None
    return base_url, model_id


def enqueue_ingestion(
    document_id: str,
    stored_path: Path,
    *,
    kb_id: str | None = None,
    thread_id: str | None = None,
    embedding_model: str | None = None,
    chunking_strategy: str = "standard",
    mode: str = "text",
) -> str:
    """Create the job row, spawn the subprocess, start the pump; return job_id."""
    from utils.rag.config import resolve_embedder

    scope = _scope_for(kb_id, thread_id)
    model_name = (
        embedding_model
        or resolve_embedder(mode, chunking_strategy)
        or RAG_EMBEDDING_MODEL
    )
    # Probe the loaded chat backend so the subprocess can route figure
    # captioning to the user's own vision model (no extra VRAM). Runs
    # for both modes — text mode splices captions into markdown, and
    # multimodal mode additionally feeds them to the image-vector
    # encoder. If no vision chat model is loaded, the subprocess falls
    # back to the helper VLM (pre-cached at studio startup).
    vlm_url, vlm_model = _probe_loaded_vlm()
    if vlm_url:
        logger.info(
            "RAG ingest: will caption figures via loaded chat VLM",
            vlm_model = vlm_model,
            vlm_url = vlm_url,
        )
    else:
        logger.info(
            "RAG ingest: no vision-capable chat model loaded; "
            "subprocess will use the helper gemma-3n VLM fallback."
        )
    job_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_ingestion_jobs
            (id, document_id, status, progress, stage)
            VALUES (?, ?, 'pending', 0.0, 'queued')
            """,
            (job_id, document_id),
        )
        conn.commit()

    state = _JobState(job_id = job_id, document_id = document_id, scope = scope)
    with _jobs_lock:
        _jobs[job_id] = state

    out_queue = _CTX.Queue()
    state.out_queue = out_queue
    proc = _CTX.Process(
        target = _subprocess_worker,
        args = (
            str(stored_path),
            model_name,
            RAG_CHUNK_SIZE,
            RAG_CHUNK_OVERLAP,
            RAG_EMBED_BATCH_SIZE,
            out_queue,
            chunking_strategy,
            mode,
            document_id,
            vlm_url,
            vlm_model,
        ),
        daemon = True,
    )
    proc.start()
    state.proc = proc
    pump_thread = threading.Thread(
        target = _pump,
        args = (state, proc, out_queue),
        name = f"rag-ingest-pump-{job_id[:8]}",
        daemon = True,
    )
    pump_thread.start()
    return job_id


def cancel_ingestion(job_id: str) -> bool:
    """Stop an in-flight ingestion: wake the pump via a sentinel and kill the
    worker subprocess so it stops consuming GPU/CPU. Returns False if the job
    is unknown or already terminal. Artifact/row cleanup is the caller's job
    (the route deletes the document)."""
    state = get_job_state(job_id)
    if state is None:
        return False
    if state.status in ("completed", "failed", "cancelled"):
        return False
    state.cancelled = True
    if state.out_queue is not None:
        try:
            state.out_queue.put_nowait({"type": "__cancel__"})
        except Exception:
            pass
    proc = state.proc
    if proc is not None and proc.is_alive():
        proc.terminate()
    return True


def delete_document_artifacts(document_id: str, scope: str) -> None:
    """Drop the doc's vectors, rebuild BM25. Caller deletes the rag_documents row."""
    vector_store.delete_document(scope, document_id)
    remaining = _all_scope_chunks(scope)
    if remaining:
        bm25.rebuild_index(scope, remaining)
    else:
        bm25.delete_scope(scope)


def delete_scope_artifacts(scope: str) -> None:
    vector_store.delete_scope(scope)
    bm25.delete_scope(scope)


def purge_thread_documents(thread_ids: list[str]) -> None:
    """Drop RAG artifacts for the given thread ids (no FK cascade to chat_threads)."""
    if not thread_ids:
        return
    import os
    from pathlib import Path

    from utils.paths.storage_roots import rag_uploads_root

    placeholders = ",".join("?" for _ in thread_ids)
    uploads_root = Path(os.path.realpath(rag_uploads_root()))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT stored_path FROM rag_documents WHERE thread_id IN ({placeholders})",
            thread_ids,
        ).fetchall()
        conn.execute(
            f"DELETE FROM rag_documents WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        conn.commit()
    for row in rows:
        try:
            real = Path(os.path.realpath(row["stored_path"]))
            real.relative_to(uploads_root)
        except (OSError, ValueError):
            continue
        real.unlink(missing_ok = True)
    for thread_id in thread_ids:
        delete_scope_artifacts(thread_scope(thread_id))


def purge_all_thread_documents() -> None:
    """Drop every per-thread RAG artifact."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM rag_documents WHERE thread_id IS NOT NULL"
        ).fetchall()
    purge_thread_documents([r["thread_id"] for r in rows])
