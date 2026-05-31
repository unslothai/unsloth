"""End-to-end multimodal RAG integration test.

Marked `server` so default pytest runs skip it — downloads BGE-VL-base
(~600 MB) on first run and exercises the real embedding stack. Run
explicitly with:

    ~/.unsloth/studio/unsloth_studio/bin/python -m pytest \
        tests/python/test_rag_multimodal_integration.py -v -m server

Exercises the ingestion subprocess worker in-process (with a regular
queue rather than mp.Queue) so we cover the parse → chunk → load
embedder → encode_images → emit chunks_batch path without spinning up
a child process. The parent-side chunk insertion is covered separately
by test_rag_multimodal.py.
"""

import os
import queue as queue_module
import sys
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


@pytest.mark.server
def test_multimodal_subprocess_emits_image_and_caption_chunks(
    tmp_path,
    monkeypatch,
):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")
    pytest.importorskip("torch")

    # tmp studio root isolates the subprocess's image writes.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_RAG_EMBEDDING_MODEL", "BAAI/BGE-VL-base")
    monkeypatch.setenv("UNSLOTH_RAG_CHUNK_SIZE", "200")
    monkeypatch.setenv("UNSLOTH_RAG_CHUNK_OVERLAP", "20")

    # Reset module caches so the new env vars apply.
    import importlib

    import utils.rag.config as rag_config

    importlib.reload(rag_config)
    from core.rag import embeddings as embeddings_module

    embeddings_module._model = None
    embeddings_module._model_name = None

    # Small PDF: text + one embedded image.
    from PIL import Image

    img = Image.new("RGB", (96, 64), (200, 100, 50))
    img_buf = BytesIO()
    img.save(img_buf, format = "PNG")
    img_bytes = img_buf.getvalue()

    doc = pymupdf.open()
    page = doc.new_page(width = 612, height = 792)
    page.insert_text(
        (72, 100),
        "Architecture overview\n\nThe following diagram shows our system.",
        fontsize = 11,
    )
    image_rect = pymupdf.Rect(72, 200, 168, 264)
    page.insert_image(image_rect, stream = img_bytes)
    page.insert_text(
        (72, 290),
        "Figure 1: the architecture diagram described above.",
        fontsize = 11,
    )
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Drive the worker in-process via a regular queue.
    from core.rag.ingestion import _subprocess_worker

    out_queue: "queue_module.Queue[dict]" = queue_module.Queue()
    _subprocess_worker(
        stored_path = str(pdf_path),
        model_name = "BAAI/BGE-VL-base",
        chunk_size = 200,
        overlap = 20,
        batch_size = 4,
        out_queue = out_queue,
        chunking_strategy = "standard",
        mode = "multimodal",
        document_id = "test-doc-1",
    )

    # Drain all events (in-process queue, stable order).
    events: list[dict] = []
    while not out_queue.empty():
        events.append(out_queue.get_nowait())

    # Expect >=1 chunks_batch and exactly one terminal complete/error.
    assert any(e["type"] == "chunks_batch" for e in events)
    terminals = [e for e in events if e["type"] in ("complete", "error")]
    assert len(terminals) == 1, terminals
    assert terminals[0]["type"] == "complete"

    # Collect chunks across batches.
    all_chunks: list[dict] = []
    for e in events:
        if e["type"] == "chunks_batch":
            all_chunks.extend(e["chunks"])

    kinds = [c.get("kind") for c in all_chunks]
    assert "text" in kinds, "expected at least one text chunk"
    assert "image" in kinds, "expected at least one image chunk"
    # Paragraph right after the image triggers caption pairing.
    assert "caption" in kinds, "expected at least one caption chunk"

    # Image chunks carry a path on disk under the tmp studio root.
    image_chunks = [c for c in all_chunks if c.get("kind") == "image"]
    for chunk in image_chunks:
        assert chunk.get("image_path"), chunk
        path_on_disk = Path(chunk["image_path"])
        assert path_on_disk.is_file()
        assert str(path_on_disk).startswith(str(tmp_path))

    # Paired image + caption chunks share a pair_group.
    pair_groups: dict[str, list[str]] = {}
    for chunk in all_chunks:
        group = chunk.get("pair_group")
        if group:
            pair_groups.setdefault(group, []).append(chunk.get("kind", ""))
    paired = [
        kinds
        for kinds in pair_groups.values()
        if "image" in kinds and "caption" in kinds
    ]
    assert paired, f"expected an image/caption pair, got groups={pair_groups}"


@pytest.mark.server
def test_text_and_image_vectors_share_dimension(monkeypatch):
    """BGE-VL is a shared-space embedder — sanity-check before relying on it."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    monkeypatch.setenv("UNSLOTH_RAG_EMBEDDING_MODEL", "BAAI/BGE-VL-base")
    from core.rag import embeddings as embeddings_module

    embeddings_module._model = None
    embeddings_module._model_name = None

    from PIL import Image

    img = Image.new("RGB", (32, 32), (50, 150, 200))
    buf = BytesIO()
    img.save(buf, format = "PNG")

    image_vectors = embeddings_module.encode_images([buf.getvalue()])
    text_vectors = embeddings_module.encode(["a blue square"])

    assert (
        image_vectors[0].shape == text_vectors[0].shape
    ), f"text dim {text_vectors[0].shape} != image dim {image_vectors[0].shape}"
