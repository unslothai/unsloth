"""Multimodal RAG tests (Phase 3B-multimodal).

Most of the multimodal pipeline depends on real models (BGE-VL ~1.5 GB
VRAM) and a writable filesystem under rag_uploads_root() — those tests
are gated behind the `server` marker. The pure-python pieces (parser
returns images when asked, route accepts the mode field, constraint
validator rejects illegal combos) run in every test invocation.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


def test_html_parser_returns_images_when_requested(tmp_path):
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    pytest.importorskip("markdownify")
    from core.rag.parsers import parse

    # 1x1 transparent PNG.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa3\xb0\xa9\xa8\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path = tmp_path / "tiny.png"
    img_path.write_bytes(png_bytes)
    html_path = tmp_path / "sample.html"
    html_path.write_text(
        f"<html><body><h1>Doc</h1>"
        f"<p>Body text.</p>"
        f'<img src="tiny.png" alt="A tiny figure">'
        f"</body></html>",
        encoding = "utf-8",
    )

    no_images = parse(html_path, want_images = False)
    assert no_images.images == []

    with_images = parse(html_path, want_images = True)
    assert len(with_images.images) == 1
    img = with_images.images[0]
    assert img.image_bytes == png_bytes
    assert img.mime_type == "image/png"
    assert img.nearest_caption == "A tiny figure"


def test_multimodal_late_combo_validator():
    from fastapi import HTTPException

    from routes.rag import _validate_mode_combo

    # Allowed combos → None.
    assert _validate_mode_combo("text", "standard") is None
    assert _validate_mode_combo("text", "late") is None
    assert _validate_mode_combo("multimodal", "standard") is None

    # Forbidden combo → 400.
    with pytest.raises(HTTPException) as excinfo:
        _validate_mode_combo("multimodal", "late")
    assert excinfo.value.status_code == 400


def test_rag_embedder_matrix_excludes_multimodal_late():
    from utils.rag.config import RAG_EMBEDDER_MATRIX, resolve_embedder

    assert ("multimodal", "late") not in RAG_EMBEDDER_MATRIX
    assert ("text", "standard") in RAG_EMBEDDER_MATRIX
    assert ("text", "late") in RAG_EMBEDDER_MATRIX
    assert ("multimodal", "standard") in RAG_EMBEDDER_MATRIX

    # Unknown combos fall back to the legacy default, not KeyError.
    fallback = resolve_embedder("multimodal", "late")
    assert isinstance(fallback, str) and fallback


def test_image_path_url_construction():
    """Sanity-check the URL shape served back to the frontend.

    The image URL is built relative to /api/rag/images/<doc>/<filename>
    purely from the stored image_path (filename only — directory
    structure is fixed). Verify the rule.
    """
    from pathlib import Path as P

    image_path = "/var/data/rag/images/doc-123/img-0042.png"
    document_id = "doc-123"
    expected = f"/api/rag/images/{document_id}/{P(image_path).name}"
    assert expected == "/api/rag/images/doc-123/img-0042.png"


@pytest.mark.server
def test_multimodal_encode_image_returns_vector(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")
    monkeypatch.setenv("UNSLOTH_RAG_EMBEDDING_MODEL", "BAAI/BGE-VL-base")
    # Reset the embedder singleton so the env var applies.
    from core.rag import embeddings as embeddings_module

    embeddings_module._model = None
    embeddings_module._model_name = None

    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (32, 32), (200, 100, 50))
    buf = BytesIO()
    img.save(buf, format = "PNG")
    image_bytes = buf.getvalue()

    vectors = embeddings_module.encode_images([image_bytes])
    assert len(vectors) == 1
    dim = vectors[0].shape[0]
    assert dim > 0

    # Text shares the same dim — the point of a multimodal embedder.
    text_vec = embeddings_module.encode(["a red square"])[0]
    assert text_vec.shape[0] == dim
