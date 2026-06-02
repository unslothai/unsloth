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


def test_rag_resolve_embedder_returns_default():
    from utils.rag.config import RAG_EMBEDDING_MODEL, resolve_embedder

    assert resolve_embedder() == RAG_EMBEDDING_MODEL
    assert isinstance(resolve_embedder(), str) and resolve_embedder()
