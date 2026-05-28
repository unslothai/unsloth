"""Generate tests/fixtures/rag-preview/sample.pdf deterministically.

Run once: python tests/fixtures/rag-preview/make_fixture_pdf.py
Requires no third-party deps — builds a minimal valid single-page PDF
using only stdlib so the fixture can be regenerated in any environment.
The output is committed alongside this script so tests load it directly.
"""

import os
import struct
import zlib
from pathlib import Path

OUTPUT = Path(__file__).parent / "sample.pdf"


def _compress(data: bytes) -> bytes:
    return zlib.compress(data, level = 9)


def _pdf() -> bytes:
    # Minimal PDF 1.4 with one page, one text stream.
    # Structure: header, catalog, pages, page, content stream, xref, trailer.
    page_text = b"BT /F1 12 Tf 72 720 Td (RAG preview fixture - page 1) Tj ET"
    compressed = _compress(page_text)
    stream_len = len(compressed)

    objects: list[bytes] = []

    def obj(n: int, body: bytes) -> bytes:
        return f"{n} 0 obj\n".encode() + body + b"\nendobj\n"

    # 1: Catalog
    objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    # 2: Pages
    objects.append(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # 3: Page
    objects.append(
        obj(
            3,
            (
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 612 792] "
                b"/Contents 4 0 R "
                b"/Resources << /Font << /F1 5 0 R >> >> >>"
            ),
        )
    )
    # 4: Content stream
    objects.append(
        obj(
            4,
            (
                f"<< /Length {stream_len} /Filter /FlateDecode >>".encode()
                + b"\nstream\n"
                + compressed
                + b"\nendstream"
            ),
        )
    )
    # 5: Font
    objects.append(
        obj(
            5,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        )
    )

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets: list[int] = []
    for o in objects:
        offsets.append(len(header) + len(body))
        body += o

    xref_offset = len(header) + len(body)
    n = len(objects)
    xref = f"xref\n0 {n + 1}\n".encode()
    xref += b"0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    trailer = (
        f"trailer\n<< /Size {n + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body + xref + trailer


if __name__ == "__main__":
    pdf_bytes = _pdf()
    OUTPUT.write_bytes(pdf_bytes)
    print(f"Written {len(pdf_bytes)} bytes to {OUTPUT}")
