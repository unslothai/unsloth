# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only ebook fixtures; no model weights or network access are needed."""

import importlib.util
import ast
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest


@pytest.fixture
def ebooks():
    # Loading by path avoids Unsloth's accelerator initialization, like test_raw_text.py.
    path = Path(__file__).resolve().parents[1] / "unsloth/dataprep/ebooks.py"
    spec = importlib.util.spec_from_file_location("ebook_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer():
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = "Chapter One Two First Second paragraph hello world café Fiction Ada Book".split()
    words += [f"word{i}" for i in range(50)]
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}, unk_token = "[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "[UNK]")


@pytest.fixture
def byte_tokenizer():
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    backend = Tokenizer(models.BPE(vocab = vocab, merges = []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object = backend)


def make_epub(
    path,
    *,
    chapters = None,
    spine = None,
    extra = None,
    encoding = "utf-8",
):
    if chapters is None:
        chapters = {
            "second.xhtml": "<h1>Chapter Two</h1><p>Second paragraph.</p>",
            "first.xhtml": "<h1>Chapter One</h1><p>First <em>paragraph</em>.</p>",
        }
    if spine is None:
        spine = list(reversed(chapters))
    manifest = "".join(
        f'<item id="c{i}" href="{escape(name)}" media-type="application/xhtml+xml"/>'
        for i, name in enumerate(chapters)
    )
    ids = {name: f"c{i}" for i, name in enumerate(chapters)}
    reading_order = "".join(f'<itemref idref="{ids[name]}"/>' for name in spine)
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="uid">urn:uuid:ebook-test</dc:identifier>
        <dc:title>Book</dc:title><dc:creator>Ada</dc:creator><dc:subject>Fiction</dc:subject>
        <dc:language>en</dc:language>
      </metadata><manifest>{manifest}</manifest><spine>{reading_order}</spine>
    </package>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="Books/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""",
        )
        archive.writestr("Books/content.opf", opf)
        for name, body in chapters.items():
            html = f'<?xml version="1.0" encoding="{encoding}"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Clutter</title><style>unwanted CSS</style></head><body>{body}</body></html>'
            archive.writestr("Books/" + name.replace("%20", " "), html.encode(encoding))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_epub_reads_spine_and_preserves_paragraphs(ebooks, tokenizer, tmp_path):
    book = make_epub(tmp_path / "book.epub")
    dataset = ebooks.load_ebook_dataset(book, tokenizer = tokenizer)
    assert dataset.column_names == ["text"]
    assert (
        dataset[0]["text"] == "Chapter One\n\nFirst paragraph.\n\nChapter Two\n\nSecond paragraph."
    )


def test_cleaning_and_unicode(ebooks, tokenizer, tmp_path):
    body = '<p>hello&nbsp;<em>world</em> café &amp; tea.</p><script>bad script</script><nav>bad navigation</nav><svg><text>bad image</text></svg><p hidden="hidden">bad hidden</p><p>Next<br/>line.</p>'
    book = make_epub(
        tmp_path / "unicode.epub", chapters = {"chapter%20one.xhtml": body}, encoding = "utf-16"
    )
    dataset = ebooks.load_ebook_dataset(book, tokenizer = tokenizer)
    assert dataset[0]["text"] == "hello world café & tea.\n\nNext\nline."


@pytest.mark.parametrize(
    "content", ["The setting encoding=auto selects café.", "Use encoding=latin1 for café."]
)
def test_encoding_words_in_prose_do_not_change_decoding(ebooks, content):
    data = ("<html><body><p>" + content + "</p></body></html>").encode("utf-8")
    assert ebooks._html_text(data) == content


def test_cdata_in_xhtml_keeps_literal_text(ebooks):
    assert (
        ebooks._html_text(b"<html><body><p><![CDATA[Actual <book> & text]]></p></body></html>")
        == "Actual <book> & text"
    )


def test_preformatted_text_keeps_its_layout(ebooks):
    data = b"<html><body><p>Intro  text</p><pre><code>def f():\n    return 1</code></pre></body></html>"
    assert ebooks._html_text(data) == "Intro text\n\ndef f():\n    return 1"


def test_definition_lists_and_figures_do_not_merge_words(ebooks):
    data = (
        b"<html><body><dl><dt>Term</dt><dd>Definition</dd></dl>"
        b"<figure><img src='a.png'/><figcaption>Caption</figcaption></figure><aside>Note</aside></body></html>"
    )
    assert ebooks._html_text(data) == "Term\n\nDefinition\n\nCaption\n\nNote"


def test_bom_is_dropped_when_xml_declares_utf8(ebooks):
    data = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?><html><body><p>Chapter</p></body></html>'
    assert ebooks._html_text(data) == "Chapter"


def test_declared_html_charset_is_honored(ebooks):
    data = (
        '<html><head><meta charset="windows-1252"/></head><body><p>café</p></body></html>'.encode(
            "windows-1252"
        )
    )
    assert ebooks._html_text(data) == "café"


def test_html_body_closes_implicit_head(ebooks):
    assert ebooks._html_text(b"<head><title>Book</title><body><p>hello</p>") == "hello"


@pytest.mark.parametrize(
    "meta",
    [
        """<meta http-equiv="Content-Type" content='text/html; charset="windows-1252"'>""",
        """<meta http-equiv="Content-Type" content="text/html; charset=&quot;windows-1252&quot;">""",
    ],
)
def test_quoted_mime_charset_is_honored(ebooks, meta):
    html = f"<html><head>{meta}</head><body><p>café</p></body></html>"
    assert ebooks._html_text(html.encode("windows-1252")) == "café"


def test_optional_metadata_and_chapter_markers(ebooks, tokenizer, tmp_path):
    book = make_epub(tmp_path / "book.epub")
    row = ebooks.load_ebook_dataset(
        book, tokenizer = tokenizer, include_metadata = True, chapter_markers = True
    )[0]
    assert {key: row[key] for key in ("title", "author", "genre")} == {
        "title": "Book",
        "author": "Ada",
        "genre": "Fiction",
    }
    assert row["source"] == str(book.resolve())
    assert row["chunk_id"] == 0
    assert row["text"].count("<|chapter_start|>") == 2
    assert row["text"].count("<|chapter_end|>") == 2


def test_chunk_size_overlap_and_complete_coverage(ebooks, tokenizer, tmp_path):
    words = [f"word{i}" for i in range(23)]
    book = make_epub(
        tmp_path / "book.epub", chapters = {"one.xhtml": "<p>" + " ".join(words) + "</p>"}
    )
    rows = ebooks.load_ebook_dataset(book, tokenizer = tokenizer, chunk_size = 8, overlap = 2)["text"]
    assert [row.split() for row in rows] == [words[0:8], words[6:14], words[12:20], words[18:23]]
    assert all(len(tokenizer.encode(row, add_special_tokens = False)) <= 8 for row in rows)


def test_prefers_sentence_boundary(ebooks, tokenizer, tmp_path):
    book = make_epub(
        tmp_path / "book.epub",
        chapters = {
            "one.xhtml": "<p>One Two First paragraph. Second paragraph hello world café.</p>"
        },
    )
    rows = ebooks.load_ebook_dataset(book, tokenizer = tokenizer, chunk_size = 7, overlap = 1)["text"]
    assert rows[0] == "One Two First paragraph."
    assert rows[1].startswith("paragraph. Second")


def test_library_is_recursive_sorted_and_books_never_mix(ebooks, tokenizer, tmp_path):
    (tmp_path / "nested").mkdir()
    make_epub(tmp_path / "z.EPUB", chapters = {"a.xhtml": "<p>Second</p>"})
    make_epub(tmp_path / "nested/a.epub", chapters = {"b.xhtml": "<p>First</p>"})
    (tmp_path / "ignored.txt").write_text("not an ebook")
    rows = ebooks.load_ebook_dataset(tmp_path, tokenizer = tokenizer, file_types = [".EPUB"])["text"]
    assert rows == ["First", "Second"]


@pytest.mark.parametrize("file_types", [None, ["notes"]])
def test_new_reader_reuses_discovery_chunking_and_metadata(
    ebooks, tokenizer, tmp_path, monkeypatch, file_types
):
    def read_notes(path, chapter_markers):
        return path.read_text(encoding = "utf-8"), {
            "title": "Notes",
            "author": "Ada",
            "genre": "Fiction",
        }

    # A future format supplies only its reader and one extension mapping entry.
    monkeypatch.setattr(ebooks, "_READERS", {".notes": read_notes}, raising = False)
    (tmp_path / "book.notes").write_text("word0 word1 word2 word3 word4")
    kwargs = {} if file_types is None else {"file_types": file_types}
    dataset = ebooks.load_ebook_dataset(
        tmp_path, tokenizer = tokenizer, chunk_size = 3, overlap = 1, include_metadata = True, **kwargs
    )
    assert dataset["text"] == ["word0 word1 word2", "word2 word3 word4"]
    assert dataset["title"] == ["Notes", "Notes"]
    assert dataset["chunk_id"] == [0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 3, "overlap": 3},
        {"overlap": -1},
        {"chunk_size": 1.5},
        {"file_types": ["pdf"]},
        {"file_types": []},
    ],
)
def test_invalid_options(ebooks, tokenizer, tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        ebooks.load_ebook_dataset(tmp_path, tokenizer = tokenizer, **kwargs)


def test_requires_tokenizer_instead_of_guessing_token_counts(ebooks, tmp_path):
    with pytest.raises(ValueError, match = "tokenizer"):
        ebooks.load_ebook_dataset(tmp_path)


def test_empty_missing_unsupported_and_corrupt_inputs(ebooks, tokenizer, tmp_path):
    with pytest.raises(FileNotFoundError):
        ebooks.load_ebook_dataset(tmp_path / "missing.epub", tokenizer = tokenizer)
    with pytest.raises(ValueError, match = "No .*ebook"):
        ebooks.load_ebook_dataset(tmp_path, tokenizer = tokenizer)
    other = tmp_path / "file.txt"
    other.write_text("not a book")
    with pytest.raises(ValueError, match = "Unsupported"):
        ebooks.load_ebook_dataset(other, tokenizer = tokenizer)
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match = "bad.epub"):
        ebooks.load_ebook_dataset(bad, tokenizer = tokenizer)
    empty = make_epub(tmp_path / "empty.epub", chapters = {"empty.xhtml": "<p> </p>"})
    with pytest.raises(ValueError, match = "readable text"):
        ebooks.load_ebook_dataset(empty, tokenizer = tokenizer)


def test_non_linear_and_nav_documents_are_not_training_text(ebooks, tokenizer, tmp_path):
    book = make_epub(tmp_path / "book.epub")
    with zipfile.ZipFile(book) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    files["Books/content.opf"] = files["Books/content.opf"].replace(
        b'idref="c0"', b'idref="c0" linear="no"'
    )
    with zipfile.ZipFile(book, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    assert "Chapter Two" not in ebooks.load_ebook_dataset(book, tokenizer = tokenizer)[0]["text"]


def test_encrypted_chapter_is_rejected_but_obfuscated_fonts_are_allowed(
    ebooks, tokenizer, tmp_path
):
    encryption = """<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><CipherData><CipherReference URI="Books/first.xhtml"/></CipherData></EncryptedData></encryption>"""
    book = make_epub(tmp_path / "drm.epub", extra = {"META-INF/encryption.xml": encryption})
    with pytest.raises(ValueError, match = "[Ee]ncrypt|DRM"):
        ebooks.load_ebook_dataset(book, tokenizer = tokenizer)
    book = make_epub(
        tmp_path / "font.epub",
        extra = {
            "META-INF/encryption.xml": encryption.replace("Books/first.xhtml", "Fonts/font.otf")
        },
    )
    assert len(ebooks.load_ebook_dataset(book, tokenizer = tokenizer)) == 1


def test_archive_size_limit(ebooks, tokenizer, tmp_path, monkeypatch):
    book = make_epub(tmp_path / "book.epub")
    monkeypatch.setattr(ebooks, "_MAX_MEMBER_BYTES", 32)
    with pytest.raises(ValueError, match = "size limit"):
        ebooks.load_ebook_dataset(book, tokenizer = tokenizer)


def test_missing_mobi_converter_has_actionable_error(ebooks, tokenizer, tmp_path, monkeypatch):
    book = tmp_path / "book.mobi"
    book.write_bytes(b"placeholder")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ImportError, match = "[Cc]alibre.*ebook-convert"):
        ebooks.load_ebook_dataset(book, tokenizer = tokenizer)


def test_mobi_conversion_failure_reports_source(ebooks, tokenizer, tmp_path):
    if not shutil.which("ebook-convert"):
        pytest.skip("Calibre is optional")
    book = tmp_path / "broken.mobi"
    book.write_bytes(b"not a mobi")
    with pytest.raises(ValueError, match = "broken.mobi"):
        ebooks.load_ebook_dataset(book, tokenizer = tokenizer)


def test_real_mobi_round_trip(ebooks, tokenizer, tmp_path):
    converter = shutil.which("ebook-convert")
    if not converter:
        pytest.skip("Calibre is optional")
    epub = make_epub(tmp_path / "source.epub")
    mobi = tmp_path / "book with spaces.mobi"
    subprocess.run(
        [converter, str(epub), str(mobi), "--no-inline-toc"],
        check = True,
        capture_output = True,
        timeout = 120,
    )
    dataset = ebooks.load_ebook_dataset(mobi, tokenizer = tokenizer, include_metadata = True)
    text = dataset[0]["text"]
    assert text.index("Chapter One") < text.index("Chapter Two")
    assert "First paragraph." in text and "Second paragraph." in text
    assert dataset[0]["title"] == "Book" and dataset[0]["author"] == "Ada"
    assert dataset[0]["source"] == str(mobi.resolve())


def test_public_exports_cover_gpu_mlx_and_dataprep(ebooks):
    root = Path(__file__).resolve().parents[1] / "unsloth"
    # Execute the MLX loading block without importing its unrelated model backends.
    tree = ast.parse((root / "__init__.py").read_text(encoding = "utf-8"))
    branch = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "_IS_MLX"
    )
    statements = [n for n in branch.body if "_ebook" in ast.unparse(n)]
    namespace = {
        "__file__": str(root / "__init__.py"),
        "importlib": __import__("importlib"),
        "_Path": Path,
    }
    exec(
        compile(ast.Module(body = statements, type_ignores = []), "mlx_ebook_export", "exec"), namespace
    )
    assert callable(namespace.get("load_ebook_dataset"))
    for name in ("_gpu_init.py", "dataprep/__init__.py"):
        imports = [
            n
            for n in ast.walk(ast.parse((root / name).read_text(encoding = "utf-8")))
            if isinstance(n, ast.ImportFrom) and n.module
        ]
        assert any(
            (
                n.module.endswith("ebooks")
                and any(a.name in {"load_ebook_dataset", "*"} for a in n.names)
            )
            or (n.module.endswith("dataprep") and any(a.name == "ebooks" for a in n.names))
            for n in imports
        )


@pytest.mark.parametrize("chunk_size", [10, 13, 16, 25])
def test_byte_bpe_chunks_preserve_unicode_and_fit_budget(
    ebooks, byte_tokenizer, tmp_path, chunk_size
):
    tokenizer = byte_tokenizer
    text = "café 🦥 中文 hello world. " * 8
    book = make_epub(tmp_path / "unicode.epub", chapters = {"one.xhtml": "<p>" + text + "</p>"})
    rows = ebooks.load_ebook_dataset(book, tokenizer = tokenizer, chunk_size = chunk_size, overlap = 3)[
        "text"
    ]
    assert all(
        "\ufffd" not in row and len(tokenizer.encode(row, add_special_tokens = False)) <= chunk_size
        for row in rows
    )
    for char in "é🦥中文":
        assert sum(row.count(char) for row in rows) >= text.count(char)
    # A zero-overlap split must not lose or duplicate even a multibyte character.
    rows = ebooks.load_ebook_dataset(book, tokenizer = tokenizer, chunk_size = chunk_size, overlap = 0)[
        "text"
    ]
    assert "".join("".join(rows).split()) == "".join(text.split())


@pytest.mark.parametrize(
    "html, chunk_size, overlap, expected",
    [
        ("<p>aaa.</p><p>bbb</p>", 4, 0, ["aaa.", "bbb"]),
        ("<p>a b</p>", 1, 0, ["a", "b"]),
        ("<p>a</p><p>b</p>", 1, 0, ["a", "b"]),
        ("<p>a   b</p>", 2, 1, ["a", "b"]),
    ],
)
def test_whitespace_windows_do_not_abort_books(
    ebooks, byte_tokenizer, html, chunk_size, overlap, expected
):
    text = ebooks._html_text(html.encode())
    chunks = list(ebooks._chunk_text(text, byte_tokenizer, chunk_size, overlap))
    assert chunks == expected
    assert all(
        len(byte_tokenizer.encode(chunk, add_special_tokens = False)) <= chunk_size
        for chunk in chunks
    )


def test_tiny_budget_still_rejects_an_incomplete_unicode_character(ebooks, byte_tokenizer):
    with pytest.raises(ValueError, match = "complete character"):
        list(ebooks._chunk_text("🦥", byte_tokenizer, 1, 0))
