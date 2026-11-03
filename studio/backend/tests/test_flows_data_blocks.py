# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import httpx
import pytest

from core.flows import assets as asset_store
from core.flows import caller
from core.flows.nodes import knowledge, text, transform
from core.flows.registry import FlowNodeError

from .flows_player_fakes import FakeCtx, run

ROWS = [
    {"id": 1, "name": "Ada", "score": 7},
    {"id": 2, "name": "bob", "score": 9},
    {"id": 3, "name": "Cy", "score": None},
    {"id": 1, "name": "Ada again", "score": 3},
]


def _item(path):
    return {"var": f"item.{path}"}


def test_edit_fields_on_an_object_and_on_a_list():
    ctx = FakeCtx(
        "json.set",
        params = {"fields": {"label": {"op": "upper", "args": [_item("name")]}, "meta.ok": True}},
        inputs = {"value": ROWS[:2]},
    )
    out = run(transform.edit_fields(ctx))["value"]
    assert out[0]["label"] == "ADA" and out[0]["id"] == 1 and out[0]["meta"] == {"ok": True}
    ctx = FakeCtx(
        "json.set",
        params = {"fields": {"n": {"var": "item.id"}}, "keep_only": True},
        inputs = {"value": ROWS[1]},
    )
    assert run(transform.edit_fields(ctx))["value"] == {"n": 2}
    assert "label" not in ROWS[0]


def test_map_filter_sort_dedupe():
    mapped = run(
        transform.map_list(
            FakeCtx("list.map", params = {"expression": _item("name")}, inputs = {"items": ROWS})
        )
    )
    assert mapped["items"] == ["Ada", "bob", "Cy", "Ada again"]

    kept = run(
        transform.filter_list(
            FakeCtx(
                "list.filter",
                params = {"condition": {"op": ">", "args": [_item("score"), 5]}},
                inputs = {"items": [r for r in ROWS if r["score"] is not None]},
            )
        )
    )
    assert [r["id"] for r in kept["items"]] == [1, 2] and [r["id"] for r in kept["dropped"]] == [1]

    by_score = run(
        transform.sort_list(
            FakeCtx(
                "list.sort",
                params = {"key": _item("score"), "descending": True},
                inputs = {"items": ROWS},
            )
        )
    )
    assert [r["score"] for r in by_score["items"]] == [9, 7, 3, None]
    by_name = run(
        transform.sort_list(
            FakeCtx("list.sort", params = {"key": _item("name"), "limit": 2}, inputs = {"items": ROWS})
        )
    )
    assert [r["name"] for r in by_name["items"]] == ["Ada", "Ada again"]

    deduped = run(
        transform.dedupe_list(
            FakeCtx("list.dedupe", params = {"key": _item("id")}, inputs = {"items": ROWS})
        )
    )
    assert [r["name"] for r in deduped["items"]] == ["Ada", "bob", "Cy"] and deduped["removed"] == 1


def test_transforms_are_bounded_and_report_bad_expressions(monkeypatch):
    monkeypatch.setattr(transform, "MAX_ITEMS", 3)
    with pytest.raises(FlowNodeError, match = "more than the 3"):
        run(transform.map_list(FakeCtx("list.map", inputs = {"items": [1, 2, 3, 4]})))
    with pytest.raises(FlowNodeError, match = "Expression error on item 0"):
        run(
            transform.map_list(
                FakeCtx("list.map", params = {"expression": {"op": "nope"}}, inputs = {"items": [1]})
            )
        )
    with pytest.raises(FlowNodeError, match = "needs a list"):
        run(transform.filter_list(FakeCtx("list.filter", inputs = {"items": "text"})))


def test_merge_modes():
    def merge(a, b, **params):
        return run(
            transform.merge_data(FakeCtx("json.merge", params = params, inputs = {"a": a, "b": b}))
        )["value"]

    assert merge([1, 2], 3, mode = "append") == [1, 2, 3]
    assert merge({"a": 1, "b": 1}, {"b": 2}, mode = "combine") == {"a": 1, "b": 2}
    assert merge([{"a": 1}, {"a": 2}], [{"b": 1}], mode = "combine") == [{"a": 1, "b": 1}, {"a": 2}]
    left = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    right = [{"id": 2, "y": "B"}, {"id": 3, "y": "C"}]
    assert merge(left, right, mode = "by_key", key = "id") == [
        {"id": 1, "x": "a"},
        {"id": 2, "x": "b", "y": "B"},
        {"id": 3, "y": "C"},
    ]
    assert merge(left, right, mode = "by_key", key = "id", keep_unmatched = False) == [
        {"id": 2, "x": "b", "y": "B"}
    ]
    lines = [{"id": 2, "sku": "x"}, {"id": 2, "sku": "y"}]
    assert merge([{"id": 2, "who": "A"}], lines, mode = "by_key", key = "id") == [
        {"id": 2, "who": "A", "sku": "x"},
        {"id": 2, "who": "A", "sku": "y"},
    ]
    assert merge(left, [{"y": "N"}], mode = "by_key", key = "id") == [*left, {"y": "N"}]
    assert merge(left, [{"y": "N"}], mode = "by_key", key = "id", keep_unmatched = False) == []
    with pytest.raises(FlowNodeError, match = "at most"):
        merge(list(range(transform.MAX_ITEMS + 1)), [1], mode = "combine")


def test_edit_fields_draws_successive_random_values():
    dice = {"op": "rand"}
    ctx = FakeCtx("json.set", params = {"fields": {"a": dice, "b": dice}}, inputs = {"value": {}})
    out = run(transform.edit_fields(ctx))["value"]
    assert out["a"] != out["b"]


def test_a_long_knowledge_base_name_is_found_again(monkeypatch):
    monkeypatch.setattr(knowledge, "POLL_S", 0)
    long_name = "n" * 250
    ctx = _Rag("kb.add", params = {"knowledge_base": long_name}, inputs = {"file": _pdf_value()})
    assert run(knowledge.add_to_knowledge_base(ctx))["knowledge_base"] == "n" * 200
    again = _Rag(
        "kb.search", kbs = ctx.kbs, params = {"knowledge_base": long_name}, inputs = {"query": "q"}
    )
    assert [c for c in again.calls] == [] and run(knowledge._knowledge_base(again, create = False))[
        0
    ] == "kb2"


def test_split_default_is_unchanged_and_chunks_overlap():
    ctx = FakeCtx("text.split", inputs = {"text": "a\n b \n\nc"})
    assert run(text.split(ctx)) == {"items": ["a", "b", "c"]}
    words = " ".join(f"w{i}" for i in range(10))
    ctx = FakeCtx(
        "text.split",
        params = {"mode": "chunks", "unit": "words", "chunk_size": 4, "overlap": 1},
        inputs = {"text": words},
    )
    assert run(text.split(ctx))["items"] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert text.chunk_text("abcdef", 4, 0, "characters") == ["abcd", "ef"]
    assert text.chunk_text("One. Two! Three? Four.", 2, 0, "sentences") == [
        "One. Two!",
        "Three? Four.",
    ]
    assert text.chunk_text("a b c", 2, 5, "words") == ["a b", "b c"]


def test_documents_are_sniffed_and_text_needs_a_named_type():
    assert asset_store.sniff_mime(b"%PDF-1.7\n...") == "application/pdf"
    assert asset_store.sniff_mime(b"PK\x03\x04 ... word/document.xml ...").endswith(
        "wordprocessingml.document"
    )
    assert asset_store.sniff_mime(b"PK\x03\x04 ... xl/workbook.xml") is None
    assert asset_store.text_hint(None, "notes.md") == "text/markdown"
    assert asset_store.sniff_text(b"# Hi", "text/markdown") == "text/markdown"
    assert asset_store.sniff_text(b"\xff\xfe\x00", "text/plain") is None
    assert asset_store.sniff_text(b"plain", None) is None
    digest, mime = asset_store.store(b"hello", "text/plain")
    assert mime == "text/plain" and asset_store.media_value(digest, mime)["kind"] == "document"


class _Rag(FakeCtx):
    def __init__(
        self,
        node_type,
        kbs = None,
        **kw,
    ):
        super().__init__(node_type, **kw)
        self.kbs = list(kbs or [])
        self.calls = []

    async def call(
        self,
        method,
        path,
        json = None,
        files = None,
        **kw,
    ):
        self.calls.append((method, path))
        caller.checked_path(httpx.URL("http://studio" + path), method)
        if path == "/api/rag/knowledge-bases" and method == "GET":
            return {"knowledgeBases": self.kbs, "ragAvailable": True}
        if path == "/api/rag/knowledge-bases":
            self.kbs.append({"id": "kb2", "name": json["name"]})
            return {"id": "kb2", "name": json["name"]}
        if path.endswith("/documents"):
            assert files["file"][0] == "a.pdf"
            return {"documentId": "d1", "jobId": "j1"}
        if path == "/api/rag/jobs/j1":
            return {"status": "completed", "progress": 1.0, "numChunks": 4}
        if path == "/api/rag/extract-text":
            return {"text": "Hello", "truncated": False}
        if path == "/api/rag/search":
            assert json["kb_id"] == "kb1"
            return {"results": [{"text": "Passage", "filename": "a.pdf", "page": 2, "score": 0.9}]}
        raise AssertionError(path)


def _pdf_value():
    digest, mime = asset_store.store(b"%PDF-1.4 tiny", None)
    return asset_store.media_value(digest, mime, name = "a.pdf")


def test_load_document_and_knowledge_base_blocks(monkeypatch):
    monkeypatch.setattr(knowledge, "POLL_S", 0)
    doc = _pdf_value()
    ctx = _Rag("doc.load", inputs = {"file": doc})
    assert run(knowledge.load_document(ctx)) == {"text": "Hello", "truncated": False}

    ctx = _Rag("kb.add", params = {"knowledge_base": "Notes"}, inputs = {"file": doc})
    assert run(knowledge.add_to_knowledge_base(ctx)) == {"knowledge_base": "Notes", "chunks": 4}
    assert ("POST", "/api/rag/knowledge-bases") in ctx.calls

    ctx = _Rag("kb.search", kbs = [{"id": "kb1", "name": "Notes"}], params = {"knowledge_base": "notes"},
               inputs = {"query": "what?"})  # fmt: skip
    out = run(knowledge.search_knowledge_base(ctx))
    assert out["passages"] == ["Passage"] and out["context"].startswith("[1] a.pdf, page 2")

    with pytest.raises(FlowNodeError, match = "No knowledge base named"):
        run(knowledge.search_knowledge_base(_Rag("kb.search", inputs = {"query": "q"})))
    with pytest.raises(FlowNodeError, match = "Connect a Document"):
        run(
            knowledge.load_document(
                _Rag("doc.load", inputs = {"file": {"kind": "image", "ref": "image:1"}})
            )
        )


def test_rag_paths_open_to_flows_only():
    from studio_mcp import forward

    url = lambda p: httpx.URL("http://studio" + p)  # noqa: E731
    for path in ("/api/rag/search", "/api/rag/knowledge-bases/kb1/documents", "/api/rag/jobs/j1"):
        caller.checked_path(url(path), "POST")
        with pytest.raises(RuntimeError):
            forward.checked_path(url(path))
    for path in ("/api/rag/linked-folders", "/api/rag/documents", "/api/rag/knowledge-bases/kb1/linked-folders",
                 "/api/rag/jobs/../reembed"):  # fmt: skip
        with pytest.raises(RuntimeError):
            caller.checked_path(url(path), "POST")


def test_knowledge_examples_name_only_studios_local_models():
    from core.flows.examples.build_examples import build
    from core.rag import config

    ask = next(d for d in build() if d["id"] == "ask-documents")
    needs = {m["kind"]: m for m in ask["requirements"]["models"]}
    assert needs["embedding"]["repo"] == config.DEFAULT_EMBEDDING_MODEL
    from core.flows.specs import SPECS_BY_TYPE

    for spec_type in ("doc.load", "kb.add", "kb.search"):
        assert not any(p["kind"] in ("model", "json") for p in SPECS_BY_TYPE[spec_type]["params"])


def test_indexed_documents_keep_their_names_with_the_sniffed_extension():
    digest, mime = asset_store.store(b"%PDF-1.4 tiny", None)

    def name_for(upload_name):
        value = asset_store.media_value(digest, mime, name = upload_name)
        return run(knowledge._upload(_Rag("kb.add", inputs = {"file": value})))[0]

    assert name_for("Q3 report.pdf") == "Q3 report.pdf"
    assert name_for("../../etc/notes.exe") == "notes.pdf"
    assert name_for("<>:.pdf") == "document.pdf"
    assert name_for(None) == "document.pdf"


def test_parallel_adds_to_a_new_knowledge_base_create_it_once(monkeypatch):
    import asyncio

    monkeypatch.setattr(knowledge, "POLL_S", 0)
    kbs = []

    class Racing(_Rag):
        async def call(
            self,
            method,
            path,
            json = None,
            files = None,
            **kw,
        ):
            if path == "/api/rag/knowledge-bases" and method == "GET":
                listed = {"knowledgeBases": list(kbs), "ragAvailable": True}
                await asyncio.sleep(0.01)
                return listed
            if path == "/api/rag/knowledge-bases":
                kbs.append({"id": f"kb{len(kbs) + 1}", "name": json["name"]})
                return dict(kbs[-1])
            return await super().call(method, path, json = json, files = files, **kw)

    async def both():
        ctxs = [
            Racing(
                "kb.add", params = {"knowledge_base": "Shared Docs"}, inputs = {"file": _pdf_value()}
            )
            for _ in range(2)
        ]
        return await asyncio.gather(*(knowledge.add_to_knowledge_base(c) for c in ctxs))

    asyncio.run(both())
    assert [kb["name"] for kb in kbs] == ["Shared Docs"]
