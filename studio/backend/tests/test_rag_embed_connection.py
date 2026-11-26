# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG embeddings served by a saved connection (connection:<name>/<model>)."""

from pathlib import Path
import json
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.rag import config, embed_connection, embeddings

_PROVIDERS = [
    {
        "id": "p-1",
        "provider_type": "custom",
        "display_name": "Scheduler",
        "base_url": "http://scheduler.local:9000/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
    {
        "id": "p-2",
        "provider_type": "custom",
        "display_name": "Scheduler/gpu1",
        "base_url": "http://gpu1.local:9000/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
    {
        "id": "p-3",
        "provider_type": "custom",
        "display_name": "Off",
        "base_url": "http://off.local/v1",
        "is_enabled": False,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
]


@pytest.fixture(autouse = True)
def _providers(monkeypatch):
    from storage import credential_secrets, providers_db

    monkeypatch.setattr(providers_db, "list_providers", lambda: [dict(p) for p in _PROVIDERS])
    monkeypatch.setattr(
        credential_secrets, "get_provider_api_key", lambda pid: "sk-test" if pid == "p-1" else None
    )
    embeddings._reset_backend()
    yield
    embeddings._reset_backend()


def _server(handler):
    calls = []

    def _handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    return calls, httpx.Client(transport = httpx.MockTransport(_handle))


def _vectors(request, _n):
    body = json.loads(request.content)
    # Reversed on purpose: the spec lets a server reorder, and index puts it back.
    data = [{"index": i, "embedding": [float(i + 1), 0.0, 0.0]} for i in range(len(body["input"]))]
    return httpx.Response(200, json = {"data": list(reversed(data))})


def test_reference_resolves_by_name_id_and_longest_name():
    provider, model = embed_connection.resolve_connection_model("connection:scheduler/BAAI/bge-m3")
    assert (provider["id"], model) == ("p-1", "BAAI/bge-m3")
    provider, model = embed_connection.resolve_connection_model("connection:Scheduler/gpu1/bge-m3")
    assert (provider["id"], model) == ("p-2", "bge-m3")
    provider, model = embed_connection.resolve_connection_model("connection:p-1/nomic-embed-text")
    assert (provider["id"], model) == ("p-1", "nomic-embed-text")


@pytest.mark.parametrize(
    "ref, needle",
    [
        ("connection:Nope/bge", "No saved connection"),
        ("connection:Scheduler/", "names no model"),
        ("connection:Off/bge", "disabled"),
    ],
)
def test_bad_reference_names_the_problem(ref, needle):
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = needle):
        embed_connection.resolve_connection_model(ref)


def test_encode_batches_orders_normalizes_and_authenticates(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH", 2)
    calls, http = _server(_vectors)
    backend = embed_connection.ConnectionEmbeddingBackend()
    backend._http = http

    out = backend.encode(["a", "b", "c"], model_name = "connection:Scheduler/bge-m3")

    assert out.shape == (3, 3) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis = 1), 1.0)
    assert len(calls) == 2
    assert str(calls[0].url) == "http://scheduler.local:9000/v1/embeddings"
    assert calls[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(calls[0].content) == {
        "model": "bge-m3",
        "input": ["a", "b"],
        "encoding_format": "float",
    }
    assert backend.dim(model_name = "connection:Scheduler/bge-m3") == 3


def test_busy_server_is_retried_and_a_refusal_is_reported(monkeypatch):
    monkeypatch.setattr(embed_connection.time, "sleep", lambda s: None)
    calls, http = _server(lambda r, n: httpx.Response(503) if n == 1 else _vectors(r, n))
    backend = embed_connection.ConnectionEmbeddingBackend()
    backend._http = http
    assert backend.encode(["a"], model_name = "connection:Scheduler/bge").shape == (1, 3)
    assert len(calls) == 2

    _, backend._http = _server(lambda r, n: httpx.Response(404, text = "no such model"))
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = "HTTP 404"):
        backend.encode(["a"], model_name = "connection:Scheduler/chat-only")


def test_connection_model_uses_the_connection_backend_whatever_the_env_says(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "llama-server")
    backend = embeddings._get_backend("connection:Scheduler/bge-m3")
    assert isinstance(backend, embed_connection.ConnectionEmbeddingBackend)
    assert not embeddings.active_backend_is_llama("connection:Scheduler/bge-m3")
    assert embeddings.backend_is_loaded("connection:Scheduler/bge-m3")
    assert embeddings.batches_across_documents("connection:Scheduler/bge-m3")


def test_identity_is_tagged_and_round_trips():
    name = "connection:Scheduler/BAAI/bge-m3"
    identity = embeddings.embedding_identity(name)
    assert identity.startswith("connection:")
    assert config.embedding_identity_model(identity) == name
    assert identity != config.embedding_identity("sentence-transformers", "BAAI/bge-m3")


def test_hub_models_keep_their_backend_choice(monkeypatch):
    monkeypatch.setattr(embeddings, "_resolve_auto_for_model", lambda m = None: "llama-server")
    assert embeddings._backend_choice("auto", "unsloth/bge-small-en-v1.5") == "llama-server"
    assert embeddings._backend_choice("sentence-transformers", "org/x") == "sentence-transformers"
    assert embeddings._backend_choice("auto", "org/x", "llama-server") == "llama-server"


@pytest.fixture
def client(monkeypatch):
    import routes.settings as settings

    saved = {}
    monkeypatch.setattr(
        settings, "set_rag_embedding_model", lambda m, **kw: saved.update(model = m, **kw)
    )
    stub = settings.EmbeddingModelResponse(
        embedding_model = "m",
        embedding_gguf_repo = "",
        default_embedding_model = "d",
        default_embedding_gguf_repo = "",
        is_custom = True,
    )
    monkeypatch.setattr(settings, "_embedding_model_response", lambda: stub)
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: pytest.fail("hub path"))
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: True
    c = TestClient(app, raise_server_exceptions = False)
    c.saved = saved
    c.settings = settings
    return c


def test_resolve_plans_no_download_and_reports_missing_connections(client):
    body = client.get(
        "/embedding-model/resolve", params = {"model": "connection:Scheduler/bge-m3"}
    ).json()
    assert body["backend"] == "connection"
    assert body["download_repo"] is None and body["error"] is None
    body = client.get("/embedding-model/resolve", params = {"model": "connection:Nope/x"}).json()
    assert "No saved connection" in body["error"]


def test_save_probes_the_endpoint_and_stores_the_connection_backend(client, monkeypatch):
    probed = []
    monkeypatch.setattr(
        embed_connection.ConnectionEmbeddingBackend,
        "warm",
        lambda self, model_name = None: probed.append(model_name),
    )
    put = {"embedding_model": "connection:Scheduler/bge-m3", "backend": "connection"}
    assert client.put("/embedding-model", json = put).status_code == 200
    assert probed == ["connection:Scheduler/bge-m3"]
    assert client.saved == {"model": "connection:Scheduler/bge-m3", "backend": "connection"}


def test_save_refuses_a_failing_probe_unless_forced(client, monkeypatch):
    def _refuse(self, model_name = None):
        raise embed_connection.ConnectionEmbedderError("HTTP 404: not an embedding model")

    monkeypatch.setattr(embed_connection.ConnectionEmbeddingBackend, "warm", _refuse)
    put = {"embedding_model": "connection:Scheduler/chat-only"}
    response = client.put("/embedding-model", json = put)
    assert response.status_code == 409 and "HTTP 404" in response.json()["detail"]
    assert client.saved == {}
    assert client.put("/embedding-model", json = {**put, "force": True}).status_code == 200
    assert client.saved["backend"] == "connection"


@pytest.mark.parametrize(
    "text, bge_tokens",
    [
        ("a " * 700, 700),
        (
            "\n".join(
                ",".join(str((r * 37 + c * 101) % 1000) for c in range(8)) for r in range(60)
            ),
            1162,
        ),
        ("def f(x_1, y_2):\n    return {'k': x_1[0] + y_2.get('a', 3.14)}\n" * 15, 600),
    ],
)
def test_token_estimate_never_undercounts_a_512_token_wordpiece_embedder(text, bge_tokens):
    # Counts from unsloth/bge-small-en-v1.5's tokenizer; under-counting overflows its 512 limit.
    assert embed_connection._estimate_tokens(text) >= bge_tokens


def test_requests_stay_within_the_tei_default_client_batch(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH", 64)
    calls, http = _server(_vectors)
    backend = embed_connection.ConnectionEmbeddingBackend()
    backend._http = http
    backend.encode([str(i) for i in range(40)], model_name = "connection:Scheduler/bge")
    assert [len(json.loads(c.content)["input"]) for c in calls] == [32, 8]


def test_planner_reports_the_connection_backend():
    assert embeddings.resolved_backend_for_model("connection:Scheduler/bge") == "connection"


def test_an_edit_between_reads_never_pairs_the_new_key_with_the_old_url(monkeypatch):
    from storage import credential_secrets, providers_db

    urls = iter(["http://old.local/v1", "http://new.local/v1", "http://new.local/v1"])
    rows = []

    def _list():
        row = {**_PROVIDERS[0], "base_url": next(urls, "http://new.local/v1")}
        rows.append(row["base_url"])
        return [row]

    monkeypatch.setattr(providers_db, "list_providers", _list)
    monkeypatch.setattr(credential_secrets, "get_provider_api_key", lambda pid: "sk-new")
    client, _, _ = embed_connection._client_for("connection:Scheduler/bge")
    assert client.base_url == "http://new.local/v1"
    assert rows[:2] == ["http://old.local/v1", "http://new.local/v1"]


def _edit(monkeypatch, *rows):
    from storage import providers_db
    monkeypatch.setattr(providers_db, "list_providers", lambda: [dict(r) for r in rows])


@pytest.mark.parametrize(
    "change", [{"base_url": "http://other.local:9000/v1"}, {"provider_type": "openai"}]
)
def test_vectors_from_the_old_server_are_stale_once_the_connection_is_edited(monkeypatch, change):
    name = "connection:Scheduler/bge-m3"
    before = embeddings.embedding_identity(name)
    _edit(monkeypatch, {**_PROVIDERS[0], **change})
    assert not config.embedding_identity_matches(before, embeddings.embedding_identity(name))


def test_an_old_name_taken_by_another_server_does_not_inherit_its_vectors(monkeypatch):
    name = "connection:Scheduler/bge-m3"
    before = embeddings.embedding_identity(name)
    _edit(
        monkeypatch,
        {**_PROVIDERS[0], "display_name": "Renamed"},
        {**_PROVIDERS[0], "id": "p-9", "base_url": "http://other.local:9000/v1"},
    )
    assert not config.embedding_identity_matches(before, embeddings.embedding_identity(name))


@pytest.mark.parametrize(
    "before, after, same",
    [
        ("http://scheduler.local:9000/v1", "http://Scheduler.LOCAL:9000/", True),
        ("http://u:old@scheduler.local:9000/v1?k=1", "http://u:new@scheduler.local:9000/v1", True),
        ("http://localhost:8080/v1", "http://127.0.0.1:8080", True),
        ("http://[::1]:8080/v1", "http://localhost:8080/v1", True),
        ("http://[fd00::1]:8080/v1", "http://[FD00:0:0:0:0:0:0:1]:8080/v1", True),
        ("https://api.example.com/v1", "https://api.example.com:443/v1", True),
        ("http://localhost:8080/v1", "http://localhost:8081/v1", False),
        ("http://scheduler.local:9000/v1", "http://scheduler.local:9000/tei/v1", False),
        ("http://gpu1.local:9000/v1", "http://gpu2.local:9000/v1", False),
    ],
)
def test_only_a_different_server_counts_as_a_move(monkeypatch, before, after, same):
    name = "connection:Scheduler/bge-m3"
    _edit(monkeypatch, {**_PROVIDERS[0], "base_url": before})
    old = embeddings.embedding_identity(name)
    _edit(monkeypatch, {**_PROVIDERS[0], "base_url": after})
    assert config.embedding_identity_matches(old, embeddings.embedding_identity(name)) is same


def test_an_edit_that_keeps_the_server_keeps_the_vectors_current(monkeypatch):
    name = "connection:Scheduler/bge-m3"
    before = embeddings.embedding_identity(name)
    _edit(monkeypatch, {**_PROVIDERS[0], "is_enabled": False, "extra_headers": {"x": "1"}})
    assert config.embedding_identity_matches(before, embeddings.embedding_identity(name))


def test_vectors_carry_the_server_that_embedded_them(monkeypatch):
    name = "connection:Scheduler/bge-m3"

    def _edited_after_answering(request, n):
        _edit(monkeypatch, {**_PROVIDERS[0], "base_url": "http://other.local:9000/v1"})
        return _vectors(request, n)

    before = embeddings.embedding_identity(name)
    _, embeddings._get_backend(name)._http = _server(_edited_after_answering)
    _, identity = embeddings.encode_with_identity(["a"], model_name = name)
    assert identity == before != embeddings.embedding_identity(name)


def test_an_encode_split_across_two_servers_is_refused(monkeypatch):
    name = "connection:Scheduler/bge-m3"
    monkeypatch.setattr(config, "EMBED_BATCH", 1)

    def _moved_after_the_first_batch(request, n):
        if n == 1:
            _edit(monkeypatch, {**_PROVIDERS[0], "base_url": "http://other.local:9000/v1"})
        return _vectors(request, n)

    _, embeddings._get_backend(name)._http = _server(_moved_after_the_first_batch)
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = "changed while embedding"):
        embeddings.encode_with_identity(["a", "b"], model_name = name)


def test_documents_indexed_before_the_upgrade_stay_current_until_the_server_changes(
    rag_conn, monkeypatch
):
    from storage import rag_db

    name = "connection:Scheduler/bge-m3"
    rag_conn.execute(
        "INSERT INTO documents(id, scope, filename, sha256, status, created_at, embedding_model) "
        "VALUES ('d1', 'kb_a', 'a.txt', 'h1', 'completed', '2026-01-01', ?)",
        (config.embedding_identity("connection", name),),
    )
    rag_conn.execute("PRAGMA user_version = 1")
    rag_conn.commit()
    rag_db.reset_schema_state_for_tests()
    conn = rag_db.get_connection()
    try:
        stored = conn.execute("SELECT embedding_model FROM documents WHERE id='d1'").fetchone()[0]
    finally:
        conn.close()
    assert config.embedding_identity_matches(stored, embeddings.embedding_identity(name))
    _edit(monkeypatch, {**_PROVIDERS[0], "base_url": "http://other.local:9000/v1"})
    assert not config.embedding_identity_matches(stored, embeddings.embedding_identity(name))


def test_a_re_embed_from_the_old_server_is_not_written_after_the_connection_moves(
    rag_conn, monkeypatch
):
    from core.rag import chunking, store

    name = "connection:Scheduler/bge-m3"
    old = embeddings.embedding_identity(name)
    scope = store.kb_scope("kb_a")
    rag_conn.execute(
        "INSERT INTO documents(id, scope, filename, sha256, status, created_at, embedding_model) "
        "VALUES ('d1', ?, 'a.txt', 'h1', 'completed', '2026-01-01', 'stale')",
        (scope,),
    )
    store.add_lexical_chunks(rag_conn, scope, "d1", [chunking.Chunk("a", 1, None, 0, 0, 0, 1)])
    rag_conn.commit()
    monkeypatch.setattr(config, "effective_embedding_model", lambda fresh = False: name)
    write = lambda: store.replace_document_vectors(
        rag_conn, "d1", "stale", ["d1:0"], [[1.0, 0.0]], old, name
    )
    _edit(monkeypatch, {**_PROVIDERS[0], "base_url": "http://other.local:9000/v1"})
    assert not write()
    _edit(monkeypatch, _PROVIDERS[0])
    assert write()
