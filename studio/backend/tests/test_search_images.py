# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""web_search image results: server-side registry, model-facing tokens, the
frontend envelope, and the thumbnail proxy that keeps image hosts away from the
browser."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import search_images, tools
from core.inference.tool_loop_controller import strip_result_for_model
from core.inference.tool_stream_exec import accepts_kwarg, search_images_kwargs
from routes.inference import studio_router
from utils import chat_history_policy

RAW_IMAGES = [
    {
        "title": "Golden  Retriever\nportrait",
        "image": "https://img.example.com/golden.jpg",
        "thumbnail": "https://tse1.mm.bing.net/th?id=golden",
        "url": "https://www.akc.org/dog-breeds/golden-retriever/",
        "source": "Bing",
    },
    {
        "title": "Local only",
        "image": "http://127.0.0.1/secret.png",
        "thumbnail": "http://127.0.0.1/secret.png",
        "url": "https://example.com/ok",
    },
    {
        "title": "Bad source",
        "thumbnail": "https://cdn.example.com/t.jpg",
        "url": "javascript:alert(1)",
    },
    {"title": "Missing thumbnail", "url": "https://example.com/page"},
    "not a dict",
]


@pytest.fixture(autouse = True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_inflight", {})
    monkeypatch.setattr(search_images, "_cleared_unservable", set())
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path)


def _png_bytes(size = (64, 48), color = (200, 30, 30)) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format = "PNG")
    return out.getvalue()


def test_register_images_keeps_urls_server_side_and_filters_unsafe_results():
    entries = search_images.register_images(RAW_IMAGES)

    assert len(entries) == 1
    entry = entries[0]
    assert search_images.IMAGE_ID_RE.match(entry["id"])
    assert entry["title"] == "Golden Retriever portrait"
    assert entry["domain"] == "akc.org"
    assert entry["source"] == "https://www.akc.org/dog-breeds/golden-retriever/"
    # The thumbnail URL never reaches the model or the frontend.
    assert "bing" not in json.dumps(entry)
    stored = search_images.lookup_image(entry["id"])
    assert stored["thumbnail"] == "https://tse1.mm.bing.net/th?id=golden"


def test_no_history_policy_keeps_search_images_memory_only(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_history_policy, "NO_CHAT_HISTORY", True)
    monkeypatch.setattr(search_images, "_fetch_thumbnail_bytes", lambda *_args, **_kwargs: b"jpg")

    entry = search_images.register_images(RAW_IMAGES)[0]
    assert search_images.lookup_image(entry["id"]) is not None
    assert list(tmp_path.iterdir()) == []
    assert search_images.thumbnail_bytes(entry["id"]) == b"jpg"
    assert list(tmp_path.iterdir()) == []

    search_images._registry.clear()
    (tmp_path / f"{entry['id']}.jpg").write_bytes(b"legacy-jpg")
    (tmp_path / f"{entry['id']}.json").write_text(
        json.dumps(
            {
                "thumbnail": RAW_IMAGES[0]["thumbnail"],
                "source": RAW_IMAGES[0]["url"],
            }
        ),
        encoding = "utf-8",
    )
    assert search_images.thumbnail_bytes(entry["id"]) is None


def test_register_images_honours_the_website_policy():
    from core.inference.web_access_policy import normalize_website_policy

    policy = normalize_website_policy({"allowedDomains": ["example.com"]})
    entries = search_images.register_images(RAW_IMAGES, policy)
    assert entries == []


def test_register_images_caps_the_count():
    many = [
        {
            "title": f"img {i}",
            "thumbnail": f"https://cdn.example.com/{i}.jpg",
            "url": f"https://example.com/{i}",
        }
        for i in range(20)
    ]
    assert len(search_images.register_images(many)) == search_images.MAX_IMAGES_PER_SEARCH


def test_lookup_rejects_malformed_ids():
    assert search_images.lookup_image("../../etc/passwd") is None
    assert search_images.lookup_image("") is None
    assert search_images.lookup_image("ABCDEFABCDEF") is None


def test_strip_images_suffix_only_removes_a_valid_envelope():
    entries = search_images.register_images(RAW_IMAGES)
    body = "Title: A\nURL: https://a.test\nSnippet: s"
    wrapped = body + search_images.images_envelope(entries)

    assert search_images.strip_images_suffix(wrapped) == body
    assert strip_result_for_model(wrapped, "web_search") == body
    # A sibling sentinel after ours bounds the payload, exactly as the frontend
    # parser and _strip_files_sentinel do; the ids must not survive into the model.
    trailing = wrapped + "\n__FILES__:[]"
    text, entries = search_images.split_images_envelope(trailing)
    assert text == body + "\n__FILES__:[]"
    assert [e["id"] for e in entries] == [e["id"] for e in entries if e["id"]]
    assert entries and "__WEB_IMAGES__" not in text
    assert entries[0]["id"] not in strip_result_for_model(trailing, "web_search")

    # Tool text that merely mentions the marker is content, not an envelope.
    literal = "see\n__WEB_IMAGES__:not json"
    assert search_images.strip_images_suffix(literal) == literal
    foreign = 'x\n__WEB_IMAGES__:[{"id":"zzz"}]'
    assert search_images.strip_images_suffix(foreign) == foreign


def test_web_search_appends_tokens_and_envelope_when_images_are_on(monkeypatch):
    calls = {}

    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "AKC", "href": "https://akc.org/x", "body": "Breeds"}]

        def images(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            calls["images"] = (query, max_results, kwargs)
            return RAW_IMAGES

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = FakeDDGS))

    plain = tools._web_search("dog breeds")
    assert "[[img:" not in plain
    assert "images" not in calls

    result = tools._web_search("dog breeds", include_images = True)
    assert "Title: AKC" in result
    assert "[[img:" in result
    assert search_images.SEARCH_IMAGES_SENTINEL in result
    assert calls["images"][0] == "dog breeds"
    assert calls["images"][2]["safesearch"] == "moderate"

    model_text = strip_result_for_model(result, "web_search")
    assert search_images.SEARCH_IMAGES_SENTINEL not in model_text
    assert "[[img:" in model_text
    assert "bing.net" not in model_text

    envelope = json.loads(result.rsplit(search_images.SEARCH_IMAGES_SENTINEL, 1)[1])
    assert [e["domain"] for e in envelope] == ["akc.org"]
    token_id = envelope[0]["id"]
    assert f"[[img:{token_id}]]" in model_text


def test_web_search_survives_an_image_engine_failure(monkeypatch):
    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "AKC", "href": "https://akc.org/x", "body": "Breeds"}]

        def images(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            raise RuntimeError("engine down")

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = FakeDDGS))
    result = tools._web_search("dog breeds", include_images = True)
    assert "Title: AKC" in result
    assert search_images.SEARCH_IMAGES_SENTINEL not in result


def test_web_search_without_an_images_method_is_unchanged(monkeypatch):
    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "AKC", "href": "https://akc.org/x", "body": "Breeds"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = FakeDDGS))
    assert tools._web_search("q", include_images = True) == tools._web_search("q")


def test_execute_tool_forwards_search_images(monkeypatch):
    seen = {}

    def fake_search(query, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(tools, "_web_search", fake_search)
    tools.execute_tool("web_search", {"query": "q"}, search_images = True)
    assert seen["include_images"] is True
    tools.execute_tool("web_search", {"query": "q"})
    assert seen["include_images"] is False


class _SubjectDDGS:
    calls: list[str] = []

    def __init__(self, **_kwargs):
        pass

    def images(
        self,
        query,
        max_results = 5,
        **kwargs,
    ):
        _SubjectDDGS.calls.append(query)
        if query == "Nothing Here":
            return []
        if query == "Broken":
            raise RuntimeError("engine down")
        slug = query.replace(" ", "_")
        return [
            {
                "title": f"{query} photo",
                "thumbnail": f"https://cdn.example.com/{slug}.jpg",
                "url": f"https://example.com/{slug}",
            },
            {
                "title": f"{query} again",
                "thumbnail": f"https://cdn.example.com/{slug}-2.jpg",
                "url": f"https://example.com/{slug}-2",
            },
            {
                "title": f"{query} third",
                "thumbnail": f"https://cdn.example.com/{slug}-3.jpg",
                "url": f"https://example.com/{slug}-3",
            },
        ]


def test_image_search_returns_tokens_grouped_by_subject(monkeypatch):
    _SubjectDDGS.calls = []
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = _SubjectDDGS))

    result = tools._image_search(
        ["German Shepherd", " german shepherd ", "Labrador", "Nothing Here", "Broken"],
    )

    # Deduped case-insensitively; each subject searched once (concurrently, any order).
    assert sorted(_SubjectDDGS.calls) == sorted(
        ["German Shepherd", "Labrador", "Nothing Here", "Broken"]
    )
    model_text = strip_result_for_model(result, "web_search")
    assert search_images.SEARCH_IMAGES_SENTINEL not in model_text
    assert "German Shepherd:\n- [[img:" in model_text
    assert "Labrador:\n- [[img:" in model_text
    assert "Nothing Here: no image found" in model_text
    assert "Broken: no image found" in model_text
    # The header example plus exactly one token per found subject; spares ride only in
    # the envelope.
    assert model_text.count("[[img:") == 1 + 2
    envelope = json.loads(result.rsplit(search_images.SEARCH_IMAGES_SENTINEL, 1)[1])
    assert len(envelope) == 2 * tools.IMAGE_SEARCH_PER_QUERY
    assert all(search_images.is_image_entry(e) for e in envelope)
    assert [e["subject"] for e in envelope] == ["German Shepherd"] * 2 + ["Labrador"] * 2
    assert "cdn.example.com" not in model_text


def test_image_queries_alone_are_a_pure_image_lookup(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = _SubjectDDGS))
    off = tools.execute_tool("web_search", {"image_queries": ["Pug"]})
    assert off == tools.IMAGE_SEARCH_DISABLED
    on = tools.execute_tool("web_search", {"image_queries": ["Pug"]}, search_images = True)
    assert "Pug:\n- [[img:" in on
    assert "Title:" not in on
    assert tools._image_search([]) == "No subjects provided."
    assert tools._image_search("Pug").count("Pug:") == 1
    assert tools._image_search({"bad": 1}) == "No subjects provided."
    # Without image_queries an empty call is still the old "No query provided."
    assert tools.execute_tool("web_search", {}, search_images = True) == "No query provided."


def test_web_search_with_image_queries_gives_one_picture_per_subject(monkeypatch):
    class Both(_SubjectDDGS):
        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "AKC", "href": "https://akc.org/x", "body": "Breeds"}]

    _SubjectDDGS.calls = []
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = Both))
    result = tools._web_search(
        "top dog breeds", include_images = True, image_queries = ["Pug", "Beagle"]
    )
    assert "Title: AKC" in result
    assert "Pug:\n- [[img:" in result and "Beagle:\n- [[img:" in result
    # The subjects replace the generic pile: no query-wide image lookup ran.
    assert sorted(_SubjectDDGS.calls) == ["Beagle", "Pug"]
    # With the setting off, named subjects are acknowledged rather than dropped.
    off = tools._web_search("top dog breeds", include_images = False, image_queries = ["Pug"])
    assert "Title: AKC" in off and tools.IMAGE_SEARCH_DISABLED in off and "[[img:" not in off
    envelope = json.loads(result.rsplit(search_images.SEARCH_IMAGES_SENTINEL, 1)[1])
    assert {e["subject"] for e in envelope} == {"Pug", "Beagle"}


def test_named_subjects_survive_a_text_sweep_that_finds_nothing(monkeypatch):
    # image_queries is an explicit request that succeeds on its own without a query, so
    # an empty TEXT sweep must not take the pictures down with it.
    class NoText(_SubjectDDGS):
        def text(
            self,
            query,
            max_results = 5,
        ):
            return []

    # ddgs signals an empty sweep by raising; the name is what tools.py matches on.
    class DDGSException(Exception):
        pass

    class RaisesEmpty(_SubjectDDGS):
        def text(
            self,
            query,
            max_results = 5,
        ):
            raise DDGSException("No results found for the given query.")

    class OnlyBlocked(_SubjectDDGS):
        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "X", "href": "https://blocked.example/x", "body": "b"}]

    for engine in (NoText, RaisesEmpty):
        _SubjectDDGS.calls = []
        monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = engine))
        result = tools._web_search("top dog breeds", include_images = True, image_queries = ["Pug"])
        assert tools.EMPTY_SEARCH_RESULTS[0] in result
        assert "Pug:\n- [[img:" in result
        assert sorted(_SubjectDDGS.calls) == ["Pug"]
        # With the setting off the parameter is acknowledged, not silently dropped.
        off = tools._web_search("top dog breeds", include_images = False, image_queries = ["Pug"])
        assert tools.IMAGE_SEARCH_DISABLED in off and "[[img:" not in off
        # No image_queries: the empty answer stays exactly as it was.
        assert (
            tools._web_search("top dog breeds", include_images = True)
            == (tools.EMPTY_SEARCH_RESULTS[0])
        )

    # Every hit filtered out by the website policy is the same empty answer.
    _SubjectDDGS.calls = []
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = OnlyBlocked))
    scoped = tools._web_search(
        "top dog breeds",
        include_images = True,
        image_queries = ["Pug"],
        # The image hosts stay allowed; only the text hit's domain is out of scope.
        website_policy = {"allowedDomains": ["akc.org", "cdn.example.com", "example.com"]},
    )
    assert tools.EMPTY_SEARCH_RESULTS[1] in scoped and "Pug:\n- [[img:" in scoped
    # One image lookup, scoped by the same policy the text sweep used.
    assert len(_SubjectDDGS.calls) == 1 and _SubjectDDGS.calls[0].startswith("Pug")


def test_a_genuine_search_failure_carries_no_pictures(monkeypatch):
    class Boom(_SubjectDDGS):
        def text(
            self,
            query,
            max_results = 5,
        ):
            raise RuntimeError("upstream exploded")

    _SubjectDDGS.calls = []
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = Boom))
    result = tools._web_search("top dog breeds", include_images = True, image_queries = ["Pug"])
    # Pictures under an error would read as a partial answer.
    assert result.startswith("Search failed:") and "[[img:" not in result
    assert _SubjectDDGS.calls == []


def test_clear_all_chats_beats_a_thumbnail_write_already_in_flight(monkeypatch, tmp_path):
    # The fetch copies its registry entry up front, so without the generation check it
    # could land tmp.replace() after the clear and restore a thumbnail on disk, where
    # the cache-first path would keep serving it.
    entry = search_images.register_images(RAW_IMAGES)[0]

    def clear_then_serve(url, **kwargs):
        search_images.clear_cache()
        return None, _png_bytes((60, 40)), "image/png"

    monkeypatch.setattr(tools, "_fetch_url_raw", clear_then_serve)
    assert search_images.thumbnail_bytes(entry["id"]) is None
    assert list(tmp_path.glob("*.jpg")) == []
    assert list(tmp_path.glob("*.tmp")) == []
    # A fetch with no clear racing it still caches, so the guard is not just "never write".
    monkeypatch.setattr(
        tools,
        "_fetch_url_raw",
        lambda url, **kwargs: (None, _png_bytes((60, 40)), "image/png"),
    )
    fresh = search_images.register_images(RAW_IMAGES)[0]
    assert search_images.thumbnail_bytes(fresh["id"]) is not None
    assert (tmp_path / f"{fresh['id']}.jpg").is_file()


def test_clear_all_chats_beats_metadata_waiting_to_publish(monkeypatch, tmp_path):
    real_persist = search_images._persist_entry

    def clear_then_persist(*args):
        search_images.clear_cache()
        real_persist(*args)

    monkeypatch.setattr(search_images, "_persist_entry", clear_then_persist)
    entry = search_images.register_images(RAW_IMAGES)[0]

    assert search_images.lookup_image(entry["id"]) is None
    assert list(tmp_path.glob("*.json")) == []


def test_clear_all_chats_invalidates_an_image_lookup_already_in_a_thread(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingDDGS:
        def __init__(self, **_kwargs):
            pass

        def images(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            started.set()
            release.wait(2)
            return RAW_IMAGES

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = BlockingDDGS))

    async def race_clear_against_lookup():
        task = asyncio.create_task(asyncio.to_thread(tools._image_search, ["Pug"], 20))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            search_images.clear_cache()
        finally:
            release.set()
        return await task

    result = asyncio.run(race_clear_against_lookup())
    assert search_images.SEARCH_IMAGES_SENTINEL not in result
    assert search_images._registry == {}
    assert list(tmp_path.glob("*.json")) == []


def test_clear_all_chats_invalidates_the_plain_query_image_sweep(monkeypatch, tmp_path):
    # The same race on the path with no image_queries: the sweep runs while the user
    # clears every chat, and without the generation it registers afterwards anyway --
    # the sidecar then outlives the clear and keeps the picture fetchable.
    started = threading.Event()
    release = threading.Event()

    class BlockingClient:
        def images(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            started.set()
            release.wait(2)
            return RAW_IMAGES

    async def race_clear_against_sweep():
        task = asyncio.create_task(
            asyncio.to_thread(
                tools._web_search_images_suffix, BlockingClient(), "dog breeds", 5, None, None
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            search_images.clear_cache()
        finally:
            release.set()
        return await task

    assert asyncio.run(race_clear_against_sweep()) == ""
    assert search_images._registry == {}
    assert list(tmp_path.glob("*.json")) == []

    # No clear racing it: the same sweep still returns its images, so the guard is
    # not simply "never register".
    assert search_images.SEARCH_IMAGES_SENTINEL in tools._web_search_images_suffix(
        BlockingClient(), "dog breeds", 5, None, None
    )


def test_a_clear_between_the_two_registry_reads_still_wins(monkeypatch, tmp_path):
    # The exact interleaving the first fix missed: the entry and the cache generation
    # were read under SEPARATE acquisitions, so a clear landing in the gap handed this
    # call the post-clear generation, the check before the write matched, and the
    # thumbnail the clear had just deleted was written straight back.
    entry = search_images.register_images(RAW_IMAGES)[0]
    real_lookup = search_images._lookup_locked

    def clearing_lookup(image_id):
        found = real_lookup(image_id)
        # Inline rather than clear_cache(): the caller already holds _registry_lock. It has
        # to move _full_clear_generation as well, because that -- not the bare generation --
        # is what a clear-everything raises to abort every in-flight fetch. Bumping only the
        # generation simulates a clear that spared this id, which is a different test.
        search_images._registry.clear()
        search_images._cache_generation += 1
        search_images._full_clear_generation = search_images._cache_generation
        for stale in tmp_path.glob("*"):
            stale.unlink()
        return found

    monkeypatch.setattr(search_images, "_lookup_locked", clearing_lookup)
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )
    assert search_images.thumbnail_bytes(entry["id"]) is None
    assert list(tmp_path.glob("*.jpg")) == []


def test_an_id_still_resolves_after_a_restart_that_never_cached_it(monkeypatch, tmp_path):
    # Chat history keeps ids, not URLs, and the browser only asks for a thumbnail once
    # it nears the viewport -- so a picture nobody scrolled to has no bytes on disk. The
    # in-memory registry does not survive the process, and reopening used to 404 forever.
    entry = search_images.register_images(RAW_IMAGES)[0]
    assert (tmp_path / f"{entry['id']}.json").is_file()
    assert not (tmp_path / f"{entry['id']}.jpg").exists(), "never materialized"

    search_images._registry.clear()  # the restart
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )
    data = search_images.thumbnail_bytes(entry["id"])
    assert data is not None and data[:3] == b"\xff\xd8\xff"

    # Clear all chats takes the metadata with the bytes, so nothing resolves afterwards.
    search_images.clear_cache()
    assert list(tmp_path.glob("*.json")) == [] and list(tmp_path.glob("*.jpg")) == []
    assert search_images.thumbnail_bytes(entry["id"]) is None


def test_persisted_metadata_is_re_checked_on_the_way_back_in(monkeypatch, tmp_path):
    entry = search_images.register_images(RAW_IMAGES)[0]
    search_images._registry.clear()
    # A host that is no longer public must not be fetched just because it once was.
    (tmp_path / f"{entry['id']}.json").write_text(
        json.dumps({"thumbnail": "http://127.0.0.1/secret.png", "source": "http://127.0.0.1/x"})
    )
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: pytest.fail("must not fetch a private host")
    )
    assert search_images.thumbnail_bytes(entry["id"]) is None
    # Corrupt or truncated metadata is a miss, never an error.
    (tmp_path / f"{entry['id']}.json").write_text("{not json")
    assert search_images.thumbnail_bytes(entry["id"]) is None


def test_web_search_tool_with_images_adds_the_field_without_touching_the_base():
    with_images = tools.web_search_tool_with_images()
    props = with_images["function"]["parameters"]["properties"]
    assert "image_queries" in props
    assert "image_queries" not in tools.WEB_SEARCH_TOOL["function"]["parameters"]["properties"]
    assert with_images["function"]["name"] == "web_search"


def test_image_search_caps_the_subject_count(monkeypatch):
    _SubjectDDGS.calls = []
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = _SubjectDDGS))
    tools._image_search([f"Breed {i}" for i in range(9)])
    assert len(_SubjectDDGS.calls) == tools.IMAGE_SEARCH_MAX_QUERIES


def test_status_line_names_the_image_queries():
    from core.inference.tool_loop_controller import status_for_tool

    assert status_for_tool("web_search", {"image_queries": ["a", "b"]}) == "Finding images: a, b"
    assert (
        status_for_tool("web_search", {"query": "dogs", "image_queries": ["a"]})
        == "Searching: dogs (images: a)"
    )
    assert status_for_tool("web_search", {"query": "dogs"}) == "Searching: dogs"


def test_request_tools_swap_in_image_queries_only_with_the_setting(monkeypatch):
    # asyncio.run, not pytest.mark.asyncio: pytest-asyncio is not a backend dependency.
    asyncio.run(_request_tools_swap_case(monkeypatch))


async def _request_tools_swap_case(monkeypatch):
    import routes.inference as inference_routes

    def has_image_queries(tools_list):
        return [
            "image_queries" in t["function"]["parameters"].get("properties", {})
            for t in tools_list
            if t["function"]["name"] == "web_search"
        ]

    payload = SimpleNamespace(
        enabled_tools = ["web_search", "python"],
        rag_scope = None,
        bypass_permissions = False,
    )
    monkeypatch.setattr(inference_routes, "_search_images_enabled", lambda: True)
    selected = await inference_routes._select_request_tools(
        payload, tools_on = True, mcp_allowed = False
    )
    assert [t["function"]["name"] for t in selected] == ["web_search", "python"]
    assert has_image_queries(selected) == [True]

    monkeypatch.setattr(inference_routes, "_search_images_enabled", lambda: False)
    selected = await inference_routes._select_request_tools(
        payload, tools_on = True, mcp_allowed = False
    )
    assert has_image_queries(selected) == [False]

    payload.enabled_tools = ["python"]
    monkeypatch.setattr(inference_routes, "_search_images_enabled", lambda: True)
    selected = await inference_routes._select_request_tools(
        payload, tools_on = True, mcp_allowed = False
    )
    assert [t["function"]["name"] for t in selected] == ["python"]


def test_search_images_kwargs_follow_the_setting_and_the_signature(monkeypatch):
    monkeypatch.setattr(search_images, "search_images_enabled", lambda: True)

    def new_style(
        name,
        arguments,
        search_images = False,
        **kwargs,
    ):
        return ""

    def old_style(
        name,
        arguments,
        cancel_event = None,
        timeout = None,
    ):
        return ""

    assert accepts_kwarg(new_style, "search_images")
    assert not accepts_kwarg(old_style, "search_images")
    assert search_images_kwargs(new_style, "web_search") == {"search_images": True}
    assert search_images_kwargs(new_style, "python") == {}
    assert search_images_kwargs(old_style, "web_search") == {}

    monkeypatch.setattr(search_images, "search_images_enabled", lambda: False)
    assert search_images_kwargs(new_style, "web_search") == {}


def test_search_images_enabled_reads_the_install_setting(monkeypatch):
    import storage.studio_db as db

    monkeypatch.setattr(db, "list_chat_settings", lambda: {"searchImages": True})
    assert search_images.search_images_enabled() is True
    monkeypatch.setattr(db, "list_chat_settings", lambda: {"searchImages": "yes"})
    assert search_images.search_images_enabled() is False
    monkeypatch.setattr(db, "list_chat_settings", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert search_images.search_images_enabled() is False


class _FakeResp:
    def __init__(self, body: bytes, content_type: str | None):
        self._body = body
        self._pos = 0
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, n: int | None = None) -> bytes:
        chunk = self._body[self._pos :] if n is None else self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


class _FakeOpener:
    def __init__(self, resp):
        self._resp = resp

    def open(
        self,
        req,
        timeout = None,
    ):
        return self._resp


def _serve_bytes(
    monkeypatch,
    body: bytes,
    content_type: str = "image/png",
):
    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", "93.184.216.34")
    )
    monkeypatch.setattr(
        tools.urllib.request,
        "build_opener",
        lambda *a, **k: _FakeOpener(_FakeResp(body, content_type)),
    )


def test_fetch_url_raw_binary_mode_returns_bytes_and_caps_size(monkeypatch):
    _serve_bytes(monkeypatch, b"\x89PNG" + b"x" * 100)
    error, body, content_type = tools._fetch_url_raw(
        "https://example.com/a.png", timeout = 5, raw_bytes_max = 1024
    )
    assert error is None
    assert body == b"\x89PNG" + b"x" * 100
    assert content_type == "image/png"

    error, body, _ = tools._fetch_url_raw("https://example.com/a.png", timeout = 5, raw_bytes_max = 50)
    assert error is not None
    assert body == ""


def test_fetch_url_raw_binary_mode_still_blocks_private_hosts():
    error, body, _ = tools._fetch_url_raw("http://127.0.0.1/x.png", timeout = 5, raw_bytes_max = 1024)
    assert error is not None
    assert body == ""


def test_thumbnail_bytes_reencodes_and_caches(monkeypatch, tmp_path):
    entry = search_images.register_images(RAW_IMAGES)[0]
    _serve_bytes(monkeypatch, _png_bytes((800, 600)))

    data = search_images.thumbnail_bytes(entry["id"])
    assert data is not None and data[:3] == b"\xff\xd8\xff"
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "JPEG"
        assert max(im.size) <= search_images.THUMBNAIL_EDGE_PX
    cached = tmp_path / f"{entry['id']}.jpg"
    assert cached.is_file()

    # Served from disk afterwards: no second fetch, even once the registry forgot the id
    # (a restart), so a reopened chat keeps its pictures.
    monkeypatch.setattr(
        tools.urllib.request, "build_opener", lambda *a, **k: pytest.fail("refetched")
    )
    assert search_images.thumbnail_bytes(entry["id"]) == data
    search_images._registry.clear()
    assert search_images.thumbnail_bytes(entry["id"]) == data
    assert search_images.thumbnail_bytes("../" + entry["id"]) is None


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b"<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>", "image/svg+xml"),
        (b"<html>not an image</html>", "image/png"),
        (b"", "image/png"),
    ],
)
def test_thumbnail_bytes_rejects_non_raster_bodies(monkeypatch, body, content_type):
    entry = search_images.register_images(RAW_IMAGES)[0]
    _serve_bytes(monkeypatch, body, content_type)
    assert search_images.thumbnail_bytes(entry["id"]) is None


def test_thumbnail_bytes_rejects_decompression_bombs(monkeypatch):
    entry = search_images.register_images(RAW_IMAGES)[0]
    monkeypatch.setattr(search_images, "MAX_IMAGE_PIXELS", 1000)
    _serve_bytes(monkeypatch, _png_bytes((100, 100)))
    assert search_images.thumbnail_bytes(entry["id"]) is None


def test_thumbnail_bytes_unknown_id():
    assert search_images.thumbnail_bytes("0123456789ab") is None


def test_thumbnail_fetch_keeps_the_website_policy_of_the_search(monkeypatch):
    # register_images checks the policy, but the proxy fetch happens on a later
    # request: without carrying it, every redirect hop off an allowed image host
    # was re-checked against no policy at all.
    policy = {"allowedDomains": ["tse1.mm.bing.net", "akc.org"]}
    entry = search_images.register_images(RAW_IMAGES, policy)[0]
    seen = {}

    def fake_fetch(url, **kwargs):
        seen["url"] = url
        seen["website_policy"] = kwargs.get("website_policy")
        return None, _png_bytes((40, 30)), "image/png"

    monkeypatch.setattr(tools, "_fetch_url_raw", fake_fetch)
    assert search_images.thumbnail_bytes(entry["id"]) is not None
    assert seen["url"] == "https://tse1.mm.bing.net/th?id=golden"
    assert seen["website_policy"] == policy


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_route_serves_registered_thumbnails_only(client, monkeypatch):
    entry = search_images.register_images(RAW_IMAGES)[0]
    _serve_bytes(monkeypatch, _png_bytes())

    ok = client.get(f"/api/inference/search-images/{entry['id']}")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/jpeg"
    assert ok.headers["x-content-type-options"] == "nosniff"
    assert ok.content[:3] == b"\xff\xd8\xff"

    assert client.get("/api/inference/search-images/0123456789ab").status_code == 404
    assert client.get("/api/inference/search-images/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/inference/search-images/not-hex-at-all").status_code == 404


def test_no_history_thumbnails_are_not_browser_cacheable(client, monkeypatch):
    from utils import chat_history_policy

    monkeypatch.setattr(chat_history_policy, "NO_CHAT_HISTORY", True)
    entry = search_images.register_images(RAW_IMAGES)[0]
    _serve_bytes(monkeypatch, _png_bytes())

    response = client.get(f"/api/inference/search-images/{entry['id']}")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"


def test_lookup_route_returns_subject_images_only_when_enabled(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = _SubjectDDGS))
    monkeypatch.setattr(search_images, "search_images_enabled", lambda: False)
    assert (
        client.post("/api/inference/search-images/lookup", json = {"subjects": ["Pug"]}).status_code
        == 403
    )

    monkeypatch.setattr(search_images, "search_images_enabled", lambda: True)
    response = client.post(
        "/api/inference/search-images/lookup", json = {"subjects": ["Pug", "Beagle"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert "Pug:\n- [[img:" in body["text"]
    assert search_images.SEARCH_IMAGES_SENTINEL not in body["text"]
    assert {e["subject"] for e in body["images"]} == {"Pug", "Beagle"}
    assert all("thumbnail" not in e for e in body["images"])
    assert (
        client.post("/api/inference/search-images/lookup", json = {"subjects": []}).status_code == 422
    )
    assert (
        client.post("/api/inference/search-images/lookup", json = {"subjects": ["a"] * 6}).status_code
        == 422
    )


def test_route_has_no_url_parameter(client):
    # A URL-taking proxy would be an open fetch relay; the route must ignore one.
    response = client.get(
        "/api/inference/search-images/0123456789ab", params = {"url": "https://evil.test/x"}
    )
    assert response.status_code == 404


def test_route_requires_auth():
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    anonymous = TestClient(app)
    entry = search_images.register_images(RAW_IMAGES)[0]
    response = anonymous.get(f"/api/inference/search-images/{entry['id']}")
    assert response.status_code in (401, 403)


def test_one_locked_thumbnail_does_not_strand_the_rest_of_the_clear(monkeypatch, tmp_path):
    # A JPEG another process holds open on Windows raises on unlink. Aborting the
    # sweep there left every later file on disk, and thumbnail_bytes reads the
    # cache before the registry, so those ids kept serving after a clear.
    entries = [search_images.register_images(RAW_IMAGES)[0] for _ in range(3)]
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )
    for entry in entries:
        assert search_images.thumbnail_bytes(entry["id"]) is not None
    stuck = tmp_path / f"{entries[1]['id']}.jpg"
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == stuck:
            raise OSError(13, "in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    search_images.clear_cache()
    monkeypatch.undo()

    left = {path.name for path in tmp_path.iterdir()}
    assert left == {stuck.name}, f"only the locked file may survive, found {sorted(left)}"
    assert search_images._registry == {}
    for entry in entries:
        if entry["id"] == entries[1]["id"]:
            continue
        assert search_images.thumbnail_bytes(entry["id"]) is None


def test_a_thumbnail_the_clear_could_not_unlink_is_not_served(monkeypatch, tmp_path):
    # Same Windows situation as above, from the reader's side: the cache-first read
    # and the sidecar read both go around the registry, so the file the clear left
    # behind would have gone on answering for a picture the user had cleared.
    entry = search_images.register_images(RAW_IMAGES)[0]
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )
    assert search_images.thumbnail_bytes(entry["id"]) is not None
    stuck = tmp_path / f"{entry['id']}.jpg"
    real_unlink = Path.unlink
    held = [True]

    def unlink(self, *args, **kwargs):
        if self == stuck and held[0]:
            raise OSError(13, "in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    search_images.clear_cache()
    assert stuck.is_file()
    assert search_images.thumbnail_bytes(entry["id"]) is None
    # Once the other process lets go, the retry reaps it and the id is ordinary again.
    held[0] = False
    assert search_images.thumbnail_bytes(entry["id"]) is None
    assert not stuck.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_sidecar_the_clear_could_not_unlink_cannot_resurrect_an_id(monkeypatch, tmp_path):
    # The sidecar carries the thumbnail and source URLs, so a surviving one is the
    # half of the pair that matters: it is what lets a cleared id be fetched again.
    entry = search_images.register_images(RAW_IMAGES)[0]
    stuck = tmp_path / f"{entry['id']}.json"
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == stuck:
            raise OSError(13, "in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    search_images.clear_cache()
    assert stuck.is_file()
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )
    assert search_images.thumbnail_bytes(entry["id"]) is None
    assert not (tmp_path / f"{entry['id']}.jpg").exists()


def test_the_route_rejects_an_id_with_a_trailing_newline():
    # `$` matches before a trailing newline, so `.match` alone let one through to
    # the store, which refused it with fullmatch. Refuse it at the door instead.
    assert search_images.IMAGE_ID_RE.fullmatch("0123456789ab\n") is None
    assert (
        search_images.is_image_entry(
            {"id": "0123456789ab\n", "title": "t", "domain": "d", "source": "https://example.com/"}
        )
        is False
    )
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)
    response = client.get("/api/inference/search-images/0123456789ab%0A")
    assert response.status_code == 404


def _fake_ddgs_with_text(monkeypatch):
    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
        ):
            return [{"title": "AKC", "href": "https://akc.org/x", "body": "Breeds"}]

        def images(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            return RAW_IMAGES

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS = FakeDDGS))


def test_a_failing_subject_lookup_does_not_discard_the_text_results(monkeypatch):
    # The named-subject branch sits inside _web_search's own `except`, so a raise
    # there came back as "Search failed: ..." with the results thrown away. The two
    # sibling branches already swallow; this one has to as well.
    _fake_ddgs_with_text(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(tools, "_image_search", boom)
    result = tools._web_search(
        "dog breeds", include_images = True, image_queries = ["Golden Retriever"]
    )
    assert "Title: AKC" in result
    assert "Search failed" not in result
    assert search_images.SEARCH_IMAGES_SENTINEL not in result


def test_a_failing_image_only_lookup_still_returns_a_string(monkeypatch):
    # This branch is evaluated before the try, so it carries its own guard:
    # execute_tool answers with a string for every input the model can send.
    _fake_ddgs_with_text(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(tools, "_image_search", boom)
    result = tools._web_search("", include_images = True, image_queries = ["Golden Retriever"])
    assert result == "No images found for: Golden Retriever"


def test_an_unreadable_cache_dir_snapshots_as_clear_everything(monkeypatch, tmp_path):
    """The OSError fallback has to be the full-clear sentinel, not an empty selection.

    `registered_image_ids` bounds the reap that follows it. When it cannot enumerate
    the cache its snapshot is incomplete, so the only safe answer is None -- which
    `clear_cache` reads as "clear everything", the behaviour a clear had before the
    snapshot existed. Returning `set()` instead reads as a selective reap of nothing:
    "Clear all chats" would leave the registry populated and every thumbnail -- which
    says what the user searched for -- still fetchable.
    """

    class _UnreadableDir(type(tmp_path)):
        def glob(self, _pattern):
            raise OSError("permission denied")

    entries = search_images.register_images(RAW_IMAGES)
    assert entries, "need a registered image for the reap to have something to miss"
    monkeypatch.setattr(search_images, "_cache_dir", lambda: _UnreadableDir(tmp_path))

    snapshot = search_images.registered_image_ids()
    assert snapshot is None, "an incomplete snapshot must not bound the reap"

    search_images.clear_cache(snapshot)
    assert search_images._registry == {}, "a clear that could not snapshot must still clear"
    assert search_images.lookup_image(entries[0]["id"]) is None


def test_a_selective_clear_does_not_abort_a_fetch_for_an_image_it_spared(monkeypatch, tmp_path):
    """The whole point of the snapshot is that a spared image keeps working.

    `clear_cache` bumps the generation for a selective reap too, and the in-flight check used
    to compare that bare number: a fetch already running for an id the clear went out of its
    way to spare therefore aborted, thumbnail_bytes answered None and the endpoint 404ed. The
    frontend does not recover from that -- SearchImageThumb sets `failed`, renders nothing,
    and its effect depends only on (id, nearViewport), so nothing re-runs it. The card is gone
    until the component remounts.
    """
    spared = search_images.register_images(RAW_IMAGES)[0]
    # Only one of RAW_IMAGES survives the policy filter, and this needs a second id purely
    # as something for the clear to reap. Straight into the registry: what matters is that
    # it is NOT the id being fetched.
    doomed_id = "0123456789ab"
    search_images._registry[doomed_id] = dict(search_images._registry[spared["id"]])
    real_lookup = search_images._lookup_locked

    def clearing_lookup(image_id):
        found = real_lookup(image_id)
        # A selective clear landing mid-fetch, reaping the OTHER image. Inline because the
        # caller already holds _registry_lock; the bookkeeping matches clear_cache's.
        search_images._registry.pop(doomed_id, None)
        search_images._cache_generation += 1
        search_images._reaped_at[doomed_id] = search_images._cache_generation
        return found

    monkeypatch.setattr(search_images, "_lookup_locked", clearing_lookup)
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )

    assert (
        search_images.thumbnail_bytes(spared["id"]) is not None
    ), "a clear that deliberately kept this image must not take its in-flight fetch down"
    assert list(tmp_path.glob(f"{spared['id']}.jpg")), "and its bytes belong on disk"


def test_a_selective_clear_still_aborts_the_fetch_for_an_image_it_reaped(monkeypatch, tmp_path):
    """The other half. Publishing here would write back a thumbnail the clear just deleted,
    and the cache-first read would go on serving it."""
    doomed = search_images.register_images(RAW_IMAGES)[0]
    real_lookup = search_images._lookup_locked

    def clearing_lookup(image_id):
        found = real_lookup(image_id)
        search_images._registry.pop(doomed["id"], None)
        search_images._cache_generation += 1
        search_images._reaped_at[doomed["id"]] = search_images._cache_generation
        for stale in tmp_path.glob("*"):
            stale.unlink()
        return found

    monkeypatch.setattr(search_images, "_lookup_locked", clearing_lookup)
    monkeypatch.setattr(
        tools, "_fetch_url_raw", lambda url, **kw: (None, _png_bytes((40, 30)), "image/png")
    )

    assert search_images.thumbnail_bytes(doomed["id"]) is None
    assert list(tmp_path.glob("*.jpg")) == []


def test_an_overflowing_reap_record_drops_the_oldest_not_everything(monkeypatch, tmp_path):
    """The per-id record is bounded, and running out of room must not abort live fetches.

    Clearing it and promoting the clear to a full-clear marker was the first attempt. That
    aborts every fetch in flight, including ones for images the clear deliberately spared,
    and an aborted fetch is not a cheap retry: thumbnail_bytes answers None, the endpoint
    404s, and useSearchThumbnail records a permanent failure for that id.

    Dropping the OLDEST records and raising a floor keeps every fetch that started at or
    after the floor exactly answerable. The assertion is taken across the clear that
    actually overflows, which is the only moment the two strategies differ.
    """
    monkeypatch.setattr(search_images, "_REAPED_AT_MAX", 4)
    monkeypatch.setattr(search_images, "_reaped_at", {})
    monkeypatch.setattr(search_images, "_reaped_floor_generation", 0)
    monkeypatch.setattr(search_images, "_full_clear_generation", 0)

    # Two clears that fit, then one that does not.
    search_images.clear_cache({"000000000000", "000000000001"})
    search_images.clear_cache({"000001000000", "000001000001"})
    in_flight_generation = search_images.cache_generation()
    search_images.clear_cache({"000002000000", "000002000001"})

    assert search_images._reaped_at, "records must survive the overflow; clearing them was the bug"
    assert (
        search_images._reaped_at.get("000002000000") == search_images.cache_generation()
    ), "the reap that overflowed is exactly the one that must still be remembered"

    with search_images._registry_lock:
        # A fetch already running for an image no clear ever named. Promoting the overflow
        # to a full clear took this down with everything else.
        assert search_images._reaped_since_locked("ffffffffffff", in_flight_generation) is False
        # And the ids that clear really did take are still known to be reaped.
        assert search_images._reaped_since_locked("000002000000", in_flight_generation) is True


def test_the_overflow_floor_still_refuses_a_fetch_older_than_every_record(monkeypatch):
    """The one case the floor gives up on, kept honest: older than anything still held."""
    monkeypatch.setattr(search_images, "_REAPED_AT_MAX", 4)
    monkeypatch.setattr(search_images, "_reaped_at", {})
    monkeypatch.setattr(search_images, "_reaped_floor_generation", 0)

    for round_index in range(4):
        search_images.clear_cache({f"{round_index:06d}{index:06d}" for index in range(2)})

    floor = search_images._reaped_floor_generation
    assert floor > 0, "the cap has to have forced a floor for this to mean anything"
    with search_images._registry_lock:
        assert search_images._reaped_since_locked("ffffffffffff", floor - 1) is True
        assert search_images._reaped_since_locked("ffffffffffff", floor) is False


def test_a_full_clear_still_aborts_every_fetch_including_unknown_ids():
    """The blunt signal is still right for a clear-everything: nothing survives it."""
    before = search_images.cache_generation()
    search_images.clear_cache()
    with search_images._registry_lock:
        assert search_images._reaped_since_locked("ffffffffffff", before) is True


def test_a_lookup_already_running_when_a_clear_starts_publishes_nothing(monkeypatch, tmp_path):
    """Bounding the reap to a snapshot spares whatever registers after it. That is right for
    a chat created since the clear, and wrong for a lookup the clear is deleting the answer
    of -- `/search-images/lookup` carries no thread, so no cancellation reaches it, and it
    samples the cache generation on entry.

    Without a fence at the clear boundary its images register into the window between the
    snapshot and the reap, the selective reap spares them, and Clear all leaves their
    sidecars on disk saying what was searched for.
    """
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path)

    # What _image_search does on entry, before it goes to the network.
    sampled = search_images.cache_generation()

    # The clear reaches its boundary while that lookup is still out.
    snapshot = search_images.snapshot_and_fence_registrations()
    assert snapshot is not None

    # The lookup comes back and tries to publish.
    published = search_images.register_images(RAW_IMAGES, expected_generation = sampled)

    assert (
        published == []
    ), "an answer this clear is deleting must not get its images registered behind it"
    assert list(tmp_path.glob("*.json")) == [], "and no sidecar may be written either"


def test_a_lookup_that_starts_after_the_clear_boundary_still_registers(monkeypatch, tmp_path):
    """The complement. A chat created after the clear is exactly what the snapshot spares,
    so its lookups must go through untouched."""
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path)

    search_images.snapshot_and_fence_registrations()
    sampled = search_images.cache_generation()
    published = search_images.register_images(RAW_IMAGES, expected_generation = sampled)

    assert published, "a lookup that started after the boundary belongs to a surviving chat"


def test_the_fence_does_not_abort_an_in_flight_fetch(monkeypatch, tmp_path):
    """The fence moves the generation, and fetch invalidation is per id now, so a fetch
    already running for an image the clear will spare must be unaffected. Reading the bare
    generation here would undo the per-id work."""
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path)
    started_at = search_images.cache_generation()

    search_images.snapshot_and_fence_registrations()

    with search_images._registry_lock:
        assert search_images._reaped_since_locked("ffffffffffff", started_at) is False
