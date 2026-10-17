# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi.responses import JSONResponse

from .mcp_harness import REMOTE, bodies, queries, run_tool

IMAGES = "/api/inference/images/gallery"
VIDEOS = "/api/inference/video/gallery"
AUDIO = "/api/inference/audio/gallery"
TRANSCRIPTS = "/api/inference/audio/transcripts"
VOICES = "/v1/audio/voices"
BASE = "http://192.168.1.20:8888"
WHEN = "2026-10-17T00:00:00Z"


def _minted(base, media_id):
    return {
        ("GET", f"{base}/{media_id}/signed-url"): {
            "url": f"{base}/{media_id}/file-signed?token=t.1"
        }
    }


def _voice(n):
    return {
        "id": f"voice-{n}",
        "name": f"Narrator {n}",
        "transcript": "Hello there.",
        "language": "en",
        "duration_s": 3.0,
        "sample_rate": 24000,
        "created_at": WHEN,
        "url": f"/api/inference/audio/voices/voice-{n}/file",
    }


# Undeclared route fields (loras, gguf_filename, settings) can hold host paths and are never copied.
PAYLOADS = {
    ("GET", IMAGES): {
        "images": [
            {
                "id": "img-1",
                "url": f"{IMAGES}/img-1/file",
                "prompt": "a red fox",
                "model": "flux",
                "width": 512,
                "height": 512,
                "seed": 7,
                "created_at": 1.5,
                "loras": ["/srv/lora:1"],
                "gguf_filename": "/srv/flux.gguf",
            }
        ],
        "has_more": True,
    },
    ("GET", VIDEOS): {
        "videos": [
            {"id": "video-1", "prompt": "waves", "fps": 24, "has_audio": True, "created_at": WHEN}
        ],
        "has_more": False,
    },
    ("GET", AUDIO): {
        "audio": [
            {
                "id": "clip-1",
                "prompt": "Hi",
                "model": "kokoro",
                "workflow": "clone",
                "voice_id": "voice-1",
                "duration_s": 1.0,
                "created_at": WHEN,
                "settings": {"options": {"speaker": "/srv/x.wav"}},
            }
        ],
        "has_more": True,
        "next_before_mtime": 1760000000.5,
        "next_before_id": "clip-1",
        "next_before_pin": None,
    },
    ("GET", TRANSCRIPTS): {
        "transcripts": [
            {"id": "tr-1", "title": "meeting.wav", "text": "Hello.", "segment_count": 1}
        ],
        "next_cursor": f"{WHEN}|tr-1",
    },
    ("GET", VOICES): {"voices": [_voice(1), _voice(2), _voice(3)]},
    **_minted(IMAGES, "img-1"),
    **_minted(VIDEOS, "video-1"),
    **_minted("/v1/audio/gallery", "clip-1"),
}

# (kind, the list key, its first row, next_cursor).
KINDS = [
    (
        "image",
        "images",
        {
            "id": "img-1",
            "url": f"{BASE}{IMAGES}/img-1/file",
            "download_url": f"{BASE}{IMAGES}/img-1/file-signed?token=t.1",
            "prompt": "a red fox",
            "model": "flux",
            "workflow": None,
            "width": 512,
            "height": 512,
            "seed": 7,
            "created_at": 1.5,
            "pinned": False,
        },
        "image:1",
    ),
    (
        "video",
        "videos",
        {"download_url": f"{BASE}{VIDEOS}/video-1/file-signed?token=t.1", "fps": 24.0},
        None,
    ),
    (
        "audio",
        "audio",
        {"url": f"{BASE}/v1/audio/gallery/clip-1/file", "workflow": "clone", "voice_id": "voice-1"},
        "audio:|1760000000.5|clip-1",
    ),
    (
        "transcript",
        "transcripts",
        {"id": "tr-1", "title": "meeting.wav", "text": "Hello."},
        f"transcript:{WHEN}|tr-1",
    ),
    ("voice", "voices", {"name": "Narrator 1", "url": f"{BASE}{VOICES}/voice-1/file"}, "voice:2"),
]


@pytest.mark.parametrize("kind,key,first,after", KINDS)
def test_each_kind_lists_typed_rows_and_a_cursor(monkeypatch, kind, key, first, after):
    args = {"kind": kind, "limit": 2}
    result, _studio = run_tool(monkeypatch, PAYLOADS, "list_media", args, **REMOTE)
    out = result["structuredContent"]
    assert out["kind"] == kind and out["next_cursor"] == after
    assert first.items() <= out[key][0].items()
    assert [k for k, v in out.items() if isinstance(v, list)] == [key]


# (cursor, the route it reaches, the query sent, the next cursor).
PAGES = [
    ("image:20", IMAGES, {"limit": "2", "offset": "20"}, "image:21"),
    ("video:4", VIDEOS, {"limit": "2", "offset": "4"}, None),
    (
        "audio:3.0|1760000001.25|clip-9",
        AUDIO,
        {"limit": "2", "before_pin": "3.0", "before_mtime": "1760000001.25", "before_id": "clip-9"},
        "audio:|1760000000.5|clip-1",
    ),
    (
        "audio:|1760000001.25|clip-9",
        AUDIO,
        {"limit": "2", "before_mtime": "1760000001.25", "before_id": "clip-9"},
        "audio:|1760000000.5|clip-1",
    ),
    (
        f"transcript:{WHEN}|tr-0",
        TRANSCRIPTS,
        {"limit": "2", "before": f"{WHEN}|tr-0"},
        f"transcript:{WHEN}|tr-1",
    ),
]


@pytest.mark.parametrize("cursor,path,query,after", PAGES)
def test_a_cursor_maps_to_the_routes_own_paging(monkeypatch, cursor, path, query, after):
    kind = cursor.partition(":")[0]
    args = {"kind": kind, "limit": 2, "cursor": cursor}
    result, studio = run_tool(monkeypatch, PAYLOADS, "list_media", args)
    assert queries(studio, path) == [query]
    assert result["structuredContent"]["next_cursor"] == after


def test_voices_are_paged_locally(monkeypatch):
    args = {"kind": "voice", "limit": 2, "cursor": "voice:2"}
    result, _studio = run_tool(monkeypatch, PAYLOADS, "list_media", args)
    out = result["structuredContent"]
    assert [v["id"] for v in out["voices"]] == ["voice-3"] and out["next_cursor"] is None


def test_a_failed_mint_leaves_download_url_empty(monkeypatch):
    routes = {**PAYLOADS, ("GET", f"{IMAGES}/img-1/signed-url"): JSONResponse({}, status_code = 403)}
    result, _studio = run_tool(monkeypatch, routes, "list_media", {"kind": "image"})
    assert result["structuredContent"]["images"][0]["download_url"] is None


@pytest.mark.parametrize(
    "args,message",
    [
        ({"kind": "image", "cursor": "audio:|1.0|clip-1"}, "not from list_media(kind='image')"),
        ({"kind": "audio", "cursor": "image:5"}, "not from list_media(kind='audio')"),
        ({"kind": "image", "cursor": "image:-1"}, "malformed"),
        ({"kind": "audio", "cursor": "audio:1.0|clip-1"}, "malformed"),
        ({"kind": "audio", "cursor": "audio:|1.0|../x"}, "malformed"),
        ({"kind": "transcript", "cursor": "transcript:"}, "malformed"),
        ({"kind": "image", "limit": 0}, None),
        ({"kind": "image", "limit": 51}, None),
        ({"kind": "album"}, None),
    ],
)
def test_a_bad_list_media_call_is_refused_without_a_call(monkeypatch, args, message):
    result, studio = run_tool(monkeypatch, {}, "list_media", args, fallback = {})
    assert result["isError"] is True
    assert message is None or message in result["content"][0]["text"]
    assert studio.state.calls == []


# The voices tool, from here to the end of the file.

SAVED = {("POST", VOICES): JSONResponse(_voice(4), status_code = 201)}
VOICE_PAYLOADS = {
    **PAYLOADS,
    **SAVED,
    ("PATCH", f"{VOICES}/voice-1"): {**_voice(1), "name": "Host"},
    ("DELETE", f"{VOICES}/voice-1"): {"removed": True},
}
# (args, the call made as (method, path), its JSON body or None, a check on the result).
LIFECYCLE = [
    ({"action": "list"}, ("GET", VOICES), None, lambda out: len(out["voices"]) == 3),
    (
        {"action": "save", "source": {"clip_id": "clip-1"}, "name": "Narrator 4", "language": "en"},
        ("POST", VOICES),
        {"source": {"clip_id": "clip-1"}, "name": "Narrator 4", "language": "en"},
        lambda out: out["voice"]["id"] == "voice-4",
    ),
    (
        {"action": "rename", "voice_id": "voice-1", "name": "Host"},
        ("PATCH", f"{VOICES}/voice-1"),
        {"name": "Host"},
        lambda out: out["voice"]["name"] == "Host",
    ),
    (
        {"action": "delete", "voice_id": "voice-1"},
        ("DELETE", f"{VOICES}/voice-1"),
        None,
        lambda out: out["removed"] is True,
    ),
]


@pytest.mark.parametrize("args,call,body,check", LIFECYCLE)
def test_voices_list_save_rename_and_delete(monkeypatch, args, call, body, check):
    result, studio = run_tool(monkeypatch, VOICE_PAYLOADS, "voices", args)
    assert result["isError"] is False, result
    assert [c[:2] for c in studio.state.calls] == [call]
    assert body is None or bodies(studio, call[1]) == [body]
    assert check(result["structuredContent"])


def test_a_voice_saved_from_a_video_uses_its_sound(monkeypatch):
    routes = {**SAVED, ("POST", f"{VIDEOS}/video-1/audio-input"): {"id": "in-7"}}
    args = {"action": "save", "source": {"video_id": "video-1", "end_s": 4}, "name": "Narrator 4"}
    result, studio = run_tool(monkeypatch, routes, "voices", args)
    assert result["isError"] is False
    assert bodies(studio, f"{VIDEOS}/video-1/audio-input") == [{"end_s": 4.0}]
    assert bodies(studio, VOICES)[0]["source"] == {"input_id": "in-7"}


@pytest.mark.parametrize(
    "args,message",
    [
        (
            {"action": "save", "source": {"voice_id": "voice-1"}, "name": "x"},
            "not from another saved voice",
        ),
        ({"action": "save", "source": {"clip_id": "clip-1"}}, "save needs source and name"),
        ({"action": "rename", "name": "x"}, "rename needs voice_id"),
        ({"action": "delete", "voice_id": "../voice-1"}, "delete needs voice_id"),
        ({"action": "rename", "voice_id": "voice-1"}, "rename needs name, transcript or language"),
        ({"action": "save", "source": {"clip_id": "c"}, "name": "x" * 81}, None),
    ],
)
def test_a_bad_voices_call_is_refused_without_a_call(monkeypatch, args, message):
    result, studio = run_tool(monkeypatch, {}, "voices", args, fallback = {})
    assert result["isError"] is True
    assert message is None or message in result["content"][0]["text"]
    assert studio.state.calls == []


def test_a_missing_voice_points_at_the_listing(monkeypatch):
    missing = {
        ("DELETE", f"{VOICES}/voice-9"): JSONResponse(
            {"detail": "Voice not found."}, status_code = 404
        )
    }
    result, _studio = run_tool(
        monkeypatch, missing, "voices", {"action": "delete", "voice_id": "voice-9"}
    )
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert text.startswith("Voice not found.") and "list_media(kind='voice')" in text
