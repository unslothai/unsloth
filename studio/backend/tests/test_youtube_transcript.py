# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``core.youtube_transcript``, which backs the composer's transcript offer.

Covers the parts that decide what the model ends up reading: which URLs count as
YouTube videos, which caption track is picked when several languages exist, how json3
cues flatten into text, and the bounds on the caption fetch itself. All offline; the
caption hop runs against an httpx MockTransport rather than YouTube."""

import asyncio
import json

import httpx
import pytest

from core.youtube_transcript import (
    _MAX_CAPTION_BYTES,
    _MAX_TRANSCRIPT_CHARS,
    TranscriptUnavailable,
    _caption_url,
    _default_track_index,
    _fetch_track_text,
    _flatten_events,
    _select_track,
    _truncate_transcript,
    extract_video_id,
)

VIDEO_ID = "dQw4w9WgXcQ"
CAPTION_URL = f"https://www.youtube.com/api/timedtext?v={VIDEO_ID}&lang=en"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&t=42s",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=42",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
    ],
)
def test_extract_video_id_accepts_youtube_link_shapes(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        # A path segment must never stand in for the host, or the route would fetch
        # a player response for a video id an arbitrary site chose.
        f"https://evil.com/youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
        f"https://notyoutube.com/watch?v={VIDEO_ID}",
        f"javascript:alert(1)//youtube.com/watch?v={VIDEO_ID}",
        f"file:///etc/passwd#{VIDEO_ID}",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/@somechannel",
        "not a url",
        "",
    ],
)
def test_extract_video_id_rejects_non_video_urls(url):
    assert extract_video_id(url) is None


def _track(code, kind = None):
    return {
        "baseUrl": f"https://www.youtube.com/api/timedtext?lang={code}",
        "languageCode": code,
        "kind": kind,
    }


def test_select_track_prefers_requested_language_and_human_captions():
    tracks = [_track("ar"), _track("en", "asr"), _track("en"), _track("fr")]
    assert _select_track(tracks, {}, ["en"])["kind"] is None
    assert _select_track(tracks, {}, ["fr"])["languageCode"] == "fr"
    # A regional tag falls back to its base language rather than missing the track.
    assert _select_track([_track("en-GB")], {}, ["en-US"])["languageCode"] == "en-GB"
    # Only an auto-generated track in that language: take it rather than another language.
    assert _select_track([_track("ar"), _track("de", "asr")], {}, ["de"])["languageCode"] == "de"


def test_select_track_prefers_an_exact_locale_over_an_earlier_regional_sibling():
    # pt-PT is listed first and shares the base language, so a positional scan would
    # hand a Brazilian viewer the European Portuguese transcript.
    tracks = [_track("pt-PT"), _track("pt-BR")]
    assert _select_track(tracks, {}, ["pt-BR"])["languageCode"] == "pt-BR"
    assert _select_track(tracks, {}, ["pt-PT"])["languageCode"] == "pt-PT"
    # No exact match for the region: the base language still answers.
    assert _select_track(tracks, {}, ["pt-AO"])["languageCode"] == "pt-PT"
    # A human base-language track outranks an auto-generated exact match.
    assert (
        _select_track([_track("pt"), _track("pt-BR", "asr")], {}, ["pt-BR"])["languageCode"] == "pt"
    )


def test_select_track_falls_back_to_the_default_audio_track_language():
    # Multi-language videos list captions alphabetically, so index 0 is usually a
    # translation of a language nobody in the video is speaking.
    tracks = [_track("ar"), _track("cs"), _track("en")]
    tracklist = {
        "audioTracks": [{"defaultCaptionTrackIndex": 2}],
        "defaultAudioTrackIndex": 0,
    }
    assert _select_track(tracks, tracklist, [])["languageCode"] == "en"
    assert _select_track(tracks, tracklist, ["  "])["languageCode"] == "en"


@pytest.mark.parametrize(
    "tracklist",
    [
        {},
        {"audioTracks": [], "defaultAudioTrackIndex": 0},
        {"audioTracks": [{"defaultCaptionTrackIndex": 9}], "defaultAudioTrackIndex": 0},
        {"audioTracks": [{}], "defaultAudioTrackIndex": "0"},
    ],
)
def test_default_track_index_is_zero_when_the_pairing_is_missing(tracklist):
    assert _default_track_index([_track("en"), _track("fr")], tracklist) == 0


def test_flatten_events_drops_rolling_window_padding():
    events = [
        {"tStartMs": 0, "dDurationMs": 1119069, "id": 1},
        {"segs": [{"utf8": "[Music]"}]},
        # ASR tracks separate cues with an aAppend newline that is not transcript text.
        {"aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"segs": [{"utf8": "This"}, {"utf8": " is"}, {"utf8": " a\nthree."}]},
        {"segs": [{"utf8": "   "}]},
        "not an event",
    ]
    assert _flatten_events(events) == "[Music]\nThis is a three."


def test_flatten_events_handles_an_empty_track():
    assert _flatten_events([]) == ""


def test_caption_url_keeps_blank_valued_parameters():
    # parse_qs drops blank values by default, which would silently strip any key=
    # parameter YouTube signs the timedtext URL with.
    url = _caption_url(
        "https://www.youtube.com/api/timedtext"
        f"?v={VIDEO_ID}&opi=&caps=&exp=xbt&signature=XYZ&lang=en"
    )
    query = url.split("?", 1)[1].split("&")
    assert "opi=" in query
    assert "caps=" in query
    assert "signature=XYZ" in query
    assert "fmt=json3" in query


@pytest.mark.parametrize(
    "base_url",
    [
        f"http://www.youtube.com/api/timedtext?v={VIDEO_ID}",
        f"https://evil.com/api/timedtext?v={VIDEO_ID}",
        f"https://youtube.com.evil.com/api/timedtext?v={VIDEO_ID}",
        f"https://169.254.169.254/api/timedtext?v={VIDEO_ID}",
    ],
)
def test_caption_url_rejects_hosts_outside_the_allowlist(base_url):
    with pytest.raises(TranscriptUnavailable):
        _caption_url(base_url)


def test_truncate_transcript_caps_a_long_video_on_a_line_boundary():
    short = "one\ntwo\nthree"
    assert _truncate_transcript(short) == (short, False)

    text = "\n".join("a line of caption text" * 4 for _ in range(20_000))
    capped, truncated = _truncate_transcript(text)
    assert truncated is True
    assert len(capped) <= _MAX_TRANSCRIPT_CHARS
    assert text.startswith(capped)
    # A cut mid-sentence would hand the model a mangled final line.
    assert capped.endswith("a line of caption text")


def _fetch_captions(handler):
    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport = transport, follow_redirects = True) as client:
            return await _fetch_track_text(client, CAPTION_URL)

    return asyncio.run(run())


def test_fetch_track_text_refuses_a_redirect_off_the_allowlist():
    # The client follows redirects for the player hop, so the caption hop has to
    # re-check the host itself or a 302 would stream a loopback body back.
    def handler(request):
        return httpx.Response(302, headers = {"location": "http://169.254.169.254/latest/meta-data"})

    with pytest.raises(TranscriptUnavailable, match = "unexpected host"):
        _fetch_captions(handler)


def test_fetch_track_text_follows_a_redirect_that_stays_on_youtube():
    events = {"events": [{"segs": [{"utf8": "hello"}]}]}

    def handler(request):
        if request.url.path == "/api/timedtext":
            return httpx.Response(302, headers = {"location": "/api/timedtext/v2?fmt=json3"})
        return httpx.Response(200, json = events)

    assert _fetch_captions(handler) == "hello"


def test_fetch_track_text_stops_a_redirect_loop():
    def handler(request):
        return httpx.Response(302, headers = {"location": "https://www.youtube.com/api/timedtext"})

    with pytest.raises(TranscriptUnavailable, match = "too many times"):
        _fetch_captions(handler)


def test_fetch_track_text_caps_an_oversized_caption_body():
    body = json.dumps({"events": [{"segs": [{"utf8": "x" * (_MAX_CAPTION_BYTES + 1024)}]}]})

    def handler(request):
        return httpx.Response(200, content = body)

    with pytest.raises(TranscriptUnavailable, match = "too large"):
        _fetch_captions(handler)
