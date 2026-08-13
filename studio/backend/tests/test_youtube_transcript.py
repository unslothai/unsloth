# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``core.youtube_transcript``, which backs the composer's transcript offer.

Covers the three parts that decide what the model ends up reading: which URLs count
as YouTube videos, which caption track is picked when several languages exist, and
how json3 cues flatten into text. All offline; the network hops are not exercised."""

import pytest

from core.youtube_transcript import (
    _default_track_index,
    _flatten_events,
    _select_track,
    extract_video_id,
)

VIDEO_ID = "dQw4w9WgXcQ"


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
