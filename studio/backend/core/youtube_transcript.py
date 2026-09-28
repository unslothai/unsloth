# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read a YouTube video's captions as plain text, with no third-party client.

Two hops, both against youtube.com. ``POST /youtubei/v1/player`` with the ANDROID
InnerTube client lists the caption tracks and the video metadata, then the chosen
track's ``baseUrl`` is downloaded as ``fmt=json3`` and flattened.

The ANDROID client matters. Caption URLs taken from the watch page's
``ytInitialPlayerResponse`` belong to the WEB client, and YouTube now answers those
with 200 and an empty body unless the request carries a proof-of-origin token, which
only its BotGuard JS can mint. The ANDROID client's URLs still resolve unsigned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import SplitResult, parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import httpx

_CLIENT_VERSION = "20.10.38"
_CLIENT_NAME_ID = "3"
_USER_AGENT = f"com.google.android.youtube/{_CLIENT_VERSION} (Linux; U; Android 11) gzip"
_PLAYER_URL = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
# www. is stripped before the lookup, so only the bare forms are listed.
_WATCH_HOSTS = frozenset(
    {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}
)
_SHORT_HOSTS = frozenset({"youtu.be"})
_ID_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
_CAPTION_HOSTS = frozenset({"youtube.com", "www.youtube.com"})

_TIMEOUT = httpx.Timeout(20.0)
# Captions are text; a 4 MB track is already an outlier for a very long video.
_MAX_CAPTION_BYTES = 4 * 1024 * 1024
_MAX_PLAYER_BYTES = 4 * 1024 * 1024
# Roughly 25k tokens: a three hour video's captions would swallow the model's context window on their own.
_MAX_TRANSCRIPT_CHARS = 100_000
# Timedtext normally answers 200, but a hop is re-validated rather than refused.
_MAX_CAPTION_REDIRECTS = 3


class TranscriptUnavailable(Exception):
    """YouTube answered, but the video has no caption track we can read."""


@dataclass(frozen = True)
class Transcript:
    video_id: str
    title: str
    author: str
    length_seconds: int
    language: str
    language_code: str
    is_generated: bool
    text: str
    truncated: bool


def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-character video id in a YouTube URL, or None if it is not one.

    Accepts ``/watch?v=``, ``youtu.be/<id>``, ``/shorts/``, ``/embed/``, ``/live/``
    and ``/v/`` on the youtube.com, youtu.be and youtube-nocookie.com hosts.
    """
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    candidate = ""
    if host in _SHORT_HOSTS:
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
    elif host not in _WATCH_HOSTS:
        return None
    elif parsed.path.rstrip("/") == "/watch":
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    else:
        for prefix in _ID_PATH_PREFIXES:
            if parsed.path.startswith(prefix):
                candidate = parsed.path[len(prefix) :].split("/", 1)[0]
                break
    return candidate if _VIDEO_ID_RE.fullmatch(candidate) else None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


async def fetch_transcript(video_id: str, languages: Sequence[str] = ()) -> Transcript:
    """Download the captions for ``video_id``, preferring ``languages`` in order.

    Within a language a human-written track wins over an auto-generated one. With no
    match the track YouTube pairs with the video's default audio track is used.
    """
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise TranscriptUnavailable("That is not a YouTube video link.")

    async with httpx.AsyncClient(timeout = _TIMEOUT, follow_redirects = True) as client:
        player = await _fetch_player(client, video_id)
        status = (player.get("playabilityStatus") or {}).get("status")
        if status not in (None, "OK"):
            raise TranscriptUnavailable(
                (player.get("playabilityStatus") or {}).get("reason")
                or "YouTube will not play this video."
            )

        tracklist = (player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
        tracks = [t for t in (tracklist.get("captionTracks") or []) if t.get("baseUrl")]
        if not tracks:
            raise TranscriptUnavailable("This video has no captions.")

        track = _select_track(tracks, tracklist, languages)
        text = await _fetch_track_text(client, str(track["baseUrl"]))

    if not text:
        raise TranscriptUnavailable("This video's captions are empty.")
    text, truncated = _truncate_transcript(text)

    details = player.get("videoDetails") or {}
    return Transcript(
        video_id = video_id,
        title = str(details.get("title") or ""),
        author = str(details.get("author") or ""),
        length_seconds = _as_int(details.get("lengthSeconds")),
        language = _track_label(track),
        language_code = str(track.get("languageCode") or ""),
        is_generated = track.get("kind") == "asr",
        text = text,
        truncated = truncated,
    )


def _truncate_transcript(text: str) -> tuple[str, bool]:
    """Bound the transcript so a long video cannot swallow the model's context."""
    if len(text) <= _MAX_TRANSCRIPT_CHARS:
        return text, False
    return text[:_MAX_TRANSCRIPT_CHARS].rsplit("\n", 1)[0].rstrip(), True


async def _fetch_player(client: httpx.AsyncClient, video_id: str) -> dict[str, Any]:
    async with client.stream(
        "POST",
        _PLAYER_URL,
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "X-YouTube-Client-Name": _CLIENT_NAME_ID,
            "X-YouTube-Client-Version": _CLIENT_VERSION,
        },
        json = {
            "context": {
                "client": {
                    "clientName": "ANDROID",
                    "clientVersion": _CLIENT_VERSION,
                    "androidSdkVersion": 30,
                    "osName": "Android",
                    "osVersion": "11",
                    "hl": "en",
                    "gl": "US",
                },
            },
            "videoId": video_id,
            "contentCheckOk": True,
            "racyCheckOk": True,
        },
    ) as response:
        response.raise_for_status()
        body = await _read_capped(
            response, _MAX_PLAYER_BYTES, "YouTube returned an unexpectedly large response."
        )
    try:
        player = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise TranscriptUnavailable("YouTube returned an unreadable response.") from error
    if not isinstance(player, dict):
        raise TranscriptUnavailable("YouTube returned an unreadable response.")
    return player


def _select_track(
    tracks: list[dict[str, Any]], tracklist: dict[str, Any], languages: Sequence[str]
) -> dict[str, Any]:
    for language in languages:
        wanted = str(language).strip().lower()
        if not wanted:
            continue
        base = wanted.split("-")[0]
        for want_generated in (False, True):
            candidates = [t for t in tracks if (t.get("kind") == "asr") is want_generated]
            # exact locale before the base-language fallback: a pt-BR request must not take a pt-PT track just
            # because it is listed first
            for matches_wanted in (
                lambda code: code == wanted,
                lambda code: code.split("-")[0] == base,
            ):
                for track in candidates:
                    if matches_wanted(str(track.get("languageCode") or "").lower()):
                        return track
    return tracks[_default_track_index(tracks, tracklist)]


def _default_track_index(tracks: list[dict[str, Any]], tracklist: dict[str, Any]) -> int:
    """Index of the caption track paired with the video's default audio track.

    A multi-language video lists its tracks alphabetically, so track 0 is often an
    unrelated translation rather than the language actually spoken.
    """
    audio_tracks = tracklist.get("audioTracks") or []
    audio_index = tracklist.get("defaultAudioTrackIndex")
    if isinstance(audio_index, int) and 0 <= audio_index < len(audio_tracks):
        caption_index = (audio_tracks[audio_index] or {}).get("defaultCaptionTrackIndex")
        if isinstance(caption_index, int) and 0 <= caption_index < len(tracks):
            return caption_index
    return 0


def _validated_caption_url(url: str) -> SplitResult:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _CAPTION_HOSTS:
        raise TranscriptUnavailable("YouTube returned a caption URL from an unexpected host.")
    return parsed


def _caption_url(base_url: str) -> str:
    """Ask a caption baseUrl for json3, keeping the blank-valued params YouTube sends."""
    parsed = _validated_caption_url(base_url)
    query = parse_qs(parsed.query, keep_blank_values = True)
    query["fmt"] = ["json3"]
    return urlunsplit(parsed._replace(query = urlencode(query, doseq = True)))


async def _read_capped(response: httpx.Response, limit: int, message: str) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            raise TranscriptUnavailable(message)
    return bytes(body)


async def _fetch_track_text(client: httpx.AsyncClient, base_url: str) -> str:
    url = _caption_url(base_url)

    body = b""
    for _ in range(_MAX_CAPTION_REDIRECTS + 1):
        # Redirects are followed by hand so the host allowlist covers every hop, not just the URL the player
        # response handed us.
        async with client.stream(
            "GET", url, headers = {"User-Agent": _USER_AGENT}, follow_redirects = False
        ) as response:
            location = response.headers.get("location")
            if response.is_redirect and location:
                url = urljoin(url, location)
                _validated_caption_url(url)
                continue
            response.raise_for_status()
            body = await _read_capped(
                response, _MAX_CAPTION_BYTES, "This video's captions are too large to attach."
            )
        break
    else:
        raise TranscriptUnavailable("YouTube redirected the caption request too many times.")
    if not body:
        raise TranscriptUnavailable("YouTube returned no caption text for this video.")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise TranscriptUnavailable("YouTube returned unreadable caption data.") from error
    events = payload.get("events") if isinstance(payload, dict) else None
    return _flatten_events(events or [])


def _flatten_events(events: list[Any]) -> str:
    lines: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        # aAppend cues carry only the rolling-window newline between ASR lines.
        if event.get("aAppend") == 1:
            continue
        segments = event.get("segs")
        if not isinstance(segments, list):
            continue
        joined = "".join(
            str(segment.get("utf8") or "") for segment in segments if isinstance(segment, dict)
        )
        line = " ".join(joined.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _track_label(track: dict[str, Any]) -> str:
    name = track.get("name")
    if isinstance(name, dict):
        simple = name.get("simpleText")
        if isinstance(simple, str) and simple:
            return simple
        runs = name.get("runs")
        if isinstance(runs, list):
            label = "".join(str(run.get("text") or "") for run in runs if isinstance(run, dict))
            if label:
                return label
    return str(track.get("languageCode") or "")


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
