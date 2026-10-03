# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Timestamps and speakers from an audio.cpp ``/v1/audio/transcriptions/details`` answer.

The runtime reports spans as sample offsets: ``words`` (Qwen3-ASR with its forced aligner),
``segments`` and ``speaker_turns`` (MOSS-Transcribe-Diarize, VibeVoice-ASR). This turns them into
seconds, gives each segment the speaker it overlaps most, and cleans the text, so a caller never
sees a sample count or a ``[12.5][S01]`` marker. Pure: no I/O.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, Optional

# VibeVoice-ASR counts its spans at 24 kHz whatever rate it was fed (and reports the input's rate).
FIXED_SPAN_RATES = {"vibevoice_asr": 24000, "vibevoice_asr_streaming": 24000}
# MOSS-Transcribe-Diarize embeds "[0.12][S01]" markers in its text; its segments carry the prose.
_MARKED_TEXT_FAMILIES = frozenset({"moss_transcribe_diarize"})
_MARKER_RE = re.compile(r"\[\d+(?:\.\d+)?\]|\[S\d+\]")

# Grouping aligned words into segments: a pause, a long run, or the end of a sentence.
_SEGMENT_GAP_SECONDS = 0.6
_SEGMENT_MAX_SECONDS = 10.0
_SENTENCE_END = tuple(".?!。？！")
_CLAUSE_END = tuple(",;:，；：")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _span(item: dict, rate: int) -> Optional[tuple[float, float]]:
    """``(start, end)`` in seconds, from sample offsets (or seconds when only those are given)."""
    start, end = _number(item.get("start_sample")), _number(item.get("end_sample"))
    if start is not None and end is not None:
        start, end = start / rate, end / rate
    else:
        start, end = _number(item.get("start")), _number(item.get("end"))
        if start is None or end is None:
            return None
    if start < 0 or end < start:
        return None
    return round(start, 3), round(end, 3)


def _clean_lines(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3000 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
    )


def _join_words(words: list[str]) -> str:
    """Words joined by spaces, except between two CJK characters, which run together."""
    out = ""
    for word in words:
        if out and not (_is_cjk(out[-1]) and _is_cjk(word[0])):
            out += " "
        out += word
    return out


def _words(raw: Any, rate: int) -> list[dict]:
    words = []
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, dict):
            continue
        word = item.get("word", item.get("text"))
        span = _span(item, rate)
        if not isinstance(word, str) or not word.strip() or span is None:
            continue
        words.append({"start": span[0], "end": span[1], "word": word.strip()})
    return words


def _segments(raw: Any, rate: int) -> list[dict]:
    segments = []
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        span = _span(item, rate)
        if not isinstance(text, str) or span is None:
            continue
        segments.append({"start": span[0], "end": span[1], "text": _clean_lines(text)})
    return segments


def _turns(raw: Any, rate: int) -> list[tuple[float, float, str]]:
    turns = []
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker_id", item.get("speaker"))
        if isinstance(speaker, bool) or not isinstance(speaker, (str, int)):
            continue
        speaker = str(speaker).strip()
        span = _span(item, rate)
        if not speaker or span is None:
            continue
        turns.append((span[0], span[1], speaker))
    return turns


def _assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> None:
    """Each segment takes the speaker whose turns overlap it most; none when nothing overlaps."""
    for segment in segments:
        overlap: dict[str, float] = {}
        for start, end, speaker in turns:
            shared = min(segment["end"], end) - max(segment["start"], start)
            # A zero-length segment inside a turn still belongs to it.
            if shared > 0 or (shared == 0 and start <= segment["start"] <= end):
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
        if overlap:
            segment["speaker"] = max(overlap, key = lambda s: overlap[s])


def _letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _respell(token: str, word: str) -> str:
    """``token``'s punctuation around ``word``'s letters: a chunk can restart mid-sentence with a
    capital ("Work"), while the aligned word keeps the case the sentence needs."""
    letters = iter(ch for ch in word if ch.isalnum())
    return "".join(next(letters, ch) if ch.isalnum() else ch for ch in token)


def punctuate_words(words: list[dict], text: str) -> list[dict]:
    """The aligned words spelled as in the punctuated text ("Friends," not "Friends").

    The aligner drops punctuation, and Qwen3-ASR's punctuated text repeats the overlap between
    its fixed chunks, so the two are aligned on their letters and only matching tokens are
    used: a repeated phrase is skipped, and a word with no match (CJK text has no spaces to
    split on) keeps its bare form. Punctuation-only tokens ("—") join the word before.
    """
    tokens = text.split()
    if not words or not tokens:
        return words
    letters = [_letters(token) for token in tokens]
    matcher = difflib.SequenceMatcher(
        None, [_letters(w["word"]) for w in words], letters, autojunk = False
    )
    spelled = [dict(word) for word in words]
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            index = block.b + offset
            shown = _respell(tokens[index], spelled[block.a + offset]["word"])
            # Trailing punctuation-only tokens belong to this word.
            following = index + 1
            while following < len(tokens) and not letters[following]:
                shown += tokens[following]
                following += 1
            spelled[block.a + offset]["word"] = shown
    return spelled


def group_words(words: list[dict]) -> list[dict]:
    """Aligned words grouped into segments: split on a pause, a long run or a sentence end."""
    segments: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            segments.append(
                {
                    "start": current[0]["start"],
                    "end": max(w["end"] for w in current),
                    "text": _join_words([w["word"] for w in current]),
                }
            )
            current.clear()

    for word in words:
        if current and word["start"] - current[-1]["end"] >= _SEGMENT_GAP_SECONDS:
            flush()
        elif current and word["end"] - current[0]["start"] >= _SEGMENT_MAX_SECONDS:
            # A long sentence splits after its last comma, so its tail is not left as one word.
            cut = next(
                (
                    i
                    for i in range(len(current) - 1, 0, -1)
                    if current[i - 1]["word"].endswith(_CLAUSE_END)
                ),
                len(current),
            )
            tail = current[cut:]
            del current[cut:]
            flush()
            current.extend(tail)
        current.append(word)
        if word["word"].endswith(_SENTENCE_END):
            flush()
    flush()
    return segments


def normalize(payload: dict, family: str, sent_rate: int) -> dict:
    """``{text, language, segments?, words?, speaker_ids?}`` from a ``/details`` answer.

    ``sent_rate`` is the rate of the WAV the runtime read, used when the answer names none.
    Seconds are rounded to milliseconds; ``speaker_ids`` are the runtime's own ids (``S01``,
    ``0``) in the order they first speak. Keys are present only when the model produced them.
    """
    payload = payload if isinstance(payload, dict) else {}
    rate = FIXED_SPAN_RATES.get(family)
    if rate is None:
        reported = payload.get("sample_rate")
        rate = (
            int(reported)
            if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0
            else int(sent_rate)
        )
    words = _words(payload.get("words"), rate)
    segments = _segments(payload.get("segments"), rate)
    turns = _turns(payload.get("speaker_turns"), rate)
    if not segments and turns:
        # A diarizer that reports turns only: each turn is a segment in its own words.
        segments = _segments(payload.get("speaker_turns"), rate)
    native_segments = bool(segments)
    if turns:
        _assign_speakers(segments, turns)
    raw_text = payload.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    if not segments and words:
        # Segment text reads like the transcript, punctuation included, and splits on sentences.
        # The words are the de-duplicated transcript (the runtime's punctuated text repeats each
        # chunk's overlap), so the text is rebuilt from them.
        words = punctuate_words(words, _clean_lines(text))
        segments = group_words(words)
        text = _join_words([w["word"] for w in words])

    if native_segments and (family in _MARKED_TEXT_FAMILIES or _MARKER_RE.search(text)):
        text = " ".join(s["text"] for s in segments if s["text"])
    elif _MARKER_RE.search(text):
        text = " ".join(_MARKER_RE.sub(" ", text).split())
    text = _clean_lines(text)

    language = payload.get("language")
    result: dict = {
        "text": text,
        "language": language.strip() if isinstance(language, str) and language.strip() else None,
    }
    if segments:
        result["segments"] = segments
    if words:
        result["words"] = words
    speaker_ids = list(dict.fromkeys(s["speaker"] for s in segments if "speaker" in s))
    if speaker_ids:
        result["speaker_ids"] = speaker_ids
    return result
