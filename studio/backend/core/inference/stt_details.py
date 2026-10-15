# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp ``/details`` answers (sample offsets, markers) -> seconds, speakers, clean text."""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, Optional

# Nemotron's spans are sub-word pieces ("f ever ed"), so it is not listed: its text stays whole.
ALWAYS_TIMESTAMPED = frozenset(
    {"moss_transcribe_diarize", "vibevoice_asr", "parakeet_tdt", "kroko_asr"}
)
ON_REQUEST_TIMESTAMPS = frozenset({"qwen3_asr"})
_SPAN_KEYS = ("words", "segments", "speaker_turns")
# VibeVoice-ASR counts its spans at 24 kHz whatever rate it was fed (and reports the input's rate).
FIXED_SPAN_RATES = {"vibevoice_asr": 24000, "vibevoice_asr_streaming": 24000}
_MARKED_TEXT_FAMILIES = frozenset({"moss_transcribe_diarize"})
_MARKER_RE = re.compile(r"\[\d+(?:\.\d+)?\]|\[S\d+\]")

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
    # Han and kana run together; Hangul is left out because Korean puts spaces between words.
    code = ord(char)
    return (
        0x3000 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
    )


def _join_words(words: list[str]) -> str:
    out = ""
    for word in words:
        if out and not (_is_cjk(out[-1]) and _is_cjk(word[0])):
            out += " "
        out += word
    return out


def _spanned(raw: Any, rate: int):
    for item in raw if isinstance(raw, list) else ():
        span = _span(item, rate) if isinstance(item, dict) else None
        if span is not None:
            yield item, span


def _words(raw: Any, rate: int) -> list[dict]:
    return [
        {"start": start, "end": end, "word": word.strip()}
        for item, (start, end) in _spanned(raw, rate)
        if isinstance(word := item.get("word", item.get("text")), str) and word.strip()
    ]


def _segments(raw: Any, rate: int) -> list[dict]:
    return [
        {"start": start, "end": end, "text": _clean_lines(text)}
        for item, (start, end) in _spanned(raw, rate)
        if isinstance(text := item.get("text"), str)
    ]


def _turns(raw: Any, rate: int) -> list[tuple[float, float, str]]:
    turns = []
    for item, (start, end) in _spanned(raw, rate):
        speaker = item.get("speaker_id", item.get("speaker"))
        if (
            isinstance(speaker, (str, int))
            and not isinstance(speaker, bool)
            and str(speaker).strip()
        ):
            turns.append((start, end, str(speaker).strip()))
    return turns


def _assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> None:
    for segment in segments:
        overlap: dict[str, float] = {}
        for start, end, speaker in turns:
            shared = min(segment["end"], end) - max(segment["start"], start)
            if shared > 0 or (shared == 0 and start <= segment["start"] <= end):
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
        if overlap:
            segment["speaker"] = max(overlap, key = lambda s: overlap[s])


def _letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _respell(token: str, word: str) -> str:
    """Keeps ``word``'s case: a chunk can restart mid-sentence with a capital ("Work")."""
    letters = iter(ch for ch in word if ch.isalnum())
    return "".join(next(letters, ch) if ch.isalnum() else ch for ch in token)


def _split_cjk(token: str) -> list[str]:
    """``token`` with each CJK character on its own: CJK text has no spaces to split words on."""
    units, run = [], ""
    for ch in token:
        if ch.isalnum() and _is_cjk(ch):
            if run:
                units.append(run)
                run = ""
            units.append(ch)
        else:
            run += ch
    return units + [run] if run else units


def punctuate_words(words: list[dict], text: str) -> list[dict]:
    """Aligned on letters, matches only: the text repeats chunk overlaps. A word takes the text's
    spelling only when every one of its pieces matched."""
    tokens = [unit for token in text.split() for unit in _split_cjk(token)]
    if not words or not tokens:
        return words
    pieces = [(i, unit) for i, word in enumerate(words) for unit in _split_cjk(word["word"])]
    letters = [_letters(token) for token in tokens]
    matcher = difflib.SequenceMatcher(
        None, [_letters(unit) for _, unit in pieces], letters, autojunk = False
    )
    shown: dict[int, str] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            index = block.b + offset
            piece = _respell(tokens[index], pieces[block.a + offset][1])
            following = index + 1
            while following < len(tokens) and not letters[following]:
                piece += tokens[following]
                following += 1
            shown[block.a + offset] = piece
    spelled = [dict(word) for word in words]
    by_word: dict[int, list[int]] = {}
    for n, (i, _) in enumerate(pieces):
        by_word.setdefault(i, []).append(n)
    for i, ns in by_word.items():
        if all(n in shown for n in ns):
            spelled[i]["word"] = "".join(shown[n] for n in ns)
    return spelled


def group_words(words: list[dict]) -> list[dict]:
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
    """``sent_rate`` (the WAV's rate) applies only when the answer names none."""
    payload = payload if isinstance(payload, dict) else {}
    if family not in ALWAYS_TIMESTAMPED | ON_REQUEST_TIMESTAMPS:
        payload = {k: v for k, v in payload.items() if k not in _SPAN_KEYS}
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
        segments = _segments(payload.get("speaker_turns"), rate)
    _by_time = lambda span: (span["start"], span["end"])  # noqa: E731
    words.sort(key = _by_time)
    segments.sort(key = _by_time)
    native_segments = bool(segments)
    if turns:
        _assign_speakers(segments, turns)
    raw_text = payload.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    if not segments and words:
        # Text rebuilt from words: the runtime's punctuated text repeats each chunk's overlap.
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
