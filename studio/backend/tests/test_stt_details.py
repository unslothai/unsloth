# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stt_details.normalize: audio.cpp ``/details`` answers (as the S4 spike recorded them) in seconds."""

import pytest

from core.inference import stt_details

_MOSS_TEXTS = (
    ("S01", 2880, 75360, "Welcome back to the studio."),
    ("S01", 77040, 187680, "Check the solar forecast."),
    ("S02", 201840, 244320, "I reviewed the forecast this morning."),
    ("S03", 387600, 511920, "The radios are charged and paired."),
    ("S04", 533280, 671040, "Good. I will send everyone the final plan before dinner."),
)


def _moss(rate = 24000):
    scale = rate / 24000
    segments, turns, marked = [], [], []
    for speaker, start, end, text in _MOSS_TEXTS:
        start, end = int(start * scale), int(end * scale)
        segments.append({"start_sample": start, "end_sample": end, "confidence": 0, "text": text})
        turns.append({**segments[-1], "speaker_id": speaker})
        marked.append(f"[{start / rate:.2f}][{speaker}] {text}[{end / rate:.2f}]")
    return {
        "text": "".join(marked),
        "segments": segments,
        "speaker_turns": turns,
        "sample_rate": rate,
    }


def _seq(words, step = 0.5):
    """``words`` back to back, ``step`` seconds each."""
    return [{"start": i * step, "end": i * step + step, "word": w} for i, w in enumerate(words)]


def _payload(words, text):
    spans = [
        {"word": w["word"], "start_sample": w["start"] * 16000, "end_sample": w["end"] * 16000}
        for w in words
    ]
    return {"text": text, "words": spans, "sample_rate": 16000}


@pytest.mark.parametrize("rate", [24000, 16000])
def test_moss_markers_are_stripped_and_segments_carry_speakers(rate):
    result = stt_details.normalize(_moss(rate), "moss_transcribe_diarize", rate)
    assert result["text"] == " ".join(text for *_, text in _MOSS_TEXTS)
    segments = result["segments"]
    assert [s["speaker"] for s in segments] == ["S01", "S01", "S02", "S03", "S04"]
    assert result["speaker_ids"] == ["S01", "S02", "S03", "S04"]
    assert (segments[0]["start"], segments[0]["end"], segments[-1]["end"]) == (0.12, 3.14, 27.96)
    assert "words" not in result


def test_vibevoice_spans_are_always_24k_and_a_segment_without_a_turn_has_no_speaker():
    payload = {
        "text": "Concord returned. Later.",
        "segments": [
            {"start_sample": 0, "end_sample": 84000, "text": "Concord returned."},
            {"start_sample": 96000, "end_sample": 120000, "text": "Later."},
        ],
        "speaker_turns": [{"start_sample": 0, "end_sample": 90000, "speaker_id": "0"}],
        "sample_rate": 16000,
        "language": "en",
    }
    assert stt_details.normalize(payload, "vibevoice_asr", 16000) == {
        "text": "Concord returned. Later.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.5, "text": "Concord returned.", "speaker": "0"},
            {"start": 4.0, "end": 5.0, "text": "Later."},
        ],
        "speaker_ids": ["0"],
    }


def test_qwen3_words_become_seconds_and_take_the_texts_punctuation():
    payload = _payload(_seq(["Concord", "returned", "home"]), "Concord returned home.")
    result = stt_details.normalize({**payload, "language": "English"}, "qwen3_asr", 16000)
    assert result["text"] == "Concord returned home." and result["language"] == "English"
    assert result["words"][-1] == {"start": 1.0, "end": 1.5, "word": "home."}
    assert result["segments"] == [{"start": 0.0, "end": 1.5, "text": "Concord returned home."}]


def test_word_grouping_splits_on_a_pause_a_long_run_and_a_sentence_end():
    words = _seq(["One", "two.", "Three", "four"])
    words[3].update(start = 2.5, end = 3.0)
    assert [(g["text"], g["start"], g["end"]) for g in stt_details.group_words(words)] == [
        ("One two.", 0.0, 1.0),
        ("Three", 1.0, 1.5),
        ("four", 2.5, 3.0),
    ]
    runs = stt_details.group_words(_seq([f"w{i}" for i in range(30)]))
    assert len(runs) == 2 and all(r["end"] - r["start"] <= 10.0 for r in runs)
    text = "the great work is still done by patience, courage, memory and care."
    assert [s["text"] for s in stt_details.group_words(_seq(text.split(), 0.9))] == [
        "the great work is still done by patience, courage,",
        "memory and care.",
    ]
    # CJK words run together; a CJK word next to a Latin one keeps its space. Korean keeps spaces.
    assert (
        stt_details.group_words(_seq(["你好", "世界", "AI", "。"]))[0]["text"] == "你好世界 AI 。"
    )
    assert (
        stt_details.group_words(_seq(["안녕하세요", "반갑습니다"]))[0]["text"]
        == "안녕하세요 반갑습니다"
    )


def test_malformed_spans_are_dropped():
    payload = _moss()
    payload["segments"][1]["start_sample"] = "soon"
    payload["segments"][2]["end_sample"] = 1
    payload["segments"] += ["not a segment", {"start_sample": 0, "end_sample": 10}]
    payload["speaker_turns"].append({"start_sample": True, "end_sample": 4, "speaker_id": "S09"})
    payload["words"] = [
        {"word": "ok", "start_sample": 0, "end_sample": 16000},
        {"word": "", "start_sample": 0, "end_sample": 1},
        {"word": "bad", "start_sample": float("nan"), "end_sample": 1},
        {"word": "neg", "start_sample": -5, "end_sample": 1},
    ]
    result = stt_details.normalize(payload, "moss_transcribe_diarize", 24000)
    assert len(result["segments"]) == 3 and "S09" not in result["speaker_ids"]
    assert result["words"] == [{"start": 0.0, "end": 0.667, "word": "ok"}]
    assert stt_details.normalize("not a dict", "qwen3_asr", 16000) == {"text": "", "language": None}


@pytest.mark.parametrize(
    "words,text,expected",
    [
        (
            ["Friends", "neighbors", "lifewe", "meet", "Again"],
            "Friends, neighbors, life—we meet. Again!",
            ["Friends,", "neighbors,", "life—we", "meet.", "Again!"],
        ),
        (["Friends", "neighbors"], "Friends — neighbors", ["Friends—", "neighbors"]),
        (["hello", "world"], "helloworld", ["hello", "world"]),
        (["hello", "world"], "", ["hello", "world"]),
        (["hello", "world"], "Goodbye, world!", ["hello", "world!"]),
        (
            "the great work of a people is done by patience courage and care".split(),
            "the great work of a people is done by patience. "
            "Work of a people is done by patience, courage, and care.",
            "the great work of a people is done by patience, courage, and care.".split(),
        ),
    ],
)
def test_aligned_words_take_the_punctuation_of_the_text(words, text, expected):
    bare = _seq(words)
    spelled = stt_details.punctuate_words(bare, text)
    assert [w["word"] for w in spelled] == expected
    assert [(w["start"], w["end"]) for w in spelled] == [(w["start"], w["end"]) for w in bare]
    assert [w["word"] for w in bare] == words


@pytest.mark.parametrize(
    "payload,family,text",
    [
        # Shape audiocpp_server returned on a T4: text whole, words split mid-word.
        (
            _payload(
                _seq(["He", "was", "in", "a", "f", "ever", "ed", "sta", "te"]),
                "He was in a fevered state",
            ),
            "nemotron_asr",
            "He was in a fevered state",
        ),
        ({"text": "  hello\n world  "}, "parakeet_tdt", "hello world"),
        ({"text": "[0.12][S01] hi there[1.5]"}, "other_asr", "hi there"),
    ],
)
def test_plain_text_and_nemotron_sub_word_spans_never_replace_text(payload, family, text):
    assert stt_details.normalize(payload, family, 16000) == {"text": text, "language": None}


@pytest.mark.parametrize("split", [["你好", "世界"], ["你", "好", "世", "界"]])
def test_cjk_text_keeps_its_punctuation_through_alignment(split):
    # Qwen3-ASR's punctuated Chinese has no spaces; the aligner splits it into words.
    result = stt_details.normalize(_payload(_seq(split, 0.3), "你好，世界。"), "qwen3_asr", 16000)
    assert result["text"] == "你好，世界。" and result["segments"][0]["text"] == "你好，世界。"


def test_spans_come_back_in_time_order_whatever_order_the_runtime_used():
    """A diarized answer grouped per speaker left segments out of order, and the player's binary
    search and paragraph merge both assume time order."""
    payload = {
        "text": "a b c",
        "sample_rate": 16000,
        "segments": [
            {"start_sample": 80000, "end_sample": 96000, "text": "c"},
            {"start_sample": 0, "end_sample": 16000, "text": "a"},
            {"start_sample": 32000, "end_sample": 48000, "text": "b"},
        ],
        "words": [
            {"start_sample": 32000, "end_sample": 48000, "word": "b"},
            {"start_sample": 0, "end_sample": 16000, "word": "a"},
        ],
    }
    result = stt_details.normalize(payload, "moss_transcribe_diarize", 16000)
    assert [s["start"] for s in result["segments"]] == [0.0, 2.0, 5.0]
    assert [w["word"] for w in result["words"]] == ["a", "b"]
