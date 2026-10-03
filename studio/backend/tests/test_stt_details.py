# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stt_details.normalize: audio.cpp ``/details`` answers (as the S4 spike recorded them) in seconds."""

import copy

from core.inference import stt_details

_MOSS_TEXTS = (
    ("S01", 2880, 75360, "Welcome back to the studio. The agenda today is simple."),
    (
        "S01",
        77040,
        187680,
        "Check the solar forecast, test the backup radios, and agree on tomorrow's route.",
    ),
    ("S02", 201840, 244320, "I reviewed the forecast this morning."),
    (
        "S02",
        251520,
        341280,
        "The east ridge should stay clear, but the valley may get fog after sunrise.",
    ),
    (
        "S03",
        387600,
        511920,
        "The radios are charged and paired. I also labeled the spare batteries so nobody mixes "
        "fresh cells with used ones.",
    ),
    (
        "S04",
        533280,
        671040,
        "Good. I will update the route map, mark the fog zone, and send everyone the final plan "
        "before dinner.",
    ),
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
        "timing": {"wall_ms": 1601.5},
    }


_QWEN3_WORDS = [
    ("Concord", 9088, 19328),
    ("returned", 19328, 25728),
    ("to", 27008, 28288),
    ("its", 28288, 30848),
    ("place", 30848, 35968),
    ("amidst", 35968, 43648),
    ("the", 43648, 44928),
    ("tents", 44928, 53888),
]


def _qwen3(words = _QWEN3_WORDS, text = "Concord returned to its place amidst the tents."):
    return {
        "text": text,
        "language": "English",
        "words": [
            {"word": w, "start_sample": s, "end_sample": e, "confidence": 0} for w, s, e in words
        ],
        "sample_rate": 16000,
    }


def test_moss_markers_are_stripped_and_segments_carry_speakers():
    result = stt_details.normalize(_moss(), "moss_transcribe_diarize", 24000)
    assert "[" not in result["text"] and "]" not in result["text"]
    assert result["text"].startswith("Welcome back to the studio.")
    assert result["text"].endswith("before dinner.")
    segments = result["segments"]
    assert len(segments) == 6
    assert [s["speaker"] for s in segments] == ["S01", "S01", "S02", "S02", "S03", "S04"]
    assert result["speaker_ids"] == ["S01", "S02", "S03", "S04"]
    assert (segments[0]["start"], segments[0]["end"]) == (0.12, 3.14)
    assert segments[-1]["end"] == 27.96
    assert all(s["end"] >= s["start"] for s in segments)
    assert "words" not in result


def test_moss_at_16k_divides_by_its_reported_rate():
    result = stt_details.normalize(_moss(16000), "moss_transcribe_diarize", 16000)
    assert result["segments"][-1]["end"] == 27.96
    assert result["segments"][0]["start"] == 0.12


def test_vibevoice_spans_are_always_24k():
    payload = {
        "text": "Concord returned to its place amidst the tents.",
        "segments": [{"start_sample": 0, "end_sample": 84000, "text": "Concord returned."}],
        "speaker_turns": [{"start_sample": 0, "end_sample": 84000, "speaker_id": "0"}],
        "sample_rate": 16000,
        "language": "en",
    }
    result = stt_details.normalize(payload, "vibevoice_asr", 16000)
    assert result["segments"] == [
        {"start": 0.0, "end": 3.5, "text": "Concord returned.", "speaker": "0"}
    ]
    assert result["speaker_ids"] == ["0"]
    assert result["text"] == payload["text"] and result["language"] == "en"


def test_qwen3_words_become_seconds_and_one_segment():
    result = stt_details.normalize(_qwen3(), "qwen3_asr", 16000)
    assert result["text"] == "Concord returned to its place amidst the tents."
    assert result["language"] == "English"
    assert len(result["words"]) == 8
    assert result["words"][0] == {"start": 0.568, "end": 1.208, "word": "Concord"}
    assert result["words"][-1]["end"] == 3.368
    (segment,) = result["segments"]
    assert segment == {
        "start": 0.568,
        "end": 3.368,
        # The full stop comes from the text; the aligner's own words have none.
        "text": "Concord returned to its place amidst the tents.",
    }
    assert "speaker_ids" not in result


def test_word_grouping_splits_on_a_pause_a_long_run_and_a_sentence_end():
    words = [
        {"start": 0.0, "end": 0.4, "word": "One"},
        {"start": 0.5, "end": 0.9, "word": "two."},
        {"start": 1.0, "end": 1.4, "word": "Three"},
        {"start": 2.0, "end": 2.4, "word": "four"},
    ]
    groups = stt_details.group_words(words)
    assert [g["text"] for g in groups] == ["One two.", "Three", "four"]
    assert [(g["start"], g["end"]) for g in groups] == [(0.0, 0.9), (1.0, 1.4), (2.0, 2.4)]
    # No pause, no punctuation: a run is cut once it reaches 10 s.
    steady = [{"start": i * 0.5, "end": i * 0.5 + 0.5, "word": f"w{i}"} for i in range(30)]
    runs = stt_details.group_words(steady)
    assert len(runs) == 2 and all(r["end"] - r["start"] <= 10.0 for r in runs)
    # CJK words run together; a CJK word next to a Latin one keeps its space.
    cjk = [
        {"start": 0.0, "end": 0.2, "word": "你好"},
        {"start": 0.2, "end": 0.4, "word": "世界"},
        {"start": 0.4, "end": 0.6, "word": "AI"},
        {"start": 0.6, "end": 0.8, "word": "。"},
    ]
    assert stt_details.group_words(cjk)[0]["text"] == "你好世界 AI 。"
    assert stt_details.group_words([]) == []


def test_a_payload_without_spans_is_text_only():
    result = stt_details.normalize(
        {"text": "  hello\n world  ", "timing": {}}, "parakeet_tdt", 16000
    )
    assert result == {"text": "hello world", "language": None}
    # Markers in a family that has no segments are still removed from the prose.
    marked = stt_details.normalize({"text": "[0.12][S01] hi there[1.5]"}, "other_asr", 16000)
    assert marked == {"text": "hi there", "language": None}


def test_malformed_spans_are_dropped():
    payload = _moss()
    payload["segments"] = copy.deepcopy(payload["segments"])
    payload["segments"][1]["start_sample"] = "soon"
    payload["segments"][2]["end_sample"] = 1  # ends before it starts
    payload["segments"].append("not a segment")
    payload["segments"].append({"start_sample": 0, "end_sample": 10})  # no text
    payload["speaker_turns"].append({"start_sample": True, "end_sample": 4, "speaker_id": "S09"})
    payload["words"] = [
        {"word": "ok", "start_sample": 0, "end_sample": 16000},
        {"word": "", "start_sample": 0, "end_sample": 1},
        {"word": "bad", "start_sample": float("nan"), "end_sample": 1},
        {"word": "neg", "start_sample": -5, "end_sample": 1},
    ]
    result = stt_details.normalize(payload, "moss_transcribe_diarize", 24000)
    assert len(result["segments"]) == 4
    assert "S09" not in result["speaker_ids"]
    assert result["words"] == [{"start": 0.0, "end": 0.667, "word": "ok"}]
    assert stt_details.normalize("not a dict", "qwen3_asr", 16000) == {
        "text": "",
        "language": None,
    }


def test_a_segment_with_no_overlapping_turn_has_no_speaker():
    payload = {
        "text": "a b",
        "segments": [
            {"start_sample": 0, "end_sample": 100, "text": "a"},
            {"start_sample": 200, "end_sample": 300, "text": "b"},
        ],
        "speaker_turns": [{"start_sample": 0, "end_sample": 120, "speaker_id": "S01"}],
        "sample_rate": 100,
    }
    first, second = stt_details.normalize(payload, "moss_transcribe_diarize", 100)["segments"]
    assert first["speaker"] == "S01" and "speaker" not in second


def test_aligned_words_take_the_punctuation_of_the_text():
    words = [
        {"start": 0.0, "end": 0.4, "word": "Friends"},
        {"start": 0.4, "end": 0.9, "word": "neighbors"},
        {"start": 0.9, "end": 1.4, "word": "lifewe"},
        {"start": 1.4, "end": 1.8, "word": "meet"},
        {"start": 2.6, "end": 3.0, "word": "Again"},
    ]
    spelled = stt_details.punctuate_words(words, "Friends, neighbors, life\u2014we meet. Again!")
    assert [w["word"] for w in spelled] == [
        "Friends,",
        "neighbors,",
        "life\u2014we",
        "meet.",
        "Again!",
    ]
    assert [(w["start"], w["end"]) for w in spelled] == [(w["start"], w["end"]) for w in words]
    # Punctuation-only tokens join the word before; the sentence end now splits segments.
    dashed = stt_details.punctuate_words(words[:2], "Friends \u2014 neighbors")
    assert [w["word"] for w in dashed] == ["Friends\u2014", "neighbors"]
    payload = {
        "text": "Friends, neighbors, life\u2014we meet. Again!",
        "words": [
            {
                "word": w["word"],
                "start_sample": int(w["start"] * 16000),
                "end_sample": int(w["end"] * 16000),
            }
            for w in words
        ],
        "sample_rate": 16000,
    }
    result = stt_details.normalize(payload, "qwen3_asr", 16000)
    assert [s["text"] for s in result["segments"]] == [
        "Friends, neighbors, life\u2014we meet.",
        "Again!",
    ]
    assert result["text"] == "Friends, neighbors, life\u2014we meet. Again!"


def test_words_without_a_match_in_the_text_are_kept_bare():
    words = [
        {"start": 0.0, "end": 0.4, "word": "hello"},
        {"start": 0.4, "end": 0.9, "word": "world"},
    ]
    # A token spanning two words, or no text: nothing lines up.
    for text in ("helloworld", ""):
        assert stt_details.punctuate_words(words, text) == words
    # Only the words that match take the text's spelling; the inputs are never changed.
    spelled = stt_details.punctuate_words(words, "Goodbye, world!")
    assert [w["word"] for w in spelled] == ["hello", "world!"]
    assert words[1]["word"] == "world"


def test_chunk_overlap_repeated_in_the_text_is_skipped():
    bare = "the great work of a people is still done by patience courage and care"
    words = [
        {"start": float(i), "end": float(i) + 0.5, "word": w} for i, w in enumerate(bare.split())
    ]
    text = (
        "the great work of a people is still done by patience. "
        "Work of a people is still done by patience, courage, and care."
    )
    spelled = stt_details.punctuate_words(words, text)
    # The second chunk restarts with "Work"; the aligned word keeps the sentence's case.
    assert " ".join(w["word"] for w in spelled) == (
        "the great work of a people is still done by patience, courage, and care."
    )
    payload = {
        "text": text,
        "words": [
            {
                "word": w["word"],
                "start_sample": int(w["start"] * 16000),
                "end_sample": int(w["end"] * 16000),
            }
            for w in words
        ],
        "sample_rate": 16000,
    }
    result = stt_details.normalize(payload, "qwen3_asr", 16000)
    assert result["text"] == (
        "the great work of a people is still done by patience, courage, and care."
    )


def test_a_long_sentence_splits_after_its_last_comma():
    # 12 s without a pause or a full stop: cut after "courage," rather than strand "care."
    text = "the great work is still done by patience, courage, memory and care."
    words = [
        {"start": i * 1.0, "end": i * 1.0 + 0.9, "word": w} for i, w in enumerate(text.split())
    ]
    segments = stt_details.group_words(words)
    assert [s["text"] for s in segments] == [
        "the great work is still done by patience, courage,",
        "memory and care.",
    ]
    assert segments[1]["start"] == words[9]["start"]


def test_korean_aligned_words_keep_their_spaces():
    words = [
        {"start": 0.0, "end": 0.4, "word": "안녕하세요"},
        {"start": 0.5, "end": 0.9, "word": "반갑습니다"},
    ]
    assert stt_details.group_words(words)[0]["text"] == "안녕하세요 반갑습니다"


def test_nemotron_sub_word_spans_never_replace_its_text():
    # Shape audiocpp_server returned on a T4: text whole, words split mid-word.
    payload = {
        "text": "He was in a fevered state of mind",
        "sample_rate": 16000,
        "words": [
            {"word": w, "start_sample": i * 1600, "end_sample": i * 1600 + 1500}
            for i, w in enumerate(["He", "was", "in", "a", "f", "ever", "ed", "sta", "te"])
        ],
    }
    result = stt_details.normalize(payload, "nemotron_asr", 16000)
    assert result == {"text": "He was in a fevered state of mind", "language": None}


def test_cjk_text_keeps_its_punctuation_through_alignment():
    # Qwen3-ASR's punctuated Chinese has no spaces; the aligner splits it into words.
    for split in (["你好", "世界"], ["你", "好", "世", "界"]):
        words = [{"start": n * 0.3, "end": n * 0.3 + 0.2, "word": w} for n, w in enumerate(split)]
        spelled = stt_details.punctuate_words(words, "你好，世界。")
        assert "".join(w["word"] for w in spelled) == "你好，世界。"
        assert stt_details.group_words(spelled)[0]["text"] == "你好，世界。"
