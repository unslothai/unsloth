# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the pure parts of voice_bench.py: no Studio server needed.

Run from the repo root with the Studio venv python:

    python -m unittest discover -s studio/benchmarks/voice -p 'test_*.py'
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_bench as vb  # noqa: E402


def _wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), dtype = "float32"), rate, format = "WAV")
    return buf.getvalue()


def _turn(turn_id: int, **timings) -> vb.TurnResult:
    """A fully-measured turn unless a timing is overridden (None = stage failed)."""
    base = dict(
        stt_s = 0.2,
        llm_ttft_s = 0.3,
        llm_first_chunk_s = 0.5,
        llm_total_s = 1.0,
        tts_first_s = 0.4,
        tts_full_s = 1.5,
    )
    base.update(timings)
    r = vb.TurnResult(id = turn_id, ground_truth = f"turn {turn_id}", **base)
    if None not in (r.stt_s, r.llm_first_chunk_s, r.tts_first_s):
        r.first_audio_latency_s = r.stt_s + r.llm_first_chunk_s + r.tts_first_s
    if None not in (r.stt_s, r.llm_total_s, r.tts_full_s):
        r.turn_wall_s = r.stt_s + r.llm_total_s + r.tts_full_s
    return r


class FirstChunkTests(unittest.TestCase):
    def test_span_covers_chunk_and_its_boundary(self):
        cases = {
            "Yes.": "Yes.",
            "It costs 3.5 dollars today. Okay": "It costs 3.5 dollars today. ",
            "Sure thing, I can help you plan that whole trip today.": "Sure thing, ",
            "Well, this is a long enough sentence for a clause cut.": (
                "Well, this is a long enough sentence for a clause cut."
            ),
            "no punctuation at all here": "no punctuation at all here",
            "  Leading space. Then more.": "  Leading space. ",
        }
        for text, expected in cases.items():
            with self.subTest(text = text):
                span = vb.first_chunk_span(text)
                self.assertEqual(text[:span], expected)
                # The chunk the harness synthesizes first is inside what had arrived.
                self.assertIn(vb.first_chunk(text), text[:span])

    def test_arrival_is_first_sample_at_or_past_span(self):
        text = "Sure thing, I can help you plan that whole trip today."
        arrivals, cum = [], 0
        for i in range(0, len(text), 3):
            cum += len(text[i : i + 3])
            arrivals.append((cum, float(i)))
        span = vb.first_chunk_span(text)
        t = vb.first_chunk_arrival(arrivals, span)
        self.assertEqual(t, 9.0)  # "Sure thing, " is 12 chars -> 4th 3-char piece, sent at t=9
        self.assertGreater(t, arrivals[0][1])  # later than first token
        self.assertLess(t, arrivals[-1][1])  # earlier than the full reply

    def test_arrival_edge_cases(self):
        self.assertIsNone(vb.first_chunk_arrival([], 5))
        # Span past everything that arrived (no boundary): the last sample.
        self.assertEqual(vb.first_chunk_arrival([(3, 1.0), (6, 2.0)], 99), 2.0)


class ChatStreamTests(unittest.TestCase):
    def test_first_chunk_time_is_between_ttft_and_total(self):
        pieces = ["Sure", " thing,", " I", " can", " help", " you", " plan", " it", " today."]
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) for p in pieces
        ] + ["data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 9}}), "data: [DONE]"]

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_lines(self, decode_unicode = True):
                yield from lines

        client = vb.StudioClient("http://x", "t", seed = 1)
        client.s.post = lambda *a, **k: FakeResponse()
        out = client.chat_stream("m", [{"role": "user", "content": "hi"}], 1, 0.0, 32)
        self.assertEqual(out["text"], "Sure thing, I can help you plan it today.")
        self.assertEqual(out["completion_tokens"], 9)
        self.assertIsNotNone(out["ttft"])
        self.assertGreaterEqual(out["first_chunk_s"], out["ttft"])
        self.assertLessEqual(out["first_chunk_s"], out["total"])

    def test_speak_sends_seed_and_provider_fields(self):
        seen = {}

        class FakeResponse:
            content = b"RIFF"

            def raise_for_status(self):
                pass

        client = vb.StudioClient(
            "http://x",
            "t",
            seed = 7,
            tts_provider = {"provider_id": "p1", "model": "tts-1", "voice": "alloy"},
        )
        client.s.post = lambda url, json, timeout: seen.update(json) or FakeResponse()
        client.speak("hello")
        self.assertEqual(
            seen, {"input": "hello", "seed": 7, "provider_id": "p1", "model": "tts-1", "voice": "alloy"}
        )


class SummarizeTests(unittest.TestCase):
    def test_complete_run(self):
        s = vb.summarize([[_turn(1), _turn(2)], [_turn(1), _turn(2)]])
        self.assertTrue(s["complete"])
        self.assertEqual(s["incomplete"], [])
        self.assertAlmostEqual(s["totals"]["first_audio_latency_s"], 2 * (0.2 + 0.5 + 0.4))
        self.assertAlmostEqual(s["means"]["llm_first_chunk_s"], 0.5)

    def test_failed_stage_marks_run_incomplete(self):
        bad = _turn(2, tts_first_s = None, tts_full_s = None)
        bad.errors.append("tts_first: 400")
        s = vb.summarize([[_turn(1), bad]])
        self.assertFalse(s["complete"])
        self.assertEqual(len(s["incomplete"]), 1)
        gap = s["incomplete"][0]
        self.assertEqual((gap["pass"], gap["turn"]), (1, 2))
        self.assertEqual(gap["errors"], ["tts_first: 400"])
        self.assertEqual(gap["missing"], ["tts_first_s", "tts_full_s"])

    def test_empty_passes_rejected(self):
        with self.assertRaises(ValueError):
            vb.summarize([])

    def test_diff_skips_incomplete_run(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.json"
            base.write_text(json.dumps({"summary": vb.summarize([[_turn(1)]])}), encoding = "utf-8")
            incomplete = vb.summarize([[_turn(1, stt_s = None)]])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                vb.diff_baseline(incomplete, base)
            self.assertIn("skipping diff", out.getvalue())
            self.assertNotIn("DIFF vs baseline", out.getvalue())


class ArgTests(unittest.TestCase):
    def test_repeats_must_be_positive(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--repeats", type = vb.positive_int, default = 1)
        self.assertEqual(ap.parse_args(["--repeats", "3"]).repeats, 3)
        for bad in ("0", "-2", "x"):
            with self.subTest(value = bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    ap.parse_args(["--repeats", bad])
                self.assertEqual(cm.exception.code, 2)


class TtsRouteTests(unittest.TestCase):
    @staticmethod
    def _args(**kw):
        d = dict(tts_provider_id = None, tts_model = None, tts_voice = None)
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_main_alone_has_no_local_route(self):
        # origin/main: one resident chat model, no voice slot -> every speak() would 400.
        provider, _, err = vb.resolve_tts_route(
            self._args(), {"active_model": "gemma", "is_audio": False}, {}
        )
        self.assertIsNone(provider)
        self.assertIn("#10373", err)
        self.assertIn("--tts-provider-id", err)

    def test_resident_tts_model_means_no_chat(self):
        _, _, err = vb.resolve_tts_route(
            self._args(), {"active_model": "orpheus", "is_audio": True}, {}
        )
        self.assertIn("speech instead of text", err)

    def test_voice_slot_loaded(self):
        provider, label, err = vb.resolve_tts_route(
            self._args(), {"is_audio": False}, {"loaded": True, "model": "orpheus-3b"}
        )
        self.assertEqual((provider, label, err), (None, "orpheus-3b", ""))

    def test_provider_requires_model_and_voice(self):
        _, _, err = vb.resolve_tts_route(self._args(tts_provider_id = "p1"), {}, {})
        self.assertIn("--tts-model", err)

    def test_provider_route(self):
        provider, label, err = vb.resolve_tts_route(
            self._args(tts_provider_id = "p1", tts_model = "tts-1", tts_voice = "alloy"), {}, {}
        )
        self.assertEqual(err, "")
        self.assertEqual(provider, {"provider_id": "p1", "model": "tts-1", "voice": "alloy"})
        self.assertIn("p1", label)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (vb.FIXTURES, vb.DOWNLOADS)
        vb.FIXTURES, vb.DOWNLOADS = root / "fixtures", root / "downloads"
        self.wav = _wav()
        self.spoken: list[str] = []
        test = self

        class FakeClient:
            def speak(self, text):
                test.spoken.append(text)
                return test.wav, 0.01

        self.client = FakeClient()

    def tearDown(self):
        vb.FIXTURES, vb.DOWNLOADS = self._saved
        self.tmp.cleanup()

    def test_synthesized_fixture_is_cached_then_regenerated_on_text_change(self):
        wav, dur, generated = vb.ensure_fixture(self.client, {"id": 1, "text": "hello"})
        self.assertTrue(generated)
        self.assertEqual(wav, self.wav)
        self.assertAlmostEqual(dur, 0.1)
        self.assertEqual(
            json.loads((vb.FIXTURES / "turn_1.json").read_text()),
            {"text": "hello", "source": "synthesized"},
        )
        self.assertTrue((vb.DOWNLOADS / "turn_1.wav").exists())

        _, _, generated = vb.ensure_fixture(self.client, {"id": 1, "text": "hello"})
        self.assertFalse(generated)

        with contextlib.redirect_stdout(io.StringIO()):
            _, _, generated = vb.ensure_fixture(self.client, {"id": 1, "text": "goodbye"})
        self.assertTrue(generated)
        self.assertEqual(self.spoken, ["hello", "goodbye"])

    def test_supplied_recording_is_adopted_and_never_overwritten(self):
        vb.FIXTURES.mkdir(parents = True)
        (vb.FIXTURES / "turn_2.wav").write_bytes(self.wav)
        _, _, generated = vb.ensure_fixture(self.client, {"id": 2, "text": "mine"})
        self.assertFalse(generated)
        self.assertEqual(
            json.loads((vb.FIXTURES / "turn_2.json").read_text()),
            {"text": "mine", "source": "supplied"},
        )
        with self.assertRaises(vb.FixtureMismatch):
            vb.ensure_fixture(self.client, {"id": 2, "text": "someone else's line"})
        self.assertEqual(self.spoken, [])
        self.assertEqual((vb.FIXTURES / "turn_2.wav").read_bytes(), self.wav)

    def test_corrupt_sidecar_is_treated_as_supplied(self):
        vb.FIXTURES.mkdir(parents = True)
        (vb.FIXTURES / "turn_3.wav").write_bytes(self.wav)
        (vb.FIXTURES / "turn_3.json").write_text("{not json")
        _, _, generated = vb.ensure_fixture(self.client, {"id": 3, "text": "t"})
        self.assertFalse(generated)
        self.assertEqual(json.loads((vb.FIXTURES / "turn_3.json").read_text())["source"], "supplied")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    unittest.main()
