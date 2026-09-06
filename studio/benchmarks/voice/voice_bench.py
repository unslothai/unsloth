# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Voice-pipeline latency benchmark.

Fakes a full realtime voice conversation against a *running* Studio backend and
measures where the time goes, stage by stage:

    input speech (wav)  --STT-->  transcript
    transcript          --LLM-->  reply text        (streamed; first-token timed)
    reply text          --TTS-->  reply speech (wav)

For each of the 4 scripted turns it records, per stage, the true wall-clock
elapsed AND a length-normalized rate so a longer utterance is not unfairly
counted as "slower":

    STT real-time factor  = input_audio_seconds  / stt_seconds     (>1 = faster than real time)
    TTS real-time factor  = output_audio_seconds / tts_seconds
    LLM throughput        = completion_tokens     / llm_seconds     (tokens/sec)

The headline number to drive down is `first_audio_latency` = the time from the
end of your speech to the first audio coming back, as a clause-first streaming
pipeline would deliver it:

    first_audio_latency = stt_seconds + llm_first_chunk_seconds + tts_first_chunk_seconds

where `llm_first_chunk_seconds` is when the first synthesizable chunk (opening
clause or short sentence, plus its closing boundary) had streamed in, not the
first token: speech cannot start on a token. `llm_ttft` is still reported.

Everything is deterministic: fixed seed, temperature 0 (greedy), and the input
audio is cached to disk on first run so every later run feeds identical bytes.
A determinism check re-runs the first turn's LLM (after the measured passes, so
it cannot pre-warm them) and asserts an identical reply.

Exit status: 0 valid run; 1 measured but invalid (a turn failed a stage, or the
determinism check failed); 2 could not run.

Usage (from this folder, with Studio already running and a chat model + a TTS
voice loaded in the UI):

    python voice_bench.py                       # run + write a timestamped report
    python voice_bench.py --repeats 3           # 3 measured passes, report the median
    python voice_bench.py --baseline reports/latest.json   # diff against a prior run

Run with the Studio venv python so the token bootstrap can import auth.storage.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "audio_fixtures"
REPORTS = HERE / "reports"
# The Orpheus-synthesized input utterances are also mirrored here so they're easy
# to find / listen to (the goal asks the generated wavs to land in Downloads).
DOWNLOADS = Path.home() / "Downloads" / "voice_bench_fixtures"
DEFAULT_BASE_URL = os.environ.get("UNSLOTH_BASE_URL", "http://127.0.0.1:8888")


# ─────────────────────────────────────────── metrics helpers ──────────────

_WORD_RE = re.compile(r"[^\w\s]", flags = re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (for WER / matching)."""
    return " ".join(_WORD_RE.sub(" ", text.lower()).split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance / reference length. 0.0 = perfect."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein over word lists.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] if r == h else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1] / len(ref)


def wav_duration_seconds(wav_bytes: bytes) -> float:
    with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
        return len(f) / f.samplerate


def first_sentence(text: str) -> str:
    """The first full sentence of a reply."""
    m = re.search(r".+?[.!?](\s|$)", text.strip(), flags = re.DOTALL)
    return (m.group(0).strip() if m else text.strip()) or text.strip()


def first_chunk(text: str) -> str:
    """The smallest chunk a latency-optimized streaming TTS should emit FIRST -- what
    gates first audio. If the opening sentence is long, break at its first clause
    boundary (comma/semicolon/colon) so audio starts on a short opening clause
    instead of waiting to synthesize a whole long sentence; short sentences are used
    whole. This is the single biggest first-audio win when TTS runs near real time."""
    sent = first_sentence(text)
    if len(sent.split()) <= 6:
        return sent
    m = re.search(r".+?[,;:](\s|$)", sent, flags = re.DOTALL)
    if m:
        clause = m.group(0).strip().rstrip(",;:").strip()
        if len(clause.split()) >= 2:
            return clause
    return sent


def first_chunk_span(text: str) -> int:
    """How many characters of the raw streamed reply must have arrived before
    first_chunk(text) is known to be complete.

    That is the chunk plus the boundary that closes it: the terminator and the
    whitespace after it for a sentence (a "." followed by "5" is a decimal, not an
    end), or the comma/semicolon/colon and its trailing space for a clause cut. A
    reply with no boundary at all needs every character."""
    lead = len(text) - len(text.lstrip())
    body = text.strip()
    m = re.search(r".+?[.!?](\s|$)", body, flags = re.DOTALL)
    if not m:
        return len(text)
    sent = m.group(0).strip()
    if len(sent.split()) > 6:
        mc = re.search(r".+?[,;:](\s|$)", sent, flags = re.DOTALL)
        if mc and len(mc.group(0).strip().rstrip(",;:").split()) >= 2:
            return lead + mc.end()
    return lead + m.end()


def first_chunk_arrival(arrivals: list[tuple[int, float]], span: int) -> Optional[float]:
    """The stream time at which ``span`` characters had arrived: the earliest
    (cumulative_chars, seconds) sample at or past it, else the last sample."""
    if not arrivals:
        return None
    for cum_len, t in arrivals:
        if cum_len >= span:
            return t
    return arrivals[-1][1]


def _fmt(x: Optional[float], unit: str = "s", nd: int = 3) -> str:
    return "  n/a " if x is None else f"{x:.{nd}f}{unit}"


# ─────────────────────────────────────────── HTTP client ──────────────────


class StudioClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        seed: int,
        timeout: float = 180.0,
        tts_provider: Optional[dict] = None,
    ):
        """``tts_provider`` = {"provider_id", "model", "voice"} routes every
        /v1/audio/speech call to that saved external TTS connection instead of the
        model Studio has loaded (the [x-unsloth] provider_id extension)."""
        self.base = base_url.rstrip("/")
        self.seed = seed
        self.timeout = timeout
        self.tts_provider = tts_provider
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def status(self) -> dict:
        r = self.s.get(f"{self.base}/api/inference/status", timeout = 30)
        r.raise_for_status()
        return r.json()

    def voice_status(self) -> dict:
        try:
            r = self.s.get(f"{self.base}/api/inference/voice/status", timeout = 30)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        return {}

    def transcribe(self, wav_bytes: bytes, model: Optional[str]) -> tuple[str, float]:
        files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
        data = {"model": model} if model else None
        t0 = time.perf_counter()
        r = self.s.post(
            f"{self.base}/v1/audio/transcriptions",
            files = files,
            data = data,
            timeout = self.timeout,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return r.json().get("text", ""), elapsed

    def speak(self, text: str) -> tuple[bytes, float]:
        """Synthesize ``text`` with the benchmark seed (AudioSpeechRequest.seed is
        best-effort: honoured where the loaded TTS backend takes a seed)."""
        body: dict = {"input": text, "seed": self.seed}
        if self.tts_provider:
            body.update(self.tts_provider)
        t0 = time.perf_counter()
        r = self.s.post(
            f"{self.base}/v1/audio/speech",
            json = body,
            timeout = self.timeout,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return r.content, elapsed

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        seed: int,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> dict:
        """Stream a completion; return SPOKEN text, time-to-first-spoken-token, etc.

        Reasoning ("thinking") models stream their chain-of-thought under
        `reasoning_content`, not `content`, and can burn the whole token budget
        thinking before a single spoken word -- pure first-audio latency the user
        just waits through. For realtime voice we default thinking OFF
        (chat_template_kwargs.enable_thinking=false) and always measure TTFT against
        the first CONTENT (spoken) token, so the number reflects when audio can
        actually start. `reason_chunks` records whether the model still reasoned."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        t0 = time.perf_counter()
        ttft: Optional[float] = None  # time to first spoken (content) token
        think_first: Optional[float] = None
        think_last: Optional[float] = None
        chunks = 0
        reason_chunks = 0
        completion_tokens: Optional[int] = None
        parts: list[str] = []
        arrivals: list[tuple[int, float]] = []  # (cumulative content chars, seconds)
        cum_len = 0
        with self.s.post(
            f"{self.base}/v1/chat/completions",
            json = payload,
            stream = True,
            timeout = self.timeout,
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode = True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    completion_tokens = usage["completion_tokens"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        now = time.perf_counter() - t0
                        if think_first is None:
                            think_first = now
                        think_last = now
                        reason_chunks += 1
                    piece = delta.get("content")
                    if piece:
                        now = time.perf_counter() - t0
                        if ttft is None:
                            ttft = now
                        chunks += 1
                        parts.append(piece)
                        cum_len += len(piece)
                        arrivals.append((cum_len, now))
        total = time.perf_counter() - t0
        # Dead time spent reasoning before the first spoken token.
        think_s = (think_last or 0.0) if reason_chunks else 0.0
        raw = "".join(parts)
        # When a clause-first streaming TTS could have started: the first chunk and
        # its closing boundary have arrived. Computed on the raw stream, since the
        # chunk's position is measured in streamed characters.
        first_chunk_s = first_chunk_arrival(arrivals, first_chunk_span(raw))
        return {
            "text": raw.strip(),
            "ttft": ttft,
            "first_chunk_s": first_chunk_s,
            "total": total,
            "chunks": chunks,
            "completion_tokens": completion_tokens or chunks,
            "reason_chunks": reason_chunks,
            "think_s": think_s,
        }


# ─────────────────────────────────────────── per-turn record ──────────────


@dataclass
class TurnResult:
    id: int
    ground_truth: str
    transcript: str = ""
    reply: str = ""
    input_audio_s: float = 0.0
    output_audio_s: float = 0.0
    stt_s: Optional[float] = None
    llm_ttft_s: Optional[float] = None
    llm_first_chunk_s: Optional[float] = None
    llm_total_s: Optional[float] = None
    tts_first_s: Optional[float] = None
    tts_full_s: Optional[float] = None
    completion_tokens: int = 0
    reason_chunks: int = 0
    think_s: float = 0.0
    wer: Optional[float] = None
    stt_rtf: Optional[float] = None
    tts_rtf: Optional[float] = None
    llm_tok_s: Optional[float] = None
    first_audio_latency_s: Optional[float] = None
    turn_wall_s: Optional[float] = None
    topical_ok: Optional[bool] = None
    errors: list[str] = field(default_factory = list)


# ─────────────────────────────────────────── the benchmark ────────────────


def _mirror_to_downloads(turn_id: int, wav: bytes) -> None:
    """Best-effort copy of an input fixture into ~/Downloads for easy listening."""
    try:
        DOWNLOADS.mkdir(parents = True, exist_ok = True)
        (DOWNLOADS / f"turn_{turn_id}.wav").write_bytes(wav)
    except OSError:
        pass


class FixtureMismatch(RuntimeError):
    """A hand-supplied recording no longer matches the utterance it stands for."""


def fixture_paths(turn_id: int) -> tuple[Path, Path]:
    """The cached wav and its sidecar, which pins the wav to the utterance text."""
    return FIXTURES / f"turn_{turn_id}.wav", FIXTURES / f"turn_{turn_id}.json"


def _read_fixture_meta(meta_path: Path) -> Optional[dict]:
    try:
        meta = json.loads(meta_path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and isinstance(meta.get("text"), str) else None


def _write_fixture_meta(meta_path: Path, text: str, source: str) -> None:
    meta_path.write_text(json.dumps({"text": text, "source": source}), encoding = "utf-8")


def ensure_fixture(client: StudioClient, turn: dict) -> tuple[bytes, float, bool]:
    """Return (wav_bytes, duration_s, generated?). Prefer a real recording on disk.

    The cache is keyed on the turn id, so the sidecar records which utterance the
    wav was made for: a synthesized fixture whose text changed (another
    --conversation, or an edited turn) is re-synthesized; a hand-supplied recording
    is never overwritten and instead raises FixtureMismatch."""
    FIXTURES.mkdir(parents = True, exist_ok = True)
    wav_path, meta_path = fixture_paths(turn["id"])
    text = turn["text"]
    if wav_path.exists():
        meta = _read_fixture_meta(meta_path)
        if meta is None:
            # Dropped in by hand with no sidecar: adopt it, pinned to the current text.
            _write_fixture_meta(meta_path, text, "supplied")
            meta = {"text": text, "source": "supplied"}
        if meta["text"] == text:
            wav = wav_path.read_bytes()
            _mirror_to_downloads(turn["id"], wav)
            return wav, wav_duration_seconds(wav), False
        if meta.get("source") == "supplied":
            raise FixtureMismatch(
                f"{wav_path.name} was supplied for {meta['text']!r} but turn {turn['id']} "
                f"now reads {text!r}; replace the recording, or delete it to re-synthesize"
            )
        print(f"    (utterance for turn {turn['id']} changed; re-synthesizing its fixture)")
    # No recording supplied: synthesize the utterance with the loaded TTS voice
    # (Orpheus) and cache it, so STT input is byte-identical on every later run.
    wav, _ = client.speak(text)
    wav_path.write_bytes(wav)
    _write_fixture_meta(meta_path, text, "synthesized")
    _mirror_to_downloads(turn["id"], wav)
    return wav, wav_duration_seconds(wav), True


def run_turn(client: StudioClient, turn: dict, messages: list[dict], args) -> TurnResult:
    res = TurnResult(id = turn["id"], ground_truth = turn["text"])

    # 1) STT
    try:
        wav, dur, generated = ensure_fixture(client, turn)
        res.input_audio_s = dur
        if generated:
            print(f"    (synthesized input fixture for turn {turn['id']})")
        res.transcript, res.stt_s = client.transcribe(wav, args.stt_model)
        res.wer = word_error_rate(turn["text"], res.transcript)
        if res.stt_s and res.stt_s > 0:
            res.stt_rtf = res.input_audio_s / res.stt_s
    except Exception as e:  # noqa: BLE001 - record and continue
        res.errors.append(f"stt: {e}")
        res.transcript = turn["text"]  # fall back so the LLM stage still runs

    # 2) LLM (multi-turn: use the real transcript as the user message)
    messages.append({"role": "user", "content": res.transcript or turn["text"]})
    try:
        out = client.chat_stream(
            model = args.model,
            messages = messages,
            seed = args.seed,
            temperature = args.temperature,
            max_tokens = args.max_tokens,
            enable_thinking = args.think,
        )
        res.reply = out["text"]
        res.llm_ttft_s = out["ttft"]
        res.llm_first_chunk_s = out["first_chunk_s"]
        res.llm_total_s = out["total"]
        res.completion_tokens = out["completion_tokens"]
        res.reason_chunks = out.get("reason_chunks", 0)
        res.think_s = out.get("think_s", 0.0)
        if res.llm_total_s and res.llm_total_s > 0 and res.completion_tokens:
            res.llm_tok_s = res.completion_tokens / res.llm_total_s
        messages.append({"role": "assistant", "content": res.reply})
        low = res.reply.lower()
        res.topical_ok = any(k in low for k in turn.get("expect_any", []))
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"llm: {e}")

    # 3) TTS — first CHUNK (opening clause = what the user hears first, gates
    # first-audio latency) then the full reply for throughput.
    if res.reply:
        try:
            _, res.tts_first_s = client.speak(first_chunk(res.reply))
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"tts_first: {e}")
        try:
            wav_out, res.tts_full_s = client.speak(res.reply)
            res.output_audio_s = wav_duration_seconds(wav_out)
            if res.tts_full_s and res.tts_full_s > 0:
                res.tts_rtf = res.output_audio_s / res.tts_full_s
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"tts_full: {e}")

    # Derived latencies. First audio is charged for the LLM until the first
    # synthesizable chunk is complete, not just its first token: speak() cannot
    # start on a token, and the tokens between the two are real waiting.
    if None not in (res.stt_s, res.llm_first_chunk_s, res.tts_first_s):
        res.first_audio_latency_s = res.stt_s + res.llm_first_chunk_s + res.tts_first_s
    if None not in (res.stt_s, res.llm_total_s, res.tts_full_s):
        res.turn_wall_s = res.stt_s + res.llm_total_s + res.tts_full_s
    return res


def warmup(client: StudioClient, args) -> dict:
    """One throwaway hit per stage to pay lazy-load / MIOpen-tune costs up front.

    Returns the cold times so the first-call penalty (a real latency the user
    feels) is visible but kept out of the steady-state means."""
    cold = {"stt_s": None, "llm_s": None, "tts_s": None}
    print("  warmup (cold-start costs, not counted in steady-state)...")
    try:
        # Tiny silence clip so STT loads Whisper + tunes kernels.
        import numpy as np

        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype = "float32"), 16000, format = "WAV")
        _, cold["stt_s"] = client.transcribe(buf.getvalue(), args.stt_model)
    except Exception as e:  # noqa: BLE001
        print(f"    stt warmup failed: {e}")
    try:
        out = client.chat_stream(
            args.model, [{"role": "user", "content": "Say hi."}], args.seed, 0.0, 16, args.think
        )
        cold["llm_s"] = out["total"]
    except Exception as e:  # noqa: BLE001
        print(f"    llm warmup failed: {e}")
    try:
        _, cold["tts_s"] = client.speak("Warming up.")
    except Exception as e:  # noqa: BLE001
        print(f"    tts warmup failed: {e}")
    print(
        f"    cold: STT {_fmt(cold['stt_s'])}  LLM {_fmt(cold['llm_s'])}  "
        f"TTS {_fmt(cold['tts_s'])}"
    )
    return cold


def determinism_check(client: StudioClient, convo: dict, args) -> dict:
    """Re-run turn 1's LLM twice with the same seed; replies must be identical."""
    system = convo.get("system")
    turn1 = convo["turns"][0]["text"]
    base = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": turn1}
    ]
    try:
        a = client.chat_stream(
            args.model, base, args.seed, args.temperature, args.max_tokens, args.think
        )
        b = client.chat_stream(
            args.model, base, args.seed, args.temperature, args.max_tokens, args.think
        )
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "identical": None, "error": str(e)}
    identical = a["text"] == b["text"]
    return {"ran": True, "identical": identical, "sample": a["text"][:160]}


def prepare_fixtures(client: StudioClient, convo: dict) -> None:
    """Materialize every turn's input wav before any measured pass, so a first run
    does not synthesize (and thereby warm) TTS in the middle of a timed turn."""
    for turn in convo["turns"]:
        _, _, generated = ensure_fixture(client, turn)
        if generated:
            print(f"  synthesized input fixture for turn {turn['id']}")


def run_once(client: StudioClient, convo: dict, args) -> list[TurnResult]:
    system = convo.get("system")
    messages: list[dict] = [{"role": "system", "content": system}] if system else []
    results = []
    for turn in convo["turns"]:
        print(f"  turn {turn['id']}: {turn['text']!r}")
        res = run_turn(client, turn, messages, args)
        results.append(res)
        print(
            f"    stt {_fmt(res.stt_s)} (rtf {_fmt(res.stt_rtf, 'x', 2)})  "
            f"llm ttft {_fmt(res.llm_ttft_s)} chunk {_fmt(res.llm_first_chunk_s)} "
            f"tot {_fmt(res.llm_total_s)} ({_fmt(res.llm_tok_s, ' t/s', 1)})  "
            f"tts1 {_fmt(res.tts_first_s)} full {_fmt(res.tts_full_s)} "
            f"(rtf {_fmt(res.tts_rtf, 'x', 2)})  "
            f"=> first-audio {_fmt(res.first_audio_latency_s)}"
        )
        if res.transcript:
            print(f"      heard : {res.transcript!r}  (WER {_fmt(res.wer, '', 3)})")
        if res.reason_chunks:
            print(
                f"      THOUGHT {res.reason_chunks} chunks (~{_fmt(res.think_s)}) before "
                f"any spoken word -- pure first-audio latency"
            )
        if res.reply:
            flag = "ok" if res.topical_ok else "OFF-TOPIC?"
            print(f"      reply : {res.reply[:120]!r} [{flag}]")
        for err in res.errors:
            print(f"      ERROR {err}")
    return results


# ─────────────────────────────────────────── aggregation / report ─────────


def _median(vals: list[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return statistics.median(xs) if xs else None


_REQUIRED_METRICS = (
    "stt_s", "llm_ttft_s", "llm_first_chunk_s", "llm_total_s", "tts_first_s", "tts_full_s"
)


def incomplete_turns(passes: list[list[TurnResult]]) -> list[dict]:
    """Every (pass, turn) that errored or is missing a stage timing.

    Such a turn would otherwise drop out of the medians and totals silently, so a
    run with a broken stage could read *faster* than its baseline."""
    gaps = []
    for p_idx, turns in enumerate(passes, 1):
        for r in turns:
            missing = [k for k in _REQUIRED_METRICS if getattr(r, k) is None]
            if r.errors or missing:
                gaps.append(
                    {"pass": p_idx, "turn": r.id, "errors": list(r.errors), "missing": missing}
                )
    return gaps


def summarize(passes: list[list[TurnResult]]) -> dict:
    """Median-over-passes per turn, then sum/mean across the conversation.

    ``complete`` is False when any (pass, turn) errored or lacks a stage timing. The
    totals then cover only what ran and must not be compared to a baseline; main
    exits 1 for such a run."""
    if not passes:
        raise ValueError("summarize() needs at least one measured pass")
    n_turns = len(passes[0])
    per_turn = []
    for i in range(n_turns):
        rs = [p[i] for p in passes]
        per_turn.append(
            {
                "id": rs[0].id,
                "stt_s": _median([r.stt_s for r in rs]),
                "llm_ttft_s": _median([r.llm_ttft_s for r in rs]),
                "llm_first_chunk_s": _median([r.llm_first_chunk_s for r in rs]),
                "llm_total_s": _median([r.llm_total_s for r in rs]),
                "tts_first_s": _median([r.tts_first_s for r in rs]),
                "tts_full_s": _median([r.tts_full_s for r in rs]),
                "first_audio_latency_s": _median([r.first_audio_latency_s for r in rs]),
                "turn_wall_s": _median([r.turn_wall_s for r in rs]),
                "stt_rtf": _median([r.stt_rtf for r in rs]),
                "tts_rtf": _median([r.tts_rtf for r in rs]),
                "llm_tok_s": _median([r.llm_tok_s for r in rs]),
                "wer": _median([r.wer for r in rs]),
                "topical_ok": all(r.topical_ok for r in rs if r.topical_ok is not None),
            }
        )

    def total(key: str) -> Optional[float]:
        xs = [t[key] for t in per_turn if t[key] is not None]
        return sum(xs) if xs else None

    def mean(key: str) -> Optional[float]:
        xs = [t[key] for t in per_turn if t[key] is not None]
        return statistics.mean(xs) if xs else None

    gaps = incomplete_turns(passes)
    return {
        "complete": not gaps,
        "incomplete": gaps,
        "per_turn": per_turn,
        "totals": {
            # The headline: sum of true elapsed on the realtime critical path.
            "first_audio_latency_s": total("first_audio_latency_s"),
            "pipeline_wall_s": total("turn_wall_s"),
            "stt_s": total("stt_s"),
            "llm_total_s": total("llm_total_s"),
            "tts_full_s": total("tts_full_s"),
        },
        "means": {
            "first_audio_latency_s": mean("first_audio_latency_s"),
            "stt_rtf": mean("stt_rtf"),
            "tts_rtf": mean("tts_rtf"),
            "llm_tok_s": mean("llm_tok_s"),
            "llm_ttft_s": mean("llm_ttft_s"),
            "llm_first_chunk_s": mean("llm_first_chunk_s"),
            "wer": mean("wer"),
        },
    }


def print_report(summary: dict, meta: dict) -> None:
    print("\n" + "=" * 84)
    print("VOICE PIPELINE LATENCY  (median over passes; lower time / higher rtf better)")
    print("=" * 84)
    print(f"  chat model : {meta.get('model')}")
    print(f"  tts voice  : {meta.get('tts_voice')}")
    print(f"  stt model  : {meta.get('stt_model') or '(server default)'}")
    print(f"  seed={meta.get('seed')} temp={meta.get('temperature')} passes={meta.get('repeats')} "
          f"thinking={'ON' if meta.get('think') else 'off'}")
    print("-" * 84)
    hdr = (
        f"{'turn':>4}  {'stt':>7} {'sttRTF':>6}  {'ttft':>7} {'chunk':>7} {'llm':>7} {'tok/s':>6}  "
        f"{'tts1':>7} {'ttsRTF':>6}  {'1st-audio':>9}  wer"
    )
    print(hdr)
    for t in summary["per_turn"]:
        print(
            f"{t['id']:>4}  {_fmt(t['stt_s']):>7} {_fmt(t['stt_rtf'],'',2):>6}  "
            f"{_fmt(t['llm_ttft_s']):>7} {_fmt(t['llm_first_chunk_s']):>7} "
            f"{_fmt(t['llm_total_s']):>7} {_fmt(t['llm_tok_s'],'',1):>6}  "
            f"{_fmt(t['tts_first_s']):>7} {_fmt(t['tts_rtf'],'',2):>6}  "
            f"{_fmt(t['first_audio_latency_s']):>9}  {_fmt(t['wer'],'',3)}"
        )
    print("-" * 84)
    tot, mean = summary["totals"], summary["means"]
    print(f"  TOTAL first-audio latency (drive this down) : {_fmt(tot['first_audio_latency_s'])}")
    print(f"  TOTAL full-pipeline wall                    : {_fmt(tot['pipeline_wall_s'])}")
    print(f"  mean first-audio latency / turn             : {_fmt(mean['first_audio_latency_s'])}")
    print(
        f"  mean STT rtf {_fmt(mean['stt_rtf'],'x',2)}   mean TTS rtf {_fmt(mean['tts_rtf'],'x',2)}"
        f"   mean LLM {_fmt(mean['llm_tok_s'],' t/s',1)}   mean WER {_fmt(mean['wer'],'',3)}"
    )
    det = meta.get("determinism", {})
    if det.get("ran"):
        print(
            f"  determinism (turn1 x2, same seed)           : "
            f"{'IDENTICAL [ok]' if det.get('identical') else 'DIFFERENT [FAIL]'}"
        )
    cold = meta.get("cold", {})
    if any(cold.values()):
        print(f"  cold-start (first call)  STT {_fmt(cold.get('stt_s'))} "
              f"LLM {_fmt(cold.get('llm_s'))} TTS {_fmt(cold.get('tts_s'))}")
    if not summary.get("complete", True):
        print("  RUN INCOMPLETE [FAIL]: the totals above skip the failed turns below and "
              "are not comparable to a baseline")
        for gap in summary.get("incomplete", []):
            detail = "; ".join(gap["errors"]) or f"missing {', '.join(gap['missing'])}"
            print(f"    pass {gap['pass']} turn {gap['turn']}: {detail}")
    print("=" * 84)


def diff_baseline(summary: dict, baseline_path: Path) -> None:
    if not summary.get("complete", True):
        print(f"\n(skipping diff vs {baseline_path.name}: this run is incomplete)")
        return
    try:
        base = json.loads(baseline_path.read_text(encoding = "utf-8"))["summary"]
    except Exception as e:  # noqa: BLE001
        print(f"\n(could not read baseline {baseline_path}: {e})")
        return
    if not base.get("complete", True):
        print(f"\n(skipping diff: baseline {baseline_path.name} was itself incomplete)")
        return
    print("\n" + "-" * 78)
    print(f"DIFF vs baseline {baseline_path.name}   (negative = faster)")
    print("-" * 78)
    pairs = [
        ("totals", "first_audio_latency_s", "total first-audio latency"),
        ("totals", "pipeline_wall_s", "total pipeline wall"),
        ("means", "first_audio_latency_s", "mean first-audio/turn"),
        ("means", "llm_ttft_s", "mean LLM ttft"),
        ("means", "llm_first_chunk_s", "mean LLM first chunk"),
    ]
    for grp, key, label in pairs:
        now = summary[grp].get(key)
        was = base.get(grp, {}).get(key)
        if now is None or was is None:
            continue
        delta = now - was
        pct = (delta / was * 100) if was else 0.0
        arrow = "faster" if delta < 0 else "slower"
        print(f"  {label:<28} {was:.3f}s -> {now:.3f}s  ({delta:+.3f}s, {pct:+.1f}% {arrow})")
    # RTF / throughput: higher is better
    for grp, key, label in [
        ("means", "stt_rtf", "mean STT rtf"),
        ("means", "tts_rtf", "mean TTS rtf"),
        ("means", "llm_tok_s", "mean LLM tok/s"),
    ]:
        now = summary[grp].get(key)
        was = base.get(grp, {}).get(key)
        if now is None or was is None:
            continue
        print(f"  {label:<28} {was:.2f} -> {now:.2f}  ({now - was:+.2f})")
    print("-" * 78)


# ─────────────────────────────────────────── main ─────────────────────────


def resolve_tts_route(args, status: dict, voice_status: dict) -> tuple[Optional[dict], str, str]:
    """Decide where /v1/audio/speech will land: (provider fields or None, voice
    label, error). A non-empty error means the run cannot produce valid numbers.

    Studio serves one resident model per slot. On main that slot is the only one:
    /v1/audio/speech is reload-only and returns 400 unless the resident model is a
    TTS model, while /v1/chat/completions against a resident TTS model returns
    speech, not text. A chat model and a local voice therefore coexist only on a
    backend with a separate voice slot (PR #10373's /api/inference/voice/*), which
    voice_status reports as ``loaded``. Otherwise TTS needs a saved external
    connection (--tts-provider-id), which the speech route proxies without
    touching the resident model."""
    if status.get("is_audio"):
        return (
            None,
            "",
            f"The resident model {status.get('active_model')!r} is a TTS model, so "
            "/v1/chat/completions would return speech instead of text. Load the chat "
            "model in the main slot (and the voice beside it, or use --tts-provider-id).",
        )
    if args.tts_provider_id:
        if not (args.tts_model and args.tts_voice):
            return None, "", "--tts-provider-id needs --tts-model and --tts-voice as well."
        provider = {
            "provider_id": args.tts_provider_id,
            "model": args.tts_model,
            "voice": args.tts_voice,
        }
        label = f"{args.tts_model}/{args.tts_voice} via connection {args.tts_provider_id}"
        return provider, label, ""
    if voice_status.get("loaded"):
        return None, str(voice_status.get("model") or "(voice slot)"), ""
    return (
        None,
        "",
        "No TTS voice is loaded beside the chat model, so every /v1/audio/speech call "
        "would return 400 (this Studio serves one resident model, and the speech route "
        "never switches it). Either load a voice in the voice slot (Speak-with picker; "
        "needs the conversation-mode backend, PR #10373) or route TTS to a saved "
        "connection with --tts-provider-id/--tts-model/--tts-voice.",
    )


def positive_int(value: str) -> int:
    """argparse type: a count that must be >= 1 (``--repeats 0`` would measure nothing)."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return n


def get_token(args) -> str:
    if args.token:
        return args.token
    env = os.environ.get("UNSLOTH_BENCH_TOKEN")
    if env:
        return env
    sys.path.insert(0, str(HERE))
    import mint_token

    return mint_token.get_token()


def main() -> int:
    ap = argparse.ArgumentParser(description = "Voice pipeline latency benchmark")
    ap.add_argument("--base-url", default = DEFAULT_BASE_URL)
    ap.add_argument("--token", default = None, help = "Bearer token / API key (else auto-mint)")
    ap.add_argument("--conversation", default = str(HERE / "conversation.json"))
    ap.add_argument("--model", default = None, help = "chat model id (default: server's active model)")
    ap.add_argument("--stt-model", default = None, help = "Whisper model id (default: server default)")
    ap.add_argument(
        "--tts-provider-id",
        default = None,
        help = "Saved external TTS connection id: route every /v1/audio/speech call there "
        "instead of Studio's loaded model (needs --tts-model and --tts-voice). Measures a "
        "remote round trip, but works on a Studio that serves one resident model.",
    )
    ap.add_argument("--tts-model", default = None, help = "model id for --tts-provider-id")
    ap.add_argument("--tts-voice", default = None, help = "voice name for --tts-provider-id")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--temperature", type = float, default = 0.0)
    ap.add_argument("--max-tokens", type = int, default = 200)
    ap.add_argument(
        "--think",
        action = "store_true",
        help = "Let the chat model reason before replying (chat_template_kwargs."
        "enable_thinking=true). OFF by default: realtime voice wants the spoken "
        "answer immediately, and reasoning is pure first-audio latency.",
    )
    ap.add_argument(
        "--repeats", type = positive_int, default = 1, help = "measured passes (>= 1); median reported"
    )
    ap.add_argument("--no-warmup", action = "store_true")
    ap.add_argument("--no-determinism", action = "store_true")
    ap.add_argument("--baseline", default = None, help = "prior report JSON to diff against")
    ap.add_argument("--out", default = None, help = "report JSON path (default: reports/<ts>.json)")
    args = ap.parse_args()

    # Windows consoles default to cp1252; keep any stray non-ASCII from crashing output.
    try:
        sys.stdout.reconfigure(encoding = "utf-8")
    except (AttributeError, ValueError):
        pass

    convo = json.loads(Path(args.conversation).read_text(encoding = "utf-8"))
    token = get_token(args)
    client = StudioClient(args.base_url, token, args.seed)

    # Confirm the server is up and something is loaded to talk to.
    try:
        st = client.status()
    except requests.RequestException as e:
        print(f"Cannot reach Studio at {args.base_url}: {e}")
        print("Start Studio (default port 8888) and load a chat model + a TTS voice.")
        return 2
    if not args.model:
        args.model = st.get("active_model")
    if not args.model:
        print("No chat model loaded. Load one in the Studio UI, then re-run.")
        return 2
    # Refuse up front rather than 400 on every speak(): see resolve_tts_route.
    client.tts_provider, tts_voice, route_error = resolve_tts_route(
        args, st, client.voice_status()
    )
    if route_error:
        print(route_error)
        return 2

    print(f"Studio {args.base_url}  |  chat={args.model}  |  voice={tts_voice}")

    # Order matters for what "cold" means: warmup (optional) first, then the input
    # fixtures (a first run synthesizes them, which warms TTS), then the measured
    # passes, and only then the determinism check, whose two extra LLM generations
    # would otherwise pre-warm pass 1 even under --no-warmup.
    cold = warmup(client, args) if not args.no_warmup else {}
    try:
        prepare_fixtures(client, convo)
    except Exception as e:  # noqa: BLE001 - nothing measured yet; refuse to start
        print(f"Cannot prepare input fixtures: {e}")
        return 2

    passes = []
    for p in range(args.repeats):
        print(f"\n-- pass {p + 1}/{args.repeats} --")
        passes.append(run_once(client, convo, args))

    det = {} if args.no_determinism else determinism_check(client, convo, args)

    summary = summarize(passes)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "stt_model": args.stt_model,
        "tts_voice": tts_voice,
        "tts_provider_id": args.tts_provider_id,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "think": args.think,
        "cold": cold,
        "determinism": det,
    }
    print_report(summary, meta)
    if args.baseline:
        diff_baseline(summary, Path(args.baseline))

    REPORTS.mkdir(parents = True, exist_ok = True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else REPORTS / f"bench_{stamp}.json"
    payload = {
        "meta": meta,
        "summary": summary,
        "passes": [[asdict(r) for r in p] for p in passes],
    }
    out.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    (REPORTS / "latest.json").write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    print(f"\nwrote {out}\n      {REPORTS / 'latest.json'}")
    # The report is always written (it is the evidence), but a run whose measurements
    # are not all present, or whose seeded LLM was not reproducible, must not pass.
    rc = 0
    if not summary["complete"]:
        print("FAIL: run incomplete (see the turns listed above)")
        rc = 1
    if det and not (det.get("ran") and det.get("identical")):
        why = det.get("error") or "the two seeded replies differ"
        print(f"FAIL: determinism check did not pass: {why}")
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
