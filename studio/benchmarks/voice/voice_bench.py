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
end of your speech to the first audio coming back:

    first_audio_latency = stt_seconds + llm_time_to_first_token + tts_first_sentence_seconds

Everything is deterministic: fixed seed, temperature 0 (greedy), and the input
audio is cached to disk on first run so every later run feeds identical bytes.
A determinism check re-runs the first turn's LLM and asserts an identical reply.

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
            cur.append(
                prev[j - 1] if r == h else 1 + min(prev[j], cur[j - 1], prev[j - 1])
            )
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


def _fmt(x: Optional[float], unit: str = "s", nd: int = 3) -> str:
    return "  n/a " if x is None else f"{x:.{nd}f}{unit}"


# ─────────────────────────────────────────── HTTP client ──────────────────


class StudioClient:
    def __init__(self, base_url: str, token: str, timeout: float = 180.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
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
            f"{self.base}/v1/audio/transcribe",
            files = files,
            data = data,
            timeout = self.timeout,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return r.json().get("text", ""), elapsed

    def speak(self, text: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        r = self.s.post(
            f"{self.base}/v1/audio/speech",
            json = {"input": text},
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
        ttft: Optional[float] = None       # time to first spoken (content) token
        think_first: Optional[float] = None
        think_last: Optional[float] = None
        chunks = 0
        reason_chunks = 0
        completion_tokens: Optional[int] = None
        parts: list[str] = []
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
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                        parts.append(piece)
        total = time.perf_counter() - t0
        # Dead time spent reasoning before the first spoken token.
        think_s = (think_last or 0.0) if reason_chunks else 0.0
        return {
            "text": "".join(parts).strip(),
            "ttft": ttft,
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


def ensure_fixture(client: StudioClient, turn: dict) -> tuple[bytes, float, bool]:
    """Return (wav_bytes, duration_s, generated?). Prefer a real recording on disk."""
    FIXTURES.mkdir(parents = True, exist_ok = True)
    path = FIXTURES / f"turn_{turn['id']}.wav"
    if path.exists():
        wav = path.read_bytes()
        _mirror_to_downloads(turn["id"], wav)
        return wav, wav_duration_seconds(wav), False
    # No recording supplied: synthesize the utterance with the loaded TTS voice
    # (Orpheus) and cache it, so STT input is byte-identical on every later run.
    wav, _ = client.speak(turn["text"])
    path.write_bytes(wav)
    _mirror_to_downloads(turn["id"], wav)
    return wav, wav_duration_seconds(wav), True


def run_turn(
    client: StudioClient,
    turn: dict,
    messages: list[dict],
    args,
) -> TurnResult:
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

    # Derived latencies
    if None not in (res.stt_s, res.llm_ttft_s, res.tts_first_s):
        res.first_audio_latency_s = res.stt_s + res.llm_ttft_s + res.tts_first_s
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
        a = client.chat_stream(args.model, base, args.seed, args.temperature, args.max_tokens, args.think)
        b = client.chat_stream(args.model, base, args.seed, args.temperature, args.max_tokens, args.think)
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "identical": None, "error": str(e)}
    identical = a["text"] == b["text"]
    return {"ran": True, "identical": identical, "sample": a["text"][:160]}


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
            f"llm ttft {_fmt(res.llm_ttft_s)} tot {_fmt(res.llm_total_s)} "
            f"({_fmt(res.llm_tok_s, ' t/s', 1)})  "
            f"tts1 {_fmt(res.tts_first_s)} full {_fmt(res.tts_full_s)} "
            f"(rtf {_fmt(res.tts_rtf, 'x', 2)})  "
            f"=> first-audio {_fmt(res.first_audio_latency_s)}"
        )
        if res.transcript:
            print(f"      heard : {res.transcript!r}  (WER {_fmt(res.wer, '', 3)})")
        if res.reason_chunks:
            print(f"      THOUGHT {res.reason_chunks} chunks (~{_fmt(res.think_s)}) before "
                  f"any spoken word -- pure first-audio latency")
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


def summarize(passes: list[list[TurnResult]]) -> dict:
    """Median-over-passes per turn, then sum/mean across the conversation."""
    n_turns = len(passes[0])
    per_turn = []
    for i in range(n_turns):
        rs = [p[i] for p in passes]
        per_turn.append(
            {
                "id": rs[0].id,
                "stt_s": _median([r.stt_s for r in rs]),
                "llm_ttft_s": _median([r.llm_ttft_s for r in rs]),
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

    return {
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
            "wer": mean("wer"),
        },
    }


def print_report(summary: dict, meta: dict) -> None:
    print("\n" + "=" * 78)
    print("VOICE PIPELINE LATENCY  (median over passes; lower time / higher rtf better)")
    print("=" * 78)
    print(f"  chat model : {meta.get('model')}")
    print(f"  tts voice  : {meta.get('tts_voice')}")
    print(f"  stt model  : {meta.get('stt_model') or '(server default)'}")
    print(f"  seed={meta.get('seed')} temp={meta.get('temperature')} passes={meta.get('repeats')} "
          f"thinking={'ON' if meta.get('think') else 'off'}")
    print("-" * 78)
    hdr = (
        f"{'turn':>4}  {'stt':>7} {'sttRTF':>6}  {'ttft':>7} {'llm':>7} {'tok/s':>6}  "
        f"{'tts1':>7} {'ttsRTF':>6}  {'1st-audio':>9}  wer"
    )
    print(hdr)
    for t in summary["per_turn"]:
        print(
            f"{t['id']:>4}  {_fmt(t['stt_s']):>7} {_fmt(t['stt_rtf'],'',2):>6}  "
            f"{_fmt(t['llm_ttft_s']):>7} {_fmt(t['llm_total_s']):>7} "
            f"{_fmt(t['llm_tok_s'],'',1):>6}  "
            f"{_fmt(t['tts_first_s']):>7} {_fmt(t['tts_rtf'],'',2):>6}  "
            f"{_fmt(t['first_audio_latency_s']):>9}  {_fmt(t['wer'],'',3)}"
        )
    print("-" * 78)
    tot, mean = summary["totals"], summary["means"]
    print(f"  TOTAL first-audio latency (drive this down) : {_fmt(tot['first_audio_latency_s'])}")
    print(f"  TOTAL full-pipeline wall                    : {_fmt(tot['pipeline_wall_s'])}")
    print(f"  mean first-audio latency / turn             : {_fmt(mean['first_audio_latency_s'])}")
    print(f"  mean STT rtf {_fmt(mean['stt_rtf'],'x',2)}   mean TTS rtf {_fmt(mean['tts_rtf'],'x',2)}"
          f"   mean LLM {_fmt(mean['llm_tok_s'],' t/s',1)}   mean WER {_fmt(mean['wer'],'',3)}")
    det = meta.get("determinism", {})
    if det.get("ran"):
        print(f"  determinism (turn1 x2, same seed)           : "
              f"{'IDENTICAL [ok]' if det.get('identical') else 'DIFFERENT [FAIL]'}")
    cold = meta.get("cold", {})
    if any(cold.values()):
        print(f"  cold-start (first call)  STT {_fmt(cold.get('stt_s'))} "
              f"LLM {_fmt(cold.get('llm_s'))} TTS {_fmt(cold.get('tts_s'))}")
    print("=" * 78)


def diff_baseline(summary: dict, baseline_path: Path) -> None:
    try:
        base = json.loads(baseline_path.read_text(encoding = "utf-8"))["summary"]
    except Exception as e:  # noqa: BLE001
        print(f"\n(could not read baseline {baseline_path}: {e})")
        return
    print("\n" + "-" * 78)
    print(f"DIFF vs baseline {baseline_path.name}   (negative = faster)")
    print("-" * 78)
    pairs = [
        ("totals", "first_audio_latency_s", "total first-audio latency"),
        ("totals", "pipeline_wall_s", "total pipeline wall"),
        ("means", "first_audio_latency_s", "mean first-audio/turn"),
        ("means", "llm_ttft_s", "mean LLM ttft"),
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
    ap.add_argument("--repeats", type = int, default = 1, help = "measured passes; median reported")
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
    client = StudioClient(args.base_url, token)

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
    vst = client.voice_status()
    tts_voice = vst.get("model") or vst.get("active_model") or ("model-owned" if st.get("is_audio") else "(loaded voice)")

    print(f"Studio {args.base_url}  |  chat={args.model}  |  voice={tts_voice}")

    cold = warmup(client, args) if not args.no_warmup else {}
    det = {} if args.no_determinism else determinism_check(client, convo, args)

    passes = []
    for p in range(args.repeats):
        print(f"\n-- pass {p + 1}/{args.repeats} --")
        passes.append(run_once(client, convo, args))

    summary = summarize(passes)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "stt_model": args.stt_model,
        "tts_voice": tts_voice,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
