# Voice pipeline latency benchmark

A deterministic, scriptable speed test for the whole realtime voice loop:

```
input speech (wav)  --STT-->  transcript  --LLM-->  reply text  --TTS-->  reply speech
```

It fakes a 4-turn conversation against a **running Studio backend**, measures
where the time goes stage by stage, checks the answers aren't garbage, and
writes a JSON report you can diff against later to prove a change made things
faster (or catch a regression).

The scripted conversation (`conversation.json`):

1. "Hello, what is the weather today?"
2. "How about tomorrow then?"
3. "Can you make me a hotel trip plan to the Great Barrier Reef? I'm in New York now."
4. "How about the weather, would that cause issues?"

Turns 2 and 4 are deliberately context-dependent ("tomorrow", "that"), so the
benchmark also exercises multi-turn history, not just isolated prompts.

## What it measures

Per turn, per stage: the true wall-clock elapsed **and** a length-normalized
rate, so a longer utterance isn't unfairly counted as slower:

| metric | meaning | better |
| --- | --- | --- |
| `stt_s` / `stt_rtf` | Whisper transcribe time / `input_audio_s ÷ stt_s` | lower / higher |
| `llm_ttft_s` | LLM time-to-first-token | lower |
| `llm_first_chunk_s` | LLM time until the first synthesizable chunk (opening clause / short sentence, plus its closing boundary) has streamed in | lower |
| `llm_total_s` / `llm_tok_s` | full generation / `tokens ÷ time` | lower / higher |
| `tts_first_s` | synth time for the first CLAUSE (the opening chunk you hear first) | lower |
| `tts_full_s` / `tts_rtf` | synth time for the whole reply / `output_audio_s ÷ tts_s` | lower / higher |

`tts_first_s` measures the opening **clause** (split at the first comma/`;`/`:` when
the first sentence is long), not the whole first sentence, because a latency-tuned
streaming TTS should start audio on a short clause instead of waiting for a long
sentence. `reason_chunks` / `think_s` record whether the chat model reasoned before
speaking (see `--think`).

**Headline number — `first_audio_latency`:** the time from the end of your
speech to the first audio coming back, as a clause-first *streaming* pipeline
would deliver it:

```
first_audio_latency = stt_s + llm_first_chunk_s + tts_first_s
```

The LLM is charged until its first synthesizable chunk is complete, not just its
first token: speech cannot start on a token, and the tokens between the first one
and the end of the opening clause are real waiting. (The harness itself still
calls TTS after the whole reply has streamed, so `turn_wall_s` =
`stt_s + llm_total_s + tts_full_s` is the latency of the pipeline as implemented
today; `first_audio_latency` is the projection of what a streaming TTS would
achieve.) That's the realtime "feel". The summary prints the total across all
turns; that total is the thing to drive down.

## Accuracy / correctness

Latency is worthless if the answers get worse, so each run also reports:

- **WER** — word error rate of the transcript vs the ground-truth utterance
  (real STT accuracy; 0.000 = perfect).
- **Determinism** — turn 1's LLM is run twice with the same seed; the replies
  must be byte-identical (`IDENTICAL [ok]`). Guards against a change quietly
  turning on sampling. A `DIFFERENT [FAIL]` result fails the run (exit 1).
- **Completeness** — every stage of every turn in every pass must have produced
  a timing. A turn that hit an STT/LLM/TTS error is listed under
  `RUN INCOMPLETE [FAIL]`, its report carries `"complete": false`, the baseline
  diff is skipped (the totals would be missing that turn and could read *faster*),
  and the run exits 1.
- **Topical check** — a soft flag if a reply wandered completely off topic
  (keyword coverage; not a factual gate, the model has no live tools).
- The actual transcripts and replies are printed so you can eyeball them.

Exit status: `0` valid run, `1` run measured but invalid (incomplete or
determinism failed; the report is still written), `2` could not run (Studio
unreachable, no chat model, no usable TTS route).

## Determinism

Fixed `--seed 42`, `--temperature 0` (greedy) for the LLM, and the **input audio
is cached** to `audio_fixtures/turn_N.wav` on first run (and mirrored to
`~/Downloads/voice_bench_fixtures/`), so every later run feeds identical bytes.
A `turn_N.json` sidecar pins each wav to the utterance it was made for, so the
cache cannot serve stale audio when a turn's text changes (an edited
`conversation.json`, or another `--conversation` reusing the same turn ids): a
synthesized fixture is re-synthesized automatically. Delete a `turn_N.wav` to
regenerate it, or drop in your own real recording with that name to benchmark
against authentic speech instead of synthesized input; a supplied recording is
adopted (sidecar written on first use) and never overwritten, so if its text
later changes the run stops with `FixtureMismatch` until you replace or delete it.

The **LLM is fully deterministic** here (greedy + seed): turn 1 is run twice and
must come back `IDENTICAL`. The same `--seed` is sent as `seed` on every
`/v1/audio/speech` call (fixtures, warmup, and reply synthesis). It is best-effort:
on current `main` the speech route hands it to the transformers (non-GGUF) TTS
backends, while a GGUF voice such as Orpheus takes no seed there (a fixed default
seed for GGUF voices is part of the conversation-mode backend in PR #10373), and
an external TTS connection does not receive it at all. **TTS is not
bit-reproducible on GPU** even when seeded: Orpheus runs at temperature 0.6, and
GPU float reductions are non-deterministic, so near-tie tokens flip run to run.
TTS wall-clock also drifts a little with GPU load. So TTS is judged by *latency*
— reported as a median over `--repeats` and normalized by audio length
(`tts_rtf`) — not by exact bytes.

## Running it

Prereqs: Studio running (default port 8888), a **chat model** loaded, and a
**TTS voice** loaded (Speak-with picker). Use the Studio venv python so the
token bootstrap can import `auth.storage`:

```bash
PY="$HOME/.unsloth/studio/unsloth_studio/Scripts/python.exe"
cd studio/benchmarks/voice

"$PY" voice_bench.py                     # one measured pass + timestamped report
"$PY" voice_bench.py --repeats 3         # 3 passes, median reported (steadier)
"$PY" voice_bench.py --baseline reports/latest.json   # diff vs a previous run
```

First run auto-mints an internal API key into `.bench_token` (gitignored). The
first call to each stage pays a cold-start cost (Whisper load, MIOpen kernel
tuning, model warmup); that's shown separately as `cold-start` and kept out of
the steady-state means via a warmup pass.

A run proceeds in this order: warmup (unless `--no-warmup`), input-fixture
preparation (a first run synthesizes the four utterances here, before anything
is timed), the measured passes, then the determinism check (its two extra LLM
generations run *after* the passes so they cannot pre-warm pass 1). For a
genuinely cold first pass you therefore need pre-existing fixtures (any earlier
run leaves them in `audio_fixtures/`) plus `--no-warmup --no-determinism`.

Useful flags: `--think` (let the chat model reason before replying — OFF by
default, because for voice the chain-of-thought is pure first-audio latency you
wait through before hearing a word), `--model <id>` (default = server's active
model), `--stt-model`, `--max-tokens`, `--no-warmup`, `--no-determinism`,
`--out <path>`.

## Optimization workflow

1. Capture a baseline: `voice_bench.py --repeats 3`, keep `reports/latest.json`.
2. Make one change (e.g. smaller Whisper, streaming TTS, fewer max tokens).
3. Re-run with `--baseline reports/latest.json` and read the diff: negative
   seconds = faster; check WER / determinism / topical flags didn't regress.

### Landed ROCm latency wins (verify each here with `--baseline`)

- **MIOpen fast find-mode + persistent cache** (`main.py`) — `MIOPEN_FIND_MODE=2`
  and a persisted kernel cache under `~/.unsloth/studio/miopen`, so the one-time
  autotune is paid once ever, not per process. Watch `cold-start STT`.

The two Whisper-side wins these findings were measured with (fp16 on GPU, and a
warm-up call before the user speaks) lived in a `routes/audio.py` that this branch
no longer carries: Studio now transcribes through the STT sidecar behind
`POST /v1/audio/transcriptions`, which the harness targets. The STT numbers below
therefore describe the pipeline as it was measured, not necessarily what the
sidecar does today — re-run with `--baseline` before quoting them.

### Measured findings (gemma-4-E2B-it Q4 chat + orpheus-3b Q2_K_L voice, RX 9060 XT)

Baseline vs optimized, 4-turn conversation, `first_audio_latency` totalled:

| stage | finding |
| --- | --- |
| STT (Whisper base, GPU) | ~0.2 s/turn, **RTF ~15x** — already fast, not the bottleneck |
| LLM **thinking ON** | ~2.5–3.2 s of `reasoning_content` **before a single spoken word** — the chat model burns the token budget reasoning first |
| LLM **thinking OFF** | TTFT **~0.3 s** — a **−89.5%** cut in time-to-first-spoken-token |
| TTS (Orpheus) | **RTF ~1.08x** (synth ≈ audio length) — now the dominant cost; ~0.3–0.6 s of leading silence per clip is pure dead air |

Turning **thinking off** cut total first-audio latency **~21–43%** (variance is
mostly TTS + STT cold-start) with **no accuracy change** (WER 0.10, replies stay
on-topic, LLM determinism `IDENTICAL`). This is a **benchmark knob** (`--think`),
not a forced product behaviour: reasoning stays **user-controlled in the UI**. It's
here so you can quantify what the model's chain-of-thought costs the first-audio
feel and decide per use case.

Landed alongside: **leading-silence trim** on TTS output and a **fixed TTS seed**
(reproducible output on CPU; best-effort on GPU). `tts_first_s` measures the
opening clause as a projection of a future clause-first streaming TTS.

### Next optimizations to try (biggest lever first)

- **Stream TTS output** (decode + play SNAC frames as they generate) instead of
  the blocking `/audio/speech` — would drop first-audio from ~4 s to <1 s. This is
  the single biggest remaining win; TTS is ~90% of first-audio latency.
- **Faster / smaller Orpheus quant or a lighter TTS** — RTF ~1.08x means synth is
  barely real time; a faster voice model moves the whole floor.
- **Cap `max_tokens`** and system-prompt for short spoken replies — cuts
  `llm_total_s` and `tts_full_s` without hurting the first-audio feel.
- **Smaller / faster STT** only if WER holds — STT is already ~15x real time, so
  low priority.
