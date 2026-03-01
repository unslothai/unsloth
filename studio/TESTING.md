# Testing Matrix: Audio, Text & VLM

## Text Model Tests

| # | Test | Model | Steps | Expected |
|---|------|-------|-------|----------|
| 1 | **Text inference — basic chat** | `unsloth/Qwen3-4B` or any text model | Load model → send "Hello, how are you?" | Streaming text response, no errors |
| 2 | **Text inference — system prompt** | Any text model | Set system prompt to "You are a pirate" → send "Tell me about the ocean" | Response in pirate persona |
| 3 | **Text training — LoRA** | Any text model | Load model → pick Alpaca dataset → set max_steps=10 → start training | Training completes, checkpoint saved, loss decreases |
| 4 | **Text compare mode** | Any text model with trained LoRA | Open compare view → send message | Both Base and LoRA columns respond, responses differ |

## VLM (Vision) Tests

| # | Test | Model | Steps | Expected |
|---|------|-------|-------|----------|
| 5 | **VLM inference — image description** | `unsloth/Llama-3.2-11B-Vision` or Gemma-3 vision | Load → attach image via paperclip → "What's in this image?" | Describes the image content accurately |
| 6 | **VLM inference — text only (no image)** | Same VLM | Send text message without image | Normal text response (no crash) |
| 7 | **VLM training — vision LoRA** | Any VLM | Load → pick image-text dataset → check finetune_vision_layers is ON → train max_steps=10 | Training completes with vision+language LoRA |
| 8 | **VLM compare mode with image** | VLM with trained LoRA | Open compare view → upload image → send | Both panels describe the image, LoRA panel should differ |
| 9 | **VLM dataset mapping** | Any VLM | Pick a dataset that needs manual column mapping | Mapping card shows correctly, VLM-specific labels appear |

## Audio TTS (Text-to-Speech) Tests

| # | Test | Model | Steps | Expected |
|---|------|-------|-------|----------|
| 10 | **TTS inference — SNAC/Orpheus** | `canopylabs/orpheus-3b-0.1-ft` | Load → send text message | AudioPlayer renders with playable WAV audio |
| 11 | **TTS inference — SparkTTS (BiCodec)** | `SparkAudio/Spark-TTS-0.5B` | Load → send text "Hello world" | AudioPlayer with synthesized speech (not raw bicodec tokens) |
| 12 | **TTS inference — OuteTTS (DAC)** | `OuteAI/Llama-OuteTTS-1.0-1B` | Load → send text | AudioPlayer with audio output |
| 13 | **TTS inference — CSM/Sesame** | `sesame/csm-1b` | Load → send text | AudioPlayer with audio output |
| 14 | **TTS training — LoRA** | Any TTS model (e.g., Orpheus) | Load → pick audio dataset → train max_steps=10 | Training completes, checkpoint saved |
| 15 | **TTS compare mode** | TTS with LoRA adapter | Open compare → send text | Both panels show AudioPlayer, LoRA should sound different |
| 16 | **SparkTTS LoRA inference** | SparkTTS with trained adapter | Load LoRA checkpoint → send text | Plays audio without 404 error (bicodec loads from local path) |

## Audio ASR (Speech-to-Text) Tests

| # | Test | Model | Steps | Expected |
|---|------|-------|-------|----------|
| 17 | **Whisper inference — audio transcription** | `unsloth/whisper-large-v3` | Load → upload audio via headphones button → send | Transcribed text appears (no "only accepts audio" error) |
| 18 | **Whisper inference — no audio error** | `unsloth/whisper-large-v3` | Load → send text without audio | Clear error: "Whisper models require audio input" |
| 19 | **Gemma 3n inference — audio ASR** | `unsloth/gemma-3n-E4B-it` | Load → upload audio → "Transcribe this audio" | Accurate transcription, uses greedy decoding |
| 20 | **Gemma 3n inference — text only** | `unsloth/gemma-3n-E4B-it` | Load → send text without audio | Normal text response (model is also a text/vision model) |
| 21 | **Gemma 3n inference — image** | `unsloth/gemma-3n-E4B-it` | Load → attach image → "What's in this image?" | Describes image (Gemma 3n supports vision too) |
| 22 | **Gemma 3n training — audio dataset** | `unsloth/gemma-3n-E4B-it` | Load → pick speech dataset → train max_steps=10 | Training completes, audio mapping card shows "audio and text" |
| 23 | **Audio chip in user message** | Any ASR model | Upload audio → send message | Audio filename chip appears in user message bubble |

## Cross-Cutting / Edge Case Tests

| # | Test | Model | Steps | Expected |
|---|------|-------|-------|----------|
| 24 | **Model switch — text to TTS** | Text model → TTS model | Load text model → chat → switch to TTS → chat | First gives text, second gives AudioPlayer — no leftover state |
| 25 | **Model switch — TTS to ASR** | TTS → Whisper or Gemma 3n | Load TTS → generate audio → switch to ASR → upload audio | TTS gives audio, ASR gives text — clean transition |
| 26 | **Model switch — VLM to text** | VLM → text model | Load VLM → send image → switch to text model → send text | No vision errors on text model, image attachment ignored |
| 27 | **Abort mid-generation** | Any streaming model | Send message → click stop button mid-stream | Generation stops cleanly, partial response visible, no crash |
| 28 | **Large audio file rejection** | Any ASR model | Try uploading audio > 50MB | Upload rejected (MAX_AUDIO_SIZE), no crash |
| 29 | **No model loaded error** | No model | Open chat → send message | Toast: "No model loaded — Pick model in top bar, then retry" |
| 30 | **Training then inference** | Any model | Train LoRA → load checkpoint → chat | Trained checkpoint responds (different from base) |

## Quick Smoke Test Order (prioritized)

If running a fast subset, do these 10 in order:

1. **#1** — Text basic chat (sanity check)
2. **#5** — VLM image description
3. **#10** — TTS audio generation (SNAC)
4. **#19** — Gemma 3n audio ASR
5. **#17** — Whisper transcription
6. **#4** — Text compare mode
7. **#3** — Text LoRA training
8. **#23** — Audio chip in user message
9. **#24** — Model switch text→TTS
10. **#29** — No model loaded error
