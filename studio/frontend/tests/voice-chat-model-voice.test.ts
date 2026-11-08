// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { joinWavBase64, pcmWavBase64 } from "../src/features/chat/utils/wav-chunks.ts";
import {
  coalescedTurnAudio,
  createTurnAudio,
  lastUserTurnHasAudio,
  voiceTurnAttachment,
  withoutAudioPlayer,
} from "../src/features/chat/voice/turn-audio.ts";
import {
  VOICE_CHAT_MODEL_VOICE,
  VOICE_DEFAULT_TTS_MODEL,
  VOICE_TTS_MODELS,
  chatModelSpeaks,
  isChatModelVoice,
  migrateVoiceTtsModel,
  servedVoiceTtsModel,
  speakerCarriesOver,
} from "../src/features/chat/voice/voice-models.ts";
import { holdForServing, releaseBeforeBorrow } from "../src/features/chat/voice/voice-ownership.ts";

function wavInfo(base64: string) {
  const bytes = Buffer.from(base64, "base64");
  return {
    riff: bytes.subarray(0, 4).toString("ascii"),
    rate: bytes.readUInt32LE(24),
    bits: bytes.readUInt16LE(34),
    dataBytes: bytes.readUInt32LE(40),
    riffSize: bytes.readUInt32LE(4),
    length: bytes.length,
  };
}

test("a turn's speech becomes one 16-bit WAV at the mic's rate", () => {
  const wav = wavInfo(pcmWavBase64(new Float32Array([0, 0.5, -0.5, 1]), 48000));
  assert.deepEqual(
    { riff: wav.riff, rate: wav.rate, bits: wav.bits, dataBytes: wav.dataBytes },
    { riff: "RIFF", rate: 48000, bits: 16, dataBytes: 8 },
  );
  assert.equal(wav.riffSize, wav.length - 8);
});

test("spoken reply chunks join into one WAV of all their samples, in order", () => {
  const a = pcmWavBase64(new Float32Array(1920).fill(0.25), 24000);
  const b = pcmWavBase64(new Float32Array(3840).fill(-0.25), 24000);
  const joined = joinWavBase64([a, b]);
  const info = wavInfo(joined);
  assert.equal(info.rate, 24000);
  assert.equal(info.dataBytes, (1920 + 3840) * 2);
  const bytes = Buffer.from(joined, "base64");
  assert.ok(bytes.readInt16LE(44) > 0);
  assert.ok(bytes.readInt16LE(44 + 1920 * 2) < 0);
  // One chunk is returned as it came; a chunk in another format is left out.
  assert.equal(joinWavBase64([a]), a);
  const other = pcmWavBase64(new Float32Array(10), 16000);
  assert.equal(wavInfo(joinWavBase64([a, other])).dataBytes, 1920 * 2);
});

test("the fragments of one turn are sent together, then the turn is empty", () => {
  const turn = createTurnAudio();
  assert.equal(turn.take(), null);
  turn.add(new Float32Array(100), 48000);
  turn.add(new Float32Array(50), 48000);
  assert.equal(wavInfo(turn.take() ?? "").dataBytes, 300);
  assert.equal(turn.take(), null);
  turn.add(new Float32Array(10), 48000);
  turn.clear();
  assert.equal(turn.take(), null);
});

test("only a turn sent with its speech is answered as a spoken reply", () => {
  const attachment = voiceTurnAttachment("UklGRg==", "voice-1");
  const spoken = { role: "user", content: [{ type: "text" }], attachments: [attachment] };
  const typed = { role: "user", content: [{ type: "text" }] };
  const reply = { role: "assistant", content: [{ type: "text" }] };
  assert.equal(lastUserTurnHasAudio([spoken, reply]), true);
  assert.equal(lastUserTurnHasAudio([spoken, reply, typed]), false);
  assert.equal(lastUserTurnHasAudio([{ role: "user", content: [{ type: "audio" }] }]), true);
  assert.equal(lastUserTurnHasAudio([]), false);
  assert.equal(attachment.contentType, "audio/wav");
});

test("the reply's player markup is never spoken or matched as an echo", () => {
  const text = 'Paris is the capital.<audio-player src="data:audio/wav;base64,AAAA" />';
  assert.equal(withoutAudioPlayer(text), "Paris is the capital.");
  assert.equal(withoutAudioPlayer("  plain reply "), "  plain reply ");
});

test("the chat model's voice is an opt-in pick, never the default and never adopted", () => {
  assert.equal(isChatModelVoice(VOICE_CHAT_MODEL_VOICE), true);
  assert.equal(isChatModelVoice(VOICE_DEFAULT_TTS_MODEL), false);
  assert.ok(!VOICE_TTS_MODELS.includes(VOICE_CHAT_MODEL_VOICE));
  assert.equal(VOICE_TTS_MODELS[0], VOICE_DEFAULT_TTS_MODEL);
  // A saved pick survives a reload; anything else migrates exactly as before.
  assert.equal(migrateVoiceTtsModel(VOICE_CHAT_MODEL_VOICE), VOICE_CHAT_MODEL_VOICE);
  assert.equal(migrateVoiceTtsModel(""), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(migrateVoiceTtsModel("unsloth/orpheus-3b-0.1-ft-GGUF"), VOICE_DEFAULT_TTS_MODEL);
  assert.equal(servedVoiceTtsModel([VOICE_CHAT_MODEL_VOICE, "LiquidAI/LFM3-Audio-2B"]), null);
});

test("whether the loaded chat model speaks comes from its status", () => {
  const models = [{ id: "LiquidAI/LFM3-Audio-2B", speaks: true }, { id: "unsloth/Qwen3-4B" }];
  assert.equal(chatModelSpeaks({ models, params: { checkpoint: "LiquidAI/LFM3-Audio-2B" } }), true);
  assert.equal(chatModelSpeaks({ models, params: { checkpoint: "unsloth/Qwen3-4B" } }), false);
  assert.equal(chatModelSpeaks({ models, params: { checkpoint: "" } }), false);
});

test("bubbles coalesced into one run send all their speech on the newest", () => {
  const a = pcmWavBase64(new Float32Array(100), 48000);
  const b = pcmWavBase64(new Float32Array(50), 48000);
  const c = pcmWavBase64(new Float32Array(25), 48000);
  const reply = { role: "assistant", content: [{ type: "text" }] };
  const bubble = (wav: string) => ({
    role: "user",
    content: [{ type: "text" }],
    attachments: [voiceTurnAttachment(wav, "x")],
  });
  // Answered speech stays out; the unanswered bubble before this one comes first.
  assert.equal(wavInfo(coalescedTurnAudio([bubble(a), reply, bubble(b)], c)).dataBytes, (50 + 25) * 2);
  // Three in one window: the middle bubble already holds the first's speech, so it is not doubled.
  const ab = coalescedTurnAudio([reply, bubble(a)], b);
  assert.equal(wavInfo(coalescedTurnAudio([reply, bubble(a), bubble(ab)], c)).dataBytes, (100 + 50 + 25) * 2);
  assert.equal(coalescedTurnAudio([bubble(a), reply], c), c);
  assert.equal(coalescedTurnAudio([], c), c);
});

test("switching an owned audio.cpp voice to the chat model's voice unloads it", () => {
  const owned = { id: "audiocpp-kokoro", owned: true };
  // The chat model's voice is borrowed like any other serving voice, so the owned one is released.
  assert.equal(releaseBeforeBorrow(owned, VOICE_CHAT_MODEL_VOICE), "audiocpp-kokoro");
  assert.equal(releaseBeforeBorrow({ id: "audiocpp-kokoro", owned: false }, VOICE_CHAT_MODEL_VOICE), null);
  assert.equal(releaseBeforeBorrow(null, VOICE_CHAT_MODEL_VOICE), null);
  assert.equal(releaseBeforeBorrow(owned, "audiocpp-kokoro"), null);
  assert.deepEqual(holdForServing(owned, VOICE_CHAT_MODEL_VOICE), { id: VOICE_CHAT_MODEL_VOICE, owned: false });
});

test("the speaker resets only when crossing to or from the chat model's voice", () => {
  assert.equal(speakerCarriesOver("audiocpp-kokoro", VOICE_CHAT_MODEL_VOICE), false);
  assert.equal(speakerCarriesOver(VOICE_CHAT_MODEL_VOICE, "audiocpp-kokoro"), false);
  assert.equal(speakerCarriesOver("audiocpp-kokoro", "audiocpp-other"), true);
  assert.equal(speakerCarriesOver(VOICE_CHAT_MODEL_VOICE, VOICE_CHAT_MODEL_VOICE), true);
});
