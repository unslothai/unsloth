// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Live = typeof import("../src/features/chat/voice/live-transcript.ts");
const load = () =>
  loadWithStubs<Live>(new URL("../src/features/chat/voice/live-transcript.ts", import.meta.url), {
    "@/features/auth": { authFetch: () => Promise.reject(new Error("unused")) },
  });
const settle = () => new Promise((r) => setTimeout(r, 0));

test("a microphone's rate comes down to 16 kHz in pieces as it would whole", async () => {
  const { createResampler } = await load();
  const tone = (hz: number, rate: number) =>
    Float32Array.from({ length: rate }, (_, i) => 0.5 * Math.sin((2 * Math.PI * hz * i) / rate));
  const inPieces = (input: Float32Array, rate: number) => {
    const resample = createResampler(rate, 16000);
    const out: number[] = [];
    for (let i = 0; i < input.length; i += 4096) out.push(...resample(input.subarray(i, i + 4096)));
    return out;
  };
  const level = (x: number[]) =>
    Math.sqrt(x.slice(1000, -1000).reduce((sum, v) => sum + v * v, 0) / (x.length - 2000));
  for (const rate of [48000, 44100]) {
    const speech = inPieces(tone(1000, rate), rate);
    assert.ok(Math.abs(speech.length - 16000) < 40, `${rate}: ${speech.length} samples`);
    assert.ok(Math.abs(level(speech) - 0.3536) < 0.01, "a tone in the speech band keeps its level");
    assert.ok(level(inPieces(tone(12000, rate), rate)) < 0.005, "what 16 kHz cannot carry is removed, not folded in");
    const whole = createResampler(rate, 16000)(tone(1000, rate));
    for (let i = 0; i < speech.length; i++) assert.ok(Math.abs(speech[i] - whole[i]) < 1e-6);
  }
  const same = new Float32Array([0.1, 0.2]);
  assert.equal(createResampler(16000, 16000)(same), same);
});

test("words come back as the audio is posted, and finish gives the whole transcript", async () => {
  const { openLiveTranscript } = await load();
  const posts: { url: string; samples: number }[] = [];
  let events!: ReadableStreamDefaultController<Uint8Array>;
  const say = (event: object) => events.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
  const fetcher = (async (url: string, init?: RequestInit) => {
    if (url.includes("/live?")) {
      return new Response(new ReadableStream<Uint8Array>({ start: (c) => void (events = c) }));
    }
    posts.push({ url, samples: (init?.body as ArrayBuffer).byteLength / 2 });
    return new Response(null, { status: 204 });
  }) as never;
  const heard: string[] = [];
  const live = openLiveTranscript({
    params: new URLSearchParams({ model: "m" }),
    sampleRate: 16000,
    onText: (text) => heard.push(text),
    fetcher,
  });
  live.feed(new Float32Array(1600).fill(0.5));
  live.feed(new Float32Array(1600).fill(-0.5));
  await settle();
  assert.equal(posts.length, 0);
  say({ type: "ready", id: "s1" });
  await settle();
  await settle();
  assert.deepEqual(posts, [{ url: "/api/inference/audio/stt/live/s1/audio", samples: 3200 }]);
  say({ type: "delta", text: "Hello" });
  say({ type: "delta", text: "  there" });
  await settle();
  assert.deepEqual(heard, ["Hello", "Hello there"]);
  live.feed(new Float32Array(800));
  await settle();
  assert.equal(posts.length, 1);
  const text = live.finish();
  await settle();
  assert.deepEqual(posts[1], { url: "/api/inference/audio/stt/live/s1/audio?last=true", samples: 800 });
  say({ type: "done", text: "Hello  there." });
  assert.equal(await text, "Hello there.");
});

test("a model that cannot transcribe live says so once, and finish rejects so the recording is used", async () => {
  const { openLiveTranscript } = await load();
  let refused = 0;
  const live = openLiveTranscript({
    params: new URLSearchParams({ model: "m" }),
    sampleRate: 48000,
    onRefused: () => refused++,
    fetcher: (async () => new Response(JSON.stringify({ detail: "no" }), { status: 409 })) as never,
  });
  live.feed(new Float32Array(4096));
  await assert.rejects(live.finish());
  assert.equal(refused, 1);
});
