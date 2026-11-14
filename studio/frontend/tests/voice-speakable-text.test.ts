// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What reaches the TTS endpoint for a streamed reply: speakableReply, then
// splitIntoSentences, then stripForSpeech per sentence, the same chain the player runs.
// A string with no words in it is voiced by OmniVoice as noise.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  speakableReply,
  splitIntoSentences,
  splitStreaming,
  stripForSpeech,
} from "../src/features/chat/voice/speech-text.ts";
import { cleanTtsClip } from "../src/features/chat/voice/tts-clip-clean.ts";

const spoken = (reply: string) =>
  splitIntoSentences(speakableReply(reply)).map(stripForSpeech).filter(Boolean);

test("thinking and tool-call text never reaches the voice", () => {
  assert.deepEqual(spoken("<think>The user wants a joke.</think>Sure. Here goes."), ["Sure.", "Here goes."]);
  assert.deepEqual(spoken("Okay.<think>still reasoning about"), ["Okay."]);
  assert.deepEqual(spoken("<|channel>thought\nPlan it.<channel|>Hi there."), ["Hi there."]);
  assert.deepEqual(spoken('Let me look.\n{"query": "weather in Paris"}\nIt is sunny.'), ["Let me look.", "It is sunny."]);
  assert.deepEqual(spoken('<tool_call>{"name":"web_search"}</tool_call>Done.'), ["Done."]);
});

test("code fences are dropped, inline code keeps its words", () => {
  assert.deepEqual(spoken("Run this:\n```bash\nls -la /tmp\n```\nThat lists it."), ["Run this:", "That lists it."]);
  assert.deepEqual(spoken("Use `npm test` first."), ["Use npm test first."]);
  assert.deepEqual(spoken("Start:\n```py\nprint(1)"), ["Start:"]);
});

test("list markers do not become one-word sentences", () => {
  assert.deepEqual(spoken("Three steps.\n1. Back up.\n2. Upgrade.\n- Test it."), [
    "Three steps.",
    "Back up.",
    "Upgrade.",
    "Test it.",
  ]);
});

test("markup, slashes, URLs, emoji and LaTeX are not voiced", () => {
  assert.equal(stripForSpeech("**bold** and __under__ 😀"), "bold and under");
  assert.equal(stripForSpeech("it's \\s or /s, and/or"), "it's s or s, and or");
  assert.equal(stripForSpeech("See https://unsloth.ai/docs?x=1 for more."), "See for more.");
  assert.equal(stripForSpeech("Read [the docs](https://x.y/z) now."), "Read the docs now.");
  assert.equal(stripForSpeech("It costs $5 for 1/2."), "It costs $5 for 1/2.");
  assert.equal(stripForSpeech("$a \\times b$ and \\frac{1}{2}"), "a times b and 1 2");
  assert.equal(stripForSpeech("$\\alpha + \\beta$"), "alpha + beta");
});

test("a sentence with no words is not sent at all", () => {
  for (const junk of [")$.", "...", "…", "**", "```", "{ }", "😀", "—", "$$"]) {
    assert.equal(stripForSpeech(junk), "", junk);
  }
  assert.deepEqual(spoken("Here is the product $(a + b)$.\n)$."), ["Here is the product (a + b)."]);
});

test("a streamed prefix splits like the finished reply", () => {
  const full = "Sure.\n1. Back up first.\n<think>x</think>Then go.";
  const done = splitIntoSentences(speakableReply(full));
  for (let n = 1; n <= full.length; n++) {
    const part = splitIntoSentences(speakableReply(full.slice(0, n)));
    // Every complete sentence of a prefix is the same sentence in the final split.
    for (let i = 0; i < part.length - 1; i++) assert.equal(part[i], done[i], `prefix ${n}`);
  }
});

test("decimals and abbreviations survive streaming, token by token", () => {
  const reply =
    "Egypt gets about 2.5 inches of rain. Ask Dr. Smith in the U.S. about it, she says 3.14 is pi. Neat!";
  const done = splitIntoSentences(reply);
  const said: string[] = [];
  for (const m of reply.matchAll(/\S+\s*/g)) {
    const { complete } = splitStreaming(reply.slice(0, m.index! + m[0].length));
    while (said.length < complete.length) said.push(complete[said.length]!);
  }
  said.push(...done.slice(said.length));
  assert.deepEqual(said, done);
  assert.ok(said[0]!.includes("2.5 inches"), said[0]);
  assert.deepEqual(splitStreaming("Egypt gets about 2.").complete, []);
  assert.deepEqual(splitStreaming("Ask Dr. ").complete, []);
});

// OmniVoice's clip shape: 100 ms of digital zero, a ramp onto a -0.06 DC offset with
// hiss ~55 dB down, the voice on top, a ramp back to zero.
function omniLikeClip(sr: number, voice = 0.1): Float32Array {
  const x = new Float32Array(Math.round(sr * 1.2));
  let seed = 7;
  const hiss = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5) * 0.006;
  const z = Math.round(sr * 0.1);
  const ramp = Math.round(sr * 0.08);
  for (let i = z; i < x.length - z; i++) {
    const k = Math.min(1, (i - z) / ramp, (x.length - z - i) / ramp);
    const inVoice = i > z + sr * 0.25 && i < x.length - z - sr * 0.2;
    x[i] = -0.06 * k + hiss() + (inVoice ? voice * Math.sin((2 * Math.PI * 180 * i) / sr) : 0);
  }
  return x;
}

function lowBandShare(x: Float32Array, sr: number, hz = 40): number {
  // Energy left after a 25 ms moving average, against the total.
  const w = Math.round(sr / hz);
  let total = 0;
  let low = 0;
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    sum += x[i]! - (i >= w ? x[i - w]! : 0);
    low += (sum / w) ** 2;
    total += x[i]! ** 2;
  }
  return low / total;
}

test("clip cleaning removes the DC bump and trims to the voice", () => {
  const sr = 24000;
  const x = omniLikeClip(sr);
  assert.ok(lowBandShare(x, sr) > 0.3, `raw clip is mostly sub-40 Hz: ${lowBandShare(x, sr)}`);
  const [y] = cleanTtsClip([x], sr)!;
  assert.ok(y);
  assert.ok(lowBandShare(y, sr) < 0.02, `cleaned: ${lowBandShare(y, sr)}`);
  // Voice runs 0.35-0.9 s: the zeros, the ramps and the hiss around it are cut.
  assert.ok(y.length < sr * 0.7, `trimmed to ${y.length / sr} s`);
  assert.ok(y.length > sr * 0.55, `voice kept: ${y.length / sr} s`);
  assert.equal(Math.abs(y[0]!), 0);
  assert.equal(Math.abs(y[y.length - 1]!), 0);
  let mean = 0;
  for (const v of y) mean += v / y.length;
  assert.ok(Math.abs(mean) < 0.002, `offset left: ${mean}`);
});

test("a clip with no voice on it is dropped", () => {
  assert.equal(cleanTtsClip([omniLikeClip(24000, 0)], 24000), null);
  assert.equal(cleanTtsClip([new Float32Array(4800)], 24000), null);
});
