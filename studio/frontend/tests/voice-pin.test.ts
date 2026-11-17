// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearVoicePin,
  type VoiceReference,
  voicePinGeneration,
  wavSeconds,
  withVoicePin,
} from "../src/features/chat/voice/voice-pin.ts";

const speak = async (ref: VoiceReference | null) => ref?.inputId ?? "drawn";

function counted(id: string) {
  const draw = async (): Promise<VoiceReference | null> => {
    draw.calls += 1;
    return { inputId: `${id}-${draw.calls}`, text: "reference" };
  };
  draw.calls = 0;
  return draw;
}

test("one reference per session: every sentence clones it, including ones sent while it was drawn", async () => {
  clearVoicePin();
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  let calls = 0;
  const slowDraw = async (): Promise<VoiceReference> => {
    calls += 1;
    await gate;
    return { inputId: "ref", text: "reference" };
  };
  const first = withVoicePin("m", speak, slowDraw);
  const second = withVoicePin("m", speak, slowDraw);
  release();
  assert.deepEqual(await Promise.all([first, second]), ["ref", "ref"]);
  assert.equal(await withVoicePin("m", speak, slowDraw), "ref");
  assert.equal(calls, 1);
});

test("another model, or a new session, draws a new voice and retires cached audio", async () => {
  clearVoicePin();
  const draw = counted("v");
  assert.equal(await withVoicePin("a", speak, draw), "v-1");
  assert.equal(await withVoicePin("b", speak, draw), "v-2");
  assert.equal(await withVoicePin("b", speak, draw), "v-2");
  const before = voicePinGeneration();
  clearVoicePin();
  assert.notEqual(voicePinGeneration(), before);
  assert.equal(await withVoicePin("b", speak, draw), "v-3");
});

test("a failed or rejected draw speaks as drawn and the next sentence tries again", async () => {
  clearVoicePin();
  assert.equal(
    await withVoicePin("m", speak, async () => {
      throw new Error("model loading");
    }),
    "drawn",
  );
  assert.equal(await withVoicePin("m", speak, async () => null), "drawn");
  assert.equal(await withVoicePin("m", speak, counted("ok")), "ok-1");
});

test("a refused clone drops the pin so the caller's retry starts over", async () => {
  clearVoicePin();
  await withVoicePin("m", speak, counted("gone"));
  await assert.rejects(
    withVoicePin(
      "m",
      async (ref) => {
        if (ref) throw new Error("clip not found");
        return "drawn";
      },
      counted("x"),
    ),
  );
  assert.equal(await withVoicePin("m", speak, counted("fresh")), "fresh-1");
});

function wav(seconds: number, rate = 24000): ArrayBuffer {
  const data = Math.round(seconds * rate) * 2;
  const buf = new ArrayBuffer(44 + data);
  const v = new DataView(buf);
  const put = (at: number, s: string) =>
    [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  put(0, "RIFF");
  v.setUint32(4, 36 + data, true);
  put(8, "WAVE");
  put(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  put(36, "data");
  v.setUint32(40, data, true);
  return buf;
}

test("wavSeconds reads a clip's length, so a broken reference is refused", () => {
  assert.equal(wavSeconds(wav(4.5)), 4.5);
  assert.equal(wavSeconds(wav(0.6, 16000)), 0.6);
  assert.equal(wavSeconds(new ArrayBuffer(10)), null);
});
