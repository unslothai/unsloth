// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  outputSpectrumNode,
  pauseOutputSpectrum,
} from "../src/features/chat/voice/voice-spectrum.ts";

type FakeNode = {
  kind: string;
  links: Set<FakeNode>;
  connect(to: FakeNode): void;
  disconnect(to?: FakeNode): void;
  onaudioprocess?: (() => void) | null;
  gain?: { value: number };
  fftSize?: number;
  smoothingTimeConstant?: number;
  frequencyBinCount?: number;
  context?: unknown;
};

function fakeContext() {
  const made: FakeNode[] = [];
  const node = (kind: string, extra: Partial<FakeNode> = {}): FakeNode => {
    const n: FakeNode = {
      kind,
      links: new Set(),
      connect(to) {
        n.links.add(to);
      },
      disconnect(to) {
        if (to) n.links.delete(to);
        else n.links.clear();
      },
      ...extra,
    };
    made.push(n);
    return n;
  };
  const ctx = {
    sampleRate: 48000,
    destination: node("destination"),
    createAnalyser: () => node("analyser", { fftSize: 2048, frequencyBinCount: 1024, context: ctx }),
    createScriptProcessor: () => node("clock", { onaudioprocess: null }),
    createGain: () => node("gain", { gain: { value: 1 } }),
  };
  return { ctx, made };
}

test("voice off stops the output clock and the next clip restarts it", () => {
  const { ctx, made } = fakeContext();
  const analyser = outputSpectrumNode(ctx as unknown as AudioContext) as unknown as FakeNode;
  const clock = made.find((n) => n.kind === "clock");
  assert.ok(clock && analyser.links.has(clock), "clock is fed by the analyser");
  assert.equal(typeof clock.onaudioprocess, "function");

  pauseOutputSpectrum();
  assert.equal(clock.onaudioprocess, null, "no more callbacks");
  assert.equal(analyser.links.has(clock), false, "analyser no longer feeds the clock");
  assert.equal(clock.links.size, 0, "clock detached from the graph");
  assert.ok(analyser.links.has(ctx.destination), "playback path is untouched");

  // Next clip on the same context: same analyser, a fresh clock.
  const again = outputSpectrumNode(ctx as unknown as AudioContext) as unknown as FakeNode;
  assert.equal(again, analyser);
  const clocks = made.filter((n) => n.kind === "clock");
  assert.equal(clocks.length, 2);
  assert.ok(analyser.links.has(clocks[1] as FakeNode));
  assert.equal(typeof clocks[1]?.onaudioprocess, "function");

  // Repeated plays do not stack clocks; a second pause is a no-op.
  outputSpectrumNode(ctx as unknown as AudioContext);
  assert.equal(made.filter((n) => n.kind === "clock").length, 2);
  pauseOutputSpectrum();
  pauseOutputSpectrum();
});

test("turning voice off pauses the output tap", () => {
  const engine = readFileSync(
    new URL("../src/features/chat/voice/voice-engine.tsx", import.meta.url),
    "utf8",
  );
  const turnOff = engine.slice(engine.indexOf("const turnOff = useCallback"));
  const end = turnOff.indexOf("}, [");
  assert.ok(turnOff.slice(0, end).includes("pauseOutputSpectrum();"));
});
