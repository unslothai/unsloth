// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// Resolves the module's extensionless relative imports, which Vite handles in the
// app and Node does not.
register("./helpers/tauri-core-resolver.mjs", import.meta.url);

// PcmCapture only touches these factory methods, so stubs suffice. `state` is
// part of the fake because the shared context is replaced once it closes.
class FakeNode {
  disconnected = false;
  connect(): void {}
  disconnect(): void {
    this.disconnected = true;
  }
}

class FakeGainNode extends FakeNode {
  gain = { value: 1 };
}

class FakeProcessorNode extends FakeNode {
  listener: ((event: unknown) => void) | null = null;
  addEventListener(_type: string, listener: (event: unknown) => void): void {
    this.listener = listener;
  }
  removeEventListener(_type: string, listener: (event: unknown) => void): void {
    if (this.listener === listener) this.listener = null;
  }
}

class FakeAudioContext {
  static created = 0;
  static last: FakeAudioContext | null = null;

  sampleRate: number;
  state: AudioContextState = "running";
  destination = new FakeNode();
  processor = new FakeProcessorNode();
  gain = new FakeGainNode();
  resumed = false;
  closed = false;

  constructor(options?: { sampleRate?: number }) {
    FakeAudioContext.created += 1;
    this.sampleRate = options?.sampleRate ?? 48_000;
    FakeAudioContext.last = this;
  }
  createMediaStreamSource(): FakeNode {
    return new FakeNode();
  }
  createScriptProcessor(): FakeProcessorNode {
    return this.processor;
  }
  createGain(): FakeGainNode {
    return this.gain;
  }
  resume(): Promise<void> {
    this.resumed = true;
    return Promise.resolve();
  }
  close(): Promise<void> {
    this.closed = true;
    this.state = "closed";
    return Promise.resolve();
  }
}

Object.assign(globalThis, {
  window: { AudioContext: FakeAudioContext },
});

const { PcmCapture, prewarmPcmCapture } = await import(
  "../src/features/chat/adapters/pcm-capture.ts"
);

const capture = () =>
  new PcmCapture({} as MediaStream, {
    onSegment: () => {},
  });

test("every recording shares one audio context", () => {
  const before = FakeAudioContext.created;
  const first = capture();
  const second = capture();
  assert.equal(
    FakeAudioContext.created,
    before + 1,
    "a per-recording context pays for an audio thread on every press",
  );
  assert.equal(first.sampleRate, second.sampleRate);
  first.stop();
  second.stop();
});

test("stopping disconnects the graph but keeps the context for the next press", () => {
  const recorder = capture();
  const context = FakeAudioContext.last;
  assert.ok(context);
  recorder.start();
  recorder.stop();
  assert.equal(context.closed, false, "closing would pay the open cost again");
  assert.equal(context.processor.disconnected, true);
  assert.equal(context.gain.disconnected, true);
  // The listener holds the capture alive; a shared context outlives it.
  assert.equal(context.processor.listener, null);
});

test("a context the browser closed is replaced instead of reused", () => {
  const first = capture();
  first.stop();
  const closed = FakeAudioContext.last;
  assert.ok(closed);
  void closed.close();
  const before = FakeAudioContext.created;
  const second = capture();
  assert.equal(FakeAudioContext.created, before + 1);
  assert.notEqual(FakeAudioContext.last, closed);
  second.stop();
});

test("prewarm reuses the open context and is safe to repeat", () => {
  const before = FakeAudioContext.created;
  prewarmPcmCapture();
  prewarmPcmCapture();
  assert.equal(
    FakeAudioContext.created,
    before,
    "prewarming must not open a second audio thread",
  );
  assert.equal(FakeAudioContext.last?.resumed, true);
});
