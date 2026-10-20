// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The 07:01 trace: a stand-in "Let me think." went on a timeline whose clock never
// moved, no onended came, the orb sat on speaking and "Still thinking." and the
// approval prompt were both dropped behind clips:1.

import assert from "node:assert/strict";
import test from "node:test";

import {
  asideBlocked,
  CLIP_WATCHDOG_GRACE_MS,
  createClipLedger,
  type LedgerSource,
} from "../src/features/chat/voice/tts-clip-ledger.ts";

function rig() {
  const ctx = { currentTime: 10, state: "running", resumes: 0, resume: async () => void ctx.resumes++ };
  let now = 0;
  let timers: { at: number; fn: () => void; id: number }[] = [];
  let nextId = 1;
  const events: { event: string; detail: Record<string, unknown> }[] = [];
  const state = { playing: false, frozen: false };
  const ledger = createClipLedger({
    ctx: () => ctx,
    frozen: () => state.frozen,
    onBusy: () => (state.playing = true),
    onIdle: () => (state.playing = false),
    trace: (event, detail) => events.push({ event, detail }),
    now: () => now,
    setTimer: (fn, ms) => {
      const id = nextId++;
      timers.push({ at: now + ms, fn, id });
      return id;
    },
    clearTimer: (id) => (timers = timers.filter((t) => t.id !== id)),
  });
  // Advance wall time; the audio clock moves only while `clock` is true.
  const advance = (ms: number, clock: boolean) => {
    const until = now + ms;
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      const t = timers[0];
      if (!t || t.at > until) break;
      if (clock) ctx.currentTime += (t.at - now) / 1000;
      now = t.at;
      timers.shift();
      t.fn();
    }
    if (clock) ctx.currentTime += (until - now) / 1000;
    now = until;
  };
  const source = () => {
    const s: LedgerSource & { started: number | null; stopped: boolean } = {
      onended: null,
      started: null,
      stopped: false,
      start: (when) => void (s.started = when),
      stop: () => void (s.stopped = true),
    };
    return s;
  };
  const add = (text: string, duration: number) => {
    const s = source();
    ledger.add({ source: s, start: ctx.currentTime + 0.02, duration, text, env: new Float32Array(1) });
    return s;
  };
  return { ctx, ledger, state, events, advance, add };
}

test("a stand-in that never sounds is force-ended after its duration + grace", () => {
  const r = rig();
  r.ctx.state = "suspended";
  const s = r.add("Let me think.", 1.1);
  assert.equal(r.state.playing, true);
  // "Still thinking." at +3.5 s: the stuck clip still blocks a filler.
  r.advance(1100, false);
  assert.equal(asideBlocked(r.ledger, { patient: false, replySynth: false }), true);
  r.advance(1100 + CLIP_WATCHDOG_GRACE_MS, false);
  assert.equal(r.ledger.size, 0);
  assert.equal(r.state.playing, false, "orb leaves speaking");
  assert.equal(s.stopped, true);
  const dog = r.events.find((e) => e.event === "tts.clipWatchdog");
  assert.ok(dog);
  assert.equal(dog.detail.text, "Let me think.");
  assert.ok(Number(dog.detail.ms) >= 1100 + CLIP_WATCHDOG_GRACE_MS);
  assert.equal(r.ctx.resumes, 1, "tries to wake the parked context");
});

test("a running clock that froze mid-clip is caught too", () => {
  const r = rig();
  r.add("Let me think.", 1.1);
  r.advance(500, true);
  r.advance(60_000, false);
  assert.equal(r.ledger.size, 0);
  assert.equal(r.state.playing, false);
});

test("a clip that is playing normally is never reaped", () => {
  const r = rig();
  const s = r.add("A long sentence.", 5);
  r.advance(4000, true);
  assert.equal(r.ledger.size, 1);
  assert.equal(r.ledger.audible(), true);
  s.onended?.(new Event("ended"));
  assert.equal(r.state.playing, false);
  assert.equal(r.events.some((e) => e.event === "tts.clipWatchdog"), false);
});

test("a clip started late on a moving clock gets its time", () => {
  const r = rig();
  r.add("Late.", 2);
  r.advance(1000, false);
  // The context came back and is now partway through the clip.
  r.advance(2500, true);
  assert.equal(r.ledger.size, 1);
  assert.equal(r.events.some((e) => e.event === "tts.clipWatchdog"), false);
});

test("pause and hold freeze the clock on purpose: no watchdog", () => {
  const r = rig();
  r.state.frozen = true;
  r.ctx.state = "suspended";
  r.add("Held reply.", 1);
  r.advance(30_000, false);
  assert.equal(r.ledger.size, 1);
  assert.equal(r.state.playing, true);
  r.state.frozen = false;
  r.ctx.state = "running";
  r.advance(5000, true);
  assert.equal(r.ledger.size, 0, "resumed and still silent: reaped");
});

test("the approval aside plays over a stuck clip instead of being dropped", () => {
  const r = rig();
  r.ctx.state = "suspended";
  r.add("Let me think.", 1.1);
  r.advance(500, false);
  r.ctx.state = "running";
  // A filler still defers to anything on the timeline...
  assert.equal(asideBlocked(r.ledger, { patient: false, replySynth: false }), true);
  // ...but the patient line only to something audible. Nothing is: reap and play.
  r.ctx.currentTime += 5;
  assert.equal(asideBlocked(r.ledger, { patient: true, replySynth: false }), false);
  assert.equal(r.ledger.size, 0);
  assert.equal(r.state.playing, false);
  r.add("Can you allow that in the chat? I'll wait.", 2.5);
  assert.equal(r.state.playing, true);
  r.advance(100, true);
  assert.equal(r.ledger.audible(), true);
});

test("the approval aside still yields to speech that is really audible", () => {
  const r = rig();
  r.add("The answer is Paris.", 3);
  r.advance(500, true);
  assert.equal(asideBlocked(r.ledger, { patient: true, replySynth: false }), true);
  assert.equal(r.ledger.size, 1);
});

test("the player routes stand-ins, pauses and patient asides through the ledger", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url), "utf8");
  assert.match(src, /asideBlocked\(clipsRef\.current, \{ patient, replySynth: realSynthRef\.current \}\)/);
  assert.match(src, /play\(asides\(\)\.get\(standIn\)!, true, standIn\)/);
  assert.match(src, /heldRef\.current \|\| \(pausedRef\.current && !patient\)/);
  // isPlaying is set only by the ledger's busy/idle callbacks.
  assert.equal((src.match(/setIsPlaying\(true\)/g) ?? []).length, 1);
  assert.match(src, /onBusy: \(\) => setIsPlaying\(true\)/);
});
