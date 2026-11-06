// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createFenceSpeculator,
  type SpeculationGate,
} from "../src/components/assistant-ui/fence-speculation.ts";

type Gate = SpeculationGate & { name: string; at: number; language: string };

const harness = (overrides: Partial<Parameters<typeof createFenceSpeculator<Gate>>[0]> = {}) => {
  const idle: (() => void)[] = [];
  const waits: { callback: () => void; ms: number }[] = [];
  const settled = new Map<Gate, (ok: boolean) => void>();
  const warmed = new Set<string>();
  const speculator = createFenceSpeculator<Gate>({
    idle: (callback) => {
      idle.push(callback);
      return () => {
        const at = idle.indexOf(callback);
        if (at >= 0) idle.splice(at, 1);
      };
    },
    wait: (callback, ms) => {
      waits.push({ callback, ms });
      return () => {
        const at = waits.findIndex((entry) => entry.callback === callback);
        if (at >= 0) waits.splice(at, 1);
      };
    },
    enabled: () => true,
    busy: () => false,
    eligible: (gate) => warmed.has(gate.language),
    rank: (gates) => gates.map((gate) => gate.at),
    maxChars: 20_000,
    budgetChars: 100_000,
    budgetFences: 4,
    ...overrides,
  });
  const gate = (name: string, at: number, chars = 100, language = "js"): Gate => ({
    name,
    at,
    chars,
    language,
    speculate: (settle) => {
      settled.set(gate_(name), settle);
      return () => settled.delete(gate_(name));
    },
  });
  const gates = new Map<string, Gate>();
  const gate_ = (name: string) => gates.get(name)!;
  const make = (name: string, at: number, chars?: number, language?: string) => {
    const g = gate(name, at, chars, language);
    gates.set(name, g);
    return g;
  };
  const runIdle = () => {
    const callback = idle.shift();
    callback?.();
  };
  return { speculator, idle, waits, settled, warmed, make, gate_, runIdle };
};

test("the nearest warmed fence goes first, one at a time, until the budget is spent", () => {
  const h = harness();
  h.warmed.add("js");
  const far = h.make("far", 900);
  const near = h.make("near", 10);
  const middle = h.make("middle", 400);
  h.speculator.add(far);
  h.speculator.add(near);
  h.speculator.add(middle);
  assert.equal(h.idle.length, 1, "one idle slot for three candidates");
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["near"]);
  assert.equal(h.idle.length, 0, "nothing else is picked while one is in flight");
  h.settled.get(near)!(true);
  assert.equal(h.speculator.snapshot().seeded, 1);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["near", "middle"]);
  h.settled.get(middle)!(true);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["near", "middle", "far"]);
});

test("fences that are empty or over the cap are never candidates", () => {
  const h = harness();
  h.warmed.add("js");
  h.speculator.add(h.make("empty", 0, 0));
  h.speculator.add(h.make("huge", 0, 20_001));
  assert.equal(h.speculator.snapshot().candidates, 0);
  assert.equal(h.idle.length, 0);
});

test("a language the main thread has not warmed waits instead of being seeded", () => {
  const h = harness();
  const cold = h.make("cold", 0, 100, "python");
  h.speculator.add(cold);
  h.runIdle();
  assert.equal(h.settled.size, 0);
  assert.equal(h.waits.length, 1);
  assert.equal(h.waits[0].ms, 1000);
  h.warmed.add("python");
  h.waits.shift()!.callback();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["cold"]);
});

test("removing the in-flight fence cancels it and frees the slot; removing a seeded one releases its budget", () => {
  const h = harness();
  h.warmed.add("js");
  const a = h.make("a", 0, 1000);
  const b = h.make("b", 1, 1000);
  h.speculator.add(a);
  h.speculator.add(b);
  h.runIdle();
  assert.ok(h.settled.has(a));
  h.speculator.remove(a);
  assert.equal(h.settled.has(a), false, "the cancel closure ran");
  assert.equal(h.speculator.snapshot().inFlight, false);
  h.runIdle();
  assert.ok(h.settled.has(b));
  h.settled.get(b)!(true);
  assert.equal(h.speculator.snapshot().seededChars, 1000);
  h.speculator.remove(b);
  assert.equal(h.speculator.snapshot().seededChars, 0);
});

test("the character budget skips a nearer fence that does not fit and the fence budget stops picking", () => {
  const h = harness({ budgetChars: 1500, budgetFences: 2 });
  h.warmed.add("js");
  const big = h.make("big", 0, 1400);
  const near = h.make("near", 3, 200);
  const small = h.make("small", 5, 50);
  const tiny = h.make("tiny", 9, 10);
  for (const g of [big, near, small, tiny]) h.speculator.add(g);
  h.runIdle();
  h.settled.get(big)!(true);
  assert.equal(h.speculator.snapshot().seededChars, 1400);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["big", "small"], "the 200-character fence is nearer but would pass 1500, so the 50-character one is picked");
  h.settled.get(small)!(true);
  h.runIdle();
  assert.equal(h.settled.has(tiny), false, "two seeded fences is the fence budget");
});

test("a failed speculation spends no budget", () => {
  const h = harness({ budgetChars: 1000, budgetFences: 1 });
  h.warmed.add("js");
  const a = h.make("a", 0, 900);
  const b = h.make("b", 1, 900);
  h.speculator.add(a);
  h.speculator.add(b);
  h.runIdle();
  h.settled.get(a)!(false);
  assert.deepEqual([h.speculator.snapshot().seeded, h.speculator.snapshot().seededChars], [0, 0]);
  h.runIdle();
  assert.ok(h.settled.has(b), "the next fence still fits");
});

test("a pause while the worker is late keeps every candidate, and speculation resumes once it answers", () => {
  let busy = true;
  const h = harness({ busy: () => busy });
  h.warmed.add("js");
  const gates = ["a", "b", "c"].map((name, at) => h.make(name, at));
  for (const g of gates) h.speculator.add(g);
  h.runIdle();
  assert.equal(h.settled.size, 0);
  assert.equal(h.speculator.snapshot().candidates, 3);
  h.waits.shift()!.callback();
  assert.equal(h.speculator.snapshot().candidates, 3, "still paused, nothing dropped");
  busy = false;
  h.waits.shift()!.callback();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["a"]);
});

test("a stream defers the pick, a disabled worker drops every candidate, and a throw settles false", () => {
  let busy = true;
  let enabled = true;
  const h = harness({ busy: () => busy, enabled: () => enabled });
  h.warmed.add("js");
  const a = h.make("a", 0);
  h.speculator.add(a);
  h.runIdle();
  assert.equal(h.settled.size, 0);
  assert.equal(h.waits.length, 1);
  busy = false;
  h.waits.shift()!.callback();
  assert.ok(h.settled.has(a));
  h.settled.get(a)!(false);
  const thrower: Gate = { name: "t", at: 0, chars: 10, language: "js", speculate: () => { throw new Error("boom"); } };
  h.speculator.add(thrower);
  h.runIdle();
  assert.equal(h.speculator.snapshot().inFlight, false, "a throw never wedges the slot");
  enabled = false;
  h.speculator.add(h.make("b", 1));
  h.runIdle();
  assert.equal(h.speculator.snapshot().candidates, 0);
});

test("with several slots, one idle task launches nearest first up to the limit and a settle refills it", () => {
  const h = harness({ maxInFlight: 3 });
  h.warmed.add("js");
  const gates = ["a", "b", "c", "d", "e"].map((name, at) => h.make(name, at * 10));
  for (const g of gates) h.speculator.add(g);
  assert.equal(h.idle.length, 1);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["a", "b", "c"]);
  assert.equal(h.speculator.snapshot().inFlightCount, 3);
  assert.equal(h.idle.length, 0, "a full set of slots schedules nothing more");
  h.settled.get(gates[1])!(true);
  assert.equal(h.idle.length, 1, "a settled slot asks for one more idle task");
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["a", "b", "c", "d"]);
  h.speculator.remove(gates[0]);
  assert.equal(h.settled.has(gates[0]), false, "removing an in-flight fence cancels it");
  assert.equal(h.speculator.snapshot().inFlightCount, 2);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["b", "c", "d", "e"]);
});

test("characters and fences in flight count against the budget, so concurrency never overshoots it", () => {
  const h = harness({ maxInFlight: 3, budgetChars: 250, budgetFences: 3 });
  h.warmed.add("js");
  const a = h.make("a", 0, 100);
  const b = h.make("b", 1, 100);
  const c = h.make("c", 2, 100);
  const d = h.make("d", 3, 40);
  for (const g of [a, b, c, d]) h.speculator.add(g);
  h.runIdle();
  assert.deepEqual([...h.settled.keys()].map((g) => g.name), ["a", "b", "d"], "the third 100-char fence would pass 250 with two in flight, the 40-char one fits");
  h.settled.get(a)!(true);
  h.settled.get(b)!(true);
  h.settled.get(d)!(true);
  h.runIdle();
  assert.equal(h.settled.has(c), false, "three seeded fences is the fence budget");
});

test("a settle for a fence that is no longer in flight is ignored and ranking refreshes on demand", () => {
  const h = harness();
  h.warmed.add("js");
  const a = h.make("a", 0);
  const b = h.make("b", 1);
  h.speculator.add(a);
  h.speculator.add(b);
  h.runIdle();
  const settleA = h.settled.get(a)!;
  h.speculator.remove(a);
  settleA(true);
  assert.equal(h.speculator.snapshot().seeded, 0, "a late settle for a removed fence does not count");
  h.speculator.rerank();
  h.runIdle();
  assert.ok(h.settled.has(b));
});
