// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { AgentDomCostIndex, agentPhysicalNeighborWindow, agentWheelPrefetchTarget, agentBudgetWindow, AGENT_DOM_TARGETS, AGENT_FIXED_VIEWPORT_DOM_TARGET, AGENT_FIXED_VIEWPORT_MAX_ROWS, needsAgentBudgetRebalance } from "../src/components/assistant-ui/agent-history-dom-budget.ts";

test("low and high adapt to measured row complexity", () => {
  const costs = new AgentDomCostIndex(40);
  // The first render must never mount multiple unmeasured, potentially
  // huge tool results. Each later render may admit only one unknown neighbor.
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.low), { start: 39, end: 40 });
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.high), { start: 39, end: 40 });
  costs.measure(39, 4_800);
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.low), { start: 39, end: 40 });
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.high), { start: 38, end: 40 });
  costs.measure(38, 100);
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.high), { start: 37, end: 40 });
  costs.measure(39, 100);
  assert.deepEqual(agentBudgetWindow(40, 39, costs, AGENT_DOM_TARGETS.low), { start: 37, end: 40 });
});

test("oversized target stays mounted and no window skips its target", () => {
  const costs = new AgentDomCostIndex(100);
  costs.measure(50, 100_000);
  for (const at of [0, 1, 15, 50, 98, 99]) {
    for (const budget of Object.values(AGENT_DOM_TARGETS)) {
      const window = agentBudgetWindow(100, at, costs, budget);
      assert.ok(window.start <= at && at < window.end);
      assert.ok(window.end - window.start <= 32);
    }
  }
  assert.deepEqual(agentBudgetWindow(100, 50, costs, AGENT_DOM_TARGETS.low), { start: 50, end: 51 });
  assert.deepEqual(agentBudgetWindow(0, 0, costs, AGENT_DOM_TARGETS.low), { start: 0, end: 0 });
});

test("invalid measurements cannot poison the budget policy", () => {
  const costs = new AgentDomCostIndex(3);
  assert.equal(costs.measure(10, 1), false);
  assert.equal(costs.measure(0, Infinity), false);
  assert.equal(costs.measure(0, 3_300), true);
  costs.resize(0);
  assert.equal(costs.cost(0), 1_000);
});

test("a five-row run of unknown giant outputs never mounts together", () => {
  const costs = new AgentDomCostIndex(1000);
  for (const level of Object.values(AGENT_DOM_TARGETS)) {
    assert.deepEqual(agentBudgetWindow(1000, 999, costs, level),
      { start: 999, end: 1000 });
  }
  costs.measure(999, 80_000);
  for (const level of Object.values(AGENT_DOM_TARGETS)) {
    assert.deepEqual(agentBudgetWindow(1000, 999, costs, level),
      { start: 999, end: 1000 });
  }
});


test("invalid target coordinates cannot create NaN or disjoint windows", () => {
  const costs = new AgentDomCostIndex(40);
  for (const at of [Number.NaN, Infinity, -Infinity, -1, 40, 1.9]) {
    const window = agentBudgetWindow(40, at, costs, AGENT_DOM_TARGETS.low);
    assert.ok(Number.isSafeInteger(window.start) && Number.isSafeInteger(window.end));
    assert.ok(window.start >= 0 && window.end <= 40);
    assert.ok(window.end - window.start >= 1 && window.end - window.start <= 32);
  }
});

test("repeated adversarial measurement updates never expand beyond measured budget", () => {
  const total = 400;
  const costs = new AgentDomCostIndex(total);
  const target = 200;
  const budgets = Object.values(AGENT_DOM_TARGETS);
  for (let phase = 0; phase < 250; phase++) {
    // Simulate a tool result that grows, shrinks, then expands again.
    costs.measure(target, phase % 3 === 0 ? 250_000 : phase % 3 === 1 ? 20 : 4_800);
    for (const budget of budgets) {
      const window = agentBudgetWindow(total, target, costs, budget);
      assert.ok(window.start <= target && target < window.end);
      assert.ok(window.end - window.start <= 32);
      const used = Array.from({ length: window.end - window.start }, (_, i) =>
        costs.cost(window.start + i)).reduce((a, b) => a + b, 0);
      assert.ok(used <= budget || window.end - window.start === 1,
        `phase ${phase}: ${used} nodes across ${window.end - window.start} rows`);
      const unknown = Array.from({ length: window.end - window.start }, (_, i) =>
        costs.measured(window.start + i) ? 0 : 1).reduce<number>((a, b) => a + b, 0);
      assert.ok(unknown <= 1, "never mount a batch of unknown huge tool rows");
    }
  }
});

test("Low and High stop adding extra messages after the viewport is filled", () => {
  const single = { start: 99, end: 100 };
  const expanded = { start: 98, end: 100 };
  assert.equal(needsAgentBudgetRebalance(single, expanded, 99, 500, 900), true);
  assert.equal(needsAgentBudgetRebalance(single, expanded, 99, 920, 900), false);
  assert.equal(needsAgentBudgetRebalance(single, expanded, 99, 890, 900), false);
  assert.equal(needsAgentBudgetRebalance({ start: 96, end: 100 },
    single, 99, 2200, 900), true);
  assert.equal(needsAgentBudgetRebalance(single, expanded, 50, 920, 900), true);
  assert.equal(needsAgentBudgetRebalance(single, single, 99, 500, 900), false);
});

test("dynamic Low never admits more than five rows and High never exceeds 32", () => {
  const costs = new AgentDomCostIndex(200);
  for (let row = 0; row < 200; row++) costs.measure(row, 10);
  for (let target = 0; target < 200; target += 7) {
    const low = agentBudgetWindow(200, target, costs, AGENT_DOM_TARGETS.low, 5);
    const high = agentBudgetWindow(200, target, costs, AGENT_DOM_TARGETS.high, 32);
    assert.ok(low.end - low.start >= 1 && low.end - low.start <= 5);
    assert.ok(high.end - high.start >= 1 && high.end - high.start <= 32);
    assert.ok(low.start <= target && target < low.end);
    assert.ok(high.start <= target && target < high.end);
  }
});


test("canonical dynamic admits at most 16 rows under an 8k DOM-node budget", () => {
  assert.equal(AGENT_DOM_TARGETS.medium, 8_000);
  const costs = new AgentDomCostIndex(200);
  for (let row = 0; row < 200; row++) costs.measure(row, 1);
  assert.deepEqual(agentBudgetWindow(200, 199, costs, AGENT_DOM_TARGETS.medium, 16),
    { start: 184, end: 200 });
  costs.measure(199, 7_900);
  const tight = agentBudgetWindow(200, 199, costs, AGENT_DOM_TARGETS.medium, 16);
  assert.ok(tight.end - tight.start <= 16);
  assert.ok(tight.start <= 199 && tight.end > 199);
});


test("viewport-adaptive Fixed grows beyond five short turns while respecting hard policy limits", () => {
  const total = 150;
  const costs = new AgentDomCostIndex(total);
  const anchor = 70;
  assert.equal(AGENT_FIXED_VIEWPORT_MAX_ROWS, 32);
  assert.equal(AGENT_FIXED_VIEWPORT_DOM_TARGET, 8_000);
  // Only one unmeasured turn may mount at a time; prevents loading many
  // unknown, potentially huge tool cards during the first render.
  assert.deepEqual(agentBudgetWindow(total, anchor, costs, AGENT_FIXED_VIEWPORT_DOM_TARGET,
    AGENT_FIXED_VIEWPORT_MAX_ROWS), { start: anchor, end: anchor + 1 });
  let current = agentBudgetWindow(total, anchor, costs, AGENT_FIXED_VIEWPORT_DOM_TARGET,
    AGENT_FIXED_VIEWPORT_MAX_ROWS);
  for (let frame = 0; frame < 100; frame++) {
    for (let i = current.start; i < current.end; i++) costs.measure(i, 2);
    // 40px tall lightweight rows; 20+ rows are necessary for a 900px viewport.
    const proposed = agentBudgetWindow(total, anchor, costs, AGENT_FIXED_VIEWPORT_DOM_TARGET,
      AGENT_FIXED_VIEWPORT_MAX_ROWS);
    const height = 40 * (current.end - current.start);
    if (!needsAgentBudgetRebalance(current, proposed, anchor, height, 900)) break;
    current = proposed;
  }
  assert.ok(current.end - current.start >= 20,
    "Fixed must be permitted to render substantially more than five lightweight turns");
  assert.ok(current.end - current.start <= AGENT_FIXED_VIEWPORT_MAX_ROWS);
  assert.ok(current.start <= anchor && current.end > anchor);
  const used = Array.from({ length: current.end-current.start }, (_, i) =>
    costs.cost(current.start+i)).reduce((sum, v) => sum+v, 0);
  assert.ok(used <= AGENT_FIXED_VIEWPORT_DOM_TARGET);
});

test("Fixed viewport growth cannot admit multiple unknown massive tool outputs", () => {
  const costs = new AgentDomCostIndex(200);
  for (let anchor = 0; anchor < 200; anchor += 11) {
    const start = agentBudgetWindow(200, anchor, costs, AGENT_FIXED_VIEWPORT_DOM_TARGET,
      AGENT_FIXED_VIEWPORT_MAX_ROWS);
    assert.equal(start.end - start.start, 1);
    costs.measure(anchor, 25_000);
    const single = agentBudgetWindow(200, anchor, costs, AGENT_FIXED_VIEWPORT_DOM_TARGET,
      AGENT_FIXED_VIEWPORT_MAX_ROWS);
    assert.equal(single.end - single.start, 1,
      "a single overbudget row must never cause additional heavy rows to mount");
  }
});

test("physical neighbor recovery retains measured windows and bounded DOM", () => {
  const costs = new AgentDomCostIndex(400);
  for (let row = 300; row < 400; row++) costs.measure(row, 55);
  // An adjacent still-unknown short text row must not cause a one-row
  // renderer collapse; drop the opposite edge within the 16-row cap.
  assert.deepEqual(
    agentPhysicalNeighborWindow({ start: 384, end: 400 }, 383, 400, costs, 8000, 16),
    { start: 383, end: 399 },
  );
  assert.deepEqual(
    agentPhysicalNeighborWindow({ start: 300, end: 316 }, 316, 400, costs, 8000, 16),
    { start: 301, end: 317 },
  );
  // The old target may be one unusually huge tool row; only the necessary
  // adjacent target is mandatory and may exceed the soft DOM budget.
  costs.measure(383, 18000);
  assert.deepEqual(
    agentPhysicalNeighborWindow({ start: 384, end: 400 }, 383, 400, costs, 8000, 16),
    { start: 383, end: 384 },
  );
  assert.equal(agentPhysicalNeighborWindow({ start: 384, end: 400 }, 381, 400, costs, 8000, 16), null);
  assert.equal(agentPhysicalNeighborWindow({ start: 384, end: 400 }, 400, 400, costs, 8000, 16), null);
});

test("physical neighbor recovery does not admit two unknown heavy rows", () => {
  const costs = new AgentDomCostIndex(400);
  for (let row = 384; row < 400; row++) costs.measure(row, 70);
  // Current mounted row 392 was never measured; adding 383 could create
  // two simultaneous unbounded tool cards, so the helper must refuse.
  costs.resize(401);
  const partial = new AgentDomCostIndex(400);
  for (let row = 384; row < 400; row++) if (row !== 392) partial.measure(row, 70);
  assert.equal(
    agentPhysicalNeighborWindow({ start: 384, end: 400 }, 383, 400, partial, 8000, 16),
    null,
  );
});

test("wheel prefetch only predicts an adjacent older row close to viewport", () => {
  // Initial real top is 180px ABOVE viewport; upcoming -240 wheel would
  // put it 60px inside, so preload one neighbor before it becomes blank.
  assert.equal(agentWheelPrefetchTarget(-240, -180, 0, 960, 380), 379);
  assert.equal(agentWheelPrefetchTarget(-240, -500, 0, 960, 380), null);
  assert.equal(agentWheelPrefetchTarget(-720, -600, 0, 960, 380), 379);
  // Long leaps still use existing conservative progressive staging.
  assert.equal(agentWheelPrefetchTarget(-1680, -180, 0, 960, 380), null);
  assert.equal(agentWheelPrefetchTarget(240, -180, 0, 960, 380), null);
  assert.equal(agentWheelPrefetchTarget(-240, -180, 0, 960, 0), null);
  assert.equal(agentWheelPrefetchTarget(NaN, -180, 0, 960, 380), null);
  assert.equal(agentWheelPrefetchTarget(-240, -180, 0, 0, 380), null);
  assert.equal(agentWheelPrefetchTarget(-240, NaN, 0, 960, 380), null);
});
