// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Exhaustive sweep of the API-examples agent picker's decision space. The component
// runs pickCompatibleAgent in an effect that re-runs on its own output, so the two
// properties that matter are not "the answer is X" but "it settles" and "it never
// settles somewhere unrunnable". Enumerating every reachable input is cheap here
// because the decision is a pure function.

import assert from "node:assert/strict";
import test from "node:test";
import {
  GGUF_ONLY_AGENTS,
  UNIVERSAL_AGENT,
  agentRunsOnActiveModel,
  fallbackAgent,
  pickCompatibleAgent,
} from "../src/features/settings/components/agent-command.ts";

// Mirrors DEFAULT_AGENTS in usage-examples.tsx and CODING_AGENTS in
// studio/backend/utils/coding_agents.py minus HIDDEN_AGENTS ("pi").
const VISIBLE_AGENTS = ["claude", "codex", "openclaw", "opencode", "hermes"];

function powerset<T>(items: readonly T[]): T[][] {
  return items.reduce<T[][]>(
    (acc, item) => [...acc, ...acc.map((subset) => [...subset, item])],
    [[]],
  );
}

// Every ordering is not enumerated: the backend returns CODING_AGENTS order and the
// frontend only filters it, so subsets preserve that order.
const DETECTED_SETS = powerset(VISIBLE_AGENTS);

// The component loop: apply the decision until it reports "nothing to change".
function settle(
  detected: readonly string[],
  start: string,
  isGguf: boolean,
  offered: readonly string[],
): { agent: string; steps: number } {
  let agent = start;
  for (let steps = 1; steps <= 10; steps++) {
    const next = pickCompatibleAgent(detected, agent, isGguf, offered);
    if (next === null || next === agent) {
      return { agent, steps };
    }
    agent = next;
  }
  throw new Error(
    `did not settle: detected=${JSON.stringify(detected)} start=${start} isGguf=${isGguf}`,
  );
}

test("the state space is the size we think it is", () => {
  assert.equal(DETECTED_SETS.length, 32);
  assert.equal(DETECTED_SETS.length * VISIBLE_AGENTS.length * 2, 320);
});

test("every reachable state settles, and settles in one step", () => {
  let checked = 0;
  for (const detected of DETECTED_SETS) {
    for (const start of VISIBLE_AGENTS) {
      for (const isGguf of [true, false]) {
        const { steps } = settle(detected, start, isGguf, VISIBLE_AGENTS);
        // More than one step would be a visible flip between paints, not just churn.
        assert.ok(
          steps <= 2,
          `took ${steps} steps: detected=${JSON.stringify(detected)} start=${start}`,
        );
        checked++;
      }
    }
  }
  assert.equal(checked, 320);
});

test("the settled agent always runs on the active model", () => {
  for (const detected of DETECTED_SETS) {
    for (const start of VISIBLE_AGENTS) {
      for (const isGguf of [true, false]) {
        const { agent } = settle(detected, start, isGguf, VISIBLE_AGENTS);
        assert.ok(
          agentRunsOnActiveModel(agent, isGguf),
          `settled on ${agent} with isGguf=${isGguf}, detected=${JSON.stringify(detected)}`,
        );
      }
    }
  }
});

test("the settled agent is always one the panel offers", () => {
  // Under version skew the backend can return a narrower list than DEFAULT_AGENTS;
  // naming an agent with no chip to click would be worse than naming a wrong one.
  for (const offered of DETECTED_SETS.filter((s) => s.length > 0)) {
    for (const detected of [[], offered]) {
      for (const start of offered) {
        for (const isGguf of [true, false]) {
          const { agent } = settle(detected, start, isGguf, offered);
          assert.ok(
            offered.includes(agent),
            `settled on ${agent}, not in offered=${JSON.stringify(offered)}`,
          );
        }
      }
    }
  }
});

test("settling is idempotent", () => {
  for (const detected of DETECTED_SETS) {
    for (const start of VISIBLE_AGENTS) {
      for (const isGguf of [true, false]) {
        const first = settle(detected, start, isGguf, VISIBLE_AGENTS).agent;
        const second = settle(detected, first, isGguf, VISIBLE_AGENTS).agent;
        assert.equal(second, first);
      }
    }
  }
});

test("a GGUF model never steers away from a detected GGUF-only agent", () => {
  // The re-steer has to work in both directions or the picker is one-way sticky.
  for (const detected of DETECTED_SETS) {
    const firstGgufOnly = detected.find((a) => GGUF_ONLY_AGENTS.includes(a));
    if (firstGgufOnly === undefined) continue;
    const { agent } = settle(detected, UNIVERSAL_AGENT, true, VISIBLE_AGENTS);
    assert.equal(agent, detected[0]);
  }
});

test("the pre-PR rule and the current rule differ only where Claude was unrunnable", () => {
  // The old auto-pick, verbatim from usage-examples.tsx before this change.
  const before = (detected: readonly string[], agent: string, isGguf: boolean) => {
    const preferred = detected.find((a) => a !== "codex" || isGguf);
    if (preferred) return preferred;
    if (agent === "codex" && !isGguf) return "claude";
    return null;
  };
  const deltas: string[] = [];
  for (const detected of DETECTED_SETS) {
    for (const start of VISIBLE_AGENTS) {
      for (const isGguf of [true, false]) {
        const b = before(detected, start, isGguf) ?? start;
        const a = pickCompatibleAgent(detected, start, isGguf, VISIBLE_AGENTS) ?? start;
        if (a !== b) {
          // Every difference must be one that removes an unrunnable answer.
          assert.ok(
            !agentRunsOnActiveModel(b, isGguf),
            `changed a runnable answer: ${b} -> ${a} (isGguf=${isGguf})`,
          );
          assert.ok(agentRunsOnActiveModel(a, isGguf));
          deltas.push(`${JSON.stringify(detected)}/${start}/${isGguf}: ${b} -> ${a}`);
        }
      }
    }
  }
  // The change is not a no-op either.
  assert.ok(deltas.length > 0);
});

test("UNIVERSAL_AGENT is an agent the panel actually offers", () => {
  // Pins the coupling to studio/backend/utils/coding_agents.py CODING_AGENTS and to
  // HIDDEN_AGENTS in ../src/features/settings/api/coding-agents.ts. Dropping opencode
  // from either should break here rather than in the UI.
  assert.ok(VISIBLE_AGENTS.includes(UNIVERSAL_AGENT));
  assert.ok(agentRunsOnActiveModel(UNIVERSAL_AGENT, false));
  assert.ok(agentRunsOnActiveModel(UNIVERSAL_AGENT, true));
});

test("fallbackAgent stays inside a narrowed offered list", () => {
  assert.equal(fallbackAgent(false, ["claude", "codex", "hermes"]), "hermes");
  assert.equal(fallbackAgent(false, ["claude", "codex"]), null); // nothing offered runs
  assert.equal(fallbackAgent(false, []), UNIVERSAL_AGENT); // caller had no list
  assert.equal(fallbackAgent(true, ["claude", "codex"]), "claude");
  assert.equal(fallbackAgent(false, VISIBLE_AGENTS), UNIVERSAL_AGENT);
});

test("a panel offering only GGUF-only agents leaves the pick alone", () => {
  // Reachable only under frontend/backend skew, but the honest answer is "no change":
  // there is nothing runnable to move to, so moving would just relabel the problem.
  for (const start of ["claude", "codex"]) {
    assert.equal(pickCompatibleAgent([], start, false, ["claude", "codex"]), null);
    assert.equal(
      pickCompatibleAgent(["claude"], start, false, ["claude", "codex"]),
      null,
    );
  }
});
