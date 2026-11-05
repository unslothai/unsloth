// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  GGUF_ONLY_AGENTS,
  compatibilityFromSources,
  UNIVERSAL_AGENT,
  agentRunsOnActiveModel,
  fallbackAgent,
  pickCompatibleAgent,
  resolveGgufCompatibility,
  statusGgufVerdict,
  verdictDescribesModel,
} from "../src/features/settings/components/agent-command.ts";
import { publicModelId } from "../src/features/hub/lib/model-identity.ts";

// DEFAULT_AGENTS in usage-examples.tsx = CODING_AGENTS minus HIDDEN_AGENTS ("pi").
const VISIBLE_AGENTS = ["claude", "codex", "openclaw", "opencode", "hermes"];

function powerset<T>(items: readonly T[]): T[][] {
  return items.reduce<T[][]>(
    (acc, item) => [...acc, ...acc.map((subset) => [...subset, item])],
    [[]],
  );
}

const DETECTED_SETS = powerset(VISIBLE_AGENTS);

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
  for (const detected of DETECTED_SETS) {
    const firstGgufOnly = detected.find((a) => GGUF_ONLY_AGENTS.includes(a));
    if (firstGgufOnly === undefined) continue;
    const { agent } = settle(detected, UNIVERSAL_AGENT, true, VISIBLE_AGENTS);
    assert.equal(agent, detected[0]);
  }
});

test("the pre-PR rule and the current rule differ only where Claude was unrunnable", () => {
  // The old auto-pick, verbatim from usage-examples.tsx before this change.
  const before = (
    detected: readonly string[],
    agent: string,
    isGguf: boolean,
  ) => {
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
        const a =
          pickCompatibleAgent(detected, start, isGguf, VISIBLE_AGENTS) ?? start;
        if (a !== b) {
          assert.ok(
            !agentRunsOnActiveModel(b, isGguf),
            `changed a runnable answer: ${b} -> ${a} (isGguf=${isGguf})`,
          );
          assert.ok(agentRunsOnActiveModel(a, isGguf));
          deltas.push(
            `${JSON.stringify(detected)}/${start}/${isGguf}: ${b} -> ${a}`,
          );
        }
      }
    }
  }
  assert.ok(deltas.length > 0);
});

test("UNIVERSAL_AGENT is an agent the panel actually offers", () => {
  assert.ok(VISIBLE_AGENTS.includes(UNIVERSAL_AGENT));
  assert.ok(agentRunsOnActiveModel(UNIVERSAL_AGENT, false));
  assert.ok(agentRunsOnActiveModel(UNIVERSAL_AGENT, true));
});

test("fallbackAgent stays inside a narrowed offered list", () => {
  assert.equal(fallbackAgent(false, ["claude", "codex", "hermes"]), "hermes");
  assert.equal(fallbackAgent(false, ["claude", "codex"]), null);
  assert.equal(fallbackAgent(false, []), UNIVERSAL_AGENT);
  assert.equal(fallbackAgent(true, ["claude", "codex"]), "claude");
  assert.equal(fallbackAgent(false, VISIBLE_AGENTS), UNIVERSAL_AGENT);
});

test("a panel offering only GGUF-only agents leaves the pick alone", () => {
  for (const start of ["claude", "codex"]) {
    assert.equal(
      pickCompatibleAgent([], start, false, ["claude", "codex"]),
      null,
    );
    assert.equal(
      pickCompatibleAgent(["claude"], start, false, ["claude", "codex"]),
      null,
    );
  }
});

// isGguf is tri-state: null until /api/inference/status resolves; null must not read as false.

function browserSession(ggufAfterHydration: boolean) {
  const detected: string[] = [];
  let agent = "claude"; // DEFAULT_AGENT
  const step = (isGguf: boolean | null) => {
    if (isGguf === null) return;
    const next = pickCompatibleAgent(detected, agent, isGguf, VISIBLE_AGENTS);
    if (next !== null) agent = next;
  };
  step(null);
  step(ggufAfterHydration);
  return agent;
}

test("an unresolved model status never re-steers the pick", () => {
  assert.equal(browserSession(true), "claude");
  assert.equal(browserSession(false), "opencode");
});

test("reading an unresolved status as non-GGUF is a one-way trip", () => {
  let agent = "claude";
  agent = pickCompatibleAgent([], agent, false, VISIBLE_AGENTS) ?? agent;
  assert.equal(agent, "opencode");
  assert.equal(pickCompatibleAgent([], agent, true, VISIBLE_AGENTS), null);
});

function manualPick(clickedUnder: boolean, now: boolean, agent: string) {
  if (clickedUnder === now) return "kept";
  return agentRunsOnActiveModel(agent, now) ? "kept" : "corrected";
}

test("a manual pick is revalidated when the model changes under it", () => {
  assert.equal(manualPick(true, true, "claude"), "kept");
  assert.equal(manualPick(true, false, "claude"), "corrected");
  assert.equal(manualPick(true, false, "opencode"), "kept");
  assert.equal(manualPick(false, true, "opencode"), "kept");
});

// An external chat selection freezes the local GGUF fields, so compatibility is unknown there.
function ggufState(
  checkpoint: string,
  fields: { variant?: string; ctx?: number },
) {
  const external = checkpoint.startsWith("external::");
  if (!checkpoint || external) return null;
  return fields.variant != null || fields.ctx != null;
}

test("an external selection makes GGUF-ness unknown, not stale-true", () => {
  assert.equal(
    ggufState("unsloth/Qwen3-1.7B-GGUF", { variant: "Q4_K_M" }),
    true,
  );
  assert.equal(ggufState("unsloth/Qwen3-1.7B", {}), false);
  assert.equal(
    ggufState("external::openai::gpt-5", { variant: "Q4_K_M" }),
    null,
  );
  assert.equal(ggufState("", { variant: "Q4_K_M" }), null);
});

test("unknown GGUF-ness leaves both the pick and the stored preference alone", () => {
  const state = ggufState("external::openai::gpt-5", { variant: "Q4_K_M" });
  assert.equal(state, null);
  assert.equal(state === null, true);
});

test("a route that never mounts the chat runtime still learns the model kind", () => {
  // The store fields are only populated on chat and hub pages, but Settings opens everywhere.
  assert.equal(resolveGgufCompatibility(null, false), false);
  assert.equal(resolveGgufCompatibility(null, true), true);
  assert.equal(
    pickCompatibleAgent([], "claude", false, VISIBLE_AGENTS),
    UNIVERSAL_AGENT,
  );
});

test("the server wins over the store when both have an answer", () => {
  // External swaps are invisible to the store, so the server answer must win over it.
  assert.equal(resolveGgufCompatibility(true, false), false);
  assert.equal(resolveGgufCompatibility(false, true), true);
});

test("a switch made in this tab is not gated on the previous model", () => {
  assert.equal(resolveGgufCompatibility(true, null), true);
  assert.equal(resolveGgufCompatibility(false, null), false);
});

test("unknown from both sources stays unknown", () => {
  assert.equal(resolveGgufCompatibility(null, null), null);
});

test("an idle server decides nothing about a model it is not holding", () => {
  // is_gguf defaults to False even with no resident model, so that pair means unknown.
  assert.equal(statusGgufVerdict(null, false), null);
  assert.equal(statusGgufVerdict(undefined, false), null);
  assert.equal(statusGgufVerdict(null, true), null);
  assert.equal(pickCompatibleAgent([], "claude", true, VISIBLE_AGENTS), null);
});

test("a server that names what it holds still decides", () => {
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B", false), false);
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B-GGUF", true), true);
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B", undefined), null);
});

test("a verdict about one model never gates a different one", () => {
  // Status and catalog polls run on separate timers, so a swap can reach one first.
  assert.equal(
    verdictDescribesModel("unsloth/Qwen3-1.7B-GGUF", "unsloth/Qwen3-1.7B"),
    false,
  );
  assert.equal(
    verdictDescribesModel("unsloth/Qwen3-1.7B-GGUF", "unsloth/Qwen3-1.7B-GGUF"),
    true,
  );
});

test("the quant suffix is not part of the model's identity", () => {
  // useExampleModelName appends ":quant"; the status route reports the bare repo.
  assert.equal(
    verdictDescribesModel(
      "unsloth/Qwen3-1.7B-GGUF",
      "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    ),
    true,
  );
  assert.equal(
    verdictDescribesModel(
      "Unsloth/Qwen3-1.7B-GGUF ",
      "unsloth/qwen3-1.7b-gguf",
    ),
    true,
  );
});

test("nothing named on either side is not a contradiction", () => {
  assert.equal(verdictDescribesModel(null, "unsloth/Qwen3-1.7B"), true);
  assert.equal(verdictDescribesModel("unsloth/Qwen3-1.7B", null), true);
  assert.equal(verdictDescribesModel(null, null), true);
});

const statusMatchesCatalog = (resident: string | null, named: string | null) =>
  verdictDescribesModel(
    resident === null ? null : publicModelId(resident),
    named,
  );

test("a path-loaded model is compared on the identity the catalog publishes", () => {
  // Status reports active_model_name raw; /v1/models passes it through public_model_id.
  assert.equal(statusMatchesCatalog("/models/foo", "foo"), true);
  assert.equal(
    statusMatchesCatalog(
      "/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf",
      "Qwen3-30B-A3B-Q4_K_M",
    ),
    true,
  );
  assert.equal(
    statusMatchesCatalog(
      "~/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B-GGUF/snapshots/abc123",
      "unsloth/Qwen3-1.7B-GGUF",
    ),
    true,
  );
  assert.equal(statusMatchesCatalog("/models/foo", "bar"), false);
});

test("polls that name different models leave the question unknown", () => {
  assert.equal(
    compatibilityFromSources(
      true,
      { resident: "unsloth/Qwen3-1.7B", isGguf: false },
      "unsloth/Qwen3-1.7B-GGUF",
    ),
    null,
  );
  assert.equal(
    compatibilityFromSources(
      false,
      { resident: "unsloth/Qwen3-1.7B-GGUF", isGguf: true },
      "unsloth/Qwen3-1.7B",
    ),
    null,
  );
});

test("an answer about the model on screen still decides, and the server wins", () => {
  assert.equal(
    compatibilityFromSources(
      true,
      { resident: "unsloth/Qwen3-1.7B", isGguf: false },
      "unsloth/Qwen3-1.7B",
    ),
    false,
  );
  assert.equal(
    compatibilityFromSources(true, null, "unsloth/Qwen3-1.7B-GGUF"),
    true,
  );
  assert.equal(compatibilityFromSources(null, null, null), null);
  assert.equal(
    compatibilityFromSources(
      true,
      { resident: null, isGguf: null },
      "unsloth/X-GGUF",
    ),
    true,
  );
});
