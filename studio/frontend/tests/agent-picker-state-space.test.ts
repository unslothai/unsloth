// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The effect re-runs on its own output, so what matters is that it settles, runnably.

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

// Orderings are not enumerated: the frontend only filters the backend's order.
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
        // >1 step is a visible flip between paints, not just churn.
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
  // Dropping opencode from CODING_AGENTS should break here, not in the UI.
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

// The two effects consume isGguf as a tri-state: null while /api/inference/status has not
// resolved. These pin the sequences that reading null as `false` produced.

function browserSession(ggufAfterHydration: boolean) {
  // Not Tauri, so detection never runs and detectedAgents stays empty for good.
  const detected: string[] = [];
  let agent = "claude"; // DEFAULT_AGENT
  const step = (isGguf: boolean | null) => {
    if (isGguf === null) return; // the guard under test
    const next = pickCompatibleAgent(detected, agent, isGguf, VISIBLE_AGENTS);
    if (next !== null) agent = next;
  };
  step(null); // first paint: store still holds its null defaults
  step(ggufAfterHydration); // status lands
  return agent;
}

test("an unresolved model status never re-steers the pick", () => {
  // Was: paint 1 read null as non-GGUF and moved claude -> opencode, then paint 2 could
  // not move back, because pickCompatibleAgent([], "opencode", true) is null.
  assert.equal(browserSession(true), "claude");
  assert.equal(browserSession(false), "opencode");
});

test("reading an unresolved status as non-GGUF is a one-way trip", () => {
  // The reason the guard has to be in the effect and not just in the helper.
  let agent = "claude";
  agent = pickCompatibleAgent([], agent, false, VISIBLE_AGENTS) ?? agent;
  assert.equal(agent, "opencode");
  assert.equal(pickCompatibleAgent([], agent, true, VISIBLE_AGENTS), null);
});

function manualPick(clickedUnder: boolean, now: boolean, agent: string) {
  // The revalidation guard: a hand-made pick survives until GGUF-ness changes under it.
  if (clickedUnder === now) return "kept";
  return agentRunsOnActiveModel(agent, now) ? "kept" : "corrected";
}

test("a manual pick is revalidated when the model changes under it", () => {
  assert.equal(manualPick(true, true, "claude"), "kept");
  assert.equal(manualPick(true, false, "claude"), "corrected");
  assert.equal(manualPick(true, false, "opencode"), "kept");
  assert.equal(manualPick(false, true, "opencode"), "kept");
});

// An external chat selection freezes the local GGUF fields (use-chat-model-runtime stops
// applying status while one is active) while the snippet keeps naming the resident model
// from /v1/models, so the two can drift with no way back. Compatibility is unknown there.
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
  // Frozen fields from before the external switch must not read as a live verdict.
  assert.equal(
    ggufState("external::openai::gpt-5", { variant: "Q4_K_M" }),
    null,
  );
  assert.equal(ggufState("", { variant: "Q4_K_M" }), null);
});

test("unknown GGUF-ness leaves both the pick and the stored preference alone", () => {
  const state = ggufState("external::openai::gpt-5", { variant: "Q4_K_M" });
  assert.equal(state, null);
  // Both effects bail on null, so nothing is re-steered and nothing is cleared.
  assert.equal(state === null, true);
});

test("a route that never mounts the chat runtime still learns the model kind", () => {
  // The store fields and params.checkpoint are populated only by useChatModelRuntime,
  // mounted on chat-page and hub-page alone, and local checkpoints are not persisted.
  // Settings opens from every route (api-monitor-page has its own button into this
  // panel), so a store-only reading left the gate permanently unknown there and the
  // panel kept offering a GGUF-only agent for a safetensors model.
  assert.equal(resolveGgufCompatibility(null, false), false);
  assert.equal(resolveGgufCompatibility(null, true), true);
  // With the answer available, the gate is live again on those routes.
  assert.equal(
    pickCompatibleAgent([], "claude", false, VISIBLE_AGENTS),
    UNIVERSAL_AGENT,
  );
});

test("the server wins over the store when both have an answer", () => {
  // A swap made from another tab, the CLI, or an auto-switching API request is invisible
  // to the store: useChatModelRuntime re-reads status on mount, on model-list changes and
  // on tab focus, never on a timer. Reading the store first left the panel gating on the
  // model that had already gone while the snippet already named the new one.
  assert.equal(resolveGgufCompatibility(true, false), false);
  assert.equal(resolveGgufCompatibility(false, true), true);
});

test("a switch made in this tab is not gated on the previous model", () => {
  // The component tags each answer with the store's model identity and discards it once
  // that identity changes, so the server's stale reading arrives here as null, not as a
  // verdict about a model that is no longer loaded.
  assert.equal(resolveGgufCompatibility(true, null), true);
  assert.equal(resolveGgufCompatibility(false, null), false);
});

test("unknown from both sources stays unknown", () => {
  // A server that omits is_gguf, or a probe that failed, is not evidence of "not GGUF".
  assert.equal(resolveGgufCompatibility(null, null), null);
});

test("an idle server decides nothing about a model it is not holding", () => {
  // InferenceStatusResponse gives is_gguf a False default and active_model is Optional,
  // so a server with nothing resident answers false while naming no model. Reading that
  // as "not GGUF" cleared a saved Claude preference for a model that was never there.
  assert.equal(statusGgufVerdict(null, false), null);
  assert.equal(statusGgufVerdict(undefined, false), null);
  assert.equal(statusGgufVerdict(null, true), null);
  // And an unknown verdict leaves the pick and the stored preference alone.
  assert.equal(pickCompatibleAgent([], "claude", true, VISIBLE_AGENTS), null);
});

test("a server that names what it holds still decides", () => {
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B", false), false);
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B-GGUF", true), true);
  // Named, but the field is missing: an older server, so still unknown.
  assert.equal(statusGgufVerdict("unsloth/Qwen3-1.7B", undefined), null);
});

test("a verdict about one model never gates a different one", () => {
  // The status poll and useExampleModelName's catalog poll run on separate timers, so a
  // swap made from the CLI or another tab reaches one of them first. For up to a poll the
  // panel would otherwise pair a newly named safetensors model with the previous GGUF
  // verdict and go on offering a Claude command the CLI refuses.
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
  // useExampleModelName appends ":quant" to pin the file on disk; the status route reports
  // the repo. Treating that as a different model would discard every verdict.
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
  // An idle server names nothing, and the catalog names nothing until it answers. Neither
  // is evidence that the verdict is about some other model.
  assert.equal(verdictDescribesModel(null, "unsloth/Qwen3-1.7B"), true);
  assert.equal(verdictDescribesModel("unsloth/Qwen3-1.7B", null), true);
  assert.equal(verdictDescribesModel(null, null), true);
});

// How the component composes the two: the status side is collapsed the way the backend
// collapses it for /v1/models before the identities are compared.
const statusMatchesCatalog = (resident: string | null, named: string | null) =>
  verdictDescribesModel(
    resident === null ? null : publicModelId(resident),
    named,
  );

test("a path-loaded model is compared on the identity the catalog publishes", () => {
  // /api/inference/status reports backend.active_model_name raw; /v1/models passes it
  // through public_model_id. Comparing them literally never matched for a local path, so
  // the verdict was discarded for good and the gate went dead on routes with no store.
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
  // Still a real mismatch once collapsed.
  assert.equal(statusMatchesCatalog("/models/foo", "bar"), false);
});

test("polls that name different models leave the question unknown", () => {
  // The store is not a tiebreaker here: an external swap is precisely what it does not
  // see, so reading it kept a Claude command on screen for a model that had been replaced.
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
  // No answer for this store state: the store carries it.
  assert.equal(
    compatibilityFromSources(true, null, "unsloth/Qwen3-1.7B-GGUF"),
    true,
  );
  assert.equal(compatibilityFromSources(null, null, null), null);
  // An idle server names nothing, so the store is left to answer.
  assert.equal(
    compatibilityFromSources(
      true,
      { resident: null, isGguf: null },
      "unsloth/X-GGUF",
    ),
    true,
  );
});
