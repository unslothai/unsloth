// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

// Execute the shipped status patch and rollback payload, like resident-status-baselines.test.ts.
// The browser store graph is not needed to exercise the request/effective round trip.
const applier = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const start = applier.indexOf("    // Rollback needs the request,");
const end = applier.indexOf("    // AFTER that clear,", start);
assert.ok(start >= 0 && end > start);
type State = {
  loadedReasoningBudget: number;
  loadedReasoningBudgetMessage: string;
  loadedReasoningBudgetRequested: number | null;
  loadedReasoningBudgetMessageRequested: string | null;
};
const hydrate = new Function(
  "previous",
  "status",
  "seedLoadParams",
  "reasoningBudgetApplicable",
  `return { ...previous, ${applier.slice(start, end)} };`,
) as (
  previous: State,
  status: Record<string, unknown>,
  settled: boolean,
  gguf: boolean,
) => State;

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const rollbackStart = runtime.indexOf(
  "n_parallel: rollbackState.loadedNParallel,",
);
const rollbackEnd = runtime.indexOf("// omit unset fields:", rollbackStart);
assert.ok(rollbackStart >= 0 && rollbackEnd > rollbackStart);
const rollback = new Function(
  "rollbackState",
  `return { ${runtime.slice(rollbackStart, rollbackEnd)} };`,
) as (state: State) => {
  reasoning_budget: number;
  reasoning_budget_message: string;
};

const inherited: State = {
  loadedReasoningBudget: 512,
  loadedReasoningBudgetMessage: "from env",
  loadedReasoningBudgetRequested: null,
  loadedReasoningBudgetMessageRequested: null,
};

test("adoption then rollback preserves inherited defaults rather than making them explicit", () => {
  const adopted = hydrate(
    inherited,
    {
      reasoning_budget: 512,
      reasoning_budget_message: "from env",
      requested_reasoning_budget: -1,
      requested_reasoning_budget_message: "",
    },
    true,
    true,
  );
  const request = rollback(adopted);
  assert.equal(request.reasoning_budget, -1);
  assert.equal(request.reasoning_budget_message, "");
  assert.equal(adopted.loadedReasoningBudget, 512);
});

test("a same-model reload updates the request even when its effective value is unchanged", () => {
  const explicit = hydrate(
    inherited,
    {
      requested_reasoning_budget: 512,
      requested_reasoning_budget_message: "from env",
    },
    true,
    true,
  );
  assert.equal(rollback(explicit).reasoning_budget, 512);
  const next = hydrate(
    explicit,
    {
      requested_reasoning_budget: -1,
      requested_reasoning_budget_message: "",
    },
    true,
    true,
  );
  assert.equal(rollback(next).reasoning_budget, -1);
  assert.equal(rollback(next).reasoning_budget_message, "");
});

test("zero budget and verbatim messages survive the rollback request", () => {
  const state = hydrate(
    inherited,
    {
      requested_reasoning_budget: 0,
      requested_reasoning_budget_message: "  conclude now  ",
    },
    true,
    true,
  );
  assert.equal(rollback(state).reasoning_budget, 0);
  assert.equal(rollback(state).reasoning_budget_message, "  conclude now  ");
});

test("a mid-load poll and an older backend cannot overwrite a known request", () => {
  const known = {
    ...inherited,
    loadedReasoningBudgetRequested: 128,
    loadedReasoningBudgetMessageRequested: "Stop",
  };
  assert.deepEqual(
    hydrate(
      known,
      {
        requested_reasoning_budget: 0,
        requested_reasoning_budget_message: "other model",
      },
      false,
      true,
    ),
    known,
  );
  assert.deepEqual(
    hydrate(known, { reasoning_budget: 512 }, true, true),
    known,
  );
});

test("non-GGUF and diffusion adoption retires the previous request", () => {
  const cleared = hydrate(
    {
      ...inherited,
      loadedReasoningBudgetRequested: 128,
      loadedReasoningBudgetMessageRequested: "Stop",
    },
    {},
    true,
    false,
  );
  assert.equal(rollback(cleared).reasoning_budget, -1);
  assert.equal(rollback(cleared).reasoning_budget_message, "");
});
