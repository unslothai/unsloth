// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { agentHistoryEnabledFor, agentHistoryLevelFor, agentHistoryWindowFor } from "../src/features/chat/stores/agent-history-preference-policy.ts";
import type { AgentHistoryPreferenceSnapshot } from "../src/features/chat/stores/agent-history-preference-policy.ts";

const initial: AgentHistoryPreferenceSnapshot = {
  globalEnabled: false,
  globalWindowRows: 5,
  enabledThreads: { optedIn: true },
  disabledThreads: {},
  windowRowsByThreadId: {},
};

test("legacy per-chat opt-in is preserved when the global switch is off", () => {
  assert.equal(agentHistoryEnabledFor(initial, "optedIn"), true);
  assert.equal(agentHistoryEnabledFor(initial, "newChat"), false);
  assert.equal(agentHistoryEnabledFor(initial, null), false);
  assert.equal(
    agentHistoryEnabledFor({ ...initial, globalEnabled: true, globalDomLevel: "dynamic-high" }, null),
    false,
    "a missing thread identity must not activate the virtualizer",
  );
  assert.equal(
    agentHistoryLevelFor({ ...initial, globalEnabled: true, globalDomLevel: "dynamic-high" }, null),
    "off",
  );
});

test("global switch covers all chats and permits a per-chat exception", () => {
  const global = { ...initial, globalEnabled: true };
  assert.equal(agentHistoryEnabledFor(global, "newChat"), true);
  assert.equal(agentHistoryEnabledFor(global, "optedIn"), true);
  const overridden = { ...global, disabledThreads: { optedIn: true as const } };
  assert.equal(agentHistoryEnabledFor(overridden, "optedIn"), false);
  assert.equal(agentHistoryEnabledFor(overridden, "newChat"), true);
});

test("a per-chat size overrides the global default without changing any messages", () => {
  assert.equal(agentHistoryWindowFor(initial, "newChat"), 5);
  const global = { ...initial, globalWindowRows: 8 as const };
  assert.equal(agentHistoryWindowFor(global, "newChat"), 8);
  assert.equal(agentHistoryWindowFor({ ...global, windowRowsByThreadId: { optedIn: 32 } }, "optedIn"), 32);
});

test("effective enabled decision always equals the canonical resolved mode", () => {
  const cases: AgentHistoryPreferenceSnapshot[] = [
    initial,
    { ...initial, globalEnabled: true },
    { ...initial, globalDomLevel: "low" },
    { ...initial, globalEnabled: false, globalDomLevel: "dynamic-high" },
    { ...initial, globalEnabled: true, globalDomLevel: "fixed-low" },
    { ...initial, globalEnabled: true, domLevelByThreadId: { optedIn: "off" } },
    { ...initial, domLevelByThreadId: { newChat: "dynamic-high" } },
    { ...initial, domLevelByThreadId: { newChat: "fixed-high" } },
    {
      ...initial,
      globalEnabled: true,
      disabledThreads: { optedIn: true },
      domLevelByThreadId: { optedIn: "dynamic-low" },
    },
  ];
  for (const state of cases) {
    for (const threadId of ["optedIn", "newChat", null]) {
      assert.equal(
        agentHistoryEnabledFor(state, threadId),
        agentHistoryLevelFor(state, threadId) !== "off",
      );
    }
  }
  assert.equal(
    agentHistoryEnabledFor({
      ...initial, domLevelByThreadId: { newChat: "dynamic-high" },
    }, "newChat"),
    true,
    "an explicit per-chat mode must activate even if global enable is false",
  );
});
