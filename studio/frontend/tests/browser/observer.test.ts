// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { observedTarget } from "../../src/features/browser/observer-state.ts";

test("observation stays separate from the viewer's target and ends when ownership ends", () => {
  const tabs = [{ id: "a", chat: "A" }, { id: "b", chat: "B" }];
  const agents = { A: { shared: true, tabId: "a" }, B: { shared: true, tabId: "b" } };
  const selection = { chat: "B", tabId: "a" };
  assert.equal(observedTarget("B", selection, tabs, agents)?.id, "a");
  assert.equal(observedTarget("B", selection, tabs, { ...agents, B: { shared: true, tabId: "new" } })?.id, "a");
  assert.equal(observedTarget("A", selection, tabs, agents), undefined);
  assert.equal(observedTarget("B", { chat: "B", tabId: "b" }, tabs, agents), undefined);
  assert.equal(observedTarget("B", selection, tabs, { ...agents, A: { shared: false, tabId: "a" } })?.id, "a");
  assert.equal(observedTarget("B", selection, tabs, { ...agents, A: { shared: false, tabId: null } }), undefined);
  assert.equal(observedTarget("B", selection, tabs, { ...agents, A: { shared: true, tabId: "new" } }), undefined);
});
