// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/computer-resolver.mjs", import.meta.url);

const { useComputerAgentStore } = await import("../src/features/computer/agent-store.ts");
const { parseComputerAction } = await import("../src/features/computer/executor.ts");
const { COMPUTER_AGENT_TOOLS, computerAgentToolsForTurn } = await import("../src/features/computer/tools.ts");

test("computer agent tools offered only when enabled (#15532)", () => {
  useComputerAgentStore.getState().setEnabled(false);
  assert.deepEqual(computerAgentToolsForTurn(), []);

  useComputerAgentStore.getState().setEnabled(true);
  const tools = computerAgentToolsForTurn();
  assert.equal(tools.length, COMPUTER_AGENT_TOOLS.length);
  assert.ok(tools.includes("computer_screenshot"));
  assert.ok(tools.includes("computer_click"));
  assert.ok(tools.includes("computer_move_mouse"));
  assert.ok(tools.includes("computer_drag"));
  assert.ok(tools.includes("computer_type"));
  assert.ok(tools.includes("computer_press_key"));
  assert.ok(tools.includes("computer_scroll"));
  assert.ok(tools.includes("computer_interval_screenshot"));
});

test("parseComputerAction validates and standardizes desktop requests (#15532)", () => {
  assert.equal(parseComputerAction(null), null);
  assert.equal(parseComputerAction({}), null);
  assert.equal(parseComputerAction({ id: "" }), null);

  const clickAction = parseComputerAction({
    id: "act_123",
    session_id: "sess_abc",
    op: "click",
    x: 450,
    y: 320,
    button: "right",
    click_count: 2,
    modifiers: ["ctrl"],
  });
  assert.ok(clickAction);
  assert.equal(clickAction.id, "act_123");
  assert.equal(clickAction.sessionId, "sess_abc");
  assert.equal(clickAction.op, "click");
  assert.equal(clickAction.x, 450);
  assert.equal(clickAction.y, 320);
  assert.equal(clickAction.button, "right");
  assert.equal(clickAction.clickCount, 2);
  assert.deepEqual(clickAction.modifiers, ["ctrl"]);

  const typeAction = parseComputerAction({
    id: "act_456",
    session_id: "sess_abc",
    op: "type",
    text: "Testing Visual Computer Use",
    press_enter: true,
  });
  assert.ok(typeAction);
  assert.equal(typeAction.op, "type");
  assert.equal(typeAction.text, "Testing Visual Computer Use");
  assert.equal(typeAction.pressEnter, true);
});

test("computer agent store manages permissions and emergency abort (#15532)", () => {
  const store = useComputerAgentStore.getState();

  store.setPermissionMode("restricted");
  assert.equal(useComputerAgentStore.getState().permissionMode, "restricted");

  store.setPermissionMode("automatic");
  assert.equal(useComputerAgentStore.getState().permissionMode, "automatic");

  store.setIntervalCapture(true, 10);
  assert.equal(useComputerAgentStore.getState().intervalCaptureEnabled, true);
  assert.equal(useComputerAgentStore.getState().intervalSeconds, 10);

  store.resetAbort();
  assert.equal(useComputerAgentStore.getState().abortRequested, false);

  store.abortControl();
  assert.equal(useComputerAgentStore.getState().abortRequested, true);
  assert.equal(useComputerAgentStore.getState().activeAction, null);
});
