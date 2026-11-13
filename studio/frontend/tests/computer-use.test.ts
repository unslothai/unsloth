// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { turnRequiresLegacyStream } from "../src/features/chat/api/durable-gate.ts";
import { normalizePlusMenuOrder } from "../src/features/chat/stores/plus-menu-prefs-store.ts";
import {
  COMPUTER_USE_TOOL,
  alwaysAllowableApp,
  computerCallFullySaid,
  computerCallWhy,
  computerUseAsk,
  describeComputerCall,
  plainComputerResult,
} from "../src/features/computer-use/summary.ts";

const describe = (args: unknown) => describeComputerCall(JSON.stringify(args));

test("a computer call reads as steps, in the order they run", () => {
  const chain = [
    { action: "type", text: "hello world", submit: true },
    { action: "key", keys: "cmd+s" },
    { action: "open_app", name: "Safari" },
  ];
  assert.deepEqual(
    describeComputerCall(`{"action": "click", "x": 512.4, "y": 300, "then": ${JSON.stringify(chain)}}`),
    ['Click at (512, 300)', 'Type "hello world" and press Enter', "Press cmd+s", "Open Safari"],
  );
});

test("typed text is shown in full, since the user approves it", () => {
  const long = "x".repeat(300);
  assert.equal(describe({ action: "type", text: long })[0], `Type "${long}"`);
});

test("other dialects still read as steps", () => {
  assert.deepEqual(describe({ action: "left_click", coordinate: [10, 20] }), ["Click at (10, 20)"]);
  assert.deepEqual(describe({ actions: [{ type: "keypress", keys: ["CTRL", "L"] }] }), ["Press CTRL+L"]);
  assert.deepEqual(describe({ action: "drag", x: 1, y: 2, to_x: 30, to_y: 40 }), ["Drag from (1, 2) to (30, 40)"]);
  assert.deepEqual(describe({ action: "scroll", direction: "up" }), ["Scroll up"]);
  assert.deepEqual(describe({ action: "fly" }), ["Do: fly"]);
});

test("the model's note on a call is one short line, and never a step", () => {
  const args = { action: "click", x: 1, y: 2, why: "  Open the\n Downloads folder " };
  assert.equal(computerCallWhy(JSON.stringify(args)), "Open the Downloads folder");
  assert.deepEqual(describe(args), ["Click at (1, 2)"]);
  assert.equal(computerCallWhy(JSON.stringify({ action: "click", why: "y".repeat(200) })).length, 80);
  assert.equal(computerCallWhy(JSON.stringify({ action: "click", why: 7 })), "");
  assert.equal(computerCallWhy('{"action": "cli'), "");
});

test("a call the steps do not fully say is shown as sent too", () => {
  const said = (args: unknown) => computerCallFullySaid(JSON.stringify(args));
  assert.equal(said({ action: "click", x: 1, y: 2, why: "Open", then: [{ action: "type", text: "hi", submit: true }] }), true);
  assert.equal(said({ actions: [{ type: "keypress", keys: ["CTRL", "L"] }] }), true);
  // Another dialect's name for the text, a button, a repeat count, an unknown action.
  assert.equal(said({ action: "type", content: "send this text", press_enter: true }), false);
  assert.equal(said({ action: "click", x: 1, y: 2, button: "right" }), false);
  assert.equal(said({ action: "key", keys: "tab", repeat: 9 }), false);
  assert.equal(said({ action: "input_text", text: "x" }), false);
  assert.equal(said({ action: "click", x: 1, y: 2, then: [{ action: "type", value: "x" }] }), false);
  assert.equal(computerCallFullySaid('{"action": "cli'), false);
});

test("arguments that are still streaming in or are not an object describe nothing", () => {
  assert.deepEqual(describeComputerCall('{"action": "cli'), []);
  assert.deepEqual(describeComputerCall(undefined), []);
  assert.deepEqual(describeComputerCall("[1, 2]"), []);
});

test("the fence around window titles is for the model, not the card", () => {
  const fenced = 'Done.\nIn front: <untrusted_screen_text>Safari: "Apple"</untrusted_screen_text>';
  assert.equal(plainComputerResult(fenced), 'Done.\nIn front: Safari: "Apple"');
  const withPicture = plainComputerResult({ text: fenced, images: [{ data: "aa", mimeType: "image/jpeg" }] });
  assert.deepEqual(withPicture, {
    text: 'Done.\nIn front: Safari: "Apple"',
    images: [{ data: "aa", mimeType: "image/jpeg" }],
  });
  assert.equal(plainComputerResult(undefined), undefined);
});

test("only string reasons and app names reach the approval card", () => {
  assert.deepEqual(computerUseAsk({ risks: ["Terminal is a terminal", 5, null], apps: ["Safari", {}] }), {
    risks: ["Terminal is a terminal"],
    apps: ["Safari"],
  });
  assert.deepEqual(computerUseAsk({ risks: [] }), { risks: [], apps: [] });
  assert.equal(computerUseAsk(undefined), undefined);
  assert.equal(computerUseAsk({ risks: "all of them" }), undefined);
});

test("Always allow is offered for exactly one new app and nothing risky", () => {
  assert.equal(alwaysAllowableApp({ risks: [], apps: ["Safari"] }), "Safari");
  assert.equal(alwaysAllowableApp({ risks: ["cmd+q can close or delete things"], apps: ["Safari"] }), null);
  assert.equal(alwaysAllowableApp({ risks: [], apps: ["Safari", "Mail"] }), null);
  assert.equal(alwaysAllowableApp({ risks: [], apps: [] }), null);
  assert.equal(alwaysAllowableApp(undefined), null);
});

test("the server runs the computer tool, so its turn can outlive the tab", () => {
  assert.equal(turnRequiresLegacyStream({ enabled_tools: [COMPUTER_USE_TOOL] }), false);
  assert.equal(turnRequiresLegacyStream({ enabled_tools: [COMPUTER_USE_TOOL, "browser_click"] }), true);
});

test("an order saved before computer use gains it after browser use", () => {
  const order = normalizePlusMenuOrder(["skills", "browserUse", "deepResearch"]);
  assert.equal(order.indexOf("computerUse"), order.indexOf("browserUse") + 1);
});
