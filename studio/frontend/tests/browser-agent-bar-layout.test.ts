// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read from source: the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const BAR = readSrc("features/browser/agent/agent-bar.tsx");
const CSS = readSrc("index.css");

test("the panel body's 1rem reset is layered, so the agent bar's text-ui-12 applies", () => {
  const at = CSS.indexOf(".browser-panel-body > * {");
  assert.notEqual(at, -1);
  const before = CSS.slice(0, at);
  const opened = before.lastIndexOf("@layer base {");
  assert.notEqual(opened, -1);
  // No rule closes the layer between its opening and the reset.
  assert.doesNotMatch(before.slice(opened), /\n}/);
  assert.match(BAR, /className="flex shrink-0 [^"]*\btext-ui-12\b/);
});

test("the agent bar reads Stop, then Deny, Allow, Always allow", () => {
  const order = [
    "browser.agent.stop",
    "browser.agent.deny",
    "browser.agent.allow",
    "browser.agentSettings.alwaysAllow",
  ].map((key) => BAR.indexOf(`t("${key}")`));
  assert.ok(order.every((at) => at > 0));
  assert.deepEqual(
    [...order].sort((a, b) => a - b),
    order,
  );
});
