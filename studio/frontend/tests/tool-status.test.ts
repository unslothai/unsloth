// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  NUDGE_TOOL_CALLS_STATUS,
  toolStatusKind,
} from "../src/features/chat/utils/tool-status.ts";

test("the nudge status is the exact string the backend sends", () => {
  // Mirrors tool_call_parser.py, so a reword on either side must break here.
  assert.equal(NUDGE_TOOL_CALLS_STATUS, "Nudging tool calls");
  assert.equal(toolStatusKind(NUDGE_TOOL_CALLS_STATUS), "nudge");
});

test("sandbox tools keep the terminal glyph", () => {
  for (const status of [
    "Running Python: print(1)",
    "Running Python...",
    "Running: ls -la",
    "Running command...",
  ]) {
    assert.equal(toolStatusKind(status), "terminal", status);
  }
});

test("every other status keeps the globe", () => {
  for (const status of [
    "Searching: red square",
    "Reading: unsloth.ai",
    "Reading page...",
    "Searching documents: quarterly report",
    "Calling: get_weather",
  ]) {
    assert.equal(toolStatusKind(status), "web", status);
  }
});

test("a status that merely mentions nudging is not the nudge itself", () => {
  // Exact match only: a tool named after the phrase must not steal the spinner.
  assert.equal(toolStatusKind("Calling: Nudging tool calls"), "web");
  assert.equal(toolStatusKind("Nudging tool calls again"), "web");
});
