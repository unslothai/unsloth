// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getMcpPresets } from "../src/features/chat/mcp-presets.ts";

test("desktop offers Cua Driver as a local stdio MCP preset", () => {
  const preset = getMcpPresets(true).find(({ id }) => id === "cua-driver");

  assert.deepEqual(preset, {
    id: "cua-driver",
    displayName: "Cua Driver",
    url: "cua-driver mcp",
    label: "Cua Driver (Computer Use)",
    hint: "Requires Cua Driver on PATH. Install from cua.ai/docs/cua-driver",
  });
});

test("browser clients do not advertise a local desktop command", () => {
  assert.equal(
    getMcpPresets(false).some(({ id }) => id === "cua-driver"),
    false,
  );
});

test("desktop keeps the existing remote MCP presets", () => {
  const ids = getMcpPresets(true).map(({ id }) => id);

  for (const id of ["unsloth-docs", "context7", "exa", "huggingface"]) {
    assert.ok(ids.includes(id), `missing ${id}`);
  }
});
