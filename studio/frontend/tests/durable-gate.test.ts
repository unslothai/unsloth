// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

const { BROWSER_EXECUTED_TOOLS, turnRequiresLegacyStream } = await import(
  "../src/features/chat/api/durable-gate.ts"
);

// A turn stays durable (returns false) unless a tool the BROWSER must execute is enabled. The regression these
// tests pin: the gate used to read `requestPayload.tools`, which is absent on the local path and is the caller's
// schema catalog on the passthrough path - so a catalog-bearing turn was forced onto the cancel-on-disconnect
// stream and a closed browser halted generation mid-turn.

test("a local agentic turn with server-executed tools stays durable", () => {
  assert.equal(
    turnRequiresLegacyStream({
      enable_tools: true,
      enabled_tools: ["web_search", "python", "terminal", "edit_file"],
    }),
    false,
  );
});

test("a passthrough turn carrying the caller's schema catalog stays durable", () => {
  assert.equal(
    turnRequiresLegacyStream({
      enable_tools: true,
      enabled_tools: ["web_search"],
      tools: [{ type: "function", function: { name: "get_weather" } }],
    }),
    false,
  );
});

test("a plain text turn stays durable", () => {
  assert.equal(turnRequiresLegacyStream({ enable_tools: false }), false);
});

test("only a browser-executed tool forces the legacy stream", () => {
  assert.equal(BROWSER_EXECUTED_TOOLS.size, 0, "nothing is browser-executed today");
});

test("a media-bearing turn stays durable; media rides the payload, not the gate", () => {
  // The gate used to read this turn's attachments and force a screenshot-carrying turn onto the
  // cancel-on-disconnect stream. Replay is faithful now, so once generation starts the client
  // contributes nothing and the turn is durable like any other; if the backend toggle ever flips
  // media back off, its 400 degrades silently through isLegacyFallbackChatGenerationAdmissionError.
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.ok(
    !adapter.includes("currentTurnCarriesMedia"),
    "media re-entered the durability gate: a screenshot turn would cancel on tab close again",
  );
  // The attachment itself still rides along, scoped to THIS turn's message only - never to thread history.
  assert.match(adapter, /image_base64: findLatestUserImageBase64\(currentTurnMessages\)/);
});
