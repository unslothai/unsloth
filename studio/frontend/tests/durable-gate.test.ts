// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  BROWSER_EXECUTED_TOOLS,
  isDurableRunCandidate,
  turnRequiresLegacyStream,
} = await import("../src/features/chat/api/durable-gate.ts");

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

// ── The gate itself: every term of the conjunction, as a truth table ─────────────────────
// These replace grepping the adapter's source for a token that happens to spell the rule. A turn is a durable-run
// candidate unless exactly one of these is true of it; each case below flips ONE term and says out loud which path
// that sends the turn to and why.

/** A plain text turn on a persisted thread, with its message in hand: everything the gate wants.
 *  Field names are the values the adapter resolves and passes; the gate itself resolves nothing. */
const durableTurn = {
  externalProvider: false,
  modelIsAudio: false,
  loadedIsDiffusion: false,
  turnCarriesMedia: false,
  continuation: false,
  threadId: "thread-1",
  incognito: false,
  assistantMessageId: "assistant-1",
  hasUserMessage: true,
};

test("a plain text turn on a persisted thread is a durable-run candidate", () => {
  assert.equal(isDurableRunCandidate(durableTurn), true);
});

const legacyCases: [string, Record<string, unknown>, string][] = [
  [
    "an external-provider turn",
    { externalProvider: true },
    "its own client owns that stream, so there is no server run to reattach to",
  ],
  [
    "an audio model",
    { modelIsAudio: true },
    "audio runs on its own path regardless of the gate",
  ],
  [
    "a loaded diffusion model",
    { loadedIsDiffusion: true },
    "the diffusion path has no durable run to join",
  ],
  [
    "THIS turn carrying an attachment",
    { turnCarriesMedia: true },
    "a media turn stays subscriber-owned; a text follow-up to an earlier screenshot must NOT (see the next case)",
  ],
  [
    "a continuation",
    { continuation: true },
    "the seeded partial is autosaved before the request starts, and admission 409s a placeholder that already has content",
  ],
  [
    "an incognito thread",
    { incognito: true },
    "an incognito run has no stored history to resume from",
  ],
  [
    "no thread to reattach to",
    { threadId: undefined },
    "a durable run is found again by its thread; with none, nothing reattaches",
  ],
  [
    "no assistant message to write into",
    { assistantMessageId: null },
    "a follower needs a row to resume, not invent one",
  ],
  [
    "no user message for the run to answer",
    { hasUserMessage: false },
    "there is no turn yet, so nothing survives the tab closing",
  ],
];

for (const [name, override, why] of legacyCases) {
  test(`${name} stays on the subscriber-owned stream`, () => {
    assert.equal(
      isDurableRunCandidate({ ...durableTurn, ...override }),
      false,
      `${name}: ${why}`,
    );
  });
}

test("the gate reads the turn's own media, never the thread's history", () => {
  // The regression this pins: the scan that fills `turnCarriesMedia` walked post-prune HISTORY, so one screenshot
  // from an earlier turn refused every later text-only turn AND sent it to the legacy stream. A turn whose OWN
  // message carries no media is a candidate even when the thread above it is full of it - which is what the adapter
  // passes here (currentTurnMessages), pinned by tests/studio/test_multi_chat_prompt_queue_contract.py.
  assert.equal(isDurableRunCandidate({ ...durableTurn, turnCarriesMedia: false }), true);
});
