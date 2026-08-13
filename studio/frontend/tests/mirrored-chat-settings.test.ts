// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat toggles used to live only in localStorage, so a second browser or a
// remote session started from defaults. They now round-trip through
// /api/chat/settings, and the backend payload is extra="forbid" with literal
// and range constraints: one out-of-contract field 400s the whole save. These
// pin what the client is allowed to send and what it accepts back.

import assert from "node:assert/strict";
import test from "node:test";

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import {
  assignSanitizedMirroredSettings,
  hasNoMirroredSettings,
  normalizeStoredPermissionMode,
  normalizeStoredRagAutoInject,
} from "../src/features/chat/utils/mirrored-chat-settings.ts";

function sanitized(value: Record<string, unknown>): PersistedChatSettings {
  const settings: PersistedChatSettings = {};
  assignSanitizedMirroredSettings(value, settings);
  return settings;
}

test("a full set of mirrored settings survives the round trip", () => {
  const settings = sanitized({
    toolsEnabled: true,
    deepResearchEnabled: false,
    permissionMode: "ask",
    ragMode: "dense",
    ragTopK: 12,
    ragAutoInject: "on",
    ragAutoInjectMinScore: 0.42,
    ragSource: { type: "kb", kbId: "notes" },
    researchWebsitePolicy: {
      allowedDomains: ["unsloth.ai"],
      blockedDomains: [],
    },
    speculativeType: "ngram",
    gpuMemoryMode: "manual",
    fitOnDeviceOnly: true,
  });

  assert.deepEqual(settings, {
    toolsEnabled: true,
    deepResearchEnabled: false,
    permissionMode: "ask",
    ragMode: "dense",
    ragTopK: 12,
    ragAutoInject: "on",
    ragAutoInjectMinScore: 0.42,
    ragSource: { type: "kb", kbId: "notes" },
    researchWebsitePolicy: {
      allowedDomains: ["unsloth.ai"],
      blockedDomains: [],
    },
    speculativeType: "ngram",
    gpuMemoryMode: "manual",
    fitOnDeviceOnly: true,
  });
});

test("a thread RAG source keeps its shape", () => {
  assert.deepEqual(sanitized({ ragSource: { type: "thread" } }), {
    ragSource: { type: "thread" },
  });
});

// Full access disables the sandbox, so it has to be re-accepted each session.
test("the full permission mode is never persisted", () => {
  assert.deepEqual(sanitized({ permissionMode: "full" }), {});
});

test("values outside the backend contract are dropped, not sent", () => {
  assert.deepEqual(
    sanitized({
      toolsEnabled: "yes",
      permissionMode: "elevated",
      ragMode: "vector",
      ragTopK: 0,
      ragAutoInjectMinScore: 2,
      speculativeType: "mtp",
      gpuMemoryMode: "",
      ragSource: { type: "kb" },
      researchWebsitePolicy: "everything",
    }),
    {},
  );
});

// A stored top K is a slider value; a fractional one would 422 the payload.
test("a fractional top K is rejected", () => {
  assert.deepEqual(sanitized({ ragTopK: 7.5 }), {});
});

test("non-string domains are stripped from the research policy", () => {
  assert.deepEqual(
    sanitized({
      researchWebsitePolicy: {
        allowedDomains: ["a.example", 42, null],
        blockedDomains: ["b.example", "x".repeat(300)],
      },
    }),
    {
      researchWebsitePolicy: {
        allowedDomains: ["a.example"],
        blockedDomains: ["b.example"],
      },
    },
  );
});

// Storage predating the three-way control holds the old booleans. Backfilling one
// raw would send a value the backend rejects, so the preference never carries.
test("a legacy RAG auto-inject boolean is migrated before it is sent", () => {
  assert.equal(normalizeStoredRagAutoInject("false"), "off");
  assert.equal(normalizeStoredRagAutoInject("true"), "auto");
  assert.equal(normalizeStoredRagAutoInject("on"), "on");
  assert.equal(normalizeStoredRagAutoInject("off"), "off");
  assert.equal(normalizeStoredRagAutoInject("auto"), "auto");
  assert.equal(normalizeStoredRagAutoInject("nonsense"), "auto");
});

// The legacy localStorage import runs only against a record with nothing in it,
// so a server record holding just the mirrored toggles must not read as empty.
test("a record holding only mirrored settings is not empty", () => {
  assert.equal(hasNoMirroredSettings({}), true);
  assert.equal(hasNoMirroredSettings({ autoTitle: true }), true);
  assert.equal(hasNoMirroredSettings({ ragTopK: 5 }), false);
  assert.equal(hasNoMirroredSettings({ toolsEnabled: false }), false);
});

// Pinned here rather than through the UI: only the first browser to hydrate resolves the level
// from storage, so the mapping can be exercised exactly once per installation.
test("a legacy confirm toggle maps onto the permission level", () => {
  assert.equal(normalizeStoredPermissionMode(null, true), "ask");
  assert.equal(normalizeStoredPermissionMode(null, false), "off");
  assert.equal(normalizeStoredPermissionMode(null, null), "auto");
});

test("a stored permission level wins over the legacy toggle", () => {
  assert.equal(normalizeStoredPermissionMode("off", true), "off");
  assert.equal(normalizeStoredPermissionMode("auto", true), "auto");
  assert.equal(normalizeStoredPermissionMode("ask", false), "ask");
});

// "full" is never stored, so reading one back must not restore the sandbox bypass.
test("an unstorable or unknown level falls through to the derivation", () => {
  assert.equal(normalizeStoredPermissionMode("full", null), "auto");
  assert.equal(normalizeStoredPermissionMode("full", true), "ask");
  assert.equal(normalizeStoredPermissionMode("nonsense", false), "off");
});
