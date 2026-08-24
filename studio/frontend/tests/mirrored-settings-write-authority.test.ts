// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two writes that reach every browser on the install now that saveBool and the hydration
// backfill mirror to /api/chat/settings, so each needs a reason to fire:
//   1. setCheckpoint's deep-research clamp. Codex runs deep research (thread.tsx exempts
//      openai_codex from researchDisabled, chat-adapter accepts its research requests), so
//      clamping on every external id would turn a Codex user's preference off everywhere.
//   2. The backfill, which reads "field absent" as "unset on the server". That only holds
//      for a GET that answered; the legacy-storage fallback knows nothing about the server.
// The store and chat-settings-storage cannot be imported in a bare node test (a .tsx barrel
// sits in both graphs), so these pin the source the way the sibling store tests do.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

const store = read("../src/features/chat/stores/chat-runtime-store.ts");
const storage = read("../src/features/chat/utils/chat-settings-storage.ts");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start + from.length);
  assert.ok(start !== -1, `not found: ${from}`);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

const setCheckpoint = () =>
  slice(
    store,
    "setCheckpoint: (modelId, ggufVariant, options) =>",
    "\n  setActiveThreadId:",
  );

test("selecting a Codex checkpoint keeps deep research", () => {
  const body = setCheckpoint();
  // Main settled on externalModelSupportsStudioTools for this; the property under test is
  // that the clamp is provider-aware and computed once, not which helper answers it.
  assert.match(
    body,
    /const clampsDeepResearch =\s*isExternalModelId\(modelId\) && !externalModelSupportsStudioTools\(modelId\);/,
  );
  // Every deep-research write in the setter goes through that one clamp rather than
  // re-testing the id, which is what switched it off for capable providers.
  for (const line of body.split("\n")) {
    if (!/deepResearch/i.test(line)) continue;
    assert.doesNotMatch(
      line,
      /isExternalModelId/,
      `deep research still clamped on any external id: ${line.trim()}`,
    );
  }
  assert.match(
    body,
    /if \(clampsDeepResearch\) \{\s*saveBool\(CHAT_DEEP_RESEARCH_ENABLED_KEY, false\);/,
  );
  assert.match(body, /const nextDeepResearchEnabled = clampsDeepResearch/);
  assert.match(
    body,
    /\.\.\.\(clampsDeepResearch \? \{ deepResearchEnabled: false \} : \{\}\)/,
  );
});

// The helper resolves the provider and refuses only a known non-Codex one, so an
// unresolved provider (connection list still loading) never drops the preference.
test("the clamp exempts Codex and an unresolved provider", () => {
  const helper = slice(
    store,
    "function externalCheckpointRefusesDeepResearch(",
    "\n}",
  );
  assert.match(helper, /if \(!parsed\) return false;/);
  assert.match(
    helper,
    /provider != null && provider\.providerType !== "openai_codex"/,
  );
});

test("hydration backfills only from an authoritative read", () => {
  const hydrate = slice(
    store,
    "hydratePersistedSettings: async () => {",
    "\n  beginModelLoading:",
  );
  assert.match(
    hydrate,
    /const \{ settings, fromServer \} = await loadChatSettingsWithLegacyImport\(\);/,
  );
  assert.match(
    hydrate,
    /if \(fromServer\) backfillMirroredSettings\(hydratedSettings\);/,
  );
  // An ungated call is the bug: a failed GET would push this browser's stale values.
  assert.doesNotMatch(hydrate, /\n\s*backfillMirroredSettings\(settings\);/);
});

test("a missing Deep Research timeout keeps the finite default", () => {
  const loadTimeout = slice(
    store,
    "function loadResearchModelTimeoutSeconds(): number {",
    "\n}\n\nfunction loadResearchWebsitePolicy",
  );
  assert.match(
    loadTimeout,
    /const raw = window\.localStorage\.getItem\(CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY\);/,
  );
  assert.match(
    loadTimeout,
    /if \(raw === null\) return DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;/,
  );
  assert.match(
    loadTimeout,
    /catch \{\s*return DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;\s*\}/,
  );
});

test("only the legacy-storage fallback is non-authoritative", () => {
  const loader = slice(
    storage,
    "export async function loadChatSettingsWithLegacyImport(",
    "\nexport async function savePersistedChatSettingsPatch(",
  );
  const nonAuthoritative = loader.match(/fromServer: false/g) ?? [];
  assert.equal(nonAuthoritative.length, 1);
  // ...and it is the branch that never saw a server answer.
  assert.match(
    slice(loader, "} catch (error) {", "\n  const legacySettings"),
    /return \{ settings: legacySettings, fromServer: false \};/,
  );
  // Every other exit reports an answered GET, so absence stays meaningful.
  assert.ok((loader.match(/fromServer: true/g) ?? []).length >= 4);
});
