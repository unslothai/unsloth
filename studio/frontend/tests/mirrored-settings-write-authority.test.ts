// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const store = readText("../src/features/chat/stores/chat-runtime-store.ts");
const storage = readText("../src/features/chat/utils/chat-settings-storage.ts");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start + from.length);
  assert.ok(start !== -1, `not found: ${from}`);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("hydration backfills only from an authoritative read", () => {
  const hydrate = slice(
    store,
    "hydratePersistedSettings: async () => {",
    "\n  beginModelLoading:",
  );
  // Extra destructured fields are fine; fromServer is the one that gates backfill.
  assert.match(
    hydrate,
    /const \{[\s\S]*?\bsettings,[\s\S]*?\bfromServer,?[\s\S]*?\} = await loadChatSettingsWithLegacyImport\(\);/,
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
    /return \{ settings: legacySettings, fromServer: false,[^}]*\};/,
  );
  // Every other exit reports an answered GET, so absence stays meaningful.
  assert.ok((loader.match(/fromServer: true/g) ?? []).length >= 4);
});
