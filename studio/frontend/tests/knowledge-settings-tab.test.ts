// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { embeddedSummary } from "../src/features/rag/components/source-folder-copy.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";
import { readSrc } from "./helpers/kit.ts";

test("a knowledge base reports embedding progress only while it is unfinished", () => {
  assert.equal(embeddedSummary({ documentCount: 5, indexedCount: 2 }), " · 2 of 5 embedded (40%)");
  assert.equal(embeddedSummary({ documentCount: 3, indexedCount: 2 }), " · 2 of 3 embedded (66%)");
  assert.equal(embeddedSummary({ documentCount: 5, indexedCount: 5 }), "");
  assert.equal(embeddedSummary({ documentCount: 0, indexedCount: 0 }), "");
  // An older server sends no indexedCount: the row reads as it always did.
  assert.equal(embeddedSummary({ documentCount: 5 }), "");
});

test("knowledge bases get a Settings tab under Data & access, not a sidebar entry", () => {
  assert.ok((SETTINGS_TABS as readonly string[]).includes("knowledge"));
  const dialog = readSrc("features/settings/settings-dialog.tsx");
  assert.match(dialog, /ids: \["sandbox", "accounts", "data", "knowledge", "library"\]/);
  assert.doesNotMatch(readSrc("components/app-sidebar.tsx"), /settings\.tabs\.knowledge/);
});

test("Settings and the chat dialog render the same manager", () => {
  assert.match(readSrc("features/settings/tabs/knowledge-tab.tsx"), /<KnowledgeBaseManager\s/);
  const kbDialog = readSrc("features/rag/components/knowledge-base-dialog.tsx");
  assert.match(kbDialog, /<Dialog open=\{open\} onOpenChange=\{onOpenChange\}>\s*<KnowledgeBaseManager\s/);
  assert.match(kbDialog, /frame=\{\(content\) => \(\s*<DialogContent className="max-w-2xl"/);
});
