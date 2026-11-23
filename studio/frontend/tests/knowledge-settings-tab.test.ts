// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { embeddedSummary } from "../src/features/rag/components/source-folder-copy.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";
import { readSrc } from "./helpers/kit.ts";

test("a knowledge base reports embedding progress only while documents are embedding", () => {
  assert.equal(embeddedSummary({ indexedCount: 2, indexingCount: 3 }), " · 2 of 5 embedded (40%)");
  assert.equal(embeddedSummary({ indexedCount: 2, indexingCount: 1 }), " · 2 of 3 embedded (66%)");
  // Dividing first gives 56.99... for 57 / 100.
  assert.equal(embeddedSummary({ indexedCount: 57, indexingCount: 43 }), " · 57 of 100 embedded (57%)");
  // Nothing left to embed, even with failed rows: no summary that never reaches 100%.
  assert.equal(embeddedSummary({ indexedCount: 4, indexingCount: 0 }), "");
  // An older server sends neither count: the row reads as it always did.
  assert.equal(embeddedSummary({}), "");
});

test("Knowledge gets a Settings tab under Library, not a sidebar entry", () => {
  assert.ok((SETTINGS_TABS as readonly string[]).includes("knowledge"));
  const dialog = readSrc("features/settings/settings-dialog.tsx");
  assert.match(dialog, /ids: \["sandbox", "accounts", "data", "library", "knowledge"\]/);
  assert.doesNotMatch(readSrc("components/app-sidebar.tsx"), /settings\.tabs\.knowledge/);
});

test("Settings and the chat dialog render the same manager", () => {
  assert.match(readSrc("features/settings/tabs/knowledge-tab.tsx"), /<KnowledgeBaseManager\s/);
  const kbDialog = readSrc("features/rag/components/knowledge-base-dialog.tsx");
  assert.match(kbDialog, /<Dialog open=\{open\} onOpenChange=\{onOpenChange\}>\s*<KnowledgeBaseManager\s/);
  assert.match(kbDialog, /frame=\{\(content\) => \(\s*<DialogContent className="max-w-2xl"/);
});

test("the chat bar's retrieval menu opens Knowledge in Settings", () => {
  const menu = readSrc("features/rag/components/knowledge-base-composer-button.tsx");
  assert.match(menu, /openDialog\("knowledge"\);\s*\}\}\s*>\s*Open Knowledge in Settings/);
  // The existing dialog entry stays, so the chat bar flow is unchanged.
  assert.match(menu, /Manage knowledge bases…/);
});
