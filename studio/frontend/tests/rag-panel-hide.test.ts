// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Chat with files panel's X hides the panel; retrieval and the composer's RAG tag stay.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const bar = readSrc("features/rag/components/thread-documents-bar.tsx");
const store = readSrc("features/chat/stores/chat-runtime-store.ts");
const thread = readSrc("components/assistant-ui/thread.tsx");

test("the panel X hides the panel instead of turning RAG off", () => {
  assert.doesNotMatch(bar, /onClose=\{\(\) => setRagEnabled\(false\)\}/);
  assert.equal(bar.match(/onClose=\{\(\) => setRagPanelHidden\(true\)\}/g)?.length, 2);
  assert.match(bar, /if \(ragPanelHidden\) \{\n\s*return kbDialog;\n\s*\}/);
});

test("turning RAG on, picking a source or switching chats shows the panel again", () => {
  assert.match(store, /ragEnabled,\n\s*ragPanelHidden: false,/);
  assert.match(store, /ragSource,\n\s*ragPanelHidden: false,/);
  assert.match(bar, /setRagPanelHidden\(false\);\n\s*\}, \[threadItemId\]\);/);
  assert.match(thread, /if \(ragEnabled && ragPanelHidden\) \{\n\s*setRagPanelHidden\(false\);\n\s*return;/);
});

test("header controls use the app's tooltip, not the browser's title", () => {
  const panel = bar.slice(bar.indexOf("function ChatFilesPanel("), bar.indexOf("const ATTACHMENT_TARGETS"));
  assert.doesNotMatch(panel, /\btitle=\{/);
  assert.match(panel, /<HeaderTip label=\{compact \? "Expand files" : "Collapse files"\}>/);
  assert.match(panel, /<HeaderTip label=\{closeLabel \?\? "Close"\}>/);
});
