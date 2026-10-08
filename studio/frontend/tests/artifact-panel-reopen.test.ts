// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

// Chat HTML opens in the browser panel; there is no canvas panel of its own.
test("no canvas surface is left to open", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  assert.doesNotMatch(page, /<ArtifactSurface|artifacts\/artifact-surface/);
  assert.doesNotMatch(read("../src/features/browser/store.ts"), /openInCanvas/);
});

test("a card opens its page in a browser tab keyed by the artifact, so reopening focuses it", () => {
  const index = read("../src/features/browser/index.ts");
  assert.match(index, /const htmlOpenKey = \(key: string\) => `file:html:\$\{key\}`;/);
  // openFile prefixes "file:" to the key it is given.
  assert.match(index, /key: `html:\$\{key\}` \}\);/);
  const card = read("../src/features/chat/artifacts/artifact-card.tsx");
  assert.match(card, /key: artifact\.id,/);
});

test("a panel dragged shut closes the browser, not left at no width", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const remember = page.indexOf("const rememberArtifactPanelWidth");
  assert.ok(remember > 0);
  const closeAt = page.indexOf("closeBrowser();", remember);
  const widthAt = page.indexOf("artifactPanelWidthRef.current = `", remember);
  assert.ok(closeAt > 0 && closeAt < widthAt, "a shut panel still records a width");
});

test("the card hides the browser it is already showing and says so on aria-expanded", () => {
  const card = read("../src/features/chat/artifacts/artifact-card.tsx");
  assert.match(card, /if \(shownView === view\) \{\s*useBrowserStore\.getState\(\)\.closePanel\(\);/);
  assert.match(card, /aria-expanded=\{shownView === "preview"\}/);
});

test("a page still being written is not opened, and opens once it is complete", () => {
  const card = read("../src/features/chat/artifacts/artifact-card.tsx");
  assert.match(card, /disabled=\{isStreaming\}/);
  assert.match(card, /if \(!autoOpen \|\| isStreaming \|\| autoOpenAttemptedRef\.current\) return;/);
});

test("the panel's resize handle opts out of the double-click reset", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const handle = page.indexOf("<ResizableHandle", page.indexOf("const rememberArtifactPanelWidth"));
  assert.ok(handle > 0);
  assert.match(page.slice(handle, handle + 300), /disableDoubleClick/);
});

test("HTML runs whole in the browser; only the source view is capped", () => {
  const view = read("../src/features/browser/file-view.tsx");
  assert.match(view, /const bytes = kind === "html" \? blob\.size : MAX_TEXT_BYTES;/);
  assert.match(view, /<ArtifactHtmlFrame\s+code=\{text\}/);
  assert.match(view, /<CodeSourceView code=\{preview\.text\}/);
});

test("the browser over the chat is a modal: focus moves in, is trapped, and Escape closes it", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const overlay = page.slice(page.indexOf("function BrowserOverlay"), page.indexOf("const ProjectSourcesPanel"));
  assert.match(overlay, /role="dialog"/);
  assert.match(overlay, /aria-modal=\{true\}/);
  assert.match(overlay, /\(focusableIn\(dialog\)\[0\] \?\? dialog\)\.focus\(\)/);
  assert.match(overlay, /previous instanceof HTMLElement && previous\.isConnected\) previous\.focus\(\)/);
  assert.match(overlay, /event\.key !== "Tab"/);
  // Background tabs stay mounted but hidden; the trap counts only what is on screen.
  assert.match(page, /!element\.closest\('\[aria-hidden="true"\], \[inert\]'\) &&\s*element\.getClientRects\(\)\.length > 0 &&/);
  // Menus, fields and annotating keep their own Escape.
  assert.match(overlay, /event\.defaultPrevented \|\| typing \|\| useBrowserStore\.getState\(\)\.annotateTabId !== null/);
});

test("compare's hidden base view leaves the browser to the overlay, so a page runs once", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  assert.match(page, /const showBrowserPanel = !showResearchPanel && !isMobile && !browserOverlaid && browserOpen;/);
  assert.match(page, /<BrowserOverlaidContext\.Provider value=\{baseBackgrounded\}>/);
  assert.match(page, /const showBrowserOverlay =\s*active &&\s*browserOpen &&\s*\(view\.mode === "compare" \|\|\s*isMobile \|\|/);
});

test("a project has no side pane, so a browser opened from it shows over it, not one left open before", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  assert.match(page, /setProjectBrowserBaseline\(projectViewId \? useBrowserStore\.getState\(\)\.openSequence : null\);/);
  assert.match(page, /\(view\.mode === "project" &&\s*projectBrowserBaseline !== null &&\s*browserOpenSequence > projectBrowserBaseline\)/);
});

test("opening the browser closes Research, which would otherwise hide it in the side pane", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  assert.match(page, /if \(handledBrowserOpenRef\.current === browserOpenSequence\) return;/);
  assert.match(page, /if \(showResearchPanel && browserOpen && !browserOverlaid\) closeResearchPanel\(\);/);
});

test("Request edits stays in overlays, staging the prompt for the visible composer", () => {
  const panel = read("../src/features/browser/browser-panel.tsx");
  assert.doesNotMatch(panel, /\{requestEdits \|\| canAnnotate \? \(/);
  assert.equal(panel.match(/\(requestEdits \?\? stageEditsPrompt\)\(/g)?.length, 2);
  assert.match(read("../src/features/browser/file-view.tsx"), /onFixWithModel=\{requestEdits \?\? stageEditsPrompt\}/);
  const stage = read("../src/features/browser/stage-edits.ts");
  assert.match(stage, /stageFixPrompt\(prompt\);\s*if \(!browserPanelAvailable\(\)\) useBrowserStore\.getState\(\)\.closePanel\(\);/);
});

test("a project's browser overlay stages Request edits in the project composer", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const landing = page.slice(page.indexOf("function ProjectLanding"));
  assert.match(landing, /useStagedFixPrompt\(\s*useChatArtifactsStore\(\(state\) => state\.pendingFixPrompt\),\s*active,\s*\);/);
  assert.match(page, /useStagedFixPrompt\(pendingFixPrompt, chatActive\);/);
});

test("leaving Chat and coming back keeps a new chat's pages open", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  // outside Chat, activeThreadId is cleared and restored, so ?new uses the first shown thread id.
  assert.match(page, /newChatRef\.current\?\.nonce !== search\.new\s*\) \{\s*newChatRef\.current = \{ nonce: search\.new, threadId: activeThreadId \};/);
  assert.match(page, /view\.newThreadNonce === newChat\.nonce \? newChat\.threadId : null;/);
  const key = page.slice(page.indexOf("const shownChatKey ="), page.indexOf("closeChatPages();"));
  assert.match(key, /view\.threadId \?\? newChatShownId \?\? activeThreadId/);
});

test("a new chat started inside a project closes the previous project chat's pages", () => {
  const page = read("../src/features/chat/chat-page.tsx");
  const key = page.slice(page.indexOf("const shownChatKey ="), page.indexOf("closeChatPages();"));
  // the project URL stays ?project=, so projectNewThreadNonce distinguishes its chats.
  assert.match(key, /`project:\$\{view\.projectId\}:\$\{projectNewThreadNonce\}`/);
});
