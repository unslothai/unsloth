// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CHAT_SEARCH_TABS,
  type ChatSearchKind,
  type ChatSearchRow,
  filterRows,
  recentRows,
  stepTab,
  tabStepForKey,
} from "../src/features/chat/utils/chat-search-tabs.ts";

const DIALOG = await readFile(
  new URL("../src/features/chat/components/chat-search-dialog.tsx", import.meta.url),
  "utf8",
);
const SOURCES = await readFile(
  new URL("../src/features/chat/hooks/use-chat-search-sources.ts", import.meta.url),
  "utf8",
);

function row(kind: ChatSearchKind, title: string, time: number): ChatSearchRow {
  return { key: `${kind}:${title}`, kind, title, time, haystack: title.toLowerCase() };
}

test("the tabs run All, then each kind a user looks for", () => {
  assert.deepEqual([...CHAT_SEARCH_TABS], ["all", "chats", "projects", "files", "models"]);
});

test("left and right step through the tabs and stop at either end", () => {
  assert.equal(stepTab("all", 1), "chats");
  assert.equal(stepTab("chats", -1), "all");
  assert.equal(stepTab("all", -1), "all");
  assert.equal(stepTab("models", 1), "models");
});

test("the arrows change tab only from the ends of the query, so the caret still moves inside it", () => {
  const at = (value: string, start: number, end = start) => ({ value, selectionStart: start, selectionEnd: end });
  assert.equal(tabStepForKey("ArrowRight", at("", 0)), 1);
  assert.equal(tabStepForKey("ArrowLeft", at("", 0)), -1);
  assert.equal(tabStepForKey("ArrowRight", at("llama", 5)), 1);
  assert.equal(tabStepForKey("ArrowRight", at("llama", 2)), null);
  assert.equal(tabStepForKey("ArrowLeft", at("llama", 2)), null);
  assert.equal(tabStepForKey("ArrowLeft", at("llama", 0, 3)), null);
  assert.equal(tabStepForKey("ArrowDown", at("", 0)), null);
});

test("every query word must appear in a row", () => {
  const rows = [row("projects", "Qwen fine-tune", 1), row("projects", "Llama eval", 2)];
  assert.deepEqual(filterRows(rows, "  "), rows);
  assert.deepEqual(filterRows(rows, "QWEN tune").map((r) => r.title), ["Qwen fine-tune"]);
  assert.deepEqual(filterRows(rows, "qwen eval"), []);
});

test("Recents are the newest rows of every kind together", () => {
  const recents = recentRows(
    {
      chats: [row("chats", "c2", 50), row("chats", "c1", 10)],
      projects: [row("projects", "p", 40)],
      files: [row("files", "f", 60)],
      models: [row("models", "m", 30)],
    },
    3,
  );
  assert.deepEqual(recents.map((r) => r.title), ["f", "c2", "p"]);
});

test("Recents ranks a kind that is not listed newest first", () => {
  const chats = [40, 50, 60, 70, 80].map((t) => row("chats", `c${t}`, t));
  chats.push(row("chats", "created first, used today", 90));
  const recents = recentRows({ chats, projects: [], files: [], models: [] }, 1);
  assert.deepEqual(recents.map((r) => r.title), ["created first, used today"]);
});

test("the dialog keeps chat search's surface, header, rows and shortcut", () => {
  assert.match(DIALOG, /<CommandDialog[\s\S]*className="chat-search-surface /);
  assert.ok(DIALOG.includes('className="flex items-center gap-3 border-b border-border/40 px-4 py-3"'));
  assert.match(DIALOG, /rounded-full px-3 py-2\.5 text-sm outline-hidden data-selected:bg-muted/);
  assert.match(DIALOG, /useShortcut\("searchChats"/);
  assert.match(DIALOG, /role="tablist"/);
});

test("the Library store loads with the dialog, not with the app, and recipes are not searched", () => {
  assert.match(SOURCES, /import\("@\/features\/library\/store"\)/);
  assert.doesNotMatch(SOURCES, /from "@\/features\/library\/store"/);
  assert.doesNotMatch(SOURCES + DIALOG, /data-recipes|recipe/i);
});

test("Models lists every complete download the Hub knows of, plus the Library's fine-tunes", () => {
  assert.match(DIALOG, /useHubInventory\(\{\s*kind: "models",\s*enabled: isOpen,\s*\}\)/);
  assert.match(DIALOG, /cachedRows\s*\.filter\(\s*\(row\) =>\s*!row\.partial &&/);
  assert.match(DIALOG, /localRows\s*\.filter\(\(row\) => !row\.partial\)/);
  assert.match(DIALOG, /search: \{ tab: "downloaded", model: id \}/);
  assert.match(DIALOG, /sources\.fineTunes\.map/);
});

test("Enter never runs a stale action or row, and a focused tab only switches", () => {
  assert.ok(DIALOG.includes("haystackMatches(t(action.labelKey).toLowerCase(), queryTokens(query)),"));
  // cmdk's root runs the highlighted row on Enter, so the tabs stop propagation.
  assert.match(DIALOG, /onClick=\{\(\) => switchTab\(entry\)\}[\s\S]{0,200}?if \(e\.key === "Enter"\) e\.stopPropagation\(\);/);
  assert.match(DIALOG, /value=\{moved && shownKeys\.has\(selected\) \? selected : firstKey\}/);
  assert.equal(DIALOG.match(/setMoved\(false\);\n\s*setSelected\(""\);/g)?.length, 3);
});

test("empty states wait for their source, and chats rank by last activity", () => {
  assert.match(SOURCES, /ready: prev\.ready \|\| isSettled\(useLibraryStore\.getState\(\)\.status\)/);
  assert.match(DIALOG, /files: !sources\.ready,/);
  assert.match(DIALOG, /projects: !projectsLoaded,/);
  assert.match(DIALOG, /time: item\.updatedAt \?\? item\.createdAt,/);
});

test("an action that becomes unavailable leaves the selection keys too", () => {
  assert.match(DIALOG, /available\[action\.id\] &&\s*haystackMatches\(/);
  assert.doesNotMatch(DIALOG.slice(DIALOG.indexOf("function ActionItem(")), /useShortcutAvailable|return null/);
});

test("Enter on a stale auto-highlight runs what the caught-up list shows first", () => {
  assert.match(DIALOG, /if \(moved\) return;\n\s*const first = groupsFor\(live, queryTokens\(query\)\.length > 0\)/);
  assert.match(DIALOG, /else if \(showActions\) go\(\(\) => void triggerShortcut\(visibleActions\[0\]\.id\)\)\(\);/);
});

test("the compact height counts every kind, not only chats", () => {
  assert.match(DIALOG, /isCompactChatSearchList\(true, otherRows \|\| chatSearchIndexHasRows\(\)\)/);
  assert.match(DIALOG, /isCompactChatSearchList\(compactList, otherRows \|\| items\.length > 0\)/);
});

test("model rows are searchable by the labels they show", () => {
  assert.equal(DIALOG.match(/FORMAT_LABELS\[row\.modelFormat\] \?\? "",/g)?.length, 2);
  assert.match(DIALOG, /t\(modelLabelKey\(entry\.item\) \?\? "library\.modelKind\.model"\)/);
  assert.doesNotMatch(DIALOG, /shell\.search\.fineTuned/);
});

test("projects rank by their newest chat, as the sidebar does", () => {
  assert.match(DIALOG, /Math\.max\(project\.updatedAt \?\? project\.createdAt, newestChat\.get\(project\.id\) \?\? 0\)/);
  assert.match(DIALOG, /time: activityAt\(project\),/);
});

test("untitled and compare chats are searchable by their shown labels", () => {
  assert.match(DIALOG, /item\.title \? "" : untitled,/);
  assert.match(DIALOG, /item\.type === "compare" \? compare : "",/);
  assert.match(DIALOG, /selectVisibleChats\(chats, search\)/);
});

test("project ages count empty threads, which the search index skips", () => {
  assert.match(DIALOG, /useChatSidebarItems\(\{\s*enabled: isOpen,\s*requireMessages: false,\s*\}\)/);
  assert.match(DIALOG, /for \(const thread of threads\) \{/);
});

test("infrastructure models stay out of the list until queried", () => {
  assert.match(DIALOG, /hidden: isHiddenModelId\(row\.id, row\.repoId, row\.path, row\.title\),/);
  assert.match(DIALOG, /row\.optimistic && isHiddenModelId\(row\.id, row\.repoId, row\.cachePath\)/);
  assert.match(DIALOG, /: rowsByKind\.models\.filter\(\(row\) => !row\.hidden\),/);
});

test("tab arrows leave IME composition alone", () => {
  assert.match(
    DIALOG,
    /if \(isImeComposing\(event\.nativeEvent\)\) return;\n[^\n]*\n\s*const step = tabStepForKey/,
  );
});
