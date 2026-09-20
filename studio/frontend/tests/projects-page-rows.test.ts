// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Projects page is a list of folders with nothing to say about what is in them: opening a
// project was the only way to find out, and pinning one meant going through a menu.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";
import { formatWorkedFor } from "../src/lib/format-worked-for.ts";

const PAGE = await readSrcAsync("features/chat/projects-page.tsx");

test("a project row opens its chats in place", () => {
  assert.match(PAGE, /aria-label=\{chatsOpen \? "Hide chats" : "Show chats"\}/);
  assert.match(PAGE, /aria-expanded=\{chatsOpen\}/);
  // Loaded the first time the row is opened, and again when chat history changes.
  assert.match(PAGE, /if \(projectChats\[projectId\] !== undefined\) return;\n\s*loadProjectChats\(projectId\);/);
  assert.match(
    PAGE,
    /listStoredChatThreads\(\{ projectId, includeArchived: false \}\)/,
  );
  assert.match(PAGE, /window\.addEventListener\(CHAT_HISTORY_UPDATED_EVENT, refresh\);/);
  // Debounced, since streaming fires the event per chunk, and open rows reload in place.
  assert.match(PAGE, /timer = setTimeout\(\(\) => \{[\s\S]*?for \(const id of open\) loadProjectChats\(id, true\);\n\s*\}, PROJECT_CHATS_REFRESH_DEBOUNCE_MS\);/);
  assert.match(PAGE, /if \(!silent\) \{\n\s*setProjectChats\(\(prev\) => \(\{ \.\.\.prev, \[projectId\]: "loading" \}\)\);/);
  // A closed project's pending load is invalidated with its cache entry.
  assert.match(PAGE, /for \(const \[id, seq\] of loadSeqRef\.current\) \{\n\s*if \(!open\.has\(id\)\) loadSeqRef\.current\.set\(id, seq \+ 1\);/);
  // A response a newer request overtook is dropped.
  assert.match(PAGE, /if \(loadSeqRef\.current\.get\(projectId\) !== seq\) return;\n\s*setProjectChats/);
  // Grouped as the sidebar groups them, newest first, and a row with none says so.
  assert.match(PAGE, /groupThreads\(threads\)\.sort\(\n\s*\(a, b\) => b\.updatedAt - a\.updatedAt,/);
  assert.match(PAGE, /No chats<\/p>/);
  // Each one opens the chat it names; a comparison opens as one.
  assert.match(PAGE, /onClick=\{\(\) => openChat\(chat, project\.id\)\}/);
  assert.match(PAGE, /search: \{ compare: item\.id, project: projectId \}/);
  // A key on a control inside the row is that control's, not the row's.
  assert.match(PAGE, /if \(e\.target !== e\.currentTarget\) return;\n\s*if \(e\.key === "Enter" \|\| e\.key === " "\)/);
});

test("pinning a project takes one click, and says which way it goes", () => {
  assert.match(PAGE, /aria-label=\{pinned \? "Unpin project" : "Pin project"\}/);
  assert.match(PAGE, /togglePinProject\(project\.id\);/);
  // A pinned row keeps its pin showing; the rest reveal on hover.
  assert.match(PAGE, /pinned \? "opacity-100" : "opacity-0",/);
  // And the menu no longer repeats the noun the row already is.
  assert.match(PAGE, /<span>\{pinned \? "Unpin" : "Pin"\}<\/span>/);
});

// "Thought for 216 seconds" is neither what the model did nor a readable duration.
test("a finished run says how long it worked, in units that read", () => {
  assert.equal(formatWorkedFor(0), "0s");
  assert.equal(formatWorkedFor(45), "45s");
  assert.equal(formatWorkedFor(60), "1m 0s");
  assert.equal(formatWorkedFor(216), "3m 36s");
  assert.equal(formatWorkedFor(3600), "1h 0m");
  assert.equal(formatWorkedFor(3840), "1h 4m");
  // Never a negative, whatever the clock did.
  assert.equal(formatWorkedFor(-5), "0s");
});

test("the reasoning header says Worked for", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(reasoning, /<span>Worked for \{formatWorkedFor\(duration \?\? 0\)\}<\/span>/);
  assert.ok(!reasoning.includes("Thought for"), "the old label is still rendered");
});
