// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGE = readSrc("features/chat/projects-page.tsx");
const PROJECTS = readSrc("features/chat/hooks/use-chat-projects.ts");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const LINKED = readSrc("features/rag/components/use-linked-folders.ts");
const DOCS = readSrc("features/rag/components/use-rag-documents.ts");

test("project recency is chat activity, everywhere a recent list is drawn", () => {
  assert.match(
    PROJECTS,
    /return Math\.max\(project\.lastActivityAt \?\? 0, project\.updatedAt\);/,
  );
  assert.match(PAGE, /projectActivityAt\(b\) - projectActivityAt\(a\)/);
  assert.match(PAGE, /formatUpdated\(projectActivityAt\(project\), t, locale\)/);
  assert.ok(!PAGE.includes("b.updatedAt - a.updatedAt : "), "the page still sorts by edit time");
  assert.match(THREAD, /sortProjectsByActivity\(projects\)\.slice\(0, 3\)/);
});

test("deleting a project on the Projects page clears what the sidebar's delete clears", () => {
  assert.match(
    PAGE,
    /await deleteChatProject\(target\.id, \{ deleteFiles \}\);[\s\S]{0,300}notifyChatHistoryUpdated\(\);[\s\S]{0,200}runtime\.activeProjectId === target\.id\) runtime\.setActiveProjectId\(null\);/,
  );
});

test("an import defaults to Recents, not whichever project the server listed first", () => {
  assert.ok(!PAGE.includes("setImportTargetId(projects[0]"));
  assert.equal(PAGE.match(/setImportTargetId\(null\);/g)?.length, 2);
});

test("exporting every project lists chats once, not once per project", () => {
  assert.ok(!/projects\.map\(\(p\) =>\s*listStoredChatThreads/.test(PAGE));
});

test("streamed chunk saves reload open project rows on a longer lull", () => {
  assert.match(
    PAGE,
    /PROJECT_CHATS_REFRESH_DEBOUNCE_MS,\n\s*PROJECT_CHATS_STREAM_REFRESH_DEBOUNCE_MS,\n\s*\);/,
  );
  assert.match(PAGE, /addEventListener\(CHAT_HISTORY_UPDATED_EVENT, debounce\.onEvent\)/);
});

test("a mount reuses a fresh project list instead of refetching it", () => {
  assert.match(
    PROJECTS,
    /projectsLoaded && Date\.now\(\) - projectsLoadedAt > MOUNT_REVALIDATE_AFTER_MS/,
  );
});

test("linked folders poll fast only while syncing, and not from a hidden tab", () => {
  assert.match(LINKED, /if \(document\.hidden\) return;/);
  assert.match(LINKED, /folder\.status === "syncing" \|\| folder\.activeJobId/);
  assert.match(LINKED, /if \(busy \|\| Date\.now\(\) - lastPoll >= LINKED_FOLDERS_IDLE_POLL_MS\) poll\(\);/);
  assert.match(LINKED, /document\.addEventListener\("visibilitychange", onVisible\);/);
  assert.match(LINKED, /document\.removeEventListener\("visibilitychange", onVisible\);/);
});

test("an idle chat page backs off its wait for a CLI-started model", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    runtime,
    /sawLoad \|\| elapsed < CLI_LOAD_POLL_FAST_MS\n\s*\? CLI_LOAD_POLL_FAST_INTERVAL_MS\n\s*: CLI_LOAD_POLL_IDLE_INTERVAL_MS,/,
  );
  assert.match(runtime, /const CLI_LOAD_POLL_IDLE_INTERVAL_MS = 2_000;/);
  assert.ok(
    !runtime.includes("setTimeout(resolve, 500)"),
    "the wait polls twice a second for its whole minute again",
  );
});

test("a document refresh indexes the previous rows and keeps unchanged ones", () => {
  assert.ok(!DOCS.includes("prev.find((p) => p.id === row.id)"), "the merge is quadratic again");
  assert.match(DOCS, /const prevById = new Map\(prev\.map\(\(p\) => \[p\.id, p\]\)\);/);
  assert.match(DOCS, /return tracked && sameDocument\(tracked, next\) \? tracked : next;/);
  assert.match(DOCS, /next\.every\(\(row, i\) => row === prev\[i\]\)\n\s*\? prev/);
});
