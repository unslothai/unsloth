// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The attach target can be chosen before the chat exists, so it is parked under
// a pending key until one does. A chat is created by sending a message just as
// often as by attaching a file, and only the second path went through
// ensureThreadId: the choice was dropped, and the next new chat inherited it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const PENDING = "__pending__";
type Target = "project" | "chat";

/** adoptPendingProjectAttachmentTarget, as the store reducer applies it. */
function adopt(
  byThread: Record<string, Target>,
  threadId: string,
): Record<string, Target> {
  const pending = byThread[PENDING];
  if (pending === undefined || threadId in byThread) {
    return byThread;
  }
  const next = { ...byThread };
  delete next[PENDING];
  next[threadId] = pending;
  return next;
}

test("the chat that gets an id takes the choice made before it existed", () => {
  const after = adopt({ [PENDING]: "chat" }, "thread-1");
  assert.equal(after["thread-1"], "chat");
  assert.equal(PENDING in after, false, "and the next new chat starts clean");
});

test("a chat that made its own choice keeps it", () => {
  const before: Record<string, Target> = {
    [PENDING]: "chat",
    "thread-1": "project",
  };
  assert.equal(adopt(before, "thread-1")["thread-1"], "project");
});

test("nothing pending leaves the chat on the saved default", () => {
  const before: Record<string, Target> = { "thread-2": "chat" };
  assert.equal(adopt(before, "thread-1"), before, "no entry, so no override");
});

/** clearPendingProjectAttachmentTarget, as the store reducer applies it. */
function clearPending(
  byThread: Record<string, Target>,
): Record<string, Target> {
  if (!(PENDING in byThread)) {
    return byThread;
  }
  const next = { ...byThread };
  delete next[PENDING];
  return next;
}

// A composer abandoned without sending or attaching left its choice parked, and
// the next fresh chat read it as its own.
test("a choice made in a composer that never became a chat is dropped", () => {
  const after = clearPending({ [PENDING]: "chat", "thread-1": "project" });
  assert.equal(PENDING in after, false);
  assert.equal(after["thread-1"], "project", "real chats are untouched");
});

test("clearing with nothing pending changes nothing", () => {
  const before: Record<string, Target> = { "thread-1": "chat" };
  assert.equal(clearPending(before), before);
});

// Adoption has to run on both paths that turn a fresh composer into a chat.
test("both chat-creating paths adopt the pending choice", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Attaching a file first: ensureThreadId materializes the thread.
  assert.match(
    source,
    /initialize\(\)[\s\S]{0,300}?adoptPendingProjectAttachmentTarget\(remoteId\)/,
  );
  // Sending a message first: the id arrives as a prop.
  assert.match(
    source,
    /if \(!hadThreadId\) \{[\s\S]{0,200}?adoptPendingProjectAttachmentTarget\(threadId\)/,
  );
});

test("the composer clears its pending choice when it goes away", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /clearPendingProjectAttachmentTarget\(\)/);
});

// Membership is read from the chat's own row, so there is a window where it is
// unknown. Attaching in it would file the file by guess.
test("attaching is held until the chat's project is known", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /const projectUnresolved = threadProjectId === undefined;/,
    "unresolved has to be distinguishable from no project",
  );
  assert.match(
    source,
    /uploading \|\| projectUploading \|\| projectUnresolved/,
    "the attach controls hold",
  );
  assert.match(
    source,
    /if \(projectUnresolved\) \{\s*return;\s*\}/,
    "and a desktop drop stays in the store rather than draining",
  );
});

// A send outlives navigation, and hydration and a model load both run before the
// project is first resolved.
test("the run keeps the project it started in", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /const composerProjectIdAtSend =\s*useChatRuntimeStore\.getState\(\)\.activeProjectId \?\? null;\s*await useChatRuntimeStore\.getState\(\)\.hydratePersistedSettings\(\);/,
    "captured before the first await",
  );
  assert.match(
    source,
    /rememberComposerProjectForRun\(resolvedThreadId, composerProjectIdAtSend\)/,
  );
});
