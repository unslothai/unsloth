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
