// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROMPT_QUEUE_DRAG_TYPE,
  hasPendingPromptQueueStart,
  isPromptQueueChord,
  isPromptQueueDragTypes,
  pastedTextQueueKey,
} from "../src/features/chat/utils/prompt-queue-input.ts";

function key(
  overrides: Partial<{
    key: string;
    shiftKey: boolean;
    metaKey: boolean;
    ctrlKey: boolean;
  }> = {},
) {
  return {
    key: "Enter",
    shiftKey: false,
    metaKey: false,
    ctrlKey: false,
    ...overrides,
  };
}

test("Cmd or Ctrl plus Enter is the queue chord", () => {
  assert.ok(isPromptQueueChord(key({ metaKey: true })));
  assert.ok(isPromptQueueChord(key({ ctrlKey: true })));
});

test("Shift+Enter stays a newline, with or without a modifier", () => {
  assert.equal(isPromptQueueChord(key({ shiftKey: true })), false);
  assert.equal(
    isPromptQueueChord(key({ shiftKey: true, metaKey: true })),
    false,
  );
  assert.equal(
    isPromptQueueChord(key({ shiftKey: true, ctrlKey: true })),
    false,
  );
});

test("plain Enter and other keys are left alone", () => {
  assert.equal(isPromptQueueChord(key()), false);
  assert.equal(isPromptQueueChord(key({ key: "k", metaKey: true })), false);
  assert.equal(isPromptQueueChord(key({ key: "NumpadEnter" })), false);
});

test("a drag carrying the queue type is a reorder", () => {
  assert.ok(isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE]));
  assert.ok(isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE, "text/plain"]));
});

test("file and foreign drags are not claimed", () => {
  // The page dropzone skips prevented events, so a claimed file drag never attaches.
  assert.equal(isPromptQueueDragTypes(["Files"]), false);
  assert.equal(isPromptQueueDragTypes(["text/plain"]), false);
  assert.equal(isPromptQueueDragTypes(["text/uri-list", "text/html"]), false);
  assert.equal(isPromptQueueDragTypes([]), false);
  assert.equal(isPromptQueueDragTypes(undefined), false);
  assert.equal(isPromptQueueDragTypes(null), false);
});

test("the drag type is private to the queue", () => {
  // A generic type would collide with drags from other panes.
  assert.match(PROMPT_QUEUE_DRAG_TYPE, /^application\/x-unsloth-/);
});

const pending = (threadId: string | null, cancelled = false) => ({
  cancelled,
  threadId,
});

// Queue start awaits hydration; without this a plain Enter double-sends the prompt.
test("a pending start counts as queueing for its own thread", () => {
  assert.equal(
    hasPendingPromptQueueStart([pending("thread-1")], "thread-1"),
    true,
  );
});

test("another thread's pending start does not block this one", () => {
  assert.equal(
    hasPendingPromptQueueStart([pending("thread-2")], "thread-1"),
    false,
  );
});

test("a cancelled reservation stops counting", () => {
  assert.equal(
    hasPendingPromptQueueStart([pending("thread-1", true)], "thread-1"),
    false,
  );
});

test("nothing pending is not queueing", () => {
  assert.equal(hasPendingPromptQueueStart([], "thread-1"), false);
});

// A new chat has no id until it persists, so null matches null.
test("a new chat's pending start is matched on null", () => {
  assert.equal(hasPendingPromptQueueStart([pending(null)], null), true);
  assert.equal(hasPendingPromptQueueStart([pending(null)], "thread-1"), false);
});

test("one live reservation among cancelled ones still counts", () => {
  assert.equal(
    hasPendingPromptQueueStart(
      [pending("thread-1", true), pending("thread-1")],
      "thread-1",
    ),
    true,
  );
});

test("concurrent pasted-text reads are tracked per thread", () => {
  const reads = [pending("thread-1"), pending("thread-2")];
  assert.equal(hasPendingPromptQueueStart(reads, "thread-1"), true);
  assert.equal(hasPendingPromptQueueStart(reads, "thread-2"), true);
  assert.equal(hasPendingPromptQueueStart(reads, "thread-3"), false);
  reads.splice(0, 1);
  assert.equal(hasPendingPromptQueueStart(reads, "thread-1"), false);
  assert.equal(hasPendingPromptQueueStart(reads, "thread-2"), true);
});

// A submit during the read is queued, so the read must be recognisable to avoid a duplicate.
test("the same pasted prompt reads under one key", () => {
  const key = () => pastedTextQueueKey("t1", "notes", ["a1", "a2"]);
  assert.equal(key(), key());
});

test("a different thread, text or attachment is a different read", () => {
  const base = pastedTextQueueKey("t1", "notes", ["a1"]);
  assert.notEqual(base, pastedTextQueueKey("t2", "notes", ["a1"]));
  assert.notEqual(base, pastedTextQueueKey("t1", "other", ["a1"]));
  assert.notEqual(base, pastedTextQueueKey("t1", "notes", ["a2"]));
});

test("reordered attachments are a different read", () => {
  assert.notEqual(
    pastedTextQueueKey("t1", "notes", ["a1", "a2"]),
    pastedTextQueueKey("t1", "notes", ["a2", "a1"]),
  );
});

test("a null thread does not collide with a named one", () => {
  assert.notEqual(
    pastedTextQueueKey(null, "notes", []),
    pastedTextQueueKey("t1", "notes", []),
  );
});
