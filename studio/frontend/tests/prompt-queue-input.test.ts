// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROMPT_QUEUE_DRAG_TYPE,
  isPromptQueueChord,
  isPromptQueueDragTypes,
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
  // A claimed file drag would be swallowed: the page dropzone skips events the
  // row already prevented, so the file would never attach.
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
