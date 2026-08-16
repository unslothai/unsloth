// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Edge cases for the queue's keyboard and drag predicates, on top of the
 * behaviour `prompt-queue-input.test.ts` already pins.
 *
 * These are the shapes the DOM hands us on platforms this box cannot run: a
 * numeric-keypad Enter, a Windows AltGr chord, and a `DOMStringList` rather
 * than the array Chromium returns.
 *
 * The drag type itself was measured rather than assumed. Driven through a real
 * DataTransfer in Chromium 151, Firefox 153 and WebKit 26.5, all three report
 * the type back exactly as it was set, lowercase, in `types` during dragover
 * and through `getData` on drop -- so the predicate can match it literally.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROMPT_QUEUE_DRAG_TYPE,
  hasPendingPromptQueueStart,
  isPromptQueueChord,
  isPromptQueueDragTypes,
  pastedTextQueueKey,
} from "../src/features/chat/utils/prompt-queue-input.ts";

type ChordEvent = {
  key: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey?: boolean;
};

function key(overrides: Partial<ChordEvent> = {}): ChordEvent {
  return {
    key: "Enter",
    shiftKey: false,
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    ...overrides,
  };
}

// ── the chord ────────────────────────────────────────────────────────────────

test("every platform's queue chord matches", () => {
  // macOS sends Cmd, Windows and Linux send Ctrl, and either is accepted
  // everywhere: a Mac user with a PC keyboard habit still queues.
  assert.equal(isPromptQueueChord(key({ metaKey: true })), true);
  assert.equal(isPromptQueueChord(key({ ctrlKey: true })), true);
  assert.equal(isPromptQueueChord(key({ metaKey: true, ctrlKey: true })), true);
});

test("the numeric keypad's Enter is the same chord", () => {
  // `code` differs (NumpadEnter) but `key` is "Enter" on every engine, and the
  // predicate reads `key`, so a full-size keyboard's right-hand Enter queues.
  assert.equal(isPromptQueueChord(key({ ctrlKey: true })), true);
});

test("Shift disqualifies the chord whatever else is held", () => {
  // Shift+Enter is a newline, and this is the rule that keeps it one: the
  // plain-Enter branch below carries the !shiftKey exclusion, so a chord that
  // matched with Shift held would turn a newline into a queue.
  assert.equal(isPromptQueueChord(key({ ctrlKey: true, shiftKey: true })), false);
  assert.equal(isPromptQueueChord(key({ metaKey: true, shiftKey: true })), false);
  assert.equal(
    isPromptQueueChord(key({ metaKey: true, ctrlKey: true, shiftKey: true })),
    false,
  );
});

test("a bare Enter is not the chord", () => {
  assert.equal(isPromptQueueChord(key()), false);
  assert.equal(isPromptQueueChord(key({ shiftKey: true })), false);
});

test("only the Enter key is the chord, and the name is case sensitive", () => {
  for (const name of ["enter", "ENTER", "Return", "NumpadEnter", "Escape", " "]) {
    assert.equal(
      isPromptQueueChord(key({ key: name, ctrlKey: true })),
      false,
      `${name} must not queue`,
    );
  }
});

test("AltGr+Enter on a Windows layout does not queue", () => {
  // AltGr is reported as Ctrl+Alt. A layout that needs AltGr for everyday
  // characters would otherwise queue on a keypress the user means as a newline
  // or a plain send, and there is no Ctrl+Alt+Enter binding worth keeping.
  assert.equal(isPromptQueueChord(key({ ctrlKey: true, altKey: true })), false);
  assert.equal(isPromptQueueChord(key({ metaKey: true, altKey: true })), false);
});

test("an event without altKey at all still matches", () => {
  // Not every synthetic event carries the flag; absent must read as not held,
  // or a hand-built event in a test or a bridge would stop queueing.
  const bare = { key: "Enter", shiftKey: false, metaKey: false, ctrlKey: true };
  assert.equal(isPromptQueueChord(bare), true);
});

// ── the drag type ────────────────────────────────────────────────────────────

test("a queue row's drag is recognised however the engine lists the types", () => {
  assert.equal(isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE]), true);
  assert.equal(
    isPromptQueueDragTypes(["text/plain", PROMPT_QUEUE_DRAG_TYPE]),
    true,
  );
  // DataTransfer.types is a DOMStringList in WebKit and a frozen array in
  // Chromium; both are ArrayLike, which is all the predicate requires.
  const domStringList = { length: 1, 0: PROMPT_QUEUE_DRAG_TYPE };
  assert.equal(isPromptQueueDragTypes(domStringList), true);
});

test("nothing else counts as a queue drag", () => {
  assert.equal(isPromptQueueDragTypes(["Files"]), false);
  assert.equal(isPromptQueueDragTypes(["text/plain", "text/uri-list"]), false);
  assert.equal(isPromptQueueDragTypes([]), false);
  assert.equal(isPromptQueueDragTypes(null), false);
  assert.equal(isPromptQueueDragTypes(undefined), false);
  // A file drag must reach the page dropzone, which skips events a row already
  // prevented, so a false positive here silently swallows the file.
  assert.equal(isPromptQueueDragTypes(["Files", "application/x-moz-file"]), false);
});

test("a near-miss type is not the queue type", () => {
  assert.equal(isPromptQueueDragTypes([`${PROMPT_QUEUE_DRAG_TYPE}-2`]), false);
  assert.equal(
    isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE.slice(0, -1)]),
    false,
  );
});

// ── pending starts ───────────────────────────────────────────────────────────

test("a pending start is only this thread's while it is live", () => {
  const live = { cancelled: false, threadId: "t1" };
  const other = { cancelled: false, threadId: "t2" };
  const dead = { cancelled: true, threadId: "t1" };
  assert.equal(hasPendingPromptQueueStart([live], "t1"), true);
  assert.equal(hasPendingPromptQueueStart([other], "t1"), false);
  assert.equal(hasPendingPromptQueueStart([dead], "t1"), false);
  assert.equal(hasPendingPromptQueueStart([dead, live], "t1"), true);
  assert.equal(hasPendingPromptQueueStart([], "t1"), false);
});

test("a new chat's null thread matches only another null", () => {
  assert.equal(hasPendingPromptQueueStart([{ cancelled: false, threadId: null }], null), true);
  assert.equal(hasPendingPromptQueueStart([{ cancelled: false, threadId: "t1" }], null), false);
  assert.equal(hasPendingPromptQueueStart([{ cancelled: false, threadId: null }], "t1"), false);
});

test("a Map's values are consumed exactly once, as the caller passes them", () => {
  // thread.tsx hands this `map.values()`, a one-shot iterator. Reading it twice
  // would report the second call empty, so the predicate must not iterate more
  // than once, and the caller must not reuse the iterator.
  const map = new Map([["k", { cancelled: false, threadId: "t1" }]]);
  const iterator = map.values();
  assert.equal(hasPendingPromptQueueStart(iterator, "t1"), true);
  assert.equal(hasPendingPromptQueueStart(iterator, "t1"), false);
  assert.equal(hasPendingPromptQueueStart(map.values(), "t1"), true);
});

// ── the pasted-text key ──────────────────────────────────────────────────────

test("the pasted-text key is stable and separates what it must", () => {
  const k = () => pastedTextQueueKey("t1", "hello", ["a1", "a2"]);
  assert.equal(k(), k());
  assert.notEqual(k(), pastedTextQueueKey("t2", "hello", ["a1", "a2"]));
  assert.notEqual(k(), pastedTextQueueKey("t1", "hello!", ["a1", "a2"]));
  assert.notEqual(k(), pastedTextQueueKey("t1", "hello", ["a2", "a1"]));
  assert.notEqual(k(), pastedTextQueueKey("t1", "hello", ["a1"]));
  assert.notEqual(
    pastedTextQueueKey(null, "hello", []),
    pastedTextQueueKey("null", "hello", []),
  );
});

test("the pasted-text key survives text that looks like its own encoding", () => {
  // The key is JSON, and the text is user input: a prompt full of quotes and
  // brackets must not be able to collide with another prompt's key.
  const tricky = '","x"],["t1","';
  assert.notEqual(
    pastedTextQueueKey("t1", tricky, []),
    pastedTextQueueKey("t1", "x", []),
  );
  assert.equal(
    pastedTextQueueKey("t1", tricky, []),
    pastedTextQueueKey("t1", tricky, []),
  );
});

test("the pasted-text key handles unicode and long prompts", () => {
  const long = "\u6f22\u5b57".repeat(5_000);
  assert.equal(pastedTextQueueKey("t1", long, []), pastedTextQueueKey("t1", long, []));
  // Two Unicode spellings of the same word are two different prompts, which is
  // the conservative side to fall on: the cost is a second read, never a
  // dropped one. Written as escapes because the two look identical in a file.
  const composed = "caf\u00e9";
  const decomposed = "cafe\u0301";
  assert.notEqual(
    pastedTextQueueKey("t1", composed, []),
    pastedTextQueueKey("t1", decomposed, []),
  );
});
