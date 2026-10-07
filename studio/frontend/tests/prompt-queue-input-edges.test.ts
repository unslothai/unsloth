// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The drag type is matched literally: Chromium, Firefox and WebKit all report it unchanged. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROMPT_QUEUE_DRAG_TYPE,
  hasPendingPromptQueueStart,
  isPromptQueueChord,
  isPromptQueueDragTypes,
  pastedTextQueueKey,
} from "../src/features/chat/utils/prompt-queue-input.ts";
import { isGuardRetiringKey } from "../src/features/chat/utils/composer-send-guard.ts";

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

test("every platform's queue chord matches", () => {
  assert.equal(isPromptQueueChord(key({ metaKey: true })), true);
  assert.equal(isPromptQueueChord(key({ ctrlKey: true })), true);
  assert.equal(isPromptQueueChord(key({ metaKey: true, ctrlKey: true })), true);
});

test("the numeric keypad's Enter is the same chord", () => {
  // NumpadEnter has key "Enter" on every engine, and the predicate reads `key`.
  assert.equal(isPromptQueueChord(key({ ctrlKey: true })), true);
});

test("Shift disqualifies the chord whatever else is held", () => {
  // Shift+Enter must stay a newline.
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
  // AltGr is reported as Ctrl+Alt, so it must not queue.
  assert.equal(isPromptQueueChord(key({ ctrlKey: true, altKey: true })), false);
  assert.equal(isPromptQueueChord(key({ metaKey: true, altKey: true })), false);
});

test("an event without altKey at all still matches", () => {
  // Some synthetic events lack the flag; absent means not held.
  const bare = { key: "Enter", shiftKey: false, metaKey: false, ctrlKey: true };
  assert.equal(isPromptQueueChord(bare), true);
});

test("a queue row's drag is recognised however the engine lists the types", () => {
  assert.equal(isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE]), true);
  assert.equal(
    isPromptQueueDragTypes(["text/plain", PROMPT_QUEUE_DRAG_TYPE]),
    true,
  );
  // types is a DOMStringList in WebKit and an array in Chromium.
  const domStringList = { length: 1, 0: PROMPT_QUEUE_DRAG_TYPE };
  assert.equal(isPromptQueueDragTypes(domStringList), true);
});

test("nothing else counts as a queue drag", () => {
  assert.equal(isPromptQueueDragTypes(["Files"]), false);
  assert.equal(isPromptQueueDragTypes(["text/plain", "text/uri-list"]), false);
  assert.equal(isPromptQueueDragTypes([]), false);
  assert.equal(isPromptQueueDragTypes(null), false);
  assert.equal(isPromptQueueDragTypes(undefined), false);
  // The page dropzone skips prevented events, so a false positive swallows the file.
  assert.equal(isPromptQueueDragTypes(["Files", "application/x-moz-file"]), false);
});

test("a near-miss type is not the queue type", () => {
  assert.equal(isPromptQueueDragTypes([`${PROMPT_QUEUE_DRAG_TYPE}-2`]), false);
  assert.equal(
    isPromptQueueDragTypes([PROMPT_QUEUE_DRAG_TYPE.slice(0, -1)]),
    false,
  );
});

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
  // thread.tsx passes a one-shot `map.values()` iterator, so iterate only once.
  const map = new Map([["k", { cancelled: false, threadId: "t1" }]]);
  const iterator = map.values();
  assert.equal(hasPendingPromptQueueStart(iterator, "t1"), true);
  assert.equal(hasPendingPromptQueueStart(iterator, "t1"), false);
  assert.equal(hasPendingPromptQueueStart(map.values(), "t1"), true);
});

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
  // The key is JSON over user input, so quotes and brackets must not collide.
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
  // Different Unicode spellings stay distinct; written as escapes since they look identical.
  const composed = "caf\u00e9";
  const decomposed = "cafe\u0301";
  assert.notEqual(
    pastedTextQueueKey("t1", composed, []),
    pastedTextQueueKey("t1", decomposed, []),
  );
});

/** The chord and the send guard share one onKeyDown, so their AltGr rules must agree. */
test("no key is both a queue chord and a guard-retiring keystroke", () => {
  const keys = ["Enter", "a", "1", "é", "Escape", "Tab", "Shift", "Control",
    "Alt", "Meta", "CapsLock", "ArrowUp", "Backspace", "F5"];
  for (const k of keys) {
    for (const ctrlKey of [false, true]) {
      for (const metaKey of [false, true]) {
        for (const altKey of [false, true]) {
          for (const shiftKey of [false, true]) {
            const event = { key: k, ctrlKey, metaKey, altKey, shiftKey };
            const chord = isPromptQueueChord(event);
            const retires = isGuardRetiringKey(event);
            assert.equal(
              chord && retires,
              false,
              `${k} ctrl=${ctrlKey} meta=${metaKey} alt=${altKey} shift=${shiftKey}`,
            );
          }
        }
      }
    }
  }
});

test("AltGr reads the same way to the chord and to the guard", () => {
  const altGrChar = { key: "é", ctrlKey: true, altKey: true, metaKey: false,
    shiftKey: false };
  assert.equal(isPromptQueueChord(altGrChar), false);
  assert.equal(isGuardRetiringKey(altGrChar), true);
  const altGrEnter = { ...altGrChar, key: "Enter" };
  assert.equal(isPromptQueueChord(altGrEnter), false);
  assert.equal(isGuardRetiringKey(altGrEnter), false);
});
