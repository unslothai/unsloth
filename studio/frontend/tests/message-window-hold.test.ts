// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One shared hold registry serves both body windowing and progressive row
// windowing. Holds are per viewport, counted per message, and release is
// guarded at the closure so a duplicate cleanup cannot steal another claim.

import assert from "node:assert/strict";
import test from "node:test";
import {
  heldMessageIds,
  holdMessage,
  isMessageHeld,
  subscribeMessageHolds,
} from "../src/components/assistant-ui/message-window-hold.ts";

const viewport = (name: string) => ({ name }) as unknown as HTMLElement;

test("holds are scoped to one viewport and one message", () => {
  const first = viewport("first");
  const second = viewport("second");
  const release = holdMessage(first, "assistant-1");

  assert.equal(isMessageHeld(first, "assistant-1"), true);
  assert.equal(isMessageHeld(first, "assistant-2"), false);
  assert.equal(isMessageHeld(second, "assistant-1"), false);
  assert.deepEqual([...heldMessageIds(first)], ["assistant-1"]);
  assert.deepEqual([...heldMessageIds(second)], []);

  release();
  assert.equal(isMessageHeld(first, "assistant-1"), false);
  assert.deepEqual([...heldMessageIds(first)], []);
});

test("counted holds require every holder to release", () => {
  const root = viewport("counted");
  const first = holdMessage(root, "assistant-1");
  const second = holdMessage(root, "assistant-1");
  first();
  assert.equal(isMessageHeld(root, "assistant-1"), true);
  second();
  assert.equal(isMessageHeld(root, "assistant-1"), false);
});

test("release is idempotent and cannot decrement another holder", () => {
  const root = viewport("idempotent");
  const first = holdMessage(root, "assistant-1");
  const second = holdMessage(root, "assistant-1");
  first();
  first();
  assert.equal(isMessageHeld(root, "assistant-1"), true);
  assert.deepEqual([...heldMessageIds(root)], ["assistant-1"]);
  second();
  assert.equal(isMessageHeld(root, "assistant-1"), false);
});

test("acquisition and release wake subscribers for the matching viewport", () => {
  const root = viewport("watched");
  const other = viewport("other");
  const wakeups: Array<[HTMLElement, string]> = [];
  const stop = subscribeMessageHolds((candidate, messageId) => {
    wakeups.push([candidate, messageId]);
  });

  const release = holdMessage(root, "assistant-1");
  release();
  holdMessage(other, "assistant-2");

  assert.deepEqual(wakeups, [
    [root, "assistant-1"],
    [root, "assistant-1"],
    [other, "assistant-2"],
  ]);
  stop();
});

test("an empty id is a no-op hold", () => {
  const root = viewport("empty");
  const release = holdMessage(root, "");
  assert.equal(isMessageHeld(root, ""), false);
  release();
});
