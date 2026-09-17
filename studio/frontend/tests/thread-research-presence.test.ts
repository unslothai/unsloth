// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The composer's "has this thread used research already" answer, which gates whether deep
// research is still offerable. It is asked inside a useAuiState selector, so once per store
// write, and a keystroke IS a store write: before this cache the composer walked all 220 messages
// of the heavy-thread fixture per character typed.
//
// Semantics first, cache second: a memoization bug here silently turns deep research back on in a
// thread that already used it, or off in one that has not.

import assert from "node:assert/strict";
import test from "node:test";

import {
  messageHasResearchRunId,
  threadHasResearchMessage,
} from "../src/components/assistant-ui/thread-research-presence.ts";

const withRun = (id: unknown) => ({
  metadata: { custom: { researchRunId: id } },
});

test("a thread with a research reply anywhere in it answers true", () => {
  assert.equal(threadHasResearchMessage([{}, withRun("run_1"), {}]), true);
  assert.equal(threadHasResearchMessage([withRun("run_1")]), true);
});

test("a thread with no research reply answers false", () => {
  assert.equal(threadHasResearchMessage([]), false);
  assert.equal(threadHasResearchMessage([{}, { metadata: {} }]), false);
  assert.equal(threadHasResearchMessage([{ metadata: { custom: {} } }]), false);
  assert.equal(threadHasResearchMessage([{ metadata: undefined }]), false);
});

test("only a string run id counts, which is what the composer's gate meant", () => {
  // A non-string run id is how a half-written metadata blob shows up; counting it would switch
  // deep research off in a thread that never used it.
  for (const id of [undefined, null, 0, 1, true, {}, ["run_1"]]) {
    assert.equal(messageHasResearchRunId(withRun(id)), false, String(id));
    assert.equal(threadHasResearchMessage([withRun(id)]), false, String(id));
  }
  assert.equal(messageHasResearchRunId(withRun("")), true);
});

test("the answer is cached on the message array, not recomputed per call", () => {
  let reads = 0;
  const counting = [
    {
      get metadata() {
        reads += 1;
        return { custom: { researchRunId: "run_1" } };
      },
    },
  ];
  assert.equal(threadHasResearchMessage(counting), true);
  const afterFirst = reads;
  assert.ok(afterFirst > 0, "the first call has to read the messages");
  for (let i = 0; i < 50; i += 1) {
    assert.equal(threadHasResearchMessage(counting), true);
  }
  assert.equal(
    reads,
    afterFirst,
    "the scan runs again on an unchanged message array",
  );
});

test("a new message array is a new answer", () => {
  // assistant-ui rebuilds the array on every repository change, which is why it is the key: a
  // cache outliving the array would answer for the previous thread.
  const before = [{ metadata: {} }];
  assert.equal(threadHasResearchMessage(before), false);
  const after = [...before, withRun("run_1")];
  assert.equal(threadHasResearchMessage(after), true);
  // The old array still answers what it did, rather than being invalidated by the new one.
  assert.equal(threadHasResearchMessage(before), false);
});
