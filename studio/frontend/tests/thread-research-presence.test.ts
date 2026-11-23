// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Asked once per store write (every keystroke), so it is cached; correctness gates deep research.

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
  // A non-string run id is a half-written metadata blob and must not count.
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
  // assistant-ui rebuilds the array per repository change, so it is the cache key.
  const before = [{ metadata: {} }];
  assert.equal(threadHasResearchMessage(before), false);
  const after = [...before, withRun("run_1")];
  assert.equal(threadHasResearchMessage(after), true);
  assert.equal(threadHasResearchMessage(before), false);
});
