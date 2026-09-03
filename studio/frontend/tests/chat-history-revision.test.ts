// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The two channels: a structural change goes at once, the per-chunk streaming path waits for
// its quiet window. Sharing one timer leaves a delete unannounced for a whole generation.

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();

const {
  CHAT_HISTORY_REVISION_KEY,
  CROSS_TAB_REVISION_DEBOUNCE_MS,
  flushChatHistoryRevision,
  isCoalescedHistoryEvent,
  publishChatHistoryRevision,
} = await import("../src/features/chat/utils/chat-history-revision.ts");

const revision = () => store.get(CHAT_HISTORY_REVISION_KEY) ?? null;

test("a structural change publishes without waiting", () => {
  store.clear();
  publishChatHistoryRevision(false);
  assert.notEqual(
    revision(),
    null,
    "another tab must not hold a deleted chat as a live row",
  );
});

test("the streaming path waits for its quiet window", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    store.clear();
    publishChatHistoryRevision(true);
    mock.timers.tick(CROSS_TAB_REVISION_DEBOUNCE_MS - 1);
    assert.equal(
      revision(),
      null,
      "no write while the chunks are still arriving",
    );

    mock.timers.tick(1);
    assert.notEqual(revision(), null, "the quiet window publishes once");
  } finally {
    mock.timers.reset();
  }
});

test("a stream collapses into one write rather than one per window", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    store.clear();
    // Two seconds of chunks at 50ms, which under a throttle would have published four times.
    for (let i = 0; i < 40; i += 1) {
      publishChatHistoryRevision(true);
      mock.timers.tick(50);
    }
    assert.equal(
      revision(),
      null,
      "a continuous stream publishes nothing mid-flight",
    );

    mock.timers.tick(CROSS_TAB_REVISION_DEBOUNCE_MS);
    assert.notEqual(revision(), null);
  } finally {
    mock.timers.reset();
  }
});

test("a structural change during a stream does not wait for it", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    store.clear();
    publishChatHistoryRevision(true);
    mock.timers.tick(100);
    // Under a shared timer the chunks that follow would push this back to the end.
    publishChatHistoryRevision(false);
    const published = revision();
    assert.notEqual(published, null, "the delete publishes on its own");

    for (let i = 0; i < 10; i += 1) {
      publishChatHistoryRevision(true);
      mock.timers.tick(50);
    }
    assert.equal(
      revision(),
      published,
      "and the stream that follows does not publish again mid-flight",
    );
  } finally {
    mock.timers.reset();
  }
});

test("a pending write is published before the tab goes away", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    store.clear();
    publishChatHistoryRevision(true);
    assert.equal(revision(), null);

    // What the pagehide handler calls: otherwise the write leaves with the page.
    flushChatHistoryRevision();
    assert.notEqual(revision(), null);

    const published = revision();
    mock.timers.tick(CROSS_TAB_REVISION_DEBOUNCE_MS);
    assert.equal(
      revision(),
      published,
      "and the flushed timer does not fire again",
    );
  } finally {
    mock.timers.reset();
  }
});

test("flushing with nothing pending writes nothing", () => {
  store.clear();
  flushChatHistoryRevision();
  assert.equal(revision(), null);
});

// misreading a chunk save as structural starves a retiring listener for a whole generation
test("a coalesced streaming update is distinguishable from a structural change", () => {
  assert.equal(
    isCoalescedHistoryEvent(
      new CustomEvent("x", { detail: { coalesce: true } }),
    ),
    true,
  );
  assert.equal(
    isCoalescedHistoryEvent(
      new CustomEvent("x", { detail: { coalesce: false } }),
    ),
    false,
  );
});

test("an undetailed history event counts as structural", () => {
  // what the cross-tab listener re-raises, and what any caller that skips the detail sends
  assert.equal(isCoalescedHistoryEvent(new Event("x")), false);
  assert.equal(isCoalescedHistoryEvent(new CustomEvent("x")), false);
});
