// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The newThreadId guard relies on assistant-ui clearing it on new -> regular, which some
// versions did not (assistant-ui #2292), so this drives the shipped reducer.

import assert from "node:assert/strict";
import test from "node:test";

// Deep relative path: @assistant-ui/core's exports map does not publish this module.
import { createThreadMappingId, updateStatusReducer } from "../node_modules/@assistant-ui/core/dist/runtimes/remote-thread-list/remote-thread-state.js";

const LOCAL_ID = "__LOCALID_contract0001";

function pendingNewThread() {
  return {
    isLoading: false,
    newThreadId: LOCAL_ID,
    threadIds: [] as string[],
    archivedThreadIds: [] as string[],
    threadIdMap: { [LOCAL_ID]: createThreadMappingId(LOCAL_ID) },
    threadData: {
      [LOCAL_ID]: {
        id: LOCAL_ID,
        remoteId: undefined,
        externalId: undefined,
        status: "new" as const,
        title: undefined,
        custom: undefined,
      },
    },
  };
}

test("a pending new thread is the one thread newThreadId names", () => {
  const state = pendingNewThread();
  assert.equal(state.newThreadId, LOCAL_ID);
  assert.ok(
    LOCAL_ID.startsWith("__LOCALID_"),
    "the id assistant-ui mints is what Studio stores as the row's primary key",
  );
});

test("newThreadId is cleared on new -> regular, which is what initialize() drives", () => {
  // The substitution in runtime-provider.tsx is only sound because of this line.
  const after = updateStatusReducer(pendingNewThread(), LOCAL_ID, "regular");

  assert.equal(
    after.newThreadId,
    undefined,
    "assistant-ui no longer clears newThreadId on new -> regular. The pairing guard in " +
      "ThreadScopedSettingsSync (activeThreadId === pendingNewThreadId) would then be " +
      "permanently true for every app-created chat, silently restoring the bug #9639 " +
      "fixed. See assistant-ui issue #2292 for the precedent.",
  );
});

test("the id keeps its __LOCALID_ prefix after the transition", () => {
  const after = updateStatusReducer(pendingNewThread(), LOCAL_ID, "regular");

  assert.ok(
    after.threadIds.includes(LOCAL_ID),
    "a saved thread should be listed under the very id it was minted with",
  );
});

test("archived and deleted also clear it, so no transition can strand the guard", () => {
  for (const status of ["archived", "deleted"] as const) {
    const after = updateStatusReducer(pendingNewThread(), LOCAL_ID, status);
    assert.equal(
      after.newThreadId,
      undefined,
      `newThreadId survived new -> ${status}, which would strand the pairing guard`,
    );
  }
});

test("a no-op transition leaves the pending new thread pending", () => {
  const after = updateStatusReducer(pendingNewThread(), LOCAL_ID, "new" as never);
  assert.equal(after.newThreadId, LOCAL_ID);
});
