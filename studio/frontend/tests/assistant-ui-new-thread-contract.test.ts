// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ThreadScopedSettingsSync now decides "this chat has no row yet" from
// `threads.newThreadId`. That is only a valid discriminator while assistant-ui clears
// the field on the new -> regular transition. Upstream has already shipped versions
// where it was never cleared (assistant-ui issue #2292, 0.10.35 / 0.10.36), and if that
// recurs on a dependency bump the guard becomes permanently true and Studio reverts to
// exactly the bug this PR fixes: every app-created chat unpaired from its own settings,
// with its edits leaking into the installation defaults.
//
// The PR's own tests are source-text regexes over runtime-provider.tsx, so they would
// stay green through that. This drives the real reducer out of node_modules instead.

import assert from "node:assert/strict";
import test from "node:test";

// Deep relative path, not the package specifier: @assistant-ui/core's exports map does
// not publish this module, and the point is to test the shipped artifact.
import { createThreadMappingId, updateStatusReducer } from "../node_modules/@assistant-ui/core/dist/runtimes/remote-thread-list/remote-thread-state.js";

const LOCAL_ID = "__LOCALID_contract0001";

/** The state shape a pending new thread is in, as switchToNewThread leaves it. */
function pendingNewThread() {
  return {
    isLoading: false,
    newThreadId: LOCAL_ID,
    threadIds: [] as string[],
    archivedThreadIds: [] as string[],
    threadIdMap: { [LOCAL_ID]: createThreadMappingId(LOCAL_ID) },
    // RemoteThreadData's "new" variant: the id is the minted __LOCALID_ one and remoteId
    // is still undefined -- initialize()'s response is what supplies it, and Studio's
    // adapter answers with the same string it was handed.
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
  // The other half of the premise: the prefix is NOT dropped when the thread is saved,
  // which is why the prefix cannot be read as "no row yet".
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
  // The converse: while the chat really has not been sent to, the guard must stay true.
  const after = updateStatusReducer(pendingNewThread(), LOCAL_ID, "new" as never);
  assert.equal(after.newThreadId, LOCAL_ID);
});
