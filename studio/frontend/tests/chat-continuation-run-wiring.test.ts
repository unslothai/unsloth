// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `watchAutoContinueRun` against the real module-scope keeper. `chat-continuation.test.ts` proves
 * the keeper DECIDES correctly once handed a run; this proves the run reaches it, which a
 * source-text match cannot: emptying this function's body kept all of those cases green.
 *
 * Its own file because `auto-continue-run-keeper.ts` reads `runningByThreadId` off the chat runtime
 * store and so needs the store stub resolver. The observable is the renewal timer, which the keeper
 * clears as soon as nothing is held, so "the timer stopped" is "the hold was given up" with no clock
 * to advance. Replaced outright rather than mocked: the keeper captured `Date.now` as a default
 * argument at import, so mocking `Date` desynchronises its clock from the lease's and renewals
 * silently stop landing, passing this test for the wrong reason.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const { AUTO_CONTINUE_LEASE_KEY, claimAutoContinue } = await import(
  "../src/features/chat/utils/continuation.ts"
);

const { holdAutoContinueRun, watchAutoContinueRun } = await import(
  "../src/features/chat/utils/auto-continue-run-keeper.ts"
);

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

/** This tab's stored lease for `messageId`, or null once it is gone. */
function lease(messageId: string): { expires?: number; done?: boolean } | null {
  const raw = store.get(AUTO_CONTINUE_LEASE_KEY);
  if (!raw) {
    return null;
  }
  return (
    (JSON.parse(raw) as Record<string, { expires?: number; done?: boolean }>)[
      messageId
    ] ?? null
  );
}

/** Let the renewal's own promise land: the keeper drops it, so draining is the only way. */
async function settleWrites(): Promise<void> {
  for (let turn = 0; turn < 25; turn += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

test("the run the bar started reaches the keeper, so stopping it gives the lease back", async (t) => {
  useChatRuntimeStore.setState({ runningByThreadId: {} });

  // The renewal interval, captured rather than scheduled, so its ticks are this test's to drive.
  const realSetInterval = globalThis.setInterval;
  const realClearInterval = globalThis.clearInterval;
  const renewalId = Symbol("renewal-interval");
  // Collected, not a `let`: narrowing ignores a write inside a callback, so the handler would
  // type as `never` and not call.
  const renewalTicks: (() => void)[] = [];
  let renewalStopped = false;
  globalThis.setInterval = ((handler: () => void) => {
    renewalTicks.push(handler);
    return renewalId;
  }) as unknown as typeof globalThis.setInterval;
  globalThis.clearInterval = ((id: unknown) => {
    if (id === renewalId) {
      renewalStopped = true;
    }
  }) as unknown as typeof globalThis.clearInterval;
  t.after(() => {
    globalThis.setInterval = realSetInterval;
    globalThis.clearInterval = realClearInterval;
  });

  assert.equal(
    await claimAutoContinue("m1", "thread-A"),
    "started",
    "this tab has to own the message before it can hold its lease",
  );

  // Idle, the ordinary case, so the hold owns the key and a stopped preflight is decidable.
  holdAutoContinueRun("m1", "thread-A");
  assert.equal(
    renewalTicks.length,
    1,
    "holding a run has to start renewing its lease, exactly once",
  );
  const renewalTick = renewalTicks[0];

  // A real promise, because that is what `startRun` hands back and `issuedRunFrom` must accept.
  let stopTheRun: (() => void) | undefined;
  const startedRun = new Promise<void>((resolve) => {
    stopTheRun = resolve;
  });
  watchAutoContinueRun("m1", "thread-A", startedRun);

  // Renewed throughout a long preflight, which is why arming has no deadline.
  for (let renewal = 0; renewal < 3; renewal += 1) {
    renewalTick();
    await settleWrites();
  }
  assert.equal(
    renewalStopped,
    false,
    "the hold was given up while its run was still starting",
  );
  assert.ok(lease("m1"), "the lease is still held while its run is starting");

  // Stop. The stream flag never moved and no failure is announced, and those were the keeper's
  // only two signals before this change, so without the run itself nothing reports anything.
  stopTheRun?.();
  await startedRun;
  await settleWrites();

  // Still open and still ticking, but its own run is over, so the keeper stops.
  renewalTick();
  await settleWrites();

  assert.equal(
    renewalStopped,
    true,
    "the stopped preflight kept its hold, so it renews the lease for the life of the tab",
  );
  assert.equal(
    lease("m1")?.done,
    undefined,
    "a message that streamed not one token was marked continued",
  );
});
