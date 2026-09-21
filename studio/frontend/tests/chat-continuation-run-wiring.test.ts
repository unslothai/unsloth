// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `watchAutoContinueRun` against the real module-scope keeper, not one built with fakes.
 *
 * Its own file because it needs the store stub resolver: `auto-continue-run-keeper.ts` reads
 * `runningByThreadId` off the chat runtime store, which `chat-continuation.test.ts` does not load.
 * Both halves are worth having. That suite proves the keeper DECIDES correctly once it is handed a
 * run; this one proves the run actually reaches it, which a source-text match cannot. Emptying
 * this function's body while leaving its call site and the `issuedRunFrom` shape check in place
 * kept all of the other cases green, so the one wire between the bar and the keeper was the one
 * thing nothing executed.
 *
 * The observable is the renewal timer. `holdAutoContinueRun` starts it, and the keeper's own tick
 * clears it as soon as nothing is held, so "the timer stopped" is exactly "the hold was given up"
 * with no clock to advance and no elapsed time to compare. The timer is replaced outright rather
 * than mocked: the keeper captured `Date.now` as a default argument at import, so mocking `Date`
 * desynchronises its clock from the lease's and renewals silently stop landing.
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
  // Collected rather than assigned to a `let`: narrowing does not follow a write made inside a
  // callback, so the captured handler types as `never` and will not call.
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

  // The thread is idle, the ordinary case: the bar only fires on a reply that has already
  // finished. So the hold owns the key, and a preflight that is stopped is decidable.
  holdAutoContinueRun("m1", "thread-A");
  assert.equal(
    renewalTicks.length,
    1,
    "holding a run has to start renewing its lease, exactly once",
  );
  const renewalTick = renewalTicks[0];

  // The run the bar just issued, still in preflight. A real promise, because that is what
  // `startRun` hands back and what `issuedRunFrom` has to accept.
  let stopTheRun: (() => void) | undefined;
  const startedRun = new Promise<void>((resolve) => {
    stopTheRun = resolve;
  });
  watchAutoContinueRun("m1", "thread-A", startedRun);

  // Preflight runs long: this chat's settings pairing, then a large local GGUF loading. The lease
  // is renewed throughout, which is the whole reason arming has no deadline.
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

  // === The user presses Stop ===
  // The run's own promise settles. `runningByThreadId` is never touched, because no token was ever
  // on its way, and no failure is announced, because the abort is what the user asked for. Those
  // are the only two signals the keeper had before this change, so without the run itself nothing
  // here reports anything at all.
  stopTheRun?.();
  await startedRun;
  await settleWrites();

  // The tab is still open and still ticking. Its own run is over, so there is nothing left to
  // renew for and the keeper stops.
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
