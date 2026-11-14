// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Proves a run reaches the keeper, which a source match cannot. Date.now is replaced, not
 * mocked: the keeper captured it as a default argument at import.
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

async function settleWrites(): Promise<void> {
  for (let turn = 0; turn < 25; turn += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

test("the run the bar started reaches the keeper, so stopping it gives the lease back", async (t) => {
  useChatRuntimeStore.setState({ runningByThreadId: {} });

  const realSetInterval = globalThis.setInterval;
  const realClearInterval = globalThis.clearInterval;
  const renewalId = Symbol("renewal-interval");
  // Collected in an array: TS narrowing ignores writes inside callbacks, so a `let` types as never.
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

  holdAutoContinueRun("m1", "thread-A");
  assert.equal(
    renewalTicks.length,
    1,
    "holding a run has to start renewing its lease, exactly once",
  );
  const renewalTick = renewalTicks[0];

  let stopTheRun: (() => void) | undefined;
  const startedRun = new Promise<void>((resolve) => {
    stopTheRun = resolve;
  });
  watchAutoContinueRun("m1", "thread-A", startedRun);

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

  stopTheRun?.();
  await startedRun;
  await settleWrites();

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
