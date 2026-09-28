// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type RefreshSupersession,
  registerRefresh,
  supersedingRefresh,
} from "../src/features/hub/lib/superseded-refresh.ts";

interface Status {
  checkpoint: string;
  ggufVariant: string;
}

/** What the store holds before an API request switches the resident model. */
const STALE: Status = {
  checkpoint: "unsloth/Qwen3-8B-GGUF",
  ggufVariant: "Q8_0",
};
/** What every read of /api/inference/status answers once the switch has landed. */
const SWITCHED: Status = {
  checkpoint: "unsloth/Llama-3.1-8B-Instruct-GGUF",
  ggufVariant: "Q4_K_M",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

/** Let every pending microtask run, so "did not resolve" means it really has not. */
const flush = () => new Promise((r) => setTimeout(r, 0));

/**
 * hub-page.tsx's refreshResidentModelStatus, with the status read held open so a test can
 * choose the order responses land in. `coalesce` controls whether a dropped response
 * waits for the refresh that superseded it.
 */
function hubPageRefresh(coalesce: boolean) {
  let seq = 0;
  const supersession: RefreshSupersession = { latest: null };
  const store: Status = { ...STALE };
  const inFlight: Array<(status: Status) => void> = [];

  const refresh = (): Promise<void> => {
    const mine = ++seq;
    const read = deferred<Status>();
    inFlight.push(read.resolve);
    const settled = read.promise
      .then((status) => {
        // hub-page.tsx:417, the drop: a newer read owns the store, so this writes nothing.
        if (mine !== seq) {
          return coalesce ? supersedingRefresh(supersession, mine) : undefined;
        }
        store.checkpoint = status.checkpoint;
        store.ggufVariant = status.ggufVariant;
      })
      .catch(() => undefined);
    if (coalesce) registerRefresh(supersession, mine, settled);
    return settled;
  };

  return {
    refresh,
    store,
    /** Deliver the status response for the nth refresh started. */
    deliver: (n: number, status: Status = SWITCHED) => inFlight[n](status),
    /** The unmount cleanup: invalidate everything in flight without starting a read. */
    unmount: () => {
      seq += 1;
    },
  };
}

/** Resolves to true only if `promise` settles before the microtask queue drains. */
async function settledEarly(promise: Promise<void>): Promise<boolean> {
  let done = false;
  void promise.then(() => {
    done = true;
  });
  await flush();
  return done;
}

test("the initial safety read waits out a newer focus refresh", async () => {
  const hub = hubPageRefresh(true);
  const initialRead = hub.refresh();
  hub.refresh();

  hub.deliver(0);
  assert.equal(
    await settledEarly(initialRead),
    false,
    "a dropped response must not release the safety gate",
  );
  assert.deepEqual(
    hub.store,
    STALE,
    "precondition: the dropped response wrote nothing, so the store is still pre-switch",
  );

  hub.deliver(1);
  await initialRead;
  assert.deepEqual(hub.store, SWITCHED);
});

test("without coalescing the dropped response releases stale state", async () => {
  const hub = hubPageRefresh(false);
  const initialRead = hub.refresh();
  hub.refresh();

  hub.deliver(0);
  assert.equal(await settledEarly(initialRead), true);
  assert.deepEqual(hub.store, STALE);
});

test("every dropped response in a chain waits for the one read that wins", async () => {
  // Focus and visibilitychange can fire as a pair, so several reads may overlap.
  const hub = hubPageRefresh(true);
  const initialRead = hub.refresh();
  hub.refresh();
  hub.refresh();

  hub.deliver(0);
  hub.deliver(1);
  assert.equal(await settledEarly(initialRead), false);
  assert.deepEqual(hub.store, STALE);

  hub.deliver(2);
  await initialRead;
  assert.deepEqual(hub.store, SWITCHED);
});

test("responses that land out of order still leave the store on the newest read", async () => {
  const hub = hubPageRefresh(true);
  const initialRead = hub.refresh();
  hub.refresh();

  // The newest read answers first and writes the store; the older response is dropped and
  // finds its superseder already settled.
  hub.deliver(1);
  hub.deliver(0);
  await initialRead;
  assert.deepEqual(hub.store, SWITCHED);
});

test("an unmount strands nobody, since it bumps the sequence without starting a read", async () => {
  // hub-page.tsx's cleanup only invalidates. The newest refresh is then its own superseder,
  // and handing it its own promise would leave every caller waiting forever.
  const hub = hubPageRefresh(true);
  const read = hub.refresh();
  hub.unmount();
  hub.deliver(0);

  const timeout = new Promise<"hung">((r) => setTimeout(() => r("hung"), 50));
  assert.notEqual(
    await Promise.race([read.then(() => "settled" as const), timeout]),
    "hung",
  );
  assert.deepEqual(hub.store, STALE, "an unmounted Hub adopts nothing");
});

test("a superseder is only ever a strictly newer refresh", () => {
  const supersession: RefreshSupersession = { latest: null };
  assert.equal(supersedingRefresh(supersession, 1), undefined);

  const settled = Promise.resolve();
  registerRefresh(supersession, 2, settled);
  assert.equal(supersedingRefresh(supersession, 1), settled);
  assert.equal(
    supersedingRefresh(supersession, 2),
    undefined,
    "a refresh may not be handed its own promise",
  );
  assert.equal(supersedingRefresh(supersession, 3), undefined);
});
