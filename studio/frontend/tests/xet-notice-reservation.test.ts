// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two tabs starting a Xet download at the same moment used to read the same
// count and both toast. Its own file: the module's session counter is process
// wide, so these need a fresh instance of it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();

// A Web Locks stand-in that really serialises: each request queues behind the
// last one for that name, which is the property the reservation leans on.
const lockTails = new Map<string, Promise<unknown>>();
// globalThis.navigator is getter-only in node, so define over it.
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    locks: {
      request: (name: string, fn: () => boolean | Promise<boolean>) => {
        const tail = lockTails.get(name) ?? Promise.resolve();
        const run = tail.then(() => fn());
        lockTails.set(
          name,
          run.catch(() => undefined),
        );
        return run;
      },
    },
  },
});

registerBundlerResolver();

const { XET_NOTICE_LIMIT, XET_NOTICE_STORAGE_KEY, reserveXetNotice } =
  await import("../src/features/hub/download-manager/xet-progress-notice.ts");

test("a concurrent burst never hands out more than three", async () => {
  // Five reservations in flight at once, all reading the count before any
  // writes. Without the lock the tabs sharing a read would each toast; here
  // the fourth and fifth lose, which is the two-tabs-on-the-last-slot case.
  const taken = await Promise.all(
    Array.from({ length: 5 }, () => reserveXetNotice()),
  );
  assert.deepEqual(taken, [true, true, true, false, false]);
  assert.equal(store.get(XET_NOTICE_STORAGE_KEY), String(XET_NOTICE_LIMIT));
});
