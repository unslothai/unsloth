// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type RemoteHistoryRevision,
  createRemoteHistorySync,
} from "../src/features/chat/utils/remote-history-sync.ts";

function harness(initial: RemoteHistoryRevision | null) {
  const server = { value: initial, fail: false };
  const asked: string[][] = [];
  const state = { visible: true, historyChanges: 0 };
  const sync = createRemoteHistorySync({
    fetchRevision: async (ids) => {
      asked.push([...ids]);
      if (server.fail) throw new TypeError("Failed to fetch");
      return server.value;
    },
    onHistoryChanged: () => {
      state.historyChanges += 1;
    },
    isVisible: () => state.visible,
  });
  return { sync, server, asked, state };
}

test("a write from another device re-reads the open thread once", async () => {
  const { sync, server, asked } = harness({ revision: 1, threads: { a: 1 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
    },
  });

  await sync.poll();
  assert.deepEqual(asked.at(-1), ["a"]);
  assert.equal(
    recovered,
    0,
    "the first read is the baseline: the thread was just loaded",
  );

  server.value = { revision: 2, threads: { a: 2 } };
  await sync.poll();
  await sync.poll();
  assert.equal(recovered, 1);
});

test("history changes elsewhere refresh the sidebar but leave the open thread alone", async () => {
  const { sync, server, state } = harness({ revision: 5, threads: { a: 1 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();

  server.value = { revision: 6, threads: { a: 1 } };
  await sync.poll();
  assert.equal(state.historyChanges, 1);
  assert.equal(recovered, 0);
});

test("a busy view keeps the change pending until it is idle", async () => {
  const { sync, server } = harness({ revision: 1, threads: { a: 1 } });
  let idle = false;
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    isIdle: () => idle,
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();
  server.value = { revision: 2, threads: { a: 4 } };
  await sync.poll();
  assert.equal(recovered, 0);
  idle = true;
  await sync.poll();
  assert.equal(recovered, 1);
});

test("switching threads takes a new baseline instead of re-reading", async () => {
  const { sync, server } = harness({ revision: 1, threads: { a: 1, b: 9 } });
  let open = "a";
  let recovered = 0;
  sync.watch({
    threadId: () => open,
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();
  open = "b";
  server.value = { revision: 1, threads: { b: 9 } };
  await sync.poll();
  assert.equal(recovered, 0);
});

test("hidden pages, failures and old servers stay quiet", async () => {
  const { sync, server, asked, state } = harness({ revision: 1, threads: {} });
  sync.watch({});
  state.visible = false;
  await sync.poll();
  assert.equal(asked.length, 0);

  state.visible = true;
  server.fail = true;
  await sync.poll();
  assert.equal(state.historyChanges, 0);

  server.fail = false;
  server.value = null;
  await sync.poll();
  await sync.poll();
  assert.equal(asked.length, 2, "a server without the endpoint is asked once");
  assert.equal(sync.watching(), false);
});
