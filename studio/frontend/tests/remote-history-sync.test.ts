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

test("a failed re-read keeps the change pending for the next poll", async () => {
  const { sync, server } = harness({ revision: 1, threads: { a: 1 } });
  let readOk = false;
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: async () => {
      recovered += 1;
      return readOk;
    },
  });
  await sync.poll();
  server.value = { revision: 2, threads: { a: 2 } };
  await sync.poll();
  assert.equal(recovered, 1);
  readOk = true;
  await sync.poll();
  assert.equal(
    recovered,
    2,
    "the unchanged revision is retried after a failed read",
  );
  await sync.poll();
  assert.equal(recovered, 2);
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

function ownWrite(
  threadId: string,
  thread: [number, number],
  history: [number, number],
) {
  return { threadId, thread, history };
}

// A poll's first baseline can count a write the freshly loaded view lacks; own writes are skipped
// only once a re-read has synced the view. Drives one change from elsewhere and its re-read.
async function syncOnce(
  sync: ReturnType<typeof createRemoteHistorySync>,
  setServer: (value: RemoteHistoryRevision) => void,
  value: RemoteHistoryRevision,
) {
  setServer(value);
  await sync.poll();
  await new Promise((resolve) => setImmediate(resolve));
}

test("this tab's own write is not read back as another device's", async () => {
  const { sync, server, state } = harness({ revision: 9, threads: { a: 2 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
      return Promise.resolve(true);
    },
  });
  await sync.poll();
  await syncOnce(sync, (v) => (server.value = v), {
    revision: 10,
    threads: { a: 3 },
  });
  recovered = 0;
  state.historyChanges = 0;

  sync.noteOwnWrite(ownWrite("a", [3, 5], [10, 12]));
  sync.noteOwnWrite(ownWrite("a", [5, 6], [12, 13]));
  server.value = { revision: 13, threads: { a: 6 } };
  await sync.poll();
  await sync.poll();
  assert.equal(recovered, 0);
  assert.equal(state.historyChanges, 0);
});

test("a write from elsewhere before this tab's own still re-reads", async () => {
  const { sync, server, state } = harness({ revision: 10, threads: { a: 3 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();

  // Another device moved a from 3 to 4; this tab's write started from 4.
  sync.noteOwnWrite(ownWrite("a", [4, 6], [11, 13]));
  server.value = { revision: 13, threads: { a: 6 } };
  await sync.poll();
  assert.equal(recovered, 1);
  assert.equal(state.historyChanges, 1);
  await sync.poll();
  assert.equal(recovered, 1);
});

test("a write from elsewhere after this tab's own still re-reads", async () => {
  const { sync, server } = harness({ revision: 10, threads: { a: 3 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();
  sync.noteOwnWrite(ownWrite("a", [3, 4], [10, 11]));
  server.value = { revision: 12, threads: { a: 5 } };
  await sync.poll();
  assert.equal(recovered, 1);
});

test("a remote write during this tab's run is recovered once it is idle", async () => {
  const { sync, server } = harness({ revision: 10, threads: { a: 3 } });
  let idle = true;
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    isIdle: () => idle,
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();

  idle = false;
  sync.noteOwnWrite(ownWrite("a", [3, 4], [10, 11]));
  server.value = { revision: 12, threads: { a: 5 } }; // another device: 4 -> 5
  await sync.poll();
  sync.noteOwnWrite(ownWrite("a", [5, 6], [12, 13]));
  server.value = { revision: 13, threads: { a: 6 } };
  await sync.poll();
  assert.equal(recovered, 0, "a busy view keeps the change pending");
  idle = true;
  await sync.poll();
  assert.equal(recovered, 1);
});

test("own writes answered out of order or after the poll saw them stay quiet", async () => {
  const { sync, server } = harness({ revision: 9, threads: { a: 2 } });
  let idle = true;
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    isIdle: () => idle,
    onThreadChanged: () => {
      recovered += 1;
      return Promise.resolve(true);
    },
  });
  await sync.poll();
  await syncOnce(sync, (v) => (server.value = v), {
    revision: 10,
    threads: { a: 3 },
  });
  recovered = 0;

  sync.noteOwnWrite(ownWrite("a", [4, 5], [11, 12]));
  sync.noteOwnWrite(ownWrite("a", [3, 4], [10, 11]));
  server.value = { revision: 12, threads: { a: 5 } };
  await sync.poll();
  assert.equal(recovered, 0);

  // The poll read the counter before this tab's response arrived; the run held the change.
  idle = false;
  server.value = { revision: 13, threads: { a: 6 } };
  await sync.poll();
  sync.noteOwnWrite(ownWrite("a", [5, 6], [12, 13]));
  idle = true;
  await sync.poll();
  assert.equal(recovered, 0);
});

test("own writes to a chat that is not open do not hide its changes later", async () => {
  const { sync, server } = harness({ revision: 10, threads: { a: 3, b: 7 } });
  let open = "a";
  let recovered = 0;
  sync.watch({
    threadId: () => open,
    onThreadChanged: () => {
      recovered += 1;
    },
  });
  await sync.poll();
  sync.noteOwnWrite(ownWrite("b", [7, 8], [10, 11]));
  open = "b";
  server.value = { revision: 11, threads: { b: 8 } };
  await sync.poll(); // baseline for b
  server.value = { revision: 12, threads: { b: 9 } };
  await sync.poll();
  assert.equal(recovered, 1);
});

test("a failed re-read is retried even if this tab wrote meanwhile", async () => {
  const { sync, server } = harness({ revision: 10, threads: { a: 3 } });
  let finish: (ok: boolean) => void = () => {};
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    onThreadChanged: () => {
      recovered += 1;
      return new Promise<boolean>((resolve) => {
        finish = resolve;
      });
    },
  });
  await sync.poll();
  server.value = { revision: 11, threads: { a: 4 } }; // another device
  await sync.poll();
  assert.equal(recovered, 1);
  sync.noteOwnWrite(ownWrite("a", [4, 5], [11, 12]));
  server.value = { revision: 12, threads: { a: 5 } };
  finish(false);
  await new Promise<void>((resolve) => setImmediate(resolve));
  await sync.poll();
  assert.equal(recovered, 2, "the other device's write is read again");
});

test("a poll answered after this tab's own write does not roll the baseline back", async () => {
  let snapshot: RemoteHistoryRevision = { revision: 9, threads: { a: 2 } };
  let release: (() => void) | null = null;
  let hold = false;
  let historyChanges = 0;
  const sync = createRemoteHistorySync({
    fetchRevision: async () => {
      const taken = snapshot;
      if (hold) await new Promise<void>((resolve) => (release = resolve));
      return taken;
    },
    onHistoryChanged: () => {
      historyChanges += 1;
    },
    isVisible: () => true,
  });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    isIdle: () => true,
    onThreadChanged: () => {
      recovered += 1;
      return Promise.resolve(true);
    },
  });
  await sync.poll();
  await syncOnce(sync, (v) => (snapshot = v), {
    revision: 10,
    threads: { a: 3 },
  });
  recovered = 0;
  historyChanges = 0;

  // The poll reads the counters before the write lands and answers after its response.
  hold = true;
  const late = sync.poll();
  sync.noteOwnWrite(ownWrite("a", [3, 4], [10, 11]));
  snapshot = { revision: 11, threads: { a: 4 } };
  hold = false;
  (release as (() => void) | null)?.();
  await late;
  await sync.poll();
  assert.equal(recovered, 0);
  assert.equal(historyChanges, 0);

  // A later write from elsewhere is still read.
  snapshot = { revision: 12, threads: { a: 5 } };
  await sync.poll();
  assert.equal(recovered, 1);
  assert.equal(historyChanges, 1);
});

test("a write from elsewhere the first poll already counted is read after this tab's own write", async () => {
  // The view loaded at a: 3; another device wrote (a: 4) before this tab's first poll for the chat.
  const { sync, server } = harness({ revision: 11, threads: { a: 4 } });
  let recovered = 0;
  sync.watch({
    threadId: () => "a",
    isIdle: () => true,
    onThreadChanged: () => {
      recovered += 1;
      return Promise.resolve(true);
    },
  });
  await sync.poll();
  assert.equal(recovered, 0);

  sync.noteOwnWrite(ownWrite("a", [4, 6], [11, 13]));
  server.value = { revision: 13, threads: { a: 6 } };
  await sync.poll();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    recovered,
    1,
    "the first own write after opening still re-reads, as before",
  );

  // Synced now: later own writes are skipped.
  sync.noteOwnWrite(ownWrite("a", [6, 8], [13, 15]));
  server.value = { revision: 15, threads: { a: 8 } };
  await sync.poll();
  assert.equal(recovered, 1);
});
