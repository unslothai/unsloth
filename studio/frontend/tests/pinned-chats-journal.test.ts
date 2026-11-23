// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  PIN_JOURNAL_PREFIX,
  createPinnedChatsJournal,
} from "../src/features/chat/utils/pinned-chats-journal.ts";
import {
  preserveFallbackPins,
  readLegacyPinnedChatIds,
} from "../src/features/chat/utils/legacy-pinned-chats.ts";
import {
  type PinOperation,
  applyPinOperation,
  createPinnedChatsSync,
} from "../src/features/chat/utils/pinned-chats-sync.ts";

class MemoryStorage implements Storage {
  readonly values = new Map<string, string>();
  fail?: (method: "get" | "set" | "remove" | "key", key: string) => boolean;
  get length() {
    return this.values.size;
  }
  clear() {
    this.values.clear();
  }
  key(index: number) {
    if (this.fail?.("key", String(index))) {
      throw new Error("storage unavailable");
    }
    return [...this.values.keys()][index] ?? null;
  }
  getItem(key: string) {
    if (this.fail?.("get", key)) {
      throw new Error("storage unavailable");
    }
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    if (this.fail?.("set", key)) {
      throw new Error("storage unavailable");
    }
    this.values.set(key, value);
  }
  removeItem(key: string) {
    if (this.fail?.("remove", key)) {
      throw new Error("storage unavailable");
    }
    this.values.delete(key);
  }
}

function pin(operationId: string, id: string, pinned = true): PinOperation {
  return { kind: "pin", operationId, ids: [id], pinned };
}

function fallback(baseIds: string[], operations: PinOperation[] = []) {
  return {
    baseIds,
    operations,
    pinnedIds: operations.reduce(applyPinOperation, baseIds),
  };
}

const a = pin("01-a", "a");
const b = pin("02-b", "b");

test("tabs append independent durable entries without replacing an unseen intent", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "session");
  const second = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(first.readPending(), []);
  assert.deepEqual(second.readPending(), []);
  first.writePending([a]);
  second.writePending([b]);
  const reloaded = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(reloaded.readPending(), [a, b]);
  // The first request can acknowledge only what it observed before awaiting I/O.
  first.writePending([]);
  assert.deepEqual(reloaded.readPending(), [b]);
});

test("acknowledging an observed queue cannot erase a later append in another tab", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "session");
  first.writePending([a]);
  const second = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(second.readPending(), [a]);
  first.writePending([a, b]);
  second.writePending([]);
  assert.deepEqual(createPinnedChatsJournal(storage, "session").readPending(), [
    b,
  ]);
});

test("repeated writes keep an operation immutable and preserve FIFO order", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  journal.writePending([a, b]);
  const saved = [...storage.values.entries()];
  journal.writePending([a, b]);
  assert.deepEqual([...storage.values.entries()], saved);
  assert.deepEqual(createPinnedChatsJournal(storage, "session").readPending(), [
    a,
    b,
  ]);
});

test("partial append remains recoverable after reload and can be retried", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  storage.fail = (method, key) =>
    method === "set" && key.endsWith(b.operationId);
  assert.throws(() => journal.writePending([a, b]), /storage unavailable/);
  const reloaded = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(reloaded.readPending(), [a]);
  storage.fail = undefined;
  reloaded.writePending([a, b]);
  assert.deepEqual(createPinnedChatsJournal(storage, "session").readPending(), [
    a,
    b,
  ]);
});

test("partial acknowledgment retains every not-yet-removed entry for retry", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  journal.writePending([a, b]);
  storage.fail = (method, key) =>
    method === "remove" && key.endsWith(b.operationId);
  assert.throws(() => journal.writePending([]), /storage unavailable/);
  const reloaded = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(reloaded.readPending(), [b]);
  storage.fail = undefined;
  reloaded.writePending([]);
  assert.deepEqual(journal.readPending(), []);
});

test("concurrent initial fallback snapshots retain both tabs' known pins", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "session");
  const second = createPinnedChatsJournal(storage, "session");
  assert.equal(first.readFallback(), undefined);
  assert.equal(second.readFallback(), undefined);
  first.writeFallback(fallback(["from-first"]));
  second.writeFallback(fallback(["from-second"]));
  const reloaded = createPinnedChatsJournal(storage, "session");
  assert.deepEqual(
    new Set(reloaded.readFallback()?.pinnedIds),
    new Set(["from-first", "from-second"]),
  );
  assert.equal(storage.length, 2);
});

test("fallback combines separate baselines before replaying explicit unpin and pin", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "session");
  const second = createPinnedChatsJournal(storage, "session");
  const unpin = pin("01-remove", "removed", false);
  const added = pin("02-add", "added");
  first.writeFallback(fallback(["a", "removed", "c"], [unpin]));
  second.readFallback();
  second.writeFallback(
    fallback(["a", "removed", "c", "remote"], [unpin, added]),
  );
  const saved = createPinnedChatsJournal(storage, "session").readFallback();
  assert.deepEqual(saved?.baseIds, ["a", "removed", "c", "remote"]);
  assert.deepEqual(saved?.operations, [unpin, added]);
  assert.deepEqual(saved?.pinnedIds, ["added", "a", "c", "remote"]);
});

test("completed fallback replay acknowledges only entries observed before another tab wrote", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "session");
  const second = createPinnedChatsJournal(storage, "session");
  first.writeFallback(fallback(["base"], [a]));
  assert.deepEqual(first.readFallback()?.operations, [a]);
  second.writeFallback(fallback(["base"], [b]));
  first.writeFallback(undefined);
  const saved = createPinnedChatsJournal(storage, "session").readFallback();
  assert.deepEqual(saved?.operations, [b]);
  assert.deepEqual(saved?.pinnedIds, ["b", "base"]);
});

test("a fallback seed can be replaced with a durable explicit operation", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  journal.writeFallback(fallback(["base"]));
  journal.writeFallback(fallback(["base"], [a]));
  assert.equal(storage.length, 1);
  assert.deepEqual(
    createPinnedChatsJournal(storage, "session").readFallback(),
    fallback(["base"], [a]),
  );
});

test("fallback set failure leaves the previous durable snapshot intact", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  journal.writeFallback(fallback(["base"]));
  storage.fail = (method) => method === "set";
  assert.throws(
    () => journal.writeFallback(fallback(["base"], [a])),
    /storage unavailable/,
  );
  assert.deepEqual(
    createPinnedChatsJournal(storage, "session").readFallback(),
    fallback(["base"]),
  );
});

test("clear removes all entries for the auth scope but preserves other scopes and unrelated storage", () => {
  const storage = new MemoryStorage();
  const first = createPinnedChatsJournal(storage, "backend|auth:a");
  const same = createPinnedChatsJournal(storage, "backend|auth:a");
  const other = createPinnedChatsJournal(storage, "backend|auth:a:other");
  first.writePending([a]);
  same.writeFallback(fallback(["base"], [b]));
  other.writePending([b]);
  other.writeFallback(fallback(["other"]));
  storage.setItem("unrelated", "keep");
  first.clear();
  assert.deepEqual(same.readPending(), []);
  assert.equal(same.readFallback(), undefined);
  assert.deepEqual(other.readPending(), [b]);
  assert.deepEqual(other.readFallback()?.pinnedIds, ["other"]);
  assert.equal(storage.getItem("unrelated"), "keep");
});

test("journal storage read and key enumeration errors are reported without deleting entries", () => {
  for (const failure of ["get", "key"] as const) {
    const storage = new MemoryStorage();
    const journal = createPinnedChatsJournal(storage, "session");
    journal.writePending([a]);
    storage.fail = (method) => method === failure;
    assert.throws(() => journal.readPending(), /storage unavailable/);
    assert.equal(storage.length, 1);
  }
});

test("failed auth cleanup reports the error and can be retried", () => {
  const storage = new MemoryStorage();
  const journal = createPinnedChatsJournal(storage, "session");
  journal.writePending([a]);
  storage.fail = (method) => method === "remove";
  assert.throws(() => journal.clear(), /storage unavailable/);
  assert.equal(storage.length, 1);
  storage.fail = undefined;
  journal.clear();
  assert.equal(storage.length, 0);
});

test("corrupted pending and fallback entries are rejected rather than published", () => {
  for (const [kind, value] of [
    ["pending", { order: null, operation: a }],
    ["pending", { order: 1, operation: { ...a, ids: [null] } }],
    ["fallback", { order: 1, baseIds: ["duplicate", "duplicate"] }],
    ["fallback", { order: 1, operation: a }],
  ] as const) {
    const storage = new MemoryStorage();
    storage.setItem(
      `${PIN_JOURNAL_PREFIX}session:${kind}:corrupt`,
      JSON.stringify(value),
    );
    const journal = createPinnedChatsJournal(storage, "session");
    assert.throws(() =>
      kind === "pending" ? journal.readPending() : journal.readFallback(),
    );
    assert.equal(storage.length, 1);
  }
});

test("sign-out keeps an older backend's local saves for the next session's baseline", () => {
  const storage = new MemoryStorage();
  const before = createPinnedChatsJournal(storage, "server|session-1");
  before.writeFallback(fallback(["legacy"], [pin("01", "new"), pin("02", "legacy", false)]));
  preserveFallbackPins(storage, before.readFallback);
  before.clear();
  assert.equal(before.readFallback(), undefined);
  assert.deepEqual(readLegacyPinnedChatIds(storage), ["new"]);
  const after = createPinnedChatsJournal(storage, "server|session-2");
  assert.equal(after.readFallback(), undefined);
});

test("a tab holding a pin another tab's unpin cancelled does not replay it", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  const save = async (expected: string[] | undefined, next: string[]) => {
    if (JSON.stringify(expected ?? []) !== JSON.stringify(server)) return { applied: false, pinnedIds: server.slice() };
    server = next.slice();
    return { applied: true, pinnedIds: server.slice() };
  };
  function tab(online: () => boolean) {
    const disk = createPinnedChatsJournal(storage, "server|session");
    return createPinnedChatsSync({
      load: async () => server.slice(),
      compareAndSet: async (expected, next) => {
        if (!online()) throw new Error("offline");
        return save(expected, next);
      },
      readLegacy: () => [],
      readPending: disk.readPending,
      writePending: disk.writePending,
      publish: () => undefined,
    });
  }
  let aOnline = false;
  const a = tab(() => aOnline);
  await a.refresh();
  await assert.rejects(a.setPinned(["chat"], true));
  const b = tab(() => true);
  await b.refresh().catch(() => undefined);
  await b.setPinned(["chat"], false);
  assert.deepEqual(server, []);
  aOnline = true;
  await a.refresh();
  assert.deepEqual(server, []);
});

test("a tab finishing an earlier intent does not recreate one another tab cancelled meanwhile", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  let release!: () => void;
  let holding = false;
  const hold = new Promise<void>((resolve) => { release = resolve; });
  const save = async (expected: string[] | undefined, next: string[]) => {
    if (JSON.stringify(expected ?? []) !== JSON.stringify(server)) return { applied: false, pinnedIds: server.slice() };
    server = next.slice();
    return { applied: true, pinnedIds: server.slice() };
  };
  function tab(delayed: boolean) {
    const disk = createPinnedChatsJournal(storage, "server|session");
    return createPinnedChatsSync({
      load: async () => server.slice(),
      compareAndSet: async (expected, next) => {
        const result = await save(expected, next);
        if (delayed && holding) await hold;
        return result;
      },
      readLegacy: () => [],
      readPending: disk.readPending,
      writePending: disk.writePending,
      publish: () => undefined,
    });
  }
  const a = tab(true);
  await a.refresh();
  holding = true;
  const first = a.setPinned(["a"], true);
  const second = a.setPinned(["x"], true).catch(() => undefined);
  await new Promise((resolve) => setTimeout(resolve, 10));
  const b = tab(false);
  await b.setPinned(["x"], false).catch(() => undefined);
  await b.refresh().catch(() => undefined);
  release();
  await first;
  await second;
  await a.refresh().catch(() => undefined);
  assert.deepEqual(server, ["a"]);
});

test("a conflicting pin is not retried once another tab cancelled it", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let firstA = true;
  const pins: string[][] = [];
  function tab(name: "a" | "b") {
    const disk = createPinnedChatsJournal(storage, "server|session");
    return createPinnedChatsSync({
      load: async () => server.slice(),
      compareAndSet: async (expected, next) => {
        if (name === "a" && firstA) {
          firstA = false;
          await held;
        }
        if (next.includes("x")) pins.push(next.slice());
        if (JSON.stringify(expected ?? []) !== JSON.stringify(server)) return { applied: false, pinnedIds: server.slice() };
        server = next.slice();
        return { applied: true, pinnedIds: server.slice() };
      },
      readLegacy: () => [],
      readPending: disk.readPending,
      writePending: disk.writePending,
      publish: () => undefined,
    });
  }
  const a = tab("a");
  await a.refresh();
  const pinning = a.setPinned(["x"], true).catch(() => undefined);
  await new Promise((resolve) => setTimeout(resolve, 10));
  server = ["other"];
  const b = tab("b");
  await b.setPinned(["x"], false).catch(() => undefined);
  await b.refresh().catch(() => undefined);
  release();
  await pinning;
  assert.deepEqual(server, ["other"]);
  assert.equal(pins.length, 1);
});

test("unpinning one chat of a queued bulk pin keeps the rest and unblocks the queue", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  const disk = createPinnedChatsJournal(storage, "server|session");
  const sync = createPinnedChatsSync({
    load: async () => server.slice(),
    compareAndSet: async (expected, next) => {
      // A legacy comparison still awaiting import: the server cannot place it yet.
      if (next.includes("unimported") || JSON.stringify(expected ?? []) !== JSON.stringify(server)) {
        return { applied: false, pinnedIds: server.slice() };
      }
      server = next.slice();
      return { applied: true, pinnedIds: server.slice() };
    },
    readLegacy: () => [],
    readPending: disk.readPending,
    writePending: disk.writePending,
    publish: () => undefined,
  });
  await sync.refresh();
  const bulk = sync.setPinned(["unimported", "existing"], true).catch(() => undefined);
  await sync.setPinned(["unimported"], false);
  await sync.setPinned(["later"], true);
  await bulk;
  await sync.refresh();
  assert.deepEqual(server, ["later", "existing"]);
  assert.deepEqual(disk.readPending(), []);
});

test("an older intent another tab journaled replays before this tab's newer one", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  const writes: string[][] = [];
  let online = false;
  const disk = createPinnedChatsJournal(storage, "server|session");
  const a = createPinnedChatsSync({
    load: async () => server.slice(),
    compareAndSet: async (expected, next) => {
      if (!online) throw new Error("offline");
      writes.push(next.slice());
      if (JSON.stringify(expected ?? []) !== JSON.stringify(server)) return { applied: false, pinnedIds: server.slice() };
      server = next.slice();
      return { applied: true, pinnedIds: server.slice() };
    },
    readLegacy: () => [],
    readPending: disk.readPending,
    writePending: disk.writePending,
    publish: () => undefined,
  });
  await a.refresh().catch(() => undefined);
  await assert.rejects(a.setPinned(["chat"], false));
  // Tab B's pin, written before A's unpin but not yet seen by A.
  storage.setItem(
    `${PIN_JOURNAL_PREFIX}${encodeURIComponent("server|session")}:pending:b-pin`,
    JSON.stringify({ operation: pin("b-pin", "chat"), order: 1 }),
  );
  online = true;
  await a.refresh().catch(() => undefined);
  await a.refresh().catch(() => undefined);
  assert.deepEqual(server, []);
});

test("a twice-trimmed offline bulk pin keeps its place ahead of the unpins that fence it", async () => {
  const storage = new MemoryStorage();
  let server: string[] = [];
  const acknowledged = new Set<string>();
  let online = false;
  function client() {
    const disk = createPinnedChatsJournal(storage, "server|session");
    return createPinnedChatsSync({
      load: async () => server.slice(),
      compareAndSet: async (expected, next, operation) => {
        if (!online) throw new Error("offline");
        const id = operation && (operation.serverId ?? operation.operationId);
        if (id && acknowledged.has(id)) return { applied: true, pinnedIds: server.slice() };
        if (JSON.stringify(expected ?? []) !== JSON.stringify(server)) return { applied: false, pinnedIds: server.slice() };
        server = next.slice();
        if (id) acknowledged.add(id);
        operation?.cancels?.forEach((cancelled) => acknowledged.add(cancelled));
        return { applied: true, pinnedIds: server.slice() };
      },
      readLegacy: () => [],
      readPending: disk.readPending,
      writePending: disk.writePending,
      publish: () => undefined,
    });
  }
  const offline = client();
  await offline.refresh().catch(() => undefined);
  await assert.rejects(offline.setPinned(["a", "b", "c"], true));
  await offline.setPinned(["a"], false).catch(() => undefined);
  await offline.setPinned(["b"], false).catch(() => undefined);
  online = true;
  // After a reload, a fresh client replays the journal in its stored order.
  const reloaded = client();
  await reloaded.refresh().catch(() => undefined);
  await reloaded.refresh().catch(() => undefined);
  assert.deepEqual(server, ["c"]);
});
