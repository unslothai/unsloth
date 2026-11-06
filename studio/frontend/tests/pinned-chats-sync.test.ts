// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type PinOperation,
  PinnedChatsValidationError,
  createPinnedChatsSync,
  validatePinOperation,
  validatePinnedIds,
} from "../src/features/chat/utils/pinned-chats-sync.ts";
import {
  ChatSettingsRequestError,
  isRetryablePinError,
} from "../src/features/chat/utils/settings-retry.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function server(initial?: string[]) {
  let value = initial?.slice();
  const calls: { expected: string[] | undefined; next: string[] }[] = [];
  return {
    calls,
    get value() {
      return value?.slice();
    },
    load: async () => value?.slice(),
    compareAndSet: async (expected: string[] | undefined, next: string[]) => {
      calls.push({ expected: expected?.slice(), next: next.slice() });
      const applied = JSON.stringify(value) === JSON.stringify(expected);
      if (applied) {
        value = next.slice();
      }
      return { applied, pinnedIds: value?.slice() };
    },
  };
}

function client(
  backend: ReturnType<typeof server>,
  legacy: string[] = [],
  isRetryable?: (error: unknown) => boolean,
) {
  let visible: string[] = [];
  const errors: unknown[] = [];
  const history: string[][] = [];
  const sync = createPinnedChatsSync({
    load: backend.load,
    compareAndSet: backend.compareAndSet,
    readLegacy: () => legacy,
    isRetryable,
    publish: (ids) => {
      visible = ids;
      history.push(ids.slice());
    },
    onError: (error) => errors.push(error),
  });
  return {
    sync,
    errors,
    history,
    get visible() {
      return visible;
    },
  };
}

test("a fresh browser reads without claiming an empty migration value", async () => {
  const backend = server();
  const fresh = client(backend);
  await fresh.sync.refresh();
  assert.deepEqual(fresh.visible, []);
  assert.equal(backend.value, undefined);
  assert.equal(backend.calls.length, 0);

  const old = client(backend, ["old-chat", "old-chat"]);
  await old.sync.refresh();
  await fresh.sync.refresh();
  assert.deepEqual(backend.value, ["old-chat"]);
  assert.deepEqual(fresh.visible, ["old-chat"]);
  assert.equal(backend.calls.length, 1);
});

test("an explicitly empty server list prevents stale local pins returning", async () => {
  const backend = server([]);
  const old = client(backend, ["old-chat"]);
  await old.sync.refresh();
  assert.deepEqual(old.visible, []);
  assert.equal(backend.calls.length, 0);
});

test("two legacy clients cannot overwrite the migration winner", async () => {
  const backend = server();
  const a = client(backend, ["A"]);
  const b = client(backend, ["B"]);
  await Promise.all([a.sync.refresh(), b.sync.refresh()]);
  assert.deepEqual(backend.value, ["A"]);
  assert.deepEqual(a.visible, ["A"]);
  assert.deepEqual(b.visible, ["A"]);
});

test("a fresh client and reload read pins saved by another browser", async () => {
  const backend = server();
  const a = client(backend);
  await a.sync.setPinned(["chat"], true);
  const b = client(backend);
  const reloadedA = client(backend);
  await Promise.all([b.sync.refresh(), reloadedA.sync.refresh()]);
  assert.deepEqual(b.visible, ["chat"]);
  assert.deepEqual(reloadedA.visible, ["chat"]);
});

test("concurrent additions rebase on the server without losing either pin", async () => {
  const backend = server(["old"]);
  const a = client(backend);
  const b = client(backend);
  await Promise.all([a.sync.refresh(), b.sync.refresh()]);
  await Promise.all([
    a.sync.setPinned(["A"], true),
    b.sync.setPinned(["B"], true),
  ]);
  assert.deepEqual(backend.value, ["B", "A", "old"]);
  assert.equal(backend.calls.length, 3);
  await a.sync.refresh();
  assert.deepEqual(a.visible, b.visible);
});

test("bulk operations deduplicate IDs and preserve existing pin positions", async () => {
  const backend = server(["old", "existing"]);
  const a = client(backend);
  await a.sync.refresh();
  await a.sync.setPinned(["new", "existing", "new", "another"], true);
  assert.deepEqual(backend.value, ["new", "another", "old", "existing"]);
  await a.sync.setPinned(["old", "new"], false);
  assert.deepEqual(backend.value, ["another", "existing"]);
});

test("pinning before initial hydration preserves legacy pins", async () => {
  const backend = server();
  const a = client(backend, ["old"]);
  const saved = a.sync.setPinned(["new"], true);
  assert.deepEqual(a.visible, ["new"]);
  await saved;
  assert.deepEqual(backend.value, ["new", "old"]);
});

test("late hydration cannot hide a newer optimistic pin", async () => {
  const backend = server(["old"]);
  const read = deferred<string[] | undefined>();
  backend.load = () => read.promise;
  const a = client(backend);
  const refreshing = a.sync.refresh();
  await Promise.resolve();
  const saved = a.sync.setPinned(["new"], true);
  const from = a.history.length - 1;
  read.resolve(["old"]);
  await Promise.all([refreshing, saved]);
  assert.deepEqual(a.visible, ["new", "old"]);
  assert.ok(a.history.slice(from).every((ids) => ids.includes("new")));
});

test("a terminal rejected write rolls back without discarding a newer pending action", async () => {
  const backend = server(["old"]);
  const cas = backend.compareAndSet;
  let first = true;
  backend.compareAndSet = async (expected, next) => {
    if (first) {
      first = false;
      throw new Error("unavailable");
    }
    return cas(expected, next);
  };
  const a = client(backend, [], () => false);
  await a.sync.refresh();
  const results = await Promise.allSettled([
    a.sync.setPinned(["failed"], true),
    a.sync.setPinned(["saved"], true),
  ]);
  assert.equal(results[0].status, "rejected");
  assert.equal(results[1].status, "fulfilled");
  assert.deepEqual(a.visible, ["saved", "old"]);
  assert.deepEqual(backend.value, ["saved", "old"]);
  assert.equal(a.errors.length, 1);
});

test("an ambiguous failed response reads back a committed write", async () => {
  const backend = server(["old"]);
  const cas = backend.compareAndSet;
  backend.compareAndSet = async (expected, next) => {
    await cas(expected, next);
    throw new Error("connection closed after commit");
  };
  const a = client(backend);
  await a.sync.refresh();
  await assert.rejects(a.sync.setPinned(["new"], true));
  assert.deepEqual(a.visible, ["new", "old"]);
  assert.deepEqual(backend.value, a.visible);
});

test("failed recovery keeps unsaved intent visible and retries it before a later mutation", async () => {
  const backend = server(["old"]);
  const load = backend.load;
  const cas = backend.compareAndSet;
  let offline = false;
  let visible: string[] = [];
  const b = createPinnedChatsSync({
    load: () => (offline ? Promise.reject(new Error("offline")) : load()),
    compareAndSet: (expected, next) =>
      offline ? Promise.reject(new Error("offline")) : cas(expected, next),
    readLegacy: () => [],
    publish: (ids) => {
      visible = ids;
    },
  });
  await b.refresh();
  offline = true;
  await assert.rejects(b.setPinned(["failed"], true));
  assert.deepEqual(visible, ["failed", "old"]);
  offline = false;
  await b.setPinned(["new"], true);
  assert.deepEqual(visible, ["new", "failed", "old"]);
});

test("reset ignores an old hydration and prevents its queued write", async () => {
  const backend = server(["new-session"]);
  const oldRead = deferred<string[] | undefined>();
  let calls = 0;
  backend.load = () =>
    ++calls === 1 ? oldRead.promise : Promise.resolve(backend.value);
  const a = client(backend);
  const old = a.sync.setPinned(["old-session-edit"], true);
  await Promise.resolve();
  a.sync.reset();
  await a.sync.refresh();
  oldRead.resolve(["old-session"]);
  await old;
  assert.deepEqual(a.visible, ["new-session"]);
  assert.equal(backend.calls.length, 0);
});

test("reset ignores an in-flight write response and drops the next queued write", async () => {
  const backend = server(["server"]);
  const response = deferred<{
    pinnedIds: string[] | undefined;
    applied: boolean;
  }>();
  let writes = 0;
  backend.compareAndSet = () => {
    writes += 1;
    return response.promise;
  };
  const a = client(backend);
  await a.sync.refresh();
  const first = a.sync.setPinned(["old-first"], true);
  const second = a.sync.setPinned(["old-second"], true);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(writes, 1);
  a.sync.reset();
  await a.sync.refresh();
  response.resolve({ applied: true, pinnedIds: ["old-first", "server"] });
  await Promise.all([first, second]);
  assert.deepEqual(a.visible, ["server"]);
  assert.equal(writes, 1);
});

test("refresh requests coalesce while the server read is pending", async () => {
  const backend = server();
  const read = deferred<string[] | undefined>();
  let reads = 0;
  backend.load = () => {
    reads += 1;
    return read.promise;
  };
  const a = client(backend);
  const first = a.sync.refresh();
  const second = a.sync.refresh();
  assert.equal(first, second);
  read.resolve([]);
  await Promise.all([first, second]);
  assert.equal(reads, 1);
});

test("repeated compare-and-set conflicts stop after a bounded number of attempts", async () => {
  const backend = server([]);
  let calls = 0;
  backend.compareAndSet = async () => {
    calls += 1;
    return { applied: false, pinnedIds: ["remote"] };
  };
  const a = client(backend);
  await a.sync.refresh();
  await assert.rejects(
    a.sync.setPinned(["local"], true),
    /changed in another browser/,
  );
  assert.equal(calls, 8);
  assert.deepEqual(a.visible, ["local"]);
  assert.equal(a.errors.length, 1);
});

test("an absent unpin reaches the server to record migration tombstones", async () => {
  const backend = server();
  const operations: PinOperation[] = [];
  const fresh = createPinnedChatsSync({
    load: backend.load,
    compareAndSet: async (expected, next, operation) => {
      if (operation) {
        operations.push(operation);
      }
      return backend.compareAndSet(expected, next);
    },
    readLegacy: () => [],
    publish: () => undefined,
  });
  await fresh.setPinned(["unrelated-chat"], false);
  assert.deepEqual(backend.value, []);
  assert.equal(backend.calls.length, 1);
  assert.equal(operations[0]?.kind, "pin");
  assert.deepEqual(
    operations[0] && "ids" in operations[0] ? operations[0].ids : [],
    ["unrelated-chat"],
  );
  assert.equal(
    operations[0] && "pinned" in operations[0]
      ? operations[0].pinned
      : undefined,
    false,
  );
});

test("unpinning the final migrated pin persists an authoritative empty list", async () => {
  const backend = server();
  const a = client(backend, ["old-chat"]);
  await a.sync.setPinned(["old-chat"], false);
  assert.deepEqual(backend.value, []);
  const stale = client(backend, ["old-chat"]);
  await stale.sync.refresh();
  assert.deepEqual(stale.visible, []);
  assert.equal(backend.calls.length, 2);
});

test("a recovered absent setting still migrates legacy pins before the next write", async () => {
  const backend = server();
  const load = backend.load;
  let first = true;
  backend.load = async () => {
    if (first) {
      first = false;
      throw new Error("temporary read failure");
    }
    return load();
  };
  const a = client(backend, ["legacy"]);
  const results = await Promise.allSettled([
    a.sync.setPinned(["failed"], true),
    a.sync.setPinned(["saved"], true),
  ]);
  assert.equal(results[0].status, "rejected");
  assert.equal(results[1].status, "rejected");
  await a.sync.refresh();
  assert.deepEqual(backend.value, ["saved", "failed", "legacy"]);
});

function journal() {
  let entries: PinOperation[] = [];
  return {
    read: () => structuredClone(entries),
    write: (next: PinOperation[]) => {
      entries = structuredClone(next);
    },
  };
}

function durableClient(
  backend: ReturnType<typeof server>,
  disk: ReturnType<typeof journal>,
  extra: Partial<Parameters<typeof createPinnedChatsSync>[0]> = {},
) {
  let visible: string[] = [];
  const sync = createPinnedChatsSync({
    load: backend.load,
    compareAndSet: backend.compareAndSet,
    readLegacy: () => [],
    readPending: disk.read,
    writePending: disk.write,
    publish: (ids) => {
      visible = ids;
    },
    ...extra,
  });
  return {
    sync,
    get visible() {
      return visible;
    },
  };
}

test("a pin queued behind hydration is durable before optimistic publish and survives reload", async () => {
  const backend = server(["old"]);
  const disk = journal();
  const stalled = deferred<string[] | undefined>();
  const a = durableClient(backend, disk, { load: () => stalled.promise });
  const hydration = a.sync.refresh();
  const saving = a.sync.setPinned(["new"], true);
  assert.deepEqual(a.visible, ["new"]);
  assert.equal(disk.read().length, 1);
  assert.equal(disk.read()[0].kind, "pin");
  await Promise.resolve();
  a.sync.reset();
  assert.equal(disk.read().length, 1);
  const reloaded = durableClient(backend, disk);
  await reloaded.sync.refresh();
  assert.deepEqual(backend.value, ["new", "old"]);
  assert.deepEqual(reloaded.visible, ["new", "old"]);
  assert.deepEqual(disk.read(), []);
  stalled.resolve(["old"]);
  await Promise.all([hydration, saving]);
  assert.deepEqual(reloaded.visible, ["new", "old"]);
});

test("journal write failure rejects before publishing the optimistic action", async () => {
  const backend = server(["old"]);
  const disk = journal();
  const a = durableClient(backend, disk, {
    writePending: () => {
      throw new Error("storage full");
    },
  });
  await a.sync.refresh();
  await assert.rejects(a.sync.setPinned(["new"], true), /storage full/);
  assert.deepEqual(a.visible, ["old"]);
  assert.deepEqual(backend.value, ["old"]);
  assert.equal(backend.calls.length, 0);
  assert.deepEqual(disk.read(), []);
});

test("retryable failure retains durable intent and later actions wait in FIFO order", async () => {
  const backend = server(["old"]);
  const disk = journal();
  let offline = true;
  const a = durableClient(backend, disk, {
    compareAndSet: (expected, next) =>
      offline
        ? Promise.reject(new Error("offline"))
        : backend.compareAndSet(expected, next),
  });
  await a.sync.refresh();
  await assert.rejects(a.sync.setPinned(["new"], true));
  assert.deepEqual(a.visible, ["new", "old"]);
  assert.equal(disk.read().length, 1);
  await assert.rejects(a.sync.setPinned(["old"], false));
  assert.equal(disk.read().length, 2);
  assert.deepEqual(a.visible, ["new"]);
  assert.deepEqual(backend.value, ["old"]);
  offline = false;
  await a.sync.refresh();
  assert.deepEqual(
    backend.calls.map((call) => call.next),
    [["new", "old"], ["new"]],
  );
  assert.deepEqual(backend.value, ["new"]);
  assert.deepEqual(disk.read(), []);
});

test("terminal rejection removes only the failed journal entry", async () => {
  const backend = server(["old"]);
  const disk = journal();
  let first = true;
  const a = durableClient(backend, disk, {
    isRetryable: () => false,
    compareAndSet: async (expected, next) => {
      if (first) {
        first = false;
        throw new Error("invalid operation");
      }
      return backend.compareAndSet(expected, next);
    },
  });
  const results = await Promise.allSettled([
    a.sync.setPinned(["rejected"], true),
    a.sync.setPinned(["accepted"], true),
  ]);
  assert.equal(results[0].status, "rejected");
  assert.equal(results[1].status, "fulfilled");
  assert.deepEqual(a.visible, ["accepted", "old"]);
  assert.deepEqual(disk.read(), []);
});

test("acknowledged operations are not replayed when the journal is read again", async () => {
  const backend = server([]);
  const disk = journal();
  disk.write([
    {
      kind: "pin",
      ids: ["restored"],
      pinned: true,
      operationId: "restored-operation",
    },
  ]);
  const a = durableClient(backend, disk);
  await a.sync.refresh();
  await a.sync.refresh();
  assert.deepEqual(backend.value, ["restored"]);
  assert.equal(backend.calls.length, 1);
  assert.deepEqual(disk.read(), []);
});

test("server operation IDs prevent acknowledged journal replay from restoring a later unpin", async () => {
  const backend = server([]);
  const disk = journal();
  const seen = new Set<string>();
  let rejectCleanup = true;
  const compareAndSet = async (
    expected: string[] | undefined,
    next: string[],
    operation?: PinOperation,
  ) => {
    if (operation && seen.has(operation.operationId)) {
      return { applied: true, pinnedIds: backend.value };
    }
    const result = await backend.compareAndSet(expected, next);
    if (result.applied && operation) {
      seen.add(operation.operationId);
    }
    return result;
  };
  const a = durableClient(backend, disk, {
    compareAndSet,
    writePending: (entries) => {
      if (entries.length === 0 && rejectCleanup) {
        throw new Error("cannot acknowledge locally");
      }
      disk.write(entries);
    },
  });
  await assert.rejects(a.sync.setPinned(["new"], true));
  assert.deepEqual(backend.value, ["new"]);
  assert.equal(disk.read().length, 1);
  await backend.compareAndSet(["new"], []);
  a.sync.reset();
  rejectCleanup = false;
  const reloaded = durableClient(backend, disk, { compareAndSet });
  await reloaded.sync.refresh();
  assert.deepEqual(backend.value, []);
  assert.deepEqual(reloaded.visible, []);
  assert.deepEqual(disk.read(), []);
});

test("cleanup failure retries journal removal without resending a known acknowledged operation", async () => {
  const backend = server([]);
  const disk = journal();
  let failing = true;
  const a = durableClient(backend, disk, {
    writePending: (entries) => {
      if (entries.length === 0 && failing) {
        throw new Error("storage unavailable");
      }
      disk.write(entries);
    },
  });
  await assert.rejects(a.sync.setPinned(["new"], true));
  assert.equal(backend.calls.length, 1);
  failing = false;
  await a.sync.refresh();
  assert.equal(backend.calls.length, 1);
  assert.deepEqual(disk.read(), []);
});

test("explicit session reset clears the old journal while normal reset retains it", async () => {
  const backend = server([]);
  const disk = journal();
  const a = durableClient(backend, disk, {
    compareAndSet: async () => {
      throw new Error("offline");
    },
  });
  await assert.rejects(a.sync.setPinned(["private-intent"], true));
  a.sync.reset();
  assert.equal(disk.read().length, 1);
  a.sync.reset({ clearPending: true });
  assert.deepEqual(disk.read(), []);
  assert.deepEqual(a.visible, []);
});

test("legacy merge runs even when server pins exist and preserves durable unpin tombstones", async () => {
  const backend = server(["server"]);
  const tombstones = new Set(["removed"]);
  let merges = 0;
  function mergingClient(initialLegacy: string[]) {
    let legacy = initialLegacy;
    return createPinnedChatsSync({
      load: backend.load,
      compareAndSet: backend.compareAndSet,
      readLegacy: () => legacy,
      migrateLegacy: async (ids) => {
        merges++;
        const current = backend.value ?? [];
        await backend.compareAndSet(current, [
          ...current,
          ...ids.filter((id) => !(current.includes(id) || tombstones.has(id))),
        ]);
        legacy = [];
        return backend.value;
      },
      publish: () => {},
    });
  }
  const a = mergingClient(["legacy-A", "removed"]);
  const b = mergingClient(["legacy-B", "removed"]);
  await a.refresh();
  await b.refresh();
  await a.refresh();
  assert.deepEqual(backend.value, ["server", "legacy-A", "legacy-B"]);
  assert.equal(merges, 2);
});

test("malformed loaded snapshots never replace the last valid published state", async () => {
  const malformed: unknown[] = [
    "string",
    null,
    {},
    [1],
    [""],
    ["duplicate", "duplicate"],
    ["x".repeat(513)],
    Array.from({ length: 10_001 }, (_, i) => String(i)),
  ];
  for (const bad of malformed) {
    const backend = server(["valid"]);
    let loadBad = false;
    const a = durableClient(backend, journal(), {
      load: async () => (loadBad ? (bad as string[]) : backend.value),
    });
    await a.sync.refresh();
    loadBad = true;
    await assert.rejects(a.sync.refresh(), PinnedChatsValidationError);
    assert.deepEqual(a.visible, ["valid"]);
  }
});

test("malformed CAS snapshots and applied flags are rejected without corrupting the UI", async () => {
  const responses: unknown[] = [
    { applied: true, pinnedIds: "string" },
    { applied: true, pinnedIds: undefined },
    { applied: true, pinnedIds: ["duplicate", "duplicate"] },
    { applied: "false", pinnedIds: [] },
    null,
  ];
  for (const response of responses) {
    const backend = server(["valid"]);
    const disk = journal();
    const a = durableClient(backend, disk, {
      compareAndSet: async () =>
        response as { applied: boolean; pinnedIds: string[] },
    });
    await a.sync.refresh();
    await assert.rejects(
      a.sync.setPinned(["new"], true),
      PinnedChatsValidationError,
    );
    assert.deepEqual(a.visible, ["valid"]);
    assert.deepEqual(disk.read(), []);
  }
});

test("malformed migration and pending journal data are rejected before writes", async () => {
  const backend = server(["valid"]);
  const badMigration = durableClient(backend, journal(), {
    readLegacy: () => ["old"],
    migrateLegacy: async () => "string" as unknown as string[],
  });
  await assert.rejects(badMigration.sync.refresh(), PinnedChatsValidationError);
  assert.deepEqual(badMigration.visible, []);
  const badJournal = durableClient(backend, journal(), {
    readPending: () =>
      [
        {
          kind: "move",
          operationId: "op",
          id: "a",
          targetId: "b",
          placement: "sideways",
        },
      ] as unknown as PinOperation[],
  });
  await assert.rejects(badJournal.sync.refresh(), PinnedChatsValidationError);
  assert.equal(backend.calls.length, 0);
});

test("orphaned legacy IDs may leave server state absent without blocking new pins", async () => {
  const backend = server();
  let migrations = 0;
  const a = durableClient(backend, journal(), {
    readLegacy: () => ["deleted-legacy-thread"],
    migrateLegacy: async () => {
      migrations += 1;
      return backend.value;
    },
  });
  await a.sync.refresh();
  assert.deepEqual(a.visible, []);
  assert.equal(backend.value, undefined);
  await a.sync.setPinned(["existing-thread"], true);
  assert.deepEqual(backend.value, ["existing-thread"]);
  assert.equal(migrations, 1);
});

test("IDs reject lone UTF-16 surrogates while preserving valid Unicode", () => {
  for (const value of [
    "\ud800",
    "\udfff",
    "a\ud800b",
    "\ud800\ud800",
    "\udfff\udc00",
  ]) {
    assert.throws(() => validatePinnedIds([value]), PinnedChatsValidationError);
    assert.throws(
      () =>
        validatePinOperation({
          kind: "pin",
          operationId: value,
          ids: [],
          pinned: true,
        }),
      PinnedChatsValidationError,
    );
  }
  assert.deepEqual(validatePinnedIds(["\ud83d\udcac", "chat-\u00e9"]), [
    "\ud83d\udcac",
    "chat-\u00e9",
  ]);
});

test("pins save on a plain-HTTP origin, where crypto.randomUUID does not exist", async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis.crypto, "randomUUID");
  Object.defineProperty(globalThis.crypto, "randomUUID", { value: undefined, configurable: true });
  try {
    const backend = server(["old"]);
    const a = client(backend);
    await a.sync.refresh();
    await a.sync.setPinned(["new"], true);
    assert.deepEqual(backend.value, ["new", "old"]);
  } finally {
    if (original) Object.defineProperty(globalThis.crypto, "randomUUID", original);
    else delete (globalThis.crypto as { randomUUID?: unknown }).randomUUID;
  }
});

test("unpinning a chat the server cannot place yet cancels its queued pin instead of waiting behind it", async () => {
  let value: string[] = [];
  const operations: PinOperation[] = [];
  let visible: string[] = [];
  const sync = createPinnedChatsSync({
    load: async () => value.slice(),
    compareAndSet: async (expected, next, operation) => {
      if (operation) operations.push(operation);
      // A comparison deleted before its legacy import: the server never acknowledges its pin.
      if (next.includes("pair") || JSON.stringify(expected) !== JSON.stringify(value)) {
        return { applied: false, pinnedIds: value.slice() };
      }
      value = next.slice();
      return { applied: true, pinnedIds: value.slice() };
    },
    readLegacy: () => [],
    publish: (ids) => { visible = ids; },
  });
  await sync.refresh();
  const blocked = sync.setPinned(["pair"], true).catch(() => undefined);
  await sync.setPinned(["pair"], false);
  await sync.setPinned(["chat"], true);
  await blocked;
  assert.deepEqual(value, ["chat"]);
  assert.deepEqual(visible, ["chat"]);
  assert.ok(operations.some((op) => !op.pinned && op.ids.includes("pair")));
});

test("a 401 from a transient token-refresh failure keeps the queued pin for a later retry", async () => {
  const backend = server(["old"]);
  const disk = journal();
  let unauthorized = true;
  const a = durableClient(backend, disk, {
    isRetryable: isRetryablePinError,
    compareAndSet: (expected, next) =>
      unauthorized
        ? Promise.reject(new ChatSettingsRequestError("Unauthorized", 401, null))
        : backend.compareAndSet(expected, next),
  });
  await a.sync.refresh();
  await assert.rejects(a.sync.setPinned(["new"], true));
  assert.equal(disk.read().length, 1);
  unauthorized = false;
  await a.sync.refresh();
  assert.deepEqual(backend.value, ["new", "old"]);
  assert.equal(isRetryablePinError(new ChatSettingsRequestError("Bad request", 400, null)), false);
});

test("an unpin that cancels a queued pin sends that pin's ID so a delayed copy is fenced", async () => {
  let value: string[] = [];
  const sent: PinOperation[] = [];
  let offline = true;
  const sync = createPinnedChatsSync({
    load: async () => value.slice(),
    compareAndSet: async (expected, next, operation) => {
      if (offline) throw new Error("offline");
      if (operation) sent.push(operation);
      if (JSON.stringify(expected ?? []) !== JSON.stringify(value)) return { applied: false, pinnedIds: value.slice() };
      value = next.slice();
      return { applied: true, pinnedIds: value.slice() };
    },
    readLegacy: () => [],
    publish: () => undefined,
  });
  await sync.refresh();
  await assert.rejects(sync.setPinned(["chat"], true));
  offline = false;
  await sync.setPinned(["chat"], false);
  const unpin = sent.find((op) => !op.pinned);
  assert.ok(unpin?.cancels?.length === 1);
  assert.ok(!sent.some((op) => op.pinned));
  assert.deepEqual(validatePinOperation(unpin).cancels, unpin.cancels);
});

test("trimming an already-committed bulk pin does not repin what another client unpinned", async () => {
  let value: string[] = [];
  const seen = new Set<string>();
  let loseResponse = true;
  const sync = createPinnedChatsSync({
    load: async () => value.slice(),
    compareAndSet: async (expected, next, operation) => {
      const id = operation && (operation.serverId ?? operation.operationId);
      if (id && seen.has(id)) return { applied: true, pinnedIds: value.slice() };
      if (JSON.stringify(expected ?? []) !== JSON.stringify(value)) return { applied: false, pinnedIds: value.slice() };
      value = next.slice();
      if (id) seen.add(id);
      if (loseResponse) {
        loseResponse = false;
        throw new TypeError("connection reset");
      }
      return { applied: true, pinnedIds: value.slice() };
    },
    readLegacy: () => [],
    publish: () => undefined,
  });
  await sync.refresh();
  await assert.rejects(sync.setPinned(["a", "b"], true));
  assert.deepEqual(value, ["a", "b"]);
  value = ["a"]; // Another client unpins "b".
  await sync.setPinned(["a"], false);
  await sync.refresh();
  assert.deepEqual(value, []);
});

test("an unpin over an original and its trimmed copy fences their shared server ID once", async () => {
  const sent: PinOperation[] = [];
  const queued: PinOperation[] = [
    { kind: "pin", operationId: "orig", ids: ["a", "b", "c"], pinned: true },
    { kind: "pin", operationId: "copy", ids: ["b", "c"], pinned: true, serverId: "orig" },
  ];
  let journal = structuredClone(queued);
  const sync = createPinnedChatsSync({
    load: async () => [],
    compareAndSet: async (_expected, _next, operation) => {
      if (operation) sent.push(operation);
      return { applied: false, pinnedIds: [] };
    },
    readLegacy: () => [],
    readPending: () => structuredClone(journal),
    writePending: (operations) => { journal = structuredClone(operations); },
    publish: () => undefined,
  });
  const unpin = sync.setPinned(["b"], false).catch(() => undefined);
  const saved = journal.find((operation) => !operation.pinned);
  assert.deepEqual(saved?.cancels, ["orig"]);
  sync.reset();
  await unpin;
});

