// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createPinnedChatsSync, type PinOperation } from "../src/features/chat/utils/pinned-chats-sync.ts";
import { createPinnedChatsTransport, type PinnedChatsFallbackState } from "../src/features/chat/utils/pinned-chats-transport.ts";
import { ChatSettingsRequestError } from "../src/features/chat/utils/settings-retry.ts";

function unsupported(status = 400, loc = ["patch", "pinnedChatIds"]) {
  return new ChatSettingsRequestError("unsupported", status, [
    { type: "extra_forbidden", loc },
  ]);
}
function pin(operationId: string, ids: string[], pinned = true): PinOperation {
  return { operationId, kind: "pin", ids, pinned };
}
function fixture(error: unknown = unsupported()) {
  const legacy = ["legacy"];
  let remote: string[] | undefined;
  let failure = error;
  let loadFailure: unknown;
  let remoteCalls = 0;
  let loads = 0;
  let warnings = 0;
  let fallback: PinnedChatsFallbackState | undefined;
  let journalFailure = false;
  const acknowledged = new Set<string>();
  const tombstones = new Set<string>();
  const readLegacy = () => legacy.slice();
  const options = {
    load: async () => {
      loads += 1;
      if (loadFailure) throw loadFailure;
      return remote?.slice();
    },
    compareAndSet: async (expected: string[] | undefined, next: string[], operation?: PinOperation) => {
      remoteCalls += 1;
      if (failure) throw failure;
      if (operation && acknowledged.has(operation.operationId))
        return { pinnedIds: remote?.slice(), applied: true };
      const applied = JSON.stringify(expected) === JSON.stringify(remote);
      if (applied) {
        remote = next.slice();
        if (operation) acknowledged.add(operation.operationId);
      }
      return { pinnedIds: remote?.slice(), applied };
    },
    migrateLegacy: async (ids: string[]) => {
      if (failure) throw failure;
      remote = [...new Set([...(remote ?? []), ...ids.filter((id) => !tombstones.has(id))])];
      return remote.slice();
    },
    readLegacy,
    readFallback: () => fallback && structuredClone(fallback),
    writeFallback: (value: PinnedChatsFallbackState | undefined) => {
      if (journalFailure) throw new Error("journal full");
      fallback = value && structuredClone(value);
    },
    onUnsupported: () => { warnings += 1; },
  };
  const transport = createPinnedChatsTransport(options);
  let visible: string[] = [];
  const sync = createPinnedChatsSync({
    ...transport, readLegacy, publish: (ids) => { visible = ids; },
  });
  return {
    options, transport, sync, tombstones, acknowledged,
    upgrade: () => { failure = undefined; },
    set failure(value: unknown) { failure = value; },
    set loadFailure(value: unknown) { loadFailure = value; },
    set journalFailure(value: boolean) { journalFailure = value; },
    set remote(value: string[] | undefined) { remote = value; },
    get remote() { return remote; },
    get local() { return fallback?.pinnedIds ?? legacy; },
    get visible() { return visible; },
    get fallback() { return fallback; },
    get remoteCalls() { return remoteCalls; },
    get loads() { return loads; },
    get warnings() { return warnings; },
  };
}

test("an older backend journals local pin/unpin and probes again on refresh", async () => {
  const f = fixture();
  await f.sync.refresh();
  assert.deepEqual(f.visible, ["legacy"]);
  await f.sync.setPinned(["new", "another"], true);
  assert.deepEqual(f.local, ["new", "another", "legacy"]);
  await f.sync.setPinned(["legacy", "new", "another"], false);
  assert.deepEqual(f.local, []);
  assert.equal(f.fallback?.operations.length, 2);
  const before = f.loads;
  await f.sync.refresh();
  assert.equal(f.loads, before + 1);
  assert.deepEqual(f.visible, []);
  assert.equal(f.warnings, 1);
});

test("explicit missing routes and combined old-schema errors enable fallback", async () => {
  const errors = [unsupported(404), unsupported(405), new ChatSettingsRequestError("unsupported", 400, [
    { type: "extra_forbidden", loc: ["expected", "pinnedChatIds"] },
    { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
  ]), new ChatSettingsRequestError("unsupported", 400, [
    // Downgraded since operation IDs were probed: the old schema rejects them too.
    { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
    { type: "extra_forbidden", loc: ["operationId"] },
  ]), new ChatSettingsRequestError("unsupported", 400, [
    { type: "value_error", loc: ["expectedAbsent"], msg: "Value error, Unknown chat settings field(s): pinnedChatIds" },
    { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
  ])];
  for (const error of errors) {
    const f = fixture(error);
    await f.sync.setPinned(["new"], true);
    assert.deepEqual(f.local, ["new", "legacy"]);
    assert.equal(f.fallback?.operations.length, 1);
    assert.equal(f.warnings, 1);
  }
});

test("network, auth, server, and unrelated validation failures never switch to local writes", async () => {
  const errors = [
    new TypeError("network unavailable"),
    new ChatSettingsRequestError("unauthorized", 401, null),
    new ChatSettingsRequestError("forbidden", 403, null),
    new ChatSettingsRequestError("internal error", 500, null),
    new ChatSettingsRequestError("bad pins", 400, [
      { type: "too_long", loc: ["patch", "pinnedChatIds"] },
    ]),
    new ChatSettingsRequestError("other field", 400, [
      { type: "extra_forbidden", loc: ["patch", "other"] },
    ]),
    new ChatSettingsRequestError("mixed error", 400, [
      { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
      { type: "bool_type", loc: ["expected", "toolsEnabled"] },
    ]),
    new ChatSettingsRequestError("absent field alone", 400, [
      {
        type: "value_error",
        loc: ["expectedAbsent"],
        msg: "Value error, Unknown chat settings field(s): pinnedChatIds",
      },
    ]),
    new ChatSettingsRequestError("other absent field", 400, [
      { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
      {
        type: "value_error",
        loc: ["expectedAbsent"],
        msg: "Value error, Unknown chat settings field(s): other, pinnedChatIds",
      },
    ]),
    new ChatSettingsRequestError("other absent validation", 400, [
      { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
      {
        type: "list_type",
        loc: ["expectedAbsent"],
        msg: "Input should be a valid list",
      },
    ]),
    new ChatSettingsRequestError("other absent path", 400, [
      { type: "extra_forbidden", loc: ["patch", "pinnedChatIds"] },
      {
        type: "value_error",
        loc: ["expectedAbsentPaths"],
        msg: "Value error, Unknown chat settings field(s): pinnedChatIds",
      },
    ]),
    new ChatSettingsRequestError("malformed detail", 400, "pinnedChatIds"),
    new ChatSettingsRequestError("no details", 400, []),
  ];
  for (const error of errors) {
    const f = fixture(error);
    await assert.rejects(
      f.transport.compareAndSet(undefined, ["new"]),
      (caught) => caught === error,
    );
    await assert.rejects(
      f.transport.compareAndSet(undefined, ["new"]),
      (caught) => caught === error,
    );
    assert.deepEqual(f.local, ["legacy"]);
    assert.equal(f.warnings, 0);
    assert.equal(f.remoteCalls, 2);
    assert.equal(f.fallback, undefined);
  }
});


test("upgrade automatically merges baseline and replays explicit changes in order", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  await f.sync.setPinned(["another"], true);
  await f.sync.setPinned(["legacy"], false);
  f.remote = ["other-browser"];
  f.upgrade();
  await f.sync.refresh();
  assert.deepEqual(f.remote, ["another", "new", "other-browser"]);
  assert.deepEqual(f.visible, f.remote);
  assert.equal(f.fallback, undefined);
  await f.sync.setPinned(["after-upgrade"], true);
  assert.deepEqual(f.remote, ["after-upgrade", "another", "new", "other-browser"]);
});

test("downgrade starts local state from last remote pins, preserving existing pins", async () => {
  const f = fixture();
  f.upgrade();
  f.remote = ["server-a", "server-b"];
  await f.transport.load();
  f.remote = undefined;
  await f.transport.load();
  f.failure = unsupported();
  const operation = pin("downgrade", ["new"]);
  const conflict = await f.transport.compareAndSet(undefined, ["new"], operation);
  assert.deepEqual(conflict, { applied: false, pinnedIds: ["server-a", "server-b"] });
  await f.transport.compareAndSet(conflict.pinnedIds, ["new", ...conflict.pinnedIds!], operation);
  assert.deepEqual(f.local, ["new", "server-a", "server-b"]);
  assert.deepEqual(f.fallback?.baseIds, ["server-a", "server-b"]);
});

test("fallback survives adapter recreation and skips tombstoned legacy pins on upgrade", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  const restored = createPinnedChatsTransport(f.options);
  assert.deepEqual(await restored.load(), ["new", "legacy"]);
  f.tombstones.add("legacy");
  f.upgrade();
  assert.deepEqual(await restored.load(), ["new"]);
  assert.deepEqual(f.remote, ["new"]);
  assert.equal(f.fallback, undefined);
});

test("a fallback baseline without operations still probes and migrates after upgrade", async () => {
  const f = fixture();
  await f.sync.refresh();
  assert.deepEqual(f.fallback?.operations, []);
  f.upgrade();
  await f.sync.refresh();
  assert.deepEqual(f.remote, ["legacy"]);
  assert.equal(f.fallback, undefined);
});

test("journal failure cannot claim a saved local write", async () => {
  const f = fixture();
  await f.sync.refresh();
  f.journalFailure = true;
  await assert.rejects(f.transport.compareAndSet(["legacy"], ["new", "legacy"], pin("write", ["new"])), /journal full/);
  assert.deepEqual(f.local, ["legacy"]);
  assert.deepEqual(f.fallback?.operations, []);
});

test("partial replay retains operation IDs and retries without repeating acknowledged edits", async () => {
  const f = fixture();
  await f.sync.setPinned(["first"], true);
  await f.sync.setPinned(["second"], true);
  f.upgrade();
  let calls = 0;
  const interrupted = createPinnedChatsTransport({ ...f.options,
    compareAndSet: async (...args) => {
      calls += 1;
      if (calls === 2) throw new TypeError("network disconnected");
      return f.options.compareAndSet(...args);
    },
  });
  assert.deepEqual(await interrupted.load(), ["second", "first", "legacy"]);
  assert.equal(f.fallback?.operations.length, 2);
  assert.deepEqual(f.remote, ["first", "legacy"]);
  // A different browser puts a new row above the first acknowledged pin.
  f.remote = ["other-browser", ...f.remote!];
  assert.deepEqual(await interrupted.load(), ["second", "other-browser", "first", "legacy"]);
  assert.equal(f.fallback, undefined);
});

test("replay conflict retries are bounded and leave the recovery journal intact", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  f.upgrade();
  let conflicts = 0;
  const transport = createPinnedChatsTransport({ ...f.options,
    compareAndSet: async () => { conflicts += 1; return { applied: false, pinnedIds: ["racer"] }; },
  });
  // The saved view stands in until the replay can land.
  assert.deepEqual(await transport.load(), ["new", "legacy"]);
  assert.equal(conflicts, 8);
  assert.equal(f.fallback?.operations.length, 1);
});

test("auth failure propagates during recovery while transient failures keep saved local pins", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  f.loadFailure = new TypeError("offline");
  assert.deepEqual(await f.transport.load(), ["new", "legacy"]);
  f.loadFailure = new ChatSettingsRequestError("unauthorized", 401, null);
  await assert.rejects(f.transport.load(), /unauthorized/);
  assert.equal(f.fallback?.operations.length, 1);
});

test("stale unsupported response after session reset cannot create a fallback journal", async () => {
  const f = fixture();
  let reject!: (error: unknown) => void;
  const transport = createPinnedChatsTransport({ ...f.options,
    compareAndSet: () => new Promise((_yes, no) => { reject = no; }),
  });
  const before = transport.compareAndSet(["legacy"], ["new"], pin("old", ["new"]));
  transport.reset();
  reject(unsupported());
  await assert.rejects(before, /session changed/);
  assert.equal(f.fallback, undefined);
});

test("stale recovery after reset cannot clear the durable fallback", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  let resolve!: (ids: string[]) => void;
  const transport = createPinnedChatsTransport({ ...f.options,
    migrateLegacy: () => new Promise((yes) => { resolve = yes; }),
  });
  const before = transport.load();
  await Promise.resolve();
  transport.reset();
  resolve(["legacy"]);
  await assert.rejects(before, /session changed/);
  assert.equal(f.fallback?.operations.length, 1);
});

test("malformed remote responses are rejected without entering fallback", async () => {
  const f = fixture();
  const transport = createPinnedChatsTransport({ ...f.options,
    load: async () => ["duplicate", "duplicate"],
    compareAndSet: async () => ({ applied: true, pinnedIds: undefined }),
  });
  await assert.rejects(transport.load(), /Invalid pinned chats response/);
  await assert.rejects(transport.compareAndSet([], ["new"], pin("bad", ["new"])), /Invalid pinned chats response/);
  assert.equal(f.fallback, undefined);
});

test("local compare and set preserves another tab's newer snapshot", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  assert.deepEqual(await f.transport.compareAndSet(["legacy"], [], pin("stale", ["legacy"], false)), {
    applied: false, pinnedIds: ["new", "legacy"],
  });
  assert.deepEqual(f.local, ["new", "legacy"]);
});


test("orphan-only migration may leave pins absent before a valid operation is replayed", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  f.upgrade();
  const transport = createPinnedChatsTransport({ ...f.options,
    migrateLegacy: async () => undefined,
  });
  assert.deepEqual(await transport.load(), ["new"]);
  assert.equal(f.fallback, undefined);

  const fresh = fixture();
  const noValidLegacy = createPinnedChatsTransport({ ...fresh.options,
    migrateLegacy: async () => undefined,
  });
  assert.equal(await noValidLegacy.migrateLegacy(["orphan"]), undefined);
  assert.equal(fresh.fallback, undefined);
});

test("upgrade replay skips a pin whose chat was unpinned (deleted) later in the offline log", async () => {
  let remote: string[] = ["chat"];
  const sent: PinOperation[] = [];
  const tombstones = new Set<string>();
  let fallback: PinnedChatsFallbackState | undefined = {
    baseIds: [],
    pinnedIds: ["new"],
    operations: [pin("01", ["pair"]), pin("02", ["pair"], false), pin("03", ["new"])],
  };
  const transport = createPinnedChatsTransport({
    load: async () => remote.slice(),
    compareAndSet: async (expected, next, operation) => {
      if (operation) sent.push(operation);
      // The upgraded server never learned the deleted comparison's pair ID.
      if (next.includes("pair")) return { pinnedIds: remote.slice(), applied: false };
      if (JSON.stringify(expected) !== JSON.stringify(remote)) return { pinnedIds: remote.slice(), applied: false };
      if (operation && !operation.pinned) operation.ids.forEach((id) => tombstones.add(id));
      remote = next.slice();
      return { pinnedIds: remote.slice(), applied: true };
    },
    migrateLegacy: async () => remote.slice(),
    readLegacy: () => [],
    readFallback: () => fallback && structuredClone(fallback),
    writeFallback: (value) => { fallback = value && structuredClone(value); },
  });
  assert.deepEqual(await transport.load(), ["new", "chat"]);
  assert.equal(fallback, undefined);
  assert.ok(tombstones.has("pair"));
  // Another tab may already be replaying the dropped pin: the unpin fences its ID.
  assert.deepEqual(sent.find((op) => !op.pinned)?.cancels, ["01"]);
});

test("saved fallback pins hydrate when authFetch reports the network down", async () => {
  const f = fixture();
  await f.sync.setPinned(["new"], true);
  f.loadFailure = Object.assign(new Error("Failed to fetch"), { unslothTransportFailure: true });
  assert.deepEqual(await f.transport.load(), ["new", "legacy"]);
});

test("an upgrade replay retries a trimmed bulk pin under the same ID", async () => {
  let remote: string[] = [];
  const seen = new Set<string>();
  const ids: string[] = [];
  let loseResponse = true;
  let fallback: PinnedChatsFallbackState | undefined = {
    baseIds: [],
    pinnedIds: ["b"],
    operations: [pin("01", ["a", "b"]), pin("02", ["a"], false)],
  };
  const transport = createPinnedChatsTransport({
    load: async () => remote.slice(),
    compareAndSet: async (expected, next, operation) => {
      if (operation) ids.push(operation.operationId);
      if (operation && seen.has(operation.operationId)) return { pinnedIds: remote.slice(), applied: true };
      if (JSON.stringify(expected) !== JSON.stringify(remote)) return { pinnedIds: remote.slice(), applied: false };
      remote = next.slice();
      if (operation) seen.add(operation.operationId);
      if (operation?.pinned && loseResponse) {
        loseResponse = false;
        throw new TypeError("connection reset");
      }
      return { pinnedIds: remote.slice(), applied: true };
    },
    migrateLegacy: async () => remote.slice(),
    readLegacy: () => [],
    readFallback: () => fallback && structuredClone(fallback),
    writeFallback: (value) => { fallback = value && structuredClone(value); },
  });
  // The first replay commits ["b"] but its response is lost; the saved state stands in.
  assert.deepEqual(await transport.load(), ["b"]);
  assert.deepEqual(remote, ["b"]);
  // Another client explicitly unpins "b" before this one retries.
  remote = [];
  await transport.load();
  const replays = ids.filter((id) => id !== "02");
  assert.equal(new Set(replays).size, 1);
  assert.deepEqual(remote, []);
});

test("a replay the server keeps refusing still lets a later unpin cancel that pin", async () => {
  let remote: string[] = [];
  let fallback: PinnedChatsFallbackState | undefined = {
    baseIds: [],
    pinnedIds: ["pair"],
    operations: [pin("01", ["pair"])],
  };
  const options = {
    load: async () => remote.slice(),
    compareAndSet: async (expected: string[] | undefined, next: string[]) => {
      // A legacy comparison still awaiting import: the upgraded server cannot place it.
      if (next.includes("pair") || JSON.stringify(expected) !== JSON.stringify(remote)) {
        return { pinnedIds: remote.slice(), applied: false };
      }
      remote = next.slice();
      return { pinnedIds: remote.slice(), applied: true };
    },
    migrateLegacy: async () => remote.slice(),
    readLegacy: () => [],
    readFallback: () => fallback && structuredClone(fallback),
    writeFallback: (value: PinnedChatsFallbackState | undefined) => {
      fallback = value && structuredClone(value);
    },
  };
  const transport = createPinnedChatsTransport(options);
  let visible: string[] = [];
  const sync = createPinnedChatsSync({ ...transport, readLegacy: () => [], publish: (ids) => { visible = ids; } });
  await sync.refresh();
  assert.deepEqual(visible, ["pair"]);
  await sync.setPinned(["pair"], false);
  await sync.setPinned(["chat"], true);
  await sync.refresh();
  assert.deepEqual(remote, ["chat"]);
  assert.equal(fallback, undefined);
});

test("local fallback acknowledges a late pin an earlier unpin already fenced", async () => {
  const f = fixture();
  await f.sync.refresh();
  const unpin: PinOperation = { ...pin("u", ["chat"], false), cancels: ["late"] };
  await f.transport.compareAndSet(["legacy"], ["legacy"], unpin);
  const result = await f.transport.compareAndSet(["legacy"], ["chat", "legacy"], pin("late", ["chat"]));
  assert.deepEqual(result, { applied: true, pinnedIds: ["legacy"] });
  assert.equal(f.fallback?.operations.length, 1);
});

test("upgrade replay sends each fenced ID once", async () => {
  let remote: string[] = [];
  const sent: PinOperation[] = [];
  let fallback: PinnedChatsFallbackState | undefined = {
    baseIds: [],
    pinnedIds: [],
    operations: [pin("p", ["chat"]), { ...pin("u", ["chat"], false), cancels: ["p"] }],
  };
  const transport = createPinnedChatsTransport({
    load: async () => remote.slice(),
    compareAndSet: async (expected, next, operation) => {
      if (operation) sent.push(operation);
      if (JSON.stringify(expected) !== JSON.stringify(remote)) return { pinnedIds: remote.slice(), applied: false };
      remote = next.slice();
      return { pinnedIds: remote.slice(), applied: true };
    },
    migrateLegacy: async () => remote.slice(),
    readLegacy: () => [],
    readFallback: () => fallback && structuredClone(fallback),
    writeFallback: (value) => { fallback = value && structuredClone(value); },
  });
  await transport.load();
  assert.deepEqual(sent.find((op) => !op.pinned)?.cancels, ["p"]);
  assert.equal(fallback, undefined);
});

