// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SoleQuantEntry,
  type SoleQuantTarget,
  createSoleQuantReader,
  partitionSoleQuants,
  soleQuantFingerprint,
  soleQuantKey,
  takeDriftedRepos,
} from "../src/features/model-picker/components/model-selector/sole-quant-cache.ts";

const A = "unsloth/Qwen3-8B-GGUF";
const B = "unsloth/Llama-3.1-8B-Instruct-GGUF";

/** Two listed repos, each at its own cache version. */
const targetsAt = (versionA: string, versionB: string): SoleQuantTarget[] => [
  {
    repoId: A,
    localSource: null,
    fingerprint: "",
    key: soleQuantKey(versionA, null),
  },
  {
    repoId: B,
    localSource: null,
    fingerprint: "",
    key: soleQuantKey(versionB, null),
  },
];

const settled = (
  targets: SoleQuantTarget[],
  quants: (string | null)[],
): Map<string, SoleQuantEntry<string>> =>
  new Map(
    targets.map((target, index) => [
      target.repoId,
      { key: target.key, quant: quants[index] ?? null },
    ]),
  );

test("resolved repos are rows, unread repos are pending", () => {
  const targets = targetsAt("1:0", "1:0");
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled([targets[0]], ["Q4_K_M"]),
    { enabled: true },
  );
  assert.deepEqual([...quants], [[A, "Q4_K_M"]]);
  assert.deepEqual([...pending], [B]);
  assert.deepEqual(
    stale.map((target) => target.repoId),
    [B],
  );
});

test("one repo's invalidation leaves the other repo's row alone", () => {
  const before = targetsAt("1:0", "1:0");
  const entries = settled(before, ["Q4_K_M", "Q8_0"]);
  // B is downloaded into, so only B's version moves.
  const after = targetsAt("1:0", "1:7");

  const { quants, pending, stale } = partitionSoleQuants(after, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], [[A, "Q4_K_M"]]);
  assert.deepEqual([...pending], [B]);
  assert.deepEqual(
    stale.map((target) => target.repoId),
    [B],
  );
});

test("a repo pointed at another directory is re-read", () => {
  const targets = targetsAt("1:0", "1:0");
  const entries = settled(targets, ["Q4_K_M", "Q8_0"]);
  const moved: SoleQuantTarget[] = [
    {
      repoId: A,
      localSource: "/other/cache",
      fingerprint: "",
      key: soleQuantKey("1:0", "/other/cache"),
    },
    targets[1],
  ];

  const { quants, pending } = partitionSoleQuants(moved, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], [[B, "Q8_0"]]);
  assert.deepEqual([...pending], [A]);
});

test("a repo with no single quant is settled, not pending", () => {
  const targets = targetsAt("1:0", "1:0");
  // A holds two quants, or could not be read: either way no row, no re-read.
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled(targets, [null, "Q8_0"]),
    { enabled: true },
  );
  assert.deepEqual([...quants], [[B, "Q8_0"]]);
  assert.deepEqual([...pending], []);
  assert.deepEqual(stale, []);
});

test("disabled reports nothing and asks for nothing", () => {
  const targets = targetsAt("1:0", "1:0");
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled(targets, ["Q4_K_M", "Q8_0"]),
    { enabled: false },
  );
  assert.deepEqual([...quants], []);
  assert.deepEqual([...pending], []);
  assert.deepEqual(stale, []);
});

test("a global invalidation moves every repo's key", () => {
  const before = targetsAt("1:0", "1:0");
  const entries = settled(before, ["Q4_K_M", "Q8_0"]);
  const after = targetsAt("2:0", "2:0");
  const { quants, pending } = partitionSoleQuants(after, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], []);
  assert.deepEqual([...pending], [A, B]);
});

/** A read whose completion the test controls. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

const targetAt = (repoId: string, key: string): SoleQuantTarget => ({
  repoId,
  localSource: null,
  fingerprint: "",
  key,
});

test("a superseded read never overwrites the fresher result", async () => {
  const reads = new Map<string, ReturnType<typeof deferred<string | null>>>();
  const committed: [string, string | null][] = [];
  const reader = createSoleQuantReader<string>({
    workers: 4,
    read: (target) => {
      const pending = deferred<string | null>();
      reads.set(target.key, pending);
      return pending.promise;
    },
    commit: (target, quant) => committed.push([target.key, quant]),
  });

  // First read is still running when the repo is invalidated.
  reader.start([targetAt(A, "v1")]);
  await flush();
  reader.start([targetAt(A, "v2")]);
  await flush();

  // The newer read lands first, then the stale one.
  reads.get("v2")?.resolve("Q8_0");
  await flush();
  reads.get("v1")?.resolve("Q4_K_M");
  await flush();

  assert.deepEqual(committed, [["v2", "Q8_0"]]);
});

test("a repo already being read at the same key is not read again", async () => {
  let calls = 0;
  const pending = deferred<string | null>();
  const reader = createSoleQuantReader<string>({
    workers: 4,
    read: () => {
      calls += 1;
      return pending.promise;
    },
    commit: () => {},
  });

  reader.start([targetAt(A, "v1")]);
  reader.start([targetAt(A, "v1")]);
  await flush();
  assert.equal(calls, 1);
  pending.resolve(null);
});

test("reads are capped at the worker count", async () => {
  let open = 0;
  let peak = 0;
  const gates: ((value: string | null) => void)[] = [];
  const reader = createSoleQuantReader<string>({
    workers: 2,
    read: () => {
      open += 1;
      peak = Math.max(peak, open);
      const gate = deferred<string | null>();
      gates.push((value) => {
        open -= 1;
        gate.resolve(value);
      });
      return gate.promise;
    },
    commit: () => {},
  });

  reader.start(
    ["r1", "r2", "r3", "r4", "r5"].map((repoId) => targetAt(repoId, "v1")),
  );
  await flush();
  assert.equal(peak, 2);

  while (gates.length > 0) {
    gates.shift()?.(null);
    await flush();
  }
  assert.equal(peak, 2);
});

test("a failed read commits as no sole quant", async () => {
  const committed: [string, string | null][] = [];
  const reader = createSoleQuantReader<string>({
    workers: 1,
    read: () => Promise.reject(new Error("offline")),
    commit: (target, quant) => committed.push([target.repoId, quant]),
  });

  reader.start([targetAt(A, "v1")]);
  await flush();
  assert.deepEqual(committed, [[A, null]]);
});

test("bytes changing on disk moves the key, so the repo is read again", () => {
  const beforePrint = soleQuantFingerprint({
    size_bytes: 100,
    last_modified: 10,
  });
  const before = {
    repoId: A,
    localSource: null,
    fingerprint: beforePrint,
    key: soleQuantKey(
      "1:0",
      null,
      soleQuantFingerprint({ size_bytes: 100, last_modified: 10 }),
    ),
  };
  const entries = new Map([[A, { key: before.key, quant: "Q4_K_M" }]]);

  // Another tab replaced the quant: same cache version, different bytes.
  const afterPrint = soleQuantFingerprint({
    size_bytes: 250,
    last_modified: 99,
  });
  const after = [
    {
      repoId: A,
      localSource: null,
      fingerprint: afterPrint,
      key: soleQuantKey("1:0", null, afterPrint),
    },
  ];
  const { quants, pending } = partitionSoleQuants(after, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], []);
  assert.deepEqual([...pending], [A]);
});

test("unchanged bytes keep the repo settled", () => {
  const fingerprint = soleQuantFingerprint({
    size_bytes: 100,
    last_modified: 10,
  });
  const target = {
    repoId: A,
    localSource: null,
    fingerprint,
    key: soleQuantKey("1:0", null, fingerprint),
  };
  const entries = new Map([[A, { key: target.key, quant: "Q4_K_M" }]]);
  const { quants, pending } = partitionSoleQuants([target], entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], [[A, "Q4_K_M"]]);
  assert.deepEqual([...pending], []);
});

const targetWith = (
  repoId: string,
  fingerprint: string,
  version: string,
): SoleQuantTarget => ({
  repoId,
  localSource: null,
  fingerprint,
  key: soleQuantKey(version, null, fingerprint),
});

test("first sight records without asking for an invalidation", () => {
  const seen = new Map<string, string>();
  assert.deepEqual(
    takeDriftedRepos([targetWith(A, "100:10", "1:0")], seen),
    [],
  );
  assert.equal(seen.get(A), "100:10");
});

test("a version bump alone does not count as drift", () => {
  const seen = new Map<string, string>();
  takeDriftedRepos([targetWith(A, "100:10", "1:0")], seen);
  // Dropping a listing bumps the version, which moves the key. Reacting to
  // that would invalidate on its own effect forever.
  assert.deepEqual(
    takeDriftedRepos([targetWith(A, "100:10", "1:9")], seen),
    [],
  );
});

test("changed bytes drift once, not on every pass", () => {
  const seen = new Map<string, string>();
  takeDriftedRepos([targetWith(A, "100:10", "1:0")], seen);
  assert.deepEqual(takeDriftedRepos([targetWith(A, "250:99", "1:0")], seen), [
    A,
  ]);
  // The bump that follows the invalidation must not drift again.
  assert.deepEqual(
    takeDriftedRepos([targetWith(A, "250:99", "1:9")], seen),
    [],
  );
});

test("only the repo whose bytes moved drifts", () => {
  const seen = new Map<string, string>();
  const before = [targetWith(A, "100:10", "1:0"), targetWith(B, "7:1", "1:0")];
  takeDriftedRepos(before, seen);
  const after = [targetWith(A, "100:10", "1:0"), targetWith(B, "9:2", "1:0")];
  assert.deepEqual(takeDriftedRepos(after, seen), [B]);
});
