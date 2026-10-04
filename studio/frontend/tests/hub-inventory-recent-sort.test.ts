// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const { compareInventoryItemsByRecent } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);
const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { epochMillisecondsToSeconds } = await import(
  "../src/features/hub/inventory/inventory-timestamps.ts"
);
const {
  createPendingInventoryHints,
  INVENTORY_HINT_TTL_MS,
  pruneExpiredInventoryHints,
  reconcileInventoryHints,
  rememberInventoryHint,
} = await import("../src/features/hub/inventory/inventory-hints.ts");
const { pendingWithInventoryHints, useInventoryHintStore } = await import(
  "../src/features/hub/inventory/inventory-hint-store.ts"
);
const { deleteCachedModel } = await import(
  "../src/features/hub/inventory/api.ts"
);
const {
  discardDeletedInventoryHints,
  getState,
  jobKeyOf,
  patchJob,
  putJob,
  removeJob,
} = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

function cached(repoId: string, lastModified?: number) {
  return {
    variant: "cached" as const,
    row: buildCachedInventoryRow(
      {
        repo_id: repoId,
        size_bytes: 1,
        last_modified: lastModified,
      },
      "gguf",
    ),
  };
}

function local(displayName: string, updatedAt?: number) {
  const [row] = buildLocalInventoryRows([
    {
      id: `local:${displayName}`,
      display_name: displayName,
      path: `C:\\models\\${displayName}.gguf`,
      source: "lmstudio",
      updated_at: updatedAt,
    },
  ]);
  return { variant: "local" as const, row };
}

test("Recent sorts cached and local rows together by normalized timestamp", () => {
  const oldCached = cached("Org/Alpha-Old", 1_700_000_000);
  const middleLocal = local("LM Studio Middle", 1_800_000_000_000);
  const newCached = cached("Org/Zulu-New", 1_900_000_000);

  assert.equal(oldCached.row.lastModified, 1_700_000_000_000);
  assert.equal(newCached.row.lastModified, 1_900_000_000_000);

  const sorted = [oldCached, middleLocal, newCached].sort(
    compareInventoryItemsByRecent,
  );
  assert.deepEqual(
    sorted.map((item) => item.row.id),
    [newCached.row.id, middleLocal.row.id, oldCached.row.id],
  );
});

test("Recent puts unknown timestamps last and leaves ties stable", () => {
  const unknownCached = cached("Org/Unknown");
  const firstTie = local("First tie", 1_800_000_000);
  const secondTie = cached("Org/Second-Tie", 1_800_000_000);

  const sorted = [unknownCached, firstTie, secondTie].sort(
    compareInventoryItemsByRecent,
  );
  assert.deepEqual(
    sorted.map((item) => item.row.id),
    [firstTie.row.id, secondTie.row.id, unknownCached.row.id],
  );
});

test("Recent keeps a completed download first until the cache rescan", () => {
  const completedAt = Date.now();
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "gguf",
    repoId: "Org/Just-Downloaded",
    bytes: 10,
    createdAt: completedAt,
  });
  const reconciled = reconcileInventoryHints({
    pending,
    kind: "gguf",
    rows: [
      {
        repo_id: "Org/Older",
        size_bytes: 10,
        last_modified: 1_700_000_000,
      },
    ],
    previouslyObserved: new Set<string>(),
  });
  const sorted = reconciled.rows
    .map((row) => ({
      variant: "cached" as const,
      row: buildCachedInventoryRow(row, "gguf"),
    }))
    .sort(compareInventoryItemsByRecent);

  assert.equal(sorted[0]?.row.repoId, "Org/Just-Downloaded");
  assert.equal(sorted[0]?.row.lastModified, completedAt);
});

test("a newer completion refreshes the same repo timestamp and expiry", () => {
  const firstStartedAt = 1_899_999_940_000;
  const firstCompletedAt = 1_900_000_000_000;
  const secondStartedAt = firstCompletedAt + 30_000;
  const secondCompletedAt = firstCompletedAt + 60_000;
  let pending = pendingWithInventoryHints(createPendingInventoryHints(), [
    {
      kind: "gguf",
      repoId: "Org/Repeated",
      bytes: 20,
      startedAt: firstStartedAt,
      createdAt: firstCompletedAt,
    },
  ]);
  pending = pendingWithInventoryHints(pending, [
    {
      kind: "gguf",
      repoId: "Org/Repeated",
      bytes: 10,
      startedAt: secondStartedAt,
      createdAt: secondCompletedAt,
    },
  ]);

  assert.deepEqual(pending.gguf.get("org/repeated"), {
    kind: "gguf",
    repoId: "Org/Repeated",
    bytes: 20,
    startedAt: secondStartedAt,
    createdAt: secondCompletedAt,
  });
  assert.equal(
    pruneExpiredInventoryHints(
      pending,
      firstCompletedAt + INVENTORY_HINT_TTL_MS + 1,
    ).gguf.size,
    1,
  );
  assert.equal(
    pruneExpiredInventoryHints(
      pending,
      secondCompletedAt + INVENTORY_HINT_TTL_MS,
    ).gguf.size,
    0,
  );
});

test("a stale aggregate row cannot consume a newer variant hint", () => {
  const completedAt = 1_900_000_000_000;
  const startedAt = completedAt - 60_000;
  const pending = pendingWithInventoryHints(createPendingInventoryHints(), [
    {
      kind: "gguf",
      repoId: "Org/Existing",
      bytes: 200,
      startedAt,
      createdAt: completedAt,
    },
  ]);
  const stale = reconcileInventoryHints({
    pending,
    kind: "gguf",
    rows: [
      {
        repo_id: "Org/Existing",
        size_bytes: 500,
        last_modified: (startedAt - 1_000) / 1000,
      },
    ],
    previouslyObserved: new Set<string>(),
    refreshStartedAt: completedAt - 1,
  });

  assert.equal(stale.pending.gguf.size, 1);
  assert.equal(stale.rows[0]?.last_modified, completedAt);

  const rescanned = reconcileInventoryHints({
    pending: stale.pending,
    kind: "gguf",
    rows: [
      {
        repo_id: "Org/Existing",
        size_bytes: 100,
        last_modified: (startedAt - 1_000) / 1000,
      },
    ],
    previouslyObserved: new Set(["org/existing"]),
    refreshStartedAt: completedAt + 1,
  });
  assert.equal(rescanned.pending.gguf.size, 0);

  const unconfirmed = reconcileInventoryHints({
    pending: stale.pending,
    kind: "gguf",
    rows: rescanned.rows,
    previouslyObserved: new Set(["org/existing"]),
    refreshStartedAt: null,
  });
  assert.equal(unconfirmed.pending.gguf.size, 1);
});

test("a new download clears historical observation and records completion time", () => {
  const repoId = "Org/Redownload";
  const key = jobKeyOf("model", repoId, "Q4_K_M");
  const hintState = useInventoryHintStore.getState();
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: {
      ...hintState.observedKeys,
      gguf: new Set(["org/redownload"]),
    },
  });
  putJob({
    key,
    kind: "model",
    repoId,
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes: 0,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 10,
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 1_900_000_000_000,
  });
  const afterStart = useInventoryHintStore.getState();
  const staleRowReconciliation = reconcileInventoryHints({
    pending: afterStart.pending,
    kind: "gguf",
    rows: [
      {
        repo_id: repoId,
        size_bytes: 100,
        last_modified: 1_800_000_000,
      },
    ],
    previouslyObserved: afterStart.observedKeys.gguf,
  });
  const beforeCompletion = Date.now();
  patchJob(key, { state: "complete" });
  const afterCompletion = Date.now();
  afterStart.commitReconciliations([], afterStart.pending, [
    {
      kind: "gguf",
      protectedObservedKeys: new Set(["org/redownload"]),
      reconciliation: staleRowReconciliation,
    },
  ]);
  assert.equal(
    useInventoryHintStore.getState().observedKeys.gguf.has("org/redownload"),
    false,
  );
  const [hint] = getState().completedInventoryHints;
  const completedState = useInventoryHintStore.getState();
  const completedPending = pendingWithInventoryHints(
    completedState.pending,
    hint ? [hint] : [],
  );
  const missingRowReconciliation = reconcileInventoryHints({
    pending: completedPending,
    kind: "gguf",
    rows: [],
    previouslyObserved: completedState.observedKeys.gguf,
  });
  const suppressed = completedState.commitReconciliations(
    hint ? [hint] : [],
    completedPending,
    [{ kind: "gguf", reconciliation: missingRowReconciliation }],
  );

  assert.equal(
    useInventoryHintStore.getState().observedKeys.gguf.has("org/redownload"),
    false,
  );
  assert.equal(suppressed.length, 0);
  assert.equal(
    useInventoryHintStore.getState().pending.gguf.has("org/redownload"),
    true,
  );
  assert.ok(hint?.createdAt && hint.createdAt >= beforeCompletion);
  assert.ok(hint?.createdAt && hint.createdAt <= afterCompletion);
  assert.equal(hint?.startedAt, 1_900_000_000_000);

  removeJob(key);
  assert.equal(
    useInventoryHintStore.getState().pending.gguf.has("org/redownload"),
    true,
  );
  discardDeletedInventoryHints(repoId, ["gguf"]);
  assert.equal(
    useInventoryHintStore.getState().pending.gguf.has("org/redownload"),
    false,
  );
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: hintState.observedKeys,
  });
});

test("full deletion removes every completed variant contribution", () => {
  const repoId = "Org/Deleted-Variants";
  const keys = ["Q4_K_M", "Q8_0"].map((variant, index) => {
    const key = jobKeyOf("model", repoId, variant);
    putJob({
      key,
      kind: "model",
      repoId,
      variant,
      state: "running",
      downloadedBytes: 10,
      completedBytes: 10,
      completeOnDisk: true,
      expectedBytes: 10,
      fraction: 1,
      bytesPerSec: 0,
      etaSeconds: 0,
      error: null,
      startedAt: 1_900_000_000_000 + index,
    });
    patchJob(key, {
      state: "complete",
      completedAt: 1_900_000_060_000 + index,
    });
    return key;
  });

  assert.equal(getState().completedInventoryHints.length, 2);
  discardDeletedInventoryHints(repoId, ["gguf"]);
  assert.equal(getState().completedInventoryHints.length, 0);
  assert.equal(keys.some((key) => key in getState().jobs), false);

  removeJob(keys[0]);
  assert.equal(getState().completedInventoryHints.length, 0);
  removeJob(keys[1]);
  assert.equal(getState().completedInventoryHints.length, 0);
});

test("variant deletion clears its hint and permits a peer completion", async () => {
  const repoId = "Org/Deleted-Quant";
  const deletedKey = jobKeyOf("model", repoId, "Q4_K_M");
  const peerKey = jobKeyOf("model", repoId, "Q8_0");
  for (const [key, variant] of [
    [deletedKey, "Q4_K_M"],
    [peerKey, "Q8_0"],
  ] as const) {
    putJob({
      key,
      kind: "model",
      repoId,
      variant,
      state: "running",
      downloadedBytes: 10,
      completedBytes: 10,
      completeOnDisk: true,
      expectedBytes: 10,
      fraction: 1,
      bytesPerSec: 0,
      etaSeconds: 0,
      error: null,
      startedAt: 1_900_000_000_000,
    });
  }
  patchJob(deletedKey, {
    state: "complete",
    completedAt: 1_900_000_060_000,
  });

  setAuthFetchHandler(() => new Response(null, { status: 204 }));
  try {
    await deleteCachedModel(repoId, "Q4_K_M");
  } finally {
    setAuthFetchHandler(null);
  }

  assert.equal(
    useInventoryHintStore.getState().pending.gguf.has("org/deleted-quant"),
    false,
  );
  assert.equal(getState().completedInventoryHints.length, 0);
  assert.equal(getState().jobs[deletedKey], undefined);

  patchJob(peerKey, {
    state: "complete",
    completedAt: 1_900_000_120_000,
  });
  assert.equal(getState().completedInventoryHints.length, 1);

  discardDeletedInventoryHints(repoId, ["gguf"]);
  removeJob(deletedKey);
  removeJob(peerKey);
});

test("picker dto timestamps stay in epoch seconds", () => {
  assert.equal(epochMillisecondsToSeconds(1_900_000_000_500), 1_900_000_000.5);
  assert.equal(epochMillisecondsToSeconds(null), undefined);
});
