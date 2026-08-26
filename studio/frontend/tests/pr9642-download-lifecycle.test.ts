// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// S5 for PR #9642. Recent is only useful if a model you just downloaded is at
// the top of it. This walks one download through every state it passes through
// -- running, progress ticks, complete, the optimistic-hint window, the backend
// rescan, and hint expiry -- and asserts where the row sits at each step.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import type { InventoryItem } from "../src/features/hub/catalog/inventory-sort.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { compareInventoryItemsByRecent } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);
const {
  createPendingInventoryHints,
  reconcileInventoryHints,
  rememberInventoryHint,
  INVENTORY_HINT_TTL_MS,
} = await import("../src/features/hub/inventory/inventory-hints.ts");

// Anchored to the real clock on purpose: reconcileInventoryHints prunes against
// Date.now() internally and takes no injectable clock, so a fixed future NOW
// would make every hint look permanently fresh and quietly disable the TTL step.
const NOW_MS = Date.now();
const DAY_S = 86_400;
const NOW_S = Math.floor(NOW_MS / 1000);

const NEW_REPO = "unsloth/Qwen3-8B-GGUF";

/** What the backend already knows about, oldest of which is a week old. */
function settledServerRows() {
  return [
    {
      repo_id: "google/bert-base-uncased",
      size_bytes: 440_000_000,
      last_modified: NOW_S - 30 * DAY_S,
    },
    {
      repo_id: "google/gemma-3-4b-it",
      size_bytes: 4_000_000_000,
      last_modified: NOW_S - 7 * DAY_S,
    },
  ];
}

const LOCAL_ROWS = buildLocalInventoryRows([
  {
    id: "lmstudio-llama-3",
    display_name: "lmstudio-llama-3",
    path: "/models/llama-3.gguf",
    source: "lmstudio",
    updated_at: NOW_S - 3 * DAY_S,
  },
]);

function order(
  rows: readonly {
    repo_id: string;
    size_bytes: number;
    last_modified?: number;
  }[],
) {
  const items: InventoryItem[] = [
    ...rows.map((row) => ({
      variant: "cached" as const,
      row: buildCachedInventoryRow(row, "gguf"),
    })),
    ...LOCAL_ROWS.map((row) => ({ variant: "local" as const, row })),
  ];
  return [...items]
    .sort(compareInventoryItemsByRecent)
    .map((item) => (item.variant === "cached" ? item.row.repoId : item.row.id));
}

test("S5: the source really does stamp live download rows, so this fixture is not fiction", async () => {
  // liveDownloadInventoryRows is module-private, so the lifecycle below rebuilds
  // its payload. Pin the one line that matters, or the fixture could drift away
  // from the implementation and keep passing.
  const source = await readFile(
    new URL(
      "../src/features/hub/inventory/use-hub-inventory.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /last_modified:\s*job\.startedAt/);
  // And that a completed job stops being surfaced as a live row.
  assert.match(
    source,
    /if \(job\.state !== "cancelled" && job\.state !== "error"\) return false;/,
  );
});

test("S5: a running download sorts to the top of Recent", () => {
  const startedAt = NOW_MS;
  const withLive = [
    // liveDownloadInventoryRows prepends the synthetic row.
    {
      repo_id: NEW_REPO,
      size_bytes: 0,
      last_modified: startedAt,
      partial: true,
      optimistic: true,
    },
    ...settledServerRows(),
  ];
  assert.equal(order(withLive)[0], NEW_REPO);
});

test("S5: progress ticks do not move the row, because startedAt is immutable", () => {
  const startedAt = NOW_MS;
  const positions = [0, 1_000_000, 5_000_000_000].map((displayBytes) => {
    const rows = [
      {
        repo_id: NEW_REPO,
        size_bytes: displayBytes,
        last_modified: startedAt,
        partial: true,
        optimistic: true,
      },
      ...settledServerRows(),
    ];
    return order(rows).indexOf(NEW_REPO);
  });
  assert.deepEqual(positions, [0, 0, 0], "a progress tick reordered the list");
});

test("S5: a completed download must not fall to the bottom of Recent", () => {
  // The moment the job reports complete, shouldSurfaceLiveJob stops emitting the
  // live row and the download-manager hands over an inventory hint instead. The
  // backend has not rescanned yet, so the hint's optimistic row is all there is.
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "gguf",
    repoId: NEW_REPO,
    bytes: 8_000_000_000,
    createdAt: NOW_MS,
  });

  const reconciled = reconcileInventoryHints({
    pending,
    kind: "gguf",
    rows: settledServerRows(),
    previouslyObserved: new Set<string>(),
  });

  const merged = reconciled.rows as {
    repo_id: string;
    size_bytes: number;
    last_modified?: number;
  }[];
  assert.ok(
    merged.some((row) => row.repo_id === NEW_REPO),
    "the optimistic row should still represent the finished download",
  );

  const position = order(merged).indexOf(NEW_REPO);
  assert.equal(
    position,
    0,
    `a download that finished this instant sat at position ${position} of ${merged.length + LOCAL_ROWS.length} in Recent`,
  );
});

test("S5: the row stays put once the backend rescan supplies the real timestamp", () => {
  const rescanned = [
    ...settledServerRows(),
    { repo_id: NEW_REPO, size_bytes: 8_000_000_000, last_modified: NOW_S },
  ];
  assert.equal(order(rescanned)[0], NEW_REPO);
});

test("S5: an expired hint cannot re-introduce the drop", () => {
  // Five minutes later with no rescan, the hint is pruned and the row is simply
  // absent -- which is correct, and importantly not a row pinned to the bottom.
  const pending = rememberInventoryHint(createPendingInventoryHints(), {
    kind: "gguf",
    repoId: NEW_REPO,
    bytes: 8_000_000_000,
    createdAt: NOW_MS - INVENTORY_HINT_TTL_MS - 1,
  });
  const reconciled = reconcileInventoryHints({
    pending,
    kind: "gguf",
    rows: settledServerRows(),
    previouslyObserved: new Set<string>(),
  });
  assert.equal(
    (reconciled.rows as { repo_id: string }[]).some(
      (row) => row.repo_id === NEW_REPO,
    ),
    false,
  );
});

test("S5: a cancelled or errored download keeps its live-row position", () => {
  // Those states still surface as live rows, so they keep the startedAt stamp
  // and behave exactly as they did before the PR.
  const startedAt = NOW_MS - 60_000;
  const rows = [
    {
      repo_id: NEW_REPO,
      size_bytes: 1_000,
      last_modified: startedAt,
      partial: true,
      optimistic: true,
    },
    ...settledServerRows(),
  ];
  assert.equal(order(rows)[0], NEW_REPO);
});
