// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two cases the lifecycle suite does not reach, both about a repo the user
// ALREADY has complete and then downloads again.
//
// Neither is a regression: before #9642 "Recent" had no date term at all, so it
// could not have been right in these cases either. They are pinned here so the
// behaviour is a decision on record rather than an accident, and so that anyone
// who later decides a re-download should jump to the top finds the exact seam.

import assert from "node:assert/strict";
import test from "node:test";

import type { InventoryItem } from "../src/features/hub/catalog/inventory-sort.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildCachedInventoryRow } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { compareInventoryItemsByRecent } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);
const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const { createPendingInventoryHints, reconcileInventoryHints, rememberInventoryHint } =
  await import("../src/features/hub/inventory/inventory-hints.ts");

const NOW_MS = Date.now();
const NOW_S = Math.floor(NOW_MS / 1000);
const DAY_S = 86_400;
const REPO = "unsloth/Qwen3-8B-GGUF";

function cachedRow(row: {
  repo_id: string;
  size_bytes: number;
  last_modified?: number;
  partial?: boolean;
  optimistic?: boolean;
}) {
  return buildCachedInventoryRow(row, "gguf");
}

function recentOrder(rows: readonly ReturnType<typeof cachedRow>[]) {
  const items: InventoryItem[] = rows.map((row) => ({
    variant: "cached" as const,
    row,
  }));
  return [...items]
    .sort(compareInventoryItemsByRecent)
    .map((item) => item.row.repoId);
}

// ---------------------------------------------------------------------------
// H16 -- dedup drops the live row when the repo is already complete
// ---------------------------------------------------------------------------

test("dedup keeps the COMPLETE row over a live re-download of the same repo", () => {
  // preferCachedRow ranks complete above partial before it looks at anything
  // else, and a live download row is partial. So the fresh clock on the live
  // row is discarded along with the row.
  const settled = cachedRow({
    repo_id: REPO,
    size_bytes: 8_000_000_000,
    last_modified: NOW_S - 30 * DAY_S,
  });
  const live = {
    ...cachedRow({
      repo_id: REPO,
      size_bytes: 1_000_000,
      last_modified: NOW_MS,
      partial: true,
      optimistic: true,
    }),
    liveDownload: true,
  };

  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [settled, live],
    localRows: [],
  });

  assert.equal(cachedRows.length, 1, "one row survives per repo+format");
  assert.equal(
    cachedRows[0].partial,
    false,
    "the complete row is the survivor, which is right for identity",
  );
  // The documented consequence: recency still reflects the settled copy, so a
  // re-download of something you already have does not jump to the top until a
  // backend rescan lands. Asserted, not lamented.
  assert.equal(
    cachedRows[0].lastModified,
    (NOW_S - 30 * DAY_S) * 1000,
    "re-downloading an already-complete repo keeps the settled timestamp",
  );
});

test("dedup keeps the live row when the settled copy is itself partial", () => {
  // Same repo, but nothing complete on disk. Here the live row must win, and
  // with it the fresh timestamp, or an interrupted download would outrank the
  // one currently running.
  const stalled = cachedRow({
    repo_id: REPO,
    size_bytes: 500_000,
    last_modified: NOW_S - 30 * DAY_S,
    partial: true,
  });
  const live = {
    ...cachedRow({
      repo_id: REPO,
      size_bytes: 1_000_000,
      last_modified: NOW_MS,
      partial: true,
      optimistic: true,
    }),
    liveDownload: true,
  };

  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [stalled, live],
    localRows: [],
  });

  assert.equal(cachedRows.length, 1);
  assert.equal(cachedRows[0].lastModified, NOW_MS);
});

test("a different quant of the same repo is a different row, not a dedup victim", () => {
  const gguf = cachedRow({ repo_id: REPO, size_bytes: 8e9, last_modified: NOW_S - 30 * DAY_S });
  const safetensors = buildCachedInventoryRow(
    { repo_id: REPO, size_bytes: 16e9, last_modified: NOW_S },
    "safetensors",
  );
  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [gguf, safetensors],
    localRows: [],
  });
  assert.equal(cachedRows.length, 2, "dedup keys on repo AND format");
  assert.deepEqual(recentOrder(cachedRows), [REPO, REPO]);
  assert.equal(
    Math.max(...cachedRows.map((r) => r.lastModified ?? 0)),
    NOW_S * 1000,
  );
});

test("case-differing repo ids still dedup to one row", () => {
  const lower = cachedRow({ repo_id: REPO.toLowerCase(), size_bytes: 8e9, last_modified: NOW_S });
  const upper = cachedRow({ repo_id: REPO.toUpperCase(), size_bytes: 8e9, last_modified: NOW_S });
  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [lower, upper],
    localRows: [],
  });
  assert.equal(cachedRows.length, 1);
});

// ---------------------------------------------------------------------------
// H17 -- a completed hint meeting a stale-but-COMPLETE server row
// ---------------------------------------------------------------------------

test("a complete server row keeps its own authoritative timestamp over a hint", () => {
  // mergeInventoryHint spreads the server row first and the seed only when the
  // server row is partial, so a complete row's date survives. That is the right
  // default: the server observed the filesystem, the hint observed a clock that
  // may be skewed. The cost is that a re-download shows the settled date until
  // the next scan.
  let pending = createPendingInventoryHints();
  pending = rememberInventoryHint(pending, {
    kind: "model",
    repoId: REPO,
    bytes: 8_000_000_000,
    createdAt: NOW_MS,
  });

  const serverRows = [
    { repo_id: REPO, size_bytes: 8_000_000_000, last_modified: NOW_S - 30 * DAY_S },
  ];
  const { rows } = reconcileInventoryHints({
    pending,
    kind: "model",
    rows: serverRows,
    previouslyObserved: new Set<string>(),
  });

  assert.equal(rows.length, 1);
  assert.equal(
    (rows[0] as { last_modified?: number }).last_modified,
    NOW_S - 30 * DAY_S,
    "the server's observation wins over the client clock",
  );
});

test("a PARTIAL server row does take the hint's timestamp", () => {
  // The other merge branch. Here the seed overrides, so a download that finished
  // while the backend still reports a partial row is dated now, not never.
  let pending = createPendingInventoryHints();
  pending = rememberInventoryHint(pending, {
    kind: "model",
    repoId: REPO,
    bytes: 8_000_000_000,
    createdAt: NOW_MS,
  });

  const serverRows = [
    {
      repo_id: REPO,
      size_bytes: 1_000_000,
      partial: true,
      last_modified: NOW_S - 30 * DAY_S,
    },
  ];
  const { rows } = reconcileInventoryHints({
    pending,
    kind: "model",
    rows: serverRows,
    previouslyObserved: new Set<string>(),
  });

  assert.equal(rows.length, 1);
  assert.equal(
    (rows[0] as { last_modified?: number }).last_modified,
    NOW_MS,
    "a partial server row is superseded by the completed hint",
  );
});

test("a brand new repo with no server row at all is dated by the hint", () => {
  let pending = createPendingInventoryHints();
  pending = rememberInventoryHint(pending, {
    kind: "model",
    repoId: REPO,
    bytes: 8_000_000_000,
    createdAt: NOW_MS,
  });

  const { rows } = reconcileInventoryHints({
    pending,
    kind: "model",
    rows: [{ repo_id: "other/thing", size_bytes: 10, last_modified: NOW_S - 90 * DAY_S }],
    previouslyObserved: new Set<string>(),
  });

  const seeded = rows.find((r) => (r as { repo_id: string }).repo_id === REPO);
  assert.ok(seeded, "the hint appends a synthetic row");
  assert.equal((seeded as { last_modified?: number }).last_modified, NOW_MS);

  const built = rows.map((r) =>
    cachedRow(r as { repo_id: string; size_bytes: number; last_modified?: number }),
  );
  assert.equal(recentOrder(built)[0], REPO, "and it lands first in Recent");
});
