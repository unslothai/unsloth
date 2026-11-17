// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { FolderSyncJob } from "../src/features/rag/types/rag.ts";
import {
  chipProgressPercent,
  folderSyncCounts,
  folderSyncJobSummary,
  linkedFolderChipLabel,
  linkedFolderChipTitle,
} from "../src/features/rag/components/source-folder-progress.ts";

function job(partial: Partial<FolderSyncJob>): FolderSyncJob {
  return {
    id: "job",
    linkedFolderId: "folder",
    mode: "sync",
    status: "running",
    ...partial,
  };
}

test("chip percent stays under 100% while indexing is still running", () => {
  assert.equal(chipProgressPercent(1340 / 1342, true), 99);
  assert.equal(chipProgressPercent(0.999, true), 99);
  assert.equal(chipProgressPercent(1, true), 99);
  assert.equal(chipProgressPercent(1, false), 100);
  assert.equal(chipProgressPercent(0.5, true), 50);
  assert.equal(chipProgressPercent(null, true), null);
});

test("sync counts include already-indexed files so the row matches the bar", () => {
  // 1342 on disk; only 2 need work this sync; 1 already handled → 1341 ready.
  const counts = folderSyncCounts(
    job({
      discoveredFiles: 1342,
      processedFiles: 1,
      progress: 0.5,
    }),
  );
  assert.deepEqual(counts, {
    discovered: 1342,
    ready: 1341,
    workTotal: 2,
    processed: 1,
  });
  const summary = folderSyncJobSummary(
    job({
      discoveredFiles: 1342,
      processedFiles: 1,
      progress: 0.5,
    }),
  );
  assert.equal(summary.text, "1341 of 1342 files ready");
  assert.equal(summary.progressPercent, (1341 / 1342) * 100);
  assert.match(summary.title, /1341 of 1342 files ready/);
});

test("sync summary keeps a work-queue label when deletions exceed files on disk", () => {
  const summary = folderSyncJobSummary(
    job({
      discoveredFiles: 10,
      processedFiles: 5,
      progress: 0.25,
    }),
  );
  assert.equal(summary.text, "5 of 20 files this sync");
  assert.equal(summary.progressPercent, 25);
});

test("completed sync names this sync's indexed count", () => {
  const summary = folderSyncJobSummary(
    job({
      status: "completed",
      indexedFiles: 3,
      processedFiles: 4,
    }),
  );
  assert.equal(summary.text, "3 files indexed this sync");
});

test("linked folder chip labels say how many added files are indexed", () => {
  assert.equal(
    linkedFolderChipLabel("Docs", 1342, 2),
    "Docs · 1340 of 1342 indexed",
  );
  assert.equal(linkedFolderChipLabel("Docs", 1, 0), "Docs · 1 file");
  assert.match(
    linkedFolderChipTitle("Docs", 1342, 2),
    /1340 of 1342 added files finished indexing/,
  );
});
