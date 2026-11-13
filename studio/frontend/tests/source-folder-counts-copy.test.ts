// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  folderSyncLabel,
  folderSyncProgress,
  linkedFolderDocumentsSummary,
  runningPercent,
} from "../src/features/rag/components/source-folder-copy.ts";

test("a source folder's sync line counts files an earlier sync already indexed", () => {
  // 592 files were indexed before; this pass has gone through 750 of the rest.
  const progress = folderSyncProgress({ discoveredFiles: 7692, syncedFiles: 1342 });
  assert.deepEqual(progress, { synced: 1342, discovered: 7692 });
  assert.equal(folderSyncLabel(1342, 7692), "1342 of 7692 files synced");
  assert.equal(folderSyncLabel(1, 1), "1 of 1 file synced");
});

test("the folder-wide count waits for the scan and never passes the folder's total", () => {
  assert.equal(folderSyncProgress({ discoveredFiles: 0, syncedFiles: 0 }), null);
  // An older server sends no syncedFiles: the row keeps its previous wording.
  assert.equal(folderSyncProgress({ discoveredFiles: 10 }), null);
  assert.deepEqual(folderSyncProgress({ discoveredFiles: 4, syncedFiles: 6 }), {
    synced: 4,
    discovered: 4,
  });
});

test("the folder card says how many added files are ready and how many still index", () => {
  const busy = linkedFolderDocumentsSummary("OC2", 1342, 2);
  assert.equal(busy.label, "1340 ready · 2 indexing");
  assert.match(busy.title, /1340 of the 1342 files added from this folder so far are ready/);
  assert.match(busy.title, /2 are still being indexed/);
  assert.match(busy.title, /Sources tab shows how far it has got/);

  const idle = linkedFolderDocumentsSummary("OC2", 1342, 0);
  assert.equal(idle.label, "1342 files");
  assert.match(idle.title, /kept in sync and shared with every chat/);
});

test("a chip that is still indexing never reads 100%", () => {
  assert.equal(runningPercent(1340 / 1342), 99);
  assert.equal(runningPercent(1), 99);
  assert.equal(runningPercent(0.5), 50);
  assert.equal(runningPercent(42), 42);
  assert.equal(runningPercent(0), 0);
});

test("the Sources tab rows and the folder card use the shared wording", () => {
  const manager = readFileSync(
    new URL("../src/features/rag/components/linked-folders-manager.tsx", import.meta.url),
    "utf8",
  );
  assert.match(manager, /folderSyncLabel\(/);
  assert.match(manager, /title=\{running && !failed \? FOLDER_SYNC_HELP : detail\}/);
  const card = readFileSync(
    new URL("../src/features/rag/components/thread-documents-bar.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(card, /Indexing \$\{indexing\} of/);
  assert.match(card, /linkedFolderDocumentsSummary\(/);
});
