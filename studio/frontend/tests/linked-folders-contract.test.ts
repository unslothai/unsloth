// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  isLinkedFolderManaged,
  linkedFolderSourcesChanged,
  retainActiveFolderJobs,
} from "../src/features/rag/types/rag.ts";

test("documents managed by linked folders cannot be treated as individual uploads", () => {
  const document = {
    id: "doc",
    filename: "notes.md",
    status: "completed",
    managed: false,
  } as const;
  assert.equal(isLinkedFolderManaged(document), false);
  assert.equal(
    isLinkedFolderManaged({ ...document, linkedFolderId: "folder" }),
    true,
  );
  assert.equal(isLinkedFolderManaged({ ...document, managed: true }), true);
});

test("only the folder's currently active job remains visible after polling", () => {
  const folder = {
    id: "folder",
    displayName: "Docs",
    scopeType: "project",
    scopeId: "project",
    status: "idle",
    activeJobId: "job-2",
  } as const;
  const jobs = {
    folder: {
      id: "job-1",
      linkedFolderId: "folder",
      mode: "sync",
      status: "completed",
    },
  } as const;
  assert.deepEqual(retainActiveFolderJobs([folder], jobs), {});
  assert.deepEqual(
    retainActiveFolderJobs([{ ...folder, activeJobId: "job-1" }], jobs),
    jobs,
  );
  assert.deepEqual(
    retainActiveFolderJobs([{ ...folder, activeJobId: null }], jobs),
    {},
  );
});

test("folder polling detects completed backend work without a visible active job", () => {
  const folder = {
    id: "folder",
    displayName: "Docs",
    scopeType: "project",
    scopeId: "project",
    status: "idle",
    documentCount: 2,
    lastSyncedAt: "2026-08-06T10:00:00Z",
  } as const;
  assert.equal(linkedFolderSourcesChanged(null, [folder]), false);
  assert.equal(linkedFolderSourcesChanged([folder], [folder]), false);
  assert.equal(
    linkedFolderSourcesChanged(
      [folder],
      [
        {
          ...folder,
          documentCount: 3,
          lastSyncedAt: "2026-08-06T10:01:00Z",
        },
      ],
    ),
    true,
  );
});
