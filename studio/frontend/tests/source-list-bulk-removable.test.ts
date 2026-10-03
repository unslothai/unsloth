// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isBulkRemovable } = await import(
  "../src/features/rag/lib/source-list.ts"
);

const doc = (over: Record<string, unknown> = {}) => ({
  id: "doc-1",
  filename: "notes.pdf",
  status: "completed" as const,
  managed: false,
  ...over,
});

test("a finished, unmanaged upload can be bulk removed", () => {
  assert.equal(isBulkRemovable(doc()), true);
});

test("an optimistic chip has no server id to delete", () => {
  assert.equal(isBulkRemovable(doc({ id: "pending_abc123" })), false);
});

test("a row that is still indexing is not removable", () => {
  // A live ingestion worker is writing to it; deleting would race the job
  // rather than remove a finished source. The chip hides its checkbox for the
  // same reason, and "Select all" has to agree with the chips.
  for (const status of ["pending", "running"]) {
    assert.equal(isBulkRemovable(doc({ status })), false, status);
  }
});

test("a failed row is still removable", () => {
  // Nothing is writing to it, and clearing failures is the point of the action.
  assert.equal(isBulkRemovable(doc({ status: "failed" })), true);
});

test("linked-folder documents are never removable", () => {
  // The DELETE route answers 409 for these: folder sync owns them.
  assert.equal(isBulkRemovable(doc({ managed: true })), false);
  assert.equal(isBulkRemovable(doc({ linkedFolderId: "folder-1" })), false);
  // Managed wins even when the row is otherwise a normal completed document.
  assert.equal(
    isBulkRemovable(doc({ managed: true, status: "completed" })),
    false,
  );
});
