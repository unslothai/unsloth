// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isAlreadyGone } from "../src/features/rag/types/rag.ts";

/** The error `ragError` throws: an Error carrying the HTTP status, which is what the
 * delete paths have to branch on. */
function ragFailure(status: number, message: string): Error & { status: number } {
  return Object.assign(new Error(message), { status });
}

// Deleting a source that another client (or a replacement retiring the document it was
// edited from) already removed answers 404. That is the state the delete asked for, so
// the caller must not restore the row: on a completed project with no indexing poll to
// correct it, the resurrected row would sit there and 404 on every later action.

test("a 404 means the document is already gone", () => {
  assert.equal(isAlreadyGone(ragFailure(404, "Document not found")), true);
});

test("other failures are real failures", () => {
  // 409 is a linked-folder or still-indexing refusal, 500 a server fault, 503 RAG off --
  // in every one of them the document is still there and the row belongs back on screen.
  for (const status of [400, 403, 409, 413, 500, 503]) {
    assert.equal(
      isAlreadyGone(ragFailure(status, "nope")),
      false,
      `${status} must not be treated as already deleted`,
    );
  }
});

test("a transport error is not a 404", () => {
  // No response at all: the delete may well not have reached the server, so the row has
  // to come back rather than be reported as done.
  assert.equal(isAlreadyGone(new Error("network down")), false);
  assert.equal(isAlreadyGone(undefined), false);
  assert.equal(isAlreadyGone({ status: 404 }), false);
});

test("the status survives being carried as an Error", () => {
  const err = ragFailure(404, "Document not found");
  assert.ok(err instanceof Error, "callers still catch it as an Error");
  assert.equal(err.status, 404);
  assert.equal(err.message, "Document not found");
});
