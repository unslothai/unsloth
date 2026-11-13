// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type Citation,
  citedPagesPerDocument,
  pageLabel,
} from "../src/components/assistant-ui/citation-utils.ts";

test("cited pages collapse into ranges", () => {
  assert.equal(pageLabel([3]), "p.3");
  assert.equal(pageLabel([1, 2, 3, 4, 5]), "p.1–5");
  assert.equal(pageLabel([1, 2, 3, 7]), "pp.1–3, 7");
  assert.equal(pageLabel([2, 4, 6]), "pp.2, 4, 6");
});

test("each document's pages are deduplicated and sorted", () => {
  const cite = (documentId: string | null, filename: string, page: number | null): Citation => ({
    id: `${documentId}-${page}`,
    filename,
    page,
    text: "",
    documentId,
  });
  const pages = citedPagesPerDocument([
    cite("a", "a.pdf", 5),
    cite("a", "a.pdf", 1),
    cite("a", "a.pdf", 5),
    cite("b", "b.md", null),
    cite(null, "c.txt", 2),
  ]);
  assert.deepEqual(pages.get("a"), [1, 5]);
  assert.deepEqual(pages.get("b"), []);
  assert.deepEqual(pages.get("c.txt"), [2]);
});
