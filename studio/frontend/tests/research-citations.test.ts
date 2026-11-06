// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { researchCitation } from "../src/features/chat/utils/research-citations.ts";

test("MCP citations cannot open a RAG document preview, including stored reports", () => {
  const source = {
    kind: "mcp" as const,
    documentId: "mcp__notes__search",
    filename: "Notes",
    snippet: "A fact",
  };
  assert.equal(researchCitation(source, 0).documentId, undefined);
  assert.equal(researchCitation(source, 0).text, "A fact");
  assert.equal(
    researchCitation(
      { ...source, kind: "knowledge_base", documentId: "doc-1" },
      0,
    ).documentId,
    "doc-1",
  );
});
