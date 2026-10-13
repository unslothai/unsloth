// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("RAG PDF previews release their worker after a load error", async () => {
  const source = await readFile(
    new URL(
      "../src/features/rag/components/document-preview-sheet.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(source, /usePdfWorker\(error === null\)/);
  assert.match(source, /onLoadError=\{\(e\) => setError\(e\.message\)\}/);
});
