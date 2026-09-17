// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { IMAGE_SENTINEL_TOOLS, SANDBOX_FILE_TOOLS } = await import(
  "../src/components/assistant-ui/sandbox-files.ts"
);

test("only the tools that emit the image envelope are in the set", () => {
  for (const emitter of ["python", "terminal", "code_execution"]) {
    assert.ok(IMAGE_SENTINEL_TOOLS.has(emitter), emitter);
  }
  for (const reader of ["mcp__fs__read_file", "web_search", "search_knowledge_base", ""]) {
    assert.ok(!IMAGE_SENTINEL_TOOLS.has(reader), reader);
  }
  for (const sandbox of SANDBOX_FILE_TOOLS) {
    assert.ok(IMAGE_SENTINEL_TOOLS.has(sandbox), `${sandbox} emits both envelopes`);
  }
});

// Asserted against the source like the other chat-adapter tests: importing the adapter
// drags in the whole app. The backend keeps a well-formed `__IMAGES__` line from a tool
// that does not emit the envelope as content the model reads, so the card must not slice
// it off and fetch a sandbox file that was never written.
test("the adapter slices the image envelope only for the tools that emit it", () => {
  const source = readSrc("features/chat/api/chat-adapter.ts");
  const gate = source.indexOf(
    'IMAGE_SENTINEL_TOOLS.has(\n                      toolCallParts[idx].toolName ?? "",\n                    )',
  );
  const slice = source.indexOf("rawResult.lastIndexOf(imgMarker)");
  assert.ok(gate >= 0, "the __IMAGES__ slice is no longer gated on IMAGE_SENTINEL_TOOLS");
  assert.ok(slice > gate, "the slice runs before the gate");
  assert.equal(
    source.indexOf("rawResult.lastIndexOf(imgMarker)", slice + 1),
    -1,
    "a second, ungated __IMAGES__ slice appeared",
  );
});
